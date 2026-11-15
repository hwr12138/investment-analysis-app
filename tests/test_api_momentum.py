"""The 🚀 Momentum routes, through the real fetcher with only the rows faked.

The price rows are stubbed one level below ``fetch_ohlcv`` — the same seam
``test_dcf_history.py`` uses — so the renaming, the slicing and the cache all run
for real. The JSON is parsed strictly, so a NaN that slipped through would fail
here rather than reach the frontend as an invalid document.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import contract, main, momentum
from app.data import sharadar_client as sc
from app.data.sharadar import prices as sh_prices
from app.logic import momentum_analysis as ma


def _rows(start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, date.today())
    close = [100 + 10 * math.sin(i / 15) + 0.05 * i for i in range(len(dates))]
    return pd.DataFrame({
        "date": dates,
        "open": [c - 0.5 for c in close],
        "high": [c + 1.0 for c in close],
        "low": [c - 1.0 for c in close],
        "close": close,
        "volume": [1_000_000] * len(dates),
        "closeadj": close,
    })


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("EQUITI_AUTH", "off")
    monkeypatch.delenv("K_SERVICE", raising=False)
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def prices(monkeypatch):
    """Routes every ticker to SFP and answers with ``rows`` — swap it per test."""
    state = {"rows": _rows(), "calls": 0}

    def fetch(ticker, *, table, start=None, end=None):
        state["calls"] += 1
        return state["rows"]

    monkeypatch.setattr(sh_prices, "_price_table_for", lambda ticker: "SFP")
    monkeypatch.setattr(sh_prices, "_fetch_price_rows", fetch)
    return state


def _strict(response) -> dict:
    """The body parsed the way a browser's ``JSON.parse`` would: no NaN."""
    def refuse(token):
        raise AssertionError(f"non-JSON constant {token} in the response")

    return json.loads(response.text, parse_constant=refuse)


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def test_the_window_is_trimmed_to_the_period_after_the_warm_up(client, prices):
    body = _strict(client.get("/api/momentum/aapl", params={"period": "3M"}))
    sessions = [date.fromisoformat(d) for d in body["bars"]["sessions"]]
    assert body["ticker"] == "AAPL"
    assert sessions[0] >= date.today() - timedelta(days=ma.PERIOD_DAYS["3M"])
    # The first bar kept already has a settled indicator, not a warm-up gap.
    assert all(len(column) == len(sessions) for column in body["bars"].values())


def test_the_scanners_and_default_anchors_come_with_the_window(client, prices):
    body = _strict(client.get("/api/momentum/AAPL"))
    assert set(body["scanners"]) == {
        "macd_momentum", "macd_trend", "rsi_trend", "rsi_failure_swings"
    }
    first, second = body["default_anchors"]["first"], body["default_anchors"]["second"]
    sessions = body["bars"]["sessions"]
    assert second == sessions[-1]
    assert first == sessions[-momentum.DEFAULT_ANCHOR_SESSIONS]


def test_a_short_window_clamps_the_first_anchor_rather_than_indexing_off_it(client, prices):
    body = _strict(client.get("/api/momentum/AAPL", params={"period": "5D"}))
    sessions = body["bars"]["sessions"]
    assert len(sessions) < momentum.DEFAULT_ANCHOR_SESSIONS
    assert body["default_anchors"] == {"first": sessions[0], "second": sessions[-1]}


def test_the_periods_in_the_schema_are_the_logic_layers(client):
    schema = client.get("/api/openapi.json").json()
    period = next(
        p for p in schema["paths"]["/api/momentum/{ticker}"]["get"]["parameters"]
        if p["name"] == "period"
    )
    assert period["schema"]["enum"] == list(ma.PERIOD_DAYS)


def test_an_unknown_period_is_refused(client, prices):
    assert client.get("/api/momentum/AAPL", params={"period": "2Y"}).status_code == 422
    assert prices["calls"] == 0


def test_a_malformed_ticker_never_reaches_sharadar(client, prices):
    assert client.get("/api/momentum/AAPL;DROP").status_code == 422
    assert prices["calls"] == 0


# ---------------------------------------------------------------------------
# Failures say what happened
# ---------------------------------------------------------------------------


def test_no_rows_is_a_404_with_the_pages_own_message(client, prices):
    prices["rows"] = pd.DataFrame()
    response = client.get("/api/momentum/NOPE")
    assert response.status_code == 404
    assert response.json() == {"detail": "No data found for this ticker."}


def test_too_little_history_for_the_indicators_is_said_so(client, prices):
    prices["rows"] = _rows(start=str(date.today() - timedelta(days=10)))
    response = client.get("/api/momentum/NEW")
    assert response.status_code == 422
    assert "Not enough price history" in response.json()["detail"]


def test_a_sharadar_failure_carries_the_clients_message_and_never_the_key(
    client, monkeypatch
):
    def fail(ticker, *, table, start=None, end=None):
        raise sc.SharadarError("Nasdaq Data Link refused access (403).", status=403,
                               table="SFP", ticker=ticker)

    monkeypatch.setenv("SHARADAR_API_KEY", "sekrit-key")
    monkeypatch.setattr(sh_prices, "_price_table_for", lambda ticker: "SFP")
    monkeypatch.setattr(sh_prices, "_fetch_price_rows", fail)
    response = client.get("/api/momentum/AAPL")
    assert response.status_code == 502
    assert "refused access" in response.json()["detail"]
    assert "sekrit" not in response.text


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------


def test_divergence_reads_both_indicators_between_the_anchors(client, prices):
    window = _strict(client.get("/api/momentum/AAPL"))
    first, second = window["default_anchors"].values()
    body = _strict(client.get(
        "/api/momentum/AAPL/divergence",
        params={"first": first, "second": second, "anchor": "Highs"},
    ))
    sessions = window["bars"]["sessions"]
    a, b = body["first"], body["second"]
    assert (a["session"], b["session"]) == (first, second)
    assert a["close"] == window["bars"]["close"][sessions.index(first)]
    expected = ma.analyze_divergence(a["close"], b["close"], a["rsi"], b["rsi"], "Highs")
    assert (body["rsi"]["state"], body["rsi"]["bias"]) == expected


def test_a_weekend_anchor_snaps_to_the_friday_before(client, prices):
    sessions = _strict(client.get("/api/momentum/AAPL"))["bars"]["sessions"]
    friday = next(
        date.fromisoformat(d) for d in reversed(sessions)
        if date.fromisoformat(d).weekday() == 4
        and date.fromisoformat(d) < date.fromisoformat(sessions[-1])
    )
    body = _strict(client.get(
        "/api/momentum/AAPL/divergence",
        params={"first": str(friday + timedelta(days=1)), "second": sessions[-1]},
    ))
    assert body["first"]["session"] == str(friday)


def test_an_anchor_before_the_window_is_refused_not_moved(client, prices):
    response = client.get(
        "/api/momentum/AAPL/divergence",
        params={"first": "2001-01-02", "second": str(date.today())},
    )
    assert response.status_code == 422
    assert "anchor dates" in response.json()["detail"]


def test_the_divergence_costs_no_second_request(client, prices):
    window = _strict(client.get("/api/momentum/AAPL"))
    first, second = window["default_anchors"].values()
    client.get("/api/momentum/AAPL/divergence", params={"first": first, "second": second})
    assert prices["calls"] == 1


# ---------------------------------------------------------------------------
# The contract refuses NaN rather than renaming it null
# ---------------------------------------------------------------------------


def test_a_nan_in_a_plain_float_field_fails_at_construction():
    with pytest.raises(ValueError, match="finite"):
        momentum.Anchor(session=date.today(), close=math.nan, macd=0.0, rsi=0.0)
    with pytest.raises(ValueError, match="finite"):
        momentum.Bars(**{field: [] for field in momentum.Bars.model_fields} | {"rsi": [math.inf]})


def test_finite_or_none_is_how_a_missing_number_is_said():
    assert contract.finite_or_none(math.nan) is None
    assert contract.finite_or_none(math.inf) is None
    assert contract.finite_or_none(None) is None
    assert contract.finite_or_none(1.5) == 1.5
