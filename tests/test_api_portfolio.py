"""The 📊 statistics, 📈 optimisation and risk-free routes, through the real fetchers.

Only the price rows are faked, one level below ``fetch_price_history``, so the
routing, the column join and the cache run for real. Every body is parsed the
way a browser would, so a NaN that slipped past the contract fails here.
"""

from __future__ import annotations

import json
import math
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import main, portfolio
from app.data import market_data
from app.data import sharadar_client as sc
from app.data.sharadar import prices as sh_prices
from app.logic import portfolio_statistics as ps
from app.provenance import Sourced

START, END = "2024-01-02", "2026-06-30"


def _series(drift: float, swing: float, phase: float) -> pd.DataFrame:
    dates = pd.bdate_range("2014-01-01", END)
    close = [
        100 * math.exp(drift * i + swing * math.sin(i / 9 + phase)) for i in range(len(dates))
    ]
    return pd.DataFrame({"date": dates, "close": close, "closeadj": close})


#: Four assets that move differently, and the benchmark.
UNIVERSE = {
    "QQQ": _series(0.0006, 0.05, 0.0),
    "GLD": _series(0.0003, 0.03, 1.5),
    "XLE": _series(0.0002, 0.06, 3.0),
    "TLT": _series(0.0001, 0.02, 4.5),
    ps.MARKET_TICKER: _series(0.0004, 0.04, 0.3),
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("EQUITI_AUTH", "off")
    monkeypatch.delenv("K_SERVICE", raising=False)
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def prices(monkeypatch):
    """Answers each ticker from ``universe``, sliced to the request; unknown ones get no rows."""
    state = {"universe": dict(UNIVERSE), "requested": []}

    def fetch(ticker, *, table, start=None, end=None):
        state["requested"].append(ticker)
        rows = state["universe"].get(ticker)
        if rows is None:
            return pd.DataFrame()
        return rows[(rows["date"] >= start) & (rows["date"] <= end)].reset_index(drop=True)

    monkeypatch.setattr(
        sh_prices.reference, "fetch_price_tables", lambda tickers: dict.fromkeys(tickers, "SFP")
    )
    monkeypatch.setattr(sh_prices, "_fetch_price_rows", fetch)
    return state


def _strict(response) -> dict:
    def refuse(token):
        raise AssertionError(f"non-JSON constant {token} in the response")

    assert response.status_code == 200, response.text
    return json.loads(response.text, parse_constant=refuse)


def _statistics(**overrides) -> dict:
    params = {
        "tickers": ["QQQ", "GLD", "XLE"],
        "weights": [55, 30, 15],
        "start": START,
        "end": END,
        "risk_free": 0.04,
    }
    return params | overrides


def _optimization(**overrides) -> dict:
    params = {"tickers": ["QQQ", "GLD", "XLE"], "start": START, "end": END, "risk_free": 0.04}
    return params | overrides


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_statistics_match_the_logic_layers_table(client, prices):
    body = _strict(client.get("/api/portfolio/statistics", params=_statistics()))
    data = sc.fetch_price_history(("QQQ", "GLD", "XLE", "SPY"), "2016-06-30", END)
    table = ps.get_comprehensive_metrics(
        data, {"QQQ": 55, "GLD": 30, "XLE": 15}, START, END, 0.04
    )
    assert body["windows"] == list(table.columns)
    assert [row["name"] for row in body["metrics"]] == list(ps.METRICS)
    sharpe = next(row for row in body["metrics"] if row["name"] == "Sharpe Ratio")
    assert sharpe["unit"] == "ratio"
    assert sharpe["values"] == pytest.approx(list(table.loc["Sharpe Ratio"]))


def test_the_fetch_reaches_back_ten_years_for_the_longest_window(client, prices):
    body = _strict(client.get("/api/portfolio/statistics", params=_statistics()))
    ten_year = body["windows"].index("10 Year")
    assert all(row["values"][ten_year] is not None for row in body["metrics"])


def test_weights_are_returned_as_the_fractions_actually_used(client, prices):
    body = _strict(client.get("/api/portfolio/statistics", params=_statistics(weights=[45, 30, 15])))
    assert body["rescaled"] is True
    assert body["entered_total"] == 90
    assert {h["ticker"]: h["weight"] for h in body["holdings"]} == pytest.approx(
        {"QQQ": 0.5, "GLD": 1 / 3, "XLE": 1 / 6}
    )


def test_a_ticker_without_prices_is_named_and_left_out(client, prices):
    params = _statistics(tickers=["QQQ", "NOPE", "GLD"], weights=[50, 25, 25])
    body = _strict(client.get("/api/portfolio/statistics", params=params))
    assert body["missing"] == ["NOPE"]
    assert [h["ticker"] for h in body["holdings"]] == ["QQQ", "GLD"]
    assert body["relationships"]["correlation"]["tickers"] == ["QQQ", "GLD"]
    # The weights given summed to 100, so the missing one is not a rescaling.
    assert body["rescaled"] is False


def test_the_relationships_are_square_and_a_diagonal_correlation_is_one(client, prices):
    body = _strict(client.get("/api/portfolio/statistics", params=_statistics()))
    correlation = body["relationships"]["correlation"]
    assert correlation["tickers"] == ["QQQ", "GLD", "XLE"]
    assert [round(correlation["values"][i][i], 9) for i in range(3)] == [1, 1, 1]
    assert len(body["relationships"]["covariance"]["values"]) == 3


def test_one_holding_has_metrics_but_no_relationships(client, prices):
    body = _strict(client.get("/api/portfolio/statistics", params=_statistics(tickers=["QQQ"], weights=[100])))
    assert body["relationships"] is None
    assert body["metrics"][0]["values"][0] is not None


def test_without_the_benchmark_the_market_rows_are_null_not_zero(client, prices):
    del prices["universe"][ps.MARKET_TICKER]
    body = _strict(client.get("/api/portfolio/statistics", params=_statistics()))
    assert body["benchmark_priced"] is False
    beta = next(row for row in body["metrics"] if row["name"] == "Beta")
    assert beta["values"] == [None] * len(body["windows"])


def test_no_priced_holding_is_a_404(client, prices):
    response = client.get(
        "/api/portfolio/statistics", params=_statistics(tickers=["NOPE"], weights=[100])
    )
    assert response.status_code == 404
    assert "check your tickers" in response.json()["detail"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"weights": [55, 45]}, "one weight for each ticker"),
        ({"weights": [0, 0, 0]}, "more than zero"),
        ({"tickers": ["QQQ", "qqq", "GLD"]}, "only once"),
        ({"start": END, "end": START}, "Start Date is before"),
    ],
)
def test_inconsistent_inputs_are_refused_before_any_fetch(client, prices, overrides, message):
    response = client.get("/api/portfolio/statistics", params=_statistics(**overrides))
    assert response.status_code == 422
    assert message in response.json()["detail"]
    assert prices["requested"] == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"tickers": ["QQQ;DROP"], "weights": [100]},
        {"weights": [-10, 60, 50]},
        {"tickers": [f"T{i}" for i in range(portfolio.MAX_ASSETS + 1)],
         "weights": [1] * (portfolio.MAX_ASSETS + 1)},
        {"risk_free": 4},
    ],
)
def test_malformed_inputs_never_reach_sharadar(client, prices, overrides):
    response = client.get("/api/portfolio/statistics", params=_statistics(**overrides))
    assert response.status_code == 422
    assert prices["requested"] == []


# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------


def test_all_four_strategies_come_back_with_allocations_summing_to_one(client, prices):
    body = _strict(client.get("/api/portfolio/optimization", params=_optimization()))
    assert [s["strategy"] for s in body["strategies"]] == [
        "sharpe", "geometric_sharpe", "volatility_target", "kelly"
    ]
    for strategy in body["strategies"]:
        assert strategy["reason"] is None
        assert sum(w["weight"] for w in strategy["allocation"]) == pytest.approx(1, abs=1e-3)
        assert [y["year"] for y in strategy["yearly_returns"]] == [2024, 2025, 2026]


def test_kelly_carries_cash_and_no_ratio(client, prices):
    body = _strict(
        client.get("/api/portfolio/optimization", params=_optimization(kelly_fraction=0.3))
    )
    kelly = body["strategies"][-1]
    assert kelly["ratio"] is None
    assert kelly["allocation"][-1]["ticker"] == "CASH"
    # A 0.3 fraction of an allocation that cannot exceed one leaves at least 70% in cash.
    assert kelly["allocation"][-1]["weight"] >= 0.7 - 1e-6


def test_an_unreachable_volatility_cap_fails_only_its_own_strategy(client, prices):
    body = _strict(
        client.get("/api/portfolio/optimization", params=_optimization(vol_target=0.0001))
    )
    capped = body["strategies"][2]
    assert capped["reason"]
    assert capped["allocation"] == [] and capped["expected_return"] is None
    assert body["strategies"][0]["reason"] is None


def test_the_range_actually_priced_is_reported(client, prices):
    body = _strict(
        client.get("/api/portfolio/optimization", params=_optimization(tickers=["QQQ", "GLD", "NOPE"]))
    )
    assert body["tickers"] == ["QQQ", "GLD"]
    assert body["missing"] == ["NOPE"]
    assert body["first_session"] == START
    assert date.fromisoformat(body["last_session"]) <= date.fromisoformat(END)


def test_fewer_than_two_priced_assets_is_refused(client, prices):
    response = client.get(
        "/api/portfolio/optimization", params=_optimization(tickers=["QQQ", "NOPE"])
    )
    assert response.status_code == 422
    assert "Fewer than two" in response.json()["detail"]


def test_one_ticker_is_refused_before_any_fetch(client, prices):
    response = client.get("/api/portfolio/optimization", params=_optimization(tickers=["QQQ"]))
    assert response.status_code == 422
    assert prices["requested"] == []


# ---------------------------------------------------------------------------
# The risk-free rate
# ---------------------------------------------------------------------------


def test_the_risk_free_rate_comes_with_its_source(client, monkeypatch):
    monkeypatch.setattr(
        market_data,
        "fetch_current_risk_free_rate",
        lambda: Sourced.reported(0.0412, "Yahoo Finance ^IRX", as_of=date(2026, 11, 13)),
    )
    body = _strict(client.get("/api/rates/risk-free"))
    assert body == {
        "value": 0.0412, "source": "Yahoo Finance ^IRX", "as_of": "2026-11-13", "detail": ""
    }


def test_a_failed_rate_is_null_with_the_reason_never_a_stand_in(client, monkeypatch):
    monkeypatch.setattr(
        market_data,
        "fetch_current_risk_free_rate",
        lambda: Sourced.unavailable("Yahoo Finance ^IRX", "Yahoo did not answer"),
    )
    body = _strict(client.get("/api/rates/risk-free"))
    assert body["value"] is None
    assert body["detail"] == "Yahoo did not answer"
