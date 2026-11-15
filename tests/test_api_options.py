"""The 🎯 Options routes, over a local store filled by the collector's own writer.

The store holds three synthetic names priced from a known smile
(``tests/options_fixtures.py``); the Discover and Screen tables in it are written
by ``collector.pipeline.write_screens``, so what these tests read is exactly
what the nightly job leaves. Every body is parsed the way a browser would, so
a NaN that slipped past the contract fails here.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import options_layout as layout
from app.api import main
from app.api import options as options_api
from app.data import options_store
from app.logic.options import screens, snapshot, universe
from app.logic.options.history import METRIC_COLUMNS
from app.logic.options.tolerances import DEFAULT
from collector import pipeline
from collector.store import LocalStore
from tests import options_fixtures as fx

SYMBOLS = ("GLD", "SPY", "MU")
SATURDAY = date(2026, 10, 24)  # the morning after fx.SESSION's close


def _metrics(sessions: int = 200) -> pd.DataFrame:
    days = [d.date() for d in pd.bdate_range(end=fx.SESSION, periods=sessions)]
    rng = np.random.default_rng(5)
    return pd.DataFrame([
        {c: np.nan for c in METRIC_COLUMNS} | {
            "symbol": s, "session": d, "spot": fx.SPOT, "atm_iv_30": 0.2 + 0.02 * rng.standard_normal(),
            "rv_21": 0.18, "skew_30": 0.01 * rng.standard_normal(),
            "reward_risk": 1 + 0.1 * rng.standard_normal(),
        }
        for s in SYMBOLS for d in days
    ])


def _rates() -> pd.DataFrame:
    return pd.DataFrame({"session": [fx.SESSION] * 3, "tenor": [1 / 12, 1.0, 5.0], "yield_pct": fx.YIELD_PCT})


def _chains() -> pd.DataFrame:
    return pd.concat(
        [fx.build_chain(symbol="GLD"), fx.build_chain(symbol="SPY", slope=-0.2), fx.build_chain(symbol="MU")],
        ignore_index=True,
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = LocalStore(tmp_path)
    chains, metrics, rates = _chains(), _metrics(), _rates()
    s.write_frame(layout.LATEST_CHAINS, chains)
    s.write_frame(layout.METRICS, metrics)
    s.write_frame(layout.RATES, rates)
    s.write_frame(layout.EARNINGS, pd.DataFrame({
        "symbol": ["MU"], "next_earnings": [fx.SESSION + timedelta(days=10)], "fetched": [fx.SESSION],
    }))
    screens_entry = pipeline.write_screens(s, fx.SESSION, chains, metrics, rates, log=lambda m: None)
    s.write_json(layout.MANIFEST, {
        "last_session": fx.SESSION.isoformat(), "written_at": "2026-10-24T06:00:00+00:00",
        "symbols": {"GLD": {"error": None}, "SPY": {"error": None}, "MU": {"error": None},
                    "SLV": {"error": "option EOD SLV: INVALID_ARGUMENT"}},
        "screens": screens_entry,
    })
    monkeypatch.setenv("OPTIONS_STORE_URI", f"file://{tmp_path}")
    monkeypatch.setattr(options_api, "_today_et", lambda: SATURDAY)
    return s


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("EQUITI_AUTH", "off")
    monkeypatch.delenv("K_SERVICE", raising=False)
    return TestClient(main.app, raise_server_exceptions=False)


def _strict(response) -> dict:
    def refuse(token):
        raise AssertionError(f"non-JSON constant {token} in the response")

    assert response.status_code == 200, response.text
    return json.loads(response.text, parse_constant=refuse)


def _solved():
    return snapshot.build(_chains(), _metrics(), _rates(), fx.SESSION)[0]


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


def test_the_overview_names_the_close_and_hides_a_name_reporting_soon(client, store):
    body = _strict(client.get("/api/options/overview"))
    assert body["session"] == "2026-10-23" and body["sessions_behind"] == 0
    assert [n["symbol"] for n in body["names"]] == ["SPY", "MU", "GLD"]  # universe order
    assert body["visible"] == ["SPY", "GLD"]
    assert body["hidden_for_earnings"] == ["MU"] and body["earnings_known"] is True
    assert next(n for n in body["names"] if n["symbol"] == "MU")["next_earnings"] == "2026-11-02"


def test_a_failed_name_is_listed_with_its_reason_and_the_rest_as_not_collected(client, store):
    body = _strict(client.get("/api/options/overview"))
    assert body["unavailable"] == [{"symbol": "SLV", "reason": "option EOD SLV: INVALID_ARGUMENT"}]
    assert len(body["not_collected"]) == len(universe.UNIVERSE) - 4


def test_the_filters_are_applied_on_the_server(client, store):
    body = _strict(client.get("/api/options/overview", params={"earnings_window": 0}))
    assert body["visible"] == ["SPY", "MU", "GLD"] and body["hidden_for_earnings"] == []
    body = _strict(client.get("/api/options/overview", params={"themes": ["Metals"]}))
    # MU is not in the theme kept, so the earnings window took nothing out of it.
    assert body["visible"] == ["GLD"] and body["hidden_for_earnings"] == []


def test_an_unknown_theme_is_refused(client, store):
    response = client.get("/api/options/overview", params={"themes": ["Metals", "Gold"]})
    assert response.status_code == 422
    assert "Gold" in response.json()["detail"]


def test_a_store_two_sessions_behind_says_so(client, store, monkeypatch):
    monkeypatch.setattr(options_api, "_today_et", lambda: date(2026, 10, 28))
    assert _strict(client.get("/api/options/overview"))["sessions_behind"] == 2


def test_screens_solved_from_an_older_close_are_refused_not_served(client, store):
    manifest = store.read_json(layout.MANIFEST)
    manifest["last_session"] = "2026-10-26"  # an older collector advanced the chain alone
    store.write_json(layout.MANIFEST, manifest)
    for route in ("overview", "discover", "screen"):
        response = client.get(f"/api/options/{route}")
        assert response.status_code == 503
        assert "2026-10-26" in response.json()["detail"] and "2026-10-23" in response.json()["detail"]


def test_an_unconfigured_store_is_a_503_that_says_how_to_configure_it(client, monkeypatch):
    monkeypatch.delenv("OPTIONS_STORE_URI", raising=False)
    monkeypatch.setattr(options_store.st, "secrets", {})
    response = client.get("/api/options/overview")
    assert response.status_code == 503
    assert "OPTIONS_STORE_URI" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Discover and Screen
# ---------------------------------------------------------------------------


def test_discover_rows_are_the_logic_layers_table_for_the_visible_names(client, store):
    body = _strict(client.get("/api/options/discover"))
    assert [r["symbol"] for r in body["rows"]] == sorted(
        ["SPY", "GLD"], key=lambda s: -next(r["score"] for r in body["rows"] if r["symbol"] == s)
    )
    table = snapshot.discover_table(_solved(), _metrics()).set_index("symbol")
    for row in body["rows"]:
        expected = table.loc[row["symbol"]]
        assert row["score"] == pytest.approx(expected["score"])
        assert row["skew_pct"] == pytest.approx(expected["skew_pct"])
        assert row["call"]["strike"] == expected["strike"]
        assert row["call"]["expiration"] == expected["expiration"].isoformat()
        assert row["call"]["annualised"] == pytest.approx(expected["annualised"])


def test_a_name_without_a_year_of_history_has_no_score_but_still_a_call(client, store):
    metrics = _metrics(sessions=30)
    pipeline.write_screens(store, fx.SESSION, _chains(), metrics, _rates(), log=lambda m: None)
    rows = _strict(client.get("/api/options/discover"))["rows"]
    assert all(r["score"] is None and r["rr_pct"] is None for r in rows)
    assert all(r["call"] is not None for r in rows)


def test_the_screen_opens_on_the_default_expiry_and_matches_the_spread_table(client, store):
    body = _strict(client.get("/api/options/screen"))
    visible = {s: snap for s, snap in _solved().items() if s in ("SPY", "GLD")}
    choices = snapshot.common_expirations(snapshot.listed_expirations(visible))
    assert body["expirations"] == [e.isoformat() for e in choices]
    expected_expiry = snapshot.default_spread_expiry(choices, fx.SESSION)
    assert body["expiration"] == expected_expiry.isoformat()
    table = snapshot.spread_table(visible, expected_expiry, 1.0).set_index("symbol")
    assert body["rows"] and {r["symbol"] for r in body["rows"]} == set(table.index)
    for row in body["rows"]:
        assert row["payout_ask"] == pytest.approx(table.loc[row["symbol"], "payout_ask"])
        assert row["long_strike"] < row["short_strike"]
    payouts = [r["payout_ask"] for r in body["rows"]]
    assert payouts == sorted(payouts, reverse=True)
    assert body["visible_count"] == 2


def test_the_screen_answers_another_target_and_a_chosen_expiry(client, store):
    first = _strict(client.get("/api/options/screen"))["expirations"][0]
    body = _strict(client.get("/api/options/screen", params={"sd": 2, "expiration": first}))
    assert body["sd"] == 2 and body["expiration"] == first
    assert body["sds"] == list(snapshot.SD_TARGETS)


def test_an_expiry_that_is_not_a_choice_falls_back_to_the_default(client, store):
    default = _strict(client.get("/api/options/screen"))["expiration"]
    body = _strict(client.get("/api/options/screen", params={"expiration": "2030-01-18"}))
    assert body["expiration"] == default


def test_a_target_the_collector_did_not_solve_is_refused(client, store):
    assert client.get("/api/options/screen", params={"sd": 3}).status_code == 422


# ---------------------------------------------------------------------------
# Evaluate and History
# ---------------------------------------------------------------------------


def test_evaluate_is_the_replacement_grid_in_both_modes(client, store):
    body = _strict(client.get("/api/options/evaluate/gld"))
    snap = _solved()["GLD"]
    columns = screens.grid_expiries(snap.surface)
    assert [c["label"] for c in body["columns"]] == [label for label, _ in columns]
    assert [r["upside"] for r in body["single"]] == list(screens.GRID_UPSIDES)
    for mode, spread in (("single", False), ("spread", True)):
        grid = screens.replacement_grid(snap.quotes, columns, fx.SESSION, snap.spot, spread=spread)
        for row in body[mode]:
            for (label, _), cell in zip(columns, row["cells"]):
                expected = grid[row["upside"]][label]
                if expected.cost_pct is None:
                    assert cell["cost_pct"] is None and cell["annualised"] is None
                else:
                    assert cell["cost_pct"] == pytest.approx(expected.cost_pct)
    assert body["spread_width"] == screens.GRID_SPREAD_WIDTH


def test_evaluate_refuses_a_name_outside_the_universe_and_explains_a_failed_one(client, store):
    assert client.get("/api/options/evaluate/ZZZZ").status_code == 404
    response = client.get("/api/options/evaluate/SLV")
    assert response.status_code == 404
    assert "INVALID_ARGUMENT" in response.json()["detail"]


def test_history_is_the_trailing_year_with_its_percentile(client, store):
    body = _strict(client.get("/api/options/history/GLD"))
    assert len(body["points"]) == 200 and body["points"][-1]["session"] == "2026-10-23"
    assert body["stored"] == 200 and body["needed"] == DEFAULT.min_history_sessions
    stats = body["stats"]
    assert stats["sessions"] == 200 and 0 < stats["percentile"] <= 1
    assert stats["quartile"] in ("worst quartile", "below median", "above median", "best quartile")
    assert stats["q25"] <= stats["median"] <= stats["q75"]


def test_a_short_history_has_no_statistics_and_says_how_much_is_stored(client, store):
    store.write_frame(layout.METRICS, _metrics(sessions=30))
    body = _strict(client.get("/api/options/history/GLD"))
    assert body["stats"] is None and body["stored"] == 30 and len(body["points"]) == 30
