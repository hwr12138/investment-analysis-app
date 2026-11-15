"""Which names the 🎯 Options page shows: the filters, and the names it lists as missing."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.logic.options import coverage, universe

SESSION = date(2026, 10, 23)


def _earnings():
    return pd.DataFrame({
        "symbol": ["MU", "NVDA", "TSLA", "AVGO", "META"],
        "next_earnings": [SESSION + timedelta(days=21), SESSION + timedelta(days=22), None,
                          SESSION - timedelta(days=1), SESSION],
    })


def test_a_name_reporting_inside_the_window_is_hidden_both_ends_included():
    assert coverage.hidden_for_earnings(_earnings(), SESSION, 21) == {"MU", "META"}
    assert coverage.hidden_for_earnings(_earnings(), SESSION, 7) == {"META"}


def test_no_window_or_no_dates_hides_nothing():
    assert coverage.hidden_for_earnings(_earnings(), SESSION, 0) == set()
    assert coverage.hidden_for_earnings(None, SESSION, 21) == set()
    assert coverage.hidden_for_earnings(_earnings().iloc[0:0], SESSION, 21) == set()


def test_visible_keeps_universe_order_and_the_chosen_themes():
    themes = {"MU": "Semiconductors", "GLD": "Metals", "SPY": "Benchmark", "ZZZZ": "Other"}
    assert coverage.visible(themes, None, set()) == ["SPY", "MU", "GLD"]  # ZZZZ is not in the universe
    assert coverage.visible(themes, {"Metals", "Semiconductors"}, {"MU"}) == ["GLD"]
    assert coverage.visible(themes, set(), set()) == []


def test_a_failed_fetch_and_an_unpriced_chain_are_both_listed_with_their_reason():
    manifest = {"symbols": {"GLD": {"error": None}, "SLV": {"error": "option EOD SLV: INVALID_ARGUMENT"}}}
    assert coverage.unavailable(manifest, {"SPY": "no close for the session"}) == {
        "SLV": "option EOD SLV: INVALID_ARGUMENT", "SPY": "no close for the session",
    }
    assert coverage.not_collected(manifest) == [s for s in universe.UNIVERSE if s not in ("GLD", "SLV")]
