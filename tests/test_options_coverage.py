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


def test_a_stored_name_takes_its_theme_from_the_universe_and_one_taken_out_is_dropped():
    # The order the names were stored in is not the order they are shown in.
    assert coverage.themes_of(["GLD", "ZZZZ", "MU", "SPY", "GLD"]) == {
        "SPY": universe.UNIVERSE["SPY"], "MU": universe.UNIVERSE["MU"], "GLD": universe.UNIVERSE["GLD"],
    }
    assert list(coverage.themes_of(["GLD", "MU", "SPY"])) == ["SPY", "MU", "GLD"]
    assert coverage.themes_of([]) == {}


def test_a_name_taken_out_of_the_universe_is_not_listed_as_missing():
    manifest = {"symbols": {"SLV": {"error": "option EOD SLV: INVALID_ARGUMENT"}, "ZZZZ": {"error": "no data"}}}
    assert coverage.unavailable(manifest, {"YYYY": "no close for the session"}) == {
        "SLV": "option EOD SLV: INVALID_ARGUMENT",
    }


def test_a_failed_fetch_and_an_unpriced_chain_are_both_listed_with_their_reason():
    manifest = {"symbols": {"GLD": {"error": None}, "SLV": {"error": "option EOD SLV: INVALID_ARGUMENT"}}}
    assert coverage.unavailable(manifest, {"SPY": "no close for the session"}) == {
        "SLV": "option EOD SLV: INVALID_ARGUMENT", "SPY": "no close for the session",
    }
    assert coverage.not_collected(manifest) == [s for s in universe.UNIVERSE if s not in ("GLD", "SLV")]
