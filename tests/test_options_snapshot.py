"""The cross-sectional tables the Discover and Screen tabs draw."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.logic.options import snapshot
from app.logic.options.history import METRIC_COLUMNS
from tests import options_fixtures as fx


def _rates():
    return pd.DataFrame({"session": [fx.SESSION] * 3, "tenor": [1 / 12, 1.0, 5.0], "yield_pct": fx.YIELD_PCT})


def _metrics(symbols, sessions=200):
    days = [d.date() for d in pd.bdate_range(end=fx.SESSION, periods=sessions)]
    rng = np.random.default_rng(3)
    rows = []
    for symbol in symbols:
        for day in days:
            rows.append({c: np.nan for c in METRIC_COLUMNS} | {
                "symbol": symbol, "session": day, "spot": fx.SPOT,
                "atm_iv_30": 0.2 + 0.02 * rng.standard_normal(), "rv_21": 0.18,
                "skew_30": 0.01 * rng.standard_normal(), "reward_risk": 1 + 0.1 * rng.standard_normal(),
            })
    return pd.DataFrame(rows)


def _chains():
    return pd.concat([fx.build_chain(symbol="GLD"), fx.build_chain(symbol="SPY", slope=-0.2)], ignore_index=True)


def test_build_solves_each_name_and_reports_the_ones_it_cannot():
    metrics = _metrics(["GLD"])  # SPY has no close
    snaps, missing = snapshot.build(_chains(), metrics, _rates(), fx.SESSION)
    assert set(snaps) == {"GLD"} and missing == {"SPY": "no close for the session"}
    assert snaps["GLD"].theme == "Metals" and snaps["GLD"].surface


def test_without_a_rate_curve_nothing_is_priced():
    snaps, missing = snapshot.build(_chains(), _metrics(["GLD", "SPY"]), _rates().iloc[0:0], fx.SESSION)
    assert not snaps and set(missing) == {"GLD", "SPY"}


def test_discover_table_has_a_score_percentiles_and_the_monthly_25_delta_call():
    metrics = _metrics(["GLD", "SPY"])
    snaps, _ = snapshot.build(_chains(), metrics, _rates(), fx.SESSION)
    table = snapshot.discover_table(snaps, metrics).set_index("symbol")
    assert table.loc["GLD", "expiration"] == date(2026, 11, 20)
    assert table.loc["GLD", "delta"] == pytest.approx(0.25, abs=0.02)
    assert table["score"].notna().all() and table["skew_pct"].between(0, 1).all()


def test_a_short_history_has_no_score_but_still_a_call():
    metrics = _metrics(["GLD", "SPY"], sessions=30)
    snaps, _ = snapshot.build(_chains(), metrics, _rates(), fx.SESSION)
    table = snapshot.discover_table(snaps, metrics)
    assert table["score"].isna().all() and table["skew_pct"].isna().all()
    assert table["strike"].notna().all()


def test_spread_table_and_its_expiry_choices():
    metrics = _metrics(["GLD", "SPY"])
    snaps, _ = snapshot.build(_chains(), metrics, _rates(), fx.SESSION)
    assert date(2026, 12, 18) in snapshot.spread_expirations(snaps)
    table = snapshot.spread_table(snaps, date(2026, 12, 18), 1.0)
    assert set(table["symbol"]) == {"GLD", "SPY"}
    assert (table["payout_ask"] < table["payout_mid"]).all()


def test_build_can_solve_just_the_names_asked_for():
    snaps, missing = snapshot.build(_chains(), _metrics(["GLD", "SPY"]), _rates(), fx.SESSION, symbols=["SPY"])
    assert set(snaps) == {"SPY"} and not missing


def test_every_spread_is_the_spread_table_at_each_choice():
    snaps, _ = snapshot.build(_chains(), _metrics(["GLD", "SPY"]), _rates(), fx.SESSION)
    stored = snapshot.every_spread(snaps)
    assert set(stored["sd"]) == set(snapshot.SD_TARGETS)
    for expiration in snapshot.spread_expirations(snaps):
        for sd in snapshot.SD_TARGETS:
            picked = stored[(stored["expiration"] == expiration) & (stored["sd"] == sd)]
            pd.testing.assert_frame_equal(
                picked.drop(columns=["expiration", "sd"]).reset_index(drop=True),
                snapshot.spread_table(snaps, expiration, sd),
            )


def test_the_expiry_choices_count_only_the_names_given():
    d1, d2, d3 = date(2026, 11, 20), date(2026, 12, 18), date(2027, 1, 15)
    listed = pd.DataFrame({"symbol": ["A", "A", "B", "C", "D"], "expiration": [d1, d2, d1, d1, d3]})
    assert snapshot.common_expirations(listed) == [d1]  # four names: an expiry needs two
    assert snapshot.common_expirations(listed[listed["symbol"] == "A"]) == [d1, d2]
    assert snapshot.common_expirations(listed.iloc[0:0]) == []


def test_the_screen_opens_on_the_first_expiry_eight_weeks_out():
    choices = [date(2026, 10, 30), date(2026, 11, 20), date(2026, 12, 18), date(2027, 1, 15)]
    assert snapshot.default_spread_expiry(choices, fx.SESSION) == date(2026, 12, 18)
    assert snapshot.default_spread_expiry(choices[:2], fx.SESSION) == date(2026, 11, 20)
    assert snapshot.default_spread_expiry([], fx.SESSION) is None


def test_solve_screens_holds_every_table_and_names_what_it_could_not_price():
    solved = snapshot.solve_screens(_chains(), _metrics(["GLD"]), _rates(), fx.SESSION)
    assert list(solved.discover["symbol"]) == ["GLD"]
    assert solved.unpriced == {"SPY": "no close for the session"}
    assert set(solved.listed["symbol"]) == {"GLD"} and set(solved.spreads["symbol"]) == {"GLD"}


def test_a_night_with_nothing_priced_still_has_every_column():
    solved = snapshot.solve_screens(_chains(), _metrics(["GLD"]).iloc[0:0], _rates(), fx.SESSION)
    assert list(solved.discover.columns) == list(snapshot.DISCOVER_COLUMNS)
    assert solved.spreads.empty and list(solved.spreads.columns)[-2:] == ["expiration", "sd"]
