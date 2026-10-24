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
