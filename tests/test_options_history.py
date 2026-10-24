"""Session metrics, own-history statistics, the buy-vol score and the universe."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.logic.options import history, score, universe
from app.logic.options.tolerances import OptionsTolerances
from tests import options_fixtures as fx


def _closes(n=40, daily=0.01):
    # Alternating ±1% log returns: realized vol is exactly 1% × √252.
    steps = np.array([daily if i % 2 else -daily for i in range(n)])
    return pd.Series(100 * np.exp(np.cumsum(steps)))


def test_realized_vol_is_annualised_close_to_close():
    rv = history.realized_vol(_closes(), 21)
    returns = np.array([0.01 if i % 2 else -0.01 for i in range(21)])
    assert rv == pytest.approx(returns.std(ddof=1) * math.sqrt(252), rel=1e-6)


def test_realized_vol_is_nan_on_a_short_series():
    assert math.isnan(history.realized_vol(_closes(10), 21))


def test_session_metrics_reads_constant_maturity_and_the_one_month_spread():
    row = history.session_metrics("TEST", fx.build_chain(), fx.SESSION, fx.SPOT, fx.curve(), _closes())
    assert set(row) == set(history.METRIC_COLUMNS)
    assert row["atm_iv_30"] == pytest.approx(fx.atm_vol(30 / 365), abs=2e-4)
    assert row["skew_30"] == pytest.approx(row["call25_iv_30"] - row["atm_iv_30"])
    assert row["skew_30"] < 0  # the fixture's smile falls through the calls
    assert row["rr_expiration"] == date(2026, 11, 20)
    assert row["rr_payout"] > 0 and row["rr_fair_payout"] > 0
    assert 0.5 < row["reward_risk"] < 2.0  # smile-based, at constant maturity
    assert row["call25_cost_pct"] > 0


def test_session_metrics_leaves_unreadable_figures_nan():
    chain = fx.build_chain()
    near_only = chain[chain["expiration"] == date(2026, 10, 30)]
    row = history.session_metrics("TEST", near_only, fx.SESSION, fx.SPOT, fx.curve(), _closes())
    assert math.isnan(row["atm_iv_30"]) and math.isnan(row["reward_risk"])


def _series(values, end="2026-10-23"):
    return pd.Series(values, index=pd.bdate_range(end=end, periods=len(values)))


def test_own_history_percentile_counts_the_latest_value():
    stats = history.own_history_stats(_series(np.arange(1.0, 201.0)))
    assert stats.percentile == 1.0 and stats.high == 200.0
    stats = history.own_history_stats(_series(np.r_[np.arange(1.0, 200.0), 50.0]))
    assert stats.percentile == pytest.approx(51 / 200)


def test_own_history_uses_only_the_trailing_year():
    stats = history.own_history_stats(_series(np.r_[np.full(300, 1000.0), np.arange(1.0, 253.0)]))
    assert stats.sessions == 252 and stats.high == 252.0


def test_a_short_history_has_no_percentile():
    assert history.own_history_stats(_series(np.arange(1.0, 100.0))) is None


def _metrics(atm, rv, skew):
    n = len(atm)
    return pd.DataFrame({"atm_iv_30": atm, "rv_21": rv, "skew_30": skew}, index=pd.bdate_range(end="2026-10-23", periods=n))


def test_buy_vol_score_is_positive_when_all_three_readings_are_low():
    rng = np.random.default_rng(1)
    base = 0.20 + 0.02 * rng.standard_normal(200)
    atm = base.copy()
    atm[-1] = 0.14  # cheap IV
    rv = np.full(200, 0.18)
    skew = 0.01 * rng.standard_normal(200)
    skew[-1] = -0.03  # cheap call wing
    result = score.buy_vol_score(_metrics(atm, rv, skew))
    assert result.score > 50
    assert set(result.z) == {label for label, _ in score.COMPONENTS}
    assert all(z < 0 for z in result.z.values())


def test_buy_vol_score_formula():
    rng = np.random.default_rng(2)
    m = _metrics(0.2 + 0.02 * rng.standard_normal(150), 0.18 + 0.01 * rng.standard_normal(150),
                 0.01 * rng.standard_normal(150))
    result = score.buy_vol_score(m)
    assert result.score == pytest.approx(-score.SCALE * np.mean(list(result.z.values())))
    expected_iv_z = (m["atm_iv_30"].iloc[-1] - m["atm_iv_30"].mean()) / m["atm_iv_30"].std(ddof=1)
    assert result.z["IV level"] == pytest.approx(expected_iv_z)


def test_no_score_on_a_short_history():
    m = _metrics(np.full(50, 0.2), np.full(50, 0.18), np.full(50, 0.0))
    assert score.buy_vol_score(m) is None
    assert score.buy_vol_score(m, OptionsTolerances(min_history_sessions=10)) is None  # zero spread: no z


def test_universe_shape():
    assert len(universe.UNIVERSE) == len(set(universe.UNIVERSE))
    assert universe.SINGLE_STOCKS <= universe.UNIVERSE.keys()
    assert universe.INDICES <= universe.UNIVERSE.keys()
    assert set(universe.INDEX_ROOTS) <= universe.INDICES
    assert len(universe.universe_hash()) == 12
