"""Portfolio statistics: hand-checkable figures, and ``None`` where one is undefined.

Each of the undefined cases below used to print as ``0.0000`` — beta against a
market that never moved, a Sharpe ratio with no volatility, every
market-relative figure when the benchmark's prices were missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.logic import portfolio_statistics as ps
from app.logic import returns as ret

MARKET_ONLY = ("Alpha", "Beta", "Treynor Ratio")


def _returns(values, start="2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype="float64")


@pytest.fixture
def market() -> pd.Series:
    return _returns(0.01 * np.sin(np.arange(300)))


def test_a_portfolio_twice_as_volatile_as_the_market_has_a_beta_of_two(market):
    metrics = ps.calculate_all_metrics(2 * market, market, rf_rate=0.0)
    assert metrics["Beta"] == pytest.approx(2.0)
    assert metrics["Alpha"] == pytest.approx(0.0, abs=1e-12)


def test_without_a_benchmark_the_market_figures_are_missing_not_zero(market):
    metrics = ps.calculate_all_metrics(market, None, rf_rate=0.02)
    for name in MARKET_ONLY:
        assert metrics[name] is None, name
    assert metrics["Sharpe Ratio"] is not None


def test_a_market_that_never_moved_has_no_beta(market):
    flat = _returns([0.0] * len(market))
    metrics = ps.calculate_all_metrics(market, flat, rf_rate=0.02)
    assert metrics["Beta"] is None
    assert metrics["Alpha"] is None
    assert metrics["Treynor Ratio"] is None


def test_a_riskless_portfolio_has_no_risk_adjusted_returns(market):
    metrics = ps.calculate_all_metrics(_returns([0.0] * len(market)), market, rf_rate=0.02)
    assert metrics["Standard Deviation"] == pytest.approx(0.0)
    assert metrics["Sharpe Ratio"] is None
    assert metrics["Geometric Sharpe Ratio"] is None
    assert metrics["Sortino Ratio"] is None


def test_every_metric_is_present_in_display_order(market):
    assert tuple(ps.calculate_all_metrics(market, market, 0.0)) == ps.METRICS


def _history(years: float) -> pd.DataFrame:
    sessions = int(years * 252)
    index = pd.bdate_range(end="2026-06-30", periods=sessions)
    swing = 0.01 * np.sin(np.arange(sessions))
    return pd.DataFrame(
        {
            "AAA": 100 * np.cumprod(1 + 0.0004 + swing),
            ps.MARKET_TICKER: 100 * np.cumprod(1 + 0.0003 + swing / 2),
        },
        index=index,
    )


def test_a_window_longer_than_the_history_is_not_reported():
    table = ps.get_comprehensive_metrics(
        _history(2), {"AAA": 1.0}, "2025-01-01", "2026-06-30", 0.02
    )
    assert table["10 Year"].isna().all()
    assert table["5 Year"].isna().all()
    assert table["1 Year"].notna().all()
    assert table["Custom"].notna().all()


def test_missing_benchmark_prices_empty_only_the_market_rows():
    data = _history(2).drop(columns=[ps.MARKET_TICKER])
    table = ps.get_comprehensive_metrics(data, {"AAA": 1.0}, "2025-01-01", "2026-06-30", 0.02)
    assert table.loc[list(MARKET_ONLY), "1 Year"].isna().all()
    assert table.loc["Sharpe Ratio", "1 Year"] is not None


def test_weights_are_renormalised_over_the_tickers_with_prices():
    data = _history(2)
    whole = ps.get_comprehensive_metrics(data, {"AAA": 100}, "2025-01-01", "2026-06-30", 0.0)
    missing = ps.get_comprehensive_metrics(
        data, {"AAA": 60, "GONE": 40}, "2025-01-01", "2026-06-30", 0.0
    )
    pd.testing.assert_frame_equal(whole, missing)


# ---------------------------------------------------------------------------
# The weights and the relationships
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("total", "rescaled"), [(100, False), (100.005, False), (1, False), (90, True), (0.5, True)]
)
def test_only_weights_off_both_wholes_are_rescaled(total, rescaled):
    assert ps.needs_rescaling(total) is rescaled


def test_weights_become_fractions_of_their_total():
    assert ps.normalised_weights({"A": 60, "B": 30}) == pytest.approx({"A": 2 / 3, "B": 1 / 3})
    assert ps.normalised_weights({"A": 0.4, "B": 0.6}) == {"A": 0.4, "B": 0.6}
    assert ps.normalised_weights({"A": 0.0}) == {"A": 0.0}


def test_the_fetch_reaches_back_to_the_longest_window():
    from datetime import date

    assert ps.history_start(date(2024, 1, 1), date(2026, 6, 30)) == date(2016, 6, 30)
    assert ps.history_start(date(2010, 1, 1), date(2026, 6, 30)) == date(2010, 1, 1)
    # A leap day has no counterpart ten years back, so it lands on the 28th.
    assert ps.history_start(date(2024, 1, 1), date(2024, 2, 29)) == date(2014, 2, 28)


def test_relationships_cover_only_the_custom_window():
    data = _history(2)
    related = ps.relationships(data, ["AAA", ps.MARKET_TICKER], "2026-01-01", "2026-06-30")
    window = data.loc["2026-01-01":"2026-06-30"].pct_change().dropna()
    assert list(related.correlation.index) == ["AAA", ps.MARKET_TICKER]
    assert related.correlation.loc["AAA", "AAA"] == pytest.approx(1.0)
    assert related.covariance.loc["AAA", "AAA"] == pytest.approx(window["AAA"].var() * ret.TRADING_DAYS)


def test_relationships_need_two_priced_assets_and_a_shared_day():
    data = _history(2)
    assert ps.relationships(data, ["AAA", "GONE"], "2025-01-01", "2026-06-30") is None
    assert ps.relationships(data, ["AAA", ps.MARKET_TICKER], "2030-01-01", "2030-06-30") is None
