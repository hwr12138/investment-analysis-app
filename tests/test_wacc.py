"""Beta regression and discount rate construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.logic.dcf import wacc as wl


def _price_series(start: str, periods: int, returns: np.ndarray, freq: str = "D") -> pd.Series:
    index = pd.date_range(start, periods=periods, freq=freq)
    prices = 100 * np.cumprod(1 + returns)
    return pd.Series(prices, index=index)


@pytest.fixture
def paired_prices():
    """A stock built to have a beta of exactly 1.5 against its index."""
    rng = np.random.default_rng(42)
    days = 1600
    index_returns = rng.normal(0.0004, 0.01, days)
    stock_returns = 1.5 * index_returns
    return (
        _price_series("2021-01-01", days, stock_returns),
        _price_series("2021-01-01", days, index_returns),
    )


def test_beta_recovers_a_known_slope(paired_prices):
    stock, index = paired_prices
    result = wl.regress_beta(stock, index, frequency="monthly")
    assert result.beta == pytest.approx(1.5, abs=0.02)
    assert result.r_squared == pytest.approx(1.0, abs=0.01)


def test_r_squared_is_reported_alongside_beta(paired_prices):
    stock, index = paired_prices
    result = wl.regress_beta(stock, index, frequency="monthly")
    assert 0.0 <= result.r_squared <= 1.0
    assert result.observations >= wl.MIN_MONTHLY_OBSERVATIONS
    assert result.index_ticker == "^GSPC"


def test_noise_produces_a_low_r_squared():
    rng = np.random.default_rng(7)
    days = 1600
    stock = _price_series("2021-01-01", days, rng.normal(0, 0.02, days))
    index = _price_series("2021-01-01", days, rng.normal(0, 0.01, days))
    result = wl.regress_beta(stock, index, frequency="monthly")
    assert result.r_squared < 0.10


def test_blume_adjustment(paired_prices):
    stock, index = paired_prices
    raw = wl.regress_beta(stock, index, frequency="monthly")
    adjusted = wl.regress_beta(stock, index, frequency="monthly", blume=True)
    assert adjusted.beta == pytest.approx(0.67 * raw.raw_beta + 0.33)
    assert adjusted.blume_adjusted is True
    assert adjusted.raw_beta == pytest.approx(raw.raw_beta)


def test_blume_pulls_toward_one():
    assert wl.blume_adjust(2.0) == pytest.approx(1.67)
    assert wl.blume_adjust(1.0) == pytest.approx(1.0)
    assert wl.blume_adjust(0.0) == pytest.approx(0.33)


def test_insufficient_observations_raises_rather_than_regressing():
    rng = np.random.default_rng(1)
    stock = _price_series("2025-01-01", 200, rng.normal(0, 0.01, 200))
    index = _price_series("2025-01-01", 200, rng.normal(0, 0.01, 200))
    with pytest.raises(wl.InsufficientObservationsError) as caught:
        wl.regress_beta(stock, index, frequency="monthly")
    assert caught.value.required == wl.MIN_MONTHLY_OBSERVATIONS
    assert caught.value.observations < wl.MIN_MONTHLY_OBSERVATIONS


def test_mismatched_dates_are_dropped_in_pairs_not_filled():
    """A date one series never traded on may not be invented."""
    rng = np.random.default_rng(3)
    days = 1600
    stock = _price_series("2021-01-01", days, rng.normal(0, 0.01, days))
    index = _price_series("2021-01-01", days, rng.normal(0, 0.01, days))
    trimmed = index.drop(index.index[100:400])

    stock_aligned, index_aligned = wl.align_returns(
        wl.periodic_returns(stock, "monthly"), wl.periodic_returns(trimmed, "monthly")
    )
    assert len(stock_aligned) == len(index_aligned)
    assert not stock_aligned.isna().any()
    assert not index_aligned.isna().any()
    assert stock_aligned.index.equals(index_aligned.index)


def test_weekly_frequency_is_supported(paired_prices):
    stock, index = paired_prices
    result = wl.regress_beta(stock, index, frequency="weekly")
    assert result.frequency == "weekly"
    assert result.beta == pytest.approx(1.5, abs=0.05)


def test_unknown_frequency_is_rejected():
    with pytest.raises(ValueError):
        wl.periodic_returns(pd.Series(dtype="float64"), "hourly")


def test_empty_price_series_returns_no_observations():
    assert wl.periodic_returns(pd.Series(dtype="float64"), "monthly").empty


# ---------------------------------------------------------------------------
# WACC
# ---------------------------------------------------------------------------


def test_cost_of_equity_capm():
    assert wl.cost_of_equity(0.04, 1.2, 0.05) == pytest.approx(0.10)


def test_wacc_by_hand():
    result = wl.compute_wacc(
        equity_value=800, debt_value=200, risk_free_rate=0.04, beta=1.0,
        equity_risk_premium=0.05, cost_of_debt=0.06, marginal_tax_rate=0.25,
    )
    # 0.8 × 0.09 + 0.2 × 0.06 × 0.75 = 0.072 + 0.009 = 0.081
    assert result.wacc == pytest.approx(0.081)
    assert result.cost_of_equity == pytest.approx(0.09)
    assert result.after_tax_cost_of_debt == pytest.approx(0.045)
    assert result.equity_weight == pytest.approx(0.8)


def test_zero_debt_gives_a_wacc_equal_to_cost_of_equity():
    result = wl.compute_wacc(
        equity_value=1000, debt_value=0, risk_free_rate=0.04, beta=1.1,
        equity_risk_premium=0.05, cost_of_debt=0.05, marginal_tax_rate=0.21,
    )
    assert result.wacc == pytest.approx(result.cost_of_equity)
    assert result.debt_weight == 0.0


def test_wacc_rejects_zero_capital():
    with pytest.raises(ValueError):
        wl.compute_wacc(
            equity_value=0, debt_value=0, risk_free_rate=0.04, beta=1.0,
            equity_risk_premium=0.05, cost_of_debt=0.05, marginal_tax_rate=0.21,
        )


def test_implied_cost_of_debt_uses_average_debt():
    assert wl.implied_cost_of_debt(50, 1200, 800) == pytest.approx(50 / 1000)


def test_implied_cost_of_debt_falls_back_to_current_debt():
    assert wl.implied_cost_of_debt(50, 1000, None) == pytest.approx(0.05)


def test_implied_cost_of_debt_is_none_without_debt():
    assert wl.implied_cost_of_debt(50, 0, 0) is None
    assert wl.implied_cost_of_debt(None, 1000) is None


def test_market_value_of_equity():
    assert wl.market_value_of_equity(100.0, 1_000_000) == 100_000_000
