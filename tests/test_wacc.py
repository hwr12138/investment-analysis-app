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
    assert result.index_ticker == "SPY"


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


# ---------------------------------------------------------------------------
# Beta uncertainty (DCF_MODEL_REVIEW.md F15)
# ---------------------------------------------------------------------------


def test_beta_standard_error_matches_an_independent_ols(paired_prices):
    stock, index = paired_prices
    result = wl.regress_beta(stock, index, frequency="monthly")
    y, x = wl.align_returns(
        wl.periodic_returns(stock, "monthly"), wl.periodic_returns(index, "monthly")
    )
    slope, intercept = np.polyfit(x.to_numpy(), y.to_numpy(), 1)
    residuals = y.to_numpy() - (intercept + slope * x.to_numpy())
    expected = np.sqrt(
        (residuals**2).sum() / (len(y) - 2) / ((x.to_numpy() - x.mean()) ** 2).sum()
    )
    assert result.raw_standard_error == pytest.approx(expected)
    low, high = result.confidence_interval
    assert low < result.beta < high


def test_blume_scales_the_standard_error_with_the_beta(paired_prices):
    stock, index = paired_prices
    raw = wl.regress_beta(stock, index, frequency="monthly")
    adjusted = wl.regress_beta(stock, index, frequency="monthly", blume=True)
    assert adjusted.standard_error == pytest.approx(0.67 * raw.standard_error)


# ---------------------------------------------------------------------------
# Synthetic-rating cost of debt (DCF_MODEL_REVIEW.md F10)
# ---------------------------------------------------------------------------


def test_coverage_maps_to_a_rating_and_a_spread_over_the_risk_free_rate():
    strong = wl.synthetic_cost_of_debt(ebit=1000.0, interest_expense=50.0, risk_free_rate=0.04)
    weak = wl.synthetic_cost_of_debt(ebit=160.0, interest_expense=100.0, risk_free_rate=0.04)
    assert strong.rating == "AAA" and strong.coverage == pytest.approx(20.0)
    assert weak.rating == "B"
    assert strong.cost_of_debt == pytest.approx(0.04 + strong.spread)
    assert weak.cost_of_debt > strong.cost_of_debt


def test_every_band_boundary_rates_into_the_band_it_opens():
    for band in wl.SYNTHETIC_RATINGS[:-1]:
        result = wl.synthetic_cost_of_debt(band.min_coverage, 1.0, 0.04)
        assert result.rating == band.rating


def test_ratings_worsen_monotonically_down_the_table():
    spreads = [band.spread for band in wl.SYNTHETIC_RATINGS]
    coverages = [band.min_coverage for band in wl.SYNTHETIC_RATINGS]
    assert spreads == sorted(spreads)
    assert coverages == sorted(coverages, reverse=True)


def test_no_interest_line_is_not_a_rating():
    """Missing stays missing: nothing is substituted for an unreported figure."""
    assert wl.synthetic_cost_of_debt(1000.0, None, 0.04) is None
    assert wl.synthetic_cost_of_debt(None, 50.0, 0.04) is None


def test_zero_interest_with_profits_rates_at_the_top():
    result = wl.synthetic_cost_of_debt(1000.0, 0.0, 0.04)
    assert result.rating == "AAA" and result.coverage is None
    assert wl.synthetic_cost_of_debt(-10.0, 0.0, 0.04) is None


def test_a_loss_maker_rates_at_the_bottom():
    assert wl.synthetic_cost_of_debt(-50.0, 100.0, 0.04).rating == "D"


def test_the_spreads_are_damodarans_january_2026_large_firm_table():
    """DCF_ROUND2_INSTRUCTIONS.md §2.5, checked against the live page."""
    assert wl.SYNTHETIC_RATINGS_AS_OF == "January 2026"
    assert "ratings.html" in wl.SYNTHETIC_RATINGS_SOURCE
    assert {band.rating: band.spread for band in wl.SYNTHETIC_RATINGS} == pytest.approx({
        "AAA": 0.0040, "AA": 0.0055, "A+": 0.0070, "A": 0.0078, "A-": 0.0089,
        "BBB": 0.0111, "BB+": 0.0138, "BB": 0.0184, "B+": 0.0275, "B": 0.0321,
        "B-": 0.0509, "CCC": 0.0885, "CC": 0.1261, "C": 0.1600, "D": 0.1900,
    })


def test_zero_interest_beside_real_debt_reads_as_undisclosed():
    """Apple carries ~$84bn of debt and Sharadar reports its interest as 0."""
    assert wl.interest_looks_undisclosed(0.0, 84e9)
    assert wl.synthetic_cost_of_debt(150e9, 0.0, 0.04, total_debt=84e9) is None


def test_zero_interest_with_no_debt_still_rates_at_the_top():
    for debt in (0.0, None):
        assert not wl.interest_looks_undisclosed(0.0, debt)
        assert wl.synthetic_cost_of_debt(150e9, 0.0, 0.04, total_debt=debt).rating == "AAA"


# ---------------------------------------------------------------------------
# Latest disclosed coverage (DCF_ROUND3_INSTRUCTIONS.md §2.7)
# ---------------------------------------------------------------------------


def test_a_year_that_disclosed_interest_two_years_back_is_used():
    history = [
        ("FY2021", 100.0, 4.0), ("FY2022", 120.0, 2.0),
        ("FY2023", 130.0, 0.0), ("FY2024", 140.0, 0.0),
    ]
    assert wl.latest_disclosed_coverage(history) == ("FY2022", 120.0, 2.0)
    found = wl.synthetic_cost_of_debt(120.0, 2.0, 0.045)
    labelled = wl.SyntheticCostOfDebt(**{**found.__dict__, "coverage_period": "FY2022"})
    assert labelled.formula.startswith("Coverage FY2022 (EBIT ÷ interest) 60.00× → AAA")
    assert labelled.cost_of_debt == pytest.approx(0.045 + 0.0040)


def test_a_year_four_years_back_is_not_reached_for():
    history = [
        ("FY2020", 100.0, 5.0), ("FY2021", 110.0, 0.0),
        ("FY2022", 120.0, 0.0), ("FY2023", 130.0, 0.0),
    ]
    assert wl.latest_disclosed_coverage(history) is None


# ---------------------------------------------------------------------------
# Missing base-period interest (DCF_ROUND4_INSTRUCTIONS.md §2.5)
# ---------------------------------------------------------------------------


def test_a_missing_base_interest_line_uses_a_disclosed_year_inside_the_window():
    """ASML and SAP report no interest line at all — not zero, missing."""
    history = [("FY2023", 9000.0, 150.0), ("FY2024", 9500.0, None), ("FY2025", 10000.0, None)]
    found, cause = wl.resolve_synthetic_cost_of_debt(10500.0, None, 0.045, 5000.0, history)
    assert cause == "missing"
    assert found.coverage_period == "FY2023"
    assert found.rating == "AAA" and found.cost_of_debt == pytest.approx(0.045 + 0.0040)


def test_a_missing_base_interest_line_with_nothing_disclosed_in_the_window_is_a_placeholder():
    history = [("FY2021", 9000.0, 150.0), ("FY2022", 9000.0, None),
               ("FY2023", 9500.0, None), ("FY2024", 10000.0, None)]
    found, cause = wl.resolve_synthetic_cost_of_debt(10500.0, None, 0.045, 5000.0, history)
    assert found is None and cause == "missing"


def test_disclosed_base_interest_needs_no_lookback():
    found, cause = wl.resolve_synthetic_cost_of_debt(1000.0, 100.0, 0.045, 5000.0, [])
    assert cause == "" and found.coverage == pytest.approx(10.0)
    assert found.coverage_period is None


def test_the_table_date_matches_its_label():
    assert wl.SYNTHETIC_RATINGS_AS_OF_DATE.year == int(wl.SYNTHETIC_RATINGS_AS_OF.split()[-1])


def test_net_interest_income_is_not_rated_and_not_substituted():
    """ASML and SAP report a negative interest line — net interest income. It
    has no coverage ratio, and no other year is reached for without a rule."""
    history = [("FY2023", 9000.0, 456.0), ("FY2024", 9500.0, -398.0)]
    found, cause = wl.resolve_synthetic_cost_of_debt(10500.0, -534.0, 0.045, 5000.0, history)
    assert found is None and cause == "negative"
