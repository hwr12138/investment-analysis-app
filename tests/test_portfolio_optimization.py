"""The portfolio optimisers: their answers, and — mostly — how they fail.

The page used to receive ``(None, 0.0, 0.0, 0.0)`` from a failed optimisation,
which it could print as a 0% return at 0% volatility. Every failure below is
checked to carry a reason and no figures at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.logic import portfolio_optimization as po


def _prices(**daily_returns: np.ndarray) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=len(next(iter(daily_returns.values()))) + 1)
    return pd.DataFrame(
        {
            name: 100.0 * np.concatenate([[1.0], np.cumprod(1 + returns)])
            for name, returns in daily_returns.items()
        },
        index=index,
    )


@pytest.fixture
def steady_and_noisy() -> pd.DataFrame:
    """STEADY climbs with little noise; NOISY swings hard and drifts down."""
    days = np.arange(500)
    return _prices(
        STEADY=0.0008 + 0.0005 * np.sin(days),
        NOISY=-0.0003 + 0.02 * np.sin(days * 1.7),
    )


def _assert_failed(result: po.Optimization, reason: str) -> None:
    assert not result.ok
    assert reason in result.reason
    assert (result.allocation, result.expected_return, result.volatility, result.ratio) == (
        None, None, None, None,
    )


@pytest.mark.parametrize(
    "optimise",
    [
        lambda data: po.maximize_sharpe(data, 0.04),
        lambda data: po.maximize_geometric_sharpe(data, 0.04),
        lambda data: po.volatility_targeting_optimizer(data, 0.25, 0.04),
        lambda data: po.calculate_kelly_allocation(data, 0.04),
    ],
    ids=["sharpe", "geometric", "volatility", "kelly"],
)
def test_every_optimiser_refuses_a_single_asset_with_a_reason(optimise, steady_and_noisy):
    _assert_failed(optimise(steady_and_noisy[["STEADY"]]), "at least two assets")


@pytest.mark.parametrize(
    "optimise",
    [
        lambda data: po.maximize_sharpe(data, 0.04),
        lambda data: po.maximize_geometric_sharpe(data, 0.04),
        lambda data: po.volatility_targeting_optimizer(data, 0.25, 0.04),
        lambda data: po.calculate_kelly_allocation(data, 0.04),
    ],
    ids=["sharpe", "geometric", "volatility", "kelly"],
)
def test_every_optimiser_refuses_one_day_of_prices(optimise, steady_and_noisy):
    _assert_failed(optimise(steady_and_noisy.head(1)), "no returns")


def test_the_sharpe_optimiser_backs_the_steady_asset(steady_and_noisy):
    result = po.maximize_sharpe(steady_and_noisy, 0.0)
    assert result.ok
    assert sum(result.allocation.values()) == pytest.approx(1.0, abs=1e-3)
    assert result.allocation["STEADY"] > 0.9
    assert result.ratio > 0
    assert result.volatility > 0


def test_an_unreachable_volatility_cap_fails_rather_than_reporting_zeros(steady_and_noisy):
    result = po.volatility_targeting_optimizer(steady_and_noisy, -0.01, 0.04)
    _assert_failed(result, "did not converge")


def test_a_riskless_portfolio_has_no_sharpe_ratio_rather_than_zero():
    flat = pd.DataFrame(
        {"A": [100.0] * 30, "B": [50.0] * 30},
        index=pd.bdate_range("2024-01-01", periods=30),
    )
    result = po.volatility_targeting_optimizer(flat, 0.25, 0.04)
    assert result.ok
    assert result.volatility == pytest.approx(0.0)
    assert result.ratio is None


def test_kelly_holds_the_remainder_as_cash_and_reports_no_ratio(steady_and_noisy):
    result = po.calculate_kelly_allocation(steady_and_noisy, 0.04, kelly_fraction=0.5)
    assert result.ok
    assert "CASH" in result.allocation
    assert sum(result.allocation.values()) == pytest.approx(1.0, abs=1e-3)
    assert result.ratio is None


def test_yearly_returns_ignore_cash_and_renormalise_the_rest(steady_and_noisy):
    alone, _ = po.get_portfolio_yearly_returns(steady_and_noisy, {"STEADY": 1.0})
    with_cash, _ = po.get_portfolio_yearly_returns(
        steady_and_noisy, {"STEADY": 0.4, "CASH": 0.6}
    )
    pd.testing.assert_series_equal(alone, with_cash)


def test_yearly_returns_of_nothing_have_no_growth_rate():
    yearly, cagr = po.get_portfolio_yearly_returns(pd.DataFrame(), {"A": 1.0})
    assert yearly.empty
    assert cagr is None
