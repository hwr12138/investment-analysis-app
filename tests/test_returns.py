"""Return arithmetic shared by the portfolio pages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.logic import returns as ret


def test_a_year_of_sessions_compounds_to_its_own_total_return():
    daily = pd.Series([0.21] + [0.0] * (ret.TRADING_DAYS - 1))
    assert ret.cagr(daily) == pytest.approx(0.21)


def test_two_years_of_sessions_compound_to_the_annual_rate():
    daily = pd.Series([0.21] + [0.0] * (2 * ret.TRADING_DAYS - 1))
    assert ret.cagr(daily) == pytest.approx(0.1)  # 1.21 ** 0.5 − 1


def test_no_returns_have_no_growth_rate_rather_than_zero():
    assert ret.cagr(pd.Series(dtype="float64")) is None


def test_the_mean_and_volatility_annualise_by_the_same_session_count():
    daily = pd.Series([0.01, -0.01] * 50)
    assert ret.annualised_mean(daily) == pytest.approx(0.0)
    assert ret.annualised_volatility(daily) == pytest.approx(
        daily.std() * np.sqrt(ret.TRADING_DAYS)
    )


def test_the_covariance_annualises_every_cell():
    frame = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [0.0, 0.01, -0.01]})
    assert np.allclose(
        ret.annualised_covariance(frame).values, frame.cov().values * ret.TRADING_DAYS
    )
