"""Return arithmetic shared by the portfolio pages.

Pure — no Streamlit, no network. Every function takes a series (or frame) of
periodic simple returns, so Portfolio Optimization and Portfolio Statistics
annualise the same way and cannot drift apart on the convention. Before this
module each page carried its own ``TRADING_DAYS`` and one view multiplied a
covariance matrix by a bare ``251``.

A figure that cannot be computed is ``None``, never zero: a compound growth
rate over no returns is not 0%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Sessions a year the portfolio pages annualise by. The US market has averaged
#: 251–252; 251 is this app's long-standing convention, and changing it moves
#: every annualised figure on both pages.
TRADING_DAYS = 251


def annualised_mean(returns: pd.Series | pd.DataFrame):
    """Arithmetic mean return, annualised — a scalar for a series, a row for a frame."""
    return returns.mean() * TRADING_DAYS


def annualised_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def annualised_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov() * TRADING_DAYS


def cagr(returns: pd.Series) -> float | None:
    """Compound annual growth rate over the whole series. ``None`` when empty."""
    if len(returns) == 0:
        return None
    return float((1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1)
