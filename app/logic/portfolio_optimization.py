"""Portfolio optimisers: max Sharpe, max geometric Sharpe, volatility target, Kelly.

Pure — no Streamlit, no network. Each optimiser takes a frame of prices, one
column per asset, and returns an :class:`Optimization`: the allocation and what
it is expected to do, or ``None`` in every figure and the reason there is no
answer. A failed optimisation used to come back as zeros, which a caller could
print as a 0% return at 0% volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app.logic import returns as ret


@dataclass(frozen=True)
class Optimization:
    """One optimiser's answer, or why there is none.

    ``expected_return`` is whichever return the optimiser targets — the compound
    rate for the two Sharpe variants, the arithmetic mean for the volatility
    target and Kelly. ``ratio`` is the Sharpe ratio it maximised or achieved, and
    ``None`` for Kelly, which maximises growth rather than a ratio. Any of them
    is ``None`` where it is undefined, never zero.
    """

    allocation: dict[str, float] | None
    expected_return: float | None
    volatility: float | None
    ratio: float | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.allocation is not None

    @classmethod
    def failed(cls, reason: str) -> "Optimization":
        return cls(None, None, None, None, reason)


_TOO_FEW_ASSETS = "An optimisation needs at least two assets with price history."
_TOO_LITTLE_HISTORY = "The assets share fewer than two days of prices, so there are no returns."


def _too_few_assets(data: pd.DataFrame) -> bool:
    return data.empty or data.shape[1] < 2


def _did_not_converge(result) -> Optimization:
    return Optimization.failed(f"The optimiser did not converge: {result.message}")


def _ratio(excess: float, volatility: float) -> float | None:
    """A Sharpe-style ratio, undefined — not zero — at zero volatility."""
    return excess / volatility if volatility > 0 else None


# --- Optimizers ---


def maximize_sharpe(data: pd.DataFrame, risk_free_rate: float) -> Optimization:
    """The long-only allocation with the highest arithmetic Sharpe ratio.

    Reports the compound annual return of that allocation over the history, and
    its annualised volatility from the covariance matrix.
    """
    if _too_few_assets(data):
        return Optimization.failed(_TOO_FEW_ASSETS)

    returns = data.pct_change().dropna()
    if returns.empty:
        return Optimization.failed(_TOO_LITTLE_HISTORY)
    expected_returns = ret.annualised_mean(returns)
    cov_matrix = ret.annualised_covariance(returns)
    num_assets = len(data.columns)
    tickers = data.columns.tolist()

    def neg_sharpe(weights):
        port_return = np.dot(weights, expected_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if port_volatility == 0:
            return 0.0
        return -((port_return - risk_free_rate) / port_volatility)

    # Constraints: sum of weights equals 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Bounds: no short selling (weights between 0 and 1)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = minimize(neg_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        return _did_not_converge(result)

    optimal_weights = result.x
    allocation = {tickers[i]: round(optimal_weights[i], 4) for i in range(num_assets)}
    ann_vol = float(np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))))
    return Optimization(
        allocation, ret.cagr(returns.dot(optimal_weights)), ann_vol, float(-result.fun)
    )


def maximize_geometric_sharpe(data: pd.DataFrame, risk_free_rate: float) -> Optimization:
    """The long-only allocation with the highest compound-return Sharpe ratio."""
    if _too_few_assets(data):
        return Optimization.failed(_TOO_FEW_ASSETS)

    daily_returns = data.pct_change().dropna()
    if daily_returns.empty:
        return Optimization.failed(_TOO_LITTLE_HISTORY)
    num_assets = len(data.columns)
    tickers = data.columns.tolist()

    def neg_geom_sharpe(weights):
        port_returns = daily_returns.dot(weights)
        ann_geom_return = ret.cagr(port_returns)
        ann_volatility = ret.annualised_volatility(port_returns)

        if ann_volatility == 0: return 0.0
        return -((ann_geom_return - risk_free_rate) / ann_volatility)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = minimize(neg_geom_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        return _did_not_converge(result)

    optimal_weights = result.x
    allocation = {tickers[i]: round(optimal_weights[i], 4) for i in range(num_assets)}
    optimal_returns = daily_returns.dot(optimal_weights)
    return Optimization(
        allocation,
        ret.cagr(optimal_returns),
        ret.annualised_volatility(optimal_returns),
        float(-result.fun),
    )


def volatility_targeting_optimizer(
    data: pd.DataFrame, vol_constraint: float, risk_free_rate: float
) -> Optimization:
    """Maximizes expected return subject to a maximum annualized volatility constraint."""
    if _too_few_assets(data):
        return Optimization.failed(_TOO_FEW_ASSETS)

    returns = data.pct_change().dropna()
    if returns.empty:
        return Optimization.failed(_TOO_LITTLE_HISTORY)
    expected_returns = ret.annualised_mean(returns)
    cov_matrix = ret.annualised_covariance(returns)
    num_assets = len(data.columns)
    tickers = data.columns.tolist()

    def objective(weights):
        return -np.dot(weights, expected_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    sum_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    vol_limit = {'type': 'ineq', 'fun': lambda w: vol_constraint - portfolio_volatility(w)}

    constraints = [sum_constraint, vol_limit]
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_weights = np.array([1.0 / num_assets] * num_assets)

    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        return _did_not_converge(result)

    opt_weights = result.x
    opt_ret = float(np.dot(opt_weights, expected_returns))
    opt_vol = float(portfolio_volatility(opt_weights))
    allocation = {tickers[i]: round(opt_weights[i], 4) for i in range(num_assets)}
    return Optimization(allocation, opt_ret, opt_vol, _ratio(opt_ret - risk_free_rate, opt_vol))


def calculate_kelly_allocation(
    data: pd.DataFrame, risk_free_rate: float, kelly_fraction: float = 0.5
) -> Optimization:
    """The fractional Kelly allocation, with the remainder held as ``CASH``."""
    if _too_few_assets(data):
        return Optimization.failed(_TOO_FEW_ASSETS)

    daily_returns = data.pct_change().dropna()
    if daily_returns.empty:
        return Optimization.failed(_TOO_LITTLE_HISTORY)
    mean_returns = ret.annualised_mean(daily_returns)
    cov_matrix = ret.annualised_covariance(daily_returns)
    num_assets = len(data.columns)
    tickers = data.columns.tolist()

    def neg_kelly_growth(weights):
        port_excess_return = np.dot(weights, mean_returns - risk_free_rate)
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        g = risk_free_rate + port_excess_return - (0.5 * port_variance)
        return -g

    # Weight sum <= 1 (allows holding cash)
    constraints = ({'type': 'ineq', 'fun': lambda x: 1.0 - np.sum(x)})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = minimize(neg_kelly_growth, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        return _did_not_converge(result)

    raw_weights = result.x
    fractional_weights = raw_weights * kelly_fraction
    cash_allocation = 1.0 - np.sum(fractional_weights)

    port_return = np.dot(fractional_weights, mean_returns) + (cash_allocation * risk_free_rate)
    port_vol = np.sqrt(np.dot(fractional_weights.T, np.dot(cov_matrix, fractional_weights)))

    allocation = {tickers[i]: round(fractional_weights[i], 4) for i in range(num_assets)}
    allocation['CASH'] = round(cash_allocation, 4)
    return Optimization(allocation, float(port_return), float(port_vol))


# --- Yearly Returns Analysis ---


def get_portfolio_yearly_returns(
    data: pd.DataFrame, weights_dict: dict
) -> tuple[pd.Series, float | None]:
    """Calendar-year returns of a fixed-weight portfolio, and its CAGR.

    Weights are renormalised over the assets that have prices, so a Kelly
    allocation's ``CASH`` line drops out and the rest are scaled up. The CAGR is
    ``None`` when there is nothing to compound.
    """
    tickers = list(weights_dict.keys())
    # Exclude 'CASH' from historical data mapping if it was added by Kelly
    valid_tickers = [t for t in tickers if t in data.columns]

    if data.empty or not valid_tickers:
        return pd.Series(dtype="float64"), None

    returns = data[valid_tickers].pct_change().dropna()
    portfolio_weights = np.array([weights_dict[t] for t in valid_tickers])

    # Normalize if weights don't sum to 1 (ignoring cash)
    if not np.isclose(portfolio_weights.sum(), 1.0) and portfolio_weights.sum() > 0:
        portfolio_weights = portfolio_weights / portfolio_weights.sum()

    portfolio_daily_returns = (returns * portfolio_weights).sum(axis=1)

    def calculate_annual_return(daily_rets):
        return (1 + daily_rets).prod() - 1

    yearly_returns = portfolio_daily_returns.groupby(portfolio_daily_returns.index.year).apply(calculate_annual_return)
    return yearly_returns, ret.cagr(portfolio_daily_returns)
