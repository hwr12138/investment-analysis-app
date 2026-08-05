"""Discount rate construction. Pure — no Streamlit, no network, no globals.

    WACC = (E/V) × Re + (D/V) × Rd × (1 − marginal tax rate)

The tax shield uses the *marginal* rate, not the effective rate that feeds
FCFF: the shield is worth what the next dollar of interest saves, not what the
company happened to pay on average across a decade of one-off settlements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Pandas resample rules for the two regression frequencies the spec allows.
_RESAMPLE_RULES = {"monthly": "ME", "weekly": "W-FRI"}

#: Below this many aligned observations a regression is noise, not a beta.
MIN_MONTHLY_OBSERVATIONS = 36
MIN_WEEKLY_OBSERVATIONS = 52


class InsufficientObservationsError(Exception):
    """Too few overlapping return observations to regress a beta."""

    def __init__(self, observations: int, required: int, frequency: str) -> None:
        self.observations = observations
        self.required = required
        self.frequency = frequency
        super().__init__(
            f"Only {observations} aligned {frequency} observations, "
            f"below the floor of {required}."
        )


@dataclass(frozen=True)
class BetaResult:
    """A regressed beta and everything needed to judge whether to trust it."""

    beta: float
    raw_beta: float
    r_squared: float
    observations: int
    frequency: str
    blume_adjusted: bool
    index_ticker: str
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None

    @property
    def formula(self) -> str:
        base = (
            f"OLS slope of {self.frequency} stock returns on {self.index_ticker} returns, "
            f"n={self.observations}"
        )
        if self.blume_adjusted:
            return base + "; Blume adjusted: 0.67 × raw β + 0.33"
        return base


@dataclass(frozen=True)
class WaccResult:
    """The assembled discount rate with every input retained for display."""

    wacc: float
    cost_of_equity: float
    cost_of_debt: float
    after_tax_cost_of_debt: float
    equity_value: float
    debt_value: float
    equity_weight: float
    debt_weight: float
    risk_free_rate: float
    beta: float
    equity_risk_premium: float
    marginal_tax_rate: float

    @property
    def formula(self) -> str:
        return (
            f"WACC = {self.equity_weight:.2%} × {self.cost_of_equity:.2%} + "
            f"{self.debt_weight:.2%} × {self.cost_of_debt:.2%} × "
            f"(1 − {self.marginal_tax_rate:.2%})"
        )


def periodic_returns(prices: pd.Series, frequency: str = "monthly") -> pd.Series:
    """Resamples a price series and returns its period-over-period returns.

    Expects an adjusted-close series: unadjusted prices would book every
    dividend and split as a return.
    """
    if frequency not in _RESAMPLE_RULES:
        raise ValueError(f"Unsupported frequency {frequency!r}")
    if prices is None or len(prices) == 0:
        return pd.Series(dtype="float64")

    series = pd.Series(prices).dropna().sort_index()
    if series.empty:
        return pd.Series(dtype="float64")
    resampled = series.resample(_RESAMPLE_RULES[frequency]).last()
    return resampled.pct_change().dropna()


def align_returns(
    stock_returns: pd.Series, index_returns: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Aligns two return series on their shared dates.

    Dates present in only one series are dropped **in pairs**. Nothing is
    forward-filled: inventing a return for a date one series never traded on
    would bias the covariance toward zero.
    """
    joined = pd.concat(
        [pd.Series(stock_returns).rename("stock"), pd.Series(index_returns).rename("index")],
        axis=1,
        join="inner",
    ).dropna()
    return joined["stock"], joined["index"]


def blume_adjust(beta: float) -> float:
    """Blume adjustment: pulls a regressed beta toward the market's 1.0."""
    return 0.67 * beta + 0.33 * 1.0


def regress_beta(
    stock_prices: pd.Series,
    index_prices: pd.Series,
    *,
    frequency: str = "monthly",
    blume: bool = False,
    index_ticker: str = "SPY",
    minimum_observations: int | None = None,
) -> BetaResult:
    """Regresses stock returns on index returns.

    Raises :class:`InsufficientObservationsError` rather than returning a beta
    fitted to a handful of points — the caller is expected to fall through to
    manual entry and say why.
    """
    floor = minimum_observations
    if floor is None:
        floor = MIN_MONTHLY_OBSERVATIONS if frequency == "monthly" else MIN_WEEKLY_OBSERVATIONS

    stock, index = align_returns(
        periodic_returns(stock_prices, frequency), periodic_returns(index_prices, frequency)
    )

    if len(stock) < floor:
        raise InsufficientObservationsError(len(stock), floor, frequency)

    index_variance = float(np.var(index.to_numpy(), ddof=1))
    if index_variance == 0:
        raise InsufficientObservationsError(len(stock), floor, frequency)

    covariance = float(np.cov(stock.to_numpy(), index.to_numpy(), ddof=1)[0, 1])
    raw_beta = covariance / index_variance
    correlation = float(np.corrcoef(stock.to_numpy(), index.to_numpy())[0, 1])

    return BetaResult(
        beta=blume_adjust(raw_beta) if blume else raw_beta,
        raw_beta=raw_beta,
        r_squared=correlation**2,
        observations=len(stock),
        frequency=frequency,
        blume_adjusted=blume,
        index_ticker=index_ticker,
        start=stock.index.min(),
        end=stock.index.max(),
    )


def cost_of_equity(risk_free_rate: float, beta: float, equity_risk_premium: float) -> float:
    """CAPM: Re = Rf + β × ERP."""
    return risk_free_rate + beta * equity_risk_premium


def implied_cost_of_debt(
    interest_expense: float, debt_current: float, debt_prior: float | None = None
) -> float | None:
    """Interest expense ÷ average total debt.

    A *backward-looking average coupon*, not a current marginal cost: it
    understates when rates have risen since the debt was issued and overstates
    when they have fallen. Shown next to the manual YTM input for sanity, never
    fed to the model. Returns ``None`` when there is no debt to divide by.
    """
    if interest_expense is None or debt_current is None:
        return None
    if debt_prior is None or (isinstance(debt_prior, float) and math.isnan(debt_prior)):
        average_debt = debt_current
    else:
        average_debt = (debt_current + debt_prior) / 2.0
    if not average_debt or average_debt <= 0:
        return None
    return interest_expense / average_debt


def market_value_of_equity(share_price: float, shares_outstanding: float) -> float:
    """E = current share price × current common shares outstanding.

    Point-in-time share count, not the weighted average: the market is pricing
    the shares that exist today, not the average that existed over the year.
    """
    return share_price * shares_outstanding


def compute_wacc(
    *,
    equity_value: float,
    debt_value: float,
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float,
    cost_of_debt: float,
    marginal_tax_rate: float,
) -> WaccResult:
    """Assembles the weighted average cost of capital.

    A company with no debt gets a 100% equity weight and a WACC equal to its
    cost of equity — no special-casing, the weights just fall out.
    """
    total_capital = equity_value + debt_value
    if total_capital <= 0:
        raise ValueError(
            "Enterprise capital (equity + debt) must be positive to weight a WACC."
        )

    equity_weight = equity_value / total_capital
    debt_weight = debt_value / total_capital
    re = cost_of_equity(risk_free_rate, beta, equity_risk_premium)
    after_tax_rd = cost_of_debt * (1.0 - marginal_tax_rate)

    return WaccResult(
        wacc=equity_weight * re + debt_weight * after_tax_rd,
        cost_of_equity=re,
        cost_of_debt=cost_of_debt,
        after_tax_cost_of_debt=after_tax_rd,
        equity_value=equity_value,
        debt_value=debt_value,
        equity_weight=equity_weight,
        debt_weight=debt_weight,
        risk_free_rate=risk_free_rate,
        beta=beta,
        equity_risk_premium=equity_risk_premium,
        marginal_tax_rate=marginal_tax_rate,
    )
