"""Discount rate construction. Pure — no Streamlit, no network, no globals.

    WACC = (E/V) × Re + (D/V) × Rd × (1 − marginal tax rate)

The tax shield uses the *marginal* rate, not the effective rate that feeds
FCFF: the shield is worth what the next dollar of interest saves, not what the
company happened to pay on average across a decade of one-off settlements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date

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
    #: Standard error of the *raw* OLS slope. A five-year monthly beta typically
    #: carries 0.2–0.3, which is why R² alone says too little about it.
    raw_standard_error: float | None = None

    @property
    def standard_error(self) -> float | None:
        """Standard error of the beta in use — scaled by 0.67 when Blume adjusted."""
        if self.raw_standard_error is None:
            return None
        return self.raw_standard_error * (BLUME_WEIGHT if self.blume_adjusted else 1.0)

    @property
    def confidence_interval(self) -> tuple[float, float] | None:
        """An approximate 95% interval around the beta in use (± 1.96 SE)."""
        error = self.standard_error
        if error is None:
            return None
        return self.beta - 1.96 * error, self.beta + 1.96 * error

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


#: Weight the Blume adjustment leaves on the regressed beta.
BLUME_WEIGHT = 0.67


def blume_adjust(beta: float) -> float:
    """Blume adjustment: pulls a regressed beta toward the market's 1.0."""
    return BLUME_WEIGHT * beta + (1.0 - BLUME_WEIGHT) * 1.0


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

    y, x = stock.to_numpy(), index.to_numpy()
    covariance = float(np.cov(y, x, ddof=1)[0, 1])
    raw_beta = covariance / index_variance
    correlation = float(np.corrcoef(y, x)[0, 1])

    # Standard error of an OLS slope: residual variance over the spread of x.
    intercept = y.mean() - raw_beta * x.mean()
    residuals = y - (intercept + raw_beta * x)
    spread = float(((x - x.mean()) ** 2).sum())
    standard_error = (
        math.sqrt(float((residuals**2).sum()) / (len(y) - 2) / spread)
        if len(y) > 2 and spread > 0
        else None
    )

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
        raw_standard_error=standard_error,
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


@dataclass(frozen=True)
class RatingBand:
    """One row of the interest-coverage → rating → spread table."""

    min_coverage: float
    rating: str
    spread: float


#: Where :data:`SYNTHETIC_RATINGS` comes from, and the vintage of its spreads.
SYNTHETIC_RATINGS_SOURCE = (
    "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.html"
)
SYNTHETIC_RATINGS_AS_OF = "January 2026"
#: The same vintage as a date, for the staleness check. Refresh both, with the
#: table, from Damodaran's January update — see CLAUDE.md.
SYNTHETIC_RATINGS_AS_OF_DATE = date(2026, 1, 1)

#: Interest coverage (EBIT ÷ interest expense) to a synthetic rating and its
#: default spread over the Treasury: Damodaran's table for large non-financial
#: firms, data as of January 2026, checked against the page. Spreads move every
#: year, so the source and vintage travel with every figure built from them.
#: The same page carries a financial-services table (not needed — banks are
#: rejected), and Damodaran publishes wider coverage bands for smaller, riskier
#: firms; that table is not used here. Ordered from the highest coverage down;
#: the first band a company clears is its rating.
SYNTHETIC_RATINGS: tuple[RatingBand, ...] = (
    RatingBand(8.50, "AAA", 0.0040),
    RatingBand(6.50, "AA", 0.0055),
    RatingBand(5.50, "A+", 0.0070),
    RatingBand(4.25, "A", 0.0078),
    RatingBand(3.00, "A-", 0.0089),
    RatingBand(2.50, "BBB", 0.0111),
    RatingBand(2.25, "BB+", 0.0138),
    RatingBand(2.00, "BB", 0.0184),
    RatingBand(1.75, "B+", 0.0275),
    RatingBand(1.50, "B", 0.0321),
    RatingBand(1.25, "B-", 0.0509),
    RatingBand(0.80, "CCC", 0.0885),
    RatingBand(0.65, "CC", 0.1261),
    RatingBand(0.20, "C", 0.1600),
    RatingBand(float("-inf"), "D", 0.1900),
)


@dataclass(frozen=True)
class SyntheticCostOfDebt:
    """A pre-tax cost of debt estimated from interest coverage."""

    cost_of_debt: float
    rating: str
    spread: float
    coverage: float | None
    #: The fiscal year the coverage was measured in, when it is not the base
    #: period's — see :func:`latest_disclosed_coverage`.
    coverage_period: str | None = None

    @property
    def formula(self) -> str:
        coverage = "no interest expense" if self.coverage is None else f"{self.coverage:,.2f}×"
        where = f" {self.coverage_period}" if self.coverage_period else ""
        return (
            f"Coverage{where} (EBIT ÷ interest) {coverage} → {self.rating} → "
            f"risk-free + {self.spread:.2%} spread (Damodaran, large non-financial "
            f"firms, {SYNTHETIC_RATINGS_AS_OF})"
        )


def interest_looks_undisclosed(
    interest_expense: float | None, total_debt: float | None
) -> bool:
    """A reported interest expense of zero beside real debt.

    Sharadar writes 0 for a line item a filing does not break out, so zero
    interest is only believable when there is no debt to pay it on. Apple
    carries about $84bn of debt and reports interest as 0.
    """
    return (
        interest_expense is not None
        and interest_expense == 0
        and total_debt is not None
        and math.isfinite(total_debt)
        and total_debt > 0
    )


#: How many of the latest fiscal years may stand in for an undisclosed interest
#: line. Coverage older than this describes a different balance sheet.
DISCLOSED_INTEREST_LOOKBACK_YEARS = 3


def latest_disclosed_coverage(
    history: list[tuple[str, float | None, float | None]],
    lookback: int = DISCLOSED_INTEREST_LOOKBACK_YEARS,
) -> tuple[str, float, float] | None:
    """The most recent year, among the last ``lookback``, that disclosed interest.

    ``history`` is ``(period label, EBIT, interest expense)``, oldest first.
    Returns that year's ``(label, EBIT, interest)``, or ``None`` when none of the
    last ``lookback`` years reported positive interest beside a usable EBIT —
    nothing older is reached for.
    """
    for label, ebit, interest in reversed(history[-lookback:]):
        if (
            ebit is not None and interest is not None
            and math.isfinite(ebit) and math.isfinite(interest) and interest > 0
        ):
            return label, float(ebit), float(interest)
    return None


def synthetic_cost_of_debt(
    ebit: float | None,
    interest_expense: float | None,
    risk_free_rate: float,
    total_debt: float | None = None,
) -> SyntheticCostOfDebt | None:
    """Pre-tax cost of debt from the synthetic-rating method.

    Maps EBIT ÷ interest expense onto :data:`SYNTHETIC_RATINGS` and adds that
    band's spread to the risk-free rate, so an AA issuer and a B issuer no
    longer share one cost of debt. ``None`` when EBIT or interest is not
    reported — nothing is substituted — and when interest reads 0 beside debt
    (:func:`interest_looks_undisclosed`), which is the same thing in disguise.
    Zero interest rates as the top band only when there is no debt either.
    """
    if ebit is None or interest_expense is None:
        return None
    if not (math.isfinite(ebit) and math.isfinite(interest_expense) and math.isfinite(risk_free_rate)):
        return None
    if interest_expense < 0 or interest_looks_undisclosed(interest_expense, total_debt):
        return None
    if interest_expense == 0:
        if ebit <= 0:
            return None
        band = SYNTHETIC_RATINGS[0]
        return SyntheticCostOfDebt(risk_free_rate + band.spread, band.rating, band.spread, None)
    coverage = ebit / interest_expense
    band = next(b for b in SYNTHETIC_RATINGS if coverage >= b.min_coverage)
    return SyntheticCostOfDebt(risk_free_rate + band.spread, band.rating, band.spread, coverage)


def resolve_synthetic_cost_of_debt(
    ebit: float | None,
    interest_expense: float | None,
    risk_free_rate: float,
    total_debt: float | None,
    history: list[tuple[str, float | None, float | None]],
) -> tuple[SyntheticCostOfDebt | None, str]:
    """The base period's synthetic cost of debt, or the latest disclosed year's.

    Returns the result and, when the base period's coverage could not be used,
    why: ``"undisclosed"`` (interest reads 0 beside real debt — Sharadar's mark
    for a line the filing does not break out) or ``"missing"`` (no interest line
    at all). Either way the latest of the last
    :data:`DISCLOSED_INTEREST_LOOKBACK_YEARS` fiscal years in ``history`` that
    did disclose interest supplies the coverage, its spread applied to today's
    risk-free rate and its year recorded. ``(None, cause)`` when none did.
    ``(None, "negative")`` when the base period reports net interest *income*
    (ASML, SAP) — no coverage exists, and no substitute year is used.
    """
    base = synthetic_cost_of_debt(ebit, interest_expense, risk_free_rate, total_debt=total_debt)
    if base is not None:
        return base, ""
    if interest_looks_undisclosed(interest_expense, total_debt):
        cause = "undisclosed"
    elif interest_expense is None or not math.isfinite(interest_expense):
        cause = "missing"
    elif interest_expense < 0:
        # A net interest *income* (ASML, SAP): the line is a net finance
        # result, not the gross cost of debt, so it has no coverage ratio. No
        # other year is reached for — that would be a judgement, not a rule.
        return None, "negative"
    else:
        return None, ""
    disclosed = latest_disclosed_coverage(history)
    if disclosed is None:
        return None, cause
    period, year_ebit, year_interest = disclosed
    found = synthetic_cost_of_debt(year_ebit, year_interest, risk_free_rate)
    if found is None:
        return None, cause
    return replace(found, coverage_period=period), cause


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
