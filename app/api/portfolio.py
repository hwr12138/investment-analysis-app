"""📊 Portfolio Statistics and 📈 Portfolio Optimization over HTTP.

Both are GETs with every input in the query string, like the momentum routes:
they read and change nothing, and a URL that names the whole request lets the
page keep it in its own address. Holdings travel as two parallel lists,
``tickers`` and ``weights``.

The prices come from one cached fetch per request, so a second look at the
same portfolio costs no request to Sharadar.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import Field, StringConstraints

from app.api.contract import Contract, finite_or_none
from app.data import sharadar_client
from app.logic import portfolio_optimization as po
from app.logic import portfolio_statistics as ps

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

#: Each ticker is its own Sharadar request, made one after another, and Hosting
#: stops waiting after 60 seconds.
MAX_ASSETS = 20

TickerText = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9.\-]{1,10}$")]
Tickers = Annotated[list[TickerText], Query(min_length=1, max_length=MAX_ASSETS)]
Weights = Annotated[
    list[Annotated[float, Field(ge=0)]], Query(min_length=1, max_length=MAX_ASSETS)
]
Start = Annotated[date, Query()]
End = Annotated[date, Query()]
RiskFree = Annotated[
    float, Query(ge=-0.05, le=0.5, description="Annual, as a decimal: 0.04 is 4%.")
]

Strategy = Literal["sharpe", "geometric_sharpe", "volatility_target", "kelly"]


class Weight(Contract):
    ticker: str
    #: A fraction of the whole; the weights in one list sum to one.
    weight: float


class Matrix(Contract):
    """A square table, rows and columns both in ``tickers`` order. A missing cell is ``null``."""

    tickers: list[str]
    values: list[list[float | None]]


class Relationships(Contract):
    correlation: Matrix
    #: Annualised.
    covariance: Matrix


class MetricRow(Contract):
    name: str
    unit: Literal["percent", "ratio"]
    #: One per window, in the response's ``windows`` order. ``null`` where the
    #: figure is undefined or the window is longer than the history.
    values: list[float | None]


class PortfolioStatistics(Contract):
    #: The tickers that have prices, with their weights rescaled over just those.
    holdings: list[Weight]
    #: The weights given summed to neither 100 nor 1, so they were scaled to one.
    rescaled: bool
    entered_total: float
    #: Tickers given that have no prices in the range. They are left out.
    missing: list[str]
    benchmark: str
    #: Without the benchmark's prices, alpha, beta and Treynor are ``null``.
    benchmark_priced: bool
    #: Over the custom window. ``null`` with fewer than two priced holdings, or
    #: no day on which they all have a return.
    relationships: Relationships | None
    windows: list[str]
    metrics: list[MetricRow]


class YearReturn(Contract):
    year: int
    #: The calendar year's return. The first and last years cover only the
    #: sessions inside the range.
    value: float | None


class StrategyResult(Contract):
    strategy: Strategy
    #: Why there is no allocation. ``null`` when there is one.
    reason: str | None
    #: The compound rate for the two Sharpe strategies, the arithmetic mean for
    #: the volatility target and Kelly.
    expected_return: float | None
    volatility: float | None
    #: The Sharpe ratio maximised or achieved. Always ``null`` for Kelly, which
    #: maximises growth instead.
    ratio: float | None
    #: Empty when the optimiser failed. Kelly's includes a ``CASH`` line.
    allocation: list[Weight]
    #: Of the invested part: a Kelly allocation's cash drops out and the rest
    #: is scaled up.
    yearly_returns: list[YearReturn]


class PortfolioOptimization(Contract):
    #: The tickers that have prices, which is what the optimisers allocate across.
    tickers: list[str]
    missing: list[str]
    first_session: date
    last_session: date
    strategies: list[StrategyResult]


def _refuse(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail)


def _check_tickers(tickers: list[str]) -> list[str]:
    tickers = [t.upper() for t in tickers]
    if len(set(tickers)) != len(tickers):
        raise _refuse("List each ticker only once.")
    return tickers


def _check_dates(start: date, end: date) -> None:
    if start >= end:
        raise _refuse("Please ensure your Start Date is before your End Date.")


def _matrix(frame: pd.DataFrame) -> Matrix:
    return Matrix(
        tickers=[str(t) for t in frame.index],
        values=[[finite_or_none(v) for v in row] for row in frame.to_numpy()],
    )


def _weights(allocation: dict[str, float]) -> list[Weight]:
    return [Weight(ticker=t, weight=float(w)) for t, w in allocation.items()]


@router.get("/statistics")
def statistics(
    tickers: Tickers, weights: Weights, start: Start, end: End, risk_free: RiskFree
) -> PortfolioStatistics:
    """Risk and return over 1, 3, 5 and 10 years and the custom window, and how the holdings move together."""
    tickers = _check_tickers(tickers)
    _check_dates(start, end)
    if len(weights) != len(tickers):
        raise _refuse("Give one weight for each ticker.")
    entered_total = sum(weights)
    if entered_total <= 0:
        raise _refuse("The weights must add up to more than zero.")

    # The benchmark rides along in the same fetch; alpha, beta and Treynor are
    # measured against it.
    wanted = tuple(dict.fromkeys([*tickers, ps.MARKET_TICKER]))
    prices = sharadar_client.fetch_price_history(
        wanted, ps.history_start(start, end).isoformat(), end.isoformat()
    )
    held = [t for t in tickers if t in prices.columns]
    if not held:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Failed to download enough valid data. Please check your tickers.",
        )

    allocation = dict(zip(tickers, weights))
    used = {t: allocation[t] for t in held}
    table = ps.get_comprehensive_metrics(
        prices, used, start.isoformat(), end.isoformat(), risk_free
    )
    if table.empty:
        raise _refuse("Not enough data to calculate portfolio metrics.")
    related = ps.relationships(prices, held, start.isoformat(), end.isoformat())

    return PortfolioStatistics(
        holdings=_weights(ps.normalised_weights(used)),
        rescaled=ps.needs_rescaling(entered_total),
        entered_total=entered_total,
        missing=[t for t in tickers if t not in held],
        benchmark=ps.MARKET_TICKER,
        benchmark_priced=ps.MARKET_TICKER in prices.columns,
        relationships=None if related is None else Relationships(
            correlation=_matrix(related.correlation),
            covariance=_matrix(related.covariance),
        ),
        windows=[str(w) for w in table.columns],
        metrics=[
            MetricRow(
                name=metric,
                unit="percent" if metric in ps.PERCENT_METRICS else "ratio",
                values=[finite_or_none(table.at[metric, w]) for w in table.columns],
            )
            for metric in ps.METRICS
        ],
    )


def _strategy(
    strategy: Strategy, result: po.Optimization, prices: pd.DataFrame
) -> StrategyResult:
    if not result.ok:
        return StrategyResult(
            strategy=strategy,
            reason=result.reason,
            expected_return=None,
            volatility=None,
            ratio=None,
            allocation=[],
            yearly_returns=[],
        )
    yearly, _ = po.get_portfolio_yearly_returns(prices, result.allocation)
    return StrategyResult(
        strategy=strategy,
        reason=None,
        expected_return=finite_or_none(result.expected_return),
        volatility=finite_or_none(result.volatility),
        ratio=finite_or_none(result.ratio),
        allocation=_weights(result.allocation),
        yearly_returns=[
            YearReturn(year=int(year), value=finite_or_none(value))
            for year, value in yearly.items()
        ],
    )


@router.get("/optimization")
def optimization(
    tickers: Annotated[list[TickerText], Query(min_length=2, max_length=MAX_ASSETS)],
    start: Start,
    end: End,
    risk_free: RiskFree,
    vol_target: Annotated[
        float, Query(gt=0, le=2, description="The volatility cap, as a decimal: 0.25 is 25%.")
    ] = 0.25,
    kelly_fraction: Annotated[float, Query(gt=0, le=1)] = 0.5,
) -> PortfolioOptimization:
    """The four optimisers over one price history: max Sharpe, max geometric Sharpe, a volatility cap, fractional Kelly."""
    tickers = _check_tickers(tickers)
    _check_dates(start, end)
    prices = sharadar_client.fetch_price_history(tuple(tickers), start.isoformat(), end.isoformat())
    if prices.empty or prices.shape[1] < 2:
        raise _refuse(
            "Fewer than two of these tickers have prices in this range. "
            "Check the tickers and the dates."
        )

    results: list[tuple[Strategy, po.Optimization]] = [
        ("sharpe", po.maximize_sharpe(prices, risk_free)),
        ("geometric_sharpe", po.maximize_geometric_sharpe(prices, risk_free)),
        ("volatility_target", po.volatility_targeting_optimizer(prices, vol_target, risk_free)),
        ("kelly", po.calculate_kelly_allocation(prices, risk_free, kelly_fraction)),
    ]
    return PortfolioOptimization(
        tickers=[str(t) for t in prices.columns],
        missing=[t for t in tickers if t not in prices.columns],
        first_session=pd.Timestamp(prices.index[0]).date(),
        last_session=pd.Timestamp(prices.index[-1]).date(),
        strategies=[_strategy(name, result, prices) for name, result in results],
    )
