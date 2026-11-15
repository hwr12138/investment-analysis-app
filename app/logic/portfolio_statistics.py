"""Risk and return statistics for a fixed-weight portfolio, over several windows.

Pure — no Streamlit, no network; the page fetches the prices, including the
:data:`MARKET_TICKER` benchmark, and passes them in.

A statistic that is undefined is ``None``, never zero. Beta against a market
that never moved, a Sharpe ratio at zero volatility, a Treynor ratio at zero
beta: each used to print as ``0.0000``, which reads as a measurement. And with
no benchmark prices at all, the market-relative figures — alpha, beta, Treynor —
are ``None`` rather than computed against a market return of zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from app.logic import returns as ret

#: Benchmark for alpha, beta and Treynor. Sharadar carries no index levels, so
#: this is the S&P 500 tracker rather than the index itself.
MARKET_TICKER = "SPY"

#: Every statistic, in display order.
METRICS = (
    "Mean Annual Return", "CAGR", "Standard Deviation",
    "Alpha", "Beta", "Sharpe Ratio", "Geometric Sharpe Ratio",
    "Sortino Ratio", "Treynor Ratio",
)

#: The ones quoted as a percentage rather than a ratio.
PERCENT_METRICS = frozenset({"Mean Annual Return", "CAGR", "Standard Deviation"})

#: The fixed windows, in years back from the end date, each reported beside
#: the custom one. The longest decides how far back the prices are fetched.
WINDOW_YEARS = (1, 3, 5, 10)

#: A fixed window is reported only if it holds at least this share of the
#: sessions it should — a "10 Year" figure over four years of history is not one.
MIN_WINDOW_COVERAGE = 0.8


def _unavailable() -> dict[str, float | None]:
    return dict.fromkeys(METRICS)


def _ratio(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator == 0 or np.isnan(denominator):
        return None
    return float(numerator / denominator)


# --- Metric Calculations ---


def calculate_all_metrics(
    portfolio_returns: pd.Series, market_returns: pd.Series | None, rf_rate: float
) -> dict[str, float | None]:
    """Every statistic for one window of daily returns.

    With ``market_returns``, both series are first aligned on the days they
    share, so the portfolio-only figures describe the same days as the beta.
    Without it, alpha, beta and Treynor are ``None``.
    """
    if portfolio_returns.empty:
        return _unavailable()

    if market_returns is None:
        p_ret, m_ret = portfolio_returns.dropna(), None
    else:
        aligned = pd.concat([portfolio_returns, market_returns], axis=1, join='inner').dropna()
        p_ret, m_ret = aligned.iloc[:, 0], aligned.iloc[:, 1]
    if p_ret.empty:
        return _unavailable()

    ann_arith_return = float(ret.annualised_mean(p_ret))
    cagr = ret.cagr(p_ret)
    ann_volatility = ret.annualised_volatility(p_ret)

    beta = alpha = None
    if m_ret is not None and len(m_ret) > 1:
        cov_matrix = np.cov(p_ret, m_ret)
        beta = _ratio(cov_matrix[0, 1], cov_matrix[1, 1])
        if beta is not None:
            ann_market_return = float(ret.annualised_mean(m_ret))
            alpha = ann_arith_return - (rf_rate + beta * (ann_market_return - rf_rate))

    downside_returns = p_ret[p_ret < 0]
    downside_deviation = (
        float(np.sqrt((downside_returns**2).mean()) * np.sqrt(ret.TRADING_DAYS))
        if not downside_returns.empty
        else None
    )

    return {
        "Mean Annual Return": ann_arith_return,
        "CAGR": cagr,
        "Standard Deviation": ann_volatility,
        "Alpha": alpha,
        "Beta": beta,
        "Sharpe Ratio": _ratio(ann_arith_return - rf_rate, ann_volatility),
        "Geometric Sharpe Ratio": _ratio(cagr - rf_rate, ann_volatility),
        "Sortino Ratio": _ratio(ann_arith_return - rf_rate, downside_deviation),
        "Treynor Ratio": _ratio(ann_arith_return - rf_rate, beta),
    }


def get_comprehensive_metrics(
    data: pd.DataFrame,
    allocations: dict,
    custom_start: str,
    end_date: str,
    risk_free_rate: float,
    market_ticker: str = MARKET_TICKER,
) -> pd.DataFrame:
    """One column of :data:`METRICS` per window: 1, 3, 5 and 10 years, and Custom.

    ``data`` holds adjusted closes, one column per ticker, and should include
    ``market_ticker``; without it the market-relative rows are empty. Weights
    are renormalised over the tickers that have prices.
    """
    try:
        end = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        return pd.DataFrame()

    periods = {f"{years} Year": end - relativedelta(years=years) for years in WINDOW_YEARS}

    try:
        c_start = datetime.strptime(custom_start, '%Y-%m-%d')
        periods["Custom"] = c_start
    except ValueError:
        pass

    tickers = list(allocations.keys())

    valid_tickers = [t for t in tickers if t in data.columns]
    if not valid_tickers:
        return pd.DataFrame()

    daily_returns = data[valid_tickers].pct_change().dropna()
    weights = normalised_weights({t: allocations[t] for t in valid_tickers})
    valid_weights = np.array([weights[t] for t in valid_tickers])

    portfolio_returns = daily_returns.dot(valid_weights)
    market_returns = (
        data[market_ticker].pct_change().dropna() if market_ticker in data.columns else None
    )

    results = {}
    for label, start_date in periods.items():
        start_str = start_date.strftime('%Y-%m-%d')
        period_p_ret = portfolio_returns.loc[start_str:]
        period_m_ret = market_returns.loc[start_str:] if market_returns is not None else None

        expected_days = (end - start_date).days * (ret.TRADING_DAYS / 365.25)
        if len(period_p_ret) < (expected_days * MIN_WINDOW_COVERAGE) and label != "Custom":
            results[label] = _unavailable()
        else:
            results[label] = calculate_all_metrics(period_p_ret, period_m_ret, risk_free_rate)

    return pd.DataFrame(results)


# --- The weights, and how the holdings move together ---


def needs_rescaling(total: float) -> bool:
    """Whether weights summing to ``total`` are neither percentages (100) nor fractions (1).

    Such weights are still used, scaled to sum to one, and the page says so.
    """
    return abs(total - 100.0) > 0.01 and abs(total - 1.0) > 0.001


def normalised_weights(allocations: Mapping[str, float]) -> dict[str, float]:
    """Each weight as a fraction of their total.

    Weights that already sum to one, or to nothing, come back as given: there
    is nothing to divide by in the second case, and nothing to change in the first.
    """
    total = sum(allocations.values())
    if total <= 0 or np.isclose(total, 1.0):
        return dict(allocations)
    return {ticker: weight / total for ticker, weight in allocations.items()}


def history_start(start: date, end: date) -> date:
    """The first day to fetch: the custom start, or earlier if the longest window needs it."""
    return min(start, end - relativedelta(years=max(WINDOW_YEARS)))


@dataclass(frozen=True)
class Relationships:
    """How the holdings moved together over the custom window, on daily returns.

    Square frames over the same tickers in the same order. A pair with no
    variation has a ``NaN`` correlation, which is missing rather than zero.
    """

    correlation: pd.DataFrame
    #: Annualised, as :func:`app.logic.returns.annualised_covariance` scales it.
    covariance: pd.DataFrame


def relationships(
    prices: pd.DataFrame, tickers: Sequence[str], start: str, end: str
) -> Relationships | None:
    """Correlation and annualised covariance of daily returns from ``start`` to ``end``.

    ``None`` when fewer than two of ``tickers`` have prices, or when the window
    holds no day on which all of them have a return.
    """
    held = [t for t in tickers if t in prices.columns]
    if len(held) < 2:
        return None
    daily = prices[held].loc[start:end].pct_change().dropna()
    if daily.empty:
        return None
    return Relationships(daily.corr(), ret.annualised_covariance(daily))
