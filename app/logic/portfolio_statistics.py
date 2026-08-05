import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from app.data import sharadar_client

TRADING_DAYS = 251

#: Benchmark for alpha, beta and Treynor. Sharadar carries no index levels, so
#: this is the S&P 500 tracker rather than the index itself.
MARKET_TICKER = "SPY"

# --- Data Fetching ---

def fetch_data_and_market(tickers: list, start_date: str, end_date: str, market_ticker=MARKET_TICKER):
    all_tickers = tuple(dict.fromkeys(list(tickers) + [market_ticker]))
    data = sharadar_client.fetch_price_history(all_tickers, start_date, end_date)
    return data

# --- Metric Calculations ---

def calculate_all_metrics(portfolio_returns: pd.Series, market_returns: pd.Series, rf_rate: float):
    metrics_keys = [
        "Mean Annual Return", "CAGR", "Standard Deviation",
        "Alpha", "Beta", "Sharpe Ratio", "Geometric Sharpe Ratio",
        "Sortino Ratio", "Treynor Ratio"
    ]

    if portfolio_returns.empty:
        return {k: None for k in metrics_keys}

    # Align dates
    aligned = pd.concat([portfolio_returns, market_returns], axis=1, join='inner').dropna()
    if aligned.empty:
        return {k: None for k in metrics_keys}

    p_ret = aligned.iloc[:, 0]
    m_ret = aligned.iloc[:, 1]

    # Mean Annual Return
    ann_arith_return = p_ret.mean() * TRADING_DAYS

    # CAGR
    total_compound_return = (1 + p_ret).prod()
    cagr = total_compound_return ** (TRADING_DAYS / len(p_ret)) - 1

    # Standard Deviation
    ann_volatility = p_ret.std() * np.sqrt(TRADING_DAYS)

    # Beta & Alpha
    cov_matrix = np.cov(p_ret, m_ret)
    if cov_matrix[1, 1] == 0:
        beta = 0.0
    else:
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    ann_market_return = m_ret.mean() * TRADING_DAYS
    alpha = ann_arith_return - (rf_rate + beta * (ann_market_return - rf_rate))

    # Sharpe Ratio
    sharpe = (ann_arith_return - rf_rate) / ann_volatility if ann_volatility != 0 else 0.0

    # Geometric Sharpe
    geom_sharpe = (cagr - rf_rate) / ann_volatility if ann_volatility != 0 else 0.0

    # Sortino Ratio
    downside_returns = p_ret[p_ret < 0]
    downside_deviation = np.sqrt((downside_returns**2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = (ann_arith_return - rf_rate) / downside_deviation if downside_deviation != 0 else 0.0

    # Treynor Ratio
    treynor = (ann_arith_return - rf_rate) / beta if beta != 0 else 0.0

    return {
        "Mean Annual Return": ann_arith_return,
        "CAGR": cagr,
        "Standard Deviation": ann_volatility,
        "Alpha": alpha,
        "Beta": beta,
        "Sharpe Ratio": sharpe,
        "Geometric Sharpe Ratio": geom_sharpe,
        "Sortino Ratio": sortino,
        "Treynor Ratio": treynor
    }

def get_comprehensive_metrics(data: pd.DataFrame, allocations: dict, custom_start: str, end_date: str, risk_free_rate: float, market_ticker=MARKET_TICKER) -> pd.DataFrame:
    try:
        end = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        return pd.DataFrame()

    periods = {
        "1 Year": end - relativedelta(years=1),
        "3 Year": end - relativedelta(years=3),
        "5 Year": end - relativedelta(years=5),
        "10 Year": end - relativedelta(years=10),
    }

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
    valid_weights = np.array([allocations[t] for t in valid_tickers])

    if not np.isclose(valid_weights.sum(), 1.0) and valid_weights.sum() > 0:
        valid_weights = valid_weights / valid_weights.sum()

    portfolio_returns = daily_returns.dot(valid_weights)

    if market_ticker in data.columns:
        market_returns = data[market_ticker].pct_change().dropna()
    else:
        market_returns = pd.Series(0, index=portfolio_returns.index)

    results = {}
    for label, start_date in periods.items():
        start_str = start_date.strftime('%Y-%m-%d')
        period_p_ret = portfolio_returns.loc[start_str:]
        period_m_ret = market_returns.loc[start_str:]

        expected_days = (end - start_date).days * (TRADING_DAYS / 365.25)
        if len(period_p_ret) < (expected_days * 0.8) and label != "Custom":
            metrics = {k: None for k in [
                "Mean Annual Return", "CAGR", "Standard Deviation",
                "Alpha", "Beta", "Sharpe Ratio", "Geometric Sharpe Ratio",
                "Sortino Ratio", "Treynor Ratio"
            ]}
        else:
            metrics = calculate_all_metrics(period_p_ret, period_m_ret, risk_free_rate)

        results[label] = metrics

    df_results = pd.DataFrame(results)
    return df_results
