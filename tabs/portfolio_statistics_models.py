import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

TRADING_DAYS = 251

# --- Metric Calculations ---

def calculate_metrics(returns: pd.Series, rf_rate: float):
    if returns.empty:
        return None, None, None

    # 1. Traditional Sharpe (Arithmetic)
    ann_arith_return = returns.mean() * TRADING_DAYS
    ann_volatility = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_arith_return - rf_rate) / ann_volatility if ann_volatility != 0 else 0.0

    # 2. Geometric Sharpe
    total_compound_return = (1 + returns).prod()
    ann_geom_return = total_compound_return ** (TRADING_DAYS / len(returns)) - 1
    geom_sharpe = (ann_geom_return - rf_rate) / ann_volatility if ann_volatility != 0 else 0.0

    # 3. Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_deviation = np.sqrt((downside_returns**2).mean()) * np.sqrt(TRADING_DAYS)
    sortino = (ann_arith_return - rf_rate) / downside_deviation if downside_deviation != 0 else 0.0

    return sharpe, geom_sharpe, sortino

def get_portfolio_risk_metrics(data: pd.DataFrame, allocations: dict, end_date: str, risk_free_rate: float) -> pd.DataFrame:
    """
    Calculates 3, 5, and 10 year risk metrics based on pre-fetched DataFrame.
    """
    try:
        end = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        return pd.DataFrame()

    start_10yr = end - relativedelta(years=10)
    start_5yr = end - relativedelta(years=5)
    start_3yr = end - relativedelta(years=3)

    tickers = list(allocations.keys())
    weights = np.array(list(allocations.values()))

    # Normalize weights if they don't exactly equal 1
    if not np.isclose(weights.sum(), 1.0) and weights.sum() > 0:
        weights = weights / weights.sum()

    # Filter available tickers in data
    valid_tickers = [t for t in tickers if t in data.columns]
    if not valid_tickers:
        return pd.DataFrame()

    daily_returns = data[valid_tickers].pct_change().dropna()
    valid_weights = np.array([allocations[t] for t in valid_tickers])
    
    if not np.isclose(valid_weights.sum(), 1.0) and valid_weights.sum() > 0:
        valid_weights = valid_weights / valid_weights.sum()

    portfolio_returns = daily_returns.dot(valid_weights)
    
    periods = {
        "3 Year": start_3yr,
        "5 Year": start_5yr,
        "10 Year": start_10yr
    }

    results = []
    for label, start_date in periods.items():
        # Using string representation for slicing compatibility
        period_returns = portfolio_returns.loc[start_date.strftime('%Y-%m-%d'):]
        
        expected_days = (end - start_date).days * (TRADING_DAYS / 365.25)
        if len(period_returns) < (expected_days * 0.9):
            results.append({"Period": label, "Trad. Sharpe": None, "Geom. Sharpe": None, "Sortino": None})
            continue

        sharpe, geom_sharpe, sortino = calculate_metrics(period_returns, risk_free_rate)
        results.append({
            "Period": label, 
            "Trad. Sharpe": sharpe, 
            "Geom. Sharpe": geom_sharpe, 
            "Sortino": sortino
        })

    return pd.DataFrame(results)