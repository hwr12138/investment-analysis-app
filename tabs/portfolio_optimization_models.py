import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime
from dateutil.relativedelta import relativedelta
import streamlit as st

TRADING_DAYS = 251

# --- Global Helpers & Data Fetching ---

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_historical_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical adjusted close prices for a list of tickers.
    Cached to prevent redundant API calls when UI parameters change.
    """
    if not tickers:
        return pd.DataFrame()
    
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    
    if df.empty:
        return pd.DataFrame()

    # Handle multi-index columns vs single ticker series
    data = df['Adj Close']
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0])
        
    return data.dropna(how='all')

def fetch_current_risk_free_rate(ticker: str = "ZQ=F") -> float:
    """
    Fetches the current 30-Day Federal Funds Futures price to calculate the implied risk-free rate.
    Priced at 100 - rate (e.g., a price of 95.00 implies a 5.00% rate).
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d") # Fetch last 5 days to ensure we get a valid trading day
        if not hist.empty:
            latest_price = hist['Close'].iloc[-1]
            implied_rate = (100 - latest_price) / 100.0
            return implied_rate
    except Exception:
        pass
    # Fallback default if Yahoo Finance fails to fetch the futures contract
    return 0.04 

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

# --- Optimizers ---

def maximize_geometric_sharpe(data: pd.DataFrame, risk_free_rate: float):
    """
    Finds the portfolio allocation that maximizes the Geometric Sharpe Ratio.
    Returns: (allocation_dict, expected_cagr, expected_volatility, max_geo_sharpe)
    """
    if data.empty or data.shape[1] < 2:
        return None, 0.0, 0.0, 0.0

    daily_returns = data.pct_change().dropna()
    num_assets = len(data.columns)
    tickers = data.columns.tolist()

    def neg_geom_sharpe(weights):
        port_returns = daily_returns.dot(weights)
        total_compound_return = (1 + port_returns).prod()
        ann_geom_return = total_compound_return ** (TRADING_DAYS / len(port_returns)) - 1
        ann_volatility = port_returns.std() * np.sqrt(TRADING_DAYS)

        if ann_volatility == 0: return 0.0
        return -((ann_geom_return - risk_free_rate) / ann_volatility)

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]

    result = minimize(neg_geom_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        return None, 0.0, 0.0, 0.0

    optimal_weights = result.x
    max_geo_sharpe = -result.fun
    
    allocation = {tickers[i]: round(optimal_weights[i], 4) for i in range(num_assets)}

    optimal_returns = daily_returns.dot(optimal_weights)
    total_compound = (1 + optimal_returns).prod()
    ann_geom_ret = total_compound ** (TRADING_DAYS / len(optimal_returns)) - 1
    ann_vol = optimal_returns.std() * np.sqrt(TRADING_DAYS)

    return allocation, ann_geom_ret, ann_vol, max_geo_sharpe

def volatility_targeting_optimizer(data: pd.DataFrame, vol_constraint: float, risk_free_rate: float):
    """
    Maximizes expected return subject to a maximum annualized volatility constraint.
    Returns: (allocation_dict, expected_return, expected_volatility, sharpe)
    """
    if data.empty or data.shape[1] < 2:
        return None, 0.0, 0.0, 0.0

    returns = data.pct_change().dropna()
    expected_returns = returns.mean() * TRADING_DAYS
    cov_matrix = returns.cov() * TRADING_DAYS
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
        return None, 0.0, 0.0, 0.0

    opt_weights = result.x
    opt_ret = np.dot(opt_weights, expected_returns)
    opt_vol = portfolio_volatility(opt_weights)
    sharpe = (opt_ret - risk_free_rate) / opt_vol if opt_vol > 0 else 0

    allocation = {tickers[i]: round(opt_weights[i], 4) for i in range(num_assets)}
    
    return allocation, opt_ret, opt_vol, sharpe

def calculate_kelly_allocation(data: pd.DataFrame, risk_free_rate: float, kelly_fraction: float = 0.5):
    """
    Finds the optimal fractional Kelly portfolio weights.
    Returns: (allocation_dict, expected_return, expected_volatility)
    """
    if data.empty or data.shape[1] < 2:
        return None, 0.0, 0.0

    daily_returns = data.pct_change().dropna()
    mean_returns = daily_returns.mean() * TRADING_DAYS
    cov_matrix = daily_returns.cov() * TRADING_DAYS
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
        return None, 0.0, 0.0

    raw_weights = result.x
    fractional_weights = raw_weights * kelly_fraction
    cash_allocation = 1.0 - np.sum(fractional_weights)

    port_return = np.dot(fractional_weights, mean_returns) + (cash_allocation * risk_free_rate)
    port_vol = np.sqrt(np.dot(fractional_weights.T, np.dot(cov_matrix, fractional_weights)))

    allocation = {tickers[i]: round(fractional_weights[i], 4) for i in range(num_assets)}
    allocation['CASH'] = round(cash_allocation, 4)

    return allocation, port_return, port_vol

# --- Yearly Returns Analysis ---

def get_portfolio_yearly_returns(data: pd.DataFrame, weights_dict: dict):
    """
    Calculates yearly returns and the compounded annual return for a given portfolio.
    Returns: (yearly_returns_series, cagr)
    """
    tickers = list(weights_dict.keys())
    # Exclude 'CASH' from historical data mapping if it was added by Kelly
    valid_tickers = [t for t in tickers if t in data.columns]
    
    if data.empty or not valid_tickers:
        return pd.Series(), 0.0

    returns = data[valid_tickers].pct_change().dropna()
    portfolio_weights = np.array([weights_dict[t] for t in valid_tickers])
    
    # Normalize if weights don't sum to 1 (ignoring cash)
    if not np.isclose(portfolio_weights.sum(), 1.0) and portfolio_weights.sum() > 0:
        portfolio_weights = portfolio_weights / portfolio_weights.sum()

    portfolio_daily_returns = (returns * portfolio_weights).sum(axis=1)

    def calculate_annual_return(daily_rets):
        return (1 + daily_rets).prod() - 1

    yearly_returns = portfolio_daily_returns.groupby(portfolio_daily_returns.index.year).apply(calculate_annual_return)

    total_growth = (1 + portfolio_daily_returns).prod()
    n_years = len(portfolio_daily_returns) / TRADING_DAYS
    cagr = (total_growth ** (1 / n_years)) - 1 if n_years > 0 else 0.0

    return yearly_returns, cagr
