"""Shared market-data access layer.

Centralizes every external API call (Yahoo Finance, Financial Modeling Prep,
SimFin, and the ERP scrape) so that caching and API keys live in one place and
fetchers can be reused across pages.
"""

import re

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# --- Price History ---


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


# --- Treasury Rates (Financial Modeling Prep) ---


@st.cache_data(show_spinner=False, ttl=3600 * 24)
def _fetch_fmp_treasury_rates() -> dict:
    """Fetches the latest Treasury-rates record from Financial Modeling Prep.

    Returns the most recent record (a dict of {tenor: rate_percent}) or an empty
    dict if the API key is missing or the request fails.
    """
    try:
        api_key = st.secrets.get("FMP_API_KEY")
        if api_key:
            url = f"https://financialmodelingprep.com/stable/treasury-rates?apikey={api_key}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
    except Exception:
        pass
    return {}


def fetch_current_risk_free_rate(period="month1") -> float:
    """
    Returns the current short-term Treasury rate (1-month by default) to use as
    the risk-free rate.
    """
    rate_pct = _fetch_fmp_treasury_rates().get(period)
    if rate_pct is not None:
        return float(rate_pct) / 100.0
    # Fallback default if the API fails or the key is missing
    return 0.04


def fetch_10y_treasury_rate() -> float:
    """Returns the current 10-year Treasury rate from FMP."""
    rate_pct = _fetch_fmp_treasury_rates().get("year10")
    if rate_pct is not None:
        return float(rate_pct) / 100.0
    # Fallback default
    return 0.04


# --- Equity Risk Premium & Beta ---


@st.cache_data(show_spinner=False, ttl=3600 * 24)
def fetch_erp() -> float:
    """Scrapes Aswath Damodaran's website for the current Implied Equity Risk Premium."""
    try:
        url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Strip HTML tags
            text = re.sub(r'<[^>]+>', '', response.text)
            # Find "Implied ERP... = X.XX%"
            match = re.search(r'Implied ERP.*?=\s*([\d\.]+)', text, re.IGNORECASE)
            if match:
                return float(match.group(1)) / 100.0
    except Exception as e:
        st.warning(f"Failed to scrape ERP: {e}")
    return 0.055  # Default fallback
