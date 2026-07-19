"""Shared market-data access layer.

Centralizes every external API call (Yahoo Finance, Financial Modeling Prep,
SimFin, and the ERP scrape) so that caching and API keys live in one place and
fetchers can be reused across pages.
"""

import re
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta

from app.defaults import DEFAULTS
from app.provenance import Origin, Resolved

# --- Value Provenance ---


class _FetchError(Exception):
    """A fetch that failed, or succeeded without yielding a usable number.

    Raised from inside the cached fetchers rather than returning a default:
    st.cache_data stores return values but not exceptions, so failing loudly is
    what keeps a transient outage from pinning the fallback for the full TTL.
    """

    origin = Origin.FETCH_FAILED


class _MissingConfig(_FetchError):
    """A required API key is absent — a setup problem, not an outage."""

    origin = Origin.MISSING_CONFIG


def _secret(name: str):
    """Reads one secret, treating an absent secrets.toml like an absent key.

    st.secrets raises rather than returning None when no secrets file exists at
    all, which is the normal state of a fresh checkout.
    """
    try:
        return st.secrets.get(name)
    except Exception:
        return None


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

    Returns the most recent record, a dict of {tenor: rate_percent}. Raises
    _FetchError if the key is missing or the call comes back unusable, so that
    nothing lands in the cache and the next rerun retries.
    """
    api_key = _secret("FMP_API_KEY")
    if not api_key:
        raise _MissingConfig("FMP_API_KEY is not set in .streamlit/secrets.toml")

    url = f"https://financialmodelingprep.com/stable/treasury-rates?apikey={api_key}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as e:
        raise _FetchError(f"FMP treasury request failed: {e}") from e

    if not data:
        raise _FetchError("FMP returned no treasury records")
    return data[0]


def _fetch_treasury_rate(period: str) -> Resolved:
    """Resolves one Treasury tenor to a decimal rate."""
    default = DEFAULTS['risk_free_rate']
    try:
        rates = _fetch_fmp_treasury_rates()
    except _FetchError as e:
        return default.fallback(e.origin, str(e))

    rate_pct = rates.get(period)
    if rate_pct is None:
        return default.fallback(Origin.FETCH_FAILED, f"FMP quotes no '{period}' tenor")

    try:
        return Resolved.observed(float(rate_pct) / 100.0, f"fmp:{period}")
    except (TypeError, ValueError):
        return default.fallback(
            Origin.FETCH_FAILED, f"FMP returned a non-numeric '{period}' rate: {rate_pct!r}"
        )


def fetch_current_risk_free_rate(period="month1") -> Resolved:
    """
    Returns the current short-term Treasury rate (1-month by default) to use as
    the risk-free rate.
    """
    return _fetch_treasury_rate(period)


def fetch_10y_treasury_rate() -> Resolved:
    """Returns the current 10-year Treasury rate from FMP."""
    return _fetch_treasury_rate("year10")


# --- Equity Risk Premium & Beta ---


@st.cache_data(show_spinner=False, ttl=3600 * 24)
def _scrape_erp() -> float:
    """Scrapes Aswath Damodaran's website for the current Implied Equity Risk Premium.

    The page is parsed by regex off a tag-stripped copy, so a layout change
    reads exactly like an outage — both raise, and neither is cached.
    """
    url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise _FetchError(f"ERP page request failed: {e}") from e

    # Strip HTML tags, then find "Implied ERP... = X.XX%"
    text = re.sub(r'<[^>]+>', '', response.text)
    match = re.search(r'Implied ERP.*?=\s*([\d\.]+)', text, re.IGNORECASE)
    if not match:
        raise _FetchError("no 'Implied ERP' figure on the page — the layout may have changed")

    try:
        return float(match.group(1)) / 100.0
    except ValueError as e:
        raise _FetchError(f"unparseable ERP figure {match.group(1)!r}") from e


def fetch_erp() -> Resolved:
    """Returns the current Implied Equity Risk Premium as a decimal."""
    try:
        return Resolved.observed(_scrape_erp(), "damodaran")
    except _FetchError as e:
        return DEFAULTS['erp'].fallback(e.origin, str(e))


@st.cache_data(show_spinner=False, ttl=3600 * 24)
def _calculate_beta(ticker: str) -> float:
    """Calculates rolling 3-year monthly Beta against S&P 500 (^GSPC)."""
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=3)

    try:
        # Fetch monthly data
        data = yf.download(
            [ticker, "^GSPC"],
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval="1mo",
            progress=False,
        )
    except Exception as e:
        raise _FetchError(f"Yahoo Finance request failed: {e}") from e

    if data.empty:
        raise _FetchError(f"Yahoo Finance returned no monthly history for {ticker}")

    for column in ('Adj Close', 'Close'):
        if column in data:
            prices = data[column]
            break
    else:
        raise _FetchError(f"Yahoo Finance returned no closing prices for {ticker}")

    returns = prices.pct_change().dropna()
    if ticker not in returns.columns or "^GSPC" not in returns.columns:
        raise _FetchError(f"no overlapping monthly returns for {ticker} and ^GSPC")

    # A zero or absent variance would divide out to inf, which reads downstream
    # as an extraordinarily risky stock rather than as missing data.
    var = returns["^GSPC"].var()
    if pd.isna(var) or var == 0:
        raise _FetchError(f"^GSPC return variance is {var} — too few observations for {ticker}")

    beta = returns.cov().loc[ticker, "^GSPC"] / var
    if pd.isna(beta):
        raise _FetchError(f"Beta computed as NaN for {ticker}")
    return float(beta)


def fetch_beta(ticker: str) -> Resolved:
    """Returns 3-year monthly Beta against the S&P 500."""
    try:
        return Resolved.observed(_calculate_beta(ticker), "yfinance:3y-monthly")
    except _FetchError as e:
        return DEFAULTS['beta'].fallback(e.origin, str(e))


# --- Company Financials (SimFin) ---

SIMFIN_API_BASE = "https://backend.simfin.com/api/v3"


def _simfin_get(path: str, params: dict, api_key: str):
    """Performs a GET against the SimFin v3 API and returns the parsed JSON."""
    response = requests.get(
        f"{SIMFIN_API_BASE}/{path}",
        params=params,
        headers={"Authorization": f"api-key {api_key}", "Accept": "application/json"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def _statement_frame(company: dict, statement: str) -> pd.DataFrame:
    """Extracts one compact statement from a SimFin payload into a DataFrame.

    Indexed by fiscal year, most recent first — SimFin returns rows oldest-first,
    but every caller reads the latest figure off .iloc[0].
    """
    for stmt in company.get("statements", []):
        if stmt.get("statement") != statement:
            continue
        df = pd.DataFrame(stmt.get("data") or [], columns=stmt.get("columns") or [])
        if df.empty or "Fiscal Year" not in df.columns:
            break
        df = df.set_index("Fiscal Year")
        # A restated year appears twice; the later row is the newer filing.
        df = df[~df.index.duplicated(keep="last")]
        return df.sort_index(ascending=False)
    return pd.DataFrame()


def _pick(df: pd.DataFrame, names: list) -> pd.Series:
    """Coalesces candidate columns into a single numeric series.

    Each name only fills the years its predecessors left null. SimFin moves a
    line item between labels over time (Apple's capex sits in 'Acquisition of
    Fixed Assets & Intangibles' through FY2021, then in 'Change in Fixed Assets
    & Intangibles') and renames it outright on the bank/insurance templates, so
    picking a single column would silently drop years.
    """
    series = pd.Series(index=df.index, dtype="float64")
    for name in names:
        if name in df.columns:
            series = series.fillna(pd.to_numeric(df[name], errors="coerce"))
    return series


def _fetch_latest_quote(ticker: str, api_key: str) -> tuple:
    """Returns (shares outstanding, last close) from SimFin's daily price series.

    Looks back 90 days so the payload stays small while still landing on a
    trading day across long closures or a halt.
    """
    start = (datetime.today() - relativedelta(days=90)).strftime('%Y-%m-%d')
    payload = _simfin_get("companies/prices/compact", {"ticker": ticker, "start": start}, api_key)
    if not payload:
        return 0.0, 0.0

    prices = pd.DataFrame(payload[0].get("data") or [], columns=payload[0].get("columns") or [])

    def last_reported(column):
        if column not in prices.columns:
            return 0.0
        values = pd.to_numeric(prices[column], errors="coerce").dropna()
        return float(values.iloc[-1]) if not values.empty else 0.0

    return last_reported("Common Shares Outstanding"), last_reported("Last Closing Price")


@st.cache_data(show_spinner=False, ttl=3600 * 24)
def fetch_simfin_financials(ticker: str, years: int = 7) -> dict:
    """
    Fetches the last 7 years of Income Statement, Balance Sheet, and Cash Flow
    Statement from the SimFin v3 API.

    Every series is indexed by fiscal year and ordered most-recent-first, so
    line items drawn from different statements divide cleanly against each other
    and .iloc[0] is the latest year.
    """
    api_key = _secret("SIMFIN_API_KEY")
    if not api_key:
        st.error("SIMFIN_API_KEY not found in secrets.")
        return {}

    try:
        payload = _simfin_get(
            "companies/statements/compact",
            {"ticker": ticker, "statements": "pl,bs,cf", "period": "fy"},
            api_key,
        )
        shares, current_price = _fetch_latest_quote(ticker, api_key)
    except Exception as e:
        st.error(f"SimFin request failed for {ticker}: {e}")
        return {}

    if not payload:
        st.error(f"SimFin has no data for {ticker}. Check the ticker symbol.")
        return {}

    company = payload[0]
    inc = _statement_frame(company, "PL").head(years)
    bs = _statement_frame(company, "BS").head(years)
    cf = _statement_frame(company, "CF").head(years)

    if inc.empty or bs.empty or cf.empty:
        st.error(f"SimFin is missing one or more financial statements for {ticker}.")
        return {}

    # 'Net Revenue' and 'Total Interest Expense' are how the BANKS and INSURANCES
    # templates label these; the rest of the alternates cover GENERAL.
    revenue = _pick(inc, ['Revenue', 'Net Revenue', 'Sales & Services Revenue', 'Total Revenue'])
    if revenue.isna().all():
        # Bail out rather than hand back a series that quietly turns the whole
        # valuation into NaN.
        st.error(f"SimFin reports no revenue line for {ticker} "
                 f"(statement template: {company.get('template', 'unknown')}).")
        return {}

    ebit = _pick(inc, ['Operating Income (Loss)'])
    pretax_income = _pick(inc, ['Pretax Income (Loss)', 'Pretax Income (Loss), Adjusted'])
    # SimFin signs expenses negative. Negate rather than take the magnitude so a
    # genuine tax benefit stays negative and trips the caller's sanity check.
    tax_provision = -_pick(inc, ['Income Tax (Expense) Benefit, net'])
    interest_expense = _pick(inc, [
        'Interest Expense',
        'Interest Expense, net',
        'Total Interest Expense',
    ]).abs()

    dna = _pick(cf, ['Depreciation & Amortization'])
    capex = _pick(cf, [
        'Acquisition of Fixed Assets & Intangibles',
        'Change in Fixed Assets & Intangibles',
        'Change in Fixed Assets and Intangibles',  # INSURANCES spells out the "and"
        'Capital Expenditures',
        'Purchase of Fixed Assets',
    ]).abs()

    # NWC = Current Assets - Current Liabilities. Banks and insurers file an
    # unclassified balance sheet, so this stays NaN for them and the caller
    # falls back to its default NWC margin.
    nwc = _pick(bs, ['Total Current Assets']) - _pick(bs, ['Total Current Liabilities'])

    # SimFin has no single total-debt line, and companies that don't split the
    # maturities leave one side null, so add with a fill rather than summing.
    short_term_debt = _pick(bs, ['Short Term Debt', 'Short Term Borrowings'])
    long_term_debt = _pick(bs, ['Long Term Debt', 'Long Term Borrowings'])
    total_debt = short_term_debt.add(long_term_debt, fill_value=0).sort_index(ascending=False)

    cash = _pick(bs, ['Cash & Cash Equivalents', 'Cash, Cash Equivalents & Short Term Investments'])

    return {
        'revenue': revenue,
        'ebit': ebit,
        'tax_provision': tax_provision,
        'pretax_income': pretax_income,
        'dna': dna,
        'capex': capex,
        'nwc': nwc,
        'total_debt': total_debt,
        'interest_expense': interest_expense,
        'cash': cash,
        'shares': shares,
        'current_price': current_price,
        # Which statement template SimFin filed this under. BANKS and INSURANCES
        # legitimately omit line items the GENERAL template carries, so the
        # caller can tell "not reported" apart from "doesn't exist for a bank".
        'template': company.get('template', ''),
    }
