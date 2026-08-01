"""Shared market-data access layer.

Centralizes every external API call (Yahoo Finance, Financial Modeling Prep,
and the ERP scrape) so that caching and API keys live in one place and fetchers
can be reused across pages. SimFin lives in its sibling ``simfin_client``.

Two of the fetchers here return :class:`~app.provenance.Sourced` rather than a
bare float: a scrape or a third-party rate can fail, and the DCF page must be
able to tell "4.18% scraped from Damodaran on 1 July 2026" apart from "the
scrape broke and you are looking at a constant". Anything that returns a plain
float substitutes a hardcoded default on failure and says so in its docstring.
"""

import re
from datetime import date, datetime

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from app.provenance import Sourced

#: An implied ERP outside this band is not credible and is treated as a failed
#: scrape rather than passed through. Consistent with the DCF spec's 2-10% band.
ERP_PLAUSIBLE_BAND = (0.02, 0.10)

DAMODARAN_URL = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm"

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
def _fetch_fmp_treasury_rates() -> tuple[dict, str]:
    """Fetches the latest Treasury-rates record from Financial Modeling Prep.

    Returns ``(record, error)`` where ``record`` is the most recent
    ``{tenor: rate_percent}`` dict and ``error`` is an empty string on success
    or a reason on failure. The reason is what lets callers that must not
    fabricate a rate explain themselves; callers that tolerate a default can
    ignore it.
    """
    try:
        api_key = st.secrets.get("FMP_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        return {}, "FMP_API_KEY is not set in .streamlit/secrets.toml"

    try:
        # The key travels as a query parameter because that is the only auth
        # this endpoint accepts; it must never be echoed into an error message.
        url = f"https://financialmodelingprep.com/stable/treasury-rates?apikey={api_key}"
        response = requests.get(url, timeout=5)
    except Exception:
        return {}, "Could not reach Financial Modeling Prep"

    if response.status_code != 200:
        return {}, f"Financial Modeling Prep returned HTTP {response.status_code}"
    try:
        data = response.json()
    except ValueError:
        return {}, "Financial Modeling Prep returned a response that is not valid JSON"
    if not data:
        return {}, "Financial Modeling Prep returned no Treasury records"
    return data[0], ""


def fetch_current_risk_free_rate(period="month1") -> float:
    """
    Returns the current short-term Treasury rate (1-month by default) to use as
    the risk-free rate.

    Falls back to a hardcoded 4% if the API fails. Callers that must not show a
    fabricated rate should use :func:`fetch_10y_treasury_rate` instead, which
    reports the failure rather than hiding it.
    """
    rates, _ = _fetch_fmp_treasury_rates()
    rate_pct = rates.get(period)
    if rate_pct is not None:
        return float(rate_pct) / 100.0
    # Fallback default if the API fails or the key is missing
    return 0.04


def fetch_10y_treasury_rate() -> Sourced:
    """Returns the current 10-year Treasury rate with its provenance.

    On failure the returned :class:`Sourced` has ``value=None`` and carries the
    reason. Nothing is substituted — a DCF discounted at a made-up risk-free
    rate is worse than one the user had to fill in by hand.
    """
    rates, error = _fetch_fmp_treasury_rates()
    rate_pct = rates.get("year10")
    if rate_pct is None:
        return Sourced.unavailable(
            source="Financial Modeling Prep (treasury-rates, year10)",
            reason=error or "Financial Modeling Prep did not return a 10-year rate",
        )

    as_of = None
    raw_date = rates.get("date")
    if raw_date:
        try:
            as_of = datetime.strptime(str(raw_date)[:10], "%Y-%m-%d").date()
        except ValueError:
            as_of = None

    return Sourced.reported(
        value=float(rate_pct) / 100.0,
        source="Financial Modeling Prep — 10-year Treasury constant maturity",
        as_of=as_of,
    )


# --- Equity Risk Premium & Beta ---


@st.cache_data(show_spinner=False, ttl=3600 * 24)
def fetch_erp() -> Sourced:
    """Scrapes Aswath Damodaran's site for the current Implied Equity Risk Premium.

    The page states it as ``Implied ERP on July 1, 2026 = 4.18%``, so the as-of
    date comes off the same line as the number and both are reported together.

    Scraping is fragile by nature. If the request fails, the pattern stops
    matching, or the value lands outside :data:`ERP_PLAUSIBLE_BAND`, this
    returns an unavailable :class:`Sourced` with the reason. It never
    substitutes a constant — the caller is expected to fall through to manual
    entry and say why.
    """
    try:
        response = requests.get(DAMODARAN_URL, timeout=10)
    except Exception:
        return Sourced.unavailable(
            source=DAMODARAN_URL, reason="Could not reach the Damodaran page"
        )

    if response.status_code != 200:
        return Sourced.unavailable(
            source=DAMODARAN_URL,
            reason=f"The Damodaran page returned HTTP {response.status_code}",
        )

    # Tags are stripped to nothing, not to a space: the page splits the figure
    # across elements as `4.<span></span><span>18%`, so anything inserted
    # between them turns 4.18 into a 4.00 that looks perfectly reasonable.
    text = re.sub(r"<[^>]+>", "", response.text)
    text = text.replace("&nbsp;", " ")

    # The trailing % is required, not optional. Without it a truncated parse of
    # "4." matches happily and silently reports 4.00%.
    dated = re.search(
        r"Implied\s+ERP\s+on\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})\s*=\s*(\d+(?:\.\d+)?)\s*%",
        text,
        re.IGNORECASE,
    )
    as_of: date | None = None
    if dated:
        raw_value = dated.group(2)
        try:
            as_of = datetime.strptime(
                re.sub(r"\s+", " ", dated.group(1)).strip(), "%B %d, %Y"
            ).date()
        except ValueError:
            as_of = None
    else:
        # The number without a date is still usable, but say the date is missing.
        undated = re.search(
            r"Implied\s+ERP.*?=\s*(\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE
        )
        if not undated:
            return Sourced.unavailable(
                source=DAMODARAN_URL,
                reason="The 'Implied ERP' figure was not found — the page layout has changed",
            )
        raw_value = undated.group(1)

    try:
        erp = float(raw_value) / 100.0
    except ValueError:
        return Sourced.unavailable(
            source=DAMODARAN_URL,
            reason=f"Could not read '{raw_value}' as an ERP percentage",
        )

    low, high = ERP_PLAUSIBLE_BAND
    if not low <= erp <= high:
        return Sourced.unavailable(
            source=DAMODARAN_URL,
            reason=(
                f"Scraped ERP of {erp:.2%} is outside the plausible "
                f"{low:.0%}-{high:.0%} band, so the scrape is assumed broken"
            ),
        )

    return Sourced.reported(
        value=erp,
        source="Damodaran — Implied Equity Risk Premium (S&P 500)",
        as_of=as_of,
    )


# --- Index prices (for beta regression) ---


def fetch_index_prices(index_ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Adjusted closes for a market index, as a Series.

    A thin named wrapper over :func:`fetch_historical_data` so the beta
    regression has an obvious entry point without a second yfinance fetcher.
    """
    frame = fetch_historical_data([index_ticker], start_date, end_date)
    if frame.empty:
        return pd.Series(dtype="float64", name=index_ticker)
    series = frame[index_ticker] if index_ticker in frame.columns else frame.iloc[:, 0]
    return series.dropna().rename(index_ticker)
