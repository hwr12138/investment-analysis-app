"""The cross-sectional universe: every listed company at once, keyed on a date.

Every other fetch in this package is keyed on a ticker. These three are keyed on
a date, and that difference is the whole point: one request returns the multiple
for every listed company, so a peer median costs the same whether the peer group
has four members or four hundred. Measured against the live API: DAILY for one
date is 5,446 rows on a single page in ~1.1s.

Because the cache keys carry no ticker, one entry serves every company looked up
that day. A second search is free.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data.sharadar import prices, transport

#: Columns pulled from TICKERS for peer grouping. Projected rather than taking
#: the default 30-odd, which is 2 MB against ~10 MB for the same 17,800 rows.
_DIRECTORY_COLUMNS = "ticker,name,sector,industry,scalemarketcap,isdelisted"

#: SF1 indicators the universe snapshot carries. ``revenue``/``ncfo``/``ncfdiv``
#: feed the three multiples DAILY does not publish; the rest are the
#: year-on-year growth terms. Projected rather than taking the full ~110, which
#: is what keeps 36,000 rows to four pages.
_UNIVERSE_SF1_INDICATORS = (
    "revenue", "ncfo", "ncfdiv", "ebitda", "ebit", "epsdil",
    "fcf", "roe", "workingcapital", "capex", "dps",
)

_UNIVERSE_SF1_COLUMNS = "ticker,calendardate,datekey," + ",".join(
    _UNIVERSE_SF1_INDICATORS
)

#: How far back the universe fundamentals request reaches.
#:
#: Six months would be enough for the *levels* the valuation multiples need — a
#: quarter would not, because asking for ``calendardate=2026-06-30`` on
#: 2026-08-09 returned 2,818 rows against 5,133 for the quarter before, with Q2
#: filings still arriving. Eighteen months is what the **growth** comparison
#: needs: every company's latest trailing-twelve-month figures *and* the ones
#: four quarters earlier, in one request rather than two. Measured at 36,199
#: rows over four pages in ~4.8s, with 5,470 of 6,054 tickers finding their
#: exact year-ago quarter — the rest are recent listings with no year-on-year
#: to report.
UNIVERSE_LOOKBACK_DAYS = 550


def _fetch_universe_daily(as_of: str) -> pd.DataFrame:
    """Every company's valuation ratios on one date.

    ``as_of`` must be a **trading day**: Sharadar returns zero rows for a
    weekend or a holiday rather than the previous session, so a caller passing
    ``date.today()`` gets an empty universe every Saturday. Callers should pass
    the last date they have already seen in a real DAILY response.
    """
    return prices._normalise_daily(transport._request_all("DAILY", {"date": as_of}))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_universe_daily(as_of: str) -> pd.DataFrame:
    """Cached universe-wide DAILY snapshot: ``ticker date ev evebit evebitda
    marketcap pb pe ps``, with ``marketcap`` and ``ev`` in raw units.

    Shares :func:`_normalise_daily` with :func:`fetch_daily_metrics`, which is
    what keeps the millions conversion in one place across both call sites.
    """
    return _fetch_universe_daily(as_of)


def _fetch_universe_fundamentals(since: str) -> pd.DataFrame:
    """Every company's trailing-twelve-month SF1 rows across the window.

    ``ART`` rather than ``MRT``: as-reported time-indexes to the filing date, so
    ``datekey.gte`` selects on when a figure became public rather than on the
    period it covers. Filtering on the period would hand back numbers nobody
    could have known at the time.

    **Several rows per ticker, not one.** Growth is a year-on-year comparison
    and needs both ends of it, so the whole window is returned and the callers
    pick the periods they want — the latest for the multiples, the latest and
    its year-ago twin for the growth rates.

    The one collapse that does happen here is the **restatement**: a company
    that filed twice for the *same* ``calendardate`` keeps only the later
    ``datekey``. That is a correction of one period, not two periods of history,
    and letting both through would double-count a quarter.
    """
    frame = transport._request_all(
        "SF1",
        {
            "dimension": "ART",
            "datekey.gte": since,
            "qopts.columns": _UNIVERSE_SF1_COLUMNS,
        },
    )
    if frame.empty or not {"ticker", "datekey", "calendardate"}.issubset(frame.columns):
        return pd.DataFrame()

    for column in _UNIVERSE_SF1_INDICATORS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("datekey", "calendardate"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")

    frame = frame.dropna(subset=["datekey", "calendardate"]).sort_values("datekey")
    frame = frame.groupby(["ticker", "calendardate"], as_index=False).tail(1)
    return frame.sort_values(["ticker", "calendardate"]).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_universe_fundamentals(since: str) -> pd.DataFrame:
    """Cached universe-wide TTM fundamentals, one row per ticker and period."""
    return _fetch_universe_fundamentals(since)


def _fetch_ticker_directory() -> pd.DataFrame:
    """Sector and industry for every company with SF1 fundamentals."""
    frame = transport._request_all(
        "TICKERS", {"table": "SF1", "qopts.columns": _DIRECTORY_COLUMNS}
    )
    if frame.empty or "ticker" not in frame.columns:
        return pd.DataFrame()
    for column in ("sector", "industry", "isdelisted"):
        if column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()
    return frame.reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def fetch_ticker_directory() -> pd.DataFrame:
    """Cached ticker → sector/industry directory. Classifications barely move.

    Sharadar's ``industry`` is the vendor's own ~145-bucket classification, not
    the SIC one. The two disagree, sometimes wildly — a blank-cheque shell is
    ``Industrials`` here and ``Finance Insurance And Real Estate`` under SIC —
    so which is used has to be a deliberate choice at the call site. The
    financial-sector *gate* stays on SIC (see :data:`FINANCIAL_SIC_RANGE`);
    peer grouping uses this, because it is the classification a reader
    recognises.
    """
    return _fetch_ticker_directory()
