"""``TICKERS``: one company's reference record, and price-table routing."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data.sharadar import transport
from app.data.sharadar.transport import TickerNotFoundError


def _fetch_ticker_meta(ticker: str) -> dict:
    frame = transport._request_all(
        "TICKERS", {"table": "SF1", "ticker": ticker}, ticker=ticker
    )
    if frame.empty:
        raise TickerNotFoundError(
            f"Sharadar has no SF1 company record for {ticker}.",
            table="TICKERS",
            ticker=ticker,
        )
    record = frame.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in record.items()}


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def fetch_ticker_meta(ticker: str) -> dict:
    """Reference data for one company.

    Carries ``name``, ``currency``, ``sector``, ``industry``,
    ``siccode``/``sicsector`` for the financial-sector gate, and ``secfilings``
    — an EDGAR company URL, which is the closest thing Sharadar has to a
    per-filing source link.
    """
    return _fetch_ticker_meta(ticker)


def _fetch_price_tables(tickers: tuple[str, ...]) -> dict[str, str]:
    """Maps each ticker to the price table that carries it: ``SEP`` or ``SFP``.

    Equities live in SEP and funds in SFP, and nothing in a ticker string says
    which. One TICKERS lookup routes the whole batch.
    """
    if not tickers:
        return {}
    frame = transport._request_all("TICKERS", {"ticker": ",".join(tickers)})
    if frame.empty or not {"ticker", "table"}.issubset(frame.columns):
        return {}
    routes: dict[str, str] = {}
    for row in frame.itertuples():
        table = str(row.table).upper()
        if table in ("SEP", "SFP"):
            # A ticker can appear under SF1 as well; only price tables count.
            routes.setdefault(str(row.ticker).upper(), table)
    return routes


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def fetch_price_tables(tickers: tuple[str, ...]) -> dict[str, str]:
    """Cached ticker → price-table routing. Listings barely move; cached a week."""
    return _fetch_price_tables(tickers)
