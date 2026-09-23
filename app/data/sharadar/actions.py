"""``ACTIONS``: dividends, splits, listings and ticker changes for one company.

The action names and :func:`~app.data.sharadar.vocabulary.split_factors` live in
the vocabulary, because the logic layer reads them.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data.sharadar import transport

#: ACTIONS fills the contra columns with the *string* ``"N/A"`` rather than a
#: null when an action has no counterparty, which would otherwise render as a
#: literal "N/A" ticker on screen.
_ACTIONS_NULL_TEXT = frozenset({"N/A", "NA", ""})


def _fetch_actions(ticker: str, start: str | None = None) -> pd.DataFrame:
    params: dict = {"ticker": ticker}
    if start:
        params["date.gte"] = start

    frame = transport._request_all("ACTIONS", params, ticker=ticker)
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if "value" in frame.columns:
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    if "action" in frame.columns:
        frame["action"] = frame["action"].astype(str).str.strip().str.lower()
    for column in ("contraticker", "contraname"):
        if column in frame.columns:
            frame[column] = frame[column].where(
                ~frame[column].astype(str).str.strip().str.upper().isin(_ACTIONS_NULL_TEXT)
            )

    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_actions(ticker: str, start: str | None = None) -> pd.DataFrame:
    """Corporate actions for one ticker, oldest first.

    Columns are Sharadar's own: ``date action ticker name value contraticker
    contraname``. On a ``dividend`` row ``date`` **is the ex-dividend date** and
    ``value`` is the dividend per share; on a ``split`` row ``value`` is the
    ratio. That mapping is documented nowhere in the vendor's column titles —
    it was confirmed against live responses.

    Returns an **empty frame rather than raising** when there is nothing: a
    company that has never paid a dividend or split is ordinary, not an error,
    and an empty response cannot be told apart from an unknown ticker here.
    Callers that need the ticker validated should use ``fetch_ticker_meta``.

    Rows arrive newest-first from the API — the reverse of every other table
    here — so the sort is load-bearing, not defensive.
    """
    return _fetch_actions(ticker, start)
