"""Nasdaq Data Link transport: auth, retry, pagination, datatable parsing.

Every request any table module makes goes through :func:`_get`, which is why
tests patch ``transport._get`` and nothing else to take the network away. Table
modules call it through this module's attribute — ``transport._request_all`` —
rather than importing the function, so the patch reaches them.

The API key is read from the ``SHARADAR_API_KEY`` environment variable (how
Cloud Run hands over a Secret Manager secret) or else from ``st.secrets``, and
never leaves this module. Nasdaq
only accepts it as a query parameter, so no exception raised here carries the
params or the full URL — only the table, the ticker, and the status.
"""

from __future__ import annotations

import os
import random
import time

import pandas as pd
import requests
import streamlit as st

BASE_URL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"

_TIMEOUT = 45
_MAX_ATTEMPTS = 4

#: Nasdaq truncates at 10,000 rows and pages through an opaque cursor.
_PAGE_SIZE = 10000
#: A five-year daily price series is ~1,260 rows, so this is a runaway guard
#: rather than an expected limit.
_MAX_PAGES = 50


class SharadarError(Exception):
    """A Nasdaq Data Link request failed.

    Deliberately carries only the table name, the HTTP status, and the ticker.
    The request URL and query params are never attached — Nasdaq only accepts
    the API key as a query parameter, so anything that echoes the params leaks
    the key into a traceback, a log line, or a Streamlit error block.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        table: str | None = None,
        ticker: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.table = table
        self.ticker = ticker

    def __str__(self) -> str:
        bits = [self.message]
        if self.table:
            bits.append(f"table SHARADAR/{self.table}")
        if self.ticker:
            bits.append(f"ticker {self.ticker}")
        if self.status:
            bits.append(f"HTTP {self.status}")
        return " · ".join(bits)


class TickerNotFoundError(SharadarError):
    """The ticker resolved to no data at Sharadar."""


def _api_key() -> str:
    """The environment's key, else Streamlit's. Never logged, never returned to a caller."""
    key = os.environ.get("SHARADAR_API_KEY")
    if not key:
        try:
            key = st.secrets.get("SHARADAR_API_KEY")
        except Exception:  # secrets.toml absent entirely
            key = None
    if not key:
        raise SharadarError(
            "SHARADAR_API_KEY is not set. Add it to .streamlit/secrets.toml, or set "
            "it in the environment, to use this page."
        )
    return str(key)


def _status_message(status: int) -> str:
    """A distinct, actionable message per documented status code."""
    return {
        400: "Nasdaq Data Link rejected the request as malformed (400). This is a "
             "bug in the request the app built, not something you can fix by "
             "retrying.",
        403: "Nasdaq Data Link refused access (403). Check SHARADAR_API_KEY in "
             ".streamlit/secrets.toml, and that your subscription covers this "
             "table.",
        404: "Nasdaq Data Link has no such table, or no data for this request (404).",
        429: "Nasdaq Data Link rate limit exceeded (429) and the request kept "
             "failing after several backed-off retries. Wait a minute and try again.",
    }.get(status, f"Nasdaq Data Link returned an unexpected HTTP {status}.")


def _get(table: str, params: dict, *, ticker: str | None = None) -> dict:
    """GETs one page of a datatable and returns parsed JSON.

    Retries 429 and 5xx with exponential backoff plus jitter. Raises
    :class:`SharadarError` on anything else — carrying no URL or params, so the
    key cannot leak into an error surface.
    """
    query = dict(params)
    query["api_key"] = _api_key()
    url = f"{BASE_URL}/{table}.json"

    last_status: int | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = requests.get(url, params=query, timeout=_TIMEOUT)
        except requests.exceptions.Timeout:
            last_status = None
            if attempt == _MAX_ATTEMPTS - 1:
                raise SharadarError(
                    "Nasdaq Data Link did not respond within the timeout.",
                    table=table,
                    ticker=ticker,
                ) from None
        except requests.exceptions.RequestException:
            # The underlying exception can echo the request URL, which carries
            # the key as a query parameter; do not chain it.
            raise SharadarError(
                "Could not reach Nasdaq Data Link. Check your network connection.",
                table=table,
                ticker=ticker,
            ) from None
        else:
            last_status = response.status_code
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    raise SharadarError(
                        "Nasdaq Data Link returned a response that is not valid JSON.",
                        status=200,
                        table=table,
                        ticker=ticker,
                    ) from None
            if response.status_code not in (429, 500, 502, 503, 504):
                raise SharadarError(
                    _status_message(response.status_code),
                    status=response.status_code,
                    table=table,
                    ticker=ticker,
                )

        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep((2**attempt) + random.uniform(0, 0.5))

    raise SharadarError(
        _status_message(last_status) if last_status else "Nasdaq Data Link request failed.",
        status=last_status,
        table=table,
        ticker=ticker,
    )


def parse_datatable(payload: dict) -> pd.DataFrame:
    """Zips a datatable's ``columns`` array against each row of ``data``.

    Column order is never assumed stable — Sharadar's docs say new indicators
    get appended over time, so the names come from this response and a
    reordered response parses identically. Values are left in their reported
    type here; numeric coercion happens per-table, because ``TICKERS`` is all
    text and coercing it would silently blank it.
    """
    table = (payload or {}).get("datatable") or {}
    names = [str(col.get("name", "")) for col in table.get("columns") or []]
    if not names:
        return pd.DataFrame()
    return pd.DataFrame(list(table.get("data") or []), columns=names)


def _request_all(table: str, params: dict, *, ticker: str | None = None) -> pd.DataFrame:
    """Fetches every page of a datatable and concatenates them.

    Nasdaq truncates responses at 10,000 rows and continues through an opaque
    ``meta.next_cursor_id``. A five-year daily price series is only ~1,260 rows,
    so this rarely loops — but a quarterly history request against a long-lived
    company will, and silently truncating it would drop the oldest years.
    """
    query = dict(params)
    query["qopts.per_page"] = _PAGE_SIZE

    pages: list[pd.DataFrame] = []
    cursor: str | None = None
    for _ in range(_MAX_PAGES):
        if cursor:
            query["qopts.cursor_id"] = cursor
        payload = _get(table, query, ticker=ticker)
        page = parse_datatable(payload)
        if not page.empty:
            pages.append(page)
        cursor = ((payload or {}).get("meta") or {}).get("next_cursor_id")
        if not cursor:
            break

    if not pages:
        return pd.DataFrame()
    return pd.concat(pages, ignore_index=True)
