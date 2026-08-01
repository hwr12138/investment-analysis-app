"""SimFin API v3 client.

Everything that talks to ``backend.simfin.com`` lives here: auth, retry,
caching, and the columnar parsing. Callers get pandas frames with resolved
column names and nothing else.

Three things about this API are not in its OpenAPI definition and were
confirmed against live responses. They are the reason this module exists rather
than a handful of inline ``requests.get`` calls:

1. **Three different response shapes.** ``/companies/list`` returns a plain
   array of objects; ``/companies/general/compact`` returns a single
   ``{columns, data}`` dict even for one ticker; the statement and price
   endpoints return an array with the columnar payload nested one level down.
2. **Rows are not chronological.** Asking for ``period=Q1,Q2,Q3,Q4`` returns
   rows grouped by fiscal period and then year, so the last row is the oldest
   fiscal period's most recent year — not the newest quarter. Every frame
   leaves here sorted by report date.
3. **``details=true`` changes the type of ``columns``** from ``list[str]`` to
   ``list[dict]`` keyed on ``caption``. Both shapes parse here.

The API key is read from ``st.secrets`` and never leaves this module: it is
sent as a header, never a query parameter, and no exception raised here carries
the request headers, the params, or the full URL.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import pandas as pd
import requests
import streamlit as st

BASE_URL = "https://backend.simfin.com/api/v3"

#: Literal prefix required by SimFin. Not a bearer token, not a bare key —
#: getting this wrong produces a 401 indistinguishable from an invalid key.
_AUTH_PREFIX = "api-key "

_TIMEOUT = 45
_MAX_ATTEMPTS = 4

#: Columns that are metadata rather than financial line items. Everything else
#: is coerced to numeric; these keep their reported type.
META_COLUMNS = frozenset(
    {
        "Fiscal Period",
        "Fiscal Year",
        "Report Date",
        "Publish Date",
        "Restated",
        "Source",
        "TTM",
        "Value Check",
        "Data Model",
        "Date",
    }
)

_STATEMENT_KEYS = ("PL", "BS", "CF", "DERIVED")


class SimFinError(Exception):
    """A SimFin request failed.

    Deliberately carries only the endpoint path, the HTTP status, and the
    ticker. The request URL and headers are never attached, so an API key
    cannot reach a traceback, a log line, or a Streamlit error block.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        endpoint: str | None = None,
        ticker: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.endpoint = endpoint
        self.ticker = ticker

    def __str__(self) -> str:
        bits = [self.message]
        if self.endpoint:
            bits.append(f"endpoint /{self.endpoint}")
        if self.ticker:
            bits.append(f"ticker {self.ticker}")
        if self.status:
            bits.append(f"HTTP {self.status}")
        return " · ".join(bits)


class TickerNotFoundError(SimFinError):
    """The ticker resolved to no data at SimFin."""


@dataclass(frozen=True)
class StatementSet:
    """Statements for one company, one period type, one ``asreported`` setting.

    ``asreported`` and ``ttm`` are recorded so the validation layer can prove a
    single model run never mixes restated with as-filed figures.
    """

    ticker: str
    simfin_id: int
    name: str
    currency: str
    template: str
    isin: str
    period: str
    ttm: bool
    asreported: bool
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    def frame(self, statement: str) -> pd.DataFrame:
        """Returns the frame for ``PL``/``BS``/``CF``/``DERIVED``, or an empty one."""
        return self.frames.get(statement.upper(), pd.DataFrame())

    @property
    def is_financial_template(self) -> bool:
        """True for the bank and insurance statement templates.

        This is the authoritative sector signal: ``sectorCode`` and
        ``sectorName`` on ``/companies/list`` are null for some companies,
        but every company that returns statements has a template.
        """
        return self.template.upper() in {"BANKS", "INSURANCES"}


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _api_key() -> str:
    """Reads the key from Streamlit secrets. Never logged, never returned to a caller."""
    try:
        key = st.secrets.get("SIMFIN_API_KEY")
    except Exception:  # secrets.toml absent entirely
        key = None
    if not key:
        raise SimFinError(
            "SIMFIN_API_KEY is not set. Add it to .streamlit/secrets.toml to use this page."
        )
    return str(key)


def _status_message(status: int) -> str:
    """A distinct, actionable message per documented status code."""
    return {
        400: "SimFin rejected the request as malformed (400). This is a bug in the "
             "request the app built, not something you can fix by retrying.",
        401: "SimFin rejected the API key (401). Check SIMFIN_API_KEY in "
             ".streamlit/secrets.toml — note the value is sent with a literal "
             "'api-key ' prefix, not as a bearer token.",
        403: "SimFin refused access (403). This endpoint or ticker may not be "
             "included in your subscription tier.",
        404: "SimFin has no such endpoint or no data for this request (404).",
        429: "SimFin rate limit exceeded (429) and the request kept failing after "
             "several backed-off retries. Wait a minute and try again.",
    }.get(status, f"SimFin returned an unexpected HTTP {status}.")


def _request(endpoint: str, params: dict, *, ticker: str | None = None):
    """GETs a SimFin endpoint and returns parsed JSON.

    Retries 429 and 5xx with exponential backoff plus jitter. Raises
    :class:`SimFinError` on anything else — carrying no URL, params, or headers,
    so the key cannot leak into an error surface.
    """
    key = _api_key()
    headers = {"Authorization": f"{_AUTH_PREFIX}{key}", "Accept": "application/json"}
    url = f"{BASE_URL}/{endpoint}"

    last_status: int | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT)
        except requests.exceptions.Timeout:
            last_status = None
            if attempt == _MAX_ATTEMPTS - 1:
                raise SimFinError(
                    "SimFin did not respond within the timeout.",
                    endpoint=endpoint,
                    ticker=ticker,
                ) from None
        except requests.exceptions.RequestException:
            # The underlying exception can echo the request; do not chain it.
            raise SimFinError(
                "Could not reach SimFin. Check your network connection.",
                endpoint=endpoint,
                ticker=ticker,
            ) from None
        else:
            last_status = response.status_code
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    raise SimFinError(
                        "SimFin returned a response that is not valid JSON.",
                        status=200,
                        endpoint=endpoint,
                        ticker=ticker,
                    ) from None
            if response.status_code not in (429, 500, 502, 503, 504):
                raise SimFinError(
                    _status_message(response.status_code),
                    status=response.status_code,
                    endpoint=endpoint,
                    ticker=ticker,
                )

        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep((2**attempt) + random.uniform(0, 0.5))

    raise SimFinError(
        _status_message(last_status) if last_status else "SimFin request failed.",
        status=last_status,
        endpoint=endpoint,
        ticker=ticker,
    )


# ---------------------------------------------------------------------------
# Columnar parsing
# ---------------------------------------------------------------------------


def column_names(columns: list) -> list[str]:
    """Normalises a ``columns`` array to plain names.

    ``details=true`` turns each entry into a dict of column metadata keyed on
    ``caption``; without it each entry is already a string. Both are accepted so
    the schema map can be built from a detailed call and reused against compact
    ones.
    """
    names: list[str] = []
    for col in columns or []:
        if isinstance(col, dict):
            caption = col.get("caption")
            names.append(str(caption) if caption is not None else "")
        else:
            names.append(str(col))
    return names


def parse_columnar(columns: list, data: list, *, numeric: bool = True) -> pd.DataFrame:
    """Zips a ``columns`` array against each row of ``data``.

    Column order is never assumed stable across calls or companies — the names
    come from this response, so a reordered response parses identically.

    ``numeric`` controls whether non-metadata columns are coerced to numbers.
    Statement and price payloads are line items, so a value that will not
    convert becomes ``NaN`` — read downstream as "not reported" and excluded,
    never as a zero balance. The company-metadata endpoint returns names,
    tickers and descriptions in the same columnar shape, so it passes
    ``numeric=False``; coercing those would silently blank them.
    """
    names = column_names(columns)
    if not names:
        return pd.DataFrame()

    frame = pd.DataFrame(list(data or []), columns=names)
    if frame.empty:
        return frame

    if numeric:
        for name in frame.columns:
            if name in META_COLUMNS:
                continue
            frame[name] = pd.to_numeric(frame[name], errors="coerce")

    for date_col in ("Report Date", "Publish Date", "Date"):
        if date_col in frame.columns:
            frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")

    if "Fiscal Year" in frame.columns:
        frame["Fiscal Year"] = pd.to_numeric(frame["Fiscal Year"], errors="coerce").astype(
            "Int64"
        )
    return frame


def _order_statement_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Sorts chronologically and drops superseded restatements.

    SimFin returns rows grouped by fiscal period then year. A restated year can
    also appear twice; the row with the later publish date is the newer filing.
    """
    if frame.empty or "Report Date" not in frame.columns:
        return frame

    ordered = frame.sort_values(
        [c for c in ("Report Date", "Publish Date") if c in frame.columns]
    )
    if {"Fiscal Year", "Fiscal Period"}.issubset(ordered.columns):
        ordered = ordered.drop_duplicates(
            subset=["Fiscal Year", "Fiscal Period"], keep="last"
        )
    return ordered.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _fetch_company_list() -> pd.DataFrame:
    payload = _request("companies/list", {})
    if not isinstance(payload, list):
        raise SimFinError(
            "SimFin's company list came back in an unexpected shape.",
            endpoint="companies/list",
        )
    frame = pd.DataFrame(payload)
    if frame.empty or "ticker" not in frame.columns:
        raise SimFinError("SimFin returned an empty company list.", endpoint="companies/list")
    frame["ticker"] = frame["ticker"].astype("string").str.upper()
    return frame


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 7)
def fetch_company_list() -> pd.DataFrame:
    """Every company SimFin covers. Cached for a week; it barely moves."""
    return _fetch_company_list()


def lookup_company(ticker: str) -> dict | None:
    """Resolves a ticker against ``/companies/list``. ``None`` if not listed.

    ``sectorCode``/``sectorName`` are null for some companies, so callers must
    treat a missing sector as unknown rather than as a rejection.
    """
    listing = fetch_company_list()
    hits = listing[listing["ticker"] == ticker.strip().upper()]
    if hits.empty:
        return None
    record = hits.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in record.items()}


def _fetch_general(ticker: str) -> dict:
    payload = _request("companies/general/compact", {"ticker": ticker}, ticker=ticker)
    # This endpoint returns a bare {columns, data} dict, not an array.
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    frame = parse_columnar(
        payload.get("columns", []), payload.get("data", []), numeric=False
    )
    if frame.empty:
        raise TickerNotFoundError(
            f"SimFin returned no company record for {ticker}.",
            endpoint="companies/general/compact",
            ticker=ticker,
        )
    record = frame.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in record.items()}


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 3)
def fetch_general(ticker: str) -> dict:
    """Company metadata. ``endFy`` is the fiscal year end *month* (AAPL: 9)."""
    return _fetch_general(ticker)


def _fetch_statements(
    ticker: str,
    statements: str = "PL,BS,CF,DERIVED",
    period: str = "FY",
    ttm: bool = False,
    asreported: bool = False,
    details: bool = False,
) -> StatementSet:
    params = {
        "ticker": ticker,
        "statements": statements,
        "period": period,
    }
    if ttm:
        params["ttm"] = "true"
    if asreported:
        params["asreported"] = "true"
    if details:
        params["details"] = "true"

    payload = _request("companies/statements/compact", params, ticker=ticker)
    if not isinstance(payload, list) or not payload:
        raise TickerNotFoundError(
            f"SimFin returned no statements for {ticker}.",
            endpoint="companies/statements/compact",
            ticker=ticker,
        )

    company = payload[0]
    frames: dict[str, pd.DataFrame] = {}
    for statement in company.get("statements") or []:
        name = str(statement.get("statement", "")).upper()
        if name not in _STATEMENT_KEYS:
            continue
        frames[name] = _order_statement_rows(
            parse_columnar(statement.get("columns", []), statement.get("data", []))
        )

    return StatementSet(
        ticker=str(company.get("ticker") or ticker).upper(),
        simfin_id=int(company.get("id") or 0),
        name=str(company.get("name") or ""),
        currency=str(company.get("currency") or ""),
        template=str(company.get("template") or ""),
        isin=str(company.get("isin") or ""),
        period=period,
        ttm=ttm,
        asreported=asreported,
        frames=frames,
    )


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 3)
def fetch_statements(
    ticker: str,
    statements: str = "PL,BS,CF,DERIVED",
    period: str = "FY",
    ttm: bool = False,
    asreported: bool = False,
    details: bool = False,
) -> StatementSet:
    """Statements for one company, sorted chronologically and de-restated."""
    return _fetch_statements(ticker, statements, period, ttm, asreported, details)


def _fetch_prices(ticker: str, start: str | None = None, ratios: bool = True) -> pd.DataFrame:
    params: dict = {"ticker": ticker}
    if ratios:
        params["ratios"] = "true"
    if start:
        params["start"] = start

    payload = _request("companies/prices/compact", params, ticker=ticker)
    if not isinstance(payload, list) or not payload:
        raise TickerNotFoundError(
            f"SimFin returned no price history for {ticker}.",
            endpoint="companies/prices/compact",
            ticker=ticker,
        )
    company = payload[0]
    frame = parse_columnar(company.get("columns", []), company.get("data", []))
    if frame.empty or "Date" not in frame.columns:
        raise TickerNotFoundError(
            f"SimFin returned an empty price series for {ticker}.",
            endpoint="companies/prices/compact",
            ticker=ticker,
        )
    return frame.sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_prices(ticker: str, start: str | None = None, ratios: bool = True) -> pd.DataFrame:
    """Daily prices.

    ``Adjusted Closing Price`` is the split/dividend-adjusted series and is what
    the beta regression uses. ``Last Closing Price`` is unadjusted and is only
    correct for the current quote and market cap.
    """
    return _fetch_prices(ticker, start, ratios)


def _fetch_common_shares(ticker: str) -> pd.DataFrame:
    payload = _request(
        "companies/common-shares-outstanding", {"ticker": ticker}, ticker=ticker
    )
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=["endDate", "value"])
    frame = pd.DataFrame(payload)
    frame["endDate"] = pd.to_datetime(frame["endDate"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["endDate", "value"]).sort_values("endDate").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_common_shares(ticker: str) -> pd.DataFrame:
    """Point-in-time common shares outstanding, dated by filing cover date."""
    return _fetch_common_shares(ticker)


def _fetch_weighted_shares(
    ticker: str, period: str = "FY", ttm: bool = False
) -> pd.DataFrame:
    params: dict = {"ticker": ticker, "period": period}
    if ttm:
        params["ttm"] = "true"
    payload = _request("companies/weighted-shares-outstanding", params, ticker=ticker)
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=["endDate", "fyear", "period", "basic", "diluted"])
    frame = pd.DataFrame(payload)
    frame["endDate"] = pd.to_datetime(frame["endDate"], errors="coerce")
    for col in ("basic", "diluted"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "fyear" in frame.columns:
        frame["fyear"] = pd.to_numeric(frame["fyear"], errors="coerce").astype("Int64")
    return frame.sort_values("endDate").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_weighted_shares(ticker: str, period: str = "FY", ttm: bool = False) -> pd.DataFrame:
    """Weighted-average basic and diluted share counts per fiscal period.

    Join to statements on ``(fyear, period)``, never on date: this endpoint
    reports the true fiscal period end (2025-09-27 for AAPL FY2025) while the
    statement endpoint normalises to month end (2025-09-30).
    """
    return _fetch_weighted_shares(ticker, period, ttm)


def _fetch_filings(ticker: str) -> pd.DataFrame:
    payload = _request("filings/by-company", {"ticker": ticker}, ticker=ticker)
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=["fyear", "period", "filingDate", "filingType", "sourceLink"])
    frame = pd.DataFrame(payload)
    keep = [c for c in ("fyear", "period", "filingDate", "filingType", "sourceLink") if c in frame]
    frame = frame[keep].copy()
    if "filingDate" in frame.columns:
        frame["filingDate"] = pd.to_datetime(frame["filingDate"], errors="coerce")
    if "fyear" in frame.columns:
        frame["fyear"] = pd.to_numeric(frame["fyear"], errors="coerce").astype("Int64")
    return frame.sort_values("filingDate").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 3)
def fetch_filings(ticker: str) -> pd.DataFrame:
    """Filing index, used to link each historical year to its source 10-K."""
    return _fetch_filings(ticker)
