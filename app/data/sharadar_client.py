"""Nasdaq Data Link (Sharadar) client.

Everything that talks to ``data.nasdaq.com`` lives here: auth, retry,
pagination, caching, and the datatable parsing. Callers get pandas frames with
resolved column names and nothing else.

Seven Sharadar tables are used:

* ``SF1``     — Core US Fundamentals. One wide row of ~110 indicators per
  (ticker, dimension, period). Partitioned here into the ``PL``/``BS``/``CF``/
  ``DERIVED`` frames the concept resolver expects.
* ``SEP``     — End-of-day equity prices.
* ``SFP``     — End-of-day fund (ETF/CEF) prices. Same columns as SEP.
* ``DAILY``   — Daily valuation ratios (market cap, EV, EV/EBITDA, PE, PB, PS).
* ``TICKERS`` — Reference data: sector, SIC code, currency, EDGAR link, and
  which price table a ticker lives in.
* ``ACTIONS`` — Corporate actions: dividends (the date *is* the ex-dividend
  date), splits, listings, ticker changes.
* ``SF2``     — Core US Insiders. One row per line item on an SEC Form 3/4/5,
  from 2008. The only table here whose rows are *people* rather than periods,
  and the only one that is event-driven rather than periodic.

Most fetches here are keyed on a ticker. The three in the *cross-sectional
universe* section at the foot of the module are keyed on a date instead, and
return every listed company at once — see that section for why that is cheap.

Six things about this API are not obvious from its docs and were confirmed
against live AAPL and JPM responses. They are the reason this module exists
rather than a handful of inline ``requests.get`` calls:

1. **Sign conventions differ from the engine's.** SF1 signs ``cor``, ``opex``,
   ``sgna``, ``rnd``, ``taxexp`` and ``intexp`` as positive magnitudes. The
   engine's contract (see ``financials`` module docstring) is costs-negative.
   :data:`SIGN_FLIPPED` is negated on ingest so nothing downstream has to know.
   ``capex`` is *already* negative and is deliberately not in that set.
2. **``DAILY`` reports market cap and EV in millions**, while ``SF1`` reports
   the same quantities in raw units. :data:`_DAILY_MILLIONS` is scaled here; a
   missed conversion passes every sanity check by a factor of a million.
3. **Zero means "absent from the filing"** for 41 indicators — Sharadar
   documents "where this item is not contained on the company consolidated
   financial statements … the value of 0 is used". The engine's rule is that
   missing stays missing, so the zeros pass through as reported but are
   enumerated in :data:`ZERO_FILLED_INDICATORS` for the validation layer to
   flag. See ``validation.check_zero_filled_indicators``.
4. **Rows are unsorted by contract** and the API truncates at 10,000 rows,
   continuing through an opaque ``meta.next_cursor_id``. Every frame leaves
   here paginated to completion and sorted.
5. **A non-trading date is not an error.** ``DAILY`` answers a weekend or a
   holiday with zero rows rather than the previous session, so any date passed
   to :func:`fetch_universe_daily` has to be one the market was open on.
6. **SF2 signs the shares, not the value.** ``transactionshares`` is signed —
   negative on a disposal — while ``transactionvalue`` is the *unsigned
   magnitude* of the same trade: 40,334 shares sold at $115.95 carries a
   ``transactionvalue`` of 4,676,727 with no minus, and is null on 48% of rows
   besides. Summing that column therefore adds every sale to every purchase and
   misses half the trades. SF2 is deliberately **not** in :data:`SIGN_FLIPPED`:
   that set negates indicators Sharadar publishes as positive magnitudes against
   a costs-negative contract, and ``transactionshares`` is already correct —
   negating it would invert every trade. The sign SF2 needs lives in a different
   column, which is a join rather than a negation, so no column-wise transform
   fixes it here. The dollar flow is ``transactionshares ×
   transactionpricepershare``, computed in ``logic/research/insiders`` and
   nowhere else.

The API key is read from ``st.secrets`` and never leaves this module. Nasdaq
only accepts it as a query parameter, so no exception raised here carries the
params or the full URL — only the table, the ticker, and the status.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

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

_STATEMENT_KEYS = ("PL", "BS", "CF", "DERIVED")

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
        "Calendar Date",
        "Dimension",
    }
)

#: SF1 indicators signed as positive magnitudes that the engine expects
#: negative. ``capex`` is NOT here — Sharadar already signs it negative
#: (AAPL FY2025: -12,715M, and their own ``fcf`` = ``ncfo + capex``).
SIGN_FLIPPED = frozenset({"cor", "opex", "sgna", "rnd", "taxexp", "intexp"})

#: Indicators Sharadar documents as "where this item is not contained on the
#: company consolidated financial statements … the value of 0 is used". A zero
#: in any of these may mean *not disclosed* rather than *nil*, which matters
#: concretely: AAPL's ``intexp`` is 0, so the FCFF interest add-back vanishes
#: and the implied cost of debt reads 0%.
ZERO_FILLED_INDICATORS = frozenset(
    {
        "accoci", "capex", "cashneq", "cor", "debt", "debtc", "debtnc",
        "deferredrev", "depamor", "deposits", "divyield", "dps", "intangibles",
        "intexp", "inventory", "investments", "investmentsc", "investmentsnc",
        "ncfbus", "ncfcommon", "ncfdebt", "ncfdiv", "ncff", "ncfi", "ncfinv",
        "ncfx", "netincdis", "netincnci", "payables", "ppnenet", "prefdivis",
        "receivables", "revenue", "rnd", "sgna", "taxassets", "taxexp",
        "taxliabilities",
    }
)

#: DAILY reports these in millions; SF1 reports the same quantities raw.
_DAILY_MILLIONS = ("marketcap", "ev")

#: Every numeric column DAILY carries. Coerced together so the per-ticker and
#: universe-wide fetches cannot drift apart on which ones are numbers.
_DAILY_NUMERIC = ("ev", "evebit", "evebitda", "marketcap", "pb", "pe", "ps")

#: SIC codes 6000-6999 are Finance, Insurance and Real Estate. Banks and
#: insurers do not have a meaningful operating-cash-flow-less-capex FCFF, so
#: they are rejected before anything is computed.
FINANCIAL_SIC_RANGE = (6000, 6999)

# ---------------------------------------------------------------------------
# SF1 statement partition
# ---------------------------------------------------------------------------

#: ``(indicator, caption, statement)`` for every SF1 indicator the app surfaces,
#: in display order. Captions are Sharadar's own ``title`` values from
#: ``SHARADAR/INDICATORS`` (trailing whitespace stripped), so the raw figures tab
#: reads as the publisher names things and ``schema.CONCEPTS`` can alias against
#: human-readable strings.
#:
#: The partition follows the ``[Income Statement]``/``[Balance Sheet]``/
#: ``[Cash Flow Statement]``/``[Metrics]`` prefix in each indicator's published
#: description, with one deliberate move: ``ebt`` is published under Metrics but
#: is the income statement's pretax income, which is where the model reads it.
#: Frozen here rather than fetched at runtime — the assignment is stable and a
#: live dependency on the dictionary endpoint would be a needless failure mode.
SF1_LAYOUT: tuple[tuple[str, str, str], ...] = (
    # --- Income statement -------------------------------------------------
    ("revenue", "Revenues", "PL"),
    ("cor", "Cost of Revenue", "PL"),
    ("gp", "Gross Profit", "PL"),
    ("sgna", "Selling General and Administrative Expense", "PL"),
    ("rnd", "Research and Development Expense", "PL"),
    ("opex", "Operating Expenses", "PL"),
    ("opinc", "Operating Income", "PL"),
    ("ebit", "Earning Before Interest & Taxes (EBIT)", "PL"),
    ("intexp", "Interest Expense", "PL"),
    ("ebt", "Earnings before Tax", "PL"),
    ("taxexp", "Income Tax Expense", "PL"),
    ("consolinc", "Consolidated Income", "PL"),
    ("netincnci", "Net Income to Non-Controlling Interests", "PL"),
    ("netinc", "Net Income", "PL"),
    ("prefdivis", "Preferred Dividends Income Statement Impact", "PL"),
    ("netinccmn", "Net Income Common Stock", "PL"),
    ("netincdis", "Net Loss Income from Discontinued Operations", "PL"),
    ("eps", "Earnings per Basic Share", "PL"),
    ("epsdil", "Earnings per Diluted Share", "PL"),
    ("shareswa", "Weighted Average Shares", "PL"),
    ("shareswadil", "Weighted Average Shares Diluted", "PL"),
    ("dps", "Dividends per Basic Common Share", "PL"),
    # --- Balance sheet ----------------------------------------------------
    ("cashneq", "Cash and Equivalents", "BS"),
    ("investmentsc", "Investments Current", "BS"),
    ("receivables", "Trade and Non-Trade Receivables", "BS"),
    ("inventory", "Inventory", "BS"),
    ("assetsc", "Current Assets", "BS"),
    ("investmentsnc", "Investments Non-Current", "BS"),
    ("ppnenet", "Property Plant & Equipment Net", "BS"),
    ("intangibles", "Goodwill and Intangible Assets", "BS"),
    ("taxassets", "Tax Assets", "BS"),
    ("investments", "Investments", "BS"),
    ("assetsnc", "Assets Non-Current", "BS"),
    ("assets", "Total Assets", "BS"),
    ("payables", "Trade and Non-Trade Payables", "BS"),
    ("deferredrev", "Deferred Revenue", "BS"),
    ("debtc", "Debt Current", "BS"),
    ("liabilitiesc", "Current Liabilities", "BS"),
    ("debtnc", "Debt Non-Current", "BS"),
    ("deposits", "Deposit Liabilities", "BS"),
    ("taxliabilities", "Tax Liabilities", "BS"),
    ("liabilitiesnc", "Liabilities Non-Current", "BS"),
    ("liabilities", "Total Liabilities", "BS"),
    ("debt", "Total Debt", "BS"),
    ("retearn", "Accumulated Retained Earnings (Deficit)", "BS"),
    ("accoci", "Accumulated Other Comprehensive Income", "BS"),
    ("equity", "Shareholders Equity Attributable to Parent", "BS"),
    # --- Cash flow --------------------------------------------------------
    ("depamor", "Depreciation Amortization & Accretion", "CF"),
    ("sbcomp", "Share Based Compensation", "CF"),
    ("ncfo", "Net Cash Flow from Operations", "CF"),
    ("capex", "Capital Expenditure", "CF"),
    ("ncfbus", "Net Cash Flow - Business Acquisitions and Disposals", "CF"),
    ("ncfinv", "Net Cash Flow - Investment Acquisitions and Disposals", "CF"),
    ("ncfi", "Net Cash Flow from Investing", "CF"),
    ("ncfdebt", "Issuance (Repayment) of Debt Securities", "CF"),
    ("ncfcommon", "Issuance (Purchase) of Equity Shares", "CF"),
    ("ncfdiv", "Payment of Dividends & Other Cash Distributions", "CF"),
    ("ncff", "Net Cash Flow from Financing", "CF"),
    ("ncfx", "Effect of Exchange Rate Changes on Cash", "CF"),
    ("ncf", "Net Cash Flow / Change in Cash & Cash Equivalents", "CF"),
    # --- Derived: metrics and entity-level figures ------------------------
    ("ebitda", "Earnings Before Interest Taxes & Depreciation Amortization (EBITDA)", "DERIVED"),
    ("fcf", "Free Cash Flow", "DERIVED"),
    ("workingcapital", "Working Capital", "DERIVED"),
    ("invcap", "Invested Capital", "DERIVED"),
    ("tangibles", "Tangible Asset Value", "DERIVED"),
    ("sharesbas", "Shares (Basic)", "DERIVED"),
    ("sharefactor", "Share Factor", "DERIVED"),
    ("price", "Share Price (Adjusted Close)", "DERIVED"),
    ("marketcap", "Market Capitalization", "DERIVED"),
    ("ev", "Enterprise Value", "DERIVED"),
    ("evebit", "Enterprise Value over EBIT", "DERIVED"),
    ("evebitda", "Enterprise Value over EBITDA", "DERIVED"),
    ("pe1", "Price to Earnings Ratio", "DERIVED"),
    ("pb", "Price to Book Value", "DERIVED"),
    ("ps1", "Price to Sales Ratio", "DERIVED"),
    ("grossmargin", "Gross Margin", "DERIVED"),
    ("ebitdamargin", "EBITDA Margin", "DERIVED"),
    ("netmargin", "Profit Margin", "DERIVED"),
    ("ros", "Return on Sales", "DERIVED"),
    ("roe", "Return on Average Equity", "DERIVED"),
    ("roa", "Return on Average Assets", "DERIVED"),
    ("roic", "Return on Invested Capital", "DERIVED"),
    ("currentratio", "Current Ratio", "DERIVED"),
    ("de", "Debt to Equity Ratio", "DERIVED"),
    ("bvps", "Book Value per Share", "DERIVED"),
    ("fcfps", "Free Cash Flow per Share", "DERIVED"),
    ("sps", "Sales per Share", "DERIVED"),
    ("divyield", "Dividend Yield", "DERIVED"),
    ("payoutratio", "Payout Ratio", "DERIVED"),
    ("assetturnover", "Asset Turnover", "DERIVED"),
    ("fxusd", "Foreign Currency to USD Exchange Rate", "DERIVED"),
)

#: Caption for the non-controlling interest column derived below. SF1 has no
#: balance-sheet NCI line, but its ``equity`` is explicitly "attributable to
#: parent", so the residual recovers it.
MINORITY_INTEREST_CAPTION = "Minority Interest (derived)"

SF1_CAPTIONS: dict[str, str] = {ind: cap for ind, cap, _ in SF1_LAYOUT}

#: Reverse of :data:`SF1_CAPTIONS`, for turning a resolved column name back into
#: the Sharadar indicator it came from.
CAPTION_TO_INDICATOR: dict[str, str] = {cap: ind for ind, cap, _ in SF1_LAYOUT}


def is_zero_filled(caption: str) -> bool:
    """Whether a resolved column is one Sharadar writes 0 into when absent.

    Lets the validation layer ask the question without importing the indicator
    vocabulary. See :data:`ZERO_FILLED_INDICATORS`.
    """
    indicator = CAPTION_TO_INDICATOR.get(caption)
    return indicator is not None and indicator in ZERO_FILLED_INDICATORS


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


@dataclass(frozen=True)
class StatementSet:
    """Statements for one company at one SF1 dimension.

    ``dimension`` and the derived ``asreported``/``ttm`` flags are recorded so
    the validation layer can prove a single model run never mixes restated with
    as-filed figures.
    """

    ticker: str
    permaticker: int
    name: str
    currency: str
    template: str
    dimension: str
    period: str
    ttm: bool
    asreported: bool
    sector: str = ""
    industry: str = ""
    sic_code: int | None = None
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    def frame(self, statement: str) -> pd.DataFrame:
        """Returns the frame for ``PL``/``BS``/``CF``/``DERIVED``, or an empty one."""
        return self.frames.get(statement.upper(), pd.DataFrame())

    @property
    def is_financial_template(self) -> bool:
        """True for banks, insurers, and the rest of SIC division H.

        SIC code is authoritative where present: Sharadar's ``sector`` field is
        a vendor classification that can be null, but every company in SF1 that
        files with the SEC has a SIC code. Both are checked.
        """
        low, high = FINANCIAL_SIC_RANGE
        if self.sic_code is not None and low <= int(self.sic_code) <= high:
            return True
        return self.template.strip().lower() in {
            "financial services",
            "finance insurance and real estate",
        }


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def _api_key() -> str:
    """Reads the key from Streamlit secrets. Never logged, never returned to a caller."""
    try:
        key = st.secrets.get("SHARADAR_API_KEY")
    except Exception:  # secrets.toml absent entirely
        key = None
    if not key:
        raise SharadarError(
            "SHARADAR_API_KEY is not set. Add it to .streamlit/secrets.toml to use "
            "this page."
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


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------


def _fetch_ticker_meta(ticker: str) -> dict:
    frame = _request_all("TICKERS", {"table": "SF1", "ticker": ticker}, ticker=ticker)
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
    frame = _request_all("TICKERS", {"ticker": ",".join(tickers)})
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


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------


def dimension_for(*, annual: bool, ttm: bool, asreported: bool) -> str:
    """Builds the SF1 ``dimension`` code.

    ``AR`` excludes restatements and time-indexes to the SEC filing date; ``MR``
    includes them and time-indexes to the report period. The ``asreported``
    toggle maps onto that pair exactly.
    """
    base = "AR" if asreported else "MR"
    if ttm:
        return f"{base}T"
    return f"{base}Y" if annual else f"{base}Q"


def _split_fiscal_period(value: object) -> tuple[int | None, str | None]:
    """``"2025-FY"`` → ``(2025, "FY")``; ``"2026-Q3"`` → ``(2026, "Q3")``."""
    text = str(value or "").strip()
    if "-" not in text:
        return None, None
    year, _, period = text.partition("-")
    try:
        return int(year), period.upper()
    except ValueError:
        return None, period.upper() or None


def _sf1_frames(
    rows: pd.DataFrame, *, source_link: str, restated: bool
) -> dict[str, pd.DataFrame]:
    """Partitions the wide SF1 result into the four statement frames.

    Every frame carries the same metadata columns, because every frame comes
    from the same SF1 row. That is why an annual request never splits a fiscal
    year across two half-populated rows.
    """
    numeric = rows.copy()
    for indicator in SF1_CAPTIONS:
        if indicator in numeric.columns:
            numeric[indicator] = pd.to_numeric(numeric[indicator], errors="coerce")

    for indicator in SIGN_FLIPPED:
        if indicator in numeric.columns:
            numeric[indicator] = -numeric[indicator]

    parsed = [_split_fiscal_period(v) for v in numeric.get("fiscalperiod", [])]
    meta = pd.DataFrame(
        {
            "Fiscal Year": pd.array([p[0] for p in parsed], dtype="Int64"),
            "Fiscal Period": [p[1] for p in parsed],
            "Report Date": pd.to_datetime(numeric.get("reportperiod"), errors="coerce"),
            "Publish Date": pd.to_datetime(numeric.get("datekey"), errors="coerce"),
            "Calendar Date": pd.to_datetime(numeric.get("calendardate"), errors="coerce"),
            "Dimension": numeric.get("dimension"),
            "Restated": "Yes" if restated else "No",
            "Source": source_link or None,
            "Value Check": None,
        },
        index=numeric.index,
    )

    # SF1 has no balance-sheet non-controlling interest line, but its `equity`
    # is explicitly "attributable to parent", so the residual recovers it.
    derived_nci: pd.Series | None = None
    if {"assets", "liabilities", "equity"}.issubset(numeric.columns):
        derived_nci = numeric["assets"] - numeric["liabilities"] - numeric["equity"]

    frames: dict[str, pd.DataFrame] = {}
    for statement in _STATEMENT_KEYS:
        captions = {
            indicator: caption
            for indicator, caption, target in SF1_LAYOUT
            if target == statement and indicator in numeric.columns
        }
        body = numeric[list(captions)].rename(columns=captions)
        if statement == "BS" and derived_nci is not None:
            body = body.assign(**{MINORITY_INTEREST_CAPTION: derived_nci})
        frames[statement] = pd.concat([meta, body], axis=1).reset_index(drop=True)

    return frames


def _order_statement_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Sorts chronologically and drops superseded filings.

    Sharadar returns rows unsorted by contract. The ``AR`` dimensions can also
    emit more than one row for a fiscal period when a company files twice in a
    quarter; the row with the later filing date supersedes.
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


def _fetch_statements(
    ticker: str,
    *,
    annual: bool = True,
    ttm: bool = False,
    asreported: bool = False,
) -> StatementSet:
    dimension = dimension_for(annual=annual, ttm=ttm, asreported=asreported)
    meta = _fetch_ticker_meta(ticker)

    rows = _request_all(
        "SF1", {"ticker": ticker, "dimension": dimension}, ticker=ticker
    )
    if rows.empty:
        raise TickerNotFoundError(
            f"Sharadar returned no {dimension} fundamentals for {ticker}.",
            table="SF1",
            ticker=ticker,
        )

    frames = {
        name: _order_statement_rows(frame)
        for name, frame in _sf1_frames(
            rows,
            source_link=str(meta.get("secfilings") or ""),
            restated=not asreported,
        ).items()
    }

    sic_code = meta.get("siccode")
    return StatementSet(
        ticker=ticker.strip().upper(),
        permaticker=int(meta.get("permaticker") or 0),
        name=str(meta.get("name") or ""),
        currency=str(meta.get("currency") or ""),
        # `template` is the generic classification string the schema resolver
        # and the unsupported-sector error report back to the user.
        template=str(meta.get("sector") or meta.get("sicsector") or ""),
        dimension=dimension,
        period="FY" if annual and not ttm else "Q1,Q2,Q3,Q4",
        ttm=ttm,
        asreported=asreported,
        sector=str(meta.get("sector") or ""),
        industry=str(meta.get("industry") or ""),
        sic_code=int(sic_code) if sic_code is not None and not pd.isna(sic_code) else None,
        frames=frames,
    )


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24 * 3)
def _cached_statements(
    ticker: str, annual: bool, ttm: bool, asreported: bool
) -> StatementSet:
    """The cache entry. Positional only — see :func:`fetch_statements`."""
    return _fetch_statements(ticker, annual=annual, ttm=ttm, asreported=asreported)


def fetch_statements(
    ticker: str,
    *,
    annual: bool = True,
    ttm: bool = False,
    asreported: bool = False,
) -> StatementSet:
    """Fundamentals for one company, sorted chronologically and de-duplicated.

    ``asreported=True`` selects Sharadar's ``AR`` dimensions (excludes
    restatements, indexed to the SEC filing date); the default selects ``MR``
    (includes restatements, indexed to the report period).

    The cache sits on ``_cached_statements`` rather than here, and this wrapper
    exists only to fill the defaults in before the key is built. ``st.cache_data``
    keys on the arguments *as written at the call site*, without applying
    defaults: ``fetch_statements(t, annual=True)`` and
    ``fetch_statements(t, annual=True, ttm=False)`` are the same request but
    hashed differently, and each would fetch the same ``MRY`` rows separately.
    That is not hypothetical — the research page passed all three keywords and
    the DCF omitted ``ttm``, so one search fetched ``MRY`` and ``MRQ`` twice
    each. Forwarding positionally makes every call form collapse onto one key.
    """
    return _cached_statements(ticker, annual, ttm, asreported)


# ---------------------------------------------------------------------------
# Views over a StatementSet
# ---------------------------------------------------------------------------
#
# Sharadar carries share counts and the filing date on the SF1 row itself, so
# these are projections of a frame already in hand rather than network calls.
# They keep the column names their consumers expect, and they keep the knowledge
# of which Sharadar indicator means what inside this module.


def common_shares_frame(statements: StatementSet) -> pd.DataFrame:
    """Point-in-time shares outstanding, as ``endDate``/``value``.

    Sharadar's ``sharesbas`` is the count stated on the cover of the 10-K or
    10-Q, adjusted for splits. ``sharefactor`` is applied because it is what
    Sharadar itself multiplies through when computing market cap — it corrects
    for ADRs representing a fraction or multiple of an underlying share, and
    for multi-class structures.
    """
    empty = pd.DataFrame(columns=["endDate", "value"])
    frame = statements.frame("DERIVED")
    caption = SF1_CAPTIONS["sharesbas"]
    if frame.empty or caption not in frame.columns:
        return empty

    factor = frame.get(SF1_CAPTIONS["sharefactor"])
    shares = pd.to_numeric(frame[caption], errors="coerce")
    if factor is not None:
        shares = shares * pd.to_numeric(factor, errors="coerce").fillna(1.0)

    out = pd.DataFrame({"endDate": frame["Report Date"], "value": shares})
    return out.dropna(subset=["endDate", "value"]).sort_values("endDate").reset_index(
        drop=True
    )


def weighted_shares_frame(statements: StatementSet) -> pd.DataFrame:
    """Weighted-average basic and diluted counts per fiscal period."""
    columns = ["endDate", "fyear", "period", "basic", "diluted"]
    frame = statements.frame("PL")
    basic_col = SF1_CAPTIONS["shareswa"]
    diluted_col = SF1_CAPTIONS["shareswadil"]
    if frame.empty or basic_col not in frame.columns:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(
        {
            "endDate": frame["Report Date"],
            "fyear": frame["Fiscal Year"],
            "period": frame["Fiscal Period"],
            "basic": pd.to_numeric(frame[basic_col], errors="coerce"),
            "diluted": pd.to_numeric(frame.get(diluted_col), errors="coerce"),
        }
    )
    return out.dropna(subset=["endDate"]).sort_values("endDate").reset_index(drop=True)


def filings_frame(statements: StatementSet) -> pd.DataFrame:
    """The filing behind each period, for the provenance panel.

    Sharadar has no per-filing URL, so ``sourceLink`` is the company's EDGAR
    filing index — enough to reach the document in one click, but not a direct
    link to it. Under the ``AR`` dimensions ``datekey`` is the SEC filing date;
    under ``MR`` it is the report period, so the filing date is only meaningful
    for as-reported runs and is labelled as such upstream.
    """
    columns = ["fyear", "period", "filingDate", "filingType", "sourceLink"]
    frame = statements.frame("PL")
    if frame.empty or "Fiscal Year" not in frame.columns:
        return pd.DataFrame(columns=columns)

    periods = frame["Fiscal Period"]
    out = pd.DataFrame(
        {
            "fyear": frame["Fiscal Year"],
            "period": periods,
            "filingDate": frame["Publish Date"],
            "filingType": ["10-K" if p == "FY" else "10-Q" for p in periods],
            "sourceLink": frame["Source"],
        }
    )
    return out.dropna(subset=["fyear"]).sort_values("filingDate").reset_index(drop=True)


def fiscal_year_end_month(statements: StatementSet) -> int | None:
    """The month a fiscal year ends in, from the latest annual report period.

    Sharadar publishes no fiscal-calendar attribute, so it is read off the
    data. Only annual statements carry it — a quarterly set would report
    whichever quarter came last.
    """
    frame = statements.frame("PL")
    if frame.empty or "Report Date" not in frame.columns:
        return None

    annual = frame
    if "Fiscal Period" in frame.columns:
        annual = frame[frame["Fiscal Period"] == "FY"]
        if annual.empty:
            return None

    dates = pd.to_datetime(annual["Report Date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return int(dates.iloc[-1].month)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

_PRICE_NUMERIC = ("open", "high", "low", "close", "volume", "closeadj", "closeunadj")


def _fetch_price_rows(
    ticker: str, *, table: str, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    params: dict = {"ticker": ticker}
    if start:
        params["date.gte"] = start
    if end:
        params["date.lte"] = end

    frame = _request_all(table, params, ticker=ticker)
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in _PRICE_NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _price_table_for(ticker: str) -> str:
    """Which of SEP/SFP carries this ticker. Defaults to SEP for equities."""
    return _fetch_price_tables((ticker.strip().upper(),)).get(
        ticker.strip().upper(), "SEP"
    )


def _normalise_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Types a raw DAILY response and rescales market cap and EV to raw units.

    DAILY publishes ``marketcap`` and ``ev`` in millions while SF1 publishes the
    same quantities raw. Everything downstream — the market-cap cross-check, the
    equity bridge, the valuation tab's peer medians — works in raw units, and a
    missed conversion looks entirely plausible while being wrong by a factor of
    a million.

    Shared by the per-ticker and universe-wide fetchers rather than inlined in
    either. There is now more than one DAILY call site, so the conversion has to
    live somewhere neither of them can forget it.
    """
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame()

    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in _DAILY_NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in _DAILY_MILLIONS:
        if column in frame.columns:
            frame[column] = frame[column] * 1_000_000.0
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _fetch_daily_metrics(ticker: str, start: str | None = None) -> pd.DataFrame:
    """One company's daily valuation ratios, oldest first."""
    params: dict = {"ticker": ticker}
    if start:
        params["date.gte"] = start
    return _normalise_daily(_request_all("DAILY", params, ticker=ticker))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_daily_metrics(ticker: str, start: str | None = None) -> pd.DataFrame:
    """Daily valuation ratios: ``ev evebit evebitda marketcap pb pe ps``.

    ``fetch_prices`` merges only ``marketcap`` and ``evebitda`` into the price
    frame because that is all the DCF needs; this exposes the whole thing for
    callers that want the multiples. Both go through the same private fetcher,
    which is the point — the millions-to-raw conversion for ``marketcap`` and
    ``ev`` happens in exactly one place, and a second call site doing its own
    DAILY request would be a factor-of-a-million bug waiting to happen.

    Columns keep Sharadar's own names and the frame is sorted oldest-first.
    """
    return _fetch_daily_metrics(ticker, start)


def _fetch_prices(ticker: str, start: str | None = None) -> pd.DataFrame:
    prices = _fetch_price_rows(ticker, table=_price_table_for(ticker), start=start)
    if prices.empty:
        raise TickerNotFoundError(
            f"Sharadar returned no price history for {ticker}.",
            table="SEP",
            ticker=ticker,
        )

    out = pd.DataFrame(
        {
            "Date": prices["date"],
            "Last Closing Price": prices.get("closeunadj"),
            "Adjusted Closing Price": prices.get("closeadj"),
            "Split-Adjusted Close": prices.get("close"),
            "Volume": prices.get("volume"),
        }
    )

    daily = _fetch_daily_metrics(ticker, start=start)
    if not daily.empty:
        merged = out.merge(
            daily[[c for c in ("date", "marketcap", "evebitda") if c in daily.columns]],
            left_on="Date",
            right_on="date",
            how="left",
        ).drop(columns=["date"])
        merged = merged.rename(
            columns={"marketcap": "Market-Cap", "evebitda": "EV/EBITDA - ttm"}
        )
        # Sharadar's market cap is sharesbas × split-adjusted close × sharefactor,
        # so dividing it back out recovers the share count on the same basis.
        # This is a cross-check against the SF1 count, not a substitute for it.
        if {"Market-Cap", "Split-Adjusted Close"}.issubset(merged.columns):
            merged["Common Shares Outstanding"] = (
                merged["Market-Cap"] / merged["Split-Adjusted Close"]
            )
        out = merged

    return out.sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_prices(ticker: str, start: str | None = None) -> pd.DataFrame:
    """Daily prices and valuation ratios for one company.

    ``Adjusted Closing Price`` is Sharadar's ``closeadj`` — adjusted for splits,
    stock dividends, cash dividends and spinoffs — and is what the beta
    regression uses. ``Last Closing Price`` is ``closeunadj``, the raw traded
    price, and is only correct for the current quote and market cap.
    """
    return _fetch_prices(ticker, start)


def _fetch_price_history(
    tickers: tuple[str, ...], start_date: str, end_date: str
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    wanted = tuple(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    if not wanted:
        return pd.DataFrame()

    routes = _fetch_price_tables(wanted)
    columns: dict[str, pd.Series] = {}
    for ticker in wanted:
        rows = _fetch_price_rows(
            ticker, table=routes.get(ticker, "SEP"), start=start_date, end=end_date
        )
        if rows.empty or "closeadj" not in rows.columns:
            continue
        columns[ticker] = rows.set_index("date")["closeadj"].dropna()

    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns).sort_index().dropna(how="all")


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_price_history(
    tickers: tuple[str, ...], start_date: str, end_date: str
) -> pd.DataFrame:
    """Adjusted closes for several tickers, one column each.

    Tickers are routed to SEP or SFP automatically, so a mixed list of stocks
    and ETFs works. Uses ``closeadj`` (total return) throughout, so every column
    is on one adjustment convention — which is what makes cross-ticker
    correlations and betas comparable.
    """
    return _fetch_price_history(tickers, start_date, end_date)


def fetch_index_prices(index_ticker: str, start_date: str, end_date: str) -> pd.Series:
    """Adjusted closes for a benchmark, as a Series.

    The benchmark is an ETF from SFP rather than an index, because Sharadar
    carries no index levels. ``SPY`` is the default stand-in for the S&P 500;
    its ``closeadj`` is a total-return series, which matches the convention on
    the company side of the regression.
    """
    frame = fetch_price_history((index_ticker,), start_date, end_date)
    if frame.empty:
        return pd.Series(dtype="float64", name=index_ticker)
    series = frame[index_ticker] if index_ticker in frame.columns else frame.iloc[:, 0]
    return series.dropna().rename(index_ticker)


def _fetch_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    rows = _fetch_price_rows(
        ticker, table=_price_table_for(ticker), start=start_date, end=end_date
    )
    if rows.empty:
        return pd.DataFrame()

    out = rows.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "closeadj": "Adj Close",
        }
    )
    keep = [c for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume") if c in out]
    return out.set_index("Date")[keep]


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Full OHLCV for one ticker, indexed by date.

    ``Open``/``High``/``Low``/``Close`` are adjusted for splits and stock
    dividends but not cash dividends, so the four stay mutually consistent and
    a candlestick drawn from them is honest. ``Adj Close`` is the fully adjusted
    series for anything that needs total return.
    """
    return _fetch_ohlcv(ticker, start_date, end_date)


# ---------------------------------------------------------------------------
# Corporate actions
# ---------------------------------------------------------------------------

#: ``action`` values seen in live ACTIONS responses (AAPL, KO, GOOGL, AMZN):
#: ``dividend``, ``split``, ``listed``, ``initiated``, ``relation``,
#: ``acquisitionof``, ``tickerchangefrom``, ``tickerchangeto``. The list is not
#: closed — Sharadar adds action types — so callers filter for the ones they
#: understand rather than exhausting it.
DIVIDEND_ACTION = "dividend"
SPLIT_ACTION = "split"

#: ACTIONS fills the contra columns with the *string* ``"N/A"`` rather than a
#: null when an action has no counterparty, which would otherwise render as a
#: literal "N/A" ticker on screen.
_ACTIONS_NULL_TEXT = frozenset({"N/A", "NA", ""})


def _fetch_actions(ticker: str, start: str | None = None) -> pd.DataFrame:
    params: dict = {"ticker": ticker}
    if start:
        params["date.gte"] = start

    frame = _request_all("ACTIONS", params, ticker=ticker)
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


# ---------------------------------------------------------------------------
# Insider transactions
# ---------------------------------------------------------------------------

#: The SF2 columns the app reads, projected with ``qopts.columns``. SF2 carries
#: 24; four are dropped, for two different reasons:
#:
#: * ``issuername`` is **null on 100% of rows** — 0 of 31,724 sampled across
#:   twelve tickers — so it is not a name source. ``fetch_ticker_meta`` is.
#: * ``dateexercisable``, ``priceexercisable`` and ``expirationdate`` describe an
#:   option's terms, and nothing built on this table reads an option's terms.
#:
#: ``rownum`` is kept although nothing displays it: it is part of the primary key
#: (ticker+ownername+filingdate+formtype+rownum) and is the only tiebreak that
#: orders two line items within one filing. Measured unique across all 5,627
#: filings sampled, in filings running up to 269 line items.
_SF2_COLUMNS = (
    "ticker,filingdate,formtype,ownername,officertitle,isdirector,isofficer,"
    "istenpercentowner,transactiondate,securityadcode,transactioncode,"
    "sharesownedbeforetransaction,transactionshares,"
    "sharesownedfollowingtransaction,transactionpricepershare,transactionvalue,"
    "securitytitle,directorindirect,natureofownership,rownum"
)

#: Every numeric column SF2 carries. Coerced per-table for the same reason as
#: :data:`_DAILY_NUMERIC`: TICKERS is all text and blanket coercion would blank
#: it.
_SF2_NUMERIC = (
    "sharesownedbeforetransaction",
    "transactionshares",
    "sharesownedfollowingtransaction",
    "transactionpricepershare",
    "transactionvalue",
    "rownum",
)

_SF2_DATES = ("filingdate", "transactiondate")

#: Columns holding short vendor codes, upper-cased on ingest so a consumer's
#: frozenset membership test cannot miss on case.
_SF2_CODES = (
    "formtype",
    "securityadcode",
    "transactioncode",
    "isdirector",
    "isofficer",
    "istenpercentowner",
    "directorindirect",
)

#: The literal SF2 writes into ``isdirector``/``isofficer``/``istenpercentowner``.
SF2_FLAG_YES = "Y"

#: ``securityadcode`` is two letters: N(on-derivative) or D(erivative), then
#: A(cquired) or D(isposed). A **bare** ``N`` or ``D`` with no second letter is a
#: holdings line rather than a trade — a Form 3, or a Form 4 line restating a
#: position. On the live sample that mapping was exact: every bare code carried a
#: null ``transactioncode`` and ``transactionshares`` of 0, and they are **28% of
#: all rows**.
#:
#: Published here because it is the vendor's vocabulary. What any of it *means*
#: to a tab — which codes count as a decision — belongs in the logic layer, the
#: same split as :data:`DIVIDEND_ACTION` against ``growth.METRICS``.
SF2_NON_DERIVATIVE = "N"
SF2_DERIVATIVE = "D"
SF2_ACQUIRED = "A"
SF2_DISPOSED = "D"

#: Identifies one reported transaction across filings. A ``RESTATED - 4``
#: supersedes the ``4`` that carried the same trade; 90 of 142 restated rows
#: sampled duplicated an existing row on exactly these fields.
_SF2_TRANSACTION_KEY = (
    "ownername",
    "transactiondate",
    "securitytitle",
    "transactioncode",
    "transactionshares",
)


def _fetch_insider_transactions(ticker: str, since: str | None = None) -> pd.DataFrame:
    # The coercions live inline rather than in a shared ``_normalise_sf2``,
    # because SF2 has one call site. ``_normalise_daily`` exists only because
    # DAILY has two and a forgotten millions conversion is a factor-of-a-million
    # bug. Split one out the moment a second appears — that would be a
    # universe-wide SF2 fetch keyed on a date rather than a ticker, which is what
    # a cross-sectional insider screen needs.
    params: dict = {"ticker": ticker, "qopts.columns": _SF2_COLUMNS}
    if since:
        params["filingdate.gte"] = since

    frame = _request_all("SF2", params, ticker=ticker)
    if frame.empty or "filingdate" not in frame.columns:
        return pd.DataFrame()

    for column in _SF2_DATES:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in _SF2_NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in _SF2_CODES:
        if column in frame.columns:
            frame[column] = (
                frame[column].astype("string").str.strip().str.upper()
            )

    frame = frame.dropna(subset=["filingdate"])
    return _order_insider_rows(frame)


def _order_insider_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Sorts chronologically and drops superseded filings.

    The sort is ``filingdate``, then ``rownum``. Both matter: rows arrive
    unsorted by contract, and two line items filed on the same day each carry a
    ``sharesownedfollowingtransaction`` balance, of which only the last in row
    order is the end-of-day one.

    Then the restatement drop, mirroring :func:`_order_statement_rows` and the
    universe dedup. A ``RESTATED - 4`` is a correction of a transaction already
    reported under form ``4``, not a second transaction: 90 of 142 restated rows
    sampled repeated an existing row on :data:`_SF2_TRANSACTION_KEY`, and letting
    both through counts one trade twice. Holdings-only rows are exempt — they
    carry no transaction to duplicate, and every field of the key is null or zero
    on them, so a single dedup would collapse a company's entire Form 3 history
    into one row.
    """
    if frame.empty or "filingdate" not in frame.columns:
        return frame

    ordered = frame.sort_values(
        [c for c in ("filingdate", "rownum") if c in frame.columns],
        kind="stable",
    )
    key = [c for c in _SF2_TRANSACTION_KEY if c in ordered.columns]
    if len(key) == len(_SF2_TRANSACTION_KEY):
        transactions = ordered["transactiondate"].notna()
        deduped = ordered[transactions].drop_duplicates(subset=key, keep="last")
        ordered = pd.concat([deduped, ordered[~transactions]])
        ordered = ordered.sort_values(
            [c for c in ("filingdate", "rownum") if c in ordered.columns],
            kind="stable",
        )
    return ordered.reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def fetch_insider_transactions(ticker: str, since: str | None = None) -> pd.DataFrame:
    """Form 3/4/5 line items for one company, oldest filing first.

    Columns are Sharadar's own. ``transactionshares`` is **signed** — negative on
    a disposal, verified across every code on a live sample — while
    ``transactionvalue`` is the unsigned magnitude of the same trade and is null
    on 48% of rows. That disagreement is left exactly as sent; see quirk 6 in the
    module docstring before summing anything here.

    ``since`` filters on **``filingdate``**, which is forced rather than chosen:
    SF2's filterable parameters are ``filingdate``, ``ownername``,
    ``securityadcode``, ``ticker`` and ``transactionvalue``, and
    ``transactiondate`` is not among them. It is a superset rather than a
    compromise — a Form 4 follows the trade it reports and never precedes it, so
    every transaction inside a window is inside the filing window too. The
    surplus is old trades filed late, which ``logic/research/insiders`` trims on
    ``transactiondate``, the date the decision was actually taken.

    Returns an **empty frame rather than raising** when there is nothing, on the
    same contract as :func:`fetch_actions`: a company whose insiders have never
    filed is ordinary, not an error, and an empty response cannot be told apart
    from an unknown ticker here. Callers needing the ticker validated should use
    ``fetch_ticker_meta``.

    Twelve hours, matching :func:`fetch_actions` — the other per-ticker event
    feed. Form 4s land continuously (two business days from the trade), so the
    week TICKERS gets would show a stale roster.

    Both parameters are forwarded positionally and neither is keyword-only. Do
    not add a keyword default here: ``st.cache_data`` keys on the arguments as
    written at the call site without applying defaults, which is the bug
    :func:`fetch_statements` exists to work around.
    """
    return _fetch_insider_transactions(ticker, since)


# ---------------------------------------------------------------------------
# Cross-sectional universe
# ---------------------------------------------------------------------------
#
# Every other fetch here is keyed on a ticker. These three are keyed on a date,
# and that difference is the whole point: one request returns the multiple for
# every listed company, so a peer median costs the same whether the peer group
# has four members or four hundred. Measured against the live API: DAILY for one
# date is 5,446 rows on a single page in ~1.1s.
#
# Because the cache keys carry no ticker, one entry serves every company looked
# up that day. A second search is free.

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
    return _normalise_daily(_request_all("DAILY", {"date": as_of}))


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
    frame = _request_all(
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
    frame = _request_all(
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
