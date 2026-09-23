"""``SF2``: Form 3/4/5 line items for one company.

**SF2 signs the shares, not the value.** ``transactionshares`` is signed —
negative on a disposal — while ``transactionvalue`` is the *unsigned magnitude*
of the same trade: 40,334 shares sold at $115.95 carries a ``transactionvalue``
of 4,676,727 with no minus, and is null on 48% of rows besides. Summing that
column therefore adds every sale to every purchase and misses half the trades.
SF2 is deliberately **not** in ``SIGN_FLIPPED``: that set negates indicators
Sharadar publishes as positive magnitudes against a costs-negative contract, and
``transactionshares`` is already correct — negating it would invert every trade.
The dollar flow is ``transactionshares × transactionpricepershare``, computed in
``logic/research/insiders`` and nowhere else.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data.sharadar import transport

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

    frame = transport._request_all("SF2", params, ticker=ticker)
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
    on 48% of rows. That disagreement is left exactly as sent; see the module
    docstring before summing anything here.

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
