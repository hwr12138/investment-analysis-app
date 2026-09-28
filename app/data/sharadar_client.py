"""Nasdaq Data Link (Sharadar) client — the public face of ``app.data.sharadar``.

Everything that talks to ``data.nasdaq.com`` is reached through here. Callers get
pandas frames with resolved column names and nothing else. This module holds no
code of its own: it re-exports the public names of the implementation package,
which is split so that each part can be read — and imported — on its own:

* ``vocabulary``   — what the vendor's columns and codes mean. **Pure**, and the
  only part the logic layer may import; see ``tests/test_architecture.py``.
* ``transport``    — auth, retry, pagination, datatable parsing, the errors.
* ``reference``    — ``TICKERS``: one company's record, price-table routing.
* ``statements``   — ``SF1`` for one company, and views over a ``StatementSet``.
* ``prices``       — ``SEP``/``SFP`` prices and ``DAILY`` ratios.
* ``actions``      — ``ACTIONS``.
* ``insiders``     — ``SF2``.
* ``institutions`` — ``SF3``/``SF3A``/``SF3B``.
* ``universe``     — the three date-keyed, every-company fetches.

Tests take the network away by patching ``transport._get``; every table module
calls it through the module attribute so that one patch reaches them all.

Eight Sharadar tables are used:

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
* ``SF3``     — Core US Institutional Investors. One row per (ticker, investor,
  security type, quarter) from SEC form 13F, from June 2013. ``SF3A`` is the
  same data summarised per ticker.

Most fetches are keyed on a ticker. The three in ``universe`` are keyed on a
date instead, and return every listed company at once — see that module for why
that is cheap.

Seven things about this API are not obvious from its docs and were confirmed
against live AAPL and JPM responses. They are the reason this module exists
rather than a handful of inline ``requests.get`` calls:

1. **Sign conventions differ from the engine's.** SF1 signs ``cor``, ``opex``,
   ``sgna``, ``rnd``, ``taxexp`` and ``intexp`` as positive magnitudes. The
   engine's contract (see ``financials`` module docstring) is costs-negative.
   ``vocabulary.SIGN_FLIPPED`` is negated on ingest so nothing downstream has
   to know. ``capex`` is *already* negative and is deliberately not in that set.
2. **``DAILY`` reports market cap and EV in millions**, while ``SF1`` reports
   the same quantities in raw units. ``prices._DAILY_MILLIONS`` is scaled on
   ingest; a missed conversion passes every sanity check by a factor of a
   million.
3. **Zero means "absent from the filing"** for 41 indicators — Sharadar
   documents "where this item is not contained on the company consolidated
   financial statements … the value of 0 is used". The engine's rule is that
   missing stays missing, so the zeros pass through as reported but are
   enumerated in ``vocabulary.ZERO_FILLED_INDICATORS`` for the validation layer
   to flag. See ``validation.check_zero_filled_indicators``.
4. **Rows are unsorted by contract** and the API truncates at 10,000 rows,
   continuing through an opaque ``meta.next_cursor_id``. Every frame leaves
   the client paginated to completion and sorted.
5. **A non-trading date is not an error.** ``DAILY`` answers a weekend or a
   holiday with zero rows rather than the previous session, so any date passed
   to ``fetch_universe_daily`` has to be one the market was open on.
6. **SF2 signs the shares, not the value.** ``transactionshares`` is signed —
   negative on a disposal — while ``transactionvalue`` is the *unsigned
   magnitude* of the same trade: 40,334 shares sold at $115.95 carries a
   ``transactionvalue`` of 4,676,727 with no minus, and is null on 48% of rows
   besides. Summing that column therefore adds every sale to every purchase and
   misses half the trades. SF2 is deliberately **not** in ``SIGN_FLIPPED``:
   that set negates indicators Sharadar publishes as positive magnitudes against
   a costs-negative contract, and ``transactionshares`` is already correct —
   negating it would invert every trade. The sign SF2 needs lives in a different
   column, which is a join rather than a negation, so no column-wise transform
   fixes it on ingest. The dollar flow is ``transactionshares ×
   transactionpricepershare``, computed in ``logic/research/insiders`` and
   nowhere else.

The API key is read from ``st.secrets`` and never leaves ``transport``. Nasdaq
only accepts it as a query parameter, so no exception the client raises carries
the params or the full URL — only the table, the ticker, and the status.
"""

from app.data.sharadar.actions import fetch_actions
from app.data.sharadar.insiders import fetch_insider_transactions
from app.data.sharadar.institutions import (
    fetch_institutional_holders,
    fetch_institutional_summary,
    fetch_investor_directory,
)
from app.data.sharadar.prices import (
    fetch_daily_metrics,
    fetch_index_prices,
    fetch_ohlcv,
    fetch_price_history,
    fetch_prices,
)
from app.data.sharadar.reference import fetch_price_tables, fetch_ticker_meta
from app.data.sharadar.statements import (
    common_shares_frame,
    dimension_for,
    fetch_statements,
    filings_frame,
    fiscal_year_end_month,
    share_factor,
    weighted_shares_frame,
)
from app.data.sharadar.transport import (
    BASE_URL,
    SharadarError,
    TickerNotFoundError,
    parse_datatable,
)
from app.data.sharadar.universe import (
    UNIVERSE_LOOKBACK_DAYS,
    fetch_ticker_directory,
    fetch_universe_daily,
    fetch_universe_fundamentals,
)
from app.data.sharadar.vocabulary import (
    CAPTION_TO_INDICATOR,
    DIVIDEND_ACTION,
    FINANCIAL_SIC_RANGE,
    META_COLUMNS,
    MINORITY_INTEREST_CAPTION,
    PRICE_CURRENCY,
    SF1_CAPTIONS,
    SF1_LAYOUT,
    SF2_ACQUIRED,
    SF2_DERIVATIVE,
    SF2_DISPOSED,
    SF2_FLAG_YES,
    SF2_NON_DERIVATIVE,
    SF3_CALLS,
    SF3_PUTS,
    SF3_SHARES,
    SIGN_FLIPPED,
    SPLIT_ACTION,
    ZERO_FILLED_INDICATORS,
    StatementSet,
    is_zero_filled,
    split_factors,
)

__all__ = [
    "BASE_URL",
    "CAPTION_TO_INDICATOR",
    "DIVIDEND_ACTION",
    "FINANCIAL_SIC_RANGE",
    "META_COLUMNS",
    "MINORITY_INTEREST_CAPTION",
    "PRICE_CURRENCY",
    "SF1_CAPTIONS",
    "SF1_LAYOUT",
    "SF2_ACQUIRED",
    "SF2_DERIVATIVE",
    "SF2_DISPOSED",
    "SF2_FLAG_YES",
    "SF2_NON_DERIVATIVE",
    "SF3_CALLS",
    "SF3_PUTS",
    "SF3_SHARES",
    "SIGN_FLIPPED",
    "SPLIT_ACTION",
    "UNIVERSE_LOOKBACK_DAYS",
    "ZERO_FILLED_INDICATORS",
    "SharadarError",
    "StatementSet",
    "TickerNotFoundError",
    "common_shares_frame",
    "dimension_for",
    "fetch_actions",
    "fetch_daily_metrics",
    "fetch_index_prices",
    "fetch_insider_transactions",
    "fetch_institutional_holders",
    "fetch_institutional_summary",
    "fetch_investor_directory",
    "fetch_ohlcv",
    "fetch_price_history",
    "fetch_price_tables",
    "fetch_prices",
    "fetch_statements",
    "fetch_ticker_directory",
    "fetch_ticker_meta",
    "fetch_universe_daily",
    "fetch_universe_fundamentals",
    "filings_frame",
    "fiscal_year_end_month",
    "is_zero_filled",
    "parse_datatable",
    "share_factor",
    "split_factors",
    "weighted_shares_frame",
]
