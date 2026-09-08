"""The one snapshot every research tab reads from, and the fetch that builds it.

:func:`load_research` is the package's single network-touching entry point, in
the same shape as ``dcf.model.load_company``: called once per ticker, never
from inside a re-render. Everything else in ``app.logic.research`` is pure.

**This module is the seam where per-tab loading would land.** The fetch is
eager — one search pulls every table the page could want, whether or not the
reader opens the tab that needs it. The blocker is not this function:
``st.tabs`` renders every tab body on every rerun and reports no active tab to
Python, so "fetch when opened" is not expressible until the tab strip becomes a
control that reports its selection. Until then a tab whose data is expensive
gates behind its own button, which is what the DCF tab does.

The peer universe was expected to be where that stopped being affordable, on
the assumption that comparables cost one ``DAILY`` request each. They do not:
``DAILY`` filtered on a date and no ticker returns every listed company in a
single page — 5,446 rows in ~1.1s — so the cost is fixed rather than
per-comparable. Better still, all three universe fetches are keyed on a date
alone, so one cache entry serves every company searched that day and the second
search pays nothing.

:func:`load_research` deliberately does **not** go through
``dcf.model.load_company``: that rejects SIC 6000–6999 because banks and
insurers have no meaningful FCFF, which is right for a valuation and wrong
here. A research page must render JPM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.data import sharadar_client
from app.logic.research import comparison
from app.logic.research.summary import (
    BENCHMARK,
    StatTile,
    _BETA_YEARS,
    build_key_stats,
)

#: SEP coverage starts in the late 1990s regardless of what ``firstpricedate``
#: claims, so the ALL window asks from here and takes whatever exists. ~7,100
#: rows for a company listed throughout, which is one page.
HISTORY_FLOOR = "1990-01-01"

#: Years of insider filings fetched. Requested on ``filingdate``, the only date
#: SF2 lets a caller filter on — ``transactiondate`` is not a filterable
#: parameter — which makes it a superset rather than a compromise: a Form 4
#: follows the trade it reports, so every transaction in the window is in the
#: filing window too. The surplus is old trades filed late, and
#: ``logic/research/insiders`` trims it on the transaction date.
#:
#: Roughly one request per ticker — 13,295 rows for twelve tickers over five
#: years arrived in two — which is cheaper than the four statement fetches
#: below, so this goes in the eager fetch rather than behind a button.
INSIDER_LOOKBACK_YEARS = 5


@dataclass(frozen=True)
class ResearchDataset:
    """Everything the page fetched for one ticker, in one snapshot."""

    ticker: str
    profile: dict
    ohlcv: pd.DataFrame
    daily: pd.DataFrame
    actions: pd.DataFrame
    #: Every Form 3/4/5 line item filed in the last :data:`INSIDER_LOOKBACK_YEARS`
    #: years, as SF2 publishes them. **Raw on purpose**: ``transactionvalue`` is
    #: unsigned here and the holdings-only rows are still in, because
    #: ``logic/research/insiders`` owns every interpretation. Empty when SF2
    #: could not be fetched, or when nobody has filed — which is an ordinary
    #: outcome for a small company, not an error.
    insiders: pd.DataFrame
    #: The four SF1 frames at each dimension the page shows. Empty when that
    #: dimension could not be fetched — see :func:`_statement_frames`.
    statements_ttm: dict[str, pd.DataFrame]
    statements_annual: dict[str, pd.DataFrame]
    statements_quarterly: dict[str, pd.DataFrame]
    #: The as-reported quarterly income statement, kept as a bare frame because
    #: the only thing that wants it is the next-earnings-date tile: the ``AR``
    #: dimensions are the ones carrying real filing dates.
    quarterly_ar: pd.DataFrame
    stats: list[StatTile]
    #: Every listed company's multiples and industry on the same session, for
    #: the Valuation tab's peer medians. One row per ticker, including this one.
    #: Empty when the universe fetches failed, which leaves that tab reporting
    #: its medians unavailable rather than the page failing.
    peer_universe: pd.DataFrame
    currency: str
    loaded_at: date

    @property
    def name(self) -> str:
        return str(self.profile.get("name") or self.ticker)

    @property
    def ttm_pl(self) -> pd.DataFrame:
        """The TTM income statement.

        A property rather than a stored field so the TTM statements live in one
        place; the statistics builder and its tests are unchanged by the move.
        """
        return self.statements_ttm.get("PL", pd.DataFrame())


def load_research(ticker: str, *, today: date | None = None) -> ResearchDataset:
    """Fetches and assembles everything the research page shows.

    Called once per search, never from a re-render — switching price windows
    slices :attr:`ResearchDataset.ohlcv` rather than refetching it. See the
    module docstring for why it fetches for every tab rather than the open one.

    Individual failures below the price history are tolerated — a company with
    no ACTIONS rows or no ``ARQ`` history still gets a page, with the affected
    tiles reporting themselves unavailable.
    """
    ticker = ticker.strip().upper()
    today = today or date.today()

    profile = sharadar_client.fetch_ticker_meta(ticker)
    ohlcv = sharadar_client.fetch_ohlcv(ticker, HISTORY_FLOOR, today.isoformat())
    if ohlcv.empty:
        raise sharadar_client.TickerNotFoundError(
            f"Sharadar returned no price history for {ticker}.",
            table="SEP",
            ticker=ticker,
        )

    daily = _tolerant(lambda: sharadar_client.fetch_daily_metrics(ticker))
    actions = _tolerant(lambda: sharadar_client.fetch_actions(ticker))

    insider_since = (
        pd.Timestamp(today) - pd.DateOffset(years=INSIDER_LOOKBACK_YEARS)
    ).date()
    insiders = _tolerant(
        lambda: sharadar_client.fetch_insider_transactions(
            ticker, insider_since.isoformat()
        )
    )

    statements_ttm = _statement_frames(ticker, annual=False, ttm=True)
    statements_annual = _statement_frames(ticker, annual=True)
    statements_quarterly = _statement_frames(ticker, annual=False)
    quarterly_ar = _tolerant(
        lambda: sharadar_client.fetch_statements(
            ticker, annual=False, asreported=True
        ).frame("PL")
    )

    beta_start = (pd.Timestamp(today) - pd.DateOffset(years=_BETA_YEARS)).date()
    stock_closes = (
        ohlcv["Adj Close"].dropna()
        if "Adj Close" in ohlcv.columns
        else pd.Series(dtype="float64")
    )
    stock_closes = stock_closes[stock_closes.index >= pd.Timestamp(beta_start)]
    index_closes = _tolerant(
        lambda: sharadar_client.fetch_index_prices(
            BENCHMARK, beta_start.isoformat(), today.isoformat()
        ),
        empty=pd.Series(dtype="float64"),
    )

    peer_universe = _peer_universe(daily, today)

    stats = build_key_stats(
        ohlcv=ohlcv,
        daily=daily,
        ttm_pl=statements_ttm.get("PL", pd.DataFrame()),
        quarterly_ar=quarterly_ar,
        actions=actions,
        stock_closes=stock_closes,
        index_closes=index_closes,
        currency=str(profile.get("currency") or ""),
        today=today,
    )

    return ResearchDataset(
        ticker=ticker,
        profile=profile,
        ohlcv=ohlcv,
        daily=daily,
        actions=actions,
        insiders=insiders,
        statements_ttm=statements_ttm,
        statements_annual=statements_annual,
        statements_quarterly=statements_quarterly,
        quarterly_ar=quarterly_ar,
        stats=stats,
        peer_universe=peer_universe,
        currency=str(profile.get("currency") or ""),
        loaded_at=today,
    )


def _peer_universe(daily: pd.DataFrame, today: date) -> pd.DataFrame:
    """The cross-sectional snapshot the Valuation tab compares against.

    Anchored on the last date in *this company's* ``DAILY`` history rather than
    on ``today``: Sharadar answers a weekend or a holiday with zero rows, and a
    date the company already has a row for is a session the market was open on.
    It also lines the peers up with the company's own latest reading, which is
    the comparison the tab claims to be making.
    """
    if daily.empty or "date" not in daily.columns:
        return pd.DataFrame()
    as_of = pd.Timestamp(daily["date"].max())
    if pd.isna(as_of):
        return pd.DataFrame()

    since = (
        pd.Timestamp(today) - pd.Timedelta(days=sharadar_client.UNIVERSE_LOOKBACK_DAYS)
    ).date()
    return comparison.build_peer_universe(
        _tolerant(lambda: sharadar_client.fetch_universe_daily(as_of.date().isoformat())),
        _tolerant(lambda: sharadar_client.fetch_universe_fundamentals(since.isoformat())),
        _tolerant(sharadar_client.fetch_ticker_directory),
    )


def _tolerant(fetch, empty=None):
    """Runs a secondary fetch, returning empty on failure rather than raising.

    Only the price history is load-bearing enough to block the page. Everything
    else degrades into an unavailable tile that says so, which is the point of
    the provenance model — a missing feed is visible, not silently absent.
    """
    fallback = pd.DataFrame() if empty is None else empty
    try:
        return fetch()
    except sharadar_client.SharadarError:
        return fallback


def _statement_frames(
    ticker: str, *, annual: bool, ttm: bool = False, asreported: bool = False
) -> dict[str, pd.DataFrame]:
    """One SF1 dimension's four frames, or ``{}`` if the request failed.

    Same contract as :func:`_tolerant`: a dimension that does not come back
    leaves the financials tab saying so rather than blocking the page. A company
    too young to have an annual filing still gets a quarterly statement view.
    """
    return _tolerant(
        lambda: dict(
            sharadar_client.fetch_statements(
                ticker, annual=annual, ttm=ttm, asreported=asreported
            ).frames
        ),
        empty={},
    )
