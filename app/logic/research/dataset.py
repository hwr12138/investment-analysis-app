"""The snapshot every research tab starts from, and each tab's own fetch.

:func:`load_research` is called once per search and fetches only what the
📊 Summary tab draws, because that is the tab a search opens on. Every other tab
that needs more has a loader here beside it — :func:`load_financials`,
:func:`load_insiders`, :func:`load_peers`, :func:`load_holdings` — and calls it
the first time it is opened. The 💰 DCF is the one exception, loading through
``dcf.model.load_company`` for the reason given at the bottom of this docstring.

That split was not expressible until Streamlit's tab strip learned to report its
selection. Before it, ``st.tabs`` rendered every body on every rerun, so one
search paid for every tab — 23 requests one after another, 27.6s on MSFT — and
the 🏛️ Holdings register hid behind a button because it was too dear to bill to
everyone. With ``on_change="rerun"`` only the open tab's body runs, and every tab
loads the same way: on first open, behind a spinner, cached after that.

Within each loader the independent requests go out together through
:func:`app.data.parallel.gather`. The ticker's reference record goes first and
alone, because every statement fetch reads it through the cache and concurrent
misses would each pay for it.

The peer universe costs a fixed amount rather than one request per comparable:
``DAILY`` filtered on a date and no ticker returns every listed company in a
single page — 5,446 rows in ~1.1s — and all three universe fetches are keyed on a
date alone, so one cache entry serves every company searched that day.

Nothing here goes through ``dcf.model.load_company``: that rejects SIC 6000–6999
because banks and insurers have no meaningful FCFF, which is right for a
valuation and wrong here. A research page must render JPM.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.data import parallel, sharadar_client
from app.logic.research import comparison, holdings
from app.logic.research.summary import (
    BENCHMARK,
    BETA_YEARS,
    StatTile,
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
#: years arrived in two.
INSIDER_LOOKBACK_YEARS = 5


@dataclass(frozen=True)
class ResearchDataset:
    """Everything the page fetched for one ticker, in one snapshot."""

    ticker: str
    profile: dict
    ohlcv: pd.DataFrame
    daily: pd.DataFrame
    actions: pd.DataFrame
    #: The four SF1 frames at the TTM dimension. Empty when that dimension could
    #: not be fetched — see :func:`_statement_frames`. The annual and quarterly
    #: dimensions are the 📄 Financials tab's, fetched by :func:`load_financials`.
    statements_ttm: dict[str, pd.DataFrame]
    #: The as-reported quarterly income statement, kept as a bare frame because
    #: the only thing that wants it is the next-earnings-date tile: the ``AR``
    #: dimensions are the ones carrying real filing dates.
    quarterly_ar: pd.DataFrame
    stats: list[StatTile]
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

    @property
    def market_session(self) -> date | None:
        """The last session in this company's ``DAILY`` history.

        What the peer universe is anchored on rather than ``loaded_at``: Sharadar
        answers a weekend or a holiday with zero rows, and a date the company
        already has a row for is a session the market was open on. It also lines
        the peers up with the company's own latest reading, which is the
        comparison the ⚖️ Relative Valuation tab claims to be making.
        """
        if self.daily.empty or "date" not in self.daily.columns:
            return None
        as_of = pd.Timestamp(self.daily["date"].max())
        return None if pd.isna(as_of) else as_of.date()


def load_research(ticker: str, *, today: date | None = None) -> ResearchDataset:
    """Fetches and assembles what the 📊 Summary tab shows.

    Called once per search, never from a re-render — switching price windows
    slices :attr:`ResearchDataset.ohlcv` rather than refetching it. Every other
    tab fetches its own on first open; see the module docstring.

    Individual failures below the price history are tolerated — a company with
    no ACTIONS rows or no ``ARQ`` history still gets a page, with the affected
    tiles reporting themselves unavailable.
    """
    ticker = ticker.strip().upper()
    today = today or date.today()

    profile = sharadar_client.fetch_ticker_meta(ticker)

    beta_start = (pd.Timestamp(today) - pd.DateOffset(years=BETA_YEARS)).date()
    fetched = parallel.gather(
        {
            "ohlcv": lambda: sharadar_client.fetch_ohlcv(
                ticker, HISTORY_FLOOR, today.isoformat()
            ),
            "daily": lambda: _tolerant(
                lambda: sharadar_client.fetch_daily_metrics(ticker)
            ),
            "actions": lambda: _tolerant(lambda: sharadar_client.fetch_actions(ticker)),
            "statements_ttm": lambda: _statement_frames(ticker, annual=False, ttm=True),
            "quarterly_ar": lambda: _tolerant(
                lambda: sharadar_client.fetch_statements(
                    ticker, annual=False, asreported=True
                ).frame("PL")
            ),
            "index_closes": lambda: _tolerant(
                lambda: sharadar_client.fetch_index_prices(
                    BENCHMARK, beta_start.isoformat(), today.isoformat()
                ),
                empty=pd.Series(dtype="float64"),
            ),
        }
    )

    ohlcv = fetched["ohlcv"]
    if ohlcv.empty:
        raise sharadar_client.TickerNotFoundError(
            f"Sharadar returned no price history for {ticker}.",
            table="SEP",
            ticker=ticker,
        )

    stock_closes = (
        ohlcv["Adj Close"].dropna()
        if "Adj Close" in ohlcv.columns
        else pd.Series(dtype="float64")
    )
    stock_closes = stock_closes[stock_closes.index >= pd.Timestamp(beta_start)]

    stats = build_key_stats(
        ohlcv=ohlcv,
        daily=fetched["daily"],
        ttm_pl=fetched["statements_ttm"].get("PL", pd.DataFrame()),
        quarterly_ar=fetched["quarterly_ar"],
        actions=fetched["actions"],
        stock_closes=stock_closes,
        index_closes=fetched["index_closes"],
        currency=str(profile.get("currency") or ""),
        today=today,
    )

    return ResearchDataset(
        ticker=ticker,
        profile=profile,
        ohlcv=ohlcv,
        daily=fetched["daily"],
        actions=fetched["actions"],
        statements_ttm=fetched["statements_ttm"],
        quarterly_ar=fetched["quarterly_ar"],
        stats=stats,
        currency=str(profile.get("currency") or ""),
        loaded_at=today,
    )


# ---------------------------------------------------------------------------
# Per-tab loaders — each called the first time its tab is opened
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinancialsData:
    """The 📄 Financials tab's statements beyond the TTM the snapshot carries.

    Each dimension is ``{}`` when it could not be fetched, independently: a
    company too young to have an annual filing still gets a quarterly view.
    """

    ticker: str
    annual: dict[str, pd.DataFrame]
    quarterly: dict[str, pd.DataFrame]


def load_financials(ticker: str) -> FinancialsData:
    """The annual and quarterly SF1 frames, fetched together.

    The DCF asks for the same two dimensions through the same cached fetcher,
    so whichever of the two tabs is opened first pays and the other does not.
    """
    ticker = ticker.strip().upper()
    fetched = parallel.gather(
        {
            "annual": lambda: _statement_frames(ticker, annual=True),
            "quarterly": lambda: _statement_frames(ticker, annual=False),
        }
    )
    return FinancialsData(ticker, fetched["annual"], fetched["quarterly"])


def load_insiders(ticker: str, *, today: date | None = None) -> pd.DataFrame:
    """Every Form 3/4/5 line item filed in the last :data:`INSIDER_LOOKBACK_YEARS`.

    **Raw on purpose**: ``transactionvalue`` is unsigned here and the
    holdings-only rows are still in, because ``logic/research/insiders`` owns
    every interpretation. Empty when nobody has filed, which is an ordinary
    outcome for a small company. Not :func:`_tolerant`: now that the 👤 Insiders
    tab loads this itself, a failed request is shown as one rather than passing
    for a company nobody has traded.
    """
    ticker = ticker.strip().upper()
    today = today or date.today()
    since = (pd.Timestamp(today) - pd.DateOffset(years=INSIDER_LOOKBACK_YEARS)).date()
    return sharadar_client.fetch_insider_transactions(ticker, since.isoformat())


def load_peers(session: date | None, *, today: date | None = None) -> pd.DataFrame:
    """Every listed company's multiples and industry on ``session``.

    What the ⚖️ Relative Valuation tab's two sub-tabs take their industry medians over.
    ``session`` is :attr:`ResearchDataset.market_session`; nothing is fetched
    without one. Keyed on dates alone, never a ticker, so one set of cache
    entries serves every company searched that day.

    Each of the three fetches is :func:`_tolerant`, and the result is empty when
    any is missing — which leaves that tab reporting its medians unavailable
    while the company's own current and five-year figures still render.
    """
    if session is None:
        return pd.DataFrame()
    today = today or date.today()
    since = (
        pd.Timestamp(today) - pd.Timedelta(days=sharadar_client.UNIVERSE_LOOKBACK_DAYS)
    ).date()
    fetched = parallel.gather(
        {
            "daily": lambda: _tolerant(
                lambda: sharadar_client.fetch_universe_daily(session.isoformat())
            ),
            "fundamentals": lambda: _tolerant(
                lambda: sharadar_client.fetch_universe_fundamentals(since.isoformat())
            ),
            "directory": lambda: _tolerant(sharadar_client.fetch_ticker_directory),
        }
    )
    return comparison.build_peer_universe(
        fetched["daily"], fetched["fundamentals"], fetched["directory"]
    )


@dataclass(frozen=True)
class HoldingsData:
    """The 🏛️ Holdings tab's own fetch, kept out of :class:`ResearchDataset`.

    The slowest thing on the page — five requests and some twenty seconds on a
    large register — and loaded like every other tab: the first time it is
    opened, and cached after that.
    """

    ticker: str
    #: SF3A, every quarter since 2013. Funds the ownership and breadth history.
    summary: pd.DataFrame
    #: SF3 detail for the last two quarters, every holder. Funds the register,
    #: the quarter-on-quarter move, churn and concentration.
    current: pd.DataFrame
    #: SF3 detail for the largest holders across every quarter. The sparklines.
    history: pd.DataFrame
    #: ``investorid`` → ``investorname``. Keyed on a date, so one entry serves
    #: every company looked at that day.
    names: pd.DataFrame
    #: SF1 quarterly ``DERIVED`` frame, for the restated share count ownership is
    #: divided by. The same cache entry the 📄 Financials tab reads.
    derived: pd.DataFrame
    loaded_at: date


def load_holdings(
    ticker: str, *, today: date | None = None, top: int = 20
) -> HoldingsData:
    """Fetch the institutional register for one company.

    Three rounds, because the order is forced: the summary names the quarters,
    the quarters bound the detail, and the detail ranks the holders whose
    history the last request asks for. What does not depend on that chain rides
    alongside it — the share count with the summary, the investor names with the
    detail.

    Everything below the summary is :func:`_tolerant`. A register with no
    sparklines beats no register, and an unresolved ``investorid`` still shows as
    a code rather than a blank.
    """
    ticker = ticker.strip().upper()
    today = today or date.today()

    first = parallel.gather(
        {
            "summary": lambda: sharadar_client.fetch_institutional_summary(ticker),
            "derived": lambda: _statement_frames(ticker, annual=False).get(
                "DERIVED", pd.DataFrame()
            ),
        }
    )
    summary, derived = first["summary"], first["derived"]
    quarters = (
        sorted(pd.to_datetime(summary["date"], errors="coerce").dropna().unique())
        if not summary.empty and "date" in summary.columns
        else []
    )
    if not quarters:
        empty = pd.DataFrame()
        return HoldingsData(ticker, summary, empty, empty, empty, derived, today)

    # Two quarters: the register, and something to measure it against.
    since = pd.Timestamp(quarters[-2] if len(quarters) > 1 else quarters[-1])
    second = parallel.gather(
        {
            "current": lambda: _tolerant(
                lambda: sharadar_client.fetch_institutional_holders(
                    ticker, since.date().isoformat()
                )
            ),
            "names": lambda: _tolerant(
                lambda: sharadar_client.fetch_investor_directory(
                    pd.Timestamp(quarters[-1]).date().isoformat()
                )
            ),
        }
    )
    current, names = second["current"], second["names"]

    investorids = holdings.top_investorids(
        current, pd.DataFrame(), tolerances=holdings.HoldingsTolerances(top_holders=top)
    )
    history = (
        _tolerant(
            lambda: sharadar_client.fetch_institutional_holders(
                ticker, None, investorids
            )
        )
        if investorids
        else pd.DataFrame()
    )
    return HoldingsData(ticker, summary, current, history, names, derived, today)


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
