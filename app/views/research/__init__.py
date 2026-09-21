"""Stock research page — one ticker, eleven tabs, one module per tab.

The general-purpose "what is this company" page. This module is the shell: the
ticker box, the search gate, the snapshot in session state, and the tab strip.
Every tab body lives in its own module beside it, and the arithmetic behind
them in ``app.logic.research``.

A tab module exposes a ``render`` function taking exactly what that tab needs —
the snapshot for the ones that read it, the bare ticker for the DCF, which
loads its own. There is no shared protocol beyond the name, because the
dispatch below is explicit and a uniform signature would only hide that the DCF
does not read the snapshot at all.

Three behaviours are deliberate and load-bearing:

* **Nothing is fetched until Search is pressed** — a page that fetches on every
  keystroke of a ticker box burns the API quota on half-typed symbols.
* **Changing the price window never refetches.** The full history is fetched
  once per search and every window is a slice of it, so the period buttons are
  free.
* **Tab bodies fetch nothing** — with two exceptions, which take opposite
  routes. ``st.tabs`` renders all eleven bodies on every rerun and tells Python
  nothing about which one is open, so anything that loaded data would fire on
  page load and on every window change.

  💰 DCF needs statements this page's search does not fetch, and it fetches them
  **ungated**: a second set of Sharadar requests per company searched, paid
  whether or not anyone opens the tab, with its own caches keeping it to once
  per company. 🏛️ Holdings needs a third set — four requests and about ten
  seconds for a register most searches never look at — and takes the other
  route: **nothing is fetched until a button is pressed.** That is the rule
  CLAUDE.md's UI conventions have stated since before any tab followed it, and
  Holdings is the first that does.

  See ``app.logic.research.dataset`` for what the same constraint costs the
  tabs that do read the snapshot.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from app.data import sharadar_client
from app.logic.research import dataset as research
from app.views.research import (
    dcf,
    financials,
    holdings,
    insiders,
    placeholders,
    summary,
    valuation,
)

#: Every tab, in display order — the one place the order is decided. The five
#: built tabs are dispatched by name in :func:`render_page` and everything else
#: must have a ``placeholders.PLACEHOLDERS`` entry, so a tab can be reordered
#: here without touching the rendering, and a name that belongs to neither
#: fails a test.
#:
#: 💰 DCF sits directly before ⚖️ Valuation because the two answer the same
#: question from opposite directions — intrinsic value from the cash flows, and
#: what the market is paying for them — so they read as a pair.
#:
#: 👤 Insiders follows that pair because it is the check on what the pair
#: concluded. The reader arrives having just formed a view of what the company
#: is worth and what it costs, and "what are the people with the best
#: information doing about it" is only a question once that view exists. Ahead
#: of the pair it is a curiosity; after Peers and History it reads as an
#: appendix.
#:
#: 🏛️ Holdings sits beside 👤 Insiders for the same reason those two sit where
#: they do: they are an ownership pair. One asks what the people *inside* the
#: company are doing with its stock, the other what the institutions *outside*
#: it are, and they share a shape — a position table, a history, and a
#: reconciliation of everything excluded — so a reader crossing between them
#: reads one form twice. Holdings comes second because a 13F register is context
#: for a view, not a substitute for one.
_TAB_ORDER = (
    "📊 Summary",
    "📄 Financials",
    "🎯 Earnings",
    "💵 Dividends",
    "💰 DCF",
    "⚖️ Valuation",
    "👤 Insiders",
    "🏛️ Holdings",
    "💰 Profitability",
    "🧭 Peers",
    "🕰️ History",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _session_reset(ticker: str) -> None:
    """Clears per-company state. The selected window survives a new search.

    Only this page's keys. The DCF tab's ``dcf_*`` state is left alone because
    that tab owns its own lifecycle: it notices the ticker has moved on its next
    render and clears itself. Reaching across from here would leave two places
    that decide when a model is stale.
    """
    for key in list(st.session_state.keys()):
        if key in {"research_output"}:
            del st.session_state[key]
    st.session_state["research_loaded_ticker"] = ticker


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _load_research_cached(ticker: str, as_of: date) -> research.ResearchDataset:
    """Caches the assembled dataset, not just the HTTP calls underneath.

    ``as_of`` is part of the key so a session left open overnight does not keep
    showing yesterday's close as today's.
    """
    return research.load_research(ticker, today=as_of)


def _needs_load(dataset: research.ResearchDataset | None, ticker: str) -> bool:
    """True when the snapshot in session state cannot serve ``ticker``.

    This is what makes the price windows free: a rerun triggered by the period
    control finds a snapshot that already matches and skips the fetch entirely.
    A snapshot for a *different* company is never reused, or a stale page would
    render under the new ticker's name.
    """
    return dataset is None or dataset.ticker != ticker


def _render_blocking(title: str, body: str, detail: str = "") -> None:
    st.error(f"**{title}**\n\n{body}")
    if detail:
        st.caption(detail)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def render_page() -> None:
    st.header("Stock Research")
    st.write(
        "Prices and fundamentals from Sharadar (Nasdaq Data Link). Every figure "
        "carries where it came from; anything that could not be obtained says so "
        "rather than showing a zero."
    )

    row = st.columns([3, 1.2, 2], vertical_alignment="bottom")
    with row[0]:
        ticker = st.text_input("Ticker", value="AAPL", key="research_ticker_input")
        ticker = ticker.strip().upper()
    with row[1]:
        search = st.button("Search", type="primary", width="stretch")

    if search and ticker:
        if st.session_state.get("research_loaded_ticker") != ticker:
            _session_reset(ticker)
        st.session_state["research_run"] = True

    if not st.session_state.get("research_run"):
        st.info(
            f"Press **Search** to load {ticker or 'a ticker'}. "
            "Nothing is fetched until you do."
        )
        return

    loaded = st.session_state.get("research_loaded_ticker") or ticker
    if loaded != ticker:
        st.warning(
            f"Showing **{loaded}**. Press **Search** to load {ticker}.", icon="⏳"
        )

    dataset = st.session_state.get("research_output")
    if _needs_load(dataset, loaded):
        try:
            with st.spinner(f"Loading {loaded} from Sharadar…"):
                dataset = _load_research_cached(loaded, date.today())
        except sharadar_client.TickerNotFoundError as error:
            _render_blocking(
                f"No Sharadar record for {loaded}",
                str(error),
                "This page covers US equities carried in Sharadar's SF1 "
                "fundamentals table. ETFs and funds have prices but no "
                "fundamentals, and are not supported yet.",
            )
            return
        except sharadar_client.SharadarError as error:
            _render_blocking(f"Sharadar request failed for {loaded}", str(error))
            return
        st.session_state["research_output"] = dataset

    # Keyed by name rather than sliced by index: the built tabs are not a
    # contiguous run at the front, and a reorder in `_TAB_ORDER` should not have
    # to be mirrored here.
    tabs = dict(zip(_TAB_ORDER, st.tabs(list(_TAB_ORDER))))

    with tabs["📊 Summary"]:
        summary.render(dataset)

    with tabs["📄 Financials"]:
        financials.render(dataset)

    with tabs["💰 DCF"]:
        # `loaded`, not the ticker box: the DCF values the company the rest of
        # this page is showing, not one half-typed into the search field.
        dcf.render(loaded)

    with tabs["⚖️ Valuation"]:
        valuation.render(dataset)

    with tabs["👤 Insiders"]:
        insiders.render(dataset)

    with tabs["🏛️ Holdings"]:
        holdings.render(dataset)

    for name, (what, source) in placeholders.PLACEHOLDERS.items():
        with tabs[name]:
            placeholders.render(name, what, source)
