"""Stock research page — one ticker, nine tabs.

The general-purpose "what is this company" page: quote, chart, key statistics.
All Streamlit calls live here; the statistics themselves are built in
``app.logic.stock_research``, which is pure.

Two behaviours are deliberate and load-bearing:

* **Nothing is fetched until Search is pressed.** Same contract as the DCF
  page, for the same reason — a page that fetches on every keystroke of a
  ticker box burns the API quota on half-typed symbols.
* **Changing the price window never refetches.** The full history is fetched
  once per search and every window is a slice of it, so the period buttons are
  free.
"""

from __future__ import annotations

from datetime import date
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.data import sharadar_client
from app.logic import stock_research as research

#: Percentage widths for the four statistics columns. Unequal on purpose: the
#: price ranges and the dividend-with-yield cell carry the longest strings, the
#: multiples the shortest, and equal quarters truncated the first and last.
_COLUMN_WIDTHS = (28, 22, 22, 28)

#: Tabs, in display order. Only Summary is built; the rest name the Sharadar
#: table that will feed them so the next person does not have to re-derive it.
_PLACEHOLDERS = {
    "📄 Financials": (
        "Income statement, balance sheet and cash flow, annual and quarterly.",
        "`SHARADAR/SF1` — the 90 line items already captioned in "
        "`sharadar_client.SF1_LAYOUT`.",
    ),
    "🎯 Earnings": (
        "EPS and revenue history, surprises, and margins by quarter.",
        "`SHARADAR/SF1` at the `ARQ` dimension, which carries real filing dates.",
    ),
    "💵 Dividends": (
        "Full dividend and split history, yield over time, payout ratio.",
        "`SHARADAR/ACTIONS` via `sharadar_client.fetch_actions` — already "
        "fetched for the ex-dividend tile on Summary.",
    ),
    "⚖️ Valuation": (
        "P/E, P/S, P/B and EV/EBITDA bands over five years, with today's "
        "reading against the company's own history.",
        "`SHARADAR/DAILY` via `sharadar_client.fetch_daily_metrics`.",
    ),
    "🌱 Growth": (
        "Revenue, earnings and free-cash-flow growth, with CAGRs.",
        "`SHARADAR/SF1` annual and TTM.",
    ),
    "💰 Profitability": (
        "Margins, ROE, ROIC and turnover, against history.",
        "`SHARADAR/SF1` DERIVED frame — `grossmargin`, `netmargin`, `roe`, "
        "`roic`, `assetturnover`.",
    ),
    "🧭 Peers": (
        "Comparison against companies in the same industry.",
        "`SHARADAR/TICKERS` on `sicindustry`/`famaindustry`, then `DAILY` for "
        "the multiples.",
    ),
    "🕰️ History": (
        "Long-run total return, drawdowns, and volatility.",
        "`SHARADAR/SEP` `closeadj` — the dividend-adjusted total-return series.",
    ),
}

_RESEARCH_CSS = """
<style>
.res-profile { font-size: 0.85rem; opacity: 0.8; }

/* The 4x4 key-statistics table. `table-layout: fixed` holds the column widths
   set by the colgroup; without it the browser reflows the grid per company as
   values change length. The columns are deliberately unequal — the ranges and
   the dividend cell carry the longest strings, the multiples the shortest. */
.res-stats {
    width: 100%; border-collapse: collapse; table-layout: fixed;
    margin: 0.25rem 0 0.75rem 0;
}
.res-stats td {
    border-top: 1px solid rgba(250, 250, 250, 0.14);
    padding: 0.5rem 0.85rem;
    vertical-align: middle;
}
.res-stats td + td { border-left: 1px solid rgba(250, 250, 250, 0.14); }
.res-stats tr:last-child td { border-bottom: 1px solid rgba(250, 250, 250, 0.14); }
.res-cell {
    display: flex; justify-content: space-between; align-items: baseline; gap: 0.6rem;
}
/* When a cell is tight the label gives way first — a wrapped label still reads,
   a number split across two lines does not. */
.res-k { font-size: 0.76rem; opacity: 0.65; min-width: 0; }
/* Values wrap rather than truncate. A number cut off by an ellipsis is worse
   than no number at all, so the row grows instead. */
.res-v {
    font-size: 0.92rem; font-weight: 600; text-align: right;
    flex-shrink: 0; max-width: 68%; overflow-wrap: anywhere;
}
.res-v-missing { font-weight: 400; font-style: italic; opacity: 0.45; }

/* Below ~900px the four columns squeeze the labels into the values; stack to
   two so each cell keeps a readable label. */
@media (max-width: 900px) {
    .res-stats, .res-stats tbody, .res-stats tr { display: block; }
    .res-stats col { width: auto !important; }
    .res-stats td {
        display: inline-block; width: 50%; box-sizing: border-box;
        border-left: none !important;
    }
}
</style>
"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _session_reset(ticker: str) -> None:
    """Clears per-company state. The selected window survives a new search."""
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
    st.markdown(_RESEARCH_CSS, unsafe_allow_html=True)
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

    tab_names = ["📊 Summary", *_PLACEHOLDERS]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        _render_summary(dataset)

    for tab, (name, (what, source)) in zip(tabs[1:], _PLACEHOLDERS.items()):
        with tab:
            _render_placeholder(name, what, source)


# ---------------------------------------------------------------------------
# Summary tab
# ---------------------------------------------------------------------------


def _render_summary(dataset: research.ResearchDataset) -> None:
    _render_profile(dataset)

    period = st.segmented_control(
        "Window",
        options=research.PERIODS,
        default="1Y",
        key="research_period",
        label_visibility="collapsed",
    )
    # segmented_control returns None when the user clears the selection.
    period = period or "1Y"

    window = research.slice_period(dataset.ohlcv, period, dataset.loaded_at)
    if window.empty:
        st.warning(f"No price history in the {period} window.")
    else:
        _render_change(window, period, dataset.currency)
        st.plotly_chart(
            _price_figure(window, dataset), width="stretch", key="research_chart"
        )
        if period == "5D":
            st.caption(
                "Sharadar publishes end-of-day prices only — 5D is five trading "
                "sessions, not intraday."
            )

    st.divider()
    st.markdown("#### Key statistics")
    _render_stats(dataset.stats)
    _render_sources(dataset.stats)


def _render_profile(dataset: research.ResearchDataset) -> None:
    profile = dataset.profile
    bits = [
        str(profile.get(field) or "").strip()
        for field in ("exchange", "sector", "industry", "currency")
    ]
    line = " · ".join(b for b in bits if b)
    st.markdown(f"### {dataset.name} ({dataset.ticker})")
    if line:
        st.markdown(f'<span class="res-profile">{line}</span>', unsafe_allow_html=True)
    filings = str(profile.get("secfilings") or "").strip()
    site = str(profile.get("companysite") or "").strip()
    links = [f"[SEC filings]({filings})" if filings else "", f"[Website]({site})" if site else ""]
    links = [link for link in links if link]
    if links:
        st.caption(" · ".join(links))


def _render_change(window: pd.DataFrame, period: str, currency: str) -> None:
    change = research.period_change(window)
    last = window["Close"].dropna()
    if last.empty:
        return
    st.metric(
        label=f"Close · {period}",
        value=research.format_money(float(last.iloc[-1]), currency),
        delta=None if not change.ok else research.format_pct(change.value),
    )


def _price_figure(window: pd.DataFrame, dataset: research.ResearchDataset) -> go.Figure:
    """Candles over volume, sharing an x-axis.

    Weekends and holidays are removed with ``rangebreaks`` — without them a
    daily series draws a gap for every non-trading day and the candles drift
    apart from each other.
    """
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.78, 0.22],
        vertical_spacing=0.03,
    )

    figure.add_trace(
        go.Candlestick(
            x=window.index,
            open=window["Open"],
            high=window["High"],
            low=window["Low"],
            close=window["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    if "Volume" in window.columns:
        rising = window["Close"] >= window["Open"]
        figure.add_trace(
            go.Bar(
                x=window.index,
                y=window["Volume"],
                name="Volume",
                marker_color=[
                    "rgba(38,166,154,0.55)" if up else "rgba(239,83,80,0.55)"
                    for up in rising
                ],
                hovertemplate="%{x|%Y-%m-%d}<br>Volume %{y:,.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    figure.update_layout(
        height=620,
        template="plotly_dark",
        margin=dict(t=30, b=10, l=10, r=10),
        xaxis_rangeslider_visible=False,
        showlegend=False,
        hovermode="x unified",
    )
    figure.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    figure.update_yaxes(title_text=dataset.currency or "Price", row=1, col=1)
    figure.update_yaxes(title_text="Volume", row=2, col=1)
    return figure


def _render_stats(stats: list[research.StatTile]) -> None:
    """The 4×4 statistics table.

    Laid out from ``research.STAT_COLUMNS`` rather than from the order the
    tiles arrive in, so the grouping is declared in one place and a renamed
    tile fails a test instead of silently leaving a hole.

    Provenance is deliberately *not* in the cells — sixteen badges and source
    lines crowded out the numbers, which are what the table is for. It moves
    to the sources panel below, where every figure is still one click away
    from where it came from.
    """
    by_label = research.stats_by_label(stats)
    columns = research.STAT_COLUMNS
    rows = []
    for row_index in range(len(columns[0])):
        cells = [
            _stat_cell(column[row_index], by_label.get(column[row_index]))
            for column in columns
        ]
        rows.append(f"<tr>{''.join(cells)}</tr>")
    colgroup = "".join(f'<col style="width:{width}%">' for width in _COLUMN_WIDTHS)
    st.markdown(
        f'<table class="res-stats">{colgroup}{"".join(rows)}</table>',
        unsafe_allow_html=True,
    )


def _stat_cell(label: str, tile: research.StatTile | None) -> str:
    """One table cell: label left, value right.

    Vendor strings reach the value (currency codes, dates), so both sides are
    escaped — nothing from Sharadar is trusted as markup.
    """
    if tile is not None and tile.ok:
        value = f'<span class="res-v">{escape(tile.display)}</span>'
    else:
        value = '<span class="res-v res-v-missing">not reported</span>'
    return (
        f'<td><div class="res-cell">'
        f'<span class="res-k">{escape(label)}</span>{value}'
        f"</div></td>"
    )


def _render_sources(stats: list[research.StatTile]) -> None:
    """Where every figure on this tab came from, and whether it arrived.

    Collapsed by default: the accuracy mandate says provenance must always be
    reachable, not that it must always be in the way. Unavailable figures are
    listed here with their reason, so a blank cell in the table above is never
    a dead end.
    """
    missing = sum(1 for tile in stats if not tile.ok)
    summary = f"Sources & data status — {len(stats)} figures"
    if missing:
        summary += f", {missing} unavailable"

    with st.expander(summary):
        frame = pd.DataFrame(
            [
                {
                    "Statistic": tile.label,
                    "Value": tile.display if tile.ok else "not reported",
                    "Status": tile.origin.label,
                    "Source": tile.source,
                    "As of": tile.as_of.isoformat() if tile.as_of else "—",
                    "Notes": " · ".join(filter(None, [tile.formula, tile.detail])) or "—",
                }
                for tile in stats
            ]
        )
        st.dataframe(frame, width="stretch", hide_index=True)
        st.caption(
            "**REPORTED** is read straight off a Sharadar line item · "
            "**DERIVED** is arithmetic over reported figures, formula shown · "
            "**ESTIMATE** is an approximation and always labelled · "
            "**UNAVAILABLE** could not be obtained and is never substituted."
        )


# ---------------------------------------------------------------------------
# Placeholder tabs
# ---------------------------------------------------------------------------


def _render_placeholder(name: str, what: str, source: str) -> None:
    """A tab that is not built yet.

    Deliberately fetches nothing: ``st.tabs`` renders every tab body on every
    rerun, so a placeholder that loaded data would fire on page load and on
    every window change.
    """
    label = name.split(" ", 1)[-1]
    st.subheader(label)
    st.info(f"**Not built yet.** Planned: {what}")
    st.caption(f"Will be sourced from {source}")
