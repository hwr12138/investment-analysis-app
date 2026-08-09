"""The 📊 Summary tab: profile, price chart, and the key-statistics grid.

Fetches nothing. ``st.tabs`` renders every tab body on every rerun, so this
runs whenever the reader touches a control on another tab; everything it needs
is already in the snapshot. Changing the price window is likewise free — the
full history arrives once per search and every window is a slice of it.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.logic.research import summary as research
from app.logic.research.dataset import ResearchDataset

#: Percentage widths for the four statistics columns. Unequal on purpose: the
#: price ranges and the dividend-with-yield cell carry the longest strings, the
#: multiples the shortest, and equal quarters truncated the first and last.
_COLUMN_WIDTHS = (28, 22, 22, 28)

_SUMMARY_CSS = """
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


def render(dataset: ResearchDataset) -> None:
    # Each tab injects its own CSS rather than the page injecting one shared
    # block: the rules travel with the markup they style, so a tab can be added
    # or dropped without anyone auditing a page-level stylesheet for orphans.
    st.markdown(_SUMMARY_CSS, unsafe_allow_html=True)
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


def _render_profile(dataset: ResearchDataset) -> None:
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


def _price_figure(window: pd.DataFrame, dataset: ResearchDataset) -> go.Figure:
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

