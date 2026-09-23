"""The 👤 Insiders tab: open-market Form 4 activity, against the price.

Fetches nothing — the filings arrive on the snapshot, and the price line is a
slice of the OHLCV history the page already has.

**The legend is not decoration on this tab.** Open-market purchases are roughly
one row in 120 live, and Apple went five years without a single one, so a tab
that showed only what it counts would look broken at exactly the companies a
reader is most likely to open first. Every excluded row is counted in the panel
at the foot, and the legend says why before the charts are read.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.formatting import format_count, format_money
from app.logic.research import insiders as research
from app.logic.research.dataset import ResearchDataset

#: Percentage widths for the roster's five columns. The owner and role carry the
#: longest strings; the code is two characters.
_ROSTER_WIDTHS = (30, 26, 18, 16, 10)

#: The buy/sell pair already used by the Summary tab's volume bars, so a green
#: bar means the same thing on both.
_BUY = "#26a69a"
_SELL = "#ef5350"

#: The largest marker on the event chart, in pixels. Areas are scaled against
#: this, so one very large trade sets the scale for the rest.
_MAX_MARKER_PX = 44

_INSIDERS_CSS = """
<style>
/* Doubled class selectors throughout: Streamlit's own hashed `table` rules
   outrank a single class, and the hash is build-specific so it cannot be
   targeted directly. */
.ins-heads.ins-heads {
    width: 100%; border-collapse: collapse; table-layout: fixed;
    margin: 0.25rem 0 0.75rem 0; font-variant-numeric: tabular-nums;
}
.ins-heads td {
    border-top: 1px solid rgba(250, 250, 250, 0.14);
    border-bottom: 1px solid rgba(250, 250, 250, 0.14);
    padding: 0.6rem 0.85rem; vertical-align: top;
}
.ins-heads td + td { border-left: 1px solid rgba(250, 250, 250, 0.14); }
.ins-k { font-size: 0.76rem; opacity: 0.65; display: block; }
.ins-v {
    font-size: 1.15rem; font-weight: 600; display: block;
    margin: 0.15rem 0 0.1rem 0; overflow-wrap: anywhere;
}
.ins-sub { font-size: 0.74rem; opacity: 0.6; display: block; }
.ins-missing { font-weight: 400; font-style: italic; opacity: 0.45; font-size: 0.9rem; }
.ins-buy { color: #5fbf82; }
.ins-sell { color: #f0857d; }
.ins-flat { opacity: 0.7; }
.ins-flag { font-size: 0.74rem; opacity: 0.75; display: block; margin-top: 0.2rem; }

.ins-roster.ins-roster {
    width: 100%; border-collapse: collapse; table-layout: fixed;
    margin: 0.25rem 0 0.5rem 0; font-variant-numeric: tabular-nums;
}
.ins-roster th {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em;
    opacity: 0.55; text-align: right; padding: 0.35rem 0.6rem; font-weight: 600;
}
.ins-roster th:first-child, .ins-roster th:nth-child(2) { text-align: left; }
.ins-roster td {
    border-top: 1px solid rgba(250, 250, 250, 0.10);
    padding: 0.38rem 0.6rem; font-size: 0.86rem; text-align: right;
}
.ins-roster td:first-child, .ins-roster td:nth-child(2) { text-align: left; }
.ins-roster td:nth-child(2) { font-size: 0.78rem; opacity: 0.7; }
.ins-scroll { overflow-x: auto; }
.ins-scroll .ins-roster { min-width: 560px; }

.ins-cluster {
    border-left: 3px solid #5fbf82; padding: 0.35rem 0 0.35rem 0.7rem;
    margin: 0.35rem 0; font-size: 0.86rem;
}
.ins-cluster-when { opacity: 0.6; font-size: 0.78rem; }

@media (max-width: 760px) {
    .ins-heads.ins-heads, .ins-heads tbody, .ins-heads tr { display: block; }
    .ins-heads col { width: auto !important; }
    .ins-heads td {
        display: block; width: 100%; box-sizing: border-box;
        border-left: none !important;
    }
}
</style>
"""


def render(dataset: ResearchDataset) -> None:
    # Each tab injects its own CSS rather than the page injecting one shared
    # block: the rules travel with the markup they style.
    st.markdown(_INSIDERS_CSS, unsafe_allow_html=True)

    tolerances = research.InsiderTolerances()
    frame = dataset.insiders

    if not research.usable(frame):
        st.info(
            f"**No insider filings for {dataset.ticker}.** Sharadar's "
            "`SHARADAR/SF2` covers SEC forms 3, 4 and 5 from 2008 onward. A "
            "company listed through that period with no rows has had nobody "
            "file — which is an answer, not a gap.",
            icon="👤",
        )
        return

    windows = research.trailing_windows(
        frame, tolerances=tolerances, today=dataset.loaded_at
    )
    _render_headline(windows, dataset.currency)
    _render_legend()

    st.plotly_chart(
        _events_figure(dataset, frame, tolerances),
        width="stretch",
        key="research_insider_events",
    )

    quarters = research.periodic_flow(
        frame, tolerances=tolerances, today=dataset.loaded_at
    )
    st.plotly_chart(
        _quarterly_flow_figure(quarters, dataset.currency),
        width="stretch",
        key="research_insider_flow",
    )
    st.caption(
        f"Quarters are bucketed on the **transaction** date, not the filing "
        f"date. The last bar covers a quarter that is still open. Monthly "
        f"buckets are not offered: they are 41–89% empty depending on the "
        f"company, and an empty bucket reads as missing data rather than as a "
        f"quiet month."
    )

    st.divider()
    _render_clusters(
        research.clusters(frame, tolerances=tolerances, today=dataset.loaded_at),
        tolerances,
        dataset.currency,
    )

    st.divider()
    roster = research.roster(frame, tolerances=tolerances, today=dataset.loaded_at)
    _render_roster(roster)

    _render_sources(dataset, windows, roster, tolerances)


# ---------------------------------------------------------------------------
# The headline row
# ---------------------------------------------------------------------------


def _render_headline(windows: list[research.InsiderWindow], currency: str) -> None:
    """Trailing net flow at three horizons.

    Hand-rolled rather than ``st.metric``: each cell carries a figure, a trade
    count and sometimes a concentration flag, and a metric holds one value and
    one delta.
    """
    cells = "".join(_headline_cell(window, currency) for window in windows)
    colgroup = "".join(f'<col style="width:{100 // len(windows)}%">' for _ in windows)
    st.markdown(
        f'<table class="ins-heads">{colgroup}<tr>{cells}</tr></table>',
        unsafe_allow_html=True,
    )


def _headline_cell(window: research.InsiderWindow, currency: str) -> str:
    label = f"Net open-market, {window.months}M"
    if not window.ok:
        body = (
            '<span class="ins-v ins-missing">not reported</span>'
            f'<span class="ins-sub">{escape(_shorten(window.net.detail))}</span>'
        )
        return f'<td><span class="ins-k">{escape(label)}</span>{body}</td>'

    net = window.net.value
    tone = "ins-buy" if net > 0 else "ins-sell" if net < 0 else "ins-flat"
    sign = "+" if net > 0 else ""
    value = f"{sign}{format_money(net, currency)}"
    sub = (
        f"{window.buy_trades} buy(s) · {window.sell_trades} sell(s) · "
        f"{window.participants} insider(s)"
    )
    flag = ""
    if window.concentration_note:
        flag = (
            f'<span class="ins-flag">⚠ {escape(window.top_owner)} is '
            f"{window.concentration:.0%} of the gross</span>"
        )
    return (
        f'<td><span class="ins-k">{escape(label)}</span>'
        f'<span class="ins-v {tone}">{escape(value)}</span>'
        f'<span class="ins-sub">{escape(sub)}</span>{flag}</td>'
    )


def _shorten(reason: str, limit: int = 120) -> str:
    """First sentence of a reason, for a cell that has room for one."""
    first = reason.split(". ")[0].strip()
    if len(first) > limit:
        first = first[: limit - 1].rstrip() + "…"
    return first if first.endswith((".", "…")) else first + "."


def _render_legend() -> None:
    st.caption(
        "**Open-market purchases (P) and sales (S) only.** Grants, option "
        "exercises and shares withheld for tax are excluded: they are "
        "compensation rather than a view on the price, and including them "
        "reverses the sign of the total at most large companies. Rows that "
        "restate a holding without reporting a trade are excluded too. Every "
        "excluded row is counted in **Sources & data status** at the foot of "
        "this tab. A 10% owner is labelled, never filtered — the code is what "
        "decides, not the role."
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _events_figure(
    dataset: ResearchDataset,
    frame: pd.DataFrame,
    tolerances: research.InsiderTolerances,
) -> go.Figure:
    """Open-market trades plotted where and when they happened.

    A line rather than candles: the subject is the events, and five years of
    candles at this width is a solid block.

    Markers sit at the price the insider actually paid, not at that day's close:
    for a ``P`` or an ``S`` the two agree inside the day's range, and snapping to
    the close would quietly redraw the trade.

    That price is put on the chart's **split-adjusted** basis first — see
    ``insiders.split_adjusted_price``. SEP's close is adjusted for splits and
    SF2's filed price never is, so at Tesla the November 2021 sales would sit at
    $981–$1,237 against a line reading $338–$410. The hover keeps the filed
    price beside it, because that is the number on the form.

    Restricted to ``P`` and ``S`` for a visual reason as well as an arithmetic
    one: an ``M`` exercise is priced at the strike, often a fraction of the
    market price, so plotting every code would scatter markers far below the
    line and imply a crash that never happened.

    Sizes use ``sizemode="area"``. Plotly's default scales the *diameter*, which
    exaggerates a large trade by the square of its value.
    """
    cutoff = pd.Timestamp(dataset.loaded_at) - pd.DateOffset(
        years=tolerances.lookback_years
    )
    figure = go.Figure()

    prices = dataset.ohlcv
    if not prices.empty and "Close" in prices.columns:
        window = prices[prices.index >= cutoff]
        figure.add_trace(
            go.Scatter(
                x=window.index,
                y=window["Close"],
                name="Close",
                mode="lines",
                line=dict(color="rgba(150, 170, 200, 0.65)", width=1.2),
                hovertemplate="%{x|%Y-%m-%d}<br>Close %{y:,.2f}<extra></extra>",
            )
        )

    market = frame[research.is_open_market(frame)].copy()
    market["flow"] = research.signed_flow(market)
    market["when"] = pd.to_datetime(market["transactiondate"], errors="coerce")
    market = market[market["flow"].notna() & market["when"].notna()]
    market = market[market["when"] >= cutoff]

    if not market.empty:
        market["role"] = research.role_of(market)
        market["plotted"] = research.split_adjusted_price(market, dataset.actions)
        filed = market["transactionpricepershare"]
        # The two differ only across a split, so the "as filed" note is added
        # per row rather than always — on a company that never split it would be
        # the same number twice on every hover.
        market["price_label"] = [
            f"{plotted:,.2f}"
            if abs(plotted - raw) < 0.005
            else f"{plotted:,.2f} split-adjusted ({raw:,.2f} as filed)"
            for plotted, raw in zip(market["plotted"], filed)
        ]
        largest = market["flow"].abs().max()
        # sizeref against the largest trade in view, so the scale is set by what
        # is actually on screen rather than by an arbitrary constant.
        sizeref = 2.0 * largest / (_MAX_MARKER_PX**2) if largest else 1.0

        for code, name, colour in (
            (research.BUY_CODE, "Insider buy", _BUY),
            (research.SELL_CODE, "Insider sell", _SELL),
        ):
            side = market[market["transactioncode"] == code]
            if side.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=side["when"],
                    y=side["plotted"],
                    name=name,
                    mode="markers",
                    marker=dict(
                        color=colour,
                        size=side["flow"].abs(),
                        sizemode="area",
                        sizeref=sizeref,
                        sizemin=5,
                        line=dict(width=0.5, color="rgba(0,0,0,0.35)"),
                        opacity=0.8,
                    ),
                    customdata=side[
                        [
                            "ownername",
                            "role",
                            "transactionshares",
                            "flow",
                            "filingdate",
                            "price_label",
                        ]
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>%{customdata[1]}"
                        "<br>%{x|%Y-%m-%d} · %{customdata[2]:,.0f} shares"
                        " @ %{customdata[5]}"
                        "<br>Value %{customdata[3]:,.0f}"
                        "<br>Filed %{customdata[4]|%Y-%m-%d}<extra></extra>"
                    ),
                )
            )

    figure.update_layout(
        template="plotly_dark",
        height=460,
        margin=dict(t=30, b=10, l=10, r=10),
        # Point events, so "closest" — "x unified" would group unrelated trades
        # that merely share an x position with the price line.
        hovermode="closest",
        legend=dict(orientation="h", y=-0.14),
        yaxis_title=f"Price ({dataset.currency})" if dataset.currency else "Price",
    )
    # Without these a daily series draws a gap for every weekend.
    figure.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return figure


def _quarterly_flow_figure(quarters: pd.DataFrame, currency: str) -> go.Figure:
    """Net open-market dollars per quarter.

    Twenty-one bars rather than sixty-one. The bar is the **net**; gross bought
    and sold ride in the hover, because a quarter that nets zero on $50m each way
    is a different fact from a quarter with no trades and the bar alone cannot
    tell them apart.
    """
    colours = [_BUY if net >= 0 else _SELL for net in quarters["net"]]
    # The open quarter is drawn back so a short final bar is not read as a
    # collapse in activity when it is simply incomplete.
    opacity = [0.45 if partial else 0.9 for partial in quarters["partial"]]

    # Categorical labels, not timestamps: an empty quarter has to occupy its own
    # slot, and a date axis would space the bars by the gaps between trades.
    labels = pd.PeriodIndex(quarters["period"], freq="Q").strftime("%Y Q%q")

    figure = go.Figure(
        go.Bar(
            x=labels,
            y=quarters["net"],
            marker=dict(color=colours, opacity=opacity),
            customdata=quarters[["bought", "sold", "trades", "owners"]],
            hovertemplate=(
                "<b>%{x}</b><br>Net %{y:,.0f}"
                "<br>Bought %{customdata[0]:,.0f}"
                "<br>Sold %{customdata[1]:,.0f}"
                "<br>%{customdata[2]} trade(s), %{customdata[3]} insider(s)"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        template="plotly_dark",
        height=280,
        margin=dict(t=30, b=10, l=10, r=10),
        showlegend=False,
        yaxis_title=f"Net ({currency})" if currency else "Net",
        bargap=0.25,
    )
    return figure


# ---------------------------------------------------------------------------
# Clusters and the roster
# ---------------------------------------------------------------------------


def _render_clusters(
    found: list[research.InsiderCluster],
    tolerances: research.InsiderTolerances,
    currency: str,
) -> None:
    """Episodes of several insiders buying at once.

    Absence is stated with the criteria rather than left blank — "no clusters"
    and "we did not look" are different facts.
    """
    st.markdown("#### Cluster buying")
    criteria = (
        f"{tolerances.cluster_min_buyers} or more insiders each buying at least "
        f"{format_money(tolerances.cluster_min_value, currency)} on the open "
        f"market within {tolerances.cluster_window_days} days. Sales are "
        f"deliberately not clustered: 10b5-1 plans and vesting schedules make "
        f"executives sell together on dates chosen months earlier, so a sell "
        f"cluster measures the plan rather than a view."
    )
    if not found:
        st.caption(f"**None in the last {tolerances.lookback_years} years.** {criteria}")
        return

    for cluster in found:
        owners = ", ".join(cluster.owners)
        st.markdown(
            f'<div class="ins-cluster"><b>{cluster.buyers} insiders bought '
            f"{format_money(cluster.value, currency)}</b> across "
            f"{cluster.trades} trades"
            f'<span class="ins-cluster-when"> — {cluster.start} to '
            f"{cluster.end}</span><br>{escape(owners)}</div>",
            unsafe_allow_html=True,
        )
    st.caption(criteria)


def _render_roster(roster: research.InsiderRoster) -> None:
    """Share balances as last reported.

    The heading says "as last reported" and every row carries its own date,
    because SF2 records no departure event: there is no date on which it asserts
    a current position.
    """
    st.markdown("#### Shares held, as last reported")
    if not roster.ok:
        st.caption(roster.reason or "No share balances on file.")
        return

    st.markdown(_roster_html(roster.rows), unsafe_allow_html=True)
    note = (
        "Each figure is the balance on that insider's most recent filing, not a "
        "confirmed holding today. Option and warrant positions are excluded — "
        "they are counted in contracts, not shares."
    )
    st.caption(f"{note} {roster.reason}".strip())


def _roster_html(rows: pd.DataFrame) -> str:
    """One line of HTML. Streamlit Markdown-parses the string first, so a
    newline inside the table breaks it into paragraphs.

    Owner names, roles and codes are vendor free text — the only genuinely
    untrusted strings on this tab — so every cell is escaped.
    """
    colgroup = "".join(f'<col style="width:{width}%">' for width in _ROSTER_WIDTHS)
    header = (
        "<tr><th>Insider</th><th>Role</th><th>Shares</th>"
        "<th>As of</th><th>Last</th></tr>"
    )
    body = "".join(
        "<tr>"
        f"<td>{escape(str(row.owner))}</td>"
        f"<td>{escape(str(row.role))}</td>"
        f"<td>{escape(format_count(row.shares))}</td>"
        f"<td>{escape(pd.Timestamp(row.as_of).date().isoformat())}</td>"
        f"<td>{escape(str(row.last_code))}</td>"
        "</tr>"
        for row in rows.itertuples()
    )
    return (
        f'<div class="ins-scroll"><table class="ins-roster">{colgroup}'
        f"<thead>{header}</thead><tbody>{body}</tbody></table></div>"
    )


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _render_sources(
    dataset: ResearchDataset,
    windows: list[research.InsiderWindow],
    roster: research.InsiderRoster,
    tolerances: research.InsiderTolerances,
) -> None:
    """Every row that came back, and which bucket it fell in.

    This is the tab's answer to "why does this look empty". Without it a reader
    at a mega-cap sees two charts with almost nothing on them and cannot tell
    deliberate filtering from a broken fetch.
    """
    panel = research.reconciliation(
        dataset.insiders, tolerances=tolerances, today=dataset.loaded_at
    )
    counted = panel[panel["Category"] == "Open market, priced"]
    shown = int(counted["Rows"].iloc[0]) if not counted.empty else 0

    with st.expander(
        f"Sources & data status — {shown} of {len(dataset.insiders)} filed rows shown"
    ):
        st.markdown(
            "**Where the figures come from.** `SHARADAR/SF2`, one row per line "
            "item on an SEC form 3, 4 or 5. Dollar values are "
            "`transaction shares × price per share`, never the reported "
            "`transactionvalue` — Sharadar publishes that unsigned, so summing "
            "it counts every sale as a purchase, and it is absent on roughly "
            "half of all rows."
        )
        st.dataframe(panel, width="stretch", hide_index=True)

        unavailable = [w for w in windows if not w.ok]
        if unavailable:
            st.markdown("**Windows with no figure**")
            for window in unavailable:
                st.caption(f"*{window.months}M* — {window.net.detail}")
        if roster.reason:
            st.caption(f"*Roster* — {roster.reason}")

        st.caption(
            "**REPORTED** is read straight off a Sharadar line item · "
            "**DERIVED** is arithmetic over reported figures, formula shown · "
            "**UNAVAILABLE** could not be obtained and is never substituted. "
            "Every figure on this tab is DERIVED: no single SF2 column is the "
            "answer, and the column that looks like it is gets the sign wrong."
        )
