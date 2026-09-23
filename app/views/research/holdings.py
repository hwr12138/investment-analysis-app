"""The 🏛️ Holdings tab: who owns the company, from SEC form 13F.

**The one tab on this page that fetches only when asked.** Four Sharadar
requests and about ten seconds, against a tab most searches never open — and
``st.tabs`` renders every body on every rerun with no way to tell which is on
screen, so an eager fetch would bill every search on the page for a tab nobody
opened. That is the case CLAUDE.md reserves the button gate for, and this is the
first tab to actually use it.

Two things the tab says out loud rather than assuming the reader knows:

* **Nothing here describes a position held today.** 13F is quarterly with a
  45-day deadline, so the freshest register on file can be four months old. The
  banner states the arithmetic every time; it does not collapse.
* **The corrections are shown, not applied silently.** A manager's filing
  entities are summed before anything is ranked, and index managers are marked
  — both from curated lists that this tab prints, because both are judgements
  rather than facts the data carries.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.formatting import format_count, format_money, format_pct
from app.logic.research import holdings as research
from app.logic.research.dataset import HoldingsData, ResearchDataset

#: Session keys this tab owns. Cleared when the page's ticker moves, by this
#: module rather than by the shell — one place decides when the payload is stale,
#: the same division ``dcf._session_reset`` keeps.
_DATA_KEY = "holdings_data"
_TICKER_KEY = "holdings_ticker"

#: Percentage widths for the holder table's seven columns.
_HOLDER_WIDTHS = (24, 9, 13, 11, 17, 12, 14)

#: Sparkline geometry, in pixels. Fixed so every row's line occupies the same
#: box and the column can be scanned even though each is scaled to its own range.
_SPARK_W, _SPARK_H, _SPARK_PAD = 104, 22, 3

_UP = "#26a69a"
_DOWN = "#ef5350"

_HOLDINGS_CSS = """
<style>
/* Doubled class selectors: Streamlit's own hashed table rules outrank a single
   class, and the hash is build-specific so it cannot be targeted directly. */
.hold-heads.hold-heads {
    width: 100%; border-collapse: collapse; table-layout: fixed;
    margin: 0.25rem 0 0.75rem 0; font-variant-numeric: tabular-nums;
}
.hold-heads td {
    border-top: 1px solid rgba(250, 250, 250, 0.14);
    border-bottom: 1px solid rgba(250, 250, 250, 0.14);
    padding: 0.6rem 0.85rem; vertical-align: top;
}
.hold-heads td + td { border-left: 1px solid rgba(250, 250, 250, 0.14); }
.hold-k { font-size: 0.76rem; opacity: 0.65; display: block; }
.hold-v {
    font-size: 1.15rem; font-weight: 600; display: block;
    margin: 0.15rem 0 0.1rem 0; overflow-wrap: anywhere;
}
.hold-sub { font-size: 0.74rem; opacity: 0.6; display: block; }
.hold-missing {
    font-weight: 400; font-style: italic; opacity: 0.45; font-size: 0.9rem;
}
.hold-up { color: #5fbf82; }
.hold-down { color: #f0857d; }
.hold-flat { opacity: 0.7; }

.hold-table.hold-table {
    width: 100%; border-collapse: collapse; table-layout: fixed;
    margin: 0.25rem 0 0.5rem 0; font-variant-numeric: tabular-nums;
}
.hold-table th {
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em;
    opacity: 0.55; text-align: right; padding: 0.35rem 0.6rem; font-weight: 600;
}
.hold-table th:first-child, .hold-table th:nth-child(2) { text-align: left; }
.hold-table td {
    border-top: 1px solid rgba(250, 250, 250, 0.10);
    padding: 0.34rem 0.6rem; font-size: 0.86rem; text-align: right;
}
.hold-table td:first-child, .hold-table td:nth-child(2) { text-align: left; }
.hold-table td:nth-child(2) { font-size: 0.72rem; opacity: 0.6; }
.hold-table tfoot td {
    border-top: 2px solid rgba(250, 250, 250, 0.22);
    font-weight: 600; opacity: 0.85;
}
.hold-spark { vertical-align: middle; }
.hold-scroll { overflow-x: auto; }
.hold-scroll .hold-table { min-width: 720px; }

.hold-gate {
    border: 1px dashed rgba(250, 250, 250, 0.22); border-radius: 0.4rem;
    padding: 0.9rem 1.1rem; margin: 0.5rem 0 0.75rem 0;
}

@media (max-width: 760px) {
    .hold-heads.hold-heads, .hold-heads tbody, .hold-heads tr { display: block; }
    .hold-heads col { width: auto !important; }
    .hold-heads td {
        display: block; width: 100%; box-sizing: border-box;
        border-left: none !important;
    }
}
</style>
"""


def render(dataset: ResearchDataset) -> None:
    # Each tab injects its own CSS rather than the page injecting one shared
    # block: the rules travel with the markup they style.
    st.markdown(_HOLDINGS_CSS, unsafe_allow_html=True)

    if st.session_state.get(_TICKER_KEY) != dataset.ticker:
        _session_reset(dataset.ticker)

    data = st.session_state.get(_DATA_KEY)
    if data is None:
        _render_gate(dataset)
        return

    tolerances = research.HoldingsTolerances()
    built = research.build(
        summary=data.summary,
        current=data.current,
        history=data.history,
        names=data.names,
        actions=dataset.actions,
        statements=dataset.statements_quarterly.get("DERIVED", pd.DataFrame()),
        tolerances=tolerances,
        today=dataset.loaded_at,
    )

    if not built.ok:
        st.info(
            f"**No 13F positions on file for {dataset.ticker}.** "
            f"{built.holders.reason}",
            icon="🏛️",
        )
        return

    _render_banner(built, dataset.loaded_at)
    _render_headline(built, dataset.currency)
    _render_legend()

    st.plotly_chart(
        _ownership_figure(built.ownership),
        width="stretch",
        key="research_holdings_ownership",
    )

    st.divider()
    _render_holders(built, dataset.currency, tolerances)

    st.divider()
    _render_churn(built.churn, built, dataset.currency, tolerances)

    _render_sources(built, data, tolerances)


def _session_reset(ticker: str) -> None:
    """Drops the previous company's register.

    Owned here, as the DCF tab owns its ``dcf_*`` keys: this tab notices on its
    next render that the page's ticker has moved and clears itself. The page's
    own ``_session_reset`` deliberately does not reach across, so there is one
    place that decides when this payload is stale.
    """
    st.session_state.pop(_DATA_KEY, None)
    st.session_state[_TICKER_KEY] = ticker


def _render_gate(dataset: ResearchDataset) -> None:
    """What the tab shows before anyone asks for the data.

    States the cost and — the part that matters — **which sections it fills**,
    so a reader can tell a page that has not been asked to load from one that
    loaded and found nothing.
    """
    st.markdown(
        '<div class="hold-gate">'
        "<b>Institutional holders are loaded on request.</b><br>"
        "Four Sharadar requests, about ten seconds: every 13F filer holding "
        f"{escape(dataset.ticker)} in the two most recent quarters, thirteen "
        "years of ownership history, and the quarter-by-quarter position of the "
        "largest holders. Cached for the rest of the day.<br>"
        "<span style='opacity:0.7'>It fills the ownership chart, the holder "
        "table and its history, the new/closed/added/trimmed counts, and the "
        "concentration figures.</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    if st.button("Load institutional holders", key="holdings_load", type="primary"):
        from app.logic.research import dataset as research_dataset

        try:
            with st.spinner(f"Loading 13F filings for {dataset.ticker}…"):
                st.session_state[_DATA_KEY] = research_dataset.load_holdings(
                    dataset.ticker, today=dataset.loaded_at
                )
        except Exception as error:  # noqa: BLE001 - surfaced, never swallowed
            # The gate stays up rather than a half-filled table going out.
            st.error(f"**Could not load 13F filings.** {error}")
            return
        st.rerun()


# ---------------------------------------------------------------------------
# Banner and headline
# ---------------------------------------------------------------------------


def _render_banner(built: research.Holdings, today) -> None:
    """The staleness notice, which is unconditional because the fact is.

    Never collapsed and never conditional: a 13F register is *always* historical,
    and a reader who takes the table for a current cap table has been misled by
    the tab rather than by the data.
    """
    quarter = built.as_of.date() if built.as_of is not None else None
    if quarter is None:
        return
    deadline = built.as_of + pd.Timedelta(days=research.FILING_LAG_DAYS)
    age = built.days_old(today) or 0
    if age < 0:
        # Read before the deadline: managers are still filing, so the register is
        # not merely old, it is incomplete — a different caution and a stronger
        # one, since holders are missing rather than merely stale.
        tail = (
            f"and filers have until {deadline.date()} to report it, so the "
            f"register is still filling in — holders are missing from it today"
        )
    else:
        tail = (
            f"and the deadline to report it passed {age} days ago, so nothing "
            "here describes a position held today"
        )
    st.warning(
        f"**Positions as at {quarter}.** 13F is reported quarterly with a "
        f"{research.FILING_LAG_DAYS}-day filing deadline, {tail}.",
        icon="🏛️",
    )


def _render_headline(built: research.Holdings, currency: str) -> None:
    latest = _latest_ownership(built.ownership)
    cells = [
        _headline_cell(
            "Reported institutional ownership",
            format_pct(latest.get("owned")) if latest.get("owned") is not None
            else None,
            f"{format_count(latest.get('units'))} shares"
            if latest.get("units") is not None
            else "",
        ),
        _headline_cell(
            "Filers holding",
            format_count(latest.get("holders")),
            f"{built.churn.opened} opened · {built.churn.closed} closed"
            if built.churn.ok
            else "",
        ),
        _headline_cell(
            f"Largest {research.HoldingsTolerances().concentration_top_n}",
            format_pct(built.top_share.value) if built.top_share.ok else None,
            "of reported institutional value",
        ),
        _headline_cell(
            "Reported value",
            format_money(built.holders.total_value, currency),
            f"across {format_count(built.holders.holders)} filers",
        ),
    ]
    colgroup = "".join('<col style="width:25%">' for _ in cells)
    st.markdown(
        f'<table class="hold-heads">{colgroup}<tr>{"".join(cells)}</tr></table>',
        unsafe_allow_html=True,
    )


def _headline_cell(label: str, value: str | None, sub: str) -> str:
    if not value or value == "not reported":
        body = '<span class="hold-v hold-missing">not reported</span>'
    else:
        body = f'<span class="hold-v">{escape(value)}</span>'
    tail = f'<span class="hold-sub">{escape(sub)}</span>' if sub else ""
    return f'<td><span class="hold-k">{escape(label)}</span>{body}{tail}</td>'


def _latest_ownership(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {}
    row = frame.iloc[-1]
    return {
        "owned": None if pd.isna(row.get("owned")) else float(row["owned"]),
        "units": None if pd.isna(row.get("units")) else float(row["units"]),
        "holders": None if pd.isna(row.get("holders")) else float(row["holders"]),
    }


def _render_legend() -> None:
    st.caption(
        "**Common stock reported on SEC form 13F.** Managers with under $100m "
        "in US equities do not file at all, so this is who *filed*, not who "
        "owns the company. Long positions only — a reported put is a long put, "
        "not short interest, so a manager's real exposure can be the opposite "
        "of what appears here. Readings above 100% are genuine and are not "
        "clamped: a lent share is reported by the lender and again by whoever "
        "bought it. Position changes are measured on **share counts**, never on "
        "value, which moves with the price whether or not anyone traded."
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _ownership_figure(frame: pd.DataFrame) -> go.Figure:
    """Reported ownership and the number of filers, on a shared quarter axis.

    Stacked rather than sharing one axis, because the interesting case is the
    two diverging: ownership flat while the filer count falls is consolidation,
    and one axis hides it.
    """
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
        vertical_spacing=0.06,
        subplot_titles=("Reported institutional ownership", "Filers holding"),
    )
    if not frame.empty:
        figure.add_trace(
            go.Scatter(
                x=frame["quarter"], y=frame["owned"], name="Owned",
                mode="lines", line=dict(color="#4fa3f7", width=1.6),
                # A quarter with no share count leaves a break rather than a
                # line drawn through a number that was never computed.
                connectgaps=False,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%}<extra></extra>",
            ),
            row=1, col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=frame["quarter"], y=frame["holders"], name="Filers",
                mode="lines", line=dict(color="rgba(150,170,200,0.8)", width=1.4),
                connectgaps=False,
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f} filers<extra></extra>",
            ),
            row=2, col=1,
        )
    figure.update_yaxes(tickformat=".0%", row=1, col=1)
    figure.update_layout(
        template="plotly_dark", height=420,
        margin=dict(t=40, b=10, l=10, r=10),
        showlegend=False, hovermode="x unified",
    )
    return figure


def _sparkline_svg(values: tuple[float | None, ...], colour: str) -> str:
    """One holder's position history as an inline SVG.

    Three rules, each of which is a correctness matter rather than a style one:

    * **Normalised to its own range**, because a shared scale draws BlackRock's
      1.03bn-to-1.16bn as a flat line and says nothing. The cost is that
      amplitude is no longer comparable between rows, which is why the table
      prints the quarter-on-quarter figure beside it and the caption says so.
    * **One polyline per contiguous run.** A polyline spanning a gap draws a
      straight segment across it, which is interpolation by omission — and a
      manager absent from a quarter did not hold a number between its
      neighbours. A single isolated reading draws a dot, since a one-point line
      has no length.
    * **A flat series draws through the middle** rather than dividing by zero.
    """
    finite = [(i, v) for i, v in enumerate(values) if v is not None]
    if not finite:
        return '<span class="hold-missing">no history</span>'

    span = max(len(values) - 1, 1)
    lo = min(v for _, v in finite)
    hi = max(v for _, v in finite)
    height = _SPARK_H - 2 * _SPARK_PAD

    def point(index: int, value: float) -> tuple[float, float]:
        x = _SPARK_PAD + index * (_SPARK_W - 2 * _SPARK_PAD) / span
        if hi == lo:
            return x, _SPARK_PAD + height / 2
        y = _SPARK_H - _SPARK_PAD - (value - lo) / (hi - lo) * height
        return x, y

    runs: list[list[tuple[int, float]]] = []
    for index, value in finite:
        if runs and index == runs[-1][-1][0] + 1:
            runs[-1].append((index, value))
        else:
            runs.append([(index, value)])

    parts = []
    for run in runs:
        coords = [point(i, v) for i, v in run]
        if len(coords) == 1:
            x, y = coords[0]
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.8" fill="{colour}"/>')
        else:
            drawn = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
            parts.append(
                f'<polyline points="{drawn}" fill="none" stroke="{colour}" '
                'stroke-width="1.4" stroke-linejoin="round"/>'
            )

    end_x, end_y = point(*finite[-1])
    parts.append(f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="2.1" fill="{colour}"/>')
    title = f"{lo:,.0f} to {hi:,.0f} shares over the window"
    return (
        f'<svg class="hold-spark" width="{_SPARK_W}" height="{_SPARK_H}" '
        f'viewBox="0 0 {_SPARK_W} {_SPARK_H}" role="img">'
        f"<title>{escape(title)}</title>{''.join(parts)}</svg>"
    )


# ---------------------------------------------------------------------------
# The holder table
# ---------------------------------------------------------------------------


def _render_holders(
    built: research.Holdings, currency: str, tolerances: research.HoldingsTolerances
) -> None:
    table = built.holders
    st.markdown(f"#### The {len(table.rows)} largest holders")
    if not table.ok:
        st.caption(table.reason)
        return

    st.markdown(_holders_html(table, currency), unsafe_allow_html=True)
    st.caption(
        f"Positions as at {table.as_of.date()}, over the last "
        f"{len(table.periods)} quarters. **Each line is scaled to its own "
        "range, so heights are not comparable between rows** — the change "
        "column carries the magnitude. A break in a line is a quarter the "
        "manager filed nothing, which is not the same as holding nothing. "
        "Managers filing under several entities are summed; ⬤ marks an index "
        "manager, whose changes are fund flows rather than decisions."
    )


def _holders_html(table: research.HolderTable, currency: str) -> str:
    """One line of HTML. Streamlit Markdown-parses the string first, so a
    newline inside the table breaks it into paragraphs.

    Investor names are vendor free text and are escaped. The sparkline is the
    one field inserted as markup, and it is built entirely from formatted
    floats and an escaped title — see :func:`_sparkline_svg`.
    """
    colgroup = "".join(f'<col style="width:{w}%">' for w in _HOLDER_WIDTHS)
    header = (
        "<tr><th>Holder</th><th>Type</th><th>Shares</th><th>Δ quarter</th>"
        "<th>History</th><th>% of shares</th><th>Value</th></tr>"
    )
    body = "".join(_holder_row(holder, currency) for holder in table.rows)
    foot = (
        "<tfoot><tr>"
        f"<td>All {format_count(table.holders)} filers</td><td></td>"
        f"<td>{escape(format_count(table.total_units))}</td>"
        f"<td>{_change_cell(table.total_change)}</td>"
        "<td></td><td></td>"
        f"<td>{escape(format_money(table.total_value, currency))}</td>"
        "</tr></tfoot>"
    )
    return (
        f'<div class="hold-scroll"><table class="hold-table">{colgroup}'
        f"<thead>{header}</thead><tbody>{body}</tbody>{foot}</table></div>"
    )


def _holder_row(holder: research.Holder, currency: str) -> str:
    colour = _UP if (holder.change or 0) >= 0 else _DOWN
    kind = "⬤ index" if holder.is_index else ""
    return (
        "<tr>"
        f"<td>{escape(holder.name)}</td>"
        f"<td>{escape(kind)}</td>"
        f"<td>{escape(format_count(holder.units))}</td>"
        f"<td>{_change_cell(holder.change)}</td>"
        f"<td>{_sparkline_svg(holder.series, colour)}</td>"
        f"<td>{escape(format_pct(holder.owned) if holder.owned is not None else '—')}</td>"
        f"<td>{escape(format_money(holder.value, currency))}</td>"
        "</tr>"
    )


def _change_cell(change: float | None) -> str:
    if change is None:
        return '<span class="hold-missing">new</span>'
    if abs(change) < 0.0005:
        return '<span class="hold-flat">—</span>'
    tone = "hold-up" if change > 0 else "hold-down"
    return f'<span class="{tone}">{change:+.1%}</span>'


# ---------------------------------------------------------------------------
# Churn, concentration and sources
# ---------------------------------------------------------------------------


def _render_churn(
    churn: research.Churn,
    built: research.Holdings,
    currency: str,
    tolerances: research.HoldingsTolerances,
) -> None:
    st.markdown("#### What the register did last quarter")
    if not churn.ok:
        st.caption(churn.reason)
        return
    st.markdown(
        f"**{churn.opened}** opened · **{churn.closed}** closed · "
        f"**{churn.added}** added · **{churn.trimmed}** trimmed · "
        f"**{churn.unchanged}** unchanged"
    )
    hhi = f"{built.hhi.value:,.0f}" if built.hhi.ok else "not reported"
    st.caption(
        f"A move of less than {tolerances.material_change:.1%} counts as "
        "unchanged — 13F counts are rounded and a manager reinvesting a "
        f"dividend has not decided anything. Concentration: the largest "
        f"{tolerances.concentration_top_n} hold "
        f"{format_pct(built.top_share.value) if built.top_share.ok else '—'} of "
        f"reported value, Herfindahl index {hhi}."
    )


def _render_sources(
    built: research.Holdings,
    data: HoldingsData,
    tolerances: research.HoldingsTolerances,
) -> None:
    shown = len(built.holders.rows)
    with st.expander(
        f"Sources & data status — {shown} of {built.holders.holders} filers shown"
    ):
        st.markdown(
            "**Where the figures come from.** `SHARADAR/SF3` and `SF3A`, one row "
            "per manager per quarter from SEC form 13F, back to June 2013. Share "
            "counts arrive **as filed and are never restated for splits**, so "
            "every count here is put on today's basis before it is compared or "
            "divided — `SHARADAR/SF1`'s share count already is, and mixing the "
            "two would be wrong by the split ratio at every earlier quarter."
        )
        if not built.reconciliation.empty:
            st.dataframe(built.reconciliation, width="stretch", hide_index=True)

        st.markdown("**Two curated lists, applied to the table above.**")
        families = ", ".join(
            f"{label} ({len(members)} filing entities)"
            for label, members in research.FILER_FAMILIES.items()
        )
        st.caption(
            f"*Summed before ranking:* {families}. A manager reorganising its "
            "filing entities otherwise shows as a large sale in one and several "
            "new positions beside it — Apple's Vanguard position read −33% on "
            "that reorganisation and −6% once summed. Deriving this from the "
            "names was tried and abandoned: it matched BANK OF MARIN to BANK OF "
            "AMERICA. Any manager not named here stands alone."
        )
        st.caption(
            "*Marked as index managers:* "
            + ", ".join(sorted(set(research.INDEX_MANAGERS.values())))
            + ". Sharadar publishes no manager type, so this is a judgement and "
            "not a measurement. It is deliberately narrow — houses running both "
            "index and active money are left unmarked rather than guessed at, "
            "and an unmarked manager is not thereby a claim to be active."
        )
        if not built.ownership.empty:
            st.caption(
                f"*Ownership history:* {len(built.ownership)} quarters, "
                f"{int(built.ownership['owned'].notna().sum())} with a share "
                "count to divide by."
            )
        st.caption(
            "**DERIVED** is arithmetic over reported figures, formula shown · "
            "**UNAVAILABLE** could not be obtained and is never substituted. "
            "Ownership, concentration and every position change on this tab are "
            "DERIVED; the filer counts and reported values are as filed."
        )
