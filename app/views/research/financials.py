"""The 📄 Financials tab: the four statements, sectioned and reconciled.

Fetches the annual and quarterly statements the first time it is opened,
through ``dataset.load_financials``; the TTM dimension arrives with the
snapshot, because Summary's tiles need it too. The layout itself
lives in ``app.logic.statements``, deliberately outside ``app.logic.research``
so that a second consumer needs no move.

This is the **only** statement viewer on the page. The DCF tab used to carry a
raw one of its own; it now points here and keeps only the filing links, which
this tab does not show.

The table is hand-rolled HTML rather than ``st.dataframe``. Bolded subtotals,
section header rows, indentation, per-cell tooltips, a derived marker and a
pinned first column are each impossible there — and the figures have to be
pre-formatted strings anyway, because this Streamlit build prints a missing
number as the word "None" in every numeric path tried: ``column_config``, a
Styler with ``na_rep``, nullable ``Float64``, and a bare frame alike. Once the
cells are strings, ``st.dataframe`` buys nothing.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from app.formatting import format_date
from app.logic import statements
from app.logic.research import dataset as research
from app.logic.research.dataset import FinancialsData, ResearchDataset
from app.provenance import Origin
from app.views.components import load_tab

_STATEMENT_CSS = """
<style>
/* ---- The statement table on the Financials tab ------------------------- */

/* Scrolls horizontally only. No `max-height`, so the container grows to the
   table's full height and the reader scrolls the page rather than a box
   inside it.
   `overflow-y` is deliberately left unset: CSS coerces it to `auto` anyway
   once `overflow-x` is `auto`, and since the container is as tall as its
   content that never produces a second scrollbar. */
.fin-scroll { overflow-x: auto; margin: 0.25rem 0 0.5rem 0; }

.fin {
    width: max-content; min-width: 100%;
    font-size: 0.85rem; font-variant-numeric: tabular-nums;
}
/* `border-collapse: separate` is required, not stylistic: with collapsed
   borders a `position: sticky` cell drops its borders while scrolling in
   WebKit, and the pinned label column loses its rule.
   The class is doubled to outrank Streamlit's own
   `.st-emotion-cache-<hash> table { border-collapse: collapse }`, which is one
   point more specific than a single class. The hash is build-specific, so it
   cannot be targeted directly — this is the stable way to win. */
.fin.fin { border-collapse: separate; border-spacing: 0; }
.fin td, .fin th { padding: 0.3rem 0.8rem; white-space: nowrap; }
.fin td { text-align: right; }

/* The pinned label column. The background must be opaque or the rows
   scrolling underneath show through it; `--background-color` is Streamlit's
   own theme variable, with its dark default as the fallback. */
.fin th.fin-label {
    position: sticky; left: 0; z-index: 2;
    text-align: left; font-weight: inherit;
    background: var(--background-color, #0e1117);
    border-right: 1px solid rgba(250, 250, 250, 0.14);
}
/* The header row is deliberately *not* sticky. The container scrolls only
   horizontally, so the page is the vertical scrollport and a `top: 0` here
   would resolve against a box that never scrolls — dead rules that look
   functional. The column headers scroll away with the page, as the rest of
   the page does. */
.fin thead th {
    background: var(--background-color, #0e1117);
    border-bottom: 1px solid rgba(250, 250, 250, 0.22);
    font-size: 0.74rem; font-weight: 600; text-align: right;
}
/* Keeps the corner above the period headers sliding under it. */
.fin th.fin-corner { z-index: 4; }
.fin .fin-p { display: block; }
.fin .fin-d { display: block; font-size: 0.68rem; opacity: 0.5; font-weight: 400; }
/* Separates the trailing-twelve-month column from the fiscal periods. */
.fin .fin-first { border-right: 1px solid rgba(250, 250, 250, 0.14); }

.fin-section th {
    padding-top: 1.1rem; font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.07em; opacity: 0.55;
}
.fin-subtotal { font-weight: 600; }
.fin-subtotal td, .fin-subtotal th { border-top: 1px solid rgba(250, 250, 250, 0.14); }
.fin-total { font-weight: 700; }
.fin-total td, .fin-total th {
    border-top: 2px solid rgba(250, 250, 250, 0.28);
    border-bottom: 1px solid rgba(250, 250, 250, 0.14);
}

/* Indentation by padding rather than by inserted spaces, so it survives
   escaping and copy-and-paste out of the table. */
.fin-i1 th.fin-label { padding-left: 1.9rem; }
.fin-i2 th.fin-label { padding-left: 3.1rem; }

/* Never reported. Kept visibly distinct from a reported zero, which is set in
   the same weight as every other figure — the two mean different things. */
.fin-missing { opacity: 0.32; font-style: italic; }
/* Reported as exactly 0 on an indicator Sharadar zero-fills, where 0 may mean
   "absent from the filing" rather than "nil". Marked, never hidden. */
.fin-zerofill { border-bottom: 1px dotted rgba(250, 250, 250, 0.45); cursor: help; }

.fin-tag {
    margin-left: 0.45rem; font-size: 0.62rem; letter-spacing: 0.06em;
    text-transform: uppercase; opacity: 0.5; cursor: help;
}
.fin-flag { margin-left: 0.4rem; color: #ffb74d; cursor: help; }
.fin-flagged td { background: rgba(255, 183, 77, 0.07); }
</style>
"""


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _load_cached(ticker: str, as_of) -> FinancialsData:
    """The tab's payload, per company per day — the key every tab's loader uses."""
    return research.load_financials(ticker)


def render(dataset: ResearchDataset) -> None:
    """The four statements, sectioned and reconciled.

    Loads the annual and quarterly dimensions on first open; after that the
    controls below re-slice frames already in the cache and fetch nothing.
    """
    st.markdown(_STATEMENT_CSS, unsafe_allow_html=True)
    st.subheader("Financials")

    data = load_tab(
        f"statements for {dataset.ticker}", _load_cached, dataset.ticker, dataset.loaded_at
    )
    if data is None:
        return

    if not data.annual and not data.quarterly:
        st.info(
            f"**Sharadar returned no fundamentals for {dataset.ticker}.** Nothing "
            "has been substituted. Prices and the summary statistics above are "
            "unaffected."
        )
        return

    controls = st.columns([3, 2, 2], vertical_alignment="bottom")
    with controls[0]:
        statement = st.radio(
            "Statement", list(statements.STATEMENTS), horizontal=True,
            key="research_fin_statement",
        )
    with controls[1]:
        period = st.radio(
            "Period", ["Annual", "Quarterly"], horizontal=True,
            key="research_fin_period",
        )
    with controls[2]:
        unit_name = st.radio(
            "Units", list(statements.UNITS), index=1, horizontal=True,
            key="research_fin_units",
            help="Display only. Per-share figures, share counts, margins and "
                 "multiples are never scaled — only money is.",
        )

    spec = statements.STATEMENTS[statement or "Income statement"]
    annual = (period or "Annual") == "Annual"

    show_empty = st.checkbox(
        f"Show the line items Sharadar left empty for {dataset.ticker}",
        value=False, key=f"research_fin_empty_{spec.key}",
        help="Indicators Sharadar carries for every company but reported no "
             "value for here. An empty row is not a zero — which items are "
             "missing is itself worth knowing.",
    )

    periodic = data.annual if annual else data.quarterly
    # The balance sheet's leading column reads MRQ: it is a point-in-time
    # snapshot, and Sharadar's TTM dimension only repeats the latest quarter-end
    # balance anyway.
    latest = (
        data.quarterly
        if spec.latest_source == "MRQ"
        else dataset.statements_ttm
    )

    table = statements.build_statement(
        spec,
        periodic=periodic,
        latest=latest,
        annual=annual,
        currency=dataset.currency,
        units=statements.UNITS[unit_name or "Millions"],
        # Every period Sharadar returned. This is a slice of a frame already in
        # the cache, so the cost is a wider table, never a fetch.
        max_periods=None,
        show_empty=show_empty,
    )

    if table.empty:
        st.info(
            f"Sharadar returned no {statement.lower()} for {dataset.ticker} at the "
            f"{'annual' if annual else 'quarterly'} dimension. Nothing has been "
            "substituted — try the other period."
        )
        return

    st.markdown(_statement_table_html(table), unsafe_allow_html=True)
    _render_statement_caption(table)
    _render_statement_sources(table)


def _statement_table_html(table: statements.StatementTable) -> str:
    """The whole table as one HTML string.

    One string with **no newlines in it**: Streamlit runs the markup through a
    Markdown parser before rendering, and a blank line inside would split the
    table into paragraphs. ``_render_stats`` above builds its table the same way
    and for the same reason.
    """
    width = len(table.periods)
    lead = " fin-first" if table.periods and table.periods[0].is_latest else ""

    corner = escape(f"{table.label}{table.units.suffix}")
    headers = [f'<th class="fin-label fin-corner">{corner}</th>']
    for index, column in enumerate(table.periods):
        classes = "fin-period" + (lead if index == 0 else "")
        headers.append(
            f'<th class="{classes}"><span class="fin-p">{escape(column.label)}</span>'
            f'<span class="fin-d">{escape(column.sublabel)}</span></th>'
        )

    body = []
    for section in table.sections:
        body.append(
            f'<tr class="fin-section"><th class="fin-label">'
            f"{escape(section.title)}</th>"
            f'<td colspan="{width}"></td></tr>'
        )
        body.extend(_statement_row_html(row, lead) for row in section.rows)

    colgroup = f'<colgroup><col class="fin-c-label"><col span="{width}"></colgroup>'
    return (
        f'<div class="fin-scroll"><table class="fin">{colgroup}'
        f"<thead><tr>{''.join(headers)}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _statement_row_html(row: statements.RenderedRow, lead: str) -> str:
    """One line item: pinned label, then a cell per period."""
    classes = ["fin-row", f"fin-{row.emphasis.value}", f"fin-i{row.indent}"]
    if row.origin is Origin.DERIVED:
        classes.append("fin-derived")
    if row.flagged:
        classes.append("fin-flagged")

    label = escape(row.label)
    if row.origin is Origin.DERIVED:
        tip = escape(row.formula or row.note or "Computed here, not a filed line item.")
        label += f'<span class="fin-tag" title="{tip}">derived</span>'
    elif row.note:
        label += f'<span class="fin-tag" title="{escape(row.note)}">note</span>'
    if row.flagged:
        label += f'<span class="fin-flag" title="{escape(row.reconcile.detail)}">≠</span>'

    cells = "".join(
        _statement_cell_html(cell, lead if index == 0 else "")
        for index, cell in enumerate(row.cells)
    )
    return (
        f'<tr class="{" ".join(classes)}">'
        f'<th class="fin-label">{label}</th>{cells}</tr>'
    )


def _statement_cell_html(cell: statements.Cell, lead: str) -> str:
    """One figure.

    Vendor strings reach the tooltips, so everything is escaped — the same rule
    ``_stat_cell`` states, extended: the row labels are ours, but the period
    labels, dates and plug reasons all trace back to Sharadar, and escaping
    uniformly is what keeps the guarantee when a field is added later.
    """
    if cell.missing:
        title = f' title="{escape(cell.reason)}"' if cell.reason else ""
        return f'<td class="fin-missing{lead}"{title}>{escape(cell.text)}</td>'
    if cell.zero_filled:
        tip = escape(
            "Sharadar writes 0 into this indicator both for a genuine nil and "
            "for a line item absent from the filing. The two cannot be told apart."
        )
        return (
            f'<td class="{lead.strip()}"><span class="fin-zerofill" title="{tip}">'
            f"{escape(cell.text)}</span></td>"
        )
    return f'<td class="{lead.strip()}">{escape(cell.text)}</td>'


def _render_statement_caption(table: statements.StatementTable) -> None:
    total = table.reported_rows + table.hidden_rows
    hidden = (
        f" · {table.hidden_rows} left empty by Sharadar and hidden"
        if table.hidden_rows
        else ""
    )
    st.caption(
        f"**{table.label}** · {table.dimension} · {table.reported_rows} of {total} "
        f"line items reported{hidden} · figures in "
        f"{table.units.label} of {table.currency or 'the reporting currency'}. "
        f"A **{statements.MISSING_TEXT}** is a figure Sharadar carries no value "
        f"for. A **0** with a dotted underline is an indicator Sharadar zero-fills, "
        f"where 0 may mean absent-from-the-filing rather than nil. Rows marked "
        f"**derived** are computed here — hover for the formula — and a **≠** "
        f"means the reported subtotal and the lines above it disagree."
    )


def _render_statement_sources(table: statements.StatementTable) -> None:
    """What was reconciled, and which filing each column came from.

    Every check is listed, passes included: the reader has to be able to see
    what was verified, not only what failed.
    """
    failed = sum(1 for result in table.diagnostics if result.failed)
    summary = f"Reconciliation and sources — {len(table.diagnostics)} checks"
    if failed:
        summary += f", {failed} failing"

    with st.expander(summary):
        if table.diagnostics:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Line item": result.row_label,
                            "Should equal": result.formula,
                            "Status": result.status,
                            "Periods checked": result.periods_checked,
                            "Largest gap": (
                                "—" if result.worst_gap is None
                                else f"{result.worst_gap:,.0f}"
                            ),
                            "In": result.worst_period or "—",
                            "Tolerance": f"{result.tolerance:.1%}",
                            "Notes": result.detail or "—",
                        }
                        for result in table.diagnostics
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Subtotals are shown **as the company reported them**, never as "
                "the sum of the rows above. Where both exist the two are compared "
                "here, and a disagreement marks the row rather than silently "
                "correcting it. A check whose inputs are missing reports *not "
                "evaluated* rather than passing."
            )
        if not table.meta.empty:
            st.markdown("**The filing behind each column**")
            filings = table.meta.copy()
            for name in ("Report Date", "Publish Date"):
                if name in filings.columns:
                    filings[name] = filings[name].apply(format_date, missing="—")
            st.dataframe(filings, width="stretch", hide_index=True)

