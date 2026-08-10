"""The ⚖️ Valuation tab: each multiple against its industry and its own past.

Fetches nothing. ``st.tabs`` renders every tab body on every rerun, so this runs
whenever the reader touches a control on another tab; the peer universe arrives
once per search on the snapshot, and every number here is arithmetic over frames
already in memory.

The table answers one question per row — *is this expensive, and against what?*
— so the two comparisons sit side by side rather than on separate tabs. A stock
can be cheap against its industry and dear against its own history, and that
disagreement is the interesting case.

Nothing here fills a gap. A metric whose denominator has gone negative, and an
industry too small to have a median, both read "not reported" and carry the
reason in the sources panel below.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from app.logic.research import valuation as research
from app.logic.research.dataset import ResearchDataset
from app.logic.research.formatting import format_pct

#: Percentage widths for the comparison table. The metric name carries the
#: longest string by a distance; the three numeric columns are equal so the
#: eye can compare down them, and the two verdict columns are narrow.
_COLUMN_WIDTHS = (24, 15, 15, 15, 16, 15)

#: Above this the reading is called rich, below its reciprocal, cheap. A tenth
#: is wide enough that ordinary noise in a peer median does not flip the label
#: from one company to the next.
_MATERIAL_PREMIUM = 0.10

_VALUATION_CSS = """
<style>
/* `table-layout: fixed` holds the colgroup widths; without it the browser
   reflows the grid per company as the numbers change length. Doubled selector
   to outrank Streamlit's own hashed table rules. */
.val-table.val-table {
    width: 100%; border-collapse: collapse; table-layout: fixed;
    margin: 0.25rem 0 0.5rem 0; font-variant-numeric: tabular-nums;
}
.val-table th {
    font-size: 0.72rem; font-weight: 600; opacity: 0.6; text-align: right;
    padding: 0.35rem 0.7rem; border-bottom: 1px solid rgba(250, 250, 250, 0.22);
    text-transform: uppercase; letter-spacing: 0.04em;
}
.val-table th.val-metric, .val-table td.val-metric { text-align: left; }
.val-table td {
    padding: 0.5rem 0.7rem; text-align: right; vertical-align: middle;
    border-top: 1px solid rgba(250, 250, 250, 0.1); font-size: 0.9rem;
}
.val-table tr:last-child td { border-bottom: 1px solid rgba(250, 250, 250, 0.14); }
.val-metric { font-size: 0.85rem; font-weight: 600; }
.val-current { font-weight: 700; }
/* Missing values are stated, never blank: an empty cell reads as zero. */
.val-missing { font-weight: 400; font-style: italic; opacity: 0.42; font-size: 0.82rem; }
.val-sub { font-size: 0.7rem; opacity: 0.5; display: block; }
.val-rich { color: #f0857d; }
.val-cheap { color: #5fbf82; }
.val-flat { opacity: 0.6; }

/* The table has six columns of numbers and a long first column; on a narrow
   viewport it scrolls inside its own box rather than the page scrolling. */
.val-scroll { overflow-x: auto; }
.val-scroll .val-table { min-width: 620px; }
</style>
"""

def render(dataset: ResearchDataset) -> None:
    # Each tab injects its own CSS rather than the page injecting one shared
    # block, so the rules travel with the markup they style.
    st.markdown(_VALUATION_CSS, unsafe_allow_html=True)
    st.subheader("Valuation")

    industry = research.industry_of(dataset.peer_universe, dataset.ticker)
    comparisons = research.build_comparison(
        daily=dataset.daily,
        statements_ttm=dataset.statements_ttm,
        peer_universe=dataset.peer_universe,
        industry=industry,
        today=dataset.loaded_at,
    )

    _render_heading(dataset, industry, comparisons)
    _render_table(comparisons)
    _render_legend(comparisons)
    _render_sources(dataset, industry, comparisons)


def _render_heading(
    dataset: ResearchDataset, industry: str, comparisons: list[research.Comparison]
) -> None:
    """What the medians are taken over, stated before the numbers are shown."""
    if dataset.peer_universe.empty:
        st.warning(
            "The peer universe could not be loaded, so the industry medians are "
            "unavailable. The company's own current and five-year figures below "
            "are unaffected.",
            icon="⚠️",
        )
        return

    if not industry:
        st.warning(
            f"Sharadar publishes no industry classification for {dataset.ticker}, "
            "so there is no peer group to compare against. The company's own "
            "history below is unaffected.",
            icon="⚠️",
        )
        return

    peers = int((dataset.peer_universe["industry"] == industry).sum())
    as_of = _universe_date(dataset.peer_universe)
    st.markdown(
        f"**{escape(dataset.name)}** against **{escape(industry)}** — "
        f"{peers} listed companies{f', prices as of {as_of}' if as_of else ''}."
    )


def _comparison_table_html(comparisons: list[research.Comparison]) -> str:
    """The comparison grid, as one string with **no newlines in it**.

    Streamlit runs the markup through a Markdown parser before rendering, and a
    blank line inside would split the table into paragraphs. Every field is
    escaped — metric labels are the only strings here today, but escaping
    uniformly is what keeps the guarantee when a vendor-supplied column is added
    later.

    Split out from :func:`_render_table` so the markup can be asserted on
    without a Streamlit runtime, which is the only way the no-newlines rule
    stays enforced.
    """
    header = "".join(
        f'<th class="{cls}">{escape(name)}</th>'
        for name, cls in (
            ("Metric", "val-metric"),
            ("Current", ""),
            ("5Y median", ""),
            ("5Y percentile", ""),
            ("Industry median", ""),
            ("vs industry", ""),
        )
    )
    rows = "".join(_row(comparison) for comparison in comparisons)
    colgroup = "".join(f'<col style="width:{width}%">' for width in _COLUMN_WIDTHS)
    return (
        f'<div class="val-scroll"><table class="val-table">{colgroup}'
        f"<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _render_table(comparisons: list[research.Comparison]) -> None:
    st.markdown(_comparison_table_html(comparisons), unsafe_allow_html=True)


def _row(comparison: research.Comparison) -> str:
    metric = comparison.metric
    cells = [
        f'<td class="val-metric">{escape(metric.label)}</td>',
        f'<td class="val-current">{_value(comparison.current, metric)}</td>',
        f"<td>{_history(comparison)}</td>",
        f"<td>{_percentile(comparison)}</td>",
        f"<td>{_industry(comparison)}</td>",
        f"<td>{_premium(comparison)}</td>",
    ]
    return f"<tr>{''.join(cells)}</tr>"


def _value(value: float | None, metric: research.Metric) -> str:
    if value is None:
        return '<span class="val-missing">not reported</span>'
    return escape(metric.format(value))


def _history(comparison: research.Comparison) -> str:
    if comparison.history_median is None:
        return '<span class="val-missing">not reported</span>'
    return (
        f"{escape(comparison.metric.format(comparison.history_median))}"
        f'<span class="val-sub">{comparison.history_observations:,} sessions</span>'
    )


def _percentile(comparison: research.Comparison) -> str:
    """Where the current reading sits in the company's own five-year spread.

    The sub-label says *what the number counts* rather than reading it as a
    change: "83%" over "of its 5Y was lower" is unambiguous, where "83% richer
    than its own 5Y" would be a different and wrong claim. It is also the one
    phrasing that stays true for the dividend yield, where a higher reading is
    the better one.
    """
    if comparison.percentile is None:
        return '<span class="val-missing">not reported</span>'
    return (
        f"{format_pct(comparison.percentile, digits=0)}"
        '<span class="val-sub">of its 5Y was lower</span>'
    )


def _industry(comparison: research.Comparison) -> str:
    if comparison.industry_median is None:
        return '<span class="val-missing">not reported</span>'
    return (
        f"{escape(comparison.metric.format(comparison.industry_median))}"
        f'<span class="val-sub">n = {comparison.peer_count:,}</span>'
    )


def _premium(comparison: research.Comparison) -> str:
    """The current reading against the industry median, with a verdict.

    The verdict flips for the dividend yield: paying more than the industry is
    the cheap direction there, and colouring it red because the number is high
    would be exactly backwards.
    """
    premium = comparison.premium_to_industry
    if premium is None:
        return '<span class="val-missing">not reported</span>'

    if abs(premium) < _MATERIAL_PREMIUM:
        return f'<span class="val-flat">{format_pct(premium, digits=0)}<span class="val-sub">in line</span></span>'

    expensive = premium > 0 if not comparison.metric.higher_is_cheaper else premium < 0
    css = "val-rich" if expensive else "val-cheap"
    word = "richer" if expensive else "cheaper"
    sign = "+" if premium > 0 else ""
    return (
        f'<span class="{css}">{sign}{format_pct(premium, digits=0)}'
        f'<span class="val-sub">{word} than industry</span></span>'
    )


def _render_legend(comparisons: list[research.Comparison]) -> None:
    unavailable = sum(1 for c in comparisons if c.current is None)
    note = (
        "Medians, not means — a multiple whose denominator approaches zero prints "
        "in the thousands, and one such quarter moves an average but not a median. "
        "Non-positive readings are excluded everywhere: a negative P/E is a loss, "
        "not a cheap stock, and a zero yield is a non-payer rather than a payer of "
        "nothing. **5Y percentile** is the share of the last five years the company "
        "traded below where it does today."
    )
    if unavailable:
        note += (
            f" {unavailable} of {len(comparisons)} metrics have no current reading; "
            "the reason for each is in the panel below."
        )
    st.caption(note)


def _render_sources(
    dataset: ResearchDataset, industry: str, comparisons: list[research.Comparison]
) -> None:
    """Every metric's definition, and the reason behind every gap.

    Collapsed by default: provenance has to be reachable, not in the way. A cell
    reading "not reported" in the table above is never a dead end — its reason
    is a row in here.
    """
    missing = sum(
        1
        for c in comparisons
        if c.current is None or c.history_median is None or c.industry_median is None
    )
    summary = f"Sources & data status — {len(comparisons)} metrics"
    if missing:
        summary += f", {missing} with a figure unavailable"

    with st.expander(summary):
        frame = pd.DataFrame(
            [
                {
                    "Metric": c.metric.label,
                    "Status": c.origin.label,
                    "Definition": c.metric.definition,
                    "Notes": " · ".join(
                        filter(
                            None,
                            [c.current_reason, c.history_reason, c.industry_reason],
                        )
                    )
                    or "—",
                }
                for c in comparisons
            ]
        )
        # Widths are pinned to the widest value each column actually holds —
        # "Dividend Yield (TTM)" and "UNAVAILABLE" — rather than left to the
        # grid, which sizes to the longest string in the row and clips the two
        # short columns to fit two sentences of prose beside them.
        st.dataframe(
            frame,
            width="stretch",
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn(width=170),
                "Status": st.column_config.TextColumn(width=110),
                "Definition": st.column_config.TextColumn(width="large"),
                "Notes": st.column_config.TextColumn(width="large"),
            },
        )

        as_of = _universe_date(dataset.peer_universe) or "not loaded"
        st.caption(
            f"Peer group: Sharadar's `industry` classification"
            f"{f' — {escape(industry)}' if industry else ''}, "
            f"priced {as_of}. The classification is the vendor's own, not SIC; "
            "the two disagree, and the financial-sector gate elsewhere in this app "
            "deliberately uses SIC instead. · "
            "**REPORTED** is read straight off a Sharadar line item · "
            "**DERIVED** is arithmetic over reported figures · "
            "**UNAVAILABLE** could not be obtained and is never substituted."
        )


def _universe_date(peer_universe: pd.DataFrame) -> str:
    if peer_universe.empty or "date" not in peer_universe.columns:
        return ""
    stamp = pd.Timestamp(peer_universe["date"].max())
    return "" if pd.isna(stamp) else stamp.date().isoformat()
