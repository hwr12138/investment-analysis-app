"""The comparison grid both sub-tabs render, and the cells they share.

Four of the six columns say the same thing on either sub-tab — the metric's
name, its current reading, its five-year median and its industry median — so
they are built once here. The two that differ stay with the sub-tab that owns
them: what the *current* cell means when it is absent, and what the *verdict*
column calls a gap. A multiple is richer or cheaper; a growth rate is faster or
slower, and is measured in percentage points rather than as a ratio.

The CSS these classes refer to is injected once by the package ``__init__``,
because one tab means one stylesheet however many sub-tabs it holds.
"""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from app.logic.research.comparison import Comparison
from app.logic.research.dataset import ResearchDataset
from app.logic.research.formatting import format_pct

#: Percentage widths for the six columns. The metric name carries the longest
#: string by a distance; the three numeric columns are equal so the eye can
#: compare down them, and the verdict column is narrow.
COLUMN_WIDTHS = (24, 15, 15, 15, 16, 15)

#: The header row, identical on both sub-tabs — which is the point: a reader
#: moving between them is reading the same shape twice.
HEADERS = (
    ("Metric", "val-metric"),
    ("Current", ""),
    ("5Y median", ""),
    ("5Y percentile", ""),
    ("Industry median", ""),
    ("vs industry", ""),
)


def comparison_table_html(comparisons: list[Comparison], verdict) -> str:
    """The grid, as one string with **no newlines in it**.

    Streamlit runs the markup through a Markdown parser before rendering, and a
    blank line inside would split the table into paragraphs. Every field is
    escaped — metric labels are the only strings here today, but escaping
    uniformly is what keeps the guarantee when a vendor-supplied column is added
    later.

    ``verdict`` is the sub-tab's own last-column renderer, passed in rather than
    branched on, so neither sub-tab has to know the other exists.
    """
    header = "".join(
        f'<th class="{cls}">{escape(name)}</th>' for name, cls in HEADERS
    )
    rows = "".join(row_html(c, verdict) for c in comparisons)
    colgroup = "".join(f'<col style="width:{width}%">' for width in COLUMN_WIDTHS)
    return (
        f'<div class="val-scroll"><table class="val-table">{colgroup}'
        f"<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def row_html(comparison: Comparison, verdict) -> str:
    cells = [
        f'<td class="val-metric">{escape(comparison.metric.label)}</td>',
        f'<td class="val-current">{value_cell(comparison.current, comparison.metric)}</td>',
        f"<td>{history_cell(comparison)}</td>",
        f"<td>{percentile_cell(comparison)}</td>",
        f"<td>{industry_cell(comparison)}</td>",
        f"<td>{verdict(comparison)}</td>",
    ]
    return f"<tr>{''.join(cells)}</tr>"


def missing() -> str:
    """What an unobtainable figure looks like. Never a blank, never a zero."""
    return '<span class="val-missing">not reported</span>'


def value_cell(value: float | None, metric) -> str:
    if value is None:
        return missing()
    return escape(metric.format(value))


def history_cell(comparison: Comparison) -> str:
    """The five-year median, with how many observations are behind it.

    The count is not decoration. A valuation multiple is sampled every trading
    session and a growth rate once a quarter, so the same column carries
    medians over 1,255 readings and over 19 — and the reader has to be able to
    tell which they are looking at.
    """
    if comparison.history_median is None:
        return missing()
    return (
        f"{escape(comparison.metric.format(comparison.history_median))}"
        f'<span class="val-sub">{comparison.history_observations:,} '
        f'{escape(comparison.history_unit)}</span>'
    )


def percentile_cell(comparison: Comparison) -> str:
    """Where the current reading sits in the company's own five-year spread.

    The sub-label says *what the number counts* rather than reading it as a
    change: "83%" over "of its 5Y was lower" is unambiguous, where "83% richer
    than its own 5Y" would be a different and wrong claim. It is also the one
    phrasing that stays true on both sub-tabs, where high means dear in one and
    fast in the other.
    """
    if comparison.percentile is None:
        return missing()
    return (
        f"{format_pct(comparison.percentile, digits=0)}"
        '<span class="val-sub">of its 5Y was lower</span>'
    )


def industry_cell(comparison: Comparison) -> str:
    if comparison.industry_median is None:
        return missing()
    return (
        f"{escape(comparison.metric.format(comparison.industry_median))}"
        f'<span class="val-sub">n = {comparison.peer_count:,}</span>'
    )


def verdict_cell(*, gap: float | None, text: str, favourable: bool | None) -> str:
    """The shared shape of a last-column verdict: a signed figure and a word.

    ``favourable`` is the sub-tab's judgement, not this module's — ``None``
    paints it neutral, which is what an in-line gap gets.
    """
    if gap is None:
        return missing()
    if favourable is None:
        return f'<span class="val-flat">{text}</span>'
    css = "val-cheap" if favourable else "val-rich"
    return f'<span class="{css}">{text}</span>'


# ---------------------------------------------------------------------------
# The frame around the grid
# ---------------------------------------------------------------------------


def render_heading(dataset: ResearchDataset, industry: str) -> None:
    """What the medians are taken over, stated before the numbers are shown.

    Shared by both sub-tabs: they compare against the same peer group on the
    same day, and saying so twice in two different ways would invite the reader
    to wonder whether they differ.
    """
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
    as_of = universe_date(dataset.peer_universe)
    st.markdown(
        f"**{escape(dataset.name)}** against **{escape(industry)}** — "
        f"{peers} listed companies{f', priced {as_of}' if as_of else ''}."
    )


def render_sources(
    dataset: ResearchDataset,
    industry: str,
    comparisons: list[Comparison],
    *,
    legend: str,
) -> None:
    """Every metric's definition, and the reason behind every gap.

    Collapsed by default: provenance has to be reachable, not in the way. A cell
    reading "not reported" in the table above is never a dead end — its reason is
    a row in here. Shared by both sub-tabs, which differ only in the closing
    legend they pass in.
    """
    unavailable = sum(
        1
        for c in comparisons
        if c.current is None or c.history_median is None or c.industry_median is None
    )
    summary = f"Sources & data status — {len(comparisons)} metrics"
    if unavailable:
        summary += f", {unavailable} with a figure unavailable"

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
        # Widths are pinned to the widest value each column actually holds
        # rather than left to the grid, which sizes to the longest string in the
        # row and clips the two short columns to fit two sentences of prose.
        st.dataframe(
            frame,
            width="stretch",
            hide_index=True,
            column_config={
                "Metric": st.column_config.TextColumn(width=210),
                "Status": st.column_config.TextColumn(width=110),
                "Definition": st.column_config.TextColumn(width="large"),
                "Notes": st.column_config.TextColumn(width="large"),
            },
        )

        as_of = universe_date(dataset.peer_universe) or "not loaded"
        st.caption(
            f"Peer group: Sharadar's `industry` classification"
            f"{f' — {escape(industry)}' if industry else ''}, "
            f"priced {as_of}. The classification is the vendor's own, not SIC; "
            "the two disagree, and the financial-sector gate elsewhere in this app "
            f"deliberately uses SIC instead. · {legend}"
        )


def universe_date(peer_universe: pd.DataFrame) -> str:
    if peer_universe.empty or "date" not in peer_universe.columns:
        return ""
    stamp = pd.Timestamp(peer_universe["date"].max())
    return "" if pd.isna(stamp) else stamp.date().isoformat()


#: The provenance legend both sub-tabs close their sources panel with.
ORIGIN_LEGEND = (
    "**REPORTED** is read straight off a Sharadar line item · "
    "**DERIVED** is arithmetic over reported figures · "
    "**UNAVAILABLE** could not be obtained and is never substituted."
)
