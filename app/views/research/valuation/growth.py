"""The Growth sub-tab: each year-on-year rate against its industry and its past.

Fetches nothing — see the package docstring. The grid, the heading and the
sources panel come from :mod:`table`; what lives here is the faster-or-slower
verdict and the legend.

Two things read differently from the Valuation sub-tab beside it, and the
wording here has to carry both:

* **A negative reading is a real answer**, not an absent one. Revenue falling 8%
  prints as −8%, where a negative P/E prints as "not reported".
* **The gap to the industry is in percentage points**, not a ratio. Growing 14%
  where the industry grows 5% is nine points faster; calling it "180% faster"
  divides one rate by another and collapses entirely when the industry median is
  itself negative.
"""

from __future__ import annotations

import streamlit as st

from app.logic.research import comparison as shared
from app.logic.research import growth as research
from app.logic.research.comparison import Comparison
from app.logic.research.dataset import ResearchDataset
from app.logic.research.formatting import format_pct
from app.views.research.valuation import table

#: Percentage points. Below this the two rates are called level with each other
#: — a peer median built from a few dozen companies moves by more than this
#: between quarters, and a verdict that flips on that is noise.
_MATERIAL_GAP = 0.02


def render(dataset: ResearchDataset) -> None:
    industry = shared.industry_of(dataset.peer_universe, dataset.ticker)
    comparisons = research.build_comparison(
        statements_ttm=dataset.statements_ttm,
        peer_universe=dataset.peer_universe,
        industry=industry,
        today=dataset.loaded_at,
    )

    table.render_heading(dataset, industry)
    st.markdown(
        table.comparison_table_html(comparisons, _verdict), unsafe_allow_html=True
    )
    _render_legend(comparisons)
    table.render_sources(
        dataset, industry, comparisons, legend=table.ORIGIN_LEGEND
    )


def _verdict(comparison: Comparison) -> str:
    """The company's rate against the industry's, in percentage points.

    Green is faster and red is slower throughout, which is the reading a
    shareholder wants for revenue, earnings and cash flow. It is *not* a
    judgement for capex or working capital — growing those faster is neither
    good nor bad on its own — and the legend says so rather than the colour
    pretending otherwise.
    """
    gap = comparison.industry_gap
    if gap is None:
        return table.missing()

    points = f"{gap * 100:+,.1f} pp"
    if abs(gap) < _MATERIAL_GAP:
        return table.verdict_cell(
            gap=gap,
            text=f'{points}<span class="val-sub">in line</span>',
            favourable=None,
        )

    word = "faster" if gap > 0 else "slower"
    return table.verdict_cell(
        gap=gap,
        text=f'{points}<span class="val-sub">{word} than industry</span>',
        favourable=gap > 0,
    )


def _render_legend(comparisons: list[Comparison]) -> None:
    unavailable = sum(1 for c in comparisons if c.current is None)
    note = (
        "Trailing twelve months against the same period a year earlier. **A "
        "falling figure is shown** — it is the answer, not a gap. What is not "
        "shown is a rate measured from a **negative or zero base**: a loss that "
        "narrowed from −4.75 to −2.12 would print −55% and read as a collapse, so "
        "those say *not reported* with the reason below. CAPEX growth is taken on "
        "the magnitude, so a rise means the company spent more — for capex and "
        "working capital, faster is a fact rather than a verdict. **vs industry** "
        "is a difference in percentage points, not a ratio."
    )
    if unavailable:
        note += (
            f" {unavailable} of {len(comparisons)} metrics have no current reading; "
            "the reason for each is in the panel below."
        )
    st.caption(note)
