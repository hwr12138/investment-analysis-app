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

import pandas as pd
import streamlit as st

from app.formatting import format_pct
from app.logic.research import comparison as shared
from app.logic.research import growth as research
from app.logic.research.comparison import Comparison
from app.logic.research.dataset import ResearchDataset
from app.views.research.valuation import table


def render(dataset: ResearchDataset, peer_universe: pd.DataFrame) -> None:
    industry = shared.industry_of(peer_universe, dataset.ticker)
    comparisons = research.build_comparison(
        statements_ttm=dataset.statements_ttm,
        peer_universe=peer_universe,
        industry=industry,
        today=dataset.loaded_at,
    )

    table.render_heading(dataset, peer_universe, industry)
    st.markdown(
        table.comparison_table_html(comparisons, _verdict), unsafe_allow_html=True
    )
    _render_legend(comparisons)
    table.render_sources(
        dataset, peer_universe, industry, comparisons, legend=table.ORIGIN_LEGEND
    )


def _verdict(comparison: Comparison) -> str:
    """The gap in percentage points, with ``growth.verdict``'s word for it.

    Green is faster and red is slower throughout; the legend says why that is
    not a judgement for capex or working capital.
    """
    reading = research.verdict(comparison)
    if reading.gap is None:
        return table.missing()
    return table.verdict_cell(
        gap=reading.gap,
        text=f'{reading.gap * 100:+,.1f} pp<span class="val-sub">{reading.word}</span>',
        favourable=reading.favourable,
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
