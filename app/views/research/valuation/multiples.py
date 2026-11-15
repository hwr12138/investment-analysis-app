"""The Valuation sub-tab: each multiple against its industry and its own past.

Fetches nothing — see the package docstring for why that matters, and
:mod:`table` for the grid, the heading and the sources panel it shares with
Growth. What lives here is only what is specific to a *level*: the metric list,
the rich-or-cheap verdict, and the legend.

The table answers one question per row — *is this expensive, and against what?*
— so the two comparisons sit side by side. A stock can be cheap against its
industry and dear against its own history, and that disagreement is the
interesting case.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.formatting import format_pct
from app.logic.research import comparison as shared
from app.logic.research import valuation as research
from app.logic.research.comparison import Comparison
from app.logic.research.dataset import ResearchDataset
from app.views.research.valuation import table


def render(dataset: ResearchDataset, peer_universe: pd.DataFrame) -> None:
    industry = shared.industry_of(peer_universe, dataset.ticker)
    comparisons = research.build_comparison(
        daily=dataset.daily,
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
    """The premium to the industry median, with ``valuation.verdict``'s word for it."""
    reading = research.verdict(comparison)
    if reading.gap is None:
        return table.missing()
    sign = "+" if reading.favourable is not None and reading.gap > 0 else ""
    return table.verdict_cell(
        gap=reading.gap,
        text=(
            f"{sign}{format_pct(reading.gap, digits=0)}"
            f'<span class="val-sub">{reading.word}</span>'
        ),
        favourable=reading.favourable,
    )


def _render_legend(comparisons: list[Comparison]) -> None:
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
