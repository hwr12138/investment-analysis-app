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

import streamlit as st

from app.formatting import format_pct
from app.logic.research import comparison as shared
from app.logic.research import valuation as research
from app.logic.research.comparison import Comparison
from app.logic.research.dataset import ResearchDataset
from app.views.research.valuation import table

#: Above this the reading is called rich, below its reciprocal, cheap. A tenth
#: is wide enough that ordinary noise in a peer median does not flip the label
#: from one company to the next.
_MATERIAL_PREMIUM = 0.10


def render(dataset: ResearchDataset) -> None:
    industry = shared.industry_of(dataset.peer_universe, dataset.ticker)
    comparisons = research.build_comparison(
        daily=dataset.daily,
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
    """The current reading against the industry median, as a premium.

    The judgement flips for the dividend yield: paying more than the industry is
    the *cheap* direction there, and colouring it red because the number is high
    would be exactly backwards.
    """
    premium = comparison.industry_gap
    if premium is None:
        return table.missing()

    if abs(premium) < _MATERIAL_PREMIUM:
        return table.verdict_cell(
            gap=premium,
            text=f'{format_pct(premium, digits=0)}<span class="val-sub">in line</span>',
            favourable=None,
        )

    expensive = premium > 0 if not comparison.metric.higher_is_cheaper else premium < 0
    sign = "+" if premium > 0 else ""
    word = "richer" if expensive else "cheaper"
    return table.verdict_cell(
        gap=premium,
        text=(
            f"{sign}{format_pct(premium, digits=0)}"
            f'<span class="val-sub">{word} than industry</span>'
        ),
        favourable=not expensive,
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
