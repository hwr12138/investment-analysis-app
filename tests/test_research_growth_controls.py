"""The Growth sub-tab's rendering: the verdict wording and what a fall looks like.

The shell, the shared grid and the common cells are covered in
``test_research_valuation_controls``. What is tested here is only what Growth
words differently from the Valuation sub-tab beside it.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.logic.research import growth as logic
from app.logic.research.comparison import Comparison
from app.views.research.valuation import growth as view
from app.views.research.valuation import table

CONSUMER = "Consumer Electronics"


def _comparison(**kwargs) -> Comparison:
    kwargs.setdefault("metric", logic.METRICS[0])
    return Comparison(**kwargs)


def _verdict(current: float, median: float) -> str:
    return view._verdict(
        _comparison(current=current, industry_median=median, industry_gap=current - median)
    )


# ---------------------------------------------------------------------------
# The verdict column
# ---------------------------------------------------------------------------


def test_growing_faster_than_the_industry_reads_as_faster():
    cell = _verdict(0.14, 0.05)
    assert "val-cheap" in cell and "faster" in cell
    assert "+9.0 pp" in cell


def test_growing_slower_reads_as_slower():
    cell = _verdict(0.02, 0.20)
    assert "val-rich" in cell and "slower" in cell
    assert "-18.0 pp" in cell


def test_the_gap_is_stated_in_points_not_as_a_ratio():
    """14% against 5% is nine points faster, not '180% faster'."""
    cell = _verdict(0.14, 0.05)
    assert "pp" in cell
    assert "180" not in cell


def test_a_gap_against_a_negative_industry_median_still_renders():
    """The case a ratio could not express: the industry is shrinking."""
    cell = _verdict(0.05, -0.10)
    assert "+15.0 pp" in cell
    assert "faster" in cell


def test_a_small_gap_is_called_in_line():
    """A peer median of a few dozen companies moves by more than a point."""
    cell = _verdict(0.101, 0.10)
    assert "in line" in cell
    assert "val-cheap" not in cell and "val-rich" not in cell


def test_no_verdict_without_an_industry_median():
    assert "not reported" in view._verdict(_comparison(current=0.10))


# ---------------------------------------------------------------------------
# A falling figure is an answer, not a gap
# ---------------------------------------------------------------------------


def test_a_negative_growth_rate_renders_as_a_number():
    """The inverse of the Valuation sub-tab, where a negative reads as absent."""
    cell = table.value_cell(-0.08, logic.METRICS[0])
    assert "-8.00%" in cell
    assert "not reported" not in cell


def test_a_blocked_base_still_renders_as_missing():
    assert "not reported" in table.value_cell(None, logic.METRICS[0])


def test_every_growth_row_is_formatted_as_a_percentage():
    for metric in logic.METRICS:
        assert metric.format(0.125) == "12.50%", metric.label


# ---------------------------------------------------------------------------
# The legend
# ---------------------------------------------------------------------------


def test_the_legend_explains_the_base_rule_and_the_capex_sign():
    """The two things a reader cannot infer from the numbers on screen."""
    import inspect

    source = inspect.getsource(view._render_legend)
    assert "negative or zero base" in source
    assert "magnitude" in source
    assert "percentage points" in source


def test_the_history_cell_says_quarters_for_growth(peer_universe):
    """Not sessions — growth steps once per filing, and the column is shared."""
    periods = pd.date_range(start="2021-06-30", periods=20, freq="QE")
    caption = "Revenues"
    statements = {
        "PL": pd.DataFrame(
            {
                "Calendar Date": periods,
                "Publish Date": periods + pd.Timedelta(days=30),
                caption: [100.0 * (1.1**i) for i in range(20)],
            }
        )
    }
    comparison = [
        c
        for c in logic.build_comparison(
            statements_ttm=statements,
            peer_universe=peer_universe,
            industry=CONSUMER,
            today=date(2026, 8, 5),
        )
        if c.metric.label == "Revenue Growth (YoY)"
    ][0]
    assert "quarters" in table.history_cell(comparison)
    assert "sessions" not in table.history_cell(comparison)
