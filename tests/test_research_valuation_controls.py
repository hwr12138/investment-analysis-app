"""The ⚖️ Valuation tab's rendering: the markup contract and what a gap says.

Imports the view module directly and drives its private builders, the same way
``test_research_page_controls`` does — there is no Streamlit runtime here, so
anything calling ``st`` is exercised through the pieces below it.
"""

from __future__ import annotations

import inspect
from datetime import date

import pandas as pd
import pytest

from app.logic.research import valuation as logic
from app.views.research import valuation as view

CONSUMER = "Consumer Electronics"


@pytest.fixture
def comparisons(peer_universe):
    """The eight rows for the fixture company, with a flat 5Y price history."""
    daily = pd.DataFrame(
        {
            "date": pd.bdate_range(end="2026-08-05", periods=400),
            "pe": 20.0,
            "ps": 2.0,
            "pb": 4.0,
            "evebit": 15.0,
            "evebitda": 12.0,
            "ev": 2.2e9,
            "marketcap": 2.0e9,
        }
    )
    return logic.build_comparison(
        daily=daily,
        statements_ttm={},
        peer_universe=peer_universe,
        industry=CONSUMER,
        today=date(2026, 8, 5),
    )


# ---------------------------------------------------------------------------
# The markup contract
# ---------------------------------------------------------------------------


def test_the_tab_takes_the_snapshot(peer_universe):
    """It reads the peer universe off the snapshot; it must not fetch its own."""
    assert list(inspect.signature(view.render).parameters) == ["dataset"]
    source = inspect.getsource(view)
    assert "sharadar_client" not in source
    assert "fetch_" not in source


def test_the_table_is_one_line_of_html(comparisons):
    """Streamlit parses the string as Markdown first; a blank line splits it."""
    html = view._comparison_table_html(comparisons)
    assert "\n" not in html


def test_the_table_scrolls_inside_its_own_box(comparisons):
    """Six numeric columns overflow a narrow viewport; the page must not."""
    html = view._comparison_table_html(comparisons)
    assert html.startswith('<div class="val-scroll">')
    assert html.endswith("</table></div>")
    assert ".val-scroll { overflow-x: auto; }" in view._VALUATION_CSS


def test_every_metric_gets_a_row(comparisons):
    html = view._comparison_table_html(comparisons)
    for metric in logic.METRICS:
        assert metric.label in html, metric.label
    assert html.count("<tr>") == len(logic.METRICS) + 1  # + the header


def test_the_markup_is_escaped():
    """Vendor strings reach these cells, so nothing from Sharadar is markup."""
    hostile = logic.Metric(label="<script>x</script>", definition="d", daily_column="pe")
    row = view._row(logic.Comparison(metric=hostile, current=1.0))
    assert "<script>" not in row
    assert "&lt;script&gt;" in row


def test_the_css_is_namespaced_to_this_tab():
    """Each tab owns its rules, so a dropped tab cannot orphan a shared block."""
    classes = {
        chunk.split("{")[0].strip()
        for chunk in view._VALUATION_CSS.split(".")
        if chunk.strip().startswith("val-")
    }
    assert classes, "expected .val-* rules"


# ---------------------------------------------------------------------------
# What a gap says
# ---------------------------------------------------------------------------


def test_a_missing_value_is_stated_rather_than_blank():
    """An empty cell reads as zero; 'not reported' does not."""
    metric = logic.METRICS[0]
    assert "not reported" in view._value(None, metric)
    assert "val-missing" in view._value(None, metric)


def test_a_thin_industry_renders_as_missing_not_as_zero(peer_universe):
    comparison = [
        c
        for c in logic.build_comparison(
            daily=pd.DataFrame(),
            statements_ttm={},
            peer_universe=peer_universe,
            industry="Aluminum",
            today=date(2026, 8, 5),
        )
        if c.metric.label == "P/E (TTM)"
    ][0]
    cell = view._industry(comparison)
    assert "not reported" in cell
    assert "0" not in cell.replace("val-missing", "")


def test_the_peer_count_travels_with_the_median(comparisons):
    """A median of six is not a median of six hundred, and the cell says which."""
    pe = [c for c in comparisons if c.metric.label == "P/E (TTM)"][0]
    assert "n = 6" in view._industry(pe)


def test_the_history_cell_states_how_many_sessions(comparisons):
    pe = [c for c in comparisons if c.metric.label == "P/E (TTM)"][0]
    assert "400 sessions" in view._history(pe)


# ---------------------------------------------------------------------------
# The verdict column
# ---------------------------------------------------------------------------


def _premium(label: str, current: float, median: float) -> str:
    metric = {m.label: m for m in logic.METRICS}[label]
    return view._premium(
        logic.Comparison(metric=metric, current=current, industry_median=median)
    )


def test_a_high_multiple_reads_as_rich():
    cell = _premium("P/E (TTM)", 30.0, 20.0)
    assert "val-rich" in cell and "richer" in cell
    assert "+50%" in cell


def test_a_low_multiple_reads_as_cheap():
    cell = _premium("P/E (TTM)", 10.0, 20.0)
    assert "val-cheap" in cell and "cheaper" in cell


def test_a_high_dividend_yield_reads_as_cheap_not_rich():
    """The one metric where a bigger number is the better one.

    Colouring a 4% yield red against a 2% industry because the number is larger
    would be exactly backwards.
    """
    cell = _premium("Dividend Yield (TTM)", 0.04, 0.02)
    assert "val-cheap" in cell and "cheaper" in cell
    assert "val-rich" not in cell


def test_a_low_dividend_yield_reads_as_rich():
    cell = _premium("Dividend Yield (TTM)", 0.01, 0.02)
    assert "val-rich" in cell


def test_a_small_gap_is_called_in_line():
    """Noise in a peer median must not flip the verdict company to company."""
    cell = _premium("P/E (TTM)", 20.5, 20.0)
    assert "in line" in cell
    assert "val-rich" not in cell and "val-cheap" not in cell


def test_no_verdict_without_an_industry_median():
    metric = logic.METRICS[0]
    cell = view._premium(logic.Comparison(metric=metric, current=20.0))
    assert "not reported" in cell
