"""The ⚖️ Valuation tab's rendering: the shell, the shared grid, and Valuation.

Imports the view modules directly and drives their builders, the same way
``test_research_page_controls`` does — there is no Streamlit runtime here, so
anything calling ``st`` is exercised through the pieces below it. The Growth
sub-tab's own wording lives in ``test_research_growth_controls``.
"""

from __future__ import annotations

import inspect
from datetime import date

import pandas as pd
import pytest

from app.logic.research import valuation as logic
from app.views.research import valuation as shell
from app.views.research.valuation import growth as growth_view
from app.views.research.valuation import multiples, table

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


def _html(comparisons) -> str:
    return table.comparison_table_html(comparisons, multiples._verdict)


# ---------------------------------------------------------------------------
# The shell
# ---------------------------------------------------------------------------


def test_the_tab_takes_the_snapshot():
    """The shell loads the peer universe through ``dataset.load_peers`` and
    hands it down; nothing in the package reaches for the client."""
    assert list(inspect.signature(shell.render).parameters) == ["dataset"]
    assert "load_tab(" in inspect.getsource(shell.render)
    for module in (shell, table, multiples, growth_view):
        source = inspect.getsource(module)
        assert "sharadar_client" not in source, module.__name__
        assert "fetch_" not in source, module.__name__


def test_both_sub_tabs_are_declared_in_order():
    """Valuation opens; Growth reads as the follow-up to it."""
    assert shell._SUB_TABS == ("Valuation", "Growth")


def test_every_sub_tab_module_exposes_render():
    for module in (multiples, growth_view):
        assert callable(getattr(module, "render", None)), module.__name__
        assert list(inspect.signature(module.render).parameters) == [
            "dataset", "peer_universe"
        ]


def test_the_peer_universe_is_keyed_on_dates_not_the_ticker():
    """One cache entry serves every company searched that day."""
    assert list(inspect.signature(shell._load_cached).parameters) == [
        "session", "as_of"
    ]


def test_only_the_open_sub_tab_is_built():
    body = inspect.getsource(shell.render)
    assert 'on_change="rerun"' in body
    assert body.count(".open:") == 2


def test_the_stylesheet_is_injected_once_by_the_shell():
    """One tab, one stylesheet, however many sub-tabs it holds."""
    assert "_VALUATION_CSS" in inspect.getsource(shell.render)
    for module in (multiples, growth_view):
        assert "<style>" not in inspect.getsource(module), module.__name__


def test_the_css_is_namespaced_to_this_tab():
    """Each tab owns its rules, so a dropped tab cannot orphan a shared block."""
    classes = {
        chunk.split("{")[0].strip()
        for chunk in shell._VALUATION_CSS.split(".")
        if chunk.strip().startswith("val-")
    }
    assert classes, "expected .val-* rules"


# ---------------------------------------------------------------------------
# The shared grid
# ---------------------------------------------------------------------------


def test_the_table_is_one_line_of_html(comparisons):
    """Streamlit parses the string as Markdown first; a blank line splits it."""
    assert "\n" not in _html(comparisons)


def test_the_table_scrolls_inside_its_own_box(comparisons):
    """Six numeric columns overflow a narrow viewport; the page must not."""
    html = _html(comparisons)
    assert html.startswith('<div class="val-scroll">')
    assert html.endswith("</table></div>")
    assert ".val-scroll { overflow-x: auto; }" in shell._VALUATION_CSS


def test_every_metric_gets_a_row(comparisons):
    html = _html(comparisons)
    for metric in logic.METRICS:
        assert metric.label in html, metric.label
    assert html.count("<tr>") == len(logic.METRICS) + 1  # + the header


def test_the_markup_is_escaped():
    """Vendor strings reach these cells, so nothing from Sharadar is markup."""
    hostile = logic.Metric(label="<script>x</script>", definition="d", daily_column="pe")
    row = table.row_html(
        _comparison(metric=hostile, current=1.0), multiples._verdict
    )
    assert "<script>" not in row
    assert "&lt;script&gt;" in row


def test_both_sub_tabs_render_the_same_column_headers():
    """A reader crossing between them is reading one shape twice."""
    assert [name for name, _ in table.HEADERS] == [
        "Metric", "Current", "5Y median", "5Y percentile",
        "Industry median", "vs industry",
    ]
    assert len(table.COLUMN_WIDTHS) == len(table.HEADERS)


# ---------------------------------------------------------------------------
# What a gap says
# ---------------------------------------------------------------------------


def _comparison(**kwargs):
    from app.logic.research.comparison import Comparison

    kwargs.setdefault("metric", logic.METRICS[0])
    return Comparison(**kwargs)


def test_a_missing_value_is_stated_rather_than_blank():
    """An empty cell reads as zero; 'not reported' does not."""
    cell = table.value_cell(None, logic.METRICS[0])
    assert "not reported" in cell and "val-missing" in cell


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
    cell = table.industry_cell(comparison)
    assert "not reported" in cell
    assert "0" not in cell.replace("val-missing", "")


def test_the_peer_count_travels_with_the_median(comparisons):
    """A median of six is not a median of six hundred, and the cell says which."""
    pe = [c for c in comparisons if c.metric.label == "P/E (TTM)"][0]
    assert "n = 6" in table.industry_cell(pe)


def test_the_history_cell_names_the_unit_it_counts(comparisons):
    """The same column carries 1,255 sessions and 19 quarters."""
    pe = [c for c in comparisons if c.metric.label == "P/E (TTM)"][0]
    assert "400 sessions" in table.history_cell(pe)


# ---------------------------------------------------------------------------
# The Valuation verdict
# ---------------------------------------------------------------------------


def _premium(label: str, current: float, median: float) -> str:
    metric = {m.label: m for m in logic.METRICS}[label]
    return multiples._verdict(
        _comparison(
            metric=metric,
            current=current,
            industry_median=median,
            industry_gap=current / median - 1.0,
        )
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
    assert "val-rich" in _premium("Dividend Yield (TTM)", 0.01, 0.02)


def test_a_small_gap_is_called_in_line():
    """Noise in a peer median must not flip the verdict company to company."""
    cell = _premium("P/E (TTM)", 20.5, 20.0)
    assert "in line" in cell
    assert "val-rich" not in cell and "val-cheap" not in cell


def test_no_verdict_without_an_industry_median():
    assert "not reported" in multiples._verdict(_comparison(current=20.0))
