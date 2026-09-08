"""Stock research page controls: search gating and session-state lifecycle.

The page's two behavioural promises are that nothing is fetched before Search
is pressed, and that changing the price window never refetches. Both are
properties of the session-state handling rather than of any arithmetic, so they
are tested here rather than in ``test_stock_research``.

No Streamlit runtime exists in the suite, so ``st.session_state`` is replaced
with a dict subclass, as in ``test_acknowledgements``.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
from datetime import date

import pandas as pd
import pytest

from app.data import sharadar_client as sc
from app.logic import statements
from app.logic.research import dataset as research_dataset
from app.logic.research import summary as research_summary
from app.provenance import Origin
from app.views import research as page
from app.views.research import (
    dcf,
    financials,
    insiders,
    placeholders,
    summary,
    valuation,
)
from tests.conftest import SF1_COLUMNS, _ANNUAL_ROWS, _TTM_ROW, blank, sf1_payload


def _frames(rows=None) -> dict:
    built = sc._sf1_frames(
        sc.parse_datatable(sf1_payload(rows)), source_link="x", restated=True
    )
    return {name: sc._order_statement_rows(frame) for name, frame in built.items()}


class _SessionState(dict):
    """Stands in for ``st.session_state``, which needs a Streamlit runtime."""

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


@pytest.fixture
def session(monkeypatch) -> _SessionState:
    state = _SessionState()
    monkeypatch.setattr(page.st, "session_state", state)
    return state


def _dataset(ticker: str = "AAPL") -> research_dataset.ResearchDataset:
    index = pd.bdate_range("2025-01-01", "2026-08-05", name="Date")
    ohlcv = pd.DataFrame(
        {
            "Open": 100.0, "High": 102.0, "Low": 98.0,
            "Close": 101.0, "Adj Close": 101.0, "Volume": 1e6,
        },
        index=index,
    )
    return research_dataset.ResearchDataset(
        ticker=ticker,
        profile={"name": f"{ticker} INC", "currency": "USD"},
        ohlcv=ohlcv,
        daily=pd.DataFrame(),
        actions=pd.DataFrame(),
        insiders=pd.DataFrame(),
        statements_ttm={},
        statements_annual={},
        statements_quarterly={},
        quarterly_ar=pd.DataFrame(),
        stats=[],
        peer_universe=pd.DataFrame(),
        currency="USD",
        loaded_at=date(2026, 8, 5),
    )


# ---------------------------------------------------------------------------
# Search gating
# ---------------------------------------------------------------------------


def test_nothing_is_loaded_before_search_is_pressed(session):
    """The gate is a sticky flag, so an unpressed page holds no dataset."""
    assert not session.get("research_run")
    assert session.get("research_output") is None


def test_reset_clears_the_previous_company(session):
    session["research_output"] = _dataset("AAPL")
    session["research_loaded_ticker"] = "AAPL"

    page._session_reset("MSFT")

    assert "research_output" not in session
    assert session["research_loaded_ticker"] == "MSFT"


def test_reset_leaves_the_selected_window_alone(session):
    """The window is a viewing preference, not a property of the company."""
    session["research_period"] = "5Y"
    session["research_output"] = _dataset("AAPL")

    page._session_reset("MSFT")

    assert session["research_period"] == "5Y"


def test_reset_leaves_other_pages_state_alone(session):
    """Every page shares one session_state; only this page's keys may be cleared."""
    session["default_rf_rate"] = 4.2
    session["ps_rf_input"] = 3.9

    page._session_reset("MSFT")

    assert session["default_rf_rate"] == 4.2
    assert session["ps_rf_input"] == 3.9


def test_a_new_search_leaves_the_dcf_tab_to_clear_itself(session):
    """One place decides when a model is stale, and it is not this one.

    The DCF tab notices on its next render that the ticker has moved and runs
    its own reset. Clearing its keys from here as well would put that decision
    in two places, and they would drift.
    """
    session["dcf_acks"] = {"tv_share": "2026-07-31T09:00:00"}
    session["dcf_output"] = {"company": "AAPL"}
    session["dcf_loaded_ticker"] = "AAPL"

    page._session_reset("MSFT")

    assert session["dcf_acks"] == {"tv_share": "2026-07-31T09:00:00"}
    assert session["dcf_output"] == {"company": "AAPL"}
    assert session["dcf_loaded_ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# The snapshot is reused, not refetched
# ---------------------------------------------------------------------------


def test_changing_the_window_does_not_reload_the_company():
    """The whole point of fetching full history once: period buttons are free."""
    assert not page._needs_load(_dataset("AAPL"), "AAPL")


def test_a_snapshot_for_a_different_ticker_is_never_reused():
    """Stale state from the previous company must not render under a new one."""
    assert page._needs_load(_dataset("AAPL"), "MSFT")


def test_an_empty_session_always_loads():
    assert page._needs_load(None, "AAPL")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def test_every_placeholder_names_what_it_will_show_and_where_it_comes_from():
    for label, (what, source) in placeholders.PLACEHOLDERS.items():
        assert what.strip()
        assert "SHARADAR" in source or "sharadar" in source, label


#: The tabs this module renders itself, plus the one it delegates to the DCF
#: module. Everything else in `_TAB_ORDER` has to be a placeholder.
_BUILT = (
    "📊 Summary",
    "📄 Financials",
    "💰 DCF",
    "⚖️ Valuation",
    "👤 Insiders",
)


def test_the_built_tabs_are_not_placeholders():
    """Summary, Financials, DCF, Valuation and Insiders are built; the rest are not."""
    for built in _BUILT:
        assert built not in placeholders.PLACEHOLDERS, built
    # Still five: Insiders was never a placeholder, so nothing was promoted and
    # no unrelated entry should be deleted to keep this number moving.
    assert len(placeholders.PLACEHOLDERS) == 5


def test_every_tab_is_either_built_or_a_placeholder():
    """`render_page` indexes the tab dict by name, so an unclaimed name is a
    ``KeyError`` at render time and a placeholder with no slot never shows."""
    assert set(page._TAB_ORDER) == set(_BUILT) | set(placeholders.PLACEHOLDERS)
    assert len(page._TAB_ORDER) == len(set(page._TAB_ORDER))


def test_the_insiders_tab_follows_the_valuation_pair():
    """It is the check on what DCF and Valuation concluded, so it reads after
    them rather than among the company's own numbers."""
    order = list(page._TAB_ORDER)
    assert order[order.index("⚖️ Valuation") + 1] == "👤 Insiders"


def test_the_dcf_tab_sits_directly_before_the_valuation_tab():
    """Intrinsic value and what the market pays for it read as a pair."""
    order = list(page._TAB_ORDER)
    assert order[order.index("💰 DCF") + 1] == "⚖️ Valuation"


def test_summary_is_the_tab_that_opens():
    """Streamlit selects the first tab, so the order is the landing tab."""
    assert page._TAB_ORDER[0] == "📊 Summary"


def test_every_built_tab_module_exposes_render():
    """The shell dispatches by that name and nothing else."""
    for module in (summary, financials, dcf, valuation, insiders, placeholders):
        assert callable(getattr(module, "render", None)), module.__name__
    assert not hasattr(dcf, "render_page")


def test_the_dcf_tab_takes_the_ticker_rather_than_asking_for_one():
    """Two ticker boxes on one page is the thing this move exists to remove."""
    parameters = inspect.signature(dcf.render).parameters
    assert list(parameters) == ["ticker"]


def test_changing_a_financials_control_does_not_reload_the_company(session):
    """Statement, period and units are viewing preferences, not new companies."""
    session["research_fin_statement"] = "Balance sheet"
    session["research_fin_period"] = "Quarterly"
    session["research_fin_units"] = "Billions"

    assert not page._needs_load(_dataset("AAPL"), "AAPL")


def test_every_period_sharadar_returned_is_shown():
    """There is no period-count control: the table carries the full history.

    Guards the call site rather than a constant, because the failure this
    prevents is a truncating default silently creeping back in.
    """
    source = inspect.getsource(financials.render)
    assert "max_periods=None" in source


def test_the_table_grows_to_its_content_rather_than_scrolling_inside_a_box():
    """The reader scrolls the page vertically and the table horizontally.

    Comments are stripped first — this file explains at length why there is no
    `max-height`, and the explanation must not satisfy the assertion.
    """
    css = re.sub(r"/\*.*?\*/", "", financials._STATEMENT_CSS, flags=re.S)

    assert "max-height" not in css
    assert "overflow-x: auto" in css
    # A sticky header would resolve against a box that never scrolls.
    assert "position: sticky; top: 0" not in css
    # ...while the label column still pins as the periods scroll past it.
    assert "position: sticky; left: 0" in css


# ---------------------------------------------------------------------------
# The statement table
# ---------------------------------------------------------------------------


def _table(spec=None, rows=None, **kwargs):
    """A real StatementTable built from the shared SF1 fixture."""
    kwargs.setdefault("units", statements.UNITS["As reported"])
    kwargs.setdefault("latest", {})
    return statements.build_statement(
        spec or statements.INCOME_STATEMENT, periodic=_frames(rows), **kwargs
    )


def test_the_table_is_one_line_of_html():
    """Streamlit parses the string as Markdown first; a blank line splits it."""
    html = financials._statement_table_html(_table())
    assert "\n" not in html
    assert html.startswith('<div class="fin-scroll">')
    assert html.endswith("</table></div>")


def test_the_label_column_is_a_header_cell_so_it_can_pin():
    html = financials._statement_table_html(_table())
    assert '<th class="fin-label fin-corner">' in html
    assert '<th class="fin-label">Revenues</th>' in html


def test_subtotals_and_totals_carry_their_class():
    html = financials._statement_table_html(_table())
    assert "fin-row fin-subtotal" in html
    assert "fin-row fin-total" in html
    assert "fin-row fin-detail" in html


def test_section_headers_render_as_their_own_row():
    html = financials._statement_table_html(_table())
    assert '<tr class="fin-section"><th class="fin-label">Revenues</th>' in html
    assert "Operating Expenses &amp; Income" in html


def test_a_derived_row_is_marked_and_carries_its_formula_in_the_tooltip():
    html = financials._statement_table_html(_table())
    assert 'class="fin-tag" title="Total Operating Expenses' in html
    assert ">derived</span>" in html


def test_a_missing_figure_renders_a_dash_and_never_the_word_none():
    """This Streamlit build prints a missing number as "None" in every numeric path."""
    table = _table(
        statements.CASH_FLOW,
        rows=[blank(row, "sbcomp") for row in _ANNUAL_ROWS],
        show_empty=True,
    )
    html = financials._statement_table_html(table)

    assert "None" not in html
    assert 'class="fin-missing"' in html
    assert statements.MISSING_TEXT in html
    assert "Share-Based Compensation not reported" in html


def test_a_zero_sharadar_may_have_invented_is_marked():
    html = financials._statement_table_html(_table())
    assert 'class="fin-zerofill"' in html


def test_a_flagged_row_shows_why_without_changing_the_reported_figure():
    broken = []
    for row in _ANNUAL_ROWS:
        out = list(row)
        out[SF1_COLUMNS.index("cor")] = float(out[SF1_COLUMNS.index("cor")]) + 25.0
        broken.append(out)
    html = financials._statement_table_html(_table(rows=broken))

    assert "fin-flagged" in html
    assert 'class="fin-flag"' in html
    assert "Reported 500 vs computed 475" in html
    assert ">500</td>" in html  # the filed figure, not the computed one


def test_vendor_supplied_strings_are_escaped_not_trusted_as_markup():
    """Period labels, dates and plug reasons all trace back to Sharadar."""
    table = _table()
    hostile = statements.PeriodColumn(
        key="x", label="<img src=x onerror=alert(1)>", sublabel="\"><script>",
    )
    table = dataclasses.replace(table, periods=(hostile, *table.periods[1:]))
    html = financials._statement_table_html(table)

    assert "<img" not in html
    assert "&lt;img" in html
    assert "<script>" not in html


def test_the_ttm_column_is_ruled_off_from_the_fiscal_periods():
    table = _table(statements.CASH_FLOW, latest=_frames([list(_TTM_ROW)]))
    html = financials._statement_table_html(table)

    assert "fin-period fin-first" in html
    assert 'class="fin-first"' in html


# ---------------------------------------------------------------------------
# The statistics table
# ---------------------------------------------------------------------------


def test_the_table_is_four_columns_of_four():
    assert len(research_summary.STAT_COLUMNS) == 4
    assert all(len(column) == 4 for column in research_summary.STAT_COLUMNS)


def test_a_cell_shows_its_value_on_the_right_of_its_label():
    tile = research_summary.StatTile(
        label="P/E", display="35.20", origin=Origin.REPORTED, source="Sharadar DAILY"
    )
    cell = summary._stat_cell("P/E", tile)
    assert cell.index("P/E") < cell.index("35.20")


def test_a_missing_cell_says_not_reported_rather_than_rendering_blank():
    cell = summary._stat_cell("P/E", research_summary.StatTile.unavailable("P/E", "Sharadar", "gone"))
    assert "not reported" in cell
    # The reason belongs in the sources panel, not crammed into the cell.
    assert "gone" not in cell


def test_a_label_with_no_matching_tile_still_renders_a_cell():
    """A layout/tile mismatch must degrade, not raise, in front of the user."""
    assert "not reported" in summary._stat_cell("Nonexistent", None)


def test_vendor_values_are_escaped_not_trusted_as_markup():
    tile = research_summary.StatTile(
        label="X", display="<img src=x onerror=alert(1)>",
        origin=Origin.REPORTED, source="Sharadar",
    )
    cell = summary._stat_cell("X", tile)
    assert "<img" not in cell
    assert "&lt;img" in cell
