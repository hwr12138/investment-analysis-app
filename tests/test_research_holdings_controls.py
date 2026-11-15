"""``app/views/research/holdings.py`` — the load, the sparklines, the table.

Streamlit is never started. The view exposes pure string and figure builders and
the tests call those; ``render`` is only ever read with ``inspect``.

One thing here is not layout and matters more than the rest: the sparkline draws
a series with holes in it, where the difference between a gap and a zero is the
difference between "did not file" and "held nothing".
"""

from __future__ import annotations

import inspect
import re
from datetime import date

import pandas as pd
import pytest

from app.data import sharadar_client as sc
from app.logic.research import holdings as logic
from app.logic.research.dataset import HoldingsData, ResearchDataset
from app.views.research import holdings as view
from tests.conftest import sf3_splits

TODAY = date(2026, 8, 5)
Q = [pd.Timestamp(d) for d in (
    "2025-03-31", "2025-06-30", "2025-09-30",
    "2025-12-31", "2026-03-31", "2026-06-30",
)]


def _prose(function) -> str:
    """Every string literal in a function, joined.

    The wording these tests pin runs past 88 columns and is written as adjacent
    literals, so a substring check against raw source breaks on wherever the line
    happened to wrap.
    """
    import ast

    tree = ast.parse(inspect.getsource(function).lstrip())
    return " ".join(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


@pytest.fixture
def built(holders_frame, holdings_summary, investor_names):
    shares = pd.DataFrame(
        {
            "Calendar Date": Q,
            sc.SF1_CAPTIONS["sharesbas"]: [10_000_000.0] * 6,
            sc.SF1_CAPTIONS["sharefactor"]: [1.0] * 6,
        }
    )
    return logic.build(
        summary=holdings_summary,
        current=holders_frame,
        history=holders_frame,
        names=investor_names,
        actions=sf3_splits(),
        statements=shares,
        today=TODAY,
    )


# ---------------------------------------------------------------------------
# The dispatch contract
# ---------------------------------------------------------------------------


def test_the_tab_takes_the_snapshot():
    assert list(inspect.signature(view.render).parameters) == ["dataset"]


def test_the_tab_never_reaches_for_the_client_itself():
    """The one tab that fetches goes through ``logic.research.dataset``, as the
    DCF goes through ``dcf.model``. No view imports the client directly."""
    source = inspect.getsource(view)
    assert "sharadar_client" not in source
    assert "import requests" not in source


def test_the_stylesheet_is_injected_first():
    body = inspect.getsource(view.render)
    first = [line for line in body.splitlines() if "st." in line][0]
    assert "_HOLDINGS_CSS" in first


def test_the_css_is_namespaced_to_this_tab():
    for rule in view._HOLDINGS_CSS.split("{"):
        selector = rule.strip().splitlines()[-1].strip() if rule.strip() else ""
        if selector.startswith("."):
            assert selector.startswith(".hold-"), selector


# ---------------------------------------------------------------------------
# The load — the same as every other tab's
# ---------------------------------------------------------------------------


def test_the_register_loads_on_render_through_the_shared_helper():
    """No button: the page runs only the open tab, so a tab's body rendering is
    the reader asking for it."""
    body = inspect.getsource(view.render)
    assert "load_tab(" in body
    assert body.index("load_tab(") < body.index("research.build")
    assert "st.button" not in inspect.getsource(view)


def test_a_failed_load_draws_nothing_else():
    """``load_tab`` has already drawn the failure; a half-filled register must
    not follow it."""
    body = inspect.getsource(view.render)
    assert "if data is None:\n        return" in body


def test_the_payload_is_cached_per_company_per_day():
    assert list(inspect.signature(view._load_cached).parameters) == ["ticker", "as_of"]


def test_the_tab_keeps_nothing_in_session_state():
    """The cache is keyed on the company, so there is no payload to clear when
    the page's ticker moves — and no second place deciding when it is stale."""
    assert "session_state" not in inspect.getsource(view)


def test_the_share_count_comes_with_the_register():
    """Ownership divides by SF1's restated count, which now arrives with the
    tab's own payload rather than on the snapshot."""
    assert "statements=data.derived" in inspect.getsource(view.render)


# ---------------------------------------------------------------------------
# The sparkline
# ---------------------------------------------------------------------------


def test_a_gap_is_never_bridged():
    """A polyline spanning a missing point draws a straight segment across it,
    which is interpolation by omission — and a manager absent from a quarter did
    not hold a number between its neighbours."""
    svg = view._sparkline_svg((10.0, None, None, 40.0, 50.0), "#26a69a")
    assert len(re.findall(r"<polyline", svg)) == 1
    assert len(re.findall(r"<circle", svg)) == 2  # the isolated point, and the end


def test_two_runs_either_side_of_a_gap_draw_two_lines():
    svg = view._sparkline_svg((10.0, 20.0, None, 40.0, 50.0), "#26a69a")
    assert len(re.findall(r"<polyline", svg)) == 2


def test_an_isolated_reading_draws_a_dot_not_a_line():
    svg = view._sparkline_svg((None, 7.0, None), "#26a69a")
    assert "<polyline" not in svg
    assert "<circle" in svg


def test_a_flat_holder_draws_through_the_middle_rather_than_dividing_by_zero():
    """BlackRock's position barely moves; the row must still draw."""
    svg = view._sparkline_svg((5.0, 5.0, 5.0), "#26a69a")
    ys = {p.split(",")[1] for p in re.search(r'points="([^"]+)"', svg)[1].split()}
    assert len(ys) == 1
    assert "nan" not in svg.lower()


def test_a_holder_with_no_history_says_so():
    assert "no history" in view._sparkline_svg((None, None), "#26a69a")


def test_the_scale_is_per_row_and_the_caption_admits_it():
    """Normalising each line to its own range is what makes a near-flat holder
    readable, and it is also what makes heights incomparable between rows."""
    small = view._sparkline_svg((1.0, 2.0), "#26a69a")
    large = view._sparkline_svg((1_000.0, 2_000.0), "#26a69a")
    assert re.search(r'points="([^"]+)"', small)[1] == re.search(
        r'points="([^"]+)"', large
    )[1]
    assert "not comparable between rows" in _prose(view._render_holders)


def test_the_sparkline_carries_its_range_for_hover():
    """The amplitude the normalisation hides has to be recoverable."""
    svg = view._sparkline_svg((100.0, 900.0), "#26a69a")
    assert "<title>100 to 900 shares over the window</title>" in svg


def test_the_sparkline_never_emits_a_nan(built):
    for holder in built.holders.rows:
        svg = view._sparkline_svg(holder.series, "#26a69a")
        assert "nan" not in svg.lower()


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_the_table_is_one_line_of_html(built):
    assert "\n" not in view._holders_html(built.holders, "USD")


def test_investor_names_are_escaped_not_trusted_as_markup(built):
    import dataclasses

    hostile = dataclasses.replace(
        built.holders.rows[0], name="<img src=x onerror=alert(1)>"
    )
    table = dataclasses.replace(built.holders, rows=(hostile,))
    html = view._holders_html(table, "USD")
    assert "<img src=x" not in html
    assert "&lt;img" in html


def test_the_foot_row_carries_the_whole_register(built):
    """The general defence against a filer reorganisation the curated map does
    not name: a single holder's collapse is always read against the aggregate."""
    html = view._holders_html(built.holders, "USD")
    assert "<tfoot>" in html
    assert f"All {built.holders.holders}" in html.replace("&nbsp;", " ")


def test_an_index_manager_is_marked_in_the_table(built):
    html = view._holders_html(built.holders, "USD")
    assert "index" in html


def test_a_new_position_reads_as_new_rather_than_as_a_change():
    assert "new" in view._change_cell(None)
    assert "+" not in view._change_cell(None)


def test_an_unmoved_position_is_a_dash_not_a_zero_percent():
    assert view._change_cell(0.0) == '<span class="hold-flat">—</span>'


# ---------------------------------------------------------------------------
# Prose the reader depends on
# ---------------------------------------------------------------------------


def test_the_banner_states_the_filing_lag():
    prose = _prose(view._render_banner)
    assert "filing deadline" in prose
    assert "nothing" in prose and "held today" in prose


def test_the_legend_says_what_thirteen_f_does_not_contain():
    prose = _prose(view._render_legend)
    assert "$100m" in prose
    assert "not short interest" in prose
    assert "above 100%" in prose and "not" in prose
    assert "never on" in prose and "value" in prose


def test_the_sources_panel_prints_both_curated_lists():
    """Both corrections are judgements, so the tab shows them rather than
    applying them invisibly."""
    prose = _prose(view._render_sources)
    assert "Summed before ranking" in prose
    assert "Marked as index managers" in prose
    assert "BANK OF MARIN" in prose  # why the heuristic was abandoned
    assert "judgement" in prose


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_the_ownership_chart_breaks_rather_than_bridging(built):
    """A quarter with no share count leaves a hole, not a line drawn through a
    number that was never computed."""
    figure = view._ownership_figure(built.ownership)
    assert all(trace.connectgaps is False for trace in figure.data)


def test_the_ownership_chart_separates_the_two_series(built):
    """Ownership flat while the filer count falls is consolidation, and one
    shared axis would hide it."""
    figure = view._ownership_figure(built.ownership)
    assert [trace.name for trace in figure.data] == ["Owned", "Filers"]
    assert figure.layout.yaxis.tickformat == ".0%"


def test_the_figure_uses_the_dark_template(built):
    figure = view._ownership_figure(built.ownership)
    assert figure.layout.template.layout.plot_bgcolor is not None


def test_an_empty_history_still_draws_a_frame():
    figure = view._ownership_figure(pd.DataFrame())
    assert figure.layout.height == 420


def test_the_register_dates_its_own_deadline(built):
    """The banner's arithmetic, as both front ends read it."""
    assert built.deadline == (built.as_of + pd.Timedelta(days=45)).date()
    assert built.days_old(built.deadline) == 0
    assert built.days_old(built.deadline - pd.Timedelta(days=3)) == -3


def test_the_latest_reading_is_the_last_quarter_with_nothing_filled(built):
    last = built.ownership.iloc[-1]
    assert built.latest["units"] == float(last["units"])
    assert built.latest["holders"] == float(last["holders"])
    assert logic.Holdings().latest == {}
