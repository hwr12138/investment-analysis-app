"""``app/views/research/insiders.py`` — the tab body.

Streamlit is never started. The view exposes pure string and figure builders and
the tests call those; ``render`` itself is only ever read with ``inspect``, which
is the idiom the other ``..._controls`` files use.

The tests that matter most here are not about layout. Open-market purchases are
roughly one row in 120 live, so this tab looks empty at exactly the companies a
reader opens first — and the legend and the reconciliation panel are the only
things standing between "deliberately filtered" and "apparently broken".
"""

from __future__ import annotations

import inspect
from datetime import date

import pandas as pd
import pytest

from app.logic.research import insiders as logic
from app.logic.research.dataset import ResearchDataset
from app.views.research import insiders as view

TODAY = date(2026, 8, 5)


def _prose(function) -> str:
    """Every string literal in a function, joined.

    The wording these tests pin runs past 88 columns and is written as adjacent
    literals, so a substring check against the raw source breaks on wherever the
    line happened to wrap. Reading the constants off the parsed tree pins the
    sentence rather than the formatting.
    """
    import ast

    tree = ast.parse(inspect.getsource(function).lstrip())
    return " ".join(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )


@pytest.fixture
def dataset(insider_frame) -> ResearchDataset:
    index = pd.bdate_range("2021-01-01", "2026-08-05", name="Date")
    ohlcv = pd.DataFrame(
        {
            "Open": 100.0, "High": 102.0, "Low": 98.0,
            "Close": 101.0, "Adj Close": 101.0, "Volume": 1e6,
        },
        index=index,
    )
    return ResearchDataset(
        ticker="TEST",
        profile={"name": "TESTCO INC", "currency": "USD"},
        ohlcv=ohlcv,
        daily=pd.DataFrame(),
        actions=pd.DataFrame(),
        insiders=insider_frame,
        statements_ttm={},
        statements_annual={},
        statements_quarterly={},
        quarterly_ar=pd.DataFrame(),
        stats=[],
        peer_universe=pd.DataFrame(),
        currency="USD",
        loaded_at=TODAY,
    )


# ---------------------------------------------------------------------------
# The dispatch contract
# ---------------------------------------------------------------------------


def test_the_tab_takes_the_snapshot():
    assert list(inspect.signature(view.render).parameters) == ["dataset"]


def test_the_tab_makes_no_request():
    """Every tab but the DCF reads the snapshot and fetches nothing."""
    source = inspect.getsource(view)
    assert "sharadar_client" not in source
    assert "fetch_" not in source


def test_the_stylesheet_is_injected_first():
    body = inspect.getsource(view.render)
    assert "_INSIDERS_CSS" in body
    first = [line for line in body.splitlines() if "st." in line][0]
    assert "_INSIDERS_CSS" in first


def test_the_css_is_namespaced_to_this_tab():
    """One prefix per tab, so a rule cannot leak into Summary or Valuation."""
    for rule in view._INSIDERS_CSS.split("{"):
        selector = rule.strip().splitlines()[-1].strip() if rule.strip() else ""
        if selector.startswith("."):
            assert selector.startswith(".ins-"), selector


# ---------------------------------------------------------------------------
# Why the tab looks empty — the part a reader has to be told
# ---------------------------------------------------------------------------


def test_the_legend_names_the_codes_that_are_counted():
    """Without this a reader at Apple sees an empty chart and files a bug."""
    prose = _prose(view._render_legend)
    assert "Open-market purchases (P) and sales (S) only" in prose
    assert "Grants, option exercises and shares withheld for tax" in prose
    assert "reverses the sign of the total" in prose


def test_the_legend_says_a_ten_percent_owner_is_not_filtered():
    assert "labelled, never filtered" in _prose(view._render_legend)


def test_the_sources_panel_counts_the_rows_that_were_excluded():
    """The expander title carries "N of M", so filtering is visible before the
    panel is even opened."""
    source = inspect.getsource(view._render_sources)
    assert "of {len(dataset.insiders)} filed rows shown" in source


def test_a_company_with_no_filings_says_so_rather_than_drawing_an_empty_chart():
    body = inspect.getsource(view.render)
    assert "research.usable(frame)" in body
    assert "No insider filings" in body
    assert body.index("return") < body.index("_events_figure")


# ---------------------------------------------------------------------------
# The headline row
# ---------------------------------------------------------------------------


def test_a_net_purchase_reads_green_and_a_net_sale_red(insider_frame):
    window = logic.trailing_flow(insider_frame, 12, today=TODAY)
    cell = view._headline_cell(window, "USD")
    assert "ins-buy" in cell
    assert "+USD 13.35M" in cell


def test_the_headline_counts_distinct_insiders_not_the_two_sides_added(
    insider_frame,
):
    """Jane both bought and sold. Adding buyers to sellers counts her twice."""
    window = logic.trailing_flow(insider_frame, 12, today=TODAY)
    assert window.buyers + window.sellers == 7
    assert window.participants == 6
    assert "6 insider(s)" in view._headline_cell(window, "USD")


def test_an_unavailable_window_shows_its_reason_not_a_blank(insider_frame):
    window = logic.trailing_flow(insider_frame, 1, today=date(2026, 12, 31))
    cell = view._headline_cell(window, "USD")
    assert "ins-missing" in cell
    assert "No insider reported a transaction" in cell


def test_the_concentration_flag_names_the_filer(insider_frame):
    window = logic.trailing_flow(insider_frame, 12, today=TODAY)
    assert "JANE OFFICER is 68% of the gross" in view._headline_cell(window, "USD")


# ---------------------------------------------------------------------------
# The event chart
# ---------------------------------------------------------------------------


def test_the_markers_sit_at_the_filed_price_not_the_close(dataset, insider_frame):
    """The filed price is what the insider actually paid; snapping to the close
    would quietly redraw the trade."""
    figure = view._events_figure(dataset, insider_frame, logic.InsiderTolerances())
    buys = [trace for trace in figure.data if trace.name == "Insider buy"][0]
    assert set(buys.y) == {100.0}
    assert 101.0 not in set(buys.y)  # the Close in the fixture


def test_a_pre_split_marker_is_moved_onto_the_price_lines_basis(
    dataset, insider_frame
):
    """SEP's close is split-adjusted and SF2's filed price never is. At Tesla
    that puts November 2021 markers at $1,050 against a line reading $350."""
    import dataclasses

    split = pd.DataFrame(
        [{"date": pd.Timestamp("2026-07-01"), "action": "split", "value": 4.0}]
    )
    figure = view._events_figure(
        dataclasses.replace(dataset, actions=split),
        insider_frame,
        logic.InsiderTolerances(),
    )
    buys = [trace for trace in figure.data if trace.name == "Insider buy"][0]
    plotted = dict(zip(pd.to_datetime(buys.x), buys.y))
    # Filed at 100 either side of the split; only the earlier ones move.
    assert plotted[pd.Timestamp("2026-06-10")] == 25.0
    assert plotted[pd.Timestamp("2026-08-01")] == 100.0
    labels = {row[5] for row in buys.customdata}
    assert any("as filed" in label for label in labels)
    assert any("as filed" not in label for label in labels)


def test_only_open_market_trades_are_plotted(dataset, insider_frame):
    """An `M` exercise is priced at the strike — $12 against a $100 market — so
    plotting every code would imply a crash that never happened."""
    figure = view._events_figure(dataset, insider_frame, logic.InsiderTolerances())
    plotted = {y for trace in figure.data if trace.name != "Close" for y in trace.y}
    assert 12.0 not in plotted


def test_the_marker_scales_by_area_not_diameter(dataset, insider_frame):
    """Plotly's default scales the diameter, exaggerating a large trade by the
    square of its value."""
    figure = view._events_figure(dataset, insider_frame, logic.InsiderTolerances())
    for trace in figure.data:
        if trace.name != "Close":
            assert trace.marker.sizemode == "area"


def test_the_hover_carries_the_owner_and_the_filing_date(dataset, insider_frame):
    figure = view._events_figure(dataset, insider_frame, logic.InsiderTolerances())
    buys = [trace for trace in figure.data if trace.name == "Insider buy"][0]
    assert "Filed" in buys.hovertemplate
    assert "customdata[0]" in buys.hovertemplate


def test_point_events_are_not_grouped_by_the_x_axis(dataset, insider_frame):
    """"x unified" would gather unrelated trades that merely share an x."""
    figure = view._events_figure(dataset, insider_frame, logic.InsiderTolerances())
    assert figure.layout.hovermode == "closest"


def test_the_price_line_is_a_slice_of_the_snapshot(dataset, insider_frame):
    figure = view._events_figure(dataset, insider_frame, logic.InsiderTolerances())
    close = [trace for trace in figure.data if trace.name == "Close"][0]
    assert min(close.x) >= pd.Timestamp("2021-08-05")


def test_a_company_with_no_open_market_trades_still_draws_the_price(dataset):
    """Apple's case: five years, no open-market purchase. The chart is not empty,
    it just has no markers."""
    quiet = dataset.insiders[dataset.insiders["transactioncode"].isin({"A", "M"})]
    figure = view._events_figure(dataset, quiet, logic.InsiderTolerances())
    assert [trace.name for trace in figure.data] == ["Close"]


# ---------------------------------------------------------------------------
# The quarterly bars
# ---------------------------------------------------------------------------


def test_the_flow_chart_draws_every_quarter_including_the_empty_ones(insider_frame):
    quarters = logic.periodic_flow(insider_frame, today=TODAY)
    figure = view._quarterly_flow_figure(quarters, "USD")
    assert len(figure.data[0].x) == 21
    assert figure.data[0].x[-1] == "2026 Q3"


def test_the_open_quarter_is_drawn_back(insider_frame):
    """A short final bar is otherwise read as a collapse in activity."""
    quarters = logic.periodic_flow(insider_frame, today=TODAY)
    figure = view._quarterly_flow_figure(quarters, "USD")
    opacity = figure.data[0].marker.opacity
    assert opacity[-1] < opacity[-2]


def test_gross_flow_rides_in_the_hover(insider_frame):
    """A quarter netting zero on $50m each way is not a quiet quarter."""
    quarters = logic.periodic_flow(insider_frame, today=TODAY)
    figure = view._quarterly_flow_figure(quarters, "USD")
    assert "Bought" in figure.data[0].hovertemplate
    assert "Sold" in figure.data[0].hovertemplate


def test_the_bars_are_sign_coloured(insider_frame):
    quarters = logic.periodic_flow(insider_frame, today=TODAY)
    figure = view._quarterly_flow_figure(quarters, "USD")
    colours = list(figure.data[0].marker.color)
    nets = list(quarters["net"])
    assert colours[nets.index(max(nets))] == view._BUY
    assert colours[nets.index(min(nets))] == view._SELL


# ---------------------------------------------------------------------------
# The roster
# ---------------------------------------------------------------------------


def test_the_roster_table_is_one_line_of_html(insider_frame):
    """Streamlit Markdown-parses the string first, so a newline breaks it into
    paragraphs."""
    roster = logic.roster(insider_frame, today=TODAY)
    assert "\n" not in view._roster_html(roster.rows)


def test_owner_names_are_escaped_not_trusted_as_markup(insider_frame):
    """Owner names are vendor free text — the only untrusted strings here."""
    roster = logic.roster(insider_frame, today=TODAY)
    rows = roster.rows.copy()
    rows.loc[0, "owner"] = "<img src=x onerror=alert(1)>"
    html = view._roster_html(rows)
    assert "<img src=x" not in html
    assert "&lt;img" in html


def test_the_roster_never_claims_a_current_holding():
    """SF2 records no departure event, so there is no date on which it asserts
    anyone still holds anything."""
    prose = _prose(view._render_roster)
    assert "as last reported" in prose
    assert "not a confirmed holding today" in prose
    assert "confirmed holding today" in prose and "currently holds" not in prose


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------


def test_no_clusters_states_the_criteria_rather_than_going_blank():
    """"No clusters" and "we did not look" are different facts."""
    prose = _prose(view._render_clusters)
    assert "None in the last" in prose
    assert "10b5-1 plans and vesting schedules" in prose
    assert "cluster_min_buyers" in inspect.getsource(view._render_clusters)


# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------


def test_the_figures_use_the_dark_template(dataset, insider_frame):
    quarters = logic.periodic_flow(insider_frame, today=TODAY)
    for figure in (
        view._events_figure(dataset, insider_frame, logic.InsiderTolerances()),
        view._quarterly_flow_figure(quarters, "USD"),
    ):
        assert figure.layout.template.layout.plot_bgcolor is not None


def test_the_buy_and_sell_colours_match_the_rest_of_the_page():
    from app.views.research import summary

    assert view._BUY in inspect.getsource(summary)
    assert view._SELL in inspect.getsource(summary)
