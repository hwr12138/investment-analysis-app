"""Page-level controls: compute gating, unit scaling, and the cross-check label."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.logic.dcf import reverse
from app.logic.dcf import valuation as dcf
from app.views.research import dcf as page


def _inputs(**overrides) -> dict:
    drivers = dcf.DriverPath.flat(
        5,
        revenue_growth=0.09,
        operating_margin=0.30,
        effective_tax_rate=0.16,
        capex_pct_revenue=0.03,
        dna_pct_revenue=0.03,
        dnwc_pct_drevenue=-0.40,
    )
    wacc = type("W", (), {"wacc": 0.0905})()
    base = dict(
        drivers=drivers,
        wacc=wacc,
        terminal_growth=0.025,
        exit_multiple=23.2,
        stub=0.167,
        mid_year=True,
        shares=14_837_004_750.0,
        share_basis="Diluted weighted-average (TTM)",
        include_sti=True,
        cash_floor=0.0,
        non_operating=None,
        headline_method="gordon",
        sbc=None,
        deduct_sbc=False,
        etr=0.1561,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Compute gating
# ---------------------------------------------------------------------------


def test_identical_assumptions_produce_an_identical_fingerprint():
    assert page._assumption_fingerprint(_inputs()) == page._assumption_fingerprint(_inputs())


def test_fingerprint_is_hashable_so_it_can_live_in_session_state():
    assert isinstance(hash(page._assumption_fingerprint(_inputs())), int)


@pytest.mark.parametrize(
    "field,value",
    [
        ("terminal_growth", 0.03),
        ("exit_multiple", 18.0),
        ("stub", 0.5),
        ("mid_year", False),
        ("shares", 15_000_000_000.0),
        ("share_basis", "Point-in-time outstanding"),
        ("include_sti", False),
        ("cash_floor", 1e9),
        ("non_operating", 5e9),
        ("headline_method", "exit_multiple"),
        ("sbc", 1.2e10),
        ("deduct_sbc", True),
        ("etr", 0.21),
    ],
)
def test_every_assumption_change_moves_the_fingerprint(field, value):
    """Anything that changes the answer has to invalidate the displayed run."""
    baseline = page._assumption_fingerprint(_inputs())
    assert page._assumption_fingerprint(_inputs(**{field: value})) != baseline


def test_a_driver_change_moves_the_fingerprint():
    changed = dcf.DriverPath.flat(
        5,
        revenue_growth=0.20,  # was 0.09
        operating_margin=0.30,
        effective_tax_rate=0.16,
        capex_pct_revenue=0.03,
        dna_pct_revenue=0.03,
        dnwc_pct_drevenue=-0.40,
    )
    assert page._assumption_fingerprint(_inputs(drivers=changed)) != page._assumption_fingerprint(
        _inputs()
    )


def test_a_wacc_change_moves_the_fingerprint():
    other = type("W", (), {"wacc": 0.10})()
    assert page._assumption_fingerprint(_inputs(wacc=other)) != page._assumption_fingerprint(
        _inputs()
    )


def test_display_only_state_is_not_in_the_fingerprint():
    """Switching the units toggle must not mark the results stale."""
    fingerprint = page._assumption_fingerprint(_inputs())
    assert "Millions" not in str(fingerprint)
    assert "Billions" not in str(fingerprint)


def test_the_reverse_dcf_is_not_in_the_fingerprint():
    """It needs no entry of its own.

    Every assumption it uses is listed already, and the market price reaches the
    tuple through the WACC — which is what the market value of equity is
    computed from. See `test_a_wacc_change_moves_the_fingerprint`.
    """
    baseline = page._assumption_fingerprint(_inputs())
    assert page._assumption_fingerprint(_inputs(reverse="anything at all")) == baseline


# ---------------------------------------------------------------------------
# Reverse DCF messages
# ---------------------------------------------------------------------------


def _reverse(status, **overrides):
    fields = dict(status=status, target=100.0, growth=None)
    fields.update(overrides)
    return reverse.ReverseDcfResult(**fields)


def test_every_reverse_status_has_a_message():
    """Adding a status without writing its message has to fail here."""
    for status in reverse.Status:
        result = _reverse(
            status,
            growth=0.12 if status is reverse.Status.SOLVED else None,
            local_slope=5.0 if status is reverse.Status.SOLVED else None,
            reason="measured detail",
        )
        level, message = page._reverse_message(result)
        assert level in {"", "info", "warning", "error"}
        if status is not reverse.Status.SOLVED:
            assert message, f"{status} renders nothing"


def test_an_ordinary_solved_rate_needs_no_banner():
    result = _reverse(reverse.Status.SOLVED, growth=0.12, local_slope=5.0)
    assert page._reverse_message(result) == ("", "")


def test_a_value_destroying_solve_says_the_reading_is_inverted():
    result = _reverse(reverse.Status.SOLVED, growth=0.12, local_slope=-5.0)
    level, message = page._reverse_message(result)
    assert level == "warning"
    assert "destroys value" in message
    assert "inverted" in message


def test_an_extreme_solved_rate_is_flagged_rather_than_presented_as_a_forecast():
    result = _reverse(reverse.Status.SOLVED, growth=0.90, local_slope=5.0)
    level, message = page._reverse_message(result)
    assert level == "warning"
    assert "not explained by revenue growth" in message


def test_the_precondition_message_says_it_is_not_a_failed_search():
    result = _reverse(reverse.Status.NOT_EVALUABLE, reason="Terminal growth is not below WACC.")
    level, message = page._reverse_message(result)
    assert level == "error"
    assert "not a failed search" in message


def test_an_unwritten_status_is_a_loud_failure_not_a_blank_panel():
    with pytest.raises(AssertionError):
        page._reverse_message(_reverse("invented_status"))


def test_the_out_of_range_reason_states_the_bounds_it_searched():
    """So the copy cannot drift from GROWTH_BOUNDS."""
    result = reverse.solve_growth(lambda g: g, 5.0)
    assert result.status is reverse.Status.TARGET_ABOVE_RANGE
    low, high = reverse.GROWTH_BOUNDS
    assert f"{low:.2%}" in result.reason
    assert f"{high:.2%}" in result.reason


def test_an_implied_revenue_path_compounds_from_the_base():
    path = page._implied_revenue_path(1000.0, 0.10, 3)
    assert path == pytest.approx([1100.0, 1210.0, 1331.0])


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_unit_scales():
    assert page._UNITS["As reported"][0] == 1.0
    assert page._UNITS["Millions"][0] == 1e6
    assert page._UNITS["Billions"][0] == 1e9


def test_unit_headers_are_distinct_and_labelled():
    suffixes = [suffix for _, suffix, _ in page._UNITS.values()]
    assert suffixes == ["", " (m)", " (bn)"]


def test_billions_scaling_of_a_real_figure():
    divisor, suffix, decimals = page._UNITS["Billions"]
    assert round(416_161_000_000 / divisor, decimals) == 416.16
    assert suffix == " (bn)"


def test_millions_scaling_of_a_real_figure():
    divisor, _, decimals = page._UNITS["Millions"]
    assert round(416_161_000_000 / divisor, decimals) == 416161.0


# ---------------------------------------------------------------------------
# Reported figures
# ---------------------------------------------------------------------------


def test_the_three_statements_map_to_frame_codes():
    assert page._STATEMENTS == {
        "Income statement": "PL", "Balance sheet": "BS", "Cash flow": "CF"
    }


def test_every_statement_offered_is_one_the_client_builds():
    """A statement in the picker the client never builds would render blank."""
    for code in page._STATEMENTS.values():
        assert code in {"PL", "BS", "CF", "DERIVED"}


def test_statement_metadata_is_not_shown_as_a_line_item():
    """Report dates and restatement flags describe the filing, not the business."""
    for column in ("Fiscal Year", "Report Date", "Publish Date", "Restated", "Source"):
        assert column in page._STATEMENT_META


def _statement(frame) -> dict:
    dataset = type("D", (), {
        "statements_annual": type("S", (), {"frame": staticmethod(lambda code: frame)})()
    })()
    table, populated, empty = page._statement_frame(dataset, "PL")
    return {"table": table, "populated": populated, "empty": empty}


def test_a_statement_is_transposed_so_line_items_read_down_the_page():
    frame = pd.DataFrame({
        "Fiscal Year": [2024, 2025],
        "Fiscal Period": ["FY", "FY"],
        "Revenue": [100.0, 120.0],
        "Gross Profit": [40.0, 50.0],
    })
    out = _statement(frame)
    assert list(out["table"].index) == ["Revenue", "Gross Profit"]
    assert list(out["table"].columns) == ["2024", "2025"]
    assert out["table"].loc["Revenue", "2025"] == 120.0


def test_the_published_line_item_order_is_preserved():
    """Column order is the statement's own order; reordering would misread it."""
    frame = pd.DataFrame({
        "Fiscal Year": [2025],
        "Revenue": [100.0], "Cost of Revenue": [60.0], "Gross Profit": [40.0],
    })
    assert list(_statement(frame)["table"].index) == [
        "Revenue", "Cost of Revenue", "Gross Profit"
    ]


def test_a_line_item_reported_in_any_year_counts_as_populated():
    frame = pd.DataFrame({
        "Fiscal Year": [2024, 2025],
        "Revenue": [100.0, 120.0],
        "Sometimes": [np.nan, 5.0],
        "Never": [np.nan, np.nan],
    })
    out = _statement(frame)
    assert out["populated"] == ["Revenue", "Sometimes"]
    assert out["empty"] == ["Never"]


def test_an_empty_line_item_is_kept_rather_than_dropped():
    """That the provider carries a field and left it blank is itself a finding."""
    frame = pd.DataFrame({
        "Fiscal Year": [2025], "Revenue": [100.0], "Short Term Debt": [np.nan],
    })
    out = _statement(frame)
    assert "Short Term Debt" in out["table"].index
    assert out["empty"] == ["Short Term Debt"]


def test_non_numeric_columns_never_reach_the_numeric_table():
    frame = pd.DataFrame({
        "Fiscal Year": [2025], "Revenue": [100.0], "Currency": ["USD"],
    })
    assert list(_statement(frame)["table"].index) == ["Revenue"]


def test_a_missing_statement_yields_nothing_rather_than_an_error():
    for frame in (pd.DataFrame(), pd.DataFrame({"Revenue": [1.0]})):
        out = _statement(frame)
        assert out["table"].empty
        assert out["populated"] == [] and out["empty"] == []


def test_a_missing_date_renders_as_a_dash_not_a_fabricated_one():
    assert page._date_text(None) == "—"
    assert page._date_text(pd.NaT) == "—"
    assert page._date_text(pd.Timestamp("2025-09-30")) == "2025-09-30"


# ---------------------------------------------------------------------------
# Terminal value cross-check
# ---------------------------------------------------------------------------


def test_gordon_row_is_restated_as_an_exit_multiple():
    terminal = dcf.TerminalValue(
        method="Gordon growth", value=1000.0, present_value=800.0,
        discount_exponent=3.5, formula="", implied_exit_multiple=13.6,
    )
    assert page._cross_implied(terminal) == "an exit multiple of 13.60×"


def test_exit_multiple_row_is_restated_as_perpetual_growth():
    terminal = dcf.TerminalValue(
        method="Exit multiple", value=1000.0, present_value=800.0,
        discount_exponent=3.5, formula="", implied_growth=0.0511,
    )
    assert page._cross_implied(terminal) == "perpetual growth of 5.11%"


def test_cross_check_is_blank_when_neither_is_computable():
    terminal = dcf.TerminalValue(
        method="Gordon growth", value=1000.0, present_value=800.0,
        discount_exponent=3.5, formula="",
    )
    assert page._cross_implied(terminal) == "—"


def test_the_two_methods_really_do_cross_imply_each_other():
    """The Gordon row's multiple and the exit row's growth are not swapped."""
    drivers = dcf.DriverPath.flat(
        3, revenue_growth=0.10, operating_margin=0.20, effective_tax_rate=0.25,
        capex_pct_revenue=0.05, dna_pct_revenue=0.04, dnwc_pct_drevenue=0.10,
    )
    projection = dcf.project(1000.0, drivers, wacc=0.08, mid_year=False)
    terminals = dcf.build_terminal_values(
        projection, terminal_growth=0.02, exit_multiple=10.0
    )

    gordon, exit_multiple = terminals["gordon"], terminals["exit_multiple"]
    # Gordon states a multiple; feeding it back reproduces Gordon's own value.
    assert gordon.implied_growth is None
    assert dcf.exit_multiple_terminal_value(
        projection.terminal_ebitda, gordon.implied_exit_multiple
    ) == pytest.approx(gordon.value)
    # The exit multiple states a growth rate; feeding it back reproduces its value.
    assert exit_multiple.implied_exit_multiple is None
    assert dcf.gordon_growth_terminal_value(
        projection.terminal_row.fcff_full_year, exit_multiple.implied_growth, projection.wacc
    ) == pytest.approx(exit_multiple.value)


# ---------------------------------------------------------------------------
# Headline terminal value method
# ---------------------------------------------------------------------------


def test_exit_multiple_is_the_default_headline_method():
    from app.logic.dcf import model as dcf_model

    assert dcf_model.DEFAULT_HEADLINE_METHOD == "exit_multiple"


def test_the_radio_offers_the_default_first():
    """Streamlit selects the first option, so the order is the default."""
    labels = list(page._HEADLINE_METHODS)
    assert labels[0] == "Exit multiple"


def test_the_ui_and_the_engine_agree_on_the_default():
    from app.logic.dcf import model as dcf_model

    first_label = list(page._HEADLINE_METHODS)[0]
    assert page._HEADLINE_METHODS[first_label] == dcf_model.DEFAULT_HEADLINE_METHOD


def test_both_methods_remain_selectable():
    assert set(page._HEADLINE_METHODS.values()) == {"exit_multiple", "gordon"}
