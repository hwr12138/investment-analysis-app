"""Page-level controls: recompute gating, unit scaling, and the cross-check label."""

from __future__ import annotations

import importlib
from datetime import date
import inspect
import pkgutil

import pandas as pd
import pytest

from app.formatting import format_date, format_pct
from app.logic.dcf import export as dcf_export
from app.logic.dcf import model as dcf_model
from app.logic.dcf import reverse
from app.logic.dcf import valuation as dcf
from app.views.research import dcf as page
from app.views.research.dcf import assumptions, common, implied, results


def _inputs(**overrides) -> dict:
    drivers = dcf.DriverPath.flat(
        5,
        revenue_growth=0.09,
        operating_margin=0.30,
        effective_tax_rate=0.16,
        capex_pct_revenue=0.03,
        dna_pct_revenue=0.03,
        nwc_pct_revenue=-0.40,
    )
    wacc = type("W", (), {"wacc": 0.0905})()
    base = dict(
        drivers=drivers,
        wacc=wacc,
        terminal_growth=0.025,
        exit_multiple=23.2,
        elapsed=0.25,
        mid_year=True,
        ronic=None,
        terminal_tax_rate=0.25,
        growth_fade_to=0.025,
        growth_hold=3,
        growth_basis="TTM year-over-year",
        assumed_margins={},
        ebita_proxy=True,
        spread_table_as_of=None,
        acquisitions_in_capex=False,
        beta=1.1,
        shares=14_837_004_750.0,
        share_basis="Diluted weighted-average (TTM)",
        include_sti=True,
        cash_floor=0.0,
        non_operating=None,
        headline_method="gordon",
        sbc=None,
        etr=0.1561,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Recompute gating
#
# There is no Compute button: the tab values whatever the page has loaded, with
# whatever the assumption panel currently shows. The fingerprint is what decides
# when that has to be rebuilt, which makes every test below a correctness test —
# a field missing from the tuple means the previous run's numbers stay on screen
# looking current, and nothing on the tab would say so.
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
        ("elapsed", 0.5),
        ("mid_year", False),
        ("ronic", 0.15),
        ("terminal_tax_rate", None),
        ("growth_fade_to", None),
        ("growth_hold", 1),
        ("assumed_margins", {"Target margin": 0.35}),
        ("ebita_proxy", False),
        ("spread_table_as_of", date(2026, 1, 1)),
        ("acquisitions_in_capex", True),
        ("beta", 1.3),
        ("shares", 15_000_000_000.0),
        ("share_basis", "Point-in-time outstanding"),
        ("include_sti", False),
        ("cash_floor", 1e9),
        ("non_operating", 5e9),
        ("headline_method", "exit_multiple"),
        ("sbc", 1.2e10),
        ("etr", 0.21),
    ],
)
def test_every_assumption_change_moves_the_fingerprint(field, value):
    """Anything that changes the answer has to force a recompute."""
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
        nwc_pct_revenue=-0.40,
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
    """Switching the units selector must not re-run the model.

    It reruns the script like any other widget, and the reverse solve underneath
    a rebuild is some fifty model evaluations.
    """
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


def _stored(ticker="AAPL", fingerprint=("f",)) -> dict:
    return {"company": ticker, "fingerprint": fingerprint}


def test_an_unchanged_rerun_reuses_the_stored_run():
    """The tab re-renders on every widget on the page, not just its own."""
    assert not page._needs_run(_stored(), "AAPL", ("f",))


def test_a_changed_assumption_recomputes_before_rendering():
    assert page._needs_run(_stored(), "AAPL", ("something else",))


def test_a_run_for_another_company_is_never_reused():
    """There is no staleness notice: a mismatch has to recompute, not warn."""
    assert page._needs_run(_stored("AAPL"), "MSFT", ("f",))


def test_an_empty_session_always_runs():
    assert page._needs_run(None, "AAPL", ("f",))


# ---------------------------------------------------------------------------
# Statement basis
# ---------------------------------------------------------------------------


def test_the_page_always_loads_the_restated_series():
    """The basis toggle is gone, so the page inherits the engine's default.

    Restated figures are what the driver averages need — a consistent time
    series. If that default ever moved, every driver here would switch to the
    as-filed series with nothing on screen to say so.
    """
    assert list(inspect.signature(page._load_company_cached).parameters) == ["ticker"]
    assert inspect.signature(dcf_model.load_company).parameters["asreported"].default is False


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
        level, message = implied._reverse_message(result)
        assert level in {"", "info", "warning", "error"}
        if status is not reverse.Status.SOLVED:
            assert message, f"{status} renders nothing"


def test_an_ordinary_solved_rate_needs_no_banner():
    result = _reverse(reverse.Status.SOLVED, growth=0.12, local_slope=5.0)
    assert implied._reverse_message(result) == ("", "")


def test_a_value_destroying_solve_says_the_reading_is_inverted():
    result = _reverse(reverse.Status.SOLVED, growth=0.12, local_slope=-5.0)
    level, message = implied._reverse_message(result)
    assert level == "warning"
    assert "destroys value" in message
    assert "inverted" in message


def test_an_extreme_solved_rate_is_flagged_rather_than_presented_as_a_forecast():
    result = _reverse(reverse.Status.SOLVED, growth=0.90, local_slope=5.0)
    level, message = implied._reverse_message(result)
    assert level == "warning"
    assert "not explained by revenue growth" in message


def test_the_precondition_message_says_it_is_not_a_failed_search():
    result = _reverse(reverse.Status.NOT_EVALUABLE, reason="Terminal growth is not below WACC.")
    level, message = implied._reverse_message(result)
    assert level == "error"
    assert "not a failed search" in message


def test_an_unwritten_status_is_a_loud_failure_not_a_blank_panel():
    with pytest.raises(AssertionError):
        implied._reverse_message(_reverse("invented_status"))


def test_the_out_of_range_reason_states_the_bounds_it_searched():
    """So the copy cannot drift from GROWTH_BOUNDS."""
    result = reverse.solve_growth(lambda g: g, 5.0)
    assert result.status is reverse.Status.TARGET_ABOVE_RANGE
    low, high = reverse.GROWTH_BOUNDS
    assert f"{low:.2%}" in result.reason
    assert f"{high:.2%}" in result.reason


def test_an_implied_revenue_path_compounds_from_the_base():
    path = implied._implied_revenue_path(1000.0, 0.10, 3)
    assert path == pytest.approx([1100.0, 1210.0, 1331.0])


def test_an_implied_revenue_path_follows_the_forecast_fade():
    path = implied._implied_revenue_path(1000.0, 0.20, 3, fade_to=0.02)
    assert path == pytest.approx([1200.0, 1200.0 * 1.14, 1200.0 * 1.14 * 1.08])


def test_an_implied_revenue_path_holds_before_it_fades():
    path = implied._implied_revenue_path(1000.0, 0.20, 3, fade_to=0.02, hold=2)
    assert path == pytest.approx([1200.0, 1440.0, 1440.0 * 1.11])


def test_the_reverse_dcf_reads_gordon_first_as_the_fundamentals():
    """§2.1: both methods are solved, Gordon leading."""
    assert implied.METHOD_ORDER == ("gordon", "exit_multiple")
    assert "fundamentals" in implied.METHOD_LABELS["gordon"]


# ---------------------------------------------------------------------------
# One set of model arguments for all three calls
# ---------------------------------------------------------------------------


def test_every_run_model_argument_comes_from_one_place():
    """The forward run, the reverse solve and the what-if price share
    `common.model_kwargs`, so the reverse DCF cannot invert a different model.
    Every keyword `run_model` takes beyond the dataset, base and drivers has to
    be in it."""
    parameters = set(inspect.signature(dcf_model.run_model).parameters)
    supplied = set(common.model_kwargs(_inputs(base_fcff=None, shares=1.0)))
    assert supplied == parameters - {"dataset", "base", "drivers"}


# ---------------------------------------------------------------------------
# Filing provenance
#
# All that survives of the tab's own statement viewer. The line items now live
# on 📄 Financials, which lists the same report and publish dates but not the
# filings — so this is the one path from a valuation back to the documents.
# ---------------------------------------------------------------------------


def test_a_missing_date_renders_as_a_dash_not_a_fabricated_one():
    assert format_date(None, missing="—") == "—"
    assert format_date(pd.NaT, missing="—") == "—"
    assert format_date(pd.Timestamp("2025-09-30"), missing="—") == "2025-09-30"


def test_a_missing_percentage_reads_as_missing_like_everywhere_else():
    """It used to read "n/a" here and "not reported" on every other tab."""
    assert format_pct(None) == "not reported"
    assert format_pct(float("nan")) == "not reported"
    assert format_pct(float("inf")) == "not reported"


def test_valuation_money_is_quoted_in_whole_units_below_a_million():
    assert common.money(123_456.78, "USD") == "USD 123,457"
    assert common.money(4.54e12, "USD") == "USD 4.54T"
    assert common.money(pd.NA) == "not reported"


def test_the_tab_no_longer_carries_a_statement_viewer():
    """It points at 📄 Financials instead. A second one would drift from it."""
    modules = [page] + [
        importlib.import_module(f"{page.__name__}.{info.name}")
        for info in pkgutil.iter_modules(page.__path__)
    ]
    assert len(modules) > 1
    for module in modules:
        for name in ("_statement_frame", "_render_reported_figures", "_STATEMENTS", "_UNITS"):
            assert not hasattr(module, name), (module.__name__, name)


# ---------------------------------------------------------------------------
# Terminal value cross-check
# ---------------------------------------------------------------------------


def test_gordon_row_is_restated_as_an_exit_multiple():
    terminal = dcf.TerminalValue(
        method="Gordon growth", value=1000.0, present_value=800.0,
        discount_exponent=3.5, formula="", implied_exit_multiple=13.6,
    )
    assert results._cross_implied(terminal) == "an exit multiple of 13.60×"


def test_exit_multiple_row_is_restated_as_growth_and_a_return_on_capital():
    terminal = dcf.TerminalValue(
        method="Exit multiple", value=1000.0, present_value=800.0,
        discount_exponent=3.5, formula="", implied_growth=0.0511, implied_ronic=0.184,
    )
    assert results._cross_implied(terminal) == (
        "perpetual growth of at least 5.11%; RONIC of 18.4% at the terminal rate"
    )


def test_cross_check_is_blank_when_neither_is_computable():
    terminal = dcf.TerminalValue(
        method="Gordon growth", value=1000.0, present_value=800.0,
        discount_exponent=3.5, formula="",
    )
    assert results._cross_implied(terminal) == "—"


def test_the_two_methods_really_do_cross_imply_each_other():
    """The Gordon row's multiple and the exit row's figures are not swapped."""
    drivers = dcf.DriverPath.flat(
        3, revenue_growth=0.10, operating_margin=0.20, effective_tax_rate=0.25,
        capex_pct_revenue=0.05, dna_pct_revenue=0.04, nwc_pct_revenue=0.10,
    )
    projection = dcf.project(1000.0, drivers, wacc=0.08, mid_year=False)
    terminals = dcf.build_terminal_values(
        projection, terminal_growth=0.02, exit_multiple=10.0
    )

    gordon, exit_multiple = terminals["gordon"], terminals["exit_multiple"]
    steady = gordon.terminal_year
    # Gordon states a multiple; feeding it back reproduces Gordon's own value.
    assert gordon.implied_growth is None
    assert dcf.exit_multiple_terminal_value(
        projection.final_ebitda, gordon.implied_exit_multiple
    ) == pytest.approx(gordon.value)
    # The exit multiple states a return on new capital; feeding it back through
    # the value-driver formula reproduces the exit value.
    assert exit_multiple.implied_exit_multiple is None
    reinvested = steady.nopat * (1 - 0.02 / exit_multiple.implied_ronic)
    assert dcf.gordon_growth_terminal_value(reinvested, 0.02, 0.08) == pytest.approx(
        exit_multiple.value
    )


# ---------------------------------------------------------------------------
# Headline terminal value method
# ---------------------------------------------------------------------------


def test_gordon_growth_is_the_default_headline_method():
    """The owner's choice, reversing DCF_ROUND2_INSTRUCTIONS.md §2.1: the headline
    rests on the company's own cash flows rather than on the multiple it traded
    at. The exit multiple is still computed and shown beside it as the
    cross-check."""
    assert dcf_model.DEFAULT_HEADLINE_METHOD == "gordon"


def test_the_radio_offers_the_default_first():
    """Streamlit selects the first option, so the order is the default."""
    labels = list(assumptions._HEADLINE_METHODS)
    assert labels[0] == "Gordon growth"


def test_the_ui_and_the_engine_agree_on_the_default():
    first_label = list(assumptions._HEADLINE_METHODS)[0]
    assert assumptions._HEADLINE_METHODS[first_label] == dcf_model.DEFAULT_HEADLINE_METHOD


def test_both_methods_remain_selectable():
    assert set(assumptions._HEADLINE_METHODS.values()) == {"exit_multiple", "gordon"}


# ---------------------------------------------------------------------------
# Capex normalisation
# ---------------------------------------------------------------------------


def _capex_target(normal_value):
    """Draws the capex target alone, as the panel does, and keeps what it returned."""
    import streamlit as st

    from app.logic.dcf import financials as fin
    from app.views.research.dcf import assumptions

    periods = ("2022", "2023", "2024", "2025", "2026")
    normal = fin.DriverDefault(
        "capex_pct_revenue", normal_value, fin.NORMAL_CAPEX_BASIS,
        len(periods) if normal_value is not None else 0,
        periods if normal_value is not None else (),
        "" if normal_value is not None else "No period had every component needed.",
    )
    records = []
    st.session_state["_target"] = assumptions._driver_target(
        "capex % of revenue", "capex", 0.35, 10, records, normal=normal
    )
    st.session_state["_origin"] = records[-1].sourced.origin.value


def _run_capex_target(normal_value):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_capex_target, args=(normal_value,))
    app.run()
    assert not app.exception
    return app


def test_capex_moves_to_its_five_year_average_by_default():
    """Today's capex is where the forecast starts, not where it stays: the
    target is on without anyone touching it, and it is derived, not typed."""
    app = _run_capex_target(0.20)
    assert app.toggle(key="dcf_input_target_capex_on").value is True
    assert app.session_state["_target"] == pytest.approx(0.20)
    assert app.session_state["_origin"] == "derived"


def test_a_typed_capex_target_reads_as_the_users():
    app = _run_capex_target(0.20)
    app.number_input(key="dcf_input_target_capex").set_value(15.0).run()
    assert app.session_state["_target"] == pytest.approx(0.15)
    assert app.session_state["_origin"] == "user"


def test_turning_normalisation_off_holds_todays_capex():
    app = _run_capex_target(0.20)
    app.toggle(key="dcf_input_target_capex_on").set_value(False).run()
    assert app.session_state["_target"] is None


def test_no_history_leaves_capex_flat_and_says_why():
    """No silent fallback: with nothing to normalise to, nothing is substituted."""
    app = _run_capex_target(None)
    assert app.toggle(key="dcf_input_target_capex_on").value is False
    assert app.session_state["_target"] is None
    assert any("No 5-year average to normalise to" in c.value for c in app.caption)


# ---------------------------------------------------------------------------
# Return on new invested capital
# ---------------------------------------------------------------------------


def _ronic_panel(ebit):
    """Draws the RONIC choice for a company whose ROIC is ebit × 0.8 ÷ 100 a year."""
    from types import SimpleNamespace

    import pandas as pd
    import streamlit as st

    from app.views.research.dcf import assumptions

    years = 6
    annual = pd.DataFrame(
        {
            "ebit": [ebit] * years, "pretax_income": [ebit] * years,
            "tax_expense": [-0.2 * ebit] * years, "total_equity": [100.0] * years,
        },
        index=list(range(2020, 2020 + years)),
    )
    ronic, sourced = assumptions._return_on_new_capital(
        SimpleNamespace(annual=annual), SimpleNamespace(wacc=0.09)
    )
    st.session_state["_ronic"] = ronic
    st.session_state["_origin"] = sourced.origin.value


def _run_ronic(ebit):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_ronic_panel, args=(ebit,))
    app.run()
    assert not app.exception
    return app


def test_the_default_ronic_is_the_companys_own_return_above_the_wacc():
    """RONIC = WACC credited nothing to growth beyond year N; the company's own
    record is the default evidence for what its new capital earns."""
    app = _run_ronic(ebit=30.0)                          # 30 × 0.8 ÷ 100 = 24%
    assert app.radio(key="dcf_input_ronic_basis").value == assumptions._OWN_ROIC
    assert app.session_state["_ronic"] == pytest.approx(0.24)
    assert app.session_state["_origin"] == "derived"


def test_an_own_return_below_the_wacc_defaults_to_the_wacc_not_to_destroying_value():
    app = _run_ronic(ebit=5.0)                           # 4%, against a 9% WACC
    assert app.session_state["_ronic"] is None
    assert app.session_state["_origin"] == "estimate"
    assert any("does not exceed its WACC" in c.value for c in app.caption)


def test_no_measurable_return_offers_only_the_wacc_and_a_manual_figure():
    app = _run_ronic(ebit=-10.0)
    radio = app.radio(key="dcf_input_ronic_basis")
    assert list(radio.options) == [assumptions._RONIC_AT_WACC, assumptions._RONIC_MANUAL]
    assert app.session_state["_ronic"] is None
    assert any("No own ROIC to default to" in c.value for c in app.caption)


def test_ronic_can_still_be_set_to_the_wacc_or_by_hand():
    app = _run_ronic(ebit=30.0)
    app.radio(key="dcf_input_ronic_basis").set_value(assumptions._RONIC_AT_WACC).run()
    assert app.session_state["_ronic"] is None
    app.radio(key="dcf_input_ronic_basis").set_value(assumptions._RONIC_MANUAL).run()
    assert app.number_input(key="dcf_input_ronic").value == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# Driver basis selector (DCF_ROUND2_INSTRUCTIONS.md §2.2)
# ---------------------------------------------------------------------------


def test_changing_a_drivers_basis_reseeds_its_value_and_its_provenance():
    state: dict = {}
    seed = lambda basis, default: assumptions._seed_driver_value(  # noqa: E731
        state, "dcf_input_growth_value", "dcf_input_growth_seeded_from", basis, default
    )

    seed("5-year CAGR", 8.0)
    assert state["dcf_input_growth_value"] == 8.0

    # Typed by hand: kept while the basis stays put, and read as an override.
    state["dcf_input_growth_value"] = 12.0
    seed("5-year CAGR", 8.0)
    assert state["dcf_input_growth_value"] == 12.0
    assert not assumptions._untouched(12.0, 8.0, 4)

    # A new basis brings its own default, which is then read as untouched —
    # derived provenance, not a user override.
    seed("5-year average", 9.5)
    assert state["dcf_input_growth_value"] == 9.5
    assert assumptions._untouched(state["dcf_input_growth_value"], 9.5, 4)



def test_the_projection_view_formats_only_columns_the_export_produces():
    """The live check caught a column the export had dropped (SBC added back)
    still being formatted by the view. Pin the two together."""
    import re

    source = inspect.getsource(results.render_projection)
    body = source.split("frame = dcf_export")[1].split("st.dataframe(display")[0]
    formatted = set(re.findall(r'"([^"]+)"', body))
    drivers = dcf.DriverPath.flat(
        2, revenue_growth=0.1, operating_margin=0.2, effective_tax_rate=0.2,
        capex_pct_revenue=0.05, dna_pct_revenue=0.04, nwc_pct_revenue=0.1,
    )
    projection = dcf.project(100.0, drivers, wacc=0.09)
    result = type("R", (), {"projection": projection})()
    columns = set(dcf_export.projection_frame(result).columns)
    assert {c for c in formatted if c[:1].isupper()} <= columns


def test_the_effective_growth_basis_names_the_rate_actually_used():
    """The TTM-window acquisitions check reads it: it applies only when the
    trailing rate is the one in use."""
    from app.logic.dcf import financials as fin

    ttm = fin.DriverDefault("revenue_growth", 0.83, fin.TTM_GROWTH_BASIS, 1, ("P",))
    cagr = fin.DriverDefault("revenue_growth", 0.67, "5-year CAGR", 5, ("P",))
    slow = fin.DriverDefault("revenue_growth", 0.04, fin.TTM_GROWTH_BASIS, 1, ("P",))
    burst = fin.lower_of_growth(ttm, cagr)
    slowdown = fin.lower_of_growth(slow, cagr)
    assert assumptions._effective_growth_basis(fin.LOWER_GROWTH_BASIS, burst, ttm) == "5-year CAGR"
    assert assumptions._effective_growth_basis(
        fin.LOWER_GROWTH_BASIS, slowdown, slow
    ) == fin.TTM_GROWTH_BASIS
    assert assumptions._effective_growth_basis(
        fin.TTM_GROWTH_BASIS, burst, ttm
    ) == fin.TTM_GROWTH_BASIS
