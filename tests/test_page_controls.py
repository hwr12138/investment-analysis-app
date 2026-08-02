"""Page-level controls: compute gating, unit scaling, and the cross-check label."""

from __future__ import annotations

import pytest

from app.logic.dcf import valuation as dcf
from app.views import dcf_valuation as page


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


def test_every_exported_historical_column_has_a_readable_label():
    for key in (
        "revenue", "ebit", "pretax_income", "net_income", "cfo", "dna",
        "total_assets", "total_liabilities", "total_equity", "cash",
    ):
        assert key in page._HISTORICAL_LABELS
        assert page._HISTORICAL_LABELS[key][0].isupper()


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
