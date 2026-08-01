"""Every formula in sections 5-8 of the spec, against hand-calculated fixtures."""

from __future__ import annotations

from datetime import date

import pytest

from app.logic.dcf import valuation as dcf


# ---------------------------------------------------------------------------
# FCFF
# ---------------------------------------------------------------------------


def test_fcff_method_a_hand_calculation():
    # 300 + 50 × (1 − 0.25) − 100 = 237.5
    assert dcf.fcff_from_cash_flow(300, 50, 0.25, 100) == pytest.approx(237.5)


def test_fcff_method_b_hand_calculation():
    # 250 × 0.75 + 60 − 100 − 10 = 137.5
    assert dcf.fcff_from_income_statement(250, 0.25, 60, 100, 10) == pytest.approx(137.5)


def test_fcff_handles_a_negative_result():
    """A cash-burning company must produce a negative FCFF, not an error."""
    assert dcf.fcff_from_income_statement(-100, 0.21, 20, 200, 30) < 0


def test_fcff_rejects_a_missing_input():
    """A missing input blocks the calculation rather than defaulting to zero."""
    with pytest.raises(dcf.ProjectionError):
        dcf.fcff_from_cash_flow(300, None, 0.25, 100)
    with pytest.raises(dcf.ProjectionError):
        dcf.fcff_from_income_statement(250, 0.25, float("nan"), 100, 10)


def test_divergence_is_symmetric_and_safe_at_zero():
    assert dcf.divergence(100, 110) == pytest.approx(dcf.divergence(110, 100))
    assert dcf.divergence(0, 0) == 0.0
    assert dcf.divergence(100, 105) == pytest.approx(5 / 105)


# ---------------------------------------------------------------------------
# Stub period
# ---------------------------------------------------------------------------


def test_stub_fraction_mid_year():
    assert dcf.stub_fraction(date(2026, 7, 31), date(2026, 9, 30)) == pytest.approx(61 / 365)


@pytest.mark.parametrize(
    "valuation_day,expected_days",
    [(date(2026, 1, 2), 271), (date(2026, 6, 30), 92), (date(2026, 9, 29), 1)],
)
def test_stub_fraction_at_several_points_in_the_year(valuation_day, expected_days):
    assert dcf.stub_fraction(valuation_day, date(2026, 9, 30)) == pytest.approx(
        expected_days / 365
    )


def test_stub_fraction_caps_at_one_full_year():
    assert dcf.stub_fraction(date(2025, 1, 1), date(2026, 9, 30)) == 1.0


def test_stub_fraction_rejects_a_fiscal_year_end_in_the_past():
    with pytest.raises(dcf.ProjectionError):
        dcf.stub_fraction(date(2026, 10, 1), date(2026, 9, 30))


# ---------------------------------------------------------------------------
# Discounting
# ---------------------------------------------------------------------------


def test_end_of_year_discount_factors():
    assert dcf.discount_factor(0.08, 1) == pytest.approx(1 / 1.08)
    assert dcf.discount_factor(0.08, 3) == pytest.approx(1 / 1.08**3)


def test_mid_year_discount_factor():
    assert dcf.discount_factor(0.08, 2.5) == pytest.approx(1 / 1.08**2.5)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


@pytest.fixture
def drivers() -> dcf.DriverPath:
    return dcf.DriverPath.flat(
        3,
        revenue_growth=0.10,
        operating_margin=0.20,
        effective_tax_rate=0.25,
        capex_pct_revenue=0.05,
        dna_pct_revenue=0.04,
        dnwc_pct_drevenue=0.10,
    )


def test_projection_year_one_by_hand(drivers):
    projection = dcf.project(1000.0, drivers, wacc=0.08, stub=1.0, mid_year=False)
    row = projection.rows[0]
    assert row.revenue == pytest.approx(1100.0)
    assert row.ebit == pytest.approx(220.0)
    assert row.nopat == pytest.approx(165.0)
    assert row.dna == pytest.approx(44.0)
    assert row.capex == pytest.approx(55.0)
    assert row.delta_nwc == pytest.approx(10.0)
    # 165 + 44 − 55 − 10 = 144
    assert row.fcff_full_year == pytest.approx(144.0)
    assert row.present_value == pytest.approx(144.0 / 1.08)


def test_mid_year_exponents_with_a_stub(drivers):
    """Stub of 0.5 gives exponents 0.25, 1.0, 2.0 under the mid-year convention."""
    projection = dcf.project(1000.0, drivers, wacc=0.08, stub=0.5, mid_year=True)
    assert [round(r.discount_exponent, 4) for r in projection.rows] == [0.25, 1.0, 2.0]


def test_stub_scales_only_the_first_period_cash_flow(drivers):
    projection = dcf.project(1000.0, drivers, wacc=0.08, stub=0.25, mid_year=False)
    first = projection.rows[0]
    assert first.fcff_period == pytest.approx(first.fcff_full_year * 0.25)
    assert projection.rows[1].fcff_period == pytest.approx(
        projection.rows[1].fcff_full_year
    )


def test_stub_does_not_shrink_later_revenue(drivers):
    """Compounding runs off full-year revenue, so the stub is not permanent."""
    full = dcf.project(1000.0, drivers, wacc=0.08, stub=1.0, mid_year=False)
    stubbed = dcf.project(1000.0, drivers, wacc=0.08, stub=0.2, mid_year=False)
    assert stubbed.rows[-1].revenue == pytest.approx(full.rows[-1].revenue)


def test_driver_path_rejects_mismatched_lengths():
    with pytest.raises(dcf.ProjectionError):
        dcf.DriverPath(
            revenue_growth=(0.1, 0.1),
            operating_margin=(0.2,),
            effective_tax_rate=(0.25,),
            capex_pct_revenue=(0.05,),
            dna_pct_revenue=(0.04,),
            dnwc_pct_drevenue=(0.1,),
        )


def test_projection_rejects_missing_base_revenue(drivers):
    with pytest.raises(dcf.ProjectionError):
        dcf.project(float("nan"), drivers, wacc=0.08)


def test_sbc_deduction_requires_a_figure(drivers):
    with pytest.raises(dcf.ProjectionError, match="stock-based compensation"):
        dcf.project(1000.0, drivers, wacc=0.08, sbc=None, deduct_sbc=True)


def test_sbc_deduction_reduces_fcff(drivers):
    without = dcf.project(1000.0, drivers, wacc=0.08, mid_year=False)
    with_sbc = dcf.project(
        1000.0, drivers, wacc=0.08, mid_year=False, sbc=20.0, deduct_sbc=True
    )
    assert with_sbc.rows[0].fcff_full_year < without.rows[0].fcff_full_year


def test_a_single_forecast_year_is_allowed():
    drivers = dcf.DriverPath.flat(
        1, revenue_growth=0.05, operating_margin=0.2, effective_tax_rate=0.2,
        capex_pct_revenue=0.05, dna_pct_revenue=0.04, dnwc_pct_drevenue=0.0,
    )
    assert len(dcf.project(100.0, drivers, wacc=0.09).rows) == 1


# ---------------------------------------------------------------------------
# Terminal value
# ---------------------------------------------------------------------------


def test_gordon_growth_by_hand():
    # 100 × 1.02 ÷ (0.08 − 0.02) = 1700
    assert dcf.gordon_growth_terminal_value(100, 0.02, 0.08) == pytest.approx(1700.0)


def test_gordon_raises_when_growth_meets_or_exceeds_wacc():
    with pytest.raises(dcf.TerminalValueError):
        dcf.gordon_growth_terminal_value(100, 0.08, 0.08)
    with pytest.raises(dcf.TerminalValueError):
        dcf.gordon_growth_terminal_value(100, 0.09, 0.08)


def test_gordon_never_returns_infinity_or_nan():
    """Nothing may propagate an infinity in place of raising."""
    for growth in (0.0799999, 0.079999999):
        value = dcf.gordon_growth_terminal_value(100, growth, 0.08)
        assert value == value and value not in (float("inf"), float("-inf"))


def test_exit_multiple_terminal_value():
    assert dcf.exit_multiple_terminal_value(300, 10) == 3000


def test_cross_implied_figures_invert_each_other():
    wacc, growth, fcff = 0.08, 0.02, 100.0
    terminal = dcf.gordon_growth_terminal_value(fcff, growth, wacc)
    assert dcf.implied_perpetuity_growth(terminal, fcff, wacc) == pytest.approx(growth)
    ebitda = 250.0
    multiple = dcf.implied_exit_multiple(terminal, ebitda)
    assert dcf.exit_multiple_terminal_value(ebitda, multiple) == pytest.approx(terminal)


def test_both_terminal_methods_use_the_final_year_convention(drivers):
    projection = dcf.project(1000.0, drivers, wacc=0.08, mid_year=True)
    terminals = dcf.build_terminal_values(
        projection, terminal_growth=0.02, exit_multiple=10.0
    )
    assert terminals["gordon"].discount_exponent == projection.rows[-1].discount_exponent
    assert terminals["exit_multiple"].discount_exponent == pytest.approx(
        terminals["gordon"].discount_exponent
    )


def test_terminal_ebitda_is_ebit_plus_dna(drivers):
    projection = dcf.project(1000.0, drivers, wacc=0.08)
    last = projection.rows[-1]
    assert projection.terminal_ebitda == pytest.approx(last.ebit + last.dna)


# ---------------------------------------------------------------------------
# Equity bridge
# ---------------------------------------------------------------------------


def test_bridge_arithmetic():
    bridge = dcf.build_equity_bridge(
        enterprise_value=10_000, total_debt=2_000, minority_interest=100,
        preferred_equity=50, cash=500, short_term_investments=300,
        non_operating_assets=200, shares=100,
        share_basis="diluted weighted-average",
    )
    assert bridge.equity_value == pytest.approx(10_000 - 2_000 - 100 - 50 + 500 + 300 + 200)
    assert bridge.value_per_share == pytest.approx(bridge.equity_value / 100)


def test_bridge_lines_sum_to_equity_value():
    bridge = dcf.build_equity_bridge(
        enterprise_value=10_000, total_debt=2_000, minority_interest=None,
        preferred_equity=None, cash=500, short_term_investments=300,
        non_operating_assets=None, shares=100, share_basis="x",
    )
    total = sum(l.amount for l in bridge.lines if l.kind in ("start", "adjustment"))
    assert total == pytest.approx(bridge.equity_value)


def test_unreported_bridge_items_get_their_own_visible_row():
    """A missing component may not vanish into the total."""
    bridge = dcf.build_equity_bridge(
        enterprise_value=10_000, total_debt=2_000, minority_interest=None,
        preferred_equity=None, cash=500, short_term_investments=300,
        non_operating_assets=None, shares=100, share_basis="x",
    )
    unreported = [l for l in bridge.lines if not l.reported]
    assert {l.label for l in unreported} >= {
        "Less: minority interest", "Less: preferred equity", "Plus: non-operating assets"
    }
    assert all("Not reported" in l.note for l in unreported)


def test_zero_debt_bridge():
    bridge = dcf.build_equity_bridge(
        enterprise_value=5_000, total_debt=0, minority_interest=None,
        preferred_equity=None, cash=100, short_term_investments=None,
        non_operating_assets=None, shares=50, share_basis="x",
    )
    assert bridge.equity_value == pytest.approx(5_100)


def test_excess_cash_toggle_moves_the_bridge():
    common = dict(
        enterprise_value=10_000, total_debt=0, minority_interest=None,
        preferred_equity=None, cash=500, short_term_investments=300,
        non_operating_assets=None, shares=100, share_basis="x",
    )
    included = dcf.build_equity_bridge(**common, include_short_term_investments=True)
    excluded = dcf.build_equity_bridge(**common, include_short_term_investments=False)
    assert included.equity_value - excluded.equity_value == pytest.approx(300)


def test_operating_cash_floor_holds_cash_back():
    bridge = dcf.build_equity_bridge(
        enterprise_value=10_000, total_debt=0, minority_interest=None,
        preferred_equity=None, cash=500, short_term_investments=None,
        non_operating_assets=None, shares=100, share_basis="x",
        operating_cash_floor=200,
    )
    assert bridge.equity_value == pytest.approx(10_300)


def test_cash_floor_never_makes_cash_negative():
    bridge = dcf.build_equity_bridge(
        enterprise_value=10_000, total_debt=0, minority_interest=None,
        preferred_equity=None, cash=100, short_term_investments=None,
        non_operating_assets=None, shares=100, share_basis="x",
        operating_cash_floor=500,
    )
    assert bridge.equity_value == pytest.approx(10_000)


def test_bridge_requires_a_positive_share_count():
    for shares in (0, -10, float("nan")):
        with pytest.raises(dcf.ProjectionError):
            dcf.build_equity_bridge(
                enterprise_value=1_000, total_debt=None, minority_interest=None,
                preferred_equity=None, cash=None, short_term_investments=None,
                non_operating_assets=None, shares=shares, share_basis="x",
            )


def test_upside_against_the_current_price():
    bridge = dcf.build_equity_bridge(
        enterprise_value=10_000, total_debt=0, minority_interest=None,
        preferred_equity=None, cash=0, short_term_investments=None,
        non_operating_assets=None, shares=100, share_basis="x",
    )
    assert bridge.upside_vs(80) == pytest.approx(0.25)
    assert bridge.upside_vs(0) is None
