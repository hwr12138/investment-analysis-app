"""Every formula in sections 5-8 of the spec, against hand-calculated fixtures."""

from __future__ import annotations

import inspect
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
# Timing
# ---------------------------------------------------------------------------


def test_elapsed_years_from_the_base_period_end():
    assert dcf.elapsed_years(date(2026, 6, 27), date(2026, 9, 26)) == pytest.approx(91 / 365)
    assert dcf.elapsed_years(date(2026, 9, 26), date(2026, 9, 26)) == 0.0


def test_elapsed_years_rejects_a_base_period_in_the_future():
    with pytest.raises(dcf.ProjectionError):
        dcf.elapsed_years(date(2026, 10, 1), date(2026, 9, 30))


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
        nwc_pct_revenue=0.10,
    )


def test_projection_year_one_by_hand(drivers):
    projection = dcf.project(1000.0, drivers, wacc=0.08, mid_year=False)
    row = projection.rows[0]
    assert row.revenue == pytest.approx(1100.0)
    assert row.ebit == pytest.approx(220.0)
    assert row.nopat == pytest.approx(165.0)
    assert row.dna == pytest.approx(44.0)
    assert row.capex == pytest.approx(55.0)
    assert row.delta_nwc == pytest.approx(10.0)
    # 165 + 44 − 55 − 10 = 144
    assert row.fcff == pytest.approx(144.0)
    assert row.present_value == pytest.approx(144.0 / 1.08)


def test_exponents_count_from_the_base_period_end(drivers):
    """Mid-year, a quarter of a year after the base period: 0.25, 1.25, 2.25."""
    projection = dcf.project(1000.0, drivers, wacc=0.08, elapsed=0.25, mid_year=True)
    assert [round(r.discount_exponent, 4) for r in projection.rows] == [0.25, 1.25, 2.25]
    assert [r.cumulative_time for r in projection.rows] == [1.0, 2.0, 3.0]


def test_every_forecast_year_is_a_whole_year(drivers):
    """No stub: year 1 is the twelve months after the base period, in full."""
    early = dcf.project(1000.0, drivers, wacc=0.08, elapsed=0.0, mid_year=False)
    late = dcf.project(1000.0, drivers, wacc=0.08, elapsed=0.9, mid_year=False)
    assert [r.fcff for r in late.rows] == pytest.approx([r.fcff for r in early.rows])


def test_value_rolls_forward_smoothly_with_the_valuation_date(drivers):
    """DCF_MODEL_REVIEW.md F4: nothing jumps when a fiscal year end passes.

    The only thing the valuation date moves is the roll-forward, so one more
    day is worth exactly one more day of the WACC — whichever side of a fiscal
    year end that day falls on. The stub convention this replaces dropped
    about five points of value overnight at the fiscal year end.
    """
    wacc = 0.09
    terminal = dict(terminal_growth=0.025, exit_multiple=12.0)
    base_end = date(2026, 6, 27)
    values = []
    for day in (date(2026, 9, 26), date(2026, 9, 27), date(2026, 10, 4), date(2026, 10, 5)):
        projection = dcf.project(
            1000.0, drivers, wacc=wacc, elapsed=dcf.elapsed_years(base_end, day)
        )
        terminals = dcf.build_terminal_values(projection, **terminal)
        values.append(projection.pv_explicit + terminals["gordon"].present_value)
    one_day = (1 + wacc) ** (1 / 365)
    assert values[1] / values[0] == pytest.approx(one_day)
    assert values[3] / values[2] == pytest.approx(one_day)


def test_driver_path_rejects_mismatched_lengths():
    with pytest.raises(dcf.ProjectionError):
        dcf.DriverPath(
            revenue_growth=(0.1, 0.1),
            operating_margin=(0.2,),
            effective_tax_rate=(0.25,),
            capex_pct_revenue=(0.05,),
            dna_pct_revenue=(0.04,),
            nwc_pct_revenue=(0.1,),
        )


def test_projection_rejects_missing_base_revenue(drivers):
    with pytest.raises(dcf.ProjectionError):
        dcf.project(float("nan"), drivers, wacc=0.08)


def test_a_single_forecast_year_is_allowed():
    drivers = dcf.DriverPath.flat(
        1, revenue_growth=0.05, operating_margin=0.2, effective_tax_rate=0.2,
        capex_pct_revenue=0.05, dna_pct_revenue=0.04, nwc_pct_revenue=0.0,
    )
    assert len(dcf.project(100.0, drivers, wacc=0.09).rows) == 1


def test_a_flat_growth_path_repeats_the_rate():
    assert dcf.growth_path(0.08, 4) == (0.08, 0.08, 0.08, 0.08)


def test_a_faded_growth_path_steps_toward_the_terminal_rate():
    """Equal steps that would land on the terminal rate in year N + 1."""
    path = dcf.growth_path(0.20, 3, fade_to=0.02)
    assert path == pytest.approx((0.20, 0.14, 0.08))
    # The step into the terminal year is the same size as every other.
    assert path[-1] - 0.02 == pytest.approx(path[0] - path[1])


# ---------------------------------------------------------------------------
# Stock-based compensation
# ---------------------------------------------------------------------------


def test_sbc_is_expensed_once_and_cannot_be_added_back(drivers):
    """DCF_ROUND2_INSTRUCTIONS.md §2.7. The margin is GAAP operating income,
    which already deducts SBC; the add-back option is gone, so FCFF is exactly
    NOPAT + D&A − capex − ΔNWC and there is no SBC argument to pass."""
    projection = dcf.project(1000.0, drivers, wacc=0.08, mid_year=False)
    for row in projection.rows:
        assert row.fcff == pytest.approx(row.nopat + row.dna - row.capex - row.delta_nwc)
    assert "add_back_sbc" not in inspect.signature(dcf.project).parameters
    assert "sbc" not in inspect.signature(dcf.project).parameters


# ---------------------------------------------------------------------------
# Terminal value
# ---------------------------------------------------------------------------


def test_gordon_growth_by_hand():
    # FCFF_{N+1} of 102 ÷ (0.08 − 0.02) = 1700
    assert dcf.gordon_growth_terminal_value(102, 0.02, 0.08) == pytest.approx(1700.0)


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


def test_terminal_year_reinvests_g_over_ronic(drivers):
    """DCF_MODEL_REVIEW.md F5: reinvestment follows g, not the forecast's growth."""
    projection = dcf.project(1000.0, drivers, wacc=0.08, mid_year=False)
    steady = dcf.terminal_year(projection, growth=0.02, ronic=0.10)
    # Revenue 1,331 × 1.02; margin 20%; tax 25%.
    assert steady.revenue == pytest.approx(1331.0 * 1.02)
    assert steady.nopat == pytest.approx(1331.0 * 1.02 * 0.20 * 0.75)
    assert steady.reinvestment_rate == pytest.approx(0.2)
    assert steady.fcff == pytest.approx(steady.nopat * 0.8)


def test_terminal_year_can_take_the_marginal_tax_rate(drivers):
    projection = dcf.project(1000.0, drivers, wacc=0.08, mid_year=False)
    steady = dcf.terminal_year(projection, growth=0.02, ronic=0.10, tax_rate=0.21)
    assert steady.tax_rate == 0.21
    assert steady.nopat == pytest.approx(steady.ebit * 0.79)


def test_terminal_year_rejects_a_non_positive_ronic(drivers):
    projection = dcf.project(1000.0, drivers, wacc=0.08)
    with pytest.raises(dcf.TerminalValueError):
        dcf.terminal_year(projection, growth=0.02, ronic=0.0)


def test_ronic_at_wacc_makes_terminal_growth_worth_nothing(drivers):
    """At RONIC = WACC the value is NOPAT_{N+1} ÷ WACC, whatever g is."""
    projection = dcf.project(1000.0, drivers, wacc=0.08, mid_year=False)
    for growth in (0.0, 0.01, 0.03):
        terminals = dcf.build_terminal_values(
            projection, terminal_growth=growth, exit_multiple=10.0
        )
        steady = terminals["gordon"].terminal_year
        assert terminals["gordon"].value == pytest.approx(steady.nopat / 0.08)


def test_cross_implied_figures_invert_each_other():
    wacc, growth, nopat_next = 0.08, 0.02, 100.0
    # RONIC recovered from a value-driver terminal value.
    terminal = nopat_next * (1 - growth / 0.15) / (wacc - growth)
    assert dcf.implied_ronic(terminal, nopat_next, growth, wacc) == pytest.approx(0.15)
    # Growth recovered from a zero-reinvestment perpetuity.
    free = (nopat_next / 1.02) * (1 + growth) / (wacc - growth)
    assert dcf.growth_required(free, nopat_next / 1.02, wacc) == pytest.approx(growth)
    ebitda = 250.0
    multiple = dcf.implied_exit_multiple(terminal, ebitda)
    assert dcf.exit_multiple_terminal_value(ebitda, multiple) == pytest.approx(terminal)


def test_a_value_beyond_free_growth_has_no_implied_ronic():
    """Worth more than NOPAT growing at g with no reinvestment: no RONIC reaches it."""
    free = 100.0 / (0.08 - 0.02)
    assert dcf.implied_ronic(free * 1.01, 100.0, 0.02, 0.08) is None


def test_exit_value_is_discounted_from_the_end_of_year_n(drivers):
    """DCF_MODEL_REVIEW.md F3: a sale price is struck at the end of year N.

    Only the Gordon value follows the mid-year convention; discounting the
    exit value from N − 0.5 overstated it by (1 + WACC)^0.5.
    """
    projection = dcf.project(1000.0, drivers, wacc=0.08, mid_year=True, elapsed=0.25)
    terminals = dcf.build_terminal_values(
        projection, terminal_growth=0.02, exit_multiple=10.0
    )
    assert terminals["gordon"].discount_exponent == pytest.approx(2.25)
    assert terminals["exit_multiple"].discount_exponent == pytest.approx(2.75)
    assert terminals["exit_multiple"].present_value == pytest.approx(
        terminals["exit_multiple"].value / 1.08**2.75
    )


def test_final_ebitda_is_ebit_plus_dna(drivers):
    projection = dcf.project(1000.0, drivers, wacc=0.08)
    last = projection.rows[-1]
    assert projection.final_ebitda == pytest.approx(last.ebit + last.dna)


def test_a_hand_computed_valuation_end_to_end(drivers):
    """An oracle, not a snapshot: every figure below is worked by hand.

    Revenue 1,000 growing 10% for three years at a 20% margin, 25% tax, capex
    5% and D&A 4% of revenue, working capital 10% of the change in revenue;
    WACC 8%, end-of-year discounting, valued on the base period's last day.

        FCFF   Y1 = 165.00 + 44.00 − 55.00 − 10.0 = 144.00
               Y2 = 181.50 + 48.40 − 60.50 − 11.0 = 158.40
               Y3 = 199.65 + 53.24 − 66.55 − 12.1 = 174.24
        Year 4 = 1,331 × 1.02 × 20% × 75% = 203.643 of NOPAT, reinvesting
                 2% ÷ 10% RONIC of it → FCFF 162.9144 → TV 2,715.24
        Exit   = (266.20 + 53.24) × 10 = 3,194.40
    """
    projection = dcf.project(1000.0, drivers, wacc=0.08, mid_year=False)
    terminals = dcf.build_terminal_values(
        projection, terminal_growth=0.02, exit_multiple=10.0, ronic=0.10
    )
    assert [row.fcff for row in projection.rows] == pytest.approx([144.0, 158.4, 174.24])
    assert projection.pv_explicit == pytest.approx(
        144.0 / 1.08 + 158.4 / 1.08**2 + 174.24 / 1.08**3
    )
    assert terminals["gordon"].value == pytest.approx(2715.24)
    assert terminals["gordon"].present_value == pytest.approx(2715.24 / 1.08**3)
    assert terminals["exit_multiple"].value == pytest.approx(3194.4)
    assert terminals["exit_multiple"].present_value == pytest.approx(3194.4 / 1.08**3)


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


# ---------------------------------------------------------------------------
# Hold-then-fade, margin path, incremental return (DCF_ROUND2_INSTRUCTIONS.md §2.3)
# ---------------------------------------------------------------------------


def test_a_hold_of_one_is_exactly_the_linear_fade():
    for years in (3, 5, 10, 15):
        linear = tuple(0.30 + (0.025 - 0.30) * i / years for i in range(years))
        assert dcf.growth_path(0.30, years, 0.025, hold=1) == pytest.approx(linear)


def test_growth_is_held_then_fades_to_terminal_growth_in_year_n_plus_one():
    path = dcf.growth_path(0.30, 10, 0.025, hold=3)
    assert path[:3] == (0.30, 0.30, 0.30)
    step = (0.025 - 0.30) / (10 - 3 + 1)
    assert path[3] == pytest.approx(0.30 + step)
    # One more step from year 10 lands exactly on the terminal rate.
    assert path[-1] + step == pytest.approx(0.025)


def test_a_hold_beyond_the_horizon_is_refused():
    with pytest.raises(dcf.ProjectionError):
        dcf.growth_path(0.30, 5, 0.025, hold=6)


def test_the_margin_moves_linearly_to_its_target_in_year_n():
    assert dcf.linear_path(0.20, 0.30, 6) == pytest.approx((0.20, 0.22, 0.24, 0.26, 0.28, 0.30))
    assert dcf.linear_path(0.25, 0.25, 4) == (0.25, 0.25, 0.25, 0.25)
    assert dcf.linear_path(0.25, 0.40, 1) == (0.25,)


def test_incremental_return_on_capital_by_hand(drivers):
    """(NOPAT_3 − NOPAT_base) ÷ Σ (capex − D&A + ΔNWC).

    NOPAT_3 = 199.65; base NOPAT 150. Reinvested: (55 − 44 + 10) + (60.5 − 48.4
    + 11) + (66.55 − 53.24 + 12.1) = 21 + 23.1 + 25.41 = 69.51.
    """
    projection = dcf.project(1000.0, drivers, wacc=0.08, mid_year=False)
    assert dcf.incremental_return_on_capital(projection, 150.0) == pytest.approx(
        (199.65 - 150.0) / 69.51
    )


def test_incremental_return_is_not_meaningful_without_net_reinvestment():
    free = dcf.DriverPath.flat(
        3, revenue_growth=0.10, operating_margin=0.20, effective_tax_rate=0.25,
        capex_pct_revenue=0.03, dna_pct_revenue=0.05, nwc_pct_revenue=0.0,
    )
    projection = dcf.project(1000.0, free, wacc=0.08)
    assert dcf.incremental_return_on_capital(projection, 150.0) is None
    assert dcf.incremental_return_on_capital(projection, None) is None


# ---------------------------------------------------------------------------
# Round 4: NWC as a level ratio (§2.1) and the EBITA proxy (§2.3)
# ---------------------------------------------------------------------------


def test_nwc_moves_with_the_change_in_revenue():
    """ΔNWC = NWC % of revenue × Δrevenue."""
    drivers = dcf.DriverPath.flat(
        2, revenue_growth=0.10, operating_margin=0.20, effective_tax_rate=0.25,
        capex_pct_revenue=0.05, dna_pct_revenue=0.04, nwc_pct_revenue=0.08,
    )
    rows = dcf.project(1000.0, drivers, wacc=0.08).rows
    assert rows[0].delta_nwc == pytest.approx(0.08 * 100.0)
    assert rows[1].delta_nwc == pytest.approx(0.08 * 110.0)


def test_negative_working_capital_releases_cash_as_revenue_grows():
    """AAPL and AMZN: customers fund the business, so growth frees cash."""
    funded = dcf.DriverPath.flat(
        1, revenue_growth=0.10, operating_margin=0.20, effective_tax_rate=0.25,
        capex_pct_revenue=0.05, dna_pct_revenue=0.04, nwc_pct_revenue=-0.10,
    )
    row = dcf.project(1000.0, funded, wacc=0.08).rows[0]
    assert row.delta_nwc == pytest.approx(-10.0)
    assert row.fcff == pytest.approx(row.nopat + row.dna - row.capex + 10.0)


def _amortising(dna: float, capex: float) -> dcf.Projection:
    drivers = dcf.DriverPath.flat(
        3, revenue_growth=0.05, operating_margin=0.25, effective_tax_rate=0.15,
        capex_pct_revenue=capex, dna_pct_revenue=dna, nwc_pct_revenue=0.0,
    )
    return dcf.project(1000.0, drivers, wacc=0.09, mid_year=False)


def test_the_ebita_add_back_is_zero_when_capex_covers_dna():
    steady = dcf.terminal_year(_amortising(dna=0.04, capex=0.06), growth=0.02, ronic=0.09)
    assert steady.amortisation == 0.0


def test_the_ebita_add_back_is_dna_less_capex_grown_and_taxed():
    """NOPAT_{N+1} = (EBIT_{N+1} + max(D&A_N − capex_N, 0) × (1 + g)) × (1 − t)."""
    projection = _amortising(dna=0.16, capex=0.01)
    final = projection.final_row
    steady = dcf.terminal_year(projection, growth=0.02, ronic=0.09, tax_rate=0.25)
    addback = (final.dna - final.capex) * 1.02
    assert steady.amortisation == pytest.approx(addback)
    assert steady.nopat == pytest.approx((steady.ebit + addback) * 0.75)
    assert steady.fcff == pytest.approx(steady.nopat * (1 - 0.02 / 0.09))


def test_the_ebita_proxy_off_reproduces_round_three():
    projection = _amortising(dna=0.16, capex=0.01)
    off = dcf.terminal_year(projection, growth=0.02, ronic=0.09, ebita_proxy=False)
    assert off.amortisation == 0.0
    assert off.nopat == pytest.approx(off.ebit * (1 - projection.final_row.tax_rate))
    on = dcf.build_terminal_values(projection, terminal_growth=0.02, exit_multiple=12.0)
    plain = dcf.build_terminal_values(
        projection, terminal_growth=0.02, exit_multiple=12.0, ebita_proxy=False
    )
    assert on["gordon"].value > plain["gordon"].value
    assert on["exit_multiple"].value == plain["exit_multiple"].value
