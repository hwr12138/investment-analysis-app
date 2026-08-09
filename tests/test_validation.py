"""Every validation check, proven to fire at the right severity.

A check that never fires is worse than no check, so each one gets a case that
trips it and — where the distinction matters — a case that does not.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.logic.dcf import validation as val

TOL = val.DEFAULT_TOLERANCES


#: Long enough by default that the history check passes, so any other check
#: firing against the baseline is firing wrongly.
BASELINE_YEARS = 12


def _annual(years: int = BASELINE_YEARS, **overrides) -> pd.DataFrame:
    """Clean, internally consistent fiscal years.

    Every identity the checks test holds exactly: assets equal liabilities plus
    equity, the cash flow ties, and EBIT reconciles to its components. Overrides
    poison one field at a time.
    """
    fiscal_years = list(range(2026 - years, 2026))
    scale = [1.0 + 0.1 * i for i in range(years)]

    def _s(value: float) -> list[float]:
        return [round(value * factor, 6) for factor in scale]

    base = {
        "revenue": _s(1000.0),
        "cost_of_revenue": _s(-600.0),
        "operating_expenses": _s(-150.0),
        "ebit": _s(250.0),
        "pretax_income": _s(240.0),
        "tax_expense": _s(-48.0),
        "net_income": _s(192.0),
        "cash": _s(100.0),
        "short_term_investments": _s(50.0),
        "total_current_assets": _s(400.0),
        "total_current_liabilities": _s(300.0),
        "short_term_debt": _s(40.0),
        "long_term_debt": _s(200.0),
        "total_assets": _s(900.0),
        "total_liabilities": _s(600.0),
        "total_equity": _s(300.0),
        "cfo": _s(280.0),
        "cfi": _s(-100.0),
        "cff": _s(-160.0),
        "fx_effect": _s(0.0),
        "net_change_in_cash": _s(20.0),
        "dna": _s(60.0),
        "capex_acquisition": _s(-80.0),
        "derived_total_debt": _s(240.0),
        "minority_interest": _s(0.0),
    }
    for key, value in overrides.items():
        base[key] = value if isinstance(value, list) else _s(value)
    return pd.DataFrame(base, index=fiscal_years)


def _poison(key: str, value: float) -> list[float]:
    """A full-length column with only the most recent year corrupted."""
    column = _annual()[key].tolist()
    column[-1] = value
    return column


def _context(**overrides) -> val.ValidationContext:
    defaults = dict(
        ticker="TEST",
        company="TESTCO",
        annual=_annual(),
        currencies_seen=("USD",),
        asreported_flags=(False,),
        valuation_date=date(2026, 1, 10),
        price_date=date(2026, 1, 8),
        last_price=100.0,
        reported_market_cap=100_000.0,
        shares_point_in_time=1000.0,
        shares_point_in_time_date=date(2025, 12, 1),
        shares_from_prices=1000.0,
        shares_weighted_diluted=1010.0,
        shares_eps_implied=1010.0,
        effective_tax_rate=0.20,
        beta_r_squared=0.45,
        beta_observations=60,
        equity_risk_premium=0.045,
        terminal_value_share=0.60,
        terminal_growth=0.02,
        risk_free_rate=0.04,
        fcff_method_a=100.0,
        fcff_method_b=102.0,
        latest_revenue=1200.0,
        debt_weight=0.05,
        balance_sheet_delta_nwc=10.0,
    )
    defaults.update(overrides)
    return val.ValidationContext(**defaults)


# ---------------------------------------------------------------------------
# The clean baseline must not trip anything
# ---------------------------------------------------------------------------


def test_clean_context_produces_no_issues():
    issues, results = val.run_all(_context(), TOL)
    assert issues == [], [i.check for i in issues]
    assert all(r.status in ("pass", "not evaluated") for r in results)


def test_every_check_appears_in_the_result_list():
    _, results = val.run_all(_context(), TOL)
    assert len(results) == len(val.ALL_CHECKS)


# ---------------------------------------------------------------------------
# CRITICAL
# ---------------------------------------------------------------------------


def test_balance_sheet_imbalance_is_critical():
    context = _context(annual=_annual(total_equity=_poison("total_equity", 999.0)))
    issues, result = val.check_balance_sheet_balances(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL
    assert issues[0].period == "FY2025"
    assert issues[0].issue_type == "ReconciliationIssue"
    assert "Total assets" in issues[0].values
    assert "Liabilities + equity + NCI" in issues[0].values
    assert result.status == "fail"


def test_cash_flow_that_does_not_tie_is_critical():
    context = _context(annual=_annual(net_change_in_cash=_poison("net_change_in_cash", 900.0)))
    issues, _ = val.check_cash_flow_ties(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL


def test_missing_fx_line_passes_when_the_identity_still_ties():
    """A company with no translation exposure must not be flagged."""
    context = _context(annual=_annual(fx_effect=[float("nan")] * BASELINE_YEARS))
    issues, result = val.check_cash_flow_ties(context, TOL)
    assert issues == []
    assert "without an FX line" in result.detail


def test_missing_fx_line_is_reported_as_unexplained_when_it_does_not_tie():
    context = _context(
        annual=_annual(fx_effect=[float("nan")] * BASELINE_YEARS,
                       net_change_in_cash=_poison("net_change_in_cash", 500.0))
    )
    issues, _ = val.check_cash_flow_ties(context, TOL)
    assert len(issues) == 1
    assert "unexplained" in issues[0].message
    assert issues[0].values["FX translation"] == "not reported"


def test_quarters_that_do_not_sum_to_the_year_is_critical():
    annual_revenue = _annual().loc[2025, "revenue"]
    quarterly = pd.DataFrame(
        {"revenue": [annual_revenue / 4] * 4},
        index=["FY2025 Q1", "FY2025 Q2", "FY2025 Q3", "FY2025 Q4"],
    )
    assert val.check_quarters_sum_to_year(_context(quarterly=quarterly), TOL)[0] == []

    quarterly.loc["FY2025 Q4", "revenue"] = annual_revenue  # now 175% of the year
    issues, _ = val.check_quarters_sum_to_year(_context(quarterly=quarterly), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL
    assert issues[0].period == "FY2025"


def test_api_ttm_diverging_from_manual_sum_is_critical():
    context = _context(
        api_ttm=pd.Series({"revenue": 1200.0, "ebit": 310.0}),
        manual_ttm=pd.Series({"revenue": 1500.0, "ebit": 310.0}),
    )
    issues, _ = val.check_api_ttm_matches_manual(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL
    assert issues[0].concept == "TTM revenue"


def test_api_ttm_matching_manual_sum_passes():
    series = pd.Series({"revenue": 1200.0, "ebit": 310.0})
    issues, result = val.check_api_ttm_matches_manual(
        _context(api_ttm=series, manual_ttm=series.copy()), TOL
    )
    assert issues == []
    assert result.status == "pass"


def test_ebit_that_does_not_reconcile_is_critical():
    context = _context(annual=_annual(ebit=_poison("ebit", 999.0)))
    issues, _ = val.check_ebit_reconciles(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL


def test_non_controlling_interest_is_included_in_the_balancing_identity():
    """Sharadar's equity is parent-only, so NCI has to be added back."""
    annual = _annual()
    annual["total_equity"] = annual["total_equity"] - 25.0
    annual["minority_interest"] = 25.0
    issues, result = val.check_balance_sheet_balances(_context(annual=annual), TOL)
    assert issues == []
    assert result.status == "pass"


def test_total_debt_mismatch_is_critical():
    context = _context(
        annual=_annual(derived_total_debt=_poison("derived_total_debt", 99999.0))
    )
    issues, _ = val.check_total_debt_reconciles(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL
    assert issues[0].values["Current debt"] == "84"


def test_share_count_scaling_error_is_critical():
    """The prices endpoint has returned counts off by orders of magnitude."""
    context = _context(shares_from_prices=1000.0, shares_point_in_time=1_000_000.0)
    issues, _ = val.check_share_counts_agree(context, TOL)
    assert issues
    assert all(i.severity is val.Severity.CRITICAL for i in issues)


def test_point_in_time_and_weighted_bases_are_compared_separately():
    """A 1% gap between the two bases is normal and must not read as CRITICAL."""
    context = _context(shares_point_in_time=1000.0, shares_weighted_diluted=1010.0)
    critical, _ = val.check_share_counts_agree(context, TOL)
    basis, _ = val.check_share_basis_difference(context, TOL)
    assert critical == []
    assert basis == []


def test_a_large_share_basis_gap_is_a_warning():
    context = _context(shares_point_in_time=1000.0, shares_weighted_diluted=1400.0)
    issues, _ = val.check_share_basis_difference(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING


def test_weighted_share_sources_disagreeing_is_critical():
    context = _context(shares_weighted_diluted=1000.0, shares_eps_implied=1500.0)
    issues, _ = val.check_weighted_share_counts_agree(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL


def test_stale_price_is_critical():
    context = _context(price_date=date(2025, 9, 1), valuation_date=date(2026, 1, 10))
    issues, _ = val.check_price_is_current(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL
    assert issues[0].issue_type == "StaleDataIssue"
    assert "131 days" in issues[0].values["Age"]


def test_fresh_price_passes():
    issues, result = val.check_price_is_current(_context(), TOL)
    assert issues == []
    assert result.status == "pass"


def test_market_cap_mismatch_is_critical():
    context = _context(reported_market_cap=500_000.0)  # implied is 100,000
    issues, _ = val.check_market_cap_agrees(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL


def test_fcff_method_divergence_is_critical():
    context = _context(fcff_method_a=100.0, fcff_method_b=200.0)
    issues, _ = val.check_fcff_methods_agree(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL
    assert issues[0].values["Divergence"] == "50.00%"


def test_fcff_methods_within_tolerance_pass():
    issues, result = val.check_fcff_methods_agree(_context(), TOL)
    assert issues == []
    assert result.status == "pass"


def test_uncomputable_fcff_method_is_not_evaluated_rather_than_passed():
    """Apple reports no interest expense; that must not read as a silent pass."""
    _, result = val.check_fcff_methods_agree(_context(fcff_method_a=None), TOL)
    assert result.status == "not evaluated"


def test_mixed_currency_is_critical():
    issues, _ = val.check_currency_consistent(_context(currencies_seen=("USD", "EUR")), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL


def test_mixed_asreported_basis_is_critical():
    issues, _ = val.check_asreported_consistent(_context(asreported_flags=(True, False)), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL


# ---------------------------------------------------------------------------
# WARNING, and the one BLOCKING case
# ---------------------------------------------------------------------------


def test_zero_usable_history_is_blocking():
    context = _context(annual=_annual(revenue=[float("nan")] * BASELINE_YEARS))
    issues, result = val.check_sufficient_history(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.BLOCKING
    assert result.severity is val.Severity.BLOCKING


def test_short_history_is_only_a_warning():
    issues, _ = val.check_sufficient_history(_context(annual=_annual(years=3)), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING
    assert issues[0].issue_type == "InsufficientHistoryIssue"


def test_sufficient_history_passes():
    issues, result = val.check_sufficient_history(_context(), TOL)
    assert issues == []
    assert result.status == "pass"


@pytest.mark.parametrize("rate", [-0.15, 0.80])
def test_implausible_effective_tax_rate_is_a_warning(rate):
    issues, _ = val.check_etr_plausible(_context(effective_tax_rate=rate), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING
    assert issues[0].issue_type == "ImplausibleValueIssue"


def test_missing_effective_tax_rate_is_not_evaluated():
    _, result = val.check_etr_plausible(_context(effective_tax_rate=None), TOL)
    assert result.status == "not evaluated"


def test_low_beta_r_squared_is_a_warning():
    issues, _ = val.check_beta_quality(_context(beta_r_squared=0.03), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING


@pytest.mark.parametrize("erp", [0.005, 0.25])
def test_implausible_erp_is_a_warning(erp):
    issues, _ = val.check_erp_plausible(_context(equity_risk_premium=erp), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING


def test_high_terminal_value_share_is_a_warning():
    issues, _ = val.check_terminal_value_share(_context(terminal_value_share=0.92), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING
    assert issues[0].values["Terminal value share"] == "92.0%"


# --- The terminal share is judged against its own horizon -------------------
#
# A five-year forecast is mechanically more terminal-heavy than a ten-year one
# at the same discount rate. Judging both against one flat number fired on
# almost every short model and stayed quiet on long models that really were
# terminal-driven.


def _tv_context(share: float, years: int, wacc: float = 0.09, growth: float = 0.025):
    return _context(
        terminal_value_share=share,
        forecast_years=years,
        wacc=wacc,
        terminal_growth=growth,
    )


def test_a_default_five_year_model_does_not_trip_the_terminal_share_check():
    """The regression this scaling exists for.

    At the page's default five-year horizon and a typical WACC the arithmetic
    alone puts ~72% of value in the terminal, so a 78% model is unremarkable and
    used to raise a warning purely for being short.
    """
    issues, result = val.check_terminal_value_share(_tv_context(0.78, years=5), TOL)
    assert issues == []
    assert result.status == "pass"


def test_a_genuinely_terminal_heavy_five_year_model_still_warns():
    issues, _ = val.check_terminal_value_share(_tv_context(0.88, years=5), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING
    assert issues[0].values["Forecast years"] == "5"


def test_a_long_horizon_tightens_rather_than_loosens_the_ceiling():
    """70% terminal passed the old flat 75% threshold; over ten years it should not.

    The scaling is not a blanket relaxation — at a long horizon the same share
    means the projection is contributing far less than it should be.
    """
    issues, _ = val.check_terminal_value_share(_tv_context(0.70, years=10), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING


@pytest.mark.parametrize(
    "years, wacc, growth",
    [
        (None, 0.09, 0.025),   # horizon unknown
        (5, None, 0.025),      # no WACC
        (5, 0.09, None),       # no terminal growth
        (5, 0.02, 0.025),      # growth above WACC — the Gordon form has no value
    ],
)
def test_the_fixed_threshold_is_used_when_the_baseline_cannot_be_computed(
    years, wacc, growth
):
    ceiling, basis = TOL.terminal_value_share_ceiling(
        years=years, wacc=wacc, growth=growth
    )
    assert ceiling == TOL.terminal_value_share
    assert "fixed" in basis


def test_the_ceiling_falls_as_the_horizon_lengthens():
    ceilings = [
        TOL.terminal_value_share_ceiling(years=y, wacc=0.09, growth=0.025)[0]
        for y in (5, 7, 10, 15)
    ]
    assert ceilings == sorted(ceilings, reverse=True)


def test_the_issue_states_the_basis_it_judged_against():
    """A bare ceiling would leave the user unable to tell why it moved."""
    issues, _ = val.check_terminal_value_share(_tv_context(0.95, years=5), TOL)
    assert "5-year horizon" in issues[0].tolerance
    assert "9.0% WACC" in issues[0].tolerance


def test_terminal_growth_above_the_economy_is_a_warning():
    issues, _ = val.check_terminal_growth_reasonable(
        _context(terminal_growth=0.09, risk_free_rate=0.04), TOL
    )
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING


def test_terminal_growth_below_the_risk_free_rate_passes():
    issues, _ = val.check_terminal_growth_reasonable(
        _context(terminal_growth=0.03, risk_free_rate=0.045), TOL
    )
    assert issues == []


def test_large_sbc_share_is_a_warning():
    issues, _ = val.check_sbc_share(_context(sbc=200.0, latest_revenue=1200.0), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING


def test_absent_sbc_is_not_evaluated_rather_than_passed():
    _, result = val.check_sbc_share(_context(sbc=None), TOL)
    assert result.status == "not evaluated"
    assert "no stock-based compensation" in result.detail


def test_a_zero_valued_line_item_is_a_warning():
    """Sharadar writes 0 both for a genuine nil and for an undisclosed line."""
    issues, result = val.check_zero_filled_indicators(
        _context(zero_filled=(("Interest Expense", "Interest Expense"),)), TOL
    )
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING
    assert "Interest Expense" in issues[0].values
    assert result.status == "fail"


def test_no_zero_valued_line_items_passes():
    issues, result = val.check_zero_filled_indicators(_context(zero_filled=()), TOL)
    assert issues == []
    assert result.status == "pass"


def test_stale_share_count_is_a_warning():
    issues, _ = val.check_share_count_currency_of_date(
        _context(shares_point_in_time_date=date(2025, 1, 1)), TOL
    )
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING
    assert issues[0].issue_type == "StaleDataIssue"


def test_heavy_book_debt_is_a_warning():
    issues, _ = val.check_book_debt_proxy(_context(debt_weight=0.45), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING
    assert "double-count" in issues[0].message


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------


def test_failures_aggregate_rather_than_short_circuiting():
    context = _context(
        annual=_annual(total_equity=_poison("total_equity", 999.0)),
        equity_risk_premium=0.5,
        terminal_value_share=0.95,
        beta_r_squared=0.01,
        effective_tax_rate=0.95,
        fcff_method_a=100.0,
        fcff_method_b=400.0,
        price_date=date(2024, 1, 1),
    )
    issues, _ = val.run_all(context, TOL)
    checks = {i.check for i in issues}
    assert len(issues) >= 6, checks
    assert "Balance sheet balances" in checks
    assert "Equity risk premium plausible" in checks
    assert "Terminal value share of enterprise value" in checks
    assert "FCFF method A agrees with method B" in checks
    assert "Price series is current" in checks


def test_issues_are_sorted_most_severe_first():
    context = _context(
        annual=_annual(total_equity=_poison("total_equity", 999.0)), equity_risk_premium=0.5
    )
    issues, _ = val.run_all(context, TOL)
    severities = [i.severity for i in issues]
    assert severities == sorted(
        severities,
        key=lambda s: {val.Severity.BLOCKING: 0, val.Severity.CRITICAL: 1, val.Severity.WARNING: 2}[s],
    )


def test_issue_keys_are_stable_across_runs():
    """The acknowledgement handle must not move between reruns."""
    first, _ = val.run_all(_context(annual=_annual(total_equity=_poison("total_equity", 999.0))), TOL)
    second, _ = val.run_all(_context(annual=_annual(total_equity=_poison("total_equity", 999.0))), TOL)
    assert [i.key for i in first] == [i.key for i in second]


def test_every_issue_names_its_figures_and_tolerance():
    context = _context(
        annual=_annual(total_equity=_poison("total_equity", 999.0)),
        fcff_method_a=100.0,
        fcff_method_b=250.0,
        terminal_value_share=0.95,
    )
    issues, _ = val.run_all(context, TOL)
    for issue in issues:
        assert issue.check
        assert issue.message
        assert issue.values, issue.check
        assert issue.tolerance, issue.check


def test_relative_gap_is_symmetric_and_safe():
    assert val.relative_gap(0, 0) == 0.0
    assert val.relative_gap(100, 110) == pytest.approx(val.relative_gap(110, 100))


def test_unsupported_sector_error_explains_itself():
    error = val.UnsupportedSectorError("JPM", "Banks", "BANKS")
    assert "JPM" in str(error)
    assert "Banks" in str(error)
    assert "meaningless" in str(error)


def test_tolerances_are_documented_in_one_table():
    frame = TOL.describe()
    assert {"Tolerance", "Default"} == set(frame.columns)
    assert len(frame) >= 15
