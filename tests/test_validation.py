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
        # These tests supply minority interest directly rather than deriving it
        # as the residual, so the balance-sheet identity is a real test here.
        minority_interest_derived=False,
        statement_currency="USD",
        price_currency="USD",
        asreported_flags=(False, False, False),
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


# --- Severity follows the window the model actually uses -------------------
#
# A break in FY1992 is real, but it cannot move a valuation whose drivers are
# averaged over the last five years. Reporting it at the same tier as a break
# in the base period drowns the ones that can.


ALL_PERIODS = frozenset(str(y) for y in range(2026 - BASELINE_YEARS, 2026))
RECENT_ONLY = frozenset(str(y) for y in range(2021, 2026))


def test_a_break_inside_the_driver_window_stays_critical():
    context = _context(
        annual=_annual(net_change_in_cash=_poison("net_change_in_cash", 900.0)),
        periods_in_use=RECENT_ONLY,
    )
    issues, result = val.check_cash_flow_ties(context, TOL)
    assert len(issues) == 1
    assert issues[0].period == "FY2025"
    assert issues[0].severity is val.Severity.CRITICAL
    assert result.severity is val.Severity.CRITICAL


def test_a_break_outside_the_driver_window_is_downgraded_to_warning():
    column = _annual()["net_change_in_cash"].tolist()
    column[0] = 900.0  # the oldest year, far outside a five-year window
    context = _context(
        annual=_annual(net_change_in_cash=column), periods_in_use=RECENT_ONLY
    )
    issues, result = val.check_cash_flow_ties(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING
    assert "does not feed the valuation shown" in issues[0].message
    assert result.severity is val.Severity.WARNING


def test_an_unknown_window_downgrades_nothing():
    """``None`` means "we do not know", and the conservative reading is CRITICAL."""
    column = _annual()["net_change_in_cash"].tolist()
    column[0] = 900.0
    context = _context(annual=_annual(net_change_in_cash=column), periods_in_use=None)
    issues, _ = val.check_cash_flow_ties(context, TOL)
    assert issues[0].severity is val.Severity.CRITICAL


def test_a_wide_window_keeps_old_periods_critical():
    """Selecting "Median (all years)" pulls old periods back into the model."""
    column = _annual()["net_change_in_cash"].tolist()
    column[0] = 900.0
    context = _context(
        annual=_annual(net_change_in_cash=column), periods_in_use=ALL_PERIODS
    )
    issues, _ = val.check_cash_flow_ties(context, TOL)
    assert issues[0].severity is val.Severity.CRITICAL


@pytest.mark.parametrize(
    "check, column, value",
    [
        (val.check_balance_sheet_balances, "total_assets", 9_999.0),
        (val.check_ebit_reconciles, "ebit", 9_999.0),
        (val.check_total_debt_reconciles, "derived_total_debt", 9_999.0),
    ],
)
def test_every_per_period_reconciliation_check_scopes_its_severity(check, column, value):
    poisoned = _annual()[column].tolist()
    poisoned[0] = value
    context = _context(
        annual=_annual(**{column: poisoned}), periods_in_use=RECENT_ONLY
    )
    issues, _ = check(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING


def test_the_roll_up_reports_the_worst_severity_actually_raised():
    """A CRITICAL summary over nothing but warnings would be its own lie."""
    column = _annual()["net_change_in_cash"].tolist()
    column[0] = 900.0   # outside the window -> WARNING
    column[-1] = 900.0  # inside it          -> CRITICAL
    context = _context(
        annual=_annual(net_change_in_cash=column), periods_in_use=RECENT_ONLY
    )
    issues, result = val.check_cash_flow_ties(context, TOL)
    assert {i.severity for i in issues} == {val.Severity.CRITICAL, val.Severity.WARNING}
    assert result.severity is val.Severity.CRITICAL


# --- Two FX presentations appear in the same feed --------------------------
#
# Sharadar folds the FX line into the reported net change in cash for recent
# vintages and reports the net change before it for older ones, switching
# mid-series (Microsoft changes over at FY2006). Testing only the first
# convention reported 13 critical failures against a history that reconciles
# exactly.


def _fx_excluded_annual(years: int = BASELINE_YEARS) -> pd.DataFrame:
    """Rows whose reported total is cfo+cfi+cff, with FX stated outside it."""
    frame = _annual(fx_effect=7.0)
    frame["net_change_in_cash"] = frame["cfo"] + frame["cfi"] + frame["cff"]
    return frame


def test_a_total_that_excludes_the_fx_line_still_ties():
    issues, result = val.check_cash_flow_ties(_context(annual=_fx_excluded_annual()), TOL)
    assert issues == []
    assert result.status == "pass"


def test_the_excluded_fx_convention_is_reported_rather_than_absorbed():
    """Silently accepting it would hide that the feed changed presentation."""
    _, result = val.check_cash_flow_ties(_context(annual=_fx_excluded_annual()), TOL)
    assert "excludes the FX line" in result.detail
    assert str(BASELINE_YEARS) in result.detail


def test_a_break_under_both_conventions_is_still_critical():
    """The FX escape hatch must not swallow a genuine reconciliation failure."""
    frame = _annual(fx_effect=7.0)
    frame.loc[2025, "net_change_in_cash"] = 900.0
    issues, _ = val.check_cash_flow_ties(_context(annual=frame), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL
    assert "either presentation" in issues[0].message


def test_a_failing_period_shows_the_gap_under_both_conventions():
    """Both are shown so it is clear the convention is not the explanation."""
    frame = _annual(fx_effect=7.0)
    frame.loc[2025, "net_change_in_cash"] = 900.0
    issues, _ = val.check_cash_flow_ties(_context(annual=frame), TOL)
    assert "Difference including FX" in issues[0].values
    assert "Difference excluding FX" in issues[0].values


def test_a_zero_fx_line_does_not_open_the_excluded_convention_path():
    """With fx == 0 the conventions coincide, so a break is a real break."""
    frame = _annual(fx_effect=0.0)
    frame.loc[2025, "net_change_in_cash"] = 900.0
    issues, _ = val.check_cash_flow_ties(_context(annual=frame), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL


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


def test_a_residual_minority_interest_is_not_counted_as_a_balanced_sheet():
    """DCF_MODEL_REVIEW.md F9. When NCI *is* assets − liabilities − equity, the
    identity holds by construction; reporting a pass would be false assurance."""
    context = _context(
        annual=_annual(total_equity=_poison("total_equity", 999.0)),
        minority_interest_derived=True,
    )
    issues, result = val.check_balance_sheet_balances(context, TOL)
    assert issues == []
    assert result.status == "not evaluated"
    assert "by construction" in result.detail


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


def test_fcff_method_divergence_is_a_warning():
    """DCF_MODEL_REVIEW.md F12: neither build feeds the projection, so a gap
    that fires for known reasons does not earn the red banner."""
    context = _context(fcff_method_a=100.0, fcff_method_b=200.0)
    issues, _ = val.check_fcff_methods_agree(context, TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.WARNING
    assert issues[0].values["Divergence"] == "50.00%"


def test_sbc_is_reconciled_out_of_the_cash_flow_build():
    """Method A adds SBC back through operating cash flow; B has expensed it."""
    context = _context(fcff_method_a=130.0, fcff_method_b=100.0, base_sbc=30.0)
    issues, result = val.check_fcff_methods_agree(context, TOL)
    assert issues == []
    assert "after SBC" in result.detail

    unexplained = _context(fcff_method_a=160.0, fcff_method_b=100.0, base_sbc=30.0)
    issues, _ = val.check_fcff_methods_agree(unexplained, TOL)
    assert issues[0].values["Unexplained residual"] == "30"


def test_fcff_methods_within_tolerance_pass():
    issues, result = val.check_fcff_methods_agree(_context(), TOL)
    assert issues == []
    assert result.status == "pass"


def test_uncomputable_fcff_method_is_not_evaluated_rather_than_passed():
    """Apple reports no interest expense; that must not read as a silent pass."""
    _, result = val.check_fcff_methods_agree(_context(fcff_method_a=None), TOL)
    assert result.status == "not evaluated"


def test_statements_in_another_currency_from_the_price_is_critical():
    """DCF_MODEL_REVIEW.md F9: the old check compared a one-element tuple."""
    issues, _ = val.check_currency_consistent(_context(statement_currency="TWD"), TOL)
    assert len(issues) == 1
    assert issues[0].severity is val.Severity.CRITICAL
    assert issues[0].values == {"Valuation inputs": "TWD", "Price": "USD"}


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


def test_a_forecast_worth_less_than_nothing_in_total_is_flagged():
    """DCF_ROUND3_INSTRUCTIONS.md §2.1: the narrow replacement for the TV-share
    warning — a terminal share above 100%."""
    issues, _ = val.check_explicit_forecast_adds_value(_context(pv_explicit=-5.0), TOL)
    assert [i.key for i in issues] == ["explicit_forecast_negative"]
    assert issues[0].severity is val.Severity.WARNING
    issues, result = val.check_explicit_forecast_adds_value(_context(pv_explicit=120.0), TOL)
    assert issues == [] and result.status == "pass"


def test_the_terminal_value_share_warning_is_gone():
    """Owner's decision: it fired on every company. The share is still shown."""
    names = {check.__name__ for check in val.ALL_CHECKS}
    assert "check_terminal_value_share" not in names
    assert not hasattr(val.Tolerances, "terminal_value_share_ceiling")


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
        pv_explicit=-10.0,
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
    assert "Explicit forecast adds value" in checks
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
        pv_explicit=-10.0,
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


# ---------------------------------------------------------------------------
# Forecast sanity (DCF_MODEL_REVIEW.md §4.4) and derived-figure plausibility
#
# The baseline context leaves every one of these unevaluated or passing, which
# `test_clean_context_produces_no_issues` already pins. Each case below moves
# one input across its line.
# ---------------------------------------------------------------------------


def _keys(issues) -> set[str]:
    return {issue.key for issue in issues}


def test_the_forecast_tax_rate_is_checked_not_the_historical_one():
    """F8: the median ETR passing said nothing about the rate actually projected."""
    context = _context(effective_tax_rate=0.20, forecast_tax_rates=(0.60,) * 5)
    assert val.check_etr_plausible(context, TOL)[0] == []
    issues, _ = val.check_forecast_tax_rate(context, TOL)
    assert _keys(issues) == {"forecast_etr"}
    assert issues[0].severity is val.Severity.WARNING
    assert val.check_forecast_tax_rate(_context(forecast_tax_rates=(0.21,) * 5), TOL)[0] == []


@pytest.mark.parametrize("wacc,growth", [(0.04, 0.0), (0.18, 0.02), (0.07, 0.05)])
def test_an_implausible_wacc_or_a_thin_spread_is_flagged(wacc, growth):
    issues, _ = val.check_wacc_sanity(_context(wacc=wacc, terminal_growth=growth), TOL)
    assert _keys(issues) == {"wacc_sanity"}


def test_a_plausible_wacc_passes():
    assert val.check_wacc_sanity(_context(wacc=0.09, terminal_growth=0.025), TOL)[0] == []


@pytest.mark.parametrize("kd", [0.03, 0.12])
def test_a_cost_of_debt_outside_risk_free_and_cost_of_equity_is_flagged(kd):
    context = _context(cost_of_debt=kd, risk_free_rate=0.04, cost_of_equity=0.10)
    assert _keys(val.check_cost_of_debt_sanity(context, TOL)[0]) == {"kd_sanity"}


def test_cost_of_debt_is_not_judged_without_debt():
    context = _context(cost_of_debt=0.01, cost_of_equity=0.10, debt_weight=0.0)
    issues, result = val.check_cost_of_debt_sanity(context, TOL)
    assert issues == [] and result.status == "not evaluated"


@pytest.mark.parametrize("beta", [-0.2, 0.1, 3.2])
def test_a_beta_outside_the_band_is_flagged(beta):
    assert _keys(val.check_beta_range(_context(beta=beta), TOL)[0]) == {"beta_range"}


def test_forecast_growth_far_above_the_realised_cagr_is_flagged():
    context = _context(forecast_growth=(0.25, 0.2), realised_cagr=0.08)
    assert _keys(val.check_growth_vs_history(context, TOL)[0]) == {"growth_vs_history"}
    close = _context(forecast_growth=(0.10,), realised_cagr=0.08)
    assert val.check_growth_vs_history(close, TOL)[0] == []


def test_a_margin_outside_anything_reported_is_flagged():
    context = _context(
        assumed_margins={"Target margin": 0.45}, historical_margin_range=(0.18, 0.30)
    )
    assert _keys(val.check_margin_vs_history(context, TOL)[0]) == {"margin_vs_history"}


def test_a_forecast_that_jumps_off_from_somewhere_else_is_flagged():
    """F12: the check that protects the valuation rather than the data."""
    context = _context(fcff_method_b=100.0, forecast_growth=(0.10,), year_one_fcff=160.0)
    issues, _ = val.check_jump_off(context, TOL)
    assert _keys(issues) == {"jump_off"}
    follows = _context(fcff_method_b=100.0, forecast_growth=(0.10,), year_one_fcff=112.0)
    assert val.check_jump_off(follows, TOL)[0] == []


def test_ronic_below_wacc_with_growth_is_flagged():
    context = _context(ronic=0.06, wacc=0.09, terminal_growth=0.025)
    assert _keys(val.check_terminal_reinvestment(context, TOL)[0]) == {"terminal_reinvestment"}


def test_growth_on_a_shrinking_asset_base_is_flagged_against_the_companys_own_ratio():
    """§2.6: judged against the company's own capex ÷ D&A, not against 1.0."""
    shrinking = _context(
        ronic=0.09, wacc=0.09, final_capex=40.0, final_dna=60.0, final_growth=0.08,
        historical_capex_to_dna=1.10,
    )
    issues, _ = val.check_terminal_reinvestment(shrinking, TOL)
    assert _keys(issues) == {"terminal_reinvestment"}
    assert "acquired intangibles" in issues[0].message


def test_capex_below_dna_is_normal_when_it_always_has_been():
    """An asset-light or acquisitive company always spends less than D&A."""
    usual = _context(
        ronic=0.09, wacc=0.09, final_capex=40.0, final_dna=60.0, final_growth=0.08,
        historical_capex_to_dna=0.70,
    )
    assert val.check_terminal_reinvestment(usual, TOL)[0] == []


def test_an_exit_multiple_needing_more_growth_than_the_ceiling_is_flagged():
    """F6: 15× implied 4.5% perpetual growth and nothing fired."""
    context = _context(exit_implied_growth=0.045, risk_free_rate=0.043)
    issues, _ = val.check_exit_implied_growth(context, TOL)
    assert _keys(issues) == {"exit_implied_growth"}
    assert "4.00%" in issues[0].tolerance


def test_the_growth_ceiling_is_the_lower_of_gdp_and_the_risk_free_rate():
    """F10: the stricter rule; the old one took the higher of the two."""
    low_rates = _context(terminal_growth=0.025, risk_free_rate=0.015)
    issues, _ = val.check_terminal_growth_reasonable(low_rates, TOL)
    assert _keys(issues) == {"terminal_growth"}
    assert "risk-free" in issues[0].tolerance


def test_diverging_terminal_methods_are_flagged_with_what_the_multiple_assumes():
    """§2.6: 35%, and the message names the value creation the multiple implies."""
    context = _context(
        gordon_value_per_share=100.0, exit_value_per_share=200.0,
        exit_implied_growth=0.055, exit_implied_ronic=0.31, ronic=0.09,
    )
    issues, _ = val.check_method_divergence(context, TOL)
    assert _keys(issues) == {"method_divergence"}
    assert "value creation beyond year N" in issues[0].message
    assert issues[0].values["RONIC the multiple implies"] == "31.0%"
    within = _context(gordon_value_per_share=100.0, exit_value_per_share=150.0)
    assert val.check_method_divergence(within, TOL)[0] == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"gordon_terminal_value": -10.0, "enterprise_values": {"gordon": 5.0}},
        {"enterprise_values": {"gordon": 100.0, "exit_multiple": -5.0}},
        {"enterprise_values": {"gordon": 100.0}, "headline_equity_value": -50.0},
    ],
    ids=["terminal value", "enterprise value", "equity value"],
)
def test_a_negative_output_is_critical(overrides):
    issues, _ = val.check_negative_outcomes(_context(**overrides), TOL)
    assert _keys(issues) == {"negative_outcome"}
    assert issues[0].severity is val.Severity.CRITICAL


def test_a_zero_equity_weight_is_critical():
    """A missing price used to collapse the WACC to after-tax Kd with no flag."""
    issues, _ = val.check_equity_weight(_context(equity_weight=0.0), TOL)
    assert issues[0].severity is val.Severity.CRITICAL


def test_an_extreme_result_asks_for_the_inputs_to_be_checked():
    assert _keys(val.check_extreme_result(_context(upside=0.85), TOL)[0]) == {"extreme_result"}
    assert val.check_extreme_result(_context(upside=-0.30), TOL)[0] == []


def test_a_negative_derived_minority_interest_is_flagged():
    annual = _annual()
    annual["minority_interest"] = -20.0
    issues, _ = val.check_minority_interest_plausible(_context(annual=annual), TOL)
    assert _keys(issues) == {"minority_interest"}
    assert "negative" in issues[0].message


def test_a_minority_interest_with_no_nci_income_is_flagged():
    annual = _annual()
    annual["minority_interest"] = 20.0
    context = _context(annual=annual, net_income_nci=(0.0, 0.0, 0.0))
    issues, _ = val.check_minority_interest_plausible(context, TOL)
    assert "mezzanine" in issues[0].message

    earning = _context(annual=annual, net_income_nci=(2.0, 2.5, 3.0))
    assert val.check_minority_interest_plausible(earning, TOL)[0] == []


def test_material_preferred_dividends_are_flagged_and_immaterial_ones_are_not():
    """§2.8 — GOOGL read $86m of 'preferred dividends' against ~$130bn of net
    income, with no preferred shares outstanding."""
    googl = _context(preferred_dividends=86e6, base_net_income=130e9)
    issues, result = val.check_preferred_equity(googl, TOL)
    assert issues == [] and result.status == "pass"
    material = _context(preferred_dividends=12.0, base_net_income=1000.0)
    issues, _ = val.check_preferred_equity(material, TOL)
    assert [i.key for i in issues] == ["preferred_equity"]
    assert "check the filing for preferred shares outstanding" in issues[0].message
    assert val.check_preferred_equity(_context(preferred_dividends=0.0), TOL)[0] == []


def test_material_acquisitions_left_out_of_reinvestment_are_flagged():
    """F14: acquired growth projected without the acquisition spend."""
    issues, _ = val.check_acquisitions(
        _context(acquisitions_to_capex=1.4, acquisitions_to_revenue=0.05), TOL
    )
    assert _keys(issues) == {"acquisitions"}
    folded = _context(
        acquisitions_to_capex=1.4, acquisitions_to_revenue=0.05, acquisitions_in_capex=True
    )
    assert val.check_acquisitions(folded, TOL)[0] == []


# ---------------------------------------------------------------------------
# Round 2: check tuning (§2.6) and currency conversion (§2.4)
# ---------------------------------------------------------------------------


def test_accelerating_ttm_growth_is_described_as_acceleration():
    context = _context(
        forecast_growth=(0.30,), realised_cagr=0.10, growth_basis="TTM year-over-year"
    )
    issues, _ = val.check_growth_vs_history(context, TOL)
    assert "accelerated" in issues[0].message
    assert issues[0].values["Five-year CAGR"] == "10.00%"


@pytest.mark.parametrize(
    "upside,severity",
    [(0.61, val.Severity.WARNING), (2.0, val.Severity.WARNING),
     (2.01, val.Severity.CRITICAL), (34.9, val.Severity.CRITICAL),
     (-0.9, val.Severity.WARNING)],
)
def test_an_extreme_upside_above_two_hundred_percent_is_critical(upside, severity):
    """DCF_ROUND3_INSTRUCTIONS.md §2.2: NVDA's +3,491% now shows red."""
    issues, _ = val.check_extreme_result(_context(upside=upside, exit_implied_growth=0.11), TOL)
    assert issues[0].severity is severity
    assert issues[0].values["Growth the exit multiple needs with no reinvestment"] == "11.00%"


def test_an_extreme_result_points_to_the_reverse_dcf_under_both_methods():
    context = _context(upside=-0.70, implied_growth={"gordon": 0.41, "exit_multiple": None})
    issues, _ = val.check_extreme_result(context, TOL)
    assert "reverse DCF" in issues[0].message
    assert issues[0].values["Implied year-1 growth (Gordon)"] == "41.00%"
    assert issues[0].values["Implied year-1 growth (exit multiple)"] == "no single rate"


def test_the_exit_growth_message_names_the_usual_fix():
    issues, _ = val.check_exit_implied_growth(_context(exit_implied_growth=0.06), TOL)
    assert "mature company" in issues[0].message


def test_the_sbc_message_says_it_is_already_expensed():
    issues, _ = val.check_sbc_share(_context(sbc=120.0, latest_revenue=1000.0), TOL)
    assert "already expensed" in issues[0].message
    assert "add SBC back" in issues[0].message


def _converted(**overrides) -> val.ValidationContext:
    fields = dict(
        reporting_currency="TWD",
        fx_spot=0.032,
        fx_spot_as_of=date(2026, 9, 25),
        fx_reported=31.27,
        fx_history=((date(2021, 12, 31), 27.7), (date(2025, 12, 31), 31.27)),
    )
    fields.update(overrides)
    return _context(**fields)


def test_a_spot_rate_agreeing_with_sharadars_passes():
    """Sharadar quotes units per dollar: 1 ÷ 31.27 ≈ 0.032 USD per TWD."""
    issues, result = val.check_fx_matches_reported(_converted(), TOL)
    assert issues == [] and result.status == "pass"


def test_an_inverted_spot_quote_is_critical():
    issues, _ = val.check_fx_matches_reported(_converted(fx_spot=31.27), TOL)
    assert _keys(issues) == {"fx_vs_reported"}
    assert issues[0].severity is val.Severity.CRITICAL


def test_a_converted_valuation_always_states_its_spot_assumption():
    issues, _ = val.check_fx_spot_assumption(_converted(), TOL)
    assert _keys(issues) == {"fx_spot_assumption"}
    assert val.check_fx_spot_assumption(_context(), TOL)[1].status == "not evaluated"


@pytest.mark.parametrize(
    "history,severity",
    [
        (((date(2021, 12, 31), 10.0), (date(2025, 12, 31), 20.0)), val.Severity.BLOCKING),
        (((date(2021, 12, 31), 10.0), (date(2025, 12, 31), 11.8)), val.Severity.WARNING),
    ],
    ids=["halved: blocking", "gentle slide: warning"],
)
def test_a_depreciating_currency_is_flagged_by_its_own_history(history, severity):
    issues, _ = val.check_currency_depreciation(_converted(fx_history=history), TOL)
    assert _keys(issues) == {"fx_depreciation"}
    assert issues[0].severity is severity


def test_a_stable_currency_passes():
    _, result = val.check_currency_depreciation(_converted(), TOL)
    assert result.status == "pass"


def test_a_foreign_filer_is_warned_about_ifrs():
    issues, _ = val.check_foreign_accounting(_context(foreign_filer=True), TOL)
    assert "IFRS" in issues[0].message
    assert "basis of preparation" in issues[0].message
    assert val.check_foreign_accounting(_context(), TOL)[0] == []


# ---------------------------------------------------------------------------
# Round 3
# ---------------------------------------------------------------------------


def _one_year_cash_flow(cfo: float, gap: float, net: float = 10e6) -> pd.DataFrame:
    """One period whose flows miss the reported net change in cash by ``gap``."""
    cff = -1e6
    cfi = net + gap - cfo - cff
    return pd.DataFrame(
        {"cfo": [cfo], "cfi": [cfi], "cff": [cff], "net_change_in_cash": [net],
         "fx_effect": [float("nan")], "cash": [100e6]},
        index=[2025],
    )


def test_a_rounding_gap_beside_billions_of_operating_cash_flow_ties():
    """§2.3: €2m against a €10m change in cash failed 0.1% of the change,
    though the company generates €6bn of operating cash flow."""
    issues, result = val.check_cash_flow_ties(
        _context(annual=_one_year_cash_flow(cfo=6e9, gap=2e6)), TOL
    )
    assert issues == [] and result.status == "pass"


def test_the_same_gap_beside_a_small_operating_cash_flow_still_fails():
    issues, _ = val.check_cash_flow_ties(
        _context(annual=_one_year_cash_flow(cfo=3e6, gap=2e6)), TOL
    )
    assert len(issues) == 1
    assert issues[0].values["Difference ÷ |CFO|"] == "66.67%"


def _two_years(
    *, cfo: float, flows: float, net: float, fx: float | None, cash: tuple[float, float]
) -> pd.DataFrame:
    """FY2025 against a FY2024 balance sheet; CFO + CFI + CFF = ``flows``."""
    nan = float("nan")
    return pd.DataFrame(
        {
            "cfo": [nan, cfo], "cfi": [nan, flows - cfo + 100.0], "cff": [nan, -100.0],
            "net_change_in_cash": [nan, net], "fx_effect": [nan, nan if fx is None else fx],
            "cash": list(cash),
        },
        index=[2024, 2025],
    )


def _tie(frame: pd.DataFrame, **overrides):
    return val.check_cash_flow_ties(_context(annual=frame, **overrides), TOL)


def test_a_missing_net_change_that_ties_to_the_balance_sheet_passes():
    """§2.4(a) — TSM FY2018: net change 0, balance sheet moved by the flows."""
    frame = _two_years(cfo=573954.0, flows=24423.0, net=0.0, fx=0.0, cash=(100000.0, 124423.0))
    issues, result = _tie(frame)
    assert issues == [] and result.status == "pass"
    assert "tie to the balance-sheet cash change instead" in result.detail


def test_a_missing_net_change_with_a_small_residual_is_a_warning():
    """TSM FY2023: 6,294 against 1.24tn of CFO — 0.5%."""
    frame = _two_years(
        cfo=1241967.0, flows=128907.0 + 8339.0, net=0.0, fx=-8339.0, cash=(0.0, 122614.0)
    )
    issues, _ = _tie(frame)
    assert issues[0].severity is val.Severity.WARNING
    assert issues[0].values["Residual ÷ |CFO|"] == "0.51%"


def test_a_missing_net_change_with_a_large_residual_stays_critical():
    frame = _two_years(cfo=1000.0, flows=500.0, net=0.0, fx=None, cash=(0.0, 400.0))
    issues, _ = _tie(frame)
    assert issues[0].severity is val.Severity.CRITICAL
    assert "possibly capex" in issues[0].message


def test_a_missing_balance_sheet_leaves_the_period_unevaluated():
    frame = _two_years(cfo=1000.0, flows=500.0, net=0.0, fx=None, cash=(float("nan"), 400.0))
    issues, result = _tie(frame)
    assert issues == [] and result.status == "not evaluated"


def test_a_zero_filled_fx_line_at_a_foreign_filer_is_the_undisclosed_fx_effect():
    """§2.4(b) — SAP FY2025: gap 836 of 9,156 CFO, net change = balance sheet."""
    frame = _two_years(cfo=9156.0, flows=-554.0, net=-1390.0, fx=0.0, cash=(10000.0, 8610.0))
    issues, _ = _tie(frame, foreign_filer=True)
    assert issues[0].severity is val.Severity.WARNING
    assert issues[0].values["Implied FX effect"] == "-836"


def test_the_same_gap_at_a_domestic_filer_stays_critical():
    frame = _two_years(cfo=9156.0, flows=-554.0, net=-1390.0, fx=0.0, cash=(10000.0, 8610.0))
    issues, _ = _tie(frame)
    assert issues[0].severity is val.Severity.CRITICAL


def test_an_implied_fx_effect_beyond_ten_percent_of_cfo_stays_critical():
    frame = _two_years(cfo=1000.0, flows=0.0, net=-200.0, fx=0.0, cash=(1000.0, 800.0))
    issues, _ = _tie(frame, foreign_filer=True)
    assert issues[0].severity is val.Severity.CRITICAL


def test_a_missing_net_change_and_a_missing_fx_line_together():
    """TSM FY2025 needs both: net change 0, FX line not reported."""
    frame = _two_years(
        cfo=2274976.0, flows=701044.0, net=0.0, fx=None, cash=(0.0, 640229.0)
    )
    issues, _ = _tie(frame, foreign_filer=True)
    assert issues[0].severity is val.Severity.WARNING
    assert issues[0].values["Implied FX effect"] == "-60,815"


def test_a_record_ttm_margin_with_no_target_does_not_fire():
    """§2.4: the TTM margin is reported data, and nothing was assumed."""
    context = _context(historical_margin_range=(0.20, 0.65), assumed_margins={})
    issues, result = val.check_margin_vs_history(context, TOL)
    assert issues == [] and result.status == "not evaluated"


def test_a_target_well_above_the_reported_range_fires():
    context = _context(
        historical_margin_range=(0.20, 0.30), assumed_margins={"Target margin": 0.38}
    )
    issues, _ = val.check_margin_vs_history(context, TOL)
    assert [i.key for i in issues] == ["margin_vs_history"]
    assert issues[0].values["Target margin"] == "38.00%"


def test_an_assumed_margin_within_the_slack_passes():
    context = _context(
        historical_margin_range=(0.20, 0.30), assumed_margins={"Starting margin": 0.34}
    )
    assert val.check_margin_vs_history(context, TOL)[0] == []


@pytest.mark.parametrize(
    "of_capex,of_revenue,fires",
    [(10.45, 0.03, True), (10.45, 0.01, False), (0.40, 0.05, False)],
    ids=["both material", "capex only (tiny capex)", "revenue only"],
)
def test_acquisitions_must_be_material_against_capex_and_revenue(of_capex, of_revenue, fires):
    """§2.5: SHOP's capex is so small that any acquisition dwarfs it."""
    context = _context(acquisitions_to_capex=of_capex, acquisitions_to_revenue=of_revenue)
    issues, _ = val.check_acquisitions(context, TOL)
    assert bool(issues) is fires
    if fires:
        assert set(issues[0].values) == {
            "Acquisitions ÷ capex (5 years)", "Acquisitions ÷ revenue (5 years)"
        }


# ---------------------------------------------------------------------------
# Round 4: acquisitions concentration, TTM window, spread table
# ---------------------------------------------------------------------------


def test_one_dominant_acquisition_year_is_named_as_a_single_deal():
    """§2.7 — SHOP: 2022 is 91% of five years' acquisitions."""
    shop = _context(
        acquisitions_to_capex=10.45, acquisitions_to_revenue=0.051,
        acquisitions_by_year=(("FY2021", 60.0), ("FY2022", 1754.0), ("FY2023", 31.0),
                              ("FY2024", 30.0), ("FY2025", 56.0)),
    )
    issues, _ = val.check_acquisitions(shop, TOL)
    assert issues[0].values["By year"] == "One year (FY2022) accounts for 91% of the 5-year total"


def test_acquisitions_spread_over_several_years_read_as_a_pattern():
    pattern = _context(
        acquisitions_to_capex=2.0, acquisitions_to_revenue=0.10,
        acquisitions_by_year=(("FY2021", 500.0), ("FY2022", 400.0), ("FY2023", 300.0),
                              ("FY2024", 0.0), ("FY2025", 200.0)),
    )
    issues, _ = val.check_acquisitions(pattern, TOL)
    assert issues[0].values["By year"].startswith("Spread across 4 years")


def test_acquisitions_inside_the_ttm_window_are_flagged_only_under_the_ttm_basis():
    fires = _context(
        growth_basis="TTM year-over-year", ttm_window_acquisitions=0.08,
        ttm_window_periods=("FY2025 Q2", "FY2026 Q2"),
    )
    issues, _ = val.check_ttm_acquired_growth(fires, TOL)
    assert [i.key for i in issues] == ["ttm_acquired_growth"]
    quiet = _context(growth_basis="TTM year-over-year", ttm_window_acquisitions=0.01)
    assert val.check_ttm_acquired_growth(quiet, TOL)[0] == []
    history = _context(growth_basis="5-year CAGR", ttm_window_acquisitions=0.08)
    assert val.check_ttm_acquired_growth(history, TOL)[1].status == "not evaluated"


@pytest.mark.parametrize(
    "valuation_day,table_date,status",
    [
        (date(2027, 3, 31), date(2026, 1, 1), "pass"),
        (date(2027, 4, 1), date(2026, 1, 1), "pass"),
        (date(2027, 4, 2), date(2026, 1, 1), "fail"),
        (date(2027, 4, 2), None, "not evaluated"),
    ],
    ids=["under fifteen months", "exactly fifteen months", "a day over: stale",
         "yield typed: not evaluated"],
)
def test_the_spread_table_is_flagged_when_stale(valuation_day, table_date, status):
    """§2.6: only when the synthetic rating actually set the cost of debt."""
    context = _context(valuation_date=valuation_day, spread_table_as_of=table_date)
    issues, result = val.check_spread_table_current(context, TOL)
    assert result.status == status
    if status == "fail":
        assert issues[0].severity is val.Severity.WARNING
        assert "SYNTHETIC_RATINGS" in issues[0].message
