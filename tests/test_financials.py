"""Statement normalisation, sign conventions, and ratio derivation."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.data import simfin_client as sc
from app.logic.dcf import financials as fin
from app.logic.dcf.schema import resolve_schema


@pytest.fixture
def concept_table(statements_payload, monkeypatch):
    monkeypatch.setattr(sc, "_request", lambda *a, **k: statements_payload)
    statements = sc._fetch_statements("TEST", period="FY")
    schema = resolve_schema(statements.frames, "TEST", statements.template)
    values, meta = fin.build_concept_table(statements, schema)
    return values, meta, schema


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def test_annual_merge_keys_on_fiscal_year_despite_the_bs_q4_label(concept_table):
    """The balance sheet labels annual periods Q4; merging must not split years."""
    values, _, _ = concept_table
    assert list(values.index) == [2023, 2024, 2025]
    assert values.loc[2025, "revenue"] == 1200
    assert values.loc[2025, "total_assets"] == 1060  # from the BS, same row


def test_metadata_carries_the_report_date_and_filing_link(concept_table):
    _, meta, _ = concept_table
    assert meta.loc[2025, "report_date"] == pd.Timestamp("2025-09-30")
    assert "sec.gov" in meta.loc[2025, "source_link"]


def test_unreported_line_items_stay_nan(concept_table):
    values, _, _ = concept_table
    assert values["sbc"].isna().all()
    assert values["minority_interest"].isna().all()


# ---------------------------------------------------------------------------
# Sign conventions
# ---------------------------------------------------------------------------


def test_tax_expense_is_positive_for_an_expense(concept_table):
    values, _, _ = concept_table
    assert fin.tax_expense(values).loc[2025] == 60


def test_a_genuine_tax_benefit_stays_negative():
    """Ford FY2025 reports a tax benefit in a loss year; abs() would hide it."""
    values = pd.DataFrame({"tax_expense": [3668.0]}, index=[2025])
    assert fin.tax_expense(values).loc[2025] == -3668.0


def test_interest_expense_is_positive_for_an_expense():
    values = pd.DataFrame(
        {"interest_expense_net": [-868.0], "interest_expense_gross": [-900.0]}, index=[2025]
    )
    assert fin.interest_expense(values, prefer="net").loc[2025] == 868.0
    assert fin.interest_expense(values, prefer="gross").loc[2025] == 900.0


def test_net_interest_income_stays_negative_rather_than_flipped():
    values = pd.DataFrame({"interest_expense_net": [400.0]}, index=[2025])
    assert fin.interest_expense(values).loc[2025] == -400.0


def test_interest_falls_back_to_the_other_line_when_one_is_missing():
    values = pd.DataFrame(
        {"interest_expense_net": [np.nan], "interest_expense_gross": [-500.0]}, index=[2025]
    )
    assert fin.interest_expense(values, prefer="net").loc[2025] == 500.0


# ---------------------------------------------------------------------------
# CapEx cascade
# ---------------------------------------------------------------------------


def test_capex_uses_the_gross_line_when_reported():
    values = pd.DataFrame(
        {
            "capex_acquisition": [-2112.0],
            "capex_net_change": [-2099.0],
            "capex_disposals": [13.0],
        },
        index=[2025],
    )
    result = fin.capex(values)
    assert result.origin == "reported"
    assert result.values.loc[2025] == 2112.0


def test_capex_reconstructs_the_gross_figure_from_disposals():
    """KO FY2025: net -2,099 less disposals +13 gives the gross -2,112."""
    values = pd.DataFrame(
        {
            "capex_acquisition": [np.nan],
            "capex_net_change": [-2099.0],
            "capex_disposals": [13.0],
        },
        index=[2025],
    )
    result = fin.capex(values)
    assert result.origin == "derived"
    assert result.values.loc[2025] == pytest.approx(2112.0)


def test_capex_falls_back_to_the_net_line_as_a_labelled_estimate():
    values = pd.DataFrame(
        {
            "capex_acquisition": [np.nan],
            "capex_net_change": [-12715.0],
            "capex_disposals": [np.nan],
        },
        index=[2025],
    )
    result = fin.capex(values)
    assert result.origin == "estimate"
    assert result.values.loc[2025] == 12715.0
    assert "net of asset disposals" in result.detail


def test_capex_is_nan_when_nothing_is_reported():
    values = pd.DataFrame(
        {"capex_acquisition": [np.nan], "capex_net_change": [np.nan]}, index=[2025]
    )
    result = fin.capex(values)
    assert result.values.isna().all()
    assert "no capital expenditure line" in result.detail


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------


def test_total_debt_sums_the_components(concept_table):
    values, _, _ = concept_table
    assert fin.total_debt(values).loc[2025] == 270  # 50 short + 220 long


def test_total_debt_is_nan_when_no_component_is_reported():
    values = pd.DataFrame(
        {"short_term_debt": [np.nan], "long_term_debt": [np.nan]}, index=[2025]
    )
    assert fin.total_debt(values).isna().all()


def test_total_debt_tolerates_one_missing_maturity():
    values = pd.DataFrame(
        {"short_term_debt": [np.nan], "long_term_debt": [200.0]}, index=[2025]
    )
    assert fin.total_debt(values).loc[2025] == 200.0


def test_effective_tax_rate(concept_table):
    values, _, _ = concept_table
    assert fin.effective_tax_rate(values).loc[2025] == pytest.approx(60 / 300)


def test_effective_tax_rate_is_undefined_on_a_pretax_loss():
    """A loss year makes the ratio meaningless; it must be NaN, not zero."""
    values = pd.DataFrame(
        {"pretax_income": [-11830.0], "tax_expense": [3668.0]}, index=[2025]
    )
    assert fin.effective_tax_rate(values).isna().all()


def test_net_working_capital_strips_cash_and_debt(concept_table):
    values, _, _ = concept_table
    # (480 − 140 − 70) − (360 − 50) = 270 − 310 = −40
    assert fin.net_working_capital(values).loc[2025] == pytest.approx(-40.0)


def test_delta_nwc_first_period_is_nan_not_zero(concept_table):
    """The earliest year has no prior balance sheet to difference against."""
    values, _, _ = concept_table
    delta = fin.delta_nwc(values)
    assert pd.isna(delta.iloc[0])
    assert not pd.isna(delta.iloc[1])


def test_cash_flow_delta_nwc_is_sign_flipped_to_match(concept_table):
    values, _, _ = concept_table
    assert fin.cash_flow_delta_nwc(values).loc[2025] == 1000.0  # reported as −1,000


# ---------------------------------------------------------------------------
# Ratios and defaults
# ---------------------------------------------------------------------------


def test_driver_ratios_are_computed_for_every_year(concept_table):
    values, _, _ = concept_table
    ratios = fin.driver_ratios(values)
    assert set(fin.DRIVER_KEYS) <= set(ratios.columns)
    assert ratios.loc[2025, "operating_margin"] == pytest.approx(310 / 1200)
    assert ratios.loc[2025, "revenue_growth"] == pytest.approx(1200 / 1100 - 1)
    assert pd.isna(ratios.loc[2023, "revenue_growth"])


def test_driver_default_reports_the_years_it_used(concept_table):
    values, _, _ = concept_table
    ratios = fin.driver_ratios(values)
    default = fin.driver_default(ratios, "operating_margin", "5-year average")
    assert default.ok
    assert default.years_used == 3
    assert "Only 3 of the requested 5 years" in default.detail


def test_driver_default_most_recent_year(concept_table):
    values, _, _ = concept_table
    ratios = fin.driver_ratios(values)
    default = fin.driver_default(ratios, "operating_margin", "Most recent year")
    assert default.value == pytest.approx(310 / 1200)
    assert default.years_used == 1


def test_driver_default_excludes_missing_years_rather_than_zero_filling():
    ratios = pd.DataFrame({"operating_margin": [0.2, np.nan, 0.3]}, index=[2023, 2024, 2025])
    default = fin.driver_default(ratios, "operating_margin", "3-year average")
    assert default.years_used == 2
    assert default.value == pytest.approx(0.25)


def test_driver_default_reports_no_value_when_nothing_is_usable():
    ratios = pd.DataFrame({"operating_margin": [np.nan, np.nan]}, index=[2024, 2025])
    default = fin.driver_default(ratios, "operating_margin", "5-year average")
    assert not default.ok
    assert default.years_used == 0
    assert "No period had every component" in default.detail


def test_nwc_ratio_skips_years_where_revenue_did_not_move():
    values = pd.DataFrame(
        {
            "revenue": [1000.0, 1000.0],
            "ebit": [200.0, 200.0],
            "dna": [40.0, 40.0],
            "total_current_assets": [400.0, 450.0],
            "total_current_liabilities": [300.0, 300.0],
            "cash": [0.0, 0.0],
        },
        index=[2024, 2025],
    )
    ratios = fin.driver_ratios(values)
    assert pd.isna(ratios.loc[2025, "dnwc_pct_drevenue"])


# ---------------------------------------------------------------------------
# TTM
# ---------------------------------------------------------------------------


def test_manual_ttm_sums_the_last_four_quarters():
    quarterly = pd.DataFrame(
        {"revenue": [10.0, 20.0, 30.0, 40.0, 50.0]},
        index=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    assert fin.manual_ttm(quarterly, ("revenue",))["revenue"] == 140.0


def test_manual_ttm_returns_none_with_fewer_than_four_quarters():
    """An incomplete sum would manufacture a divergence that isn't there."""
    quarterly = pd.DataFrame({"revenue": [10.0, 20.0, 30.0]}, index=["Q1", "Q2", "Q3"])
    assert fin.manual_ttm(quarterly, ("revenue",)) is None


def test_manual_ttm_propagates_a_missing_quarter_as_nan():
    quarterly = pd.DataFrame(
        {"revenue": [10.0, np.nan, 30.0, 40.0]}, index=["Q1", "Q2", "Q3", "Q4"]
    )
    assert pd.isna(fin.manual_ttm(quarterly, ("revenue",))["revenue"])


def test_stock_and_flow_keys_do_not_overlap():
    """Balance-sheet items must never be summed across quarters."""
    assert not set(fin.FLOW_KEYS) & set(fin.STOCK_KEYS)


# ---------------------------------------------------------------------------
# Fiscal calendar
# ---------------------------------------------------------------------------


def test_fiscal_year_end_uses_the_reported_day():
    """Apple's fiscal year ends on the Saturday nearest 30 September."""
    result = fin.fiscal_year_end(9, date(2026, 7, 31), date(2025, 9, 27))
    assert result == date(2026, 9, 27)


def test_fiscal_year_end_rolls_into_next_year_when_already_past():
    assert fin.fiscal_year_end(6, date(2026, 7, 31), date(2026, 6, 30)) == date(2027, 6, 30)


def test_fiscal_year_end_is_none_without_a_month():
    assert fin.fiscal_year_end(None, date(2026, 7, 31), None) is None


def test_calendar_and_fiscal_year_ends_are_both_handled():
    assert fin.fiscal_year_end(12, date(2026, 7, 31), date(2025, 12, 31)) == date(2026, 12, 31)
    assert fin.fiscal_year_end(1, date(2026, 7, 31), date(2026, 1, 31)) == date(2027, 1, 31)


# ---------------------------------------------------------------------------
# Base-period working capital span
# ---------------------------------------------------------------------------


def test_ttm_base_period_uses_a_trailing_twelve_month_working_capital_change():
    """A TTM cash flow must be paired with a TTM working-capital movement.

    Differencing the last two annual balance sheets against a TTM cash flow
    measures two different spans, which inflates the gap between the two FCFF
    builds into something that reads as a data defect.
    """
    from app.logic.dcf import model as dcf_model

    quarterly = pd.DataFrame(
        {
            "total_current_assets": [400.0, 410.0, 420.0, 430.0, 500.0],
            "total_current_liabilities": [300.0] * 5,
            "cash": [0.0] * 5,
            "short_term_investments": [0.0] * 5,
            "short_term_debt": [0.0] * 5,
        },
        index=["FY2025 Q1", "FY2025 Q2", "FY2025 Q3", "FY2025 Q4", "FY2026 Q1"],
    )
    annual = pd.DataFrame(
        {
            "total_current_assets": [200.0, 430.0],
            "total_current_liabilities": [300.0, 300.0],
            "cash": [0.0, 0.0],
            "short_term_investments": [0.0, 0.0],
            "short_term_debt": [0.0, 0.0],
        },
        index=[2024, 2025],
    )
    dataset = type("D", (), {"quarterly": quarterly, "annual": annual})()
    base = type("B", (), {"is_ttm": True})()

    change, span = dcf_model.base_period_delta_nwc(dataset, base)
    assert change == pytest.approx(100.0)  # 500 − 400, four quarters apart
    assert "Trailing twelve months" in span
    # The annual basis would have said 230, over a different span entirely.
    assert change != pytest.approx(230.0)


def test_annual_base_period_uses_the_annual_working_capital_change():
    from app.logic.dcf import model as dcf_model

    annual = pd.DataFrame(
        {
            "total_current_assets": [200.0, 430.0],
            "total_current_liabilities": [300.0, 300.0],
            "cash": [0.0, 0.0],
            "short_term_investments": [0.0, 0.0],
            "short_term_debt": [0.0, 0.0],
        },
        index=[2024, 2025],
    )
    dataset = type("D", (), {"quarterly": pd.DataFrame(), "annual": annual})()
    base = type("B", (), {"is_ttm": False})()

    change, span = dcf_model.base_period_delta_nwc(dataset, base)
    assert change == pytest.approx(230.0)
    assert "Fiscal year 2025" in span


def test_ttm_base_falls_back_to_annual_and_says_so_with_thin_quarters():
    from app.logic.dcf import model as dcf_model

    annual = pd.DataFrame(
        {
            "total_current_assets": [200.0, 430.0],
            "total_current_liabilities": [300.0, 300.0],
            "cash": [0.0, 0.0],
            "short_term_investments": [0.0, 0.0],
            "short_term_debt": [0.0, 0.0],
        },
        index=[2024, 2025],
    )
    dataset = type("D", (), {"quarterly": pd.DataFrame(), "annual": annual})()
    base = type("B", (), {"is_ttm": True})()

    change, span = dcf_model.base_period_delta_nwc(dataset, base)
    assert change == pytest.approx(230.0)
    assert "fewer than five quarters" in span


def test_no_consecutive_balance_sheets_yields_no_change():
    from app.logic.dcf import model as dcf_model

    annual = pd.DataFrame(
        {"total_current_assets": [200.0], "total_current_liabilities": [100.0],
         "cash": [0.0], "short_term_investments": [0.0], "short_term_debt": [0.0]},
        index=[2025],
    )
    dataset = type("D", (), {"quarterly": pd.DataFrame(), "annual": annual})()
    base = type("B", (), {"is_ttm": False})()
    change, span = dcf_model.base_period_delta_nwc(dataset, base)
    assert change is None
    assert "No two consecutive balance sheets" in span
