"""Statement normalisation, sign conventions, and ratio derivation."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.data.sharadar import statements as sh_statements
from app.data.sharadar import transport as sh_transport
from app.logic.dcf import financials as fin
from app.logic.dcf.schema import resolve_schema


@pytest.fixture
def concept_table(statements_payload, tickers_response, monkeypatch):
    def _get(table, params, ticker=None):
        return tickers_response if table == "TICKERS" else statements_payload

    monkeypatch.setattr(sh_transport, "_get", _get)
    statements = sh_statements._fetch_statements("TEST", annual=True)
    schema = resolve_schema(statements.frames, "TEST", statements.template)
    values, meta = fin.build_concept_table(statements, schema)
    return values, meta, schema


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def test_all_four_statements_merge_into_one_row_per_year(concept_table):
    """One row per fiscal year, carrying concepts from all four frames."""
    values, _, _ = concept_table
    assert list(values.index) == [2023, 2024, 2025]
    assert values.loc[2025, "revenue"] == 1200            # PL
    assert values.loc[2025, "total_assets"] == 1060       # BS
    assert values.loc[2025, "cfo"] == 340                 # CF
    assert values.loc[2025, "derived_fcf"] == 240         # DERIVED


def test_metadata_carries_the_report_date_and_filing_link(concept_table):
    _, meta, _ = concept_table
    assert meta.loc[2025, "report_date"] == pd.Timestamp("2025-09-30")
    assert "sec.gov" in meta.loc[2025, "source_link"]


def test_a_column_absent_from_the_response_stays_nan(concept_table, statements_payload,
                                                     tickers_response, monkeypatch):
    """A dropped indicator must read as absent, never as a zero balance."""
    from tests.conftest import SF1_COLUMNS

    stripped = dict(statements_payload)
    keep = [i for i, name in enumerate(SF1_COLUMNS) if name != "sbcomp"]
    stripped["datatable"] = {
        "columns": [{"name": SF1_COLUMNS[i]} for i in keep],
        "data": [[row[i] for i in keep] for row in statements_payload["datatable"]["data"]],
    }

    def _get(table, params, ticker=None):
        return tickers_response if table == "TICKERS" else stripped

    monkeypatch.setattr(sh_transport, "_get", _get)
    statements = sh_statements._fetch_statements("TEST", annual=True)
    schema = resolve_schema(statements.frames, "TEST", statements.template)
    values, _ = fin.build_concept_table(statements, schema)
    assert "sbc" not in values.columns


def test_a_derived_minority_interest_of_zero_is_a_real_zero(concept_table):
    """The fixture company has no NCI, and the residual says so exactly."""
    values, _, _ = concept_table
    assert values.loc[2025, "minority_interest"] == 0.0


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


def test_capex_negates_the_reported_outflow():
    """Sharadar signs capex negative for cash spent; the model wants it positive."""
    values = pd.DataFrame({"capex_acquisition": [-12715.0]}, index=[2025])
    result = fin.capex(values)
    assert result.origin == "reported"
    assert result.values.loc[2025] == 12715.0


def test_capex_from_the_fixture_is_reported_not_estimated(concept_table):
    values, _, _ = concept_table
    result = fin.capex(values)
    assert result.origin == "reported"
    assert result.values.loc[2025] == 100.0


def test_capex_is_nan_when_nothing_is_reported():
    values = pd.DataFrame({"capex_acquisition": [np.nan]}, index=[2025])
    result = fin.capex(values)
    assert result.values.isna().all()
    assert result.origin == "estimate"
    assert "no capital expenditure line" in result.detail


def test_capex_is_nan_when_the_column_is_absent_entirely():
    result = fin.capex(pd.DataFrame({"revenue": [1200.0]}, index=[2025]))
    assert result.values.isna().all()


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
    # Pretax is 1200 − 700 − 190 = 310; tax expense is 60.
    assert fin.effective_tax_rate(values).loc[2025] == pytest.approx(60 / 310)


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
# Realised growth
# ---------------------------------------------------------------------------


def _revenue(values, index=None) -> pd.DataFrame:
    return pd.DataFrame(
        {"revenue": values}, index=index if index is not None else range(2026 - len(values), 2026)
    )


def test_the_compound_and_arithmetic_rates_are_not_the_same_number(concept_table):
    values, _, _ = concept_table
    history = fin.revenue_growth_history(values)
    assert history.ok
    assert history.cagr == pytest.approx(0.09544511501033215)
    assert history.arithmetic_mean == pytest.approx(0.09545454545454546)
    assert history.cagr < history.arithmetic_mean
    assert (history.first_period, history.last_period) == ("2023", "2025")
    assert (history.intervals, history.observations) == (2, 2)


def test_a_round_trip_series_compounds_at_zero_while_averaging_a_quarter():
    """Why the benchmark is the CAGR: the average overstates by 25 points here.

    Revenue ends exactly where it started, so the realised compound rate is zero.
    Averaging +100% and -50% gives +25% — a number no reading of this history
    supports, and the one a reverse DCF would be benchmarked against if the
    driver panel's arithmetic basis were reused here.
    """
    history = fin.revenue_growth_history(_revenue([100.0, 200.0, 100.0]))
    assert history.cagr == pytest.approx(0.0)
    assert history.arithmetic_mean == pytest.approx(0.25)


def test_a_gap_shortens_the_span_rather_than_bridging_it():
    history = fin.revenue_growth_history(
        _revenue([500.0, np.nan, 1000.0, 1100.0, 1200.0], index=[2021, 2022, 2023, 2024, 2025])
    )
    assert history.first_period == "2023"
    assert history.intervals == 2
    assert history.cagr == pytest.approx(0.09544511501033215)
    assert "rather than bridged" in history.detail


def test_a_non_positive_period_ends_the_span():
    """A CAGR needs positive endpoints; a loss-making year cannot anchor one."""
    history = fin.revenue_growth_history(
        _revenue([-50.0, 0.0, 1000.0, 1100.0, 1200.0], index=[2021, 2022, 2023, 2024, 2025])
    )
    assert history.first_period == "2023"
    assert history.first_value == 1000.0


def test_a_single_period_yields_no_cagr_and_says_why():
    history = fin.revenue_growth_history(_revenue([1200.0], index=[2025]))
    assert not history.ok
    assert history.cagr is None
    assert history.intervals == 0
    assert "no interval to compound across" in history.detail


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(),
        pd.DataFrame({"ebit": [1.0]}, index=[2025]),
        pd.DataFrame({"revenue": [np.nan, np.nan]}, index=[2024, 2025]),
    ],
    ids=["empty", "no revenue column", "all missing"],
)
def test_nothing_is_substituted_when_no_growth_can_be_measured(frame):
    history = fin.revenue_growth_history(frame)
    assert not history.ok
    assert history.cagr is None
    assert history.arithmetic_mean is None
    assert history.detail


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
