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


def _nwc_history(revenue: list[float], nwc: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "revenue": revenue,
            "total_current_assets": [n + 300.0 for n in nwc],
            "total_current_liabilities": [300.0] * len(nwc),
            "cash": [0.0] * len(nwc),
        },
        index=[2021 + i for i in range(len(nwc))],
    )


def test_the_nwc_driver_is_a_level_ratio():
    ratios = fin.driver_ratios(_nwc_history([1000.0, 1100.0], [50.0, 66.0]))
    assert ratios["nwc_pct_revenue"].tolist() == pytest.approx([0.05, 0.06])


def test_a_near_flat_revenue_year_no_longer_dominates_the_default():
    """DCF_ROUND4_INSTRUCTIONS.md §2.1 — DHR's shape: revenue moves $15m and
    working capital $368m. ΔNWC ÷ Δrevenue read 2,453% that year and dragged
    the five-year average to 518%; the level ratio barely notices."""
    history = _nwc_history(
        [24802.0, 26643.0, 23890.0, 23875.0, 24568.0], [930.0, 2090.0, 1494.0, 1126.0, 1336.0]
    )
    default = fin.driver_default(fin.driver_ratios(history), "nwc_pct_revenue", "5-year average")
    assert 0.04 < default.value < 0.06
    old_style = (history["total_current_assets"].diff() / history["revenue"].diff()).iloc[-2]
    assert old_style > 20  # what the replaced driver read for that year


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
# Compound growth default
# ---------------------------------------------------------------------------


def _growth_ratios(rates: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"revenue_growth": rates}, index=[2020 + i for i in range(len(rates))]
    )


def test_growth_defaults_to_the_compound_rate_not_the_average():
    """DCF_MODEL_REVIEW.md F7: 100 → 200 → 100 averages +25% and compounds at 0%."""
    ratios = _growth_ratios([1.0, -0.5])
    cagr = fin.driver_default(ratios, "revenue_growth", "3-year CAGR")
    average = fin.driver_default(ratios, "revenue_growth", "3-year average")
    assert cagr.value == pytest.approx(0.0)
    assert average.value == pytest.approx(0.25)
    assert fin.GROWTH_BASES[0] == "5-year CAGR"


def test_cagr_compounds_the_reported_years():
    ratios = _growth_ratios([0.10, 0.20, 0.05])
    result = fin.driver_default(ratios, "revenue_growth", "3-year CAGR")
    assert result.value == pytest.approx((1.10 * 1.20 * 1.05) ** (1 / 3) - 1)
    assert result.years_used == 3


def test_cagr_is_refused_for_a_driver_that_does_not_compound():
    ratios = pd.DataFrame({"operating_margin": [0.2, 0.3]}, index=[2024, 2025])
    result = fin.driver_default(ratios, "operating_margin", "5-year CAGR")
    assert not result.ok


def test_cagr_across_a_total_loss_of_revenue_is_not_invented():
    result = fin.driver_default(_growth_ratios([0.1, -1.0, 0.2]), "revenue_growth", "3-year CAGR")
    assert not result.ok
    assert "no compound rate" in result.detail


# ---------------------------------------------------------------------------
# Acquisitions
# ---------------------------------------------------------------------------


def test_acquisition_spend_is_positive_for_cash_out_and_missing_stays_missing():
    values = pd.DataFrame({"acquisitions": [-50.0, np.nan, 10.0]})
    spend = fin.acquisition_spend(values)
    assert spend.iloc[0] == 50.0
    assert pd.isna(spend.iloc[1])
    assert spend.iloc[2] == -10.0  # a net disposal
    assert fin.acquisition_spend(pd.DataFrame({"revenue": [1.0]})).isna().all()


def test_acquisitions_are_a_ratio_but_not_a_forecast_driver(concept_table):
    values, _, _ = concept_table
    assert values["acquisitions"].notna().all()
    ratios = fin.driver_ratios(values)
    assert fin.ACQUISITIONS_KEY in ratios.columns
    assert fin.ACQUISITIONS_KEY not in fin.DRIVER_KEYS


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
    base = type("B", (), {"is_ttm": True, "fx_rate": 1.0})()

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
    base = type("B", (), {"is_ttm": False, "fx_rate": 1.0})()

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
    base = type("B", (), {"is_ttm": True, "fx_rate": 1.0})()

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
    base = type("B", (), {"is_ttm": False, "fx_rate": 1.0})()
    change, span = dcf_model.base_period_delta_nwc(dataset, base)
    assert change is None
    assert "No two consecutive balance sheets" in span


def test_the_base_period_working_capital_change_is_converted_at_spot():
    """ΔNWC feeds method B, which meets dollar figures, so it is converted too."""
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
    local = type("B", (), {"is_ttm": False, "fx_rate": 1.0})()
    converted = type("B", (), {"is_ttm": False, "fx_rate": 0.032})()
    change, _ = dcf_model.base_period_delta_nwc(dataset, local)
    converted_change, _ = dcf_model.base_period_delta_nwc(dataset, converted)
    assert converted_change == pytest.approx(change * 0.032)


# ---------------------------------------------------------------------------
# Trailing bases (DCF_ROUND2_INSTRUCTIONS.md §2.3)
# ---------------------------------------------------------------------------


def _ttm_series(revenues: list[float], ends: list[str]):
    labels = [f"P{i}" for i in range(len(revenues))]
    values = pd.DataFrame({"revenue": revenues}, index=labels)
    meta = pd.DataFrame({"report_date": pd.to_datetime(ends)}, index=labels)
    return values, meta


def test_ttm_growth_compares_with_the_period_four_quarters_earlier():
    values, meta = _ttm_series(
        [100.0, 104.0, 108.0, 113.0, 120.0],
        ["2025-06-28", "2025-09-27", "2025-12-27", "2026-03-28", "2026-06-27"],
    )
    result = fin.ttm_revenue_growth(values, meta)
    assert result.ok and result.basis == fin.TTM_GROWTH_BASIS
    assert result.value == pytest.approx(0.20)
    assert result.periods == ("P0", "P4")


def test_ttm_growth_is_unavailable_without_a_period_a_year_back():
    values, meta = _ttm_series([100.0, 120.0], ["2026-03-28", "2026-06-27"])
    result = fin.ttm_revenue_growth(values, meta)
    assert not result.ok
    assert "about a year before" in result.detail


def test_ttm_growth_is_unavailable_for_a_filer_with_no_ttm_series():
    result = fin.ttm_revenue_growth(pd.DataFrame(), pd.DataFrame())
    assert not result.ok and "no trailing-twelve-month" in result.detail


def test_the_ttm_margin_basis_is_the_base_periods_own_margin():
    flows = pd.Series({"revenue": 400.0, "ebit": 120.0})
    stocks = pd.Series(dtype="float64")
    margin = fin.base_period_driver("operating_margin", flows, stocks, is_ttm=True, label="TTM")
    assert margin.value == pytest.approx(0.30)
    fiscal = fin.base_period_driver(
        "operating_margin", flows, stocks, is_ttm=False, label="FY2025"
    )
    assert not fiscal.ok and "fiscal year" in fiscal.detail


def test_ttm_capex_dna_and_nwc_bases_read_the_base_period():
    """DCF_ROUND4_INSTRUCTIONS.md §2.1/§2.2: today's capex, D&A and working
    capital, not five-year averages that predate an AI build-out."""
    flows = pd.Series({"revenue": 400.0, "ebit": 120.0, "dna": 20.0, "capex_acquisition": -60.0})
    stocks = pd.Series({
        "total_current_assets": 150.0, "total_current_liabilities": 100.0,
        "cash": 20.0, "short_term_investments": 10.0, "short_term_debt": 5.0,
    })
    read = lambda key: fin.base_period_driver(  # noqa: E731
        key, flows, stocks, is_ttm=True, label="TTM to FY2026 Q2"
    )
    assert read("capex_pct_revenue").value == pytest.approx(0.15)
    assert read("dna_pct_revenue").value == pytest.approx(0.05)
    # NWC = (150 − 20 − 10) − (100 − 5) = 25, over TTM revenue of 400.
    assert read("nwc_pct_revenue").value == pytest.approx(0.0625)
    assert read("capex_pct_revenue").basis == fin.TTM_BASIS


def test_every_ttm_basis_falls_back_when_the_base_is_a_fiscal_year():
    flows = pd.Series({"revenue": 400.0, "ebit": 120.0, "dna": 20.0, "capex_acquisition": -60.0})
    for key in ("operating_margin", "capex_pct_revenue", "dna_pct_revenue", "nwc_pct_revenue"):
        result = fin.base_period_driver(
            key, flows, pd.Series(dtype="float64"), is_ttm=False, label="FY2025"
        )
        assert not result.ok and "fiscal year" in result.detail


def _capex_ratios(values: list[float]) -> pd.DataFrame:
    years = range(2026 - len(values) + 1, 2027)
    return pd.DataFrame({"capex_pct_revenue": values}, index=list(years))


def test_capex_normalises_to_its_own_five_year_average():
    """A build-out is not held forever: MSFT-like capex, doubled in two years,
    returns to the average of the last five years — not to the pre-build-out
    level, and the sixth year back plays no part."""
    normal = fin.normal_capex(_capex_ratios([0.99, 0.12, 0.13, 0.18, 0.23, 0.35]))
    assert normal.value == pytest.approx((0.12 + 0.13 + 0.18 + 0.23 + 0.35) / 5)
    assert normal.basis == fin.NORMAL_CAPEX_BASIS
    assert normal.periods == ("2022", "2023", "2024", "2025", "2026")
    assert normal.detail == ""


def test_capex_below_its_average_normalises_up():
    """Symmetric, so the rule never only flatters: a foundry off its peak reverts up."""
    normal = fin.normal_capex(_capex_ratios([0.50, 0.47, 0.43, 0.33, 0.34]))
    assert normal.value > 0.34


def test_a_missing_capex_year_is_excluded_not_zero_filled():
    normal = fin.normal_capex(_capex_ratios([0.10, np.nan, 0.20, 0.30]))
    assert normal.value == pytest.approx(0.20)
    assert normal.years_used == 3
    assert "Only 3 of the requested 5" in normal.detail


def _roic_frame(**columns) -> pd.DataFrame:
    base = {
        "ebit": [100.0, 120.0, 150.0],
        "pretax_income": [100.0, 120.0, 150.0],
        "tax_expense": [-20.0, -24.0, -30.0],   # a 20% rate, signed as Sharadar ships it
        "total_equity": [300.0, 400.0, 500.0],
        "short_term_debt": [10.0, 10.0, 10.0],
        "long_term_debt": [90.0, 90.0, 90.0],
        "minority_interest": [0.0, 0.0, 0.0],
        "cash": [50.0, 50.0, 50.0],
        "short_term_investments": [50.0, 50.0, 50.0],
    }
    base.update(columns)
    return pd.DataFrame(base, index=[2023, 2024, 2025])


def test_invested_capital_is_the_financing_side_net_of_the_bridges_cash():
    capital = fin.invested_capital(_roic_frame())
    assert list(capital) == [300.0, 400.0, 500.0]   # equity + 100 debt − 100 cash and STI


def test_an_unreported_debt_line_is_absent_not_a_reason_to_drop_the_year():
    frame = _roic_frame().drop(columns=["short_term_debt", "long_term_debt"])
    assert fin.invested_capital(frame).iloc[-1] == pytest.approx(400.0)


def test_roic_is_nopat_over_average_invested_capital():
    roic = fin.return_on_invested_capital(_roic_frame())
    assert np.isnan(roic.iloc[0])                       # no opening balance to average
    assert roic.iloc[1] == pytest.approx(120 * 0.8 / 350)
    assert roic.iloc[2] == pytest.approx(150 * 0.8 / 450)


def test_a_pretax_loss_year_has_no_return_rather_than_a_zero_one():
    frame = _roic_frame(pretax_income=[100.0, -5.0, 150.0])
    assert np.isnan(fin.return_on_invested_capital(frame).iloc[1])


def test_negative_invested_capital_has_no_meaningful_return():
    """Buybacks can run equity below zero; a return on less than nothing is noise."""
    frame = _roic_frame(total_equity=[-500.0, -500.0, 500.0])
    roic = fin.return_on_invested_capital(frame)
    assert np.isnan(roic.iloc[1])


def test_the_default_ronic_is_the_median_of_the_last_five_years():
    years = list(range(2016, 2026))
    ebit = [10.0] * 3 + [50.0, 30.0, 999.0, 20.0, 40.0, 10.0, 60.0]
    frame = pd.DataFrame(
        {
            "ebit": ebit, "pretax_income": [1.0] * 10, "tax_expense": [0.0] * 10,
            "total_equity": [100.0] * 10,
        },
        index=years,
    )
    roic = fin.historical_roic(frame)
    assert roic.value == pytest.approx(0.40)            # median of 9.99, 0.2, 0.4, 0.1, 0.6
    assert roic.periods == ("2021", "2022", "2023", "2024", "2025")
    assert roic.basis == fin.ROIC_BASIS


def test_no_measurable_return_leaves_the_default_ronic_unavailable():
    frame = _roic_frame(pretax_income=[-1.0, -1.0, -1.0])
    roic = fin.historical_roic(frame)
    assert not roic.ok and roic.detail


def test_no_capex_history_means_nothing_to_normalise_to():
    normal = fin.normal_capex(_capex_ratios([np.nan, np.nan]))
    assert not normal.ok
    assert normal.detail


# ---------------------------------------------------------------------------
# Year-1 growth: the lower of TTM and the five-year CAGR
# ---------------------------------------------------------------------------


def _growth_default(value, basis):
    return fin.DriverDefault("revenue_growth", value, basis, 1, ("P",))


def test_a_trailing_burst_above_the_five_year_record_is_not_extrapolated():
    """NVDA: 83% trailing against a 67% five-year CAGR — start from 67%."""
    lower = fin.lower_of_growth(
        _growth_default(0.834, fin.TTM_GROWTH_BASIS), _growth_default(0.669, "5-year CAGR")
    )
    assert lower.value == pytest.approx(0.669)
    assert lower.basis == fin.LOWER_GROWTH_BASIS
    assert lower.detail.startswith("Using 5-year CAGR (66.90%), the lower")


def test_a_trailing_slowdown_is_kept():
    lower = fin.lower_of_growth(
        _growth_default(0.04, fin.TTM_GROWTH_BASIS), _growth_default(0.10, "5-year CAGR")
    )
    assert lower.value == pytest.approx(0.04)
    assert "TTM year-over-year" in lower.detail


def test_the_lower_of_rule_needs_both_rates():
    """No silent fallback: with one side missing the basis is unavailable."""
    missing = fin.DriverDefault(
        "revenue_growth", None, fin.TTM_GROWTH_BASIS, 0, (), "no TTM series."
    )
    lower = fin.lower_of_growth(missing, _growth_default(0.10, "5-year CAGR"))
    assert not lower.ok and "TTM year-over-year is unavailable" in lower.detail
