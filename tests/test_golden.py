"""Golden-file test: one full model run, snapshotted.

Any change to these numbers must be deliberate. If a refactor moves them, either
the refactor is wrong or the snapshot needs updating with a reason.

The whole run is driven off recorded fixtures — no network, no live API.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.data import market_data, simfin_client as sc
from app.logic.dcf import financials as fin
from app.logic.dcf import model as dcf_model
from app.logic.dcf import validation as val
from app.logic.dcf import valuation as dcf
from app.logic.dcf import wacc as wl

VALUATION_DATE = date(2026, 7, 31)


def build_model(statements_payload, monkeypatch) -> dict:
    """Runs the whole pipeline against recorded fixtures.

    Shared with the export tests so both exercise the same assembled model.
    """
    prices = pd.DataFrame(
        {
            "Date": pd.date_range("2026-07-01", periods=20, freq="B"),
            "Last Closing Price": [50.0] * 20,
            "Adjusted Closing Price": [50.0] * 20,
            "Common Shares Outstanding": [1_000.0] * 20,
            "Market-Cap": [50_000.0] * 20,
            "EV/EBITDA - ttm": [12.0] * 20,
        }
    )
    common_shares = pd.DataFrame(
        {"endDate": [pd.Timestamp("2026-06-30")], "value": [1_000.0]}
    )
    weighted = pd.DataFrame(
        {
            "endDate": [pd.Timestamp("2025-09-30")],
            "fyear": [2025],
            "period": ["FY"],
            "basic": [1_000.0],
            "diluted": [1_010.0],
        }
    )

    def _statements(ticker, statements="PL,BS,CF,DERIVED", period="FY", **kwargs):
        """Annual data only — this company has no quarterly history at SimFin.

        That is a real situation, and it is what makes the model fall back to
        the latest fiscal year as its base period.
        """
        if period.strip().upper() != "FY":
            return sc.StatementSet(
                ticker=ticker, simfin_id=999, name="TESTCO INC", currency="USD",
                template="GENERAL", isin="", period=period, ttm=kwargs.get("ttm", False),
                asreported=False, frames={},
            )
        return sc._fetch_statements(ticker, statements, period, **kwargs)

    monkeypatch.setattr(sc, "_request", lambda *a, **k: statements_payload)
    monkeypatch.setattr(sc, "lookup_company", lambda ticker: {
        "id": 999, "ticker": "TEST", "name": "TESTCO INC",
        "sectorCode": 101001, "sectorName": "Computer Hardware",
        "industryName": "Technology",
    })
    monkeypatch.setattr(sc, "fetch_statements", _statements)
    monkeypatch.setattr(sc, "fetch_general", lambda ticker: {"endFy": 9, "ticker": "TEST"})
    monkeypatch.setattr(sc, "fetch_prices", lambda *a, **k: prices)
    monkeypatch.setattr(sc, "fetch_common_shares", lambda *a, **k: common_shares)
    monkeypatch.setattr(sc, "fetch_weighted_shares", lambda *a, **k: weighted)
    monkeypatch.setattr(sc, "fetch_filings", lambda *a, **k: pd.DataFrame())

    dataset = dcf_model.load_company("TEST")
    base = dcf_model.base_period(dataset)

    etr = float(fin.effective_tax_rate(dataset.annual).dropna().tail(5).median())
    delta_nwc = float(fin.delta_nwc(dataset.annual).dropna().iloc[-1])
    base_fcff = dcf_model.base_period_fcff(
        dataset, base, tax_rate=etr, delta_nwc=delta_nwc
    )

    wacc_result = wl.compute_wacc(
        equity_value=50_000.0,
        debt_value=float(fin.total_debt(pd.DataFrame([base.stocks])).iloc[0]),
        risk_free_rate=0.04,
        beta=1.10,
        equity_risk_premium=0.045,
        cost_of_debt=0.05,
        marginal_tax_rate=0.21,
    )

    ratios = fin.driver_ratios(dataset.annual)
    drivers = dcf.DriverPath.flat(
        5,
        **{
            key: float(fin.driver_default(ratios, key, "3-year average").value)
            for key in fin.DRIVER_KEYS
        },
    )

    fiscal_year_end = fin.fiscal_year_end(
        dataset.fiscal_year_end_month, VALUATION_DATE, date(2025, 9, 30)
    )
    stub = dcf.stub_fraction(VALUATION_DATE, fiscal_year_end)

    result = dcf_model.run_model(
        dataset,
        base,
        drivers=drivers,
        wacc_result=wacc_result,
        terminal_growth=0.025,
        exit_multiple=12.0,
        stub=stub,
        mid_year=True,
        shares=dataset.shares_weighted_diluted,
        share_basis="Diluted weighted-average (FY2025)",
        include_short_term_investments=True,
        operating_cash_floor=0.0,
        non_operating_assets=None,
        base_fcff=base_fcff,
    )

    context = dcf_model.build_validation_context(
        dataset,
        base=base,
        base_fcff=base_fcff,
        valuation_date=VALUATION_DATE,
        effective_tax_rate=etr,
        equity_risk_premium=0.045,
        result=result,
        terminal_growth=0.025,
        risk_free_rate=0.04,
    )
    issues, check_results = val.run_all(context)

    return {
        "dataset": dataset,
        "base": base,
        "result": result,
        "issues": issues,
        "check_results": check_results,
        "stub": stub,
        "etr": etr,
        # The run inputs, so a test can re-run the model with one of them moved.
        "drivers": drivers,
        "wacc_result": wacc_result,
        "base_fcff": base_fcff,
    }


@pytest.fixture
def built(statements_payload, monkeypatch):
    return build_model(statements_payload, monkeypatch)


# --- The snapshot ----------------------------------------------------------

GOLDEN = {
    "fiscal_years": 3,
    "base_label": "FY2025",
    "stub": 0.16712328767123288,
    "effective_tax_rate": 0.2,
    "wacc": 0.08923145016908693,
    "fcff_method_b": 235.0,
    "enterprise_value_gordon": 5412.386011417202,
    "value_per_share_gordon": 5.2993920905120815,
    "enterprise_value_exit": 6329.902849731228,
    "value_per_share_exit": 6.207824603694285,
}


def test_golden_scalars(built):
    result = built["result"]
    assert len(built["dataset"].annual) == GOLDEN["fiscal_years"]
    assert built["base"].label == GOLDEN["base_label"]
    assert built["stub"] == pytest.approx(GOLDEN["stub"])
    assert built["etr"] == pytest.approx(GOLDEN["effective_tax_rate"])
    assert result.wacc.wacc == pytest.approx(GOLDEN["wacc"])
    assert result.base_fcff.method_b == pytest.approx(GOLDEN["fcff_method_b"])


def test_golden_valuation(built):
    result = built["result"]
    assert result.enterprise_values["gordon"] == pytest.approx(
        GOLDEN["enterprise_value_gordon"]
    )
    assert result.bridges["gordon"].value_per_share == pytest.approx(
        GOLDEN["value_per_share_gordon"]
    )
    assert result.enterprise_values["exit_multiple"] == pytest.approx(
        GOLDEN["enterprise_value_exit"]
    )
    assert result.bridges["exit_multiple"].value_per_share == pytest.approx(
        GOLDEN["value_per_share_exit"]
    )


def test_golden_enterprise_value_is_pv_plus_terminal(built):
    """The headline number must decompose back into its two parts."""
    result = built["result"]
    for method, enterprise_value in result.enterprise_values.items():
        assert enterprise_value == pytest.approx(
            result.projection.pv_explicit + result.terminal_values[method].present_value
        )


def test_golden_fixture_ties_out_and_raises_no_critical_issue(built):
    """The fixture is internally consistent, so only the short-history warning fires."""
    critical = [i for i in built["issues"] if i.severity is val.Severity.CRITICAL]
    assert critical == [], [i.check for i in critical]
    warnings = {i.check for i in built["issues"]}
    assert "Sufficient usable history" in warnings


def test_method_a_is_uncomputable_without_an_interest_line(built):
    """The fixture reports no interest expense, exactly like AAPL."""
    assert built["result"].base_fcff.method_a is None
    assert any("interest expense" in note for note in built["result"].base_fcff.notes)


def test_financial_sector_is_rejected_before_anything_is_computed(
    statements_payload, monkeypatch
):
    payload = [dict(statements_payload[0], template="BANKS")]
    monkeypatch.setattr(sc, "_request", lambda *a, **k: payload)
    monkeypatch.setattr(sc, "lookup_company", lambda ticker: {
        "id": 1, "ticker": "JPM", "name": "JPMORGAN", "sectorCode": 104002,
        "sectorName": "Banks", "industryName": "Financial Services",
    })
    monkeypatch.setattr(sc, "fetch_statements", sc._fetch_statements)

    with pytest.raises(val.UnsupportedSectorError) as caught:
        dcf_model.load_company("JPM")
    assert caught.value.template == "BANKS"


def test_null_sector_alone_does_not_reject(statements_payload, monkeypatch):
    """SimFin leaves sectorCode null for some companies; the template rules."""
    monkeypatch.setattr(sc, "_request", lambda *a, **k: statements_payload)
    monkeypatch.setattr(sc, "lookup_company", lambda ticker: {
        "id": 1, "ticker": "TEST", "name": "TESTCO",
        "sectorCode": None, "sectorName": None, "industryName": None,
    })
    monkeypatch.setattr(sc, "fetch_statements", sc._fetch_statements)
    monkeypatch.setattr(sc, "fetch_general", lambda ticker: {"endFy": 9})
    monkeypatch.setattr(sc, "fetch_prices", lambda *a, **k: pd.DataFrame(
        {"Date": pd.date_range("2026-07-01", periods=5, freq="B"),
         "Last Closing Price": [50.0] * 5}
    ))
    monkeypatch.setattr(sc, "fetch_common_shares", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(sc, "fetch_weighted_shares", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(sc, "fetch_filings", lambda *a, **k: pd.DataFrame())

    dataset = dcf_model.load_company("TEST")
    assert dataset.template == "GENERAL"


def test_erp_scrape_failure_returns_no_value(monkeypatch):
    """A broken scrape must never substitute a constant."""
    class _Broken:
        status_code = 500
        text = ""

    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: _Broken())
    result = market_data.fetch_erp.__wrapped__()
    assert result.value is None
    assert not result.ok
    assert "500" in result.detail


def test_erp_outside_the_plausible_band_is_rejected(monkeypatch):
    class _Silly:
        status_code = 200
        text = "<p><strong>Implied ERP on July 1, 2026 </strong>= 87.5% (whatever)"

    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: _Silly())
    result = market_data.fetch_erp.__wrapped__()
    assert result.value is None
    assert "plausible" in result.detail


def test_erp_survives_the_split_element_the_page_actually_uses(monkeypatch):
    """The live page splits the figure as `4.<span></span><span>18%`."""
    class _Real:
        status_code = 200
        text = (
            '<p><strong>Implied ERP on July 1, 2026 </strong>= 4.'
            '<span class="MsoNormal"></span><span class="MsoNormal">18% (Trailing)'
        )

    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: _Real())
    result = market_data.fetch_erp.__wrapped__()
    assert result.value == pytest.approx(0.0418)
    assert result.as_of == date(2026, 7, 1)


def test_treasury_failure_returns_no_value(monkeypatch):
    monkeypatch.setattr(
        market_data, "_fetch_fmp_treasury_rates", lambda: ({}, "FMP_API_KEY is not set")
    )
    result = market_data.fetch_10y_treasury_rate()
    assert result.value is None
    assert "FMP_API_KEY" in result.detail
