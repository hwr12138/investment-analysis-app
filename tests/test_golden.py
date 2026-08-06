"""Golden-file test: one full model run, snapshotted.

Any change to these numbers must be deliberate. If a refactor moves them, either
the refactor is wrong or the snapshot needs updating with a reason.

The whole run is driven off recorded fixtures — no network, no live API.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.data import market_data, sharadar_client as sc
from app.logic.dcf import financials as fin
from app.logic.dcf import model as dcf_model
from app.logic.dcf import validation as val
from app.logic.dcf import valuation as dcf
from app.logic.dcf import wacc as wl

VALUATION_DATE = date(2026, 7, 31)


def build_model(statements_payload, tickers_response, monkeypatch) -> dict:
    """Runs the whole pipeline against recorded fixtures.

    Shared with the export and reverse tests so all three exercise the same
    assembled model.
    """
    prices = pd.DataFrame(
        {
            "Date": pd.date_range("2026-07-01", periods=20, freq="B"),
            "Last Closing Price": [50.0] * 20,
            "Adjusted Closing Price": [50.0] * 20,
            "Split-Adjusted Close": [50.0] * 20,
            "Common Shares Outstanding": [1_000.0] * 20,
            "Market-Cap": [50_000.0] * 20,
            "EV/EBITDA - ttm": [12.0] * 20,
        }
    )

    def _get(table, params, ticker=None):
        return tickers_response if table == "TICKERS" else statements_payload

    def _statements(ticker, *, annual=True, ttm=False, asreported=False):
        """Annual data only — this company has no quarterly history at Sharadar.

        That is a real situation, and it is what makes the model fall back to
        the latest fiscal year as its base period.
        """
        if not annual or ttm:
            raise sc.TickerNotFoundError(
                "No quarterly history.", table="SF1", ticker=ticker
            )
        return sc._fetch_statements(ticker, annual=True, asreported=asreported)

    monkeypatch.setattr(sc, "_get", _get)
    monkeypatch.setattr(sc, "fetch_ticker_meta", lambda ticker: sc._fetch_ticker_meta(ticker))
    monkeypatch.setattr(sc, "fetch_statements", _statements)
    monkeypatch.setattr(sc, "fetch_prices", lambda *a, **k: prices)

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
def built(statements_payload, tickers_response, monkeypatch):
    return build_model(statements_payload, tickers_response, monkeypatch)


# --- The snapshot ----------------------------------------------------------

# Re-baselined for the Sharadar feed. The fixture's pretax income is now
# operating income less interest (310) rather than a separately stated 300,
# which moves the effective tax rate and everything downstream of it.
GOLDEN = {
    "fiscal_years": 3,
    "base_label": "FY2025",
    "stub": 0.16712328767123288,
    "effective_tax_rate": 0.19285714285714287,
    "wacc": 0.08923145016908693,
    "fcff_method_a": 240.0,
    "fcff_method_b": 237.21428571428572,
    "enterprise_value_gordon": 5463.246764785289,
    "value_per_share_gordon": 5.349749272064643,
    "enterprise_value_exit": 6340.346933503249,
    "value_per_share_exit": 6.218165280696286,
}


def test_golden_scalars(built):
    result = built["result"]
    assert len(built["dataset"].annual) == GOLDEN["fiscal_years"]
    assert built["base"].label == GOLDEN["base_label"]
    assert built["stub"] == pytest.approx(GOLDEN["stub"])
    assert built["etr"] == pytest.approx(GOLDEN["effective_tax_rate"])
    assert result.wacc.wacc == pytest.approx(GOLDEN["wacc"])
    assert result.base_fcff.method_a == pytest.approx(GOLDEN["fcff_method_a"])
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


def test_a_zero_interest_line_is_flagged_rather_than_absorbed(built):
    """The fixture reports interest expense as 0, exactly like AAPL.

    Sharadar writes 0 both for a genuine nil and for a line item absent from
    the filing. Method A still computes — the add-back is just zero — but the
    ambiguity has to reach the user rather than being silently absorbed.
    """
    fcff = built["result"].base_fcff
    assert fcff.method_a is not None
    assert any("0" in note and "not disclosed" in note for note in fcff.notes)

    flagged = [i for i in built["issues"] if i.key == "zero_filled"]
    assert flagged, "the zero-fill check did not fire"
    assert "Interest Expense" in flagged[0].values


def test_financial_sector_is_rejected_before_anything_is_computed(
    statements_payload, financial_tickers_response, monkeypatch
):
    def _get(table, params, ticker=None):
        return financial_tickers_response if table == "TICKERS" else statements_payload

    monkeypatch.setattr(sc, "_get", _get)
    monkeypatch.setattr(sc, "fetch_ticker_meta", sc._fetch_ticker_meta)
    monkeypatch.setattr(sc, "fetch_statements", sc._fetch_statements)

    with pytest.raises(val.UnsupportedSectorError) as caught:
        dcf_model.load_company("JPM")
    assert caught.value.template == "Financial Services"


def test_a_null_vendor_sector_alone_does_not_reject(
    statements_payload, tickers_response, monkeypatch
):
    """Sharadar's `sector` is a vendor label and can be null; SIC decides."""
    from tests.conftest import TICKERS_COLUMNS

    row = list(tickers_response["datatable"]["data"][0])
    row[TICKERS_COLUMNS.index("sector")] = None
    row[TICKERS_COLUMNS.index("sicsector")] = None
    payload = {
        "datatable": dict(tickers_response["datatable"], data=[row]),
        "meta": {"next_cursor_id": None},
    }

    def _get(table, params, ticker=None):
        return payload if table == "TICKERS" else statements_payload

    monkeypatch.setattr(sc, "_get", _get)
    monkeypatch.setattr(sc, "fetch_ticker_meta", sc._fetch_ticker_meta)
    monkeypatch.setattr(sc, "fetch_statements", sc._fetch_statements)
    monkeypatch.setattr(sc, "fetch_prices", lambda *a, **k: pd.DataFrame(
        {"Date": pd.date_range("2026-07-01", periods=5, freq="B"),
         "Last Closing Price": [50.0] * 5}
    ))

    dataset = dcf_model.load_company("TEST")
    assert dataset.template == ""


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


def _yahoo_frame(rows: list[tuple[date, float | None]]) -> pd.DataFrame:
    """A frame shaped like ``yf.Ticker.history`` — tz-aware index, extra columns."""
    index = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="America/Chicago") for d, _ in rows], name="Date"
    )
    return pd.DataFrame(
        {"Open": [c for _, c in rows], "Close": [c for _, c in rows], "Volume": 0},
        index=index,
    )


def _patch_yahoo(monkeypatch, frame):
    """Stands in for ``market_data.yf.Ticker(...).history(...)``."""
    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            if isinstance(frame, Exception):
                raise frame
            return frame

    monkeypatch.setattr(market_data.yf, "Ticker", _Ticker)


def test_treasury_reads_the_last_populated_close(monkeypatch):
    """^TNX quotes percentage points; the newest non-NaN day is the one used."""
    _patch_yahoo(monkeypatch, _yahoo_frame([
        (date(2026, 7, 30), 4.745),
        (date(2026, 7, 31), 4.627),
        (date(2026, 8, 3), None),  # holiday or not-yet-printed
    ]))
    result = market_data._fetch_yahoo_yield.__wrapped__(market_data.TNX_SYMBOL)
    assert result == (pytest.approx(4.627), date(2026, 7, 31), "")


def test_treasury_failure_returns_no_value(monkeypatch):
    monkeypatch.setattr(
        market_data,
        "_fetch_yahoo_yield",
        lambda symbol: (None, None, f"Yahoo Finance returned no quotes for {symbol}"),
    )
    result = market_data.fetch_10y_treasury_rate()
    assert result.value is None
    assert not result.ok
    assert "^TNX" in result.detail


def test_treasury_success_reports_percent_as_decimal(monkeypatch):
    monkeypatch.setattr(
        market_data, "_fetch_yahoo_yield", lambda symbol: (4.627, date(2026, 7, 31), "")
    )
    result = market_data.fetch_10y_treasury_rate()
    assert result.value == pytest.approx(0.04627)
    assert result.as_of == date(2026, 7, 31)


def test_treasury_at_the_old_ten_times_quote_is_rejected(monkeypatch):
    """Yahoo published ^TNX at 10x the yield for years. 46.27% is not a rate."""
    _patch_yahoo(monkeypatch, _yahoo_frame([(date(2026, 7, 31), 46.27)]))
    rate, _, error = market_data._fetch_yahoo_yield.__wrapped__(market_data.TNX_SYMBOL)
    assert rate is None
    assert "plausible" in error


def test_treasury_empty_frame_is_a_failure_not_a_zero(monkeypatch):
    """yfinance returns an empty frame instead of raising on a bad fetch."""
    _patch_yahoo(monkeypatch, pd.DataFrame())
    rate, _, error = market_data._fetch_yahoo_yield.__wrapped__(market_data.TNX_SYMBOL)
    assert rate is None
    assert "no quotes" in error


def test_treasury_network_error_is_caught(monkeypatch):
    _patch_yahoo(monkeypatch, RuntimeError("connection reset"))
    rate, _, error = market_data._fetch_yahoo_yield.__wrapped__(market_data.TNX_SYMBOL)
    assert rate is None
    assert "Could not reach Yahoo Finance" in error


def test_short_rate_falls_back_when_the_fetch_fails(monkeypatch):
    """The portfolio pages take a plain float, so this one does substitute."""
    monkeypatch.setattr(
        market_data, "_fetch_yahoo_yield", lambda symbol: (None, None, "down")
    )
    assert market_data.fetch_current_risk_free_rate() == pytest.approx(0.04)

    monkeypatch.setattr(
        market_data, "_fetch_yahoo_yield", lambda symbol: (3.73, date(2026, 8, 4), "")
    )
    assert market_data.fetch_current_risk_free_rate() == pytest.approx(0.0373)
