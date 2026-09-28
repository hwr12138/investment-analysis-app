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
from app.data.sharadar import reference as sh_reference
from app.data.sharadar import statements as sh_statements
from app.data.sharadar import transport as sh_transport
from app.logic.dcf import financials as fin
from app.logic.dcf import model as dcf_model
from app.logic.dcf import validation as val
from app.logic.dcf import valuation as dcf
from app.logic.dcf import wacc as wl
from app.provenance import Origin, Sourced

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
        return sh_statements._fetch_statements(ticker, annual=True, asreported=asreported)

    monkeypatch.setattr(sh_transport, "_get", _get)
    monkeypatch.setattr(
        sc, "fetch_ticker_meta", lambda ticker: sh_reference._fetch_ticker_meta(ticker)
    )
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

    # Annual-only, so the forecast is anchored to the last fiscal year end and
    # the value rolled forward ten months to the valuation date.
    elapsed = dcf.elapsed_years(base.flows_as_of, VALUATION_DATE)

    result = dcf_model.run_model(
        dataset,
        base,
        drivers=drivers,
        wacc_result=wacc_result,
        terminal_growth=0.025,
        exit_multiple=12.0,
        elapsed=elapsed,
        mid_year=True,
        shares=dataset.shares_weighted_diluted,
        share_basis="Diluted weighted-average (FY2025)",
        include_short_term_investments=True,
        operating_cash_floor=0.0,
        non_operating_assets=None,
        base_fcff=base_fcff,
        terminal_tax_rate=wacc_result.marginal_tax_rate,
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
        drivers=drivers,
        beta_in_use=1.10,
    )
    issues, check_results = val.run_all(context)

    return {
        "dataset": dataset,
        "base": base,
        "result": result,
        "issues": issues,
        "check_results": check_results,
        "elapsed": elapsed,
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
#
# Re-baselined again for the DCF model review (DCF_MODEL_REVIEW.md):
# * F4 — the forecast is anchored to the base period's end (FY2025, 30 Sep
#   2025) and rolled forward 0.833 years to the valuation date, instead of a
#   0.167 stub to the next fiscal year end. Year 1 is now a whole year.
# * F3 — the exit value is discounted from the end of year 5, not mid-year.
# * F5/F16 — the Gordon value capitalises a normalised year N + 1: reinvestment
#   of g ÷ RONIC (RONIC = WACC) and the 21% marginal tax rate, rather than the
#   final forecast year's FCFF grown at g.
#
# Re-baselined for DCF round 4 (DCF_ROUND4_INSTRUCTIONS.md §2.1): the working
# capital driver is now NWC as a share of revenue — here a three-year average of
# −2.20% — in place of ΔNWC ÷ Δrevenue, which divides by a near-zero change in
# a flat year (DHR's FY2024 read 2,453%). ΔNWC is still that ratio × Δrevenue.
# The EBITA proxy (§2.3), on by default, adds nothing here: the fixture's final
# capex (154.7) exceeds its D&A (113.6).
GOLDEN = {
    "fiscal_years": 3,
    "base_label": "FY2025",
    "base_period_end": date(2025, 9, 30),
    "elapsed": 0.8328767123287671,
    "effective_tax_rate": 0.19285714285714287,
    "wacc": 0.08923145016908693,
    "fcff_method_a": 240.0,
    "fcff_method_b": 237.21428571428572,
    "enterprise_value_gordon": 4461.593930203337,
    "value_per_share_gordon": 4.3580137922805315,
    "enterprise_value_exit": 6268.782449734702,
    "value_per_share_exit": 6.147309356172972,
}


def test_golden_scalars(built):
    result = built["result"]
    assert len(built["dataset"].annual) == GOLDEN["fiscal_years"]
    assert built["base"].label == GOLDEN["base_label"]
    assert built["base"].flows_as_of == GOLDEN["base_period_end"]
    assert built["elapsed"] == pytest.approx(GOLDEN["elapsed"])
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
    """The fixture is internally consistent, so no CRITICAL check fires.

    Warnings still do — the short history among them, and the forecast-sanity
    checks, since the fixture's $50 price was never meant to match its value.
    """
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


def test_a_zero_fx_line_is_not_flagged_as_an_ambiguous_zero(built):
    """The fixture reports ``ncfx`` as 0, like any filer without FX exposure.

    The zero is as ambiguous as any other, but nothing turns on resolving it,
    and ``check_cash_flow_ties`` already treats an absent FX line as expected.
    Flagging it here as well contradicted that check and put a warning on the
    majority of domestic filers.
    """
    flagged = [i for i in built["issues"] if i.key == "zero_filled"]
    assert flagged, "the zero-fill check did not fire at all"
    assert not any("Exchange" in label for label in flagged[0].values), (
        f"FX was flagged as an ambiguous zero: {list(flagged[0].values)}"
    )

    tie = [r for r in built["check_results"] if r.name == "Cash flow statement ties"]
    assert tie and tie[0].status == "pass"


def test_financial_sector_is_rejected_before_anything_is_computed(
    statements_payload, financial_tickers_response, monkeypatch
):
    def _get(table, params, ticker=None):
        return financial_tickers_response if table == "TICKERS" else statements_payload

    monkeypatch.setattr(sh_transport, "_get", _get)
    monkeypatch.setattr(sc, "fetch_ticker_meta", sh_reference._fetch_ticker_meta)
    monkeypatch.setattr(sc, "fetch_statements", sh_statements._fetch_statements)

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

    monkeypatch.setattr(sh_transport, "_get", _get)
    monkeypatch.setattr(sc, "fetch_ticker_meta", sh_reference._fetch_ticker_meta)
    monkeypatch.setattr(sc, "fetch_statements", sh_statements._fetch_statements)
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
    result = market_data.fetch_erp()
    assert result.value is None
    assert not result.ok
    assert "500" in result.detail


def test_erp_outside_the_plausible_band_is_rejected(monkeypatch):
    class _Silly:
        status_code = 200
        text = "<p><strong>Implied ERP on July 1, 2026 </strong>= 87.5% (whatever)"

    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: _Silly())
    result = market_data.fetch_erp()
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
    result = market_data.fetch_erp()
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
    result = market_data._fetch_yahoo_yield(market_data.TNX_SYMBOL)
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
    rate, _, error = market_data._fetch_yahoo_yield(market_data.TNX_SYMBOL)
    assert rate is None
    assert "plausible" in error


def test_treasury_empty_frame_is_a_failure_not_a_zero(monkeypatch):
    """yfinance returns an empty frame instead of raising on a bad fetch."""
    _patch_yahoo(monkeypatch, pd.DataFrame())
    rate, _, error = market_data._fetch_yahoo_yield(market_data.TNX_SYMBOL)
    assert rate is None
    assert "no quotes" in error


def test_treasury_network_error_is_caught(monkeypatch):
    _patch_yahoo(monkeypatch, RuntimeError("connection reset"))
    rate, _, error = market_data._fetch_yahoo_yield(market_data.TNX_SYMBOL)
    assert rate is None
    assert "Could not reach Yahoo Finance" in error


def test_short_rate_reports_a_failed_fetch_rather_than_substituting(monkeypatch):
    """It used to return 4% on failure, which read exactly like a live quote."""
    monkeypatch.setattr(
        market_data, "_fetch_yahoo_yield", lambda symbol: (None, None, "down")
    )
    failed = market_data.fetch_current_risk_free_rate()
    assert failed.value is None
    assert failed.origin is Origin.UNAVAILABLE
    assert failed.detail == "down"

    monkeypatch.setattr(
        market_data, "_fetch_yahoo_yield", lambda symbol: (3.73, date(2026, 8, 4), "")
    )
    quoted = market_data.fetch_current_risk_free_rate()
    assert quoted.value == pytest.approx(0.0373)
    assert quoted.origin is Origin.REPORTED
    assert quoted.as_of == date(2026, 8, 4)
    assert market_data.IRX_SYMBOL in quoted.source


# --- Currency and ADRs (DCF_MODEL_REVIEW.md F2) ------------------------------


TWD_PER_USD = 31.25
SPOT_USD_PER_TWD = 1.0 / TWD_PER_USD


def _twd_inputs(fxusd_by_year: dict[int, float] | None = None):
    """The fixture company, reporting in TWD, with Sharadar's ``fxusd`` set."""
    from tests.conftest import (
        _ANNUAL_ROWS, SF1_COLUMNS, TICKERS_COLUMNS, TICKERS_ROW, sf1_payload, tickers_payload,
    )

    row = list(TICKERS_ROW)
    row[TICKERS_COLUMNS.index("currency")] = "TWD"
    row[TICKERS_COLUMNS.index("category")] = "ADR Common Stock"
    rows = [list(r) for r in _ANNUAL_ROWS]
    for r in rows:
        year = int(str(r[SF1_COLUMNS.index("calendardate")])[:4])
        r[SF1_COLUMNS.index("fxusd")] = (fxusd_by_year or {}).get(year, TWD_PER_USD)
    return sf1_payload(rows), tickers_payload(row)


def _spot(monkeypatch, rate: float | None, reason: str = "no quote"):
    monkeypatch.setattr(
        market_data, "fetch_fx_rate",
        lambda currency: Sourced.reported(rate, f"Yahoo Finance {currency}USD=X",
                                          as_of=date(2026, 7, 31))
        if rate is not None else Sourced.unavailable(f"Yahoo Finance {currency}", reason),
    )


def test_a_reporter_with_no_exchange_rate_is_still_blocked(monkeypatch):
    """No spot rate means no valuation — a rate is never substituted."""
    statements, tickers = _twd_inputs()
    _spot(monkeypatch, None, reason="Yahoo Finance returned no quotes for TWDUSD=X")
    with pytest.raises(val.UnsupportedCurrencyError) as caught:
        build_model(statements, tickers, monkeypatch)
    assert caught.value.statement_currency == "TWD"
    assert "no quotes" in str(caught.value)


def test_a_twd_reporter_is_valued_in_dollars_end_to_end(
    statements_payload, tickers_response, monkeypatch
):
    """DCF_ROUND2_INSTRUCTIONS.md §2.4: drivers from the TWD history, every
    monetary input converted once at spot, the valuation struck in USD."""
    import streamlit as st

    local = build_model(statements_payload, tickers_response, monkeypatch)
    # The TICKERS record is cached per ticker; the second company reuses it.
    st.cache_data.clear()
    statements, tickers = _twd_inputs()
    _spot(monkeypatch, SPOT_USD_PER_TWD)
    converted = build_model(statements, tickers, monkeypatch)

    dataset, base = converted["dataset"], converted["base"]
    assert dataset.is_converted and dataset.currency == "TWD"
    assert base.currency == "USD" and base.fx_rate == pytest.approx(SPOT_USD_PER_TWD)
    # Monetary lines at spot; the published rate itself untouched.
    assert base.flows["revenue"] == pytest.approx(
        local["base"].flows["revenue"] * SPOT_USD_PER_TWD
    )
    assert base.stocks["cash"] == pytest.approx(local["base"].stocks["cash"] * SPOT_USD_PER_TWD)
    assert base.flows["fx_to_usd"] == pytest.approx(TWD_PER_USD)
    # Ratios are currency-free, so the drivers are identical.
    assert converted["drivers"] == local["drivers"]
    # Both base-period FCFF builds meet dollar figures, so both are converted.
    assert converted["base_fcff"].method_a == pytest.approx(
        local["base_fcff"].method_a * SPOT_USD_PER_TWD
    )
    # The bridge deducts converted debt.
    debt = [l for l in converted["result"].headline_bridge.lines if l.label == "Less: total debt"]
    local_debt = [l for l in local["result"].headline_bridge.lines if l.label == "Less: total debt"]
    assert debt[0].amount == pytest.approx(local_debt[0].amount * SPOT_USD_PER_TWD)

    results = {r.name: r for r in converted["check_results"]}
    assert results["Valuation inputs in the price's currency"].status == "pass"
    assert "converted from TWD" in results["Valuation inputs in the price's currency"].detail
    assert results["Spot FX rate agrees with Sharadar's"].status == "pass"
    assert results["Reporting currency stable against the dollar"].status == "pass"
    keys = {i.key for i in converted["issues"]}
    assert {"fx_spot_assumption", "foreign_accounting"} <= keys
    assert not [i for i in converted["issues"] if i.severity is val.Severity.CRITICAL]


def test_a_collapsing_currency_blocks_the_valuation(monkeypatch):
    """TWD losing half its value in two years is not a spot-rate world."""
    statements, tickers = _twd_inputs({2023: 15.0, 2024: 21.0, 2025: 30.0})
    _spot(monkeypatch, 1.0 / 30.0)
    built = build_model(statements, tickers, monkeypatch)
    blocking = [i for i in built["issues"] if i.severity is val.Severity.BLOCKING]
    assert [i.key for i in blocking] == ["fx_depreciation"]


def test_the_historical_multiple_puts_each_periods_ebitda_in_dollars(monkeypatch):
    """DAILY EV is in dollars; EBITDA is converted at its own period's fxusd."""
    statements, tickers = _twd_inputs({2025: 25.0})
    _spot(monkeypatch, 1.0 / 25.0)
    dataset = build_model(statements, tickers, monkeypatch)["dataset"]
    ebitda_twd = float(dataset.annual["ebit"].iloc[-1] + dataset.annual["dna"].iloc[-1])
    dataset.prices = dataset.prices.assign(**{"Enterprise Value": 15.0 * ebitda_twd / 25.0})
    assert dcf_model.historical_ev_ebitda(dataset).value == pytest.approx(15.0)


def test_fx_falls_back_to_the_inverted_pair(monkeypatch):
    """Yahoo lists some pairs only one way round."""
    frames = {
        "TWDUSD=X": pd.DataFrame(),
        "USDTWD=X": _yahoo_frame([(date(2026, 7, 31), TWD_PER_USD)]),
    }

    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            return frames[self.symbol]

    monkeypatch.setattr(market_data.yf, "Ticker", _Ticker)
    rate = market_data.fetch_fx_rate("TWD")
    assert rate.value == pytest.approx(SPOT_USD_PER_TWD)
    assert "inverted" in rate.source
    assert rate.as_of == date(2026, 7, 31)


def test_a_failed_fetch_is_not_cached(monkeypatch):
    """DCF_ROUND2_INSTRUCTIONS.md §2.8: a transient outage used to be cached
    for a day. The cache is not cleared between the two calls below."""
    class _Broken:
        status_code = 503
        text = ""

    class _Real:
        status_code = 200
        text = "<p><strong>Implied ERP on July 1, 2026 </strong>= 4.18% (Trailing)"

    responses = iter([_Broken(), _Real()])
    monkeypatch.setattr(market_data.requests, "get", lambda *a, **k: next(responses))
    assert not market_data.fetch_erp().ok
    assert market_data.fetch_erp().value == pytest.approx(0.0418)

    frames = iter([pd.DataFrame(), _yahoo_frame([(date(2026, 7, 31), 4.5)])])

    class _Ticker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            return next(frames)

    monkeypatch.setattr(market_data.yf, "Ticker", _Ticker)
    assert market_data.fetch_10y_treasury_rate().value is None
    assert market_data.fetch_10y_treasury_rate().value == pytest.approx(0.045)


def test_a_successful_fetch_is_cached(monkeypatch):
    calls = []

    class _Ticker:
        def __init__(self, symbol):
            calls.append(symbol)

        def history(self, **kwargs):
            return _yahoo_frame([(date(2026, 7, 31), 0.032)])

    monkeypatch.setattr(market_data.yf, "Ticker", _Ticker)
    market_data.fetch_fx_rate("TWD")
    market_data.fetch_fx_rate("TWD")
    assert calls == ["TWDUSD=X"]


def test_an_adr_share_factor_reaches_every_share_count(tickers_response, monkeypatch):
    """The weighted counts and EPS are per underlying share; the price is per ADR.

    Only the point-in-time count used to carry the factor, so the default
    denominator was in different units from the price it was compared with.
    """
    from tests.conftest import _ANNUAL_ROWS, SF1_COLUMNS, sf1_payload

    def load(factor: float):
        rows = [list(row) for row in _ANNUAL_ROWS]
        for row in rows:
            row[SF1_COLUMNS.index("sharefactor")] = factor
        return build_model(sf1_payload(rows), tickers_response, monkeypatch)["dataset"]

    plain, adr = load(1.0), load(5.0)
    assert adr.share_factor == 5.0
    for field in (
        "shares_point_in_time",
        "shares_weighted_diluted",
        "shares_weighted_basic",
        "shares_eps_implied",
    ):
        assert getattr(adr, field) == pytest.approx(5.0 * getattr(plain, field)), field


# --- The exit multiple default (DCF_MODEL_REVIEW.md F6) ----------------------


def test_the_historical_multiple_is_measured_on_the_models_ebitda(built):
    """EV ÷ (operating income + D&A), not Sharadar's EBITDA, which carries
    non-operating items the exit multiple is never applied to."""
    dataset = built["dataset"]
    ebitda = float(dataset.annual["ebit"].iloc[-1] + dataset.annual["dna"].iloc[-1])
    dataset.prices = dataset.prices.assign(**{"Enterprise Value": 15.0 * ebitda})

    multiple = dcf_model.historical_ev_ebitda(dataset)
    assert multiple.value == pytest.approx(15.0)
    assert "operating income + D&A" in multiple.source


def test_without_a_daily_ev_the_vendor_ratio_is_the_fallback_and_says_so(built):
    multiple = dcf_model.historical_ev_ebitda(built["dataset"])
    assert multiple.value == pytest.approx(12.0)
    assert "non-operating" in multiple.formula


# --- Sensitivity (DCF_MODEL_REVIEW.md §4.5) ----------------------------------


def _grid(built, **overrides):
    kwargs = dict(
        wacc_result=built["wacc_result"],
        terminal_growth=0.025,
        exit_multiple=12.0,
        drivers=built["drivers"],
        elapsed=built["elapsed"],
        mid_year=True,
        shares=built["dataset"].shares_weighted_diluted,
        share_basis="Diluted weighted-average (FY2025)",
        include_short_term_investments=True,
        operating_cash_floor=0.0,
        non_operating_assets=None,
        base_fcff=built["base_fcff"],
        terminal_tax_rate=built["wacc_result"].marginal_tax_rate,
    )
    kwargs.update(overrides)
    return dcf_model.sensitivity_grid(built["dataset"], built["base"], **kwargs)


@pytest.mark.parametrize("method", ["gordon", "exit_multiple"])
def test_the_sensitivity_grid_is_centred_on_the_headline(built, method):
    grid = _grid(built, headline_method=method)
    middle = len(grid.waccs) // 2
    assert grid.values[middle][middle] == pytest.approx(
        built["result"].bridges[method].value_per_share
    )
    # A higher discount rate is worth less; a higher terminal input, more.
    assert grid.values[0][middle] > grid.values[-1][middle]
    assert grid.values[middle][0] < grid.values[middle][-1]


def test_an_exit_headline_grid_varies_the_multiple_not_terminal_growth(built):
    """DCF_ROUND2_INSTRUCTIONS.md §2.1: growth barely moves an exit value."""
    grid = _grid(built, headline_method="exit_multiple")
    assert grid.column_label == "Exit multiple"
    assert grid.columns == pytest.approx((9.6, 10.8, 12.0, 13.2, 14.4))
    assert grid.column_heading(12.0) == "12.00×"
    gordon = _grid(built, headline_method="gordon")
    assert gordon.column_label == "Terminal growth"
    assert gordon.columns == pytest.approx((0.015, 0.02, 0.025, 0.03, 0.035))


def test_the_sensitivity_grid_leaves_undefined_perpetuities_blank(built):
    grid = _grid(built, headline_method="gordon", terminal_growth=0.08, growth_step=0.01)
    for rate, row in zip(grid.waccs, grid.values):
        for growth, value in zip(grid.columns, row):
            assert (value is None) == (growth >= rate)
