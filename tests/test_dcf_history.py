"""The summary chart's DCF line: the DCF tab's default value, struck at past filings."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from app.data import market_data
from app.data import sharadar_client as sc
from app.data.sharadar import prices as sh_prices
from app.logic.dcf import inputs as dcf_inputs
from app.logic.dcf import model as dcf_model
from app.logic.dcf import validation as val
from app.logic.dcf import wacc as wacc_lib
from app.logic.research import dataset as research_dataset
from app.logic.research import summary as research
from app.provenance import Sourced

from .test_golden import build_model


# ---------------------------------------------------------------------------
# Rates as of a past day
# ---------------------------------------------------------------------------


def _yahoo(rows: list[tuple[date, float]]) -> pd.DataFrame:
    """Shaped like ``yf.Ticker.history`` — a tz-aware index and extra columns."""
    index = pd.DatetimeIndex([pd.Timestamp(d, tz="America/Chicago") for d, _ in rows])
    return pd.DataFrame({"Close": [c for _, c in rows], "Volume": 0}, index=index)


def _patch_yahoo(monkeypatch, frames: dict[str, pd.DataFrame]) -> list[str]:
    calls: list[str] = []

    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol
            calls.append(symbol)

        def history(self, **kwargs):
            return frames.get(self.symbol, pd.DataFrame())

    monkeypatch.setattr(market_data.yf, "Ticker", _Ticker)
    return calls


_TNX = [(date(2026, 5, 19), 4.40), (date(2026, 5, 20), 4.50), (date(2026, 5, 22), 4.60)]


def test_the_ten_year_rate_on_a_past_day_is_that_days_close(monkeypatch):
    _patch_yahoo(monkeypatch, {"^TNX": _yahoo(_TNX)})
    rate = market_data.fetch_10y_treasury_rate_on(date(2026, 5, 20))
    assert rate.value == pytest.approx(0.045)
    assert rate.as_of == date(2026, 5, 20)


def test_a_weekend_reads_the_friday_before(monkeypatch):
    _patch_yahoo(monkeypatch, {"^TNX": _yahoo(_TNX)})
    rate = market_data.fetch_10y_treasury_rate_on(date(2026, 5, 24))
    assert rate.value == pytest.approx(0.046)
    assert rate.as_of == date(2026, 5, 22)


def test_a_stale_close_is_not_passed_off_as_that_days(monkeypatch):
    _patch_yahoo(monkeypatch, {"^TNX": _yahoo(_TNX)})
    rate = market_data.fetch_10y_treasury_rate_on(date(2026, 6, 30))
    assert not rate.ok
    assert "more than 7 days" in rate.detail


def test_a_day_before_the_history_is_unavailable(monkeypatch):
    _patch_yahoo(monkeypatch, {"^TNX": _yahoo(_TNX)})
    assert not market_data.fetch_10y_treasury_rate_on(date(2020, 1, 2)).ok


def test_twenty_past_days_cost_one_request(monkeypatch):
    calls = _patch_yahoo(monkeypatch, {"^TNX": _yahoo(_TNX)})
    for day in (date(2026, 5, 19), date(2026, 5, 20), date(2026, 5, 22)):
        market_data.fetch_10y_treasury_rate_on(day)
    assert calls == ["^TNX"]


def test_a_past_exchange_rate_tries_the_inverted_pair(monkeypatch):
    _patch_yahoo(monkeypatch, {"USDTWD=X": _yahoo([(date(2025, 8, 27), 32.0)])})
    rate = market_data.fetch_fx_rate_on("TWD", date(2025, 8, 27))
    assert rate.value == pytest.approx(1 / 32.0)
    assert "inverted" in rate.source


def test_the_benchmark_is_one_request_whatever_the_window(monkeypatch):
    """A beta at every filing date used to be a range-keyed fetch per date."""
    calls = []

    def rows(ticker, *, table, start=None, end=None):
        calls.append((ticker, start, end))
        dates = pd.bdate_range("2019-01-01", "2026-09-25")
        return pd.DataFrame({"date": dates, "closeadj": range(len(dates))})

    monkeypatch.setattr(sh_prices, "_price_table_for", lambda ticker: "SFP")
    monkeypatch.setattr(sh_prices, "_fetch_price_rows", rows)
    narrow = sc.fetch_index_prices("SPY", "2021-01-01", "2021-12-31")
    wide = sc.fetch_index_prices("SPY", "2019-06-01", "2026-09-25")
    assert calls == [("SPY", None, None)]
    assert narrow.index.min() >= pd.Timestamp("2021-01-01")
    assert narrow.index.max() <= pd.Timestamp("2021-12-31")
    assert len(wide) > len(narrow)


# ---------------------------------------------------------------------------
# The company as it stood on a past date
# ---------------------------------------------------------------------------


def test_a_past_date_keeps_only_what_had_been_filed():
    frame = pd.DataFrame({
        "Fiscal Year": [2024, 2025],
        "Publish Date": pd.to_datetime(["2024-10-30", "2025-10-31"]),
    })
    statements = sc.StatementSet(
        ticker="T", permaticker=1, name="T", currency="USD", template="t",
        dimension="ARY", period="FY", ttm=False, asreported=True, frames={"PL": frame},
    )
    filed = dcf_model._filed_by(statements, date(2025, 10, 15))
    assert list(filed.frame("PL")["Fiscal Year"]) == [2024]


def test_a_past_date_needs_the_as_reported_statements():
    """Under MR, datekey is the report period, and a restatement moves it."""
    with pytest.raises(ValueError, match="as-reported"):
        dcf_model.load_company("TEST", as_of=date(2025, 1, 1))


def test_the_company_on_a_past_date_is_built_from_that_days_filings(
    statements_payload, tickers_response, monkeypatch
):
    build_model(statements_payload, tickers_response, monkeypatch)
    prices = pd.DataFrame({
        "Date": pd.bdate_range("2025-09-01", "2025-11-28"),
        "Last Closing Price": 500.0,        # before a ten-for-one split
        "Split-Adjusted Close": 50.0,       # on the basis of today's share counts
        "Adjusted Closing Price": 50.0,
    })
    monkeypatch.setattr(sc, "fetch_prices", lambda *a, **k: prices)

    before = dcf_model.load_company("TEST", asreported=True, as_of=date(2025, 10, 15))
    assert before.annual.index[-1] == 2024              # FY2025 was filed on 31 October
    assert before.price_date <= date(2025, 10, 15)
    # Share counts are restated for splits even as filed, so the price they
    # multiply is the split-adjusted one, not the traded one.
    assert before.last_price == pytest.approx(50.0)

    after = dcf_model.load_company("TEST", asreported=True, as_of=date(2025, 11, 3))
    assert after.annual.index[-1] == 2025


# ---------------------------------------------------------------------------
# A default the tab would ask the reader for is a gap
# ---------------------------------------------------------------------------

_BETA = dcf_model.BetaOutcome(
    wacc_lib.BetaResult(
        beta=1.1, raw_beta=1.15, r_squared=0.4, observations=60, frequency="monthly",
        blume_adjusted=True, index_ticker="SPY",
    ),
    "stub", None,
)
_RF = Sourced.reported(0.042, "stub")
_ERP = Sourced.reported(0.045, "stub")


@pytest.mark.parametrize(
    "rf,erp,beta,transient",
    [
        (Sourced.unavailable("stub", "down"), _ERP, _BETA, True),
        (_RF, Sourced.unavailable("stub", "down"), _BETA, True),
        (_RF, _ERP, dcf_model.BetaOutcome(None, "stub", "too few months"), False),
    ],
)
def test_an_input_the_tab_would_ask_for_is_a_gap_not_a_placeholder(
    statements_payload, tickers_response, monkeypatch, rf, erp, beta, transient
):
    built = build_model(statements_payload, tickers_response, monkeypatch)
    with pytest.raises(dcf_model.DefaultUnavailable) as raised:
        dcf_inputs.resolve(
            built["dataset"], valuation_date=date.today(), risk_free=rf, erp=erp, beta=beta,
            strict=True,
        )
    assert raised.value.transient is transient


# ---------------------------------------------------------------------------
# The research page's loaders
# ---------------------------------------------------------------------------


def _pl(rows: list[tuple[int, str, str | None]]) -> sc.StatementSet:
    frame = pd.DataFrame({
        "Fiscal Year": [r[0] for r in rows],
        "Fiscal Period": [r[1] for r in rows],
        "Publish Date": pd.to_datetime([r[2] for r in rows]),
    })
    return sc.StatementSet(
        ticker="T", permaticker=1, name="T", currency="USD", template="t",
        dimension="ART", period="Q", ttm=True, asreported=True, frames={"PL": frame},
    )


def test_filing_dates_come_from_the_trailing_statements(monkeypatch):
    def statements(ticker, *, annual=True, ttm=False, asreported=False):
        assert asreported, "only the as-reported dimensions are dated by filing"
        return _pl([(2026, "Q1", "2026-05-20"), (2026, "Q2", None), (2026, "Q3", "2026-11-19")])

    monkeypatch.setattr(research_dataset.sharadar_client, "fetch_statements", statements)
    filings = research_dataset.load_dcf_filings("T")
    # A row with no filing date is left out, never dated by guesswork.
    assert [(f.filed, f.period) for f in filings] == [
        (date(2026, 5, 20), "FY2026 Q1"), (date(2026, 11, 19), "FY2026 Q3"),
    ]


def test_an_annual_only_filer_is_dated_by_its_annual_filings(monkeypatch):
    def statements(ticker, *, annual=True, ttm=False, asreported=False):
        if ttm:
            raise sc.TickerNotFoundError("no TTM", table="SF1", ticker=ticker)
        return _pl([(2025, "FY", "2026-02-20")])

    monkeypatch.setattr(research_dataset.sharadar_client, "fetch_statements", statements)
    assert [f.filed for f in research_dataset.load_dcf_filings("T")] == [date(2026, 2, 20)]


def _valued(value):
    return SimpleNamespace(headline_bridge=SimpleNamespace(value_per_share=value))


def test_a_point_is_the_default_headline(monkeypatch):
    monkeypatch.setattr(dcf_inputs, "default_value_on", lambda t, d=None: _valued(123.0))
    point = research_dataset.load_dcf_point("T", date(2026, 5, 20))
    assert point.value == pytest.approx(123.0)


def test_missing_data_is_a_gap_with_its_reason(monkeypatch):
    def unavailable(t, d=None):
        raise dcf_model.DefaultUnavailable("No beta — too few months")

    monkeypatch.setattr(dcf_inputs, "default_value_on", unavailable)
    point = research_dataset.load_dcf_point("T", date(2026, 5, 20))
    assert point.value is None and "No beta" in point.reason


def test_a_bank_is_a_gap_not_an_error(monkeypatch):
    def bank(t, d=None):
        raise val.UnsupportedSectorError("JPM", "Financial Services", "Financial Services")

    monkeypatch.setattr(dcf_inputs, "default_value_on", bank)
    assert research_dataset.load_dcf_point("JPM").value is None


def test_a_failed_fetch_is_raised_so_no_cache_remembers_it(monkeypatch):
    def outage(t, d=None):
        raise dcf_model.DefaultUnavailable("No risk-free rate", transient=True)

    monkeypatch.setattr(dcf_inputs, "default_value_on", outage)
    with pytest.raises(dcf_model.DefaultUnavailable):
        research_dataset.load_dcf_point("T")


# ---------------------------------------------------------------------------
# Which filings a window needs
# ---------------------------------------------------------------------------

_FILED = [date(2025, 5, 28), date(2025, 8, 27), date(2025, 11, 19), date(2026, 2, 25),
          date(2026, 5, 20), date(2026, 8, 26)]


def test_a_window_values_only_its_own_filings_and_the_one_in_force_at_its_start():
    dates = research.dcf_line_dates(_FILED, date(2025, 9, 26), date(2026, 9, 25))
    assert dates.anchor == date(2025, 8, 27)
    assert dates.inside == (date(2025, 11, 19), date(2026, 2, 25), date(2026, 5, 20),
                            date(2026, 8, 26))
    assert dates.start == date(2025, 9, 26)


def test_a_short_window_after_the_last_filing_needs_one_run():
    dates = research.dcf_line_dates(_FILED, date(2026, 9, 1), date(2026, 9, 25))
    assert dates.anchor == date(2026, 8, 26) and dates.inside == ()


def test_the_line_reaches_back_five_years_at_most():
    filed = [date(y, 3, 1) for y in range(1995, 2027)]
    dates = research.dcf_line_dates(filed, date(1995, 1, 3), date(2026, 9, 25))
    assert dates.start == date(2021, 9, 25)
    assert dates.anchor == date(2021, 3, 1)
    assert len(dates.inside) == 5


# ---------------------------------------------------------------------------
# The chart's step line
# ---------------------------------------------------------------------------


def test_the_line_steps_anchor_on_the_windows_first_session_and_end_on_today():
    """The anchor is valued on the filing before the window but drawn where the
    window opens — a weekend start would otherwise draw it nowhere."""
    periods = {d: f"P{i}" for i, d in enumerate(_FILED)}
    # 2025-09-27 is a Saturday; the window's first session is the Monday.
    sessions = pd.bdate_range("2025-09-29", "2026-09-25")
    dates, steps = research.dcf_line_steps(periods, sessions)

    assert dates.start == date(2025, 9, 29)
    assert steps[0].as_of == date(2025, 8, 27)
    assert steps[0].drawn_at == date(2025, 9, 29)
    assert steps[0].label == "P1 filed 2025-08-27"
    assert [s.as_of for s in steps[1:-1]] == list(dates.inside)
    assert all(s.drawn_at == s.as_of for s in steps[1:-1])
    assert steps[-1].as_of is None and steps[-1].label == "today"
    assert steps[-1].drawn_at == date(2026, 9, 25)


def test_a_window_before_any_filing_is_today_alone():
    _, steps = research.dcf_line_steps({}, pd.bdate_range("2026-09-01", "2026-09-25"))
    assert [(s.as_of, s.label) for s in steps] == [(None, "today")]
