"""Stock research statistics: window slicing, ranges, and the dividend zero-fill guard.

Everything here runs against constructed frames — nothing touches the network,
and nothing needs a Streamlit runtime, because the module under test is pure.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app import formatting
from app.data import sharadar_client as sc
from app.logic.research import dataset as research_dataset
from app.logic.research import summary
from app.provenance import Origin

TODAY = date(2026, 8, 5)


def _ohlcv(start: str = "2019-01-01", end: str = "2026-08-05") -> pd.DataFrame:
    """A well-behaved daily bar series on business days."""
    index = pd.bdate_range(start, end, name="Date")
    base = np.linspace(100.0, 300.0, len(index))
    return pd.DataFrame(
        {
            "Open": base,
            "High": base + 2.0,
            "Low": base - 2.0,
            "Close": base + 1.0,
            "Adj Close": base + 1.0,
            "Volume": np.full(len(index), 1_000_000.0),
        },
        index=index,
    )


def _actions(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """``(action, date, value)`` triples in the client's output shape."""
    if not rows:
        return pd.DataFrame(columns=["date", "action", "value"])
    return pd.DataFrame(
        {
            "action": [r[0] for r in rows],
            "date": pd.to_datetime([r[1] for r in rows]),
            "value": [r[2] for r in rows],
        }
    ).sort_values("date").reset_index(drop=True)


def _ttm_pl(**indicators) -> pd.DataFrame:
    """A one-row TTM income statement keyed by Sharadar's published captions."""
    row = {sc.SF1_CAPTIONS[key]: value for key, value in indicators.items()}
    row["Report Date"] = pd.Timestamp("2026-06-30")
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Window slicing
# ---------------------------------------------------------------------------


def test_five_day_window_counts_sessions_not_calendar_days():
    """5D on a quote page means five bars, not five days minus a weekend."""
    assert len(summary.slice_period(_ohlcv(), "5D", TODAY)) == 5


def test_all_returns_the_untouched_frame():
    frame = _ohlcv()
    assert summary.slice_period(frame, "ALL", TODAY) is frame


def test_ytd_starts_at_january_first_of_the_current_year():
    window = summary.slice_period(_ohlcv(), "YTD", TODAY)
    assert window.index.min() >= pd.Timestamp("2026-01-01")
    assert window.index.min() < pd.Timestamp("2026-01-06")


def test_ytd_is_anchored_to_the_calendar_not_to_the_data():
    """The other windows follow the last bar; YTD must not, or it stops meaning YTD."""
    stale = _ohlcv(end="2026-03-31")
    window = summary.slice_period(stale, "YTD", TODAY)
    assert window.index.min() >= pd.Timestamp("2026-01-01")


@pytest.mark.parametrize(
    "period,years",
    [("1M", 1 / 12), ("3M", 0.25), ("6M", 0.5), ("1Y", 1.0), ("5Y", 5.0)],
)
def test_calendar_windows_span_roughly_their_label(period, years):
    window = summary.slice_period(_ohlcv(), period, TODAY)
    span_years = (window.index.max() - window.index.min()).days / 365.25
    assert span_years == pytest.approx(years, abs=0.06)


def test_a_window_longer_than_the_history_returns_everything():
    short = _ohlcv(start="2026-06-01")
    assert len(summary.slice_period(short, "5Y", TODAY)) == len(short)


def test_slicing_an_empty_frame_is_empty_not_an_error():
    for period in summary.PERIODS:
        assert summary.slice_period(pd.DataFrame(), period, TODAY).empty


def test_period_change_measures_first_to_last_close():
    frame = pd.DataFrame(
        {"Close": [100.0, 110.0]}, index=pd.to_datetime(["2026-08-04", "2026-08-05"])
    )
    change = summary.period_change(frame)
    assert change.ok
    assert change.value == pytest.approx(0.10)
    assert change.origin is Origin.DERIVED


def test_period_change_of_a_single_session_is_unavailable_not_zero():
    frame = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2026-08-05"]))
    change = summary.period_change(frame)
    assert not change.ok
    assert change.origin is Origin.UNAVAILABLE


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------


def test_days_range_reads_the_latest_bar():
    tile = summary.days_range(_ohlcv(), "USD")
    assert tile.ok
    assert tile.origin is Origin.REPORTED
    # The unit is stated once, on the left.
    assert tile.display.count("USD") == 1


def test_fifty_two_week_range_excludes_older_bars():
    """A spike thirteen months back must not widen the 52-week range."""
    frame = _ohlcv(start="2024-01-01")
    frame.loc[frame.index[0], "High"] = 10_000.0
    tile = summary.fifty_two_week_range(frame, "USD")
    assert "10,000" not in tile.display
    assert tile.origin is Origin.DERIVED


def test_ranges_on_an_empty_frame_report_unavailable():
    for tile in (
        summary.days_range(pd.DataFrame(), "USD"),
        summary.fifty_two_week_range(pd.DataFrame(), "USD"),
    ):
        assert not tile.ok
        assert tile.origin is Origin.UNAVAILABLE
        assert tile.detail


# ---------------------------------------------------------------------------
# The dividend zero-fill guard
# ---------------------------------------------------------------------------
# Sharadar writes 0 both for "pays nothing" and for "absent from the filing".
# ACTIONS is what tells the two apart.


def test_zero_dividend_with_no_actions_is_a_genuine_nil():
    tile = summary._dividend_tile(_ttm_pl(dps=0.0), _actions([]), 100.0, "USD", TODAY)
    assert tile.ok
    assert "no dividends" in tile.detail.lower()


def test_zero_dividend_contradicted_by_actions_is_unavailable_not_zero():
    """The one case where a plausible-looking 0 must not be shown."""
    paid = _actions([("dividend", "2026-03-09", 0.21), ("dividend", "2026-06-08", 0.22)])
    tile = summary._dividend_tile(_ttm_pl(dps=0.0), paid, 100.0, "USD", TODAY)
    assert not tile.ok
    assert tile.origin is Origin.UNAVAILABLE
    assert "ACTIONS" in tile.detail


def test_the_dividend_cell_carries_its_yield_in_brackets():
    """SF1 rounds ``divyield`` to three decimals; 1.05/311 must not become 0.4%."""
    paid = _actions([("dividend", "2026-05-11", 0.27)])
    tile = summary._dividend_tile(_ttm_pl(dps=1.05), paid, 311.0, "USD", TODAY)
    assert tile.display == "USD 1.05 (0.34%)"
    # Reported dividend, derived yield — the weaker origin governs the cell.
    assert tile.origin is Origin.DERIVED


def test_a_dividend_without_a_price_shows_no_yield_rather_than_a_wrong_one():
    tile = summary._dividend_tile(_ttm_pl(dps=1.05), _actions([]), None, "USD", TODAY)
    assert tile.display == "USD 1.05"
    assert "%" not in tile.display


def test_a_missing_dps_line_is_unavailable_rather_than_nil():
    """Absent is not the same as zero, and neither is shown as the other."""
    tile = summary._dividend_tile(pd.DataFrame(), _actions([]), 100.0, "USD", TODAY)
    assert not tile.ok
    assert tile.origin is Origin.UNAVAILABLE


# ---------------------------------------------------------------------------
# Ex-dividend date
# ---------------------------------------------------------------------------


def test_ex_dividend_date_is_the_latest_dividend_action():
    paid = _actions(
        [("dividend", "2026-03-09", 0.21), ("split", "2026-04-01", 4.0),
         ("dividend", "2026-06-08", 0.22)]
    )
    tile = summary._ex_dividend_tile(paid, "USD", TODAY)
    assert tile.display == "2026-06-08"
    assert "0.22" in tile.detail


def test_a_future_ex_date_is_labelled_upcoming():
    """Sharadar carries declared ex-dates ahead of time."""
    tile = summary._ex_dividend_tile(_actions([("dividend", "2026-09-08", 0.22)]), "USD", TODAY)
    assert "upcoming" in tile.detail


def test_a_company_that_never_paid_has_no_ex_date():
    tile = summary._ex_dividend_tile(_actions([("split", "2020-08-31", 4.0)]), "USD", TODAY)
    assert not tile.ok
    assert tile.origin is Origin.UNAVAILABLE


# ---------------------------------------------------------------------------
# Reporting calendar
# ---------------------------------------------------------------------------


def _quarterly_ar(periods: list[tuple[str, str]]) -> pd.DataFrame:
    """``(report period, publish date)`` pairs, oldest first."""
    return pd.DataFrame(
        {
            "Report Date": pd.to_datetime([p[0] for p in periods]),
            "Publish Date": pd.to_datetime([p[1] for p in periods]),
            "Fiscal Year": [pd.Timestamp(p[0]).year for p in periods],
            "Fiscal Period": ["Q2"] * len(periods),
        }
    )


def test_next_report_adds_the_median_filing_lag_to_the_next_period_end():
    frame = _quarterly_ar(
        [
            ("2025-09-30", "2025-10-31"),
            ("2025-12-31", "2026-01-31"),
            ("2026-03-31", "2026-04-30"),
            ("2026-06-30", "2026-07-31"),
        ]
    )
    tile = summary._next_report_tile(frame, TODAY)
    # 2026-06-30 + 3 months = 2026-09-30, + 31 days median lag.
    assert tile.display == "2026-10-31"
    assert tile.origin is Origin.ESTIMATE


def test_the_last_real_filing_date_rides_along_in_the_detail():
    """It is what the estimate is built from, so it stays reachable."""
    frame = _quarterly_ar([("2026-06-30", "2026-07-31")])
    assert "Last filed 2026-07-31" in summary._next_report_tile(frame, TODAY).detail


def test_next_report_is_never_presented_as_an_announced_earnings_date():
    frame = _quarterly_ar([("2026-06-30", "2026-07-31")])
    tile = summary._next_report_tile(frame, TODAY)
    assert tile.origin is Origin.ESTIMATE
    assert "no earnings calendar" in tile.detail


def test_an_estimate_that_has_already_passed_says_so():
    """Stale vendor data must not read as a confident future date."""
    frame = _quarterly_ar([("2026-03-31", "2026-05-01")])
    tile = summary._next_report_tile(frame, TODAY)
    assert tile.display == "2026-07-31"
    assert "has passed" in tile.detail


def test_no_filing_history_reports_unavailable():
    tile = summary._next_report_tile(pd.DataFrame(), TODAY)
    assert not tile.ok
    assert tile.origin is Origin.UNAVAILABLE


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------


def test_beta_of_a_series_that_tracks_the_index_is_one():
    index = pd.bdate_range("2020-01-01", "2026-08-05")
    rng = np.random.default_rng(0)
    market = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(index))), index=index)
    tile = summary._beta_tile(market.copy(), market)
    assert tile.ok
    assert float(tile.display) == pytest.approx(1.0, abs=0.01)
    assert tile.origin is Origin.DERIVED


def test_too_short_a_history_reports_unavailable_rather_than_a_noisy_beta():
    index = pd.bdate_range("2026-01-01", "2026-08-05")
    series = pd.Series(np.linspace(100, 120, len(index)), index=index)
    tile = summary._beta_tile(series, series * 1.01)
    assert not tile.ok
    assert tile.origin is Origin.UNAVAILABLE


def test_beta_as_of_comes_from_the_price_series_not_the_resample_label():
    """Resampling to month-end labels the last bucket in the future."""
    index = pd.bdate_range("2020-01-01", "2026-08-05")
    rng = np.random.default_rng(1)
    market = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(index))), index=index)
    tile = summary._beta_tile(market.copy(), market)
    assert tile.as_of == index.max().date()


# ---------------------------------------------------------------------------
# The grid as a whole
# ---------------------------------------------------------------------------


def _build(**overrides):
    base = dict(
        ohlcv=_ohlcv(),
        daily=pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-08-05")],
                "marketcap": [4.5e12],
                "ev": [4.6e12],
                "pe": [35.2],
                "pb": [42.2],
                "ps": [9.7],
                "evebitda": [27.1],
            }
        ),
        ttm_pl=_ttm_pl(dps=1.05, epsdil=8.71),
        quarterly_ar=_quarterly_ar([("2026-06-30", "2026-07-31")]),
        actions=_actions([("dividend", "2026-05-11", 0.27)]),
        stock_closes=pd.Series(dtype="float64"),
        index_closes=pd.Series(dtype="float64"),
        currency="USD",
        today=TODAY,
    )
    base.update(overrides)
    return summary.build_key_stats(**base)


def test_the_layout_and_the_tiles_agree_exactly():
    """A renamed tile must fail here rather than leave a hole in the table.

    The table is laid out from ``STAT_COLUMNS`` by label lookup, so the two
    lists drifting apart is the failure this guards: a label with no tile
    renders "not reported" forever, and a tile with no label vanishes.
    """
    laid_out = {label for column in summary.STAT_COLUMNS for label in column}
    produced = {tile.label for tile in _build()}
    assert laid_out | {summary.HEADLINE_STAT} == produced


def test_the_headline_stat_is_not_repeated_in_the_table():
    """Previous Close sits above the chart; showing it twice wastes a cell."""
    laid_out = {label for column in summary.STAT_COLUMNS for label in column}
    assert summary.HEADLINE_STAT not in laid_out


def test_the_table_is_four_columns_of_four():
    assert len(summary.STAT_COLUMNS) == 4
    assert all(len(column) == 4 for column in summary.STAT_COLUMNS)


def test_every_tile_carries_an_origin():
    assert all(isinstance(tile.origin, Origin) for tile in _build())


def test_every_tile_has_its_own_label():
    """The grid is laid out by label, so two tiles sharing one would hide one."""
    labels = [tile.label for tile in _build()]
    assert len(set(labels)) == len(labels)


def test_missing_feeds_produce_unavailable_tiles_rather_than_zeros():
    """A dropped feed must be visible, not silently absent."""
    tiles = _build(daily=pd.DataFrame(), ttm_pl=pd.DataFrame(), actions=_actions([]))
    by_label = {tile.label: tile for tile in tiles}
    for label in ("Market Cap", "P/E", "EPS (TTM, diluted)", "Ex-Dividend Date"):
        assert not by_label[label].ok, label
        assert by_label[label].origin is Origin.UNAVAILABLE, label
        assert by_label[label].detail, label
    # The price-derived tiles still work — one missing feed does not cascade.
    assert by_label[summary.HEADLINE_STAT].ok


def test_market_cap_is_formatted_from_raw_units():
    """DAILY publishes millions; the client rescales, so 4.5e12 must read as trillions."""
    tile = next(t for t in _build() if t.label == "Market Cap")
    assert tile.display == "USD 4.50T"


def test_an_empty_price_history_does_not_raise():
    tiles = _build(ohlcv=pd.DataFrame())
    assert all(isinstance(tile, summary.StatTile) for tile in tiles)
    assert not next(t for t in tiles if t.label == "Previous Close").ok


# ---------------------------------------------------------------------------
# StatTile
# ---------------------------------------------------------------------------


def test_an_unavailable_sourced_projects_to_a_tile_with_no_display():
    from app.provenance import Sourced

    tile = summary.StatTile.from_sourced(
        "X", Sourced.unavailable("Sharadar", "nothing there"), formatting.format_ratio
    )
    assert not tile.ok
    assert tile.display is None
    assert "nothing there" in tile.describe()


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_formatters_never_invent_a_number(value):
    assert formatting.format_money(value) == "not reported"
    assert formatting.format_count(value) == "not reported"
    assert formatting.format_ratio(value) == "not reported"
    assert formatting.format_pct(value) == "not reported"


def test_a_missing_date_renders_as_a_dash_not_a_fabricated_one():
    assert formatting.format_date(None, missing="—") == "—"
    assert formatting.format_date(pd.NaT, missing="—") == "—"
    assert formatting.format_date(pd.Timestamp("2025-09-30"), missing="—") == "2025-09-30"


# ---------------------------------------------------------------------------
# Orchestration — the statement sets the financials tab reads
# ---------------------------------------------------------------------------


def _statement_set(dimension: str) -> sc.StatementSet:
    """A StatementSet whose frames are identifiable by their dimension."""
    return sc.StatementSet(
        ticker="TEST", permaticker=999, name="TESTCO INC", currency="USD",
        template="Technology", dimension=dimension,
        period="FY" if dimension.endswith("Y") else "Q1,Q2,Q3,Q4",
        ttm=dimension.endswith("T"), asreported=dimension.startswith("AR"),
        frames={"PL": pd.DataFrame({"Fiscal Year": [2025], "Dimension": [dimension]})},
    )


@pytest.fixture
def stub_fetches(monkeypatch):
    """Every network call `load_research` makes, replaced by a local stub."""
    monkeypatch.setattr(research_dataset.sharadar_client, "fetch_ticker_meta",
                        lambda ticker: {"name": "TESTCO INC", "currency": "USD"})
    monkeypatch.setattr(research_dataset.sharadar_client, "fetch_ohlcv",
                        lambda *a, **k: _ohlcv("2025-01-01", "2026-08-05"))
    monkeypatch.setattr(research_dataset.sharadar_client, "fetch_daily_metrics",
                        lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(research_dataset.sharadar_client, "fetch_actions",
                        lambda *a, **k: _actions([]))
    monkeypatch.setattr(research_dataset.sharadar_client, "fetch_index_prices",
                        lambda *a, **k: pd.Series(dtype="float64"))
    monkeypatch.setattr(research_dataset.sharadar_client,
                        "fetch_insider_transactions",
                        lambda *a, **k: pd.DataFrame())
    for name in ("fetch_institutional_summary", "fetch_institutional_holders",
                 "fetch_investor_directory"):
        monkeypatch.setattr(research_dataset.sharadar_client, name,
                            lambda *a, **k: pd.DataFrame())

    def statements(ticker, *, annual=True, ttm=False, asreported=False):
        return _statement_set(
            sc.dimension_for(annual=annual, ttm=ttm, asreported=asreported)
        )

    monkeypatch.setattr(research_dataset.sharadar_client, "fetch_statements", statements)
    return monkeypatch


def test_load_research_fetches_only_what_the_summary_tab_draws(stub_fetches):
    """A search lands on 📊 Summary, so that is all it pays for. Every other tab
    fetches on first open, and a stub that raises proves nothing reached for it."""
    def untouchable(*args, **kwargs):
        raise AssertionError("fetched by the search, not by the tab that needs it")

    for name in ("fetch_insider_transactions", "fetch_institutional_summary",
                 "fetch_universe_daily", "fetch_universe_fundamentals",
                 "fetch_ticker_directory"):
        stub_fetches.setattr(research_dataset.sharadar_client, name, untouchable)

    def statements(ticker, *, annual=True, ttm=False, asreported=False):
        dimension = sc.dimension_for(annual=annual, ttm=ttm, asreported=asreported)
        assert dimension in {"MRT", "ARQ"}, dimension
        return _statement_set(dimension)

    stub_fetches.setattr(research_dataset.sharadar_client, "fetch_statements", statements)
    dataset = research_dataset.load_research("TEST", today=TODAY)

    assert dataset.statements_ttm["PL"]["Dimension"].iloc[0] == "MRT"
    # The as-reported quarterly income statement is still a bare frame; only
    # the next-report tile wants it.
    assert dataset.quarterly_ar["Dimension"].iloc[0] == "ARQ"
    for moved in ("statements_annual", "statements_quarterly", "insiders",
                  "peer_universe"):
        assert not hasattr(dataset, moved), moved


def test_ttm_pl_still_resolves_through_the_property(stub_fetches):
    """`build_key_stats` and its tests address `ttm_pl`; the move must be invisible."""
    dataset = research_dataset.load_research("TEST", today=TODAY)
    assert dataset.ttm_pl["Dimension"].iloc[0] == "MRT"


def test_a_missing_price_history_still_blocks_the_search(stub_fetches):
    """Fetched concurrently now, but still the one fetch the page cannot do without."""
    stub_fetches.setattr(research_dataset.sharadar_client, "fetch_ohlcv",
                         lambda *a, **k: pd.DataFrame())
    with pytest.raises(sc.TickerNotFoundError):
        research_dataset.load_research("TEST", today=TODAY)


def test_load_financials_keeps_both_dimensions_the_tab_needs(stub_fetches):
    data = research_dataset.load_financials("TEST")

    assert data.annual["PL"]["Dimension"].iloc[0] == "MRY"
    assert data.quarterly["PL"]["Dimension"].iloc[0] == "MRQ"


def test_a_dimension_that_fails_leaves_an_empty_set_rather_than_blocking(stub_fetches):
    """The `_tolerant` contract: one dimension missing leaves the other showing."""
    def statements(ticker, *, annual=True, ttm=False, asreported=False):
        dimension = sc.dimension_for(annual=annual, ttm=ttm, asreported=asreported)
        if dimension == "MRY":
            raise sc.SharadarError("no fundamentals", table="SF1", ticker=ticker)
        return _statement_set(dimension)

    stub_fetches.setattr(research_dataset.sharadar_client, "fetch_statements", statements)
    data = research_dataset.load_financials("TEST")

    assert data.annual == {}
    assert data.quarterly["PL"]["Dimension"].iloc[0] == "MRQ"
    # And the search itself never depended on either.
    assert research_dataset.load_research("TEST", today=TODAY).stats


def test_a_failed_insider_fetch_is_raised_not_passed_off_as_no_filings(stub_fetches):
    """An empty frame means nobody filed. Now that the tab loads its own, a failed
    request reaches ``load_tab`` and is shown as one."""
    def failing(*args, **kwargs):
        raise sc.SharadarError("timed out", table="SF2", ticker="TEST")

    stub_fetches.setattr(research_dataset.sharadar_client,
                         "fetch_insider_transactions", failing)
    with pytest.raises(sc.SharadarError):
        research_dataset.load_insiders("TEST", today=TODAY)


def test_the_insider_window_is_five_years_of_filing_dates(stub_fetches):
    seen = []
    stub_fetches.setattr(research_dataset.sharadar_client,
                         "fetch_insider_transactions",
                         lambda ticker, since: seen.append(since) or pd.DataFrame())
    research_dataset.load_insiders("TEST", today=TODAY)
    assert seen == ["2021-08-05"]


def test_the_peer_universe_is_anchored_on_the_last_market_session(stub_fetches):
    daily = pd.DataFrame({"date": pd.to_datetime(["2026-07-30", "2026-07-31"])})
    stub_fetches.setattr(research_dataset.sharadar_client, "fetch_daily_metrics",
                         lambda *a, **k: daily)
    dataset = research_dataset.load_research("TEST", today=TODAY)
    # TODAY is a Wednesday; the company's own last row is the Friday before.
    assert dataset.market_session == date(2026, 7, 31)


def test_no_market_session_means_no_peer_fetch(stub_fetches):
    def untouchable(*args, **kwargs):
        raise AssertionError("a universe fetch without a session to anchor it")

    stub_fetches.setattr(research_dataset.sharadar_client, "fetch_universe_daily",
                         untouchable)
    assert research_dataset.load_peers(None, today=TODAY).empty


def test_the_holdings_payload_carries_the_share_count_it_divides_by(stub_fetches):
    data = research_dataset.load_holdings("TEST", today=TODAY)
    assert data.derived.empty  # the stub's statements carry only a PL frame

    def statements(ticker, *, annual=True, ttm=False, asreported=False):
        dimension = sc.dimension_for(annual=annual, ttm=ttm, asreported=asreported)
        derived = pd.DataFrame({"Dimension": [dimension]})
        return sc.StatementSet(**{**_statement_set(dimension).__dict__,
                                  "frames": {"DERIVED": derived}})

    stub_fetches.setattr(research_dataset.sharadar_client, "fetch_statements", statements)
    data = research_dataset.load_holdings("TEST", today=TODAY)
    assert data.derived["Dimension"].iloc[0] == "MRQ"


# ---------------------------------------------------------------------------
# Requests per search — every table asked for once, however many tabs open
# ---------------------------------------------------------------------------


@pytest.fixture
def counted_requests(monkeypatch):
    """Routes ``transport._get`` by table and records every request made."""
    from app.data.sharadar import transport as sh_transport
    from tests import conftest

    prices = {
        "datatable": {
            "data": [["TEST", d, 100.0, 102.0, 98.0, 101.0, 1e6, 101.0, 101.0]
                     for d in ("2026-08-03", "2026-08-04", "2026-08-05")],
            "columns": [{"name": n, "type": "text"} for n in (
                "ticker", "date", "open", "high", "low", "close", "volume",
                "closeadj", "closeunadj")],
        },
        "meta": {"next_cursor_id": None},
    }
    empty = {"datatable": {"data": [], "columns": []}, "meta": {"next_cursor_id": None}}
    seen: list[tuple[str, dict]] = []

    def route(table, query, **kwargs):
        table = table.rsplit("/", 1)[-1]
        seen.append((table, dict(query)))
        if table == "TICKERS":
            return conftest.tickers_payload()
        if table == "SF1":
            return conftest.sf1_payload()
        if table in ("SEP", "SFP"):
            return prices
        return empty

    monkeypatch.setattr(sh_transport, "_get", route)
    return seen


def test_the_reference_record_is_fetched_once_per_company(counted_requests):
    """Every statement fetch reads the company's TICKERS record. It used to read
    it uncached, so a search paid for it once per SF1 dimension."""
    research_dataset.load_research("TEST", today=TODAY)
    research_dataset.load_financials("TEST")

    records = [q for t, q in counted_requests if t == "TICKERS" and q.get("table") == "SF1"]
    assert len(records) == 1


def test_each_sf1_dimension_is_fetched_once_across_the_tabs(counted_requests):
    research_dataset.load_research("TEST", today=TODAY)
    research_dataset.load_financials("TEST")
    research_dataset.load_holdings("TEST", today=TODAY)

    dimensions = [q["dimension"] for t, q in counted_requests if t == "SF1"]
    assert sorted(dimensions) == ["ARQ", "MRQ", "MRT", "MRY"]


def test_the_dcf_prices_reuse_the_searchs_price_history(counted_requests):
    """``fetch_ohlcv`` and ``fetch_prices`` slice one cached full history."""
    research_dataset.load_research("TEST", today=TODAY)
    sc.fetch_prices("TEST", start="2021-01-01")

    # The benchmark has a price request of its own; only the company's count.
    company = [t for t, q in counted_requests if q.get("ticker") == "TEST"]
    assert company.count("SEP") == 1
    assert company.count("DAILY") == 1
