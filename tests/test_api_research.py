"""The 🔎 Stock Research routes, over the fixture companies the tab tests use.

Each loader in ``app.logic.research.dataset`` is replaced by the frames its tab's
own tests build — the insider register whose naive total has the wrong sign, the
13F register with a split and Vanguard's reorganisation in it, the eleven-ticker
universe — so the routes run the real logic and these tests check only what the
API adds: the shapes, the strict JSON, and the cache in front of the DCF line.
The JSON is parsed with NaN refused, so a gap that slipped through unconverted
fails here rather than reaching the frontend as an invalid document.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import main
from app.api import research as api
from app.data import sharadar_client as sc
from app.data.sharadar import statements as sh_statements
from app.logic.dcf import model as dcf_model
from app.logic.research import dataset as research
from app.logic.research import summary
from app.logic.research.summary import StatTile
from app.provenance import Origin
from tests.conftest import _QUARTERLY_ROWS, _TTM_ROW, sf1_payload, sf3_splits

TODAY = date(2026, 8, 5)


def _strict(response) -> dict:
    assert response.status_code == 200, response.text

    def refuse(constant):
        raise ValueError(f"non-finite number {constant} in the response")

    return json.loads(response.text, parse_constant=refuse)


def _frames(rows=None) -> dict:
    """The four statement frames, as ``fetch_statements`` assembles them."""
    parsed = sc.parse_datatable(sf1_payload(rows))
    built = sh_statements._sf1_frames(parsed, source_link="https://example.test", restated=True)
    return {name: sh_statements._order_statement_rows(frame) for name, frame in built.items()}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("EQUITI_AUTH", "off")
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.setattr(api, "_today", lambda: TODAY)
    api._dcf_point_cached.cache_clear()
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def snapshot(monkeypatch) -> research.ResearchDataset:
    """The search's snapshot for TEST: five years of prices with one gap in them."""
    index = pd.bdate_range("2021-01-04", TODAY, name="Date")
    close = pd.Series(100.0 + 0.01 * pd.RangeIndex(len(index)), index=index)
    ohlcv = pd.DataFrame(
        {
            "Open": close - 0.5, "High": close + 1.0, "Low": close - 1.0,
            "Close": close, "Adj Close": close, "Volume": 1e6,
        },
        index=index,
    )
    ohlcv.iloc[-3, ohlcv.columns.get_loc("Volume")] = float("nan")
    daily = pd.DataFrame(
        {
            "date": pd.bdate_range(end=TODAY, periods=400),
            "pe": 20.0, "ps": 2.0, "pb": 4.0, "evebit": 15.0, "evebitda": 12.0,
            "ev": 2.2e9, "marketcap": 2.0e9,
        }
    )
    stats = [
        StatTile("Previous Close", "101.23", Origin.REPORTED, "Sharadar SEP", as_of=TODAY),
        StatTile.unavailable("Forward P/E", "Sharadar", "No consensus estimates."),
    ]
    dataset = research.ResearchDataset(
        ticker="TEST",
        profile={
            "name": "TESTCO INC", "exchange": "NASDAQ", "sector": "Technology",
            "industry": "Consumer Electronics", "currency": "USD",
            "secfilings": "https://example.test/sec", "companysite": "",
        },
        ohlcv=ohlcv,
        daily=daily,
        actions=sf3_splits(),
        statements_ttm=_frames([list(_TTM_ROW)]),
        quarterly_ar=pd.DataFrame(),
        stats=stats,
        currency="USD",
        loaded_at=TODAY,
    )
    calls = []

    def load(ticker, *, today=None):
        calls.append((ticker, today))
        if ticker != "TEST":
            raise sc.TickerNotFoundError(f"No Sharadar record for {ticker}.", table="TICKERS", ticker=ticker)
        return dataset

    monkeypatch.setattr(api.research, "load_research", load)
    dataset_calls = calls
    return dataset, dataset_calls


# ---------------------------------------------------------------------------
# Profile and Summary
# ---------------------------------------------------------------------------


def test_the_profile_names_the_company_and_its_last_close(client, snapshot):
    body = _strict(client.get("/api/research/test"))
    assert body["ticker"] == "TEST" and body["name"] == "TESTCO INC"
    assert body["industry"] == "Consumer Electronics"
    assert body["sec_filings"] == "https://example.test/sec"
    # An empty vendor string is no link, not a link to nowhere.
    assert body["website"] is None
    assert body["last_session"] == TODAY.isoformat()
    assert body["loaded_at"] == TODAY.isoformat()
    # Every loader is keyed on the same day, upper-cased once.
    assert snapshot[1] == [("TEST", TODAY)]


def test_an_unknown_ticker_is_a_404_naming_it(client, snapshot):
    response = client.get("/api/research/NOPE")
    assert response.status_code == 404
    assert "NOPE" in response.json()["detail"]


def test_the_summary_is_a_slice_of_the_snapshot_and_keeps_its_gaps(client, snapshot):
    body = _strict(client.get("/api/research/TEST/summary", params={"period": "1M"}))
    candles = body["candles"]
    window = summary.slice_period(snapshot[0].ohlcv, "1M", TODAY)
    assert len(candles["sessions"]) == len(window)
    assert candles["sessions"][-1] == TODAY.isoformat()
    # The missing volume stays missing rather than reading as zero.
    assert candles["volume"][-3] is None and candles["volume"][-1] == 1e6
    assert body["change"] == pytest.approx(summary.period_change(window).value)
    assert body["periods"] == list(summary.PERIODS)


def test_the_statistics_carry_their_provenance_and_a_missing_one_says_why(client, snapshot):
    body = _strict(client.get("/api/research/TEST/summary"))
    assert body["grid"] == [list(column) for column in summary.STAT_COLUMNS]
    by_label = {s["label"]: s for s in body["stats"]}
    assert by_label["Previous Close"]["display"] == "101.23"
    assert by_label["Previous Close"]["provenance"]["origin"] == "reported"
    missing = by_label["Forward P/E"]
    assert missing["display"] is None
    assert missing["provenance"]["origin"] == "unavailable"
    assert missing["provenance"]["detail"] == "No consensus estimates."


def test_an_unknown_period_is_refused(client, snapshot):
    assert client.get("/api/research/TEST/summary", params={"period": "2W"}).status_code == 422


# ---------------------------------------------------------------------------
# The DCF line
# ---------------------------------------------------------------------------


FILED = [date(2025, 5, 28), date(2025, 8, 27), date(2025, 11, 19), date(2026, 2, 25),
         date(2026, 5, 20)]


@pytest.fixture
def points(monkeypatch):
    """Filings for TEST and a point loader that counts its calls."""
    values: dict[date | None, float | None] = {d: 90.0 + i for i, d in enumerate(FILED)}
    values[None] = 120.0
    state = {"values": values, "calls": [], "fail": set()}

    monkeypatch.setattr(
        api.research, "load_dcf_filings",
        lambda ticker: tuple(research.DcfFiling(d, f"P{i}") for i, d in enumerate(FILED)),
    )

    def load(ticker, as_of=None):
        state["calls"].append(as_of)
        if as_of in state["fail"]:
            raise dcf_model.DefaultUnavailable("No risk-free rate", transient=True)
        value = state["values"].get(as_of)
        return research.DcfPoint(as_of, value, "" if value is not None else "Too little filed.")

    monkeypatch.setattr(api.research, "load_dcf_point", load)
    return state


def test_the_line_steps_through_each_filing_and_ends_on_today(client, snapshot, points):
    body = _strict(client.get("/api/research/TEST/dcf-line", params={"period": "1Y"}))
    steps = body["steps"]
    # The anchor (the filing in force when the window opens) draws at its start.
    assert steps[0]["label"] == "P0 filed 2025-05-28" and steps[0]["value"] == 90.0
    assert steps[0]["session"] == body["start"] == "2025-08-05"
    assert [s["session"] for s in steps[1:-1]] == [
        "2025-08-27", "2025-11-19", "2026-02-25", "2026-05-20"
    ]
    assert steps[-1] == {"session": TODAY.isoformat(), "value": 120.0, "label": "today", "reason": ""}
    assert body["max_years"] == summary.DCF_LINE_MAX_YEARS and body["reason"] == ""


def test_a_filing_that_cannot_be_valued_is_a_gap_with_its_reason(client, snapshot, points):
    points["values"][FILED[2]] = None
    steps = _strict(client.get("/api/research/TEST/dcf-line"))["steps"]
    gap = next(s for s in steps if s["session"] == FILED[2].isoformat())
    assert gap["value"] is None and gap["reason"] == "Too little filed."


def test_each_point_is_valued_once_and_a_wider_window_pays_only_for_its_own(client, snapshot, points):
    client.get("/api/research/TEST/dcf-line", params={"period": "6M"})
    first = list(points["calls"])
    client.get("/api/research/TEST/dcf-line", params={"period": "6M"})
    assert points["calls"] == first
    client.get("/api/research/TEST/dcf-line", params={"period": "1Y"})
    added = points["calls"][len(first):]
    assert added and not set(added) & set(first)


def test_an_outage_is_a_gap_today_and_is_retried_next_time(client, snapshot, points):
    points["fail"].add(FILED[3])
    steps = _strict(client.get("/api/research/TEST/dcf-line"))["steps"]
    gap = next(s for s in steps if s["session"] == FILED[3].isoformat())
    assert gap["value"] is None and "Retried" in gap["reason"]

    points["fail"].clear()
    steps = _strict(client.get("/api/research/TEST/dcf-line"))["steps"]
    assert next(s for s in steps if s["session"] == FILED[3].isoformat())["value"] == 93.0


def test_no_value_anywhere_is_no_line_with_todays_reason(client, snapshot, points):
    points["values"] = {}
    body = _strict(client.get("/api/research/TEST/dcf-line"))
    assert body["steps"] == [] and body["reason"] == "Too little filed."


# ---------------------------------------------------------------------------
# Financials
# ---------------------------------------------------------------------------


@pytest.fixture
def statements(monkeypatch):
    data = research.FinancialsData("TEST", annual=_frames(), quarterly=_frames(list(_QUARTERLY_ROWS)))
    monkeypatch.setattr(api.research, "load_financials", lambda ticker: data)
    return data


def test_the_income_statement_is_laid_out_and_reconciled(client, snapshot, statements):
    body = _strict(client.get("/api/research/TEST/financials"))
    assert body["available"] and body["label"] == "Income statement"
    assert body["statement"] == "PL" and body["unit"] == "Millions" and body["unit_suffix"] == " (m)"
    # The leading column is the trailing twelve months.
    assert body["periods"][0]["is_latest"] and body["periods"][0]["label"] == "TTM"
    rows = [row for section in body["sections"] for row in section["rows"]]
    assert all(len(row["cells"]) == len(body["periods"]) for row in rows)
    # The fixture company ties exactly, so no reconciliation may fire.
    assert not any(row["flagged"] for row in rows)
    assert body["checks"] and {c["status"] for c in body["checks"]} <= {"ok", "not evaluated"}
    assert [s["key"] for s in body["statements"]] == ["PL", "BS", "CF", "DERIVED"]


def test_the_balance_sheet_leads_with_the_latest_quarter(client, snapshot, statements):
    """Against the fiscal years its leading column is the latest quarter-end;
    against quarters that quarter is already the first column."""
    annual = _strict(client.get("/api/research/TEST/financials", params={"statement": "BS"}))
    assert annual["periods"][0]["is_latest"] and "MRQ" in annual["periods"][0]["label"]
    quarterly = _strict(client.get(
        "/api/research/TEST/financials",
        params={"statement": "BS", "period": "quarterly", "unit": "As reported"},
    ))
    assert quarterly["dimension"] == "MRQ" and quarterly["unit_suffix"] == ""
    assert not any(p["is_latest"] for p in quarterly["periods"])


def test_no_fundamentals_is_said_rather_than_shown_empty(client, snapshot, monkeypatch):
    monkeypatch.setattr(api.research, "load_financials", lambda t: research.FinancialsData(t, {}, {}))
    body = _strict(client.get("/api/research/TEST/financials"))
    assert body["available"] is False and body["sections"] == []


def test_an_unknown_statement_is_refused(client, snapshot, statements):
    assert client.get("/api/research/TEST/financials", params={"statement": "XX"}).status_code == 422


# ---------------------------------------------------------------------------
# Valuation and Growth
# ---------------------------------------------------------------------------


@pytest.fixture
def peers(monkeypatch, peer_universe):
    seen = []

    def load(session, *, today=None):
        seen.append(session)
        return peer_universe

    monkeypatch.setattr(api.research, "load_peers", load)
    return seen


def test_valuation_compares_against_the_companys_own_industry(client, snapshot, peers):
    body = _strict(client.get("/api/research/TEST/valuation"))
    assert body["kind"] == "valuation" and body["universe_loaded"]
    assert body["industry"] == "Consumer Electronics" and body["peers"] > 0
    assert body["priced"] is not None
    # Keyed on the company's last session, never on the ticker.
    assert peers == [snapshot[0].market_session]
    pe = next(r for r in body["rows"] if r["label"] == "P/E (TTM)")
    assert pe["current"] == 20.0 and not pe["percent"]
    assert pe["history_unit"] == "sessions"
    yield_row = next(r for r in body["rows"] if r["label"] == "Dividend Yield (TTM)")
    assert yield_row["percent"]


def test_a_verdict_comes_with_its_direction(client, snapshot, peers):
    rows = _strict(client.get("/api/research/TEST/valuation"))["rows"]
    for row in rows:
        if row["gap"] is None:
            assert row["verdict"] == "" and row["favourable"] is None
        else:
            assert row["verdict"]


def test_growth_reads_rates_in_quarters(client, snapshot, peers):
    body = _strict(client.get("/api/research/TEST/growth"))
    assert body["kind"] == "growth"
    assert all(r["percent"] for r in body["rows"])
    assert {r["history_unit"] for r in body["rows"] if r["history_median"] is not None} <= {"quarters"}
    for row in body["rows"]:
        if row["verdict"] not in ("", "in line"):
            assert row["verdict"].endswith("than industry")


def test_a_failed_universe_is_said_not_hidden(client, snapshot, monkeypatch):
    monkeypatch.setattr(api.research, "load_peers", lambda session, *, today=None: pd.DataFrame())
    body = _strict(client.get("/api/research/TEST/valuation"))
    assert body["universe_loaded"] is False and body["industry"] == ""
    assert all(r["industry_median"] is None for r in body["rows"])


# ---------------------------------------------------------------------------
# Insiders
# ---------------------------------------------------------------------------


def test_insiders_count_open_market_trades_only_and_get_the_sign_right(
    client, snapshot, monkeypatch, insider_frame
):
    monkeypatch.setattr(api.research, "load_insiders", lambda ticker, *, today=None: insider_frame)
    body = _strict(client.get("/api/research/TEST/insiders"))
    assert body["available"]
    assert {t["code"] for t in body["trades"]} <= {"P", "S"}
    for trade in body["trades"]:
        assert (trade["value"] > 0) == (trade["code"] == "P")
    twelve = next(w for w in body["windows"] if w["months"] == 12)
    # The fixture's reason for existing: the correct total is positive.
    assert twelve["available"] and twelve["net"] > 0
    assert body["shown_rows"] <= body["filed_rows"] == len(insider_frame)
    assert all(q["label"][:4].isdigit() and " Q" in q["label"] for q in body["quarters"])
    assert body["quarters"][-1]["partial"] is True
    assert body["reconciliation"]["columns"] == ["Category", "Rows", "Shares", "Dollars", "Note"]
    # The price line is the snapshot's own closes over the lookback.
    assert body["prices"]["sessions"][-1] == TODAY.isoformat()


def test_no_filings_is_an_answer(client, snapshot, monkeypatch):
    monkeypatch.setattr(api.research, "load_insiders", lambda ticker, *, today=None: pd.DataFrame())
    body = _strict(client.get("/api/research/TEST/insiders"))
    assert body["available"] is False and body["trades"] == []


# ---------------------------------------------------------------------------
# Holdings
# ---------------------------------------------------------------------------


@pytest.fixture
def register(monkeypatch, holders_frame, holdings_summary, investor_names):
    shares = pd.DataFrame(
        {
            "Calendar Date": pd.to_datetime(
                ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]
            ),
            sc.SF1_CAPTIONS["sharesbas"]: [10_000_000.0] * 6,
            sc.SF1_CAPTIONS["sharefactor"]: [1.0] * 6,
        }
    )
    data = research.HoldingsData(
        ticker="TEST", summary=holdings_summary, current=holders_frame, history=holders_frame,
        names=investor_names, derived=shares, loaded_at=TODAY,
    )
    monkeypatch.setattr(api.research, "load_holdings", lambda ticker, *, today=None: data)
    return data


def test_holdings_date_the_register_and_never_call_it_current(client, snapshot, register):
    body = _strict(client.get("/api/research/TEST/holdings"))
    assert body["available"]
    assert body["as_of"] == "2026-06-30"
    assert body["deadline"] == "2026-08-14" and body["days_old"] == -9
    assert body["filing_lag_days"] == 45
    assert 0 < len(body["holders"]) <= 20
    assert all(len(h["series"]) == len(body["periods"]) for h in body["holders"])


def test_a_holders_missing_quarter_stays_missing(client, snapshot, register):
    holders = _strict(client.get("/api/research/TEST/holdings"))["holders"]
    assert any(None in h["series"] for h in holders)


def test_the_curated_lists_are_printed(client, snapshot, register):
    body = _strict(client.get("/api/research/TEST/holdings"))
    assert any(f["label"].startswith("Vanguard") for f in body["families"])
    assert body["index_managers"]
