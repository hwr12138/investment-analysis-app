"""The options collector against a fake ThetaData client — no network, no ``thetadata``.

The fake answers in ThetaData's own shapes (``CALL``/``PUT``, a 17:15 ET
``created`` stamp, ``NoDataFoundError`` for an empty answer, gRPC-style errors
with ``code()``/``details()``) so the normalisation and retry code run for real.
"""

from __future__ import annotations

import sys
import types
from datetime import date, timedelta

import pandas as pd
import pytest

from app import options_layout as layout
from app.logic.options import chain as C
from app.logic.options import snapshot, universe
from collector import pipeline
from collector.client import SourceError, ThetaSource, chunks, normalise_chain
from collector.store import LocalStore
from tests import options_fixtures as fx


class NoDataFoundError(Exception):
    pass


class _Code:
    def __init__(self, name):
        self.name = name


class RpcError(Exception):
    def __init__(self, code, details):
        super().__init__(details)
        self._code, self._details = code, details

    def code(self):
        return _Code(self._code)

    def details(self):
        return self._details


SESSIONS = [d.date() for d in pd.bdate_range("2026-09-01", "2026-10-30")]
CHAIN = fx.build_chain()


def _raw_chain(root, session, max_dte=None):
    chain = CHAIN
    if max_dte is not None:
        chain = chain[[(e - session).days <= max_dte for e in chain[C.EXPIRATION]]]
    chain = chain[[e > session for e in chain[C.EXPIRATION]]]
    return pd.DataFrame({
        "symbol": root, "expiration": chain[C.EXPIRATION], "strike": chain[C.STRIKE],
        "right": chain[C.RIGHT].map({C.CALL: "CALL", C.PUT: "PUT"}),
        "created": pd.Timestamp(f"{session} 17:15:30").tz_localize("America/New_York"),
        "last_trade": pd.NaT, "open": 0.0, "high": 0.0, "low": 0.0, "close": chain[C.CLOSE],
        "volume": chain[C.VOLUME], "count": 1, "bid_size": chain[C.BID_SIZE], "bid_exchange": 1,
        "bid": chain[C.BID], "bid_condition": 50, "ask_size": chain[C.ASK_SIZE], "ask_exchange": 1,
        "ask": chain[C.ASK], "ask_condition": 50,
    })


class FakeClient:
    def __init__(self, released_through=date(2026, 10, 30), fail=None, busy=0):
        self.released_through = released_through
        self.fail = fail or {}  # symbol → exception to raise on option calls
        self.busy = busy  # how many option calls answer "Too many concurrent" first
        self.calls = []

    def _sessions(self, start, end):
        return [d for d in SESSIONS if start <= d <= min(end, self.released_through)]

    def option_history_eod(self, start_date, end_date, symbol, expiration, max_dte=None):
        self.calls.append(("option", symbol, start_date, end_date, max_dte))
        if self.busy:
            self.busy -= 1
            raise RpcError("RESOURCE_EXHAUSTED", "Too many concurrent requests. Please reduce the number")
        if symbol in self.fail:
            raise self.fail[symbol]
        days = self._sessions(start_date, end_date)
        if not days:
            raise NoDataFoundError()
        return pd.concat([_raw_chain(symbol, d, max_dte) for d in days], ignore_index=True)

    def _eod(self, symbol, start_date, end_date):
        self.calls.append(("eod", symbol, start_date, end_date))
        days = self._sessions(start_date, end_date)
        if not days:
            raise NoDataFoundError()
        created = [pd.Timestamp(f"{d} 17:15:00").tz_localize("America/New_York") for d in days]
        return pd.DataFrame({"created": created, "close": fx.SPOT})

    stock_history_eod = _eod
    index_history_eod = _eod

    def interest_rate_history_eod(self, symbol, start_date, end_date):
        self.calls.append(("rates", symbol, start_date, end_date))
        days = [start_date + timedelta(days=i) for i in range((min(end_date, self.released_through) - start_date).days + 1)]
        return pd.DataFrame({"created": [d.isoformat() for d in days], "rate": fx.YIELD_PCT})


def _source(client):
    return ThetaSource(client, sleep=lambda s: None, log=lambda m: None)


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


#: The real refresh, kept before the autouse stub below replaces it.
_REAL_REFRESH_EARNINGS = pipeline._refresh_earnings


@pytest.fixture(autouse=True)
def _no_earnings_scrape(monkeypatch):
    """The earnings refresh scrapes Yahoo; no test may reach the network."""
    monkeypatch.setattr(pipeline, "_refresh_earnings", lambda store, log, today: None)


def test_chunks_split_by_month_and_by_365_days():
    assert chunks(date(2026, 9, 15), date(2026, 11, 2), months=True) == [
        (date(2026, 9, 15), date(2026, 9, 30)), (date(2026, 10, 1), date(2026, 10, 31)), (date(2026, 11, 1), date(2026, 11, 2)),
    ]
    pieces = chunks(date(2025, 1, 1), date(2026, 6, 30))
    assert all((b - a).days < 365 for a, b in pieces) and pieces[-1][1] == date(2026, 6, 30)


def test_normalise_chain_takes_the_session_from_the_report_stamp():
    chain = normalise_chain(_raw_chain("TEST", date(2026, 10, 23)), "TEST")
    assert list(chain.columns) == list(C.COLUMNS)
    assert set(chain[C.SESSION]) == {date(2026, 10, 23)}
    assert set(chain[C.RIGHT]) == {C.CALL, C.PUT}


def test_an_index_merges_its_two_roots_keeping_the_first_where_they_overlap(monkeypatch):
    monkeypatch.setitem(universe.OPTION_ROOTS, "SPX", ("SPXW", "SPX"))
    client = FakeClient()
    chain = _source(client).option_chain("SPX", date(2026, 10, 23), date(2026, 10, 23))
    assert [c[1] for c in client.calls] == ["SPXW", "SPX"]
    assert set(chain[C.SYMBOL]) == {"SPX"}
    assert not chain.duplicated([C.SESSION, C.EXPIRATION, C.STRIKE, C.RIGHT]).any()


def test_berkshire_options_are_fetched_under_their_occ_root_and_stored_under_the_name():
    client = FakeClient()
    source = _source(client)
    chain = source.option_chain("BRK.B", date(2026, 10, 23), date(2026, 10, 23))
    source.closes("BRK.B", date(2026, 10, 23), date(2026, 10, 23))
    assert [c[:2] for c in client.calls] == [("option", "BRKB"), ("eod", "BRK.B")]
    assert set(chain[C.SYMBOL]) == {"BRK.B"}


def test_too_many_concurrent_is_retried_and_a_permission_error_is_not():
    client = FakeClient(busy=2)
    assert not _source(client).option_chain("GLD", date(2026, 10, 23), date(2026, 10, 23)).empty
    denied = FakeClient(fail={"GLD": RpcError("PERMISSION_DENIED", "requires a standard subscription")})
    with pytest.raises(SourceError, match="PERMISSION_DENIED"):
        _source(denied).option_chain("GLD", date(2026, 10, 23), date(2026, 10, 23))


def test_no_data_is_an_empty_frame_not_an_error():
    chain = _source(FakeClient()).option_chain("GLD", date(2026, 10, 24), date(2026, 10, 25))
    assert chain.empty


def test_daily_catches_up_every_released_session_and_writes_the_manifest_last(store):
    store.write_json(layout.MANIFEST, {"last_session": "2026-10-22"})
    writes = []
    original = store.write_bytes
    store.write_bytes = lambda path, data: (writes.append(path), original(path, data))
    manifest = pipeline.run_daily(_source(FakeClient()), store, ["GLD", "SPY"], date(2026, 10, 28))
    assert manifest["last_session"] == "2026-10-27"
    assert writes[-1] == layout.MANIFEST
    metrics = store.read_frame(layout.METRICS)
    assert sorted(set(metrics["session"])) == [date(2026, 10, 23), date(2026, 10, 26), date(2026, 10, 27)]
    latest = store.read_frame(layout.LATEST_CHAINS)
    assert set(latest[C.SESSION]) == {date(2026, 10, 27)} and set(latest[C.SYMBOL]) == {"GLD", "SPY"}
    assert metrics["atm_iv_30"].notna().all()


def test_daily_before_release_changes_nothing(store):
    store.write_json(layout.MANIFEST, {"last_session": "2026-10-22"})
    manifest = pipeline.run_daily(_source(FakeClient(released_through=date(2026, 10, 22))), store, ["GLD"], date(2026, 10, 24))
    assert manifest == {"last_session": "2026-10-22"}
    assert store.read_bytes(layout.METRICS) is None


def test_a_failing_name_is_recorded_and_the_rest_still_write(store):
    client = FakeClient(fail={"GLD": RpcError("INVALID_ARGUMENT", "bad symbol")})
    manifest = pipeline.run_daily(_source(client), store, ["GLD", "SPY"], date(2026, 10, 27))
    assert manifest["symbols"]["GLD"]["error"].startswith("option EOD GLD")
    assert manifest["symbols"]["SPY"]["error"] is None
    assert set(store.read_frame(layout.LATEST_CHAINS)[C.SYMBOL]) == {"SPY"}


def test_daily_fails_without_advancing_when_no_name_has_a_chain(store):
    store.write_json(layout.MANIFEST, {"last_session": "2026-10-22"})
    client = FakeClient(fail={"GLD": RpcError("INVALID_ARGUMENT", "bad")})
    with pytest.raises(SourceError):
        pipeline.run_daily(_source(client), store, ["GLD"], date(2026, 10, 24))
    assert store.read_json(layout.MANIFEST) == {"last_session": "2026-10-22"}


def test_backfill_checkpoints_each_month_and_resumes(store):
    client = FakeClient()
    pipeline.run_backfill(_source(client), store, ["GLD"], date(2026, 9, 1), date(2026, 10, 23))
    manifest = store.read_json(layout.MANIFEST)
    assert manifest["backfill"]["done"] == ["GLD:2026-09", "GLD:2026-10"]
    assert all(c[4] == pipeline.BACKFILL_MAX_DTE for c in client.calls if c[0] == "option")
    metrics = store.read_frame(layout.METRICS)
    assert len(metrics) == len([d for d in SESSIONS if d <= date(2026, 10, 23)])

    again = FakeClient()
    pipeline.run_backfill(_source(again), store, ["GLD"], date(2026, 9, 1), date(2026, 10, 23))
    assert not [c for c in again.calls if c[0] == "option"]  # nothing left to fetch


def test_recompute_restates_metrics_from_stored_chains_without_a_request(store):
    pipeline.run_backfill(_source(FakeClient()), store, ["GLD"], date(2026, 10, 1), date(2026, 10, 23))
    before = store.read_frame(layout.METRICS)
    store.write_frame(layout.METRICS, before.assign(atm_iv_30=0.0))  # a stale definition
    manifest = pipeline.run_recompute(store, ["GLD"], log=lambda m: None)
    after = store.read_frame(layout.METRICS)
    assert after["atm_iv_30"].to_numpy() == pytest.approx(before["atm_iv_30"].to_numpy())
    assert manifest["metrics_version"] == pipeline.history.METRICS_VERSION


def test_chain_paths_round_trip():
    for symbol in ("SPX", "BRK.B"):  # a share class's dot is not the extension
        path = layout.chain_path(date(2026, 10, 23), symbol)
        assert layout.parse_chain_path(path) == (date(2026, 10, 23), symbol)
    assert layout.parse_chain_path(layout.METRICS) is None


def test_a_holiday_run_writes_nothing_and_the_next_run_catches_up(store):
    # 2026-11-26 is Thanksgiving: SPY has no close, so it is not a session.
    holiday = date(2026, 11, 26)
    client = FakeClient(released_through=date(2026, 11, 27))
    global SESSIONS
    saved = SESSIONS
    SESSIONS = [d.date() for d in pd.bdate_range("2026-09-01", "2026-11-27") if d.date() != holiday]
    try:
        store.write_json(layout.MANIFEST, {"last_session": "2026-11-25"})
        manifest = pipeline.run_daily(_source(client), store, ["GLD"], date(2026, 11, 27))
        assert manifest == {"last_session": "2026-11-25"}
        assert not [c for c in client.calls if c[0] == "option"]  # no chain was asked for
        manifest = pipeline.run_daily(_source(client), store, ["GLD"], date(2026, 11, 28))
        assert manifest["last_session"] == "2026-11-27"
    finally:
        SESSIONS = saved


def test_earnings_refresh_by_age_not_by_weekday(store):
    assert pipeline._earnings_stale(store, date(2026, 10, 27))
    store.write_frame(layout.EARNINGS, pd.DataFrame({"symbol": ["MU"], "next_earnings": [date(2026, 12, 23)],
                                                     "fetched": [date(2026, 10, 27)]}))
    assert not pipeline._earnings_stale(store, date(2026, 10, 31))  # Saturday, four days on
    assert pipeline._earnings_stale(store, date(2026, 11, 3))       # the next Tuesday


def _fake_yfinance(dates_by_symbol):
    """Yahoo's answer shape: earnings dates as a tz-aware index, one row each."""
    def get_earnings_dates(symbol, limit):
        index = pd.DatetimeIndex(dates_by_symbol.get(symbol, []), name="Earnings Date").tz_localize("America/New_York")
        return pd.DataFrame({"EPS Estimate": [1.0] * len(index)}, index=index)

    module = types.ModuleType("yfinance")
    module.Ticker = lambda symbol: types.SimpleNamespace(get_earnings_dates=lambda limit=8: get_earnings_dates(symbol, limit))
    return module


def test_what_the_earnings_refresh_writes_its_staleness_check_can_read(store, monkeypatch):
    """The refresh and the check, both real, through a store round trip.

    The refresh once wrote ``fetched`` as a Timestamp, and the next run's check
    subtracted it from a date: every run crashed after its fetch, before its
    manifest. The test above writes the frame itself, so it could not see it.
    """
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance({"MU": ["2026-09-23 16:00", "2026-12-17 16:00"]}))
    monkeypatch.setattr(pipeline.universe, "SINGLE_STOCKS", frozenset({"MU", "NVDA"}))
    _REAL_REFRESH_EARNINGS(store, lambda m: None, date(2026, 11, 17))

    frame = store.read_frame(layout.EARNINGS).set_index("symbol")
    assert frame.loc["MU", "next_earnings"] == date(2026, 12, 17)
    assert frame.loc["NVDA", "next_earnings"] is None  # nothing upcoming is not a date
    assert set(frame["fetched"]) == {date(2026, 11, 17)}
    assert not pipeline._earnings_stale(store, date(2026, 11, 18))
    assert pipeline._earnings_stale(store, date(2026, 11, 24))


def test_the_earnings_refresh_asks_yahoo_for_berkshire_by_its_hyphenated_name(store, monkeypatch):
    # Yahoo answers ``BRK.B`` with nothing; the date is under ``BRK-B``.
    monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance({"BRK-B": ["2027-03-01 08:00"]}))
    monkeypatch.setattr(pipeline.universe, "SINGLE_STOCKS", frozenset({"BRK.B"}))
    _REAL_REFRESH_EARNINGS(store, lambda m: None, date(2026, 11, 21))

    frame = store.read_frame(layout.EARNINGS).set_index("symbol")
    assert frame.loc["BRK.B", "next_earnings"] == date(2027, 3, 1)


def test_earnings_staleness_reads_the_timestamp_the_store_already_holds(store):
    """The bucket's file from 2026-11-17 carries ``fetched`` as datetime64."""
    store.write_frame(layout.EARNINGS, pd.DataFrame({"symbol": ["MU"], "next_earnings": [date(2026, 12, 17)],
                                                     "fetched": pd.to_datetime(["2026-11-17"])}))
    assert not pipeline._earnings_stale(store, date(2026, 11, 21))
    assert pipeline._earnings_stale(store, date(2026, 11, 24))


def test_daily_solves_the_screens_and_writes_them_before_the_manifest(store):
    store.write_json(layout.MANIFEST, {"last_session": "2026-10-22"})
    writes = []
    original = store.write_bytes
    store.write_bytes = lambda path, data: (writes.append(path), original(path, data))
    manifest = pipeline.run_daily(_source(FakeClient()), store, ["GLD", "SPY"], date(2026, 10, 28))
    assert manifest["screens"] == {"session": "2026-10-27", "unpriced": {}}
    assert all(writes.index(p) < writes.index(layout.MANIFEST) for p in (layout.DISCOVER, layout.SPREADS, layout.EXPIRATIONS))
    assert sorted(store.read_frame(layout.DISCOVER)["symbol"]) == ["GLD", "SPY"]
    spreads = store.read_frame(layout.SPREADS)
    assert set(spreads["sd"]) == set(snapshot.SD_TARGETS) and set(spreads["symbol"]) == {"GLD", "SPY"}
    listed = store.read_frame(layout.EXPIRATIONS)
    assert set(listed["expiration"]) == {e for e in fx.EXPIRATIONS if e > date(2026, 10, 27)}


def test_the_screens_mode_rebuilds_them_from_the_store_without_a_request(store):
    pipeline.run_daily(_source(FakeClient()), store, ["GLD"], date(2026, 10, 28))
    for path in (layout.DISCOVER, layout.SPREADS, layout.EXPIRATIONS):
        (store.root / path).unlink()
    manifest = store.read_json(layout.MANIFEST)
    del manifest["screens"]  # as an older collector left it
    store.write_json(layout.MANIFEST, manifest)
    manifest = pipeline.run_screens(store, log=lambda m: None)
    assert manifest["screens"]["session"] == manifest["last_session"]
    assert list(store.read_frame(layout.DISCOVER)["symbol"]) == ["GLD"]


def test_the_screens_mode_refuses_an_empty_store(store):
    with pytest.raises(SourceError, match="no completed daily run"):
        pipeline.run_screens(store, log=lambda m: None)


def test_a_recompute_rebuilds_the_screens_from_the_restated_history(store):
    pipeline.run_daily(_source(FakeClient()), store, ["GLD"], date(2026, 10, 28))
    (store.root / layout.DISCOVER).unlink()
    manifest = pipeline.run_recompute(store, ["GLD"], log=lambda m: None)
    assert manifest["screens"]["session"] == "2026-10-27"
    assert list(store.read_frame(layout.DISCOVER)["symbol"]) == ["GLD"]


def _partly_stored(store):
    """Three sessions as a run inside the free tier's release leaves them: SPY's chains not out yet."""
    store.write_json(layout.MANIFEST, {"last_session": "2026-10-22"})
    client = FakeClient(fail={"SPY": NoDataFoundError()})
    pipeline.run_daily(_source(client), store, ["GLD", "SPY"], date(2026, 10, 28), log=lambda m: None)


def test_repair_fetches_only_the_names_a_session_is_missing(store):
    _partly_stored(store)
    assert set(store.read_frame(layout.METRICS)["symbol"]) == {"GLD"}
    latest = store.read_bytes(layout.LATEST_CHAINS)
    (store.root / layout.DISCOVER).unlink()

    client = FakeClient()
    manifest = pipeline.run_repair(_source(client), store, ["GLD", "SPY"], [date(2026, 10, 26)], log=lambda m: None)

    assert [c[1:4] for c in client.calls if c[0] == "option"] == [("SPY", date(2026, 10, 26), date(2026, 10, 26))]
    metrics = store.read_frame(layout.METRICS)
    assert list(metrics[metrics["symbol"] == "SPY"]["session"]) == [date(2026, 10, 26)]
    assert metrics["atm_iv_30"].notna().all()
    assert store.read_frame(layout.chain_path(date(2026, 10, 26), "SPY")) is not None
    # The night the page shows is untouched; its screens are rebuilt over the longer history.
    assert manifest["last_session"] == "2026-10-27"
    assert store.read_bytes(layout.LATEST_CHAINS) == latest
    assert manifest["screens"]["session"] == "2026-10-27" and store.read_frame(layout.DISCOVER) is not None
    assert manifest["repair"]["repaired"] == 1 and manifest["repair"]["errors"] == {}
    assert store.read_json(layout.MANIFEST) == manifest

    again = FakeClient()
    manifest = pipeline.run_repair(_source(again), store, ["GLD", "SPY"], [date(2026, 10, 26)], log=lambda m: None)
    assert not [c for c in again.calls if c[0] == "option"] and manifest["repair"]["repaired"] == 0


def test_a_name_still_missing_after_a_repair_is_recorded(store):
    _partly_stored(store)
    client = FakeClient(fail={"SPY": NoDataFoundError()})
    manifest = pipeline.run_repair(_source(client), store, ["GLD", "SPY"], [date(2026, 10, 26)], log=lambda m: None)
    assert manifest["repair"]["errors"] == {"SPY:2026-10-26": "no option chain"}
    assert set(store.read_frame(layout.METRICS)["symbol"]) == {"GLD"}


def test_repair_refuses_the_daily_runs_sessions_and_days_that_are_not_sessions(store):
    with pytest.raises(SourceError, match="no completed daily run"):
        pipeline.run_repair(_source(FakeClient()), store, ["SPY"], [date(2026, 10, 26)], log=lambda m: None)
    _partly_stored(store)
    before = store.read_json(layout.MANIFEST)
    with pytest.raises(SourceError, match="the daily run fetches it"):
        pipeline.run_repair(_source(FakeClient()), store, ["SPY"], [date(2026, 10, 28)], log=lambda m: None)
    with pytest.raises(SourceError, match="is not a session"):
        pipeline.run_repair(_source(FakeClient()), store, ["SPY"], [date(2026, 10, 24)], log=lambda m: None)  # a Saturday
    assert store.read_json(layout.MANIFEST) == before


def test_the_repair_mode_needs_its_sessions(monkeypatch, tmp_path, capsys):
    from collector import main

    monkeypatch.setenv("THETADATA_API_KEY", "not-a-key")
    with pytest.raises(SystemExit):
        main.main(["repair", "--store", f"file://{tmp_path}"])
    assert "repair needs --sessions" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# One run at a time (collector.lease)
# ---------------------------------------------------------------------------


def test_a_nightly_run_in_the_middle_of_a_backfill_keeps_its_session_and_screens(store):
    """The backfill gives way between months and does not write back a stale manifest.

    It used to hold the manifest it read at its start and write it back after
    every name-month, so a nightly run that finished in between had its newest
    session and screens rolled back by the backfill's next write.
    """
    from collector.lease import GuardedStore, Lease

    class Clock:
        now = 1_000_000.0

        def __call__(self):
            return self.now

    clock = Clock()
    hooks = []

    def sleep(seconds):
        clock.now += seconds
        if hooks:
            hooks.pop(0)()

    def lease(mode):
        return Lease(store, mode, owner=mode, clock=clock, sleep=sleep, log=lambda m: None, renew=False)

    backfill, daily = lease("backfill"), lease("daily")

    class AskedAfterOneMonth(FakeClient):
        def option_history_eod(self, **kwargs):
            out = super().option_history_eod(**kwargs)
            if len([c for c in self.calls if c[0] == "option"]) == 1:
                daily.ask()  # the nightly run arrives while September is being fetched
            return out

    nightly = {}

    def nightly_run():
        daily.acquire(max_wait=0)
        nightly.update(pipeline.run_daily(
            _source(FakeClient()), GuardedStore(store, daily), ["GLD", "SPY"], date(2026, 10, 31), log=lambda m: None
        ))
        daily.release()

    hooks.append(nightly_run)
    backfill.acquire()
    client = AskedAfterOneMonth()
    pipeline.run_backfill(
        _source(client), GuardedStore(store, backfill), ["GLD"], date(2026, 9, 1), date(2026, 10, 23),
        lease=backfill, log=lambda m: None,
    )
    backfill.release()

    assert nightly["last_session"] == "2026-10-30"
    manifest = store.read_json(layout.MANIFEST)
    assert manifest["last_session"] == "2026-10-30"
    assert manifest["screens"]["session"] == "2026-10-30"
    assert manifest["backfill"]["done"] == ["GLD:2026-09", "GLD:2026-10"]
    metrics = store.read_frame(layout.METRICS)
    assert {"GLD", "SPY"} <= set(metrics["symbol"])
    assert date(2026, 10, 30) in set(pd.to_datetime(metrics["session"]).dt.date)


def test_a_backfill_records_a_failure_after_its_last_finished_month(store):
    client = FakeClient(fail={"SPY": RpcError("PERMISSION_DENIED", "not on this tier")})
    pipeline.run_backfill(_source(client), store, ["GLD", "SPY"], date(2026, 10, 1), date(2026, 10, 23), log=lambda m: None)
    manifest = store.read_json(layout.MANIFEST)
    assert manifest["backfill"]["done"] == ["GLD:2026-10"]
    assert "SPY:2026-10" in manifest["backfill_errors"]


def test_a_run_that_cannot_have_the_store_writes_nothing_and_names_the_holder(monkeypatch, tmp_path, capsys):
    from collector import lease as lease_module
    from collector import main
    from collector.lease import Lease

    held = Lease(LocalStore(tmp_path), "backfill", owner="backfill on job-abc", log=lambda m: None, renew=False)
    held.acquire()
    monkeypatch.setattr(lease_module, "DEFAULT_MAX_WAIT_SECONDS", 0)
    monkeypatch.setenv("THETADATA_API_KEY", "not-a-key")
    monkeypatch.setattr(main.ThetaSource, "connect", classmethod(lambda cls, key: pytest.fail("fetched without the store")))

    assert main.main(["daily", "--store", f"file://{tmp_path}"]) == 1
    assert "in use by backfill on job-abc" in capsys.readouterr().err
    assert LocalStore(tmp_path).read_json(layout.MANIFEST) is None
    assert held.holder()["owner"] == "backfill on job-abc"


def test_the_store_is_released_when_a_run_fails(monkeypatch, tmp_path, capsys):
    from collector import main

    monkeypatch.setenv("THETADATA_API_KEY", "not-a-key")
    monkeypatch.setattr(main.ThetaSource, "connect", classmethod(lambda cls, key: _source(FakeClient())))

    def fails(*args, **kwargs):
        raise SourceError("ThetaData is down")

    monkeypatch.setattr(main.pipeline, "run_daily", fails)
    assert main.main(["daily", "--store", f"file://{tmp_path}"]) == 1
    assert "ThetaData is down" in capsys.readouterr().err
    assert LocalStore(tmp_path).read_bytes(layout.LOCK) is None
