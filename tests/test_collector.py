"""The options collector against a fake ThetaData client — no network, no ``thetadata``.

The fake answers in ThetaData's own shapes (``CALL``/``PUT``, a 17:15 ET
``created`` stamp, ``NoDataFoundError`` for an empty answer, gRPC-style errors
with ``code()``/``details()``) so the normalisation and retry code run for real.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from app import options_layout as layout
from app.logic.options import chain as C
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


def test_spx_merges_its_two_roots_keeping_spxw_where_they_overlap():
    client = FakeClient()
    chain = _source(client).option_chain("SPX", date(2026, 10, 23), date(2026, 10, 23))
    assert [c[1] for c in client.calls] == ["SPXW", "SPX"]
    assert set(chain[C.SYMBOL]) == {"SPX"}
    assert not chain.duplicated([C.SESSION, C.EXPIRATION, C.STRIKE, C.RIGHT]).any()


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
    path = layout.chain_path(date(2026, 10, 23), "SPX")
    assert layout.parse_chain_path(path) == (date(2026, 10, 23), "SPX")
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
