"""The store's lease: one run at a time, and a backfill that gives way.

Every lease here runs on a clock the test moves, and sleeping moves it, so a
wait of hours takes no time; ``renew=False`` keeps the background renewal out
of it, and a test that is about renewal calls :meth:`Lease.renew_once`.
"""

from __future__ import annotations

import pytest

from app import options_layout as layout
from collector import lease as L
from collector.lease import GuardedStore, Lease, LeaseLost, StoreBusy
from collector.store import LocalStore


class Clock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now
        self.on_sleep = []  # callables run, in turn, at each sleep

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.on_sleep:
            self.on_sleep.pop(0)()


@pytest.fixture
def store(tmp_path):
    return LocalStore(tmp_path)


@pytest.fixture
def clock():
    return Clock()


def _lease(store, clock, mode, name=None):
    return Lease(store, mode, owner=name or mode, clock=clock, sleep=clock.sleep, log=lambda m: None, renew=False)


def test_a_run_waits_for_the_store_then_gives_up_naming_the_holder(store, clock):
    backfill = _lease(store, clock, "backfill")
    backfill.acquire()
    daily = _lease(store, clock, "daily")
    with pytest.raises(StoreBusy, match="in use by backfill"):
        daily.acquire(max_wait=60)
    assert clock.now >= 1_000_060
    assert backfill.holder()["owner"] == "backfill"
    # It withdrew its request, so the backfill does not stop for nobody.
    assert not backfill.should_give_way()


def test_a_released_store_is_free_at_once(store, clock):
    first = _lease(store, clock, "daily", "first")
    first.acquire()
    first.release()
    assert first.holder() == {}
    second = _lease(store, clock, "daily", "second")
    second.acquire(max_wait=0)
    assert second.holder()["owner"] == "second"


def test_a_lease_its_holder_stopped_renewing_is_taken_over(store, clock):
    dead = _lease(store, clock, "daily", "dead")
    dead.acquire()
    clock.now += L.TTL_SECONDS + 1
    alive = _lease(store, clock, "daily", "alive")
    alive.acquire(max_wait=0)
    assert alive.holder()["owner"] == "alive"

    # The old holder finds out at its next renewal, writes nothing more, and
    # its release leaves the new holder's lease alone.
    dead.renew_once()
    assert dead.lost
    with pytest.raises(LeaseLost):
        GuardedStore(store, dead).write_json(layout.MANIFEST, {"last_session": "2026-11-20"})
    assert store.read_json(layout.MANIFEST) is None
    dead.release()
    assert alive.holder()["owner"] == "alive"


def test_a_renewed_lease_is_not_taken_over(store, clock):
    holder = _lease(store, clock, "backfill")
    holder.acquire()
    clock.now += L.TTL_SECONDS - 5
    holder.renew_once()
    clock.now += 10  # past the first expiry, inside the renewed one
    with pytest.raises(StoreBusy):
        _lease(store, clock, "daily").acquire(max_wait=0)
    assert not holder.lost


def test_a_waiting_run_asks_a_backfill_to_give_way_and_is_let_through(store, clock):
    backfill = _lease(store, clock, "backfill")
    backfill.acquire()
    daily = _lease(store, clock, "daily")

    def backfill_notices():
        assert backfill.should_give_way()
        backfill.release()  # what give_way does first

    clock.on_sleep.append(backfill_notices)
    daily.acquire(max_wait=60)
    assert daily.holder()["owner"] == "daily"
    assert store.read_bytes(layout.LOCK_WANTED) is None  # served, so cleared


def test_a_backfill_stands_aside_while_a_request_is_fresh_and_not_once_it_is_stale(store, clock):
    _lease(store, clock, "daily").ask()
    backfill = _lease(store, clock, "backfill")
    with pytest.raises(StoreBusy):
        backfill.acquire(max_wait=L.POLL_SECONDS)  # the store is free, but a run asked first
    assert backfill.holder() == {}

    clock.now += L.REQUEST_FRESH_SECONDS + 1  # the asker stopped renewing its request
    backfill.acquire(max_wait=0)
    assert backfill.holder()["owner"] == "backfill"


def test_backfills_do_not_ask_each_other_to_give_way(store, clock):
    first = _lease(store, clock, "backfill", "first")
    first.acquire()
    with pytest.raises(StoreBusy):
        _lease(store, clock, "backfill", "second").acquire(max_wait=30)
    assert not first.should_give_way()


def test_giving_way_takes_the_store_back_once_the_waiting_run_is_done(store, clock):
    backfill = _lease(store, clock, "backfill")
    backfill.acquire()
    daily = _lease(store, clock, "daily")
    daily.ask()
    assert backfill.should_give_way()

    def daily_runs():
        daily.acquire(max_wait=0)
        assert daily.holder()["owner"] == "daily"
        daily.release()

    clock.on_sleep.append(daily_runs)
    backfill.give_way()
    assert backfill.holder()["owner"] == "backfill"
    assert not backfill.should_give_way()
