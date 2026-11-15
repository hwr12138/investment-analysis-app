"""The data layer's caches: the behaviours ``st.cache_data`` gave that the
fetchers were built on, now ours to keep.

Each one is load-bearing somewhere else. An exception that was cached would
remember an outage for a day (``market_data.FetchFailed``); a key that filled
in defaults would hide the reason ``fetch_statements`` forwards positionally;
a cache that let two threads compute one key would double the requests
``parallel.gather`` exists to save; and a hit that returned the stored object
would let one caller's ``frame.drop(..., inplace=True)`` change the next
caller's data.
"""

from __future__ import annotations

import threading
import time

import pandas as pd
import pytest

from app.data import cache


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(cache, "_clock", lambda: now[0])
    return now


def _counted(**options):
    calls = []

    @cache.data(**options)
    def fetch(ticker, start=None):
        calls.append((ticker, start))
        return {"ticker": ticker, "start": start, "n": len(calls)}

    return fetch, calls


def test_a_repeat_call_is_served_from_the_cache():
    fetch, calls = _counted(ttl=60)
    assert fetch("AAPL") == fetch("AAPL")
    assert calls == [("AAPL", None)]


def test_an_exception_is_never_cached():
    attempts = []

    @cache.data(ttl=60)
    def flaky(symbol):
        attempts.append(symbol)
        if len(attempts) == 1:
            raise RuntimeError("transient outage")
        return 4.18

    with pytest.raises(RuntimeError):
        flaky("^TNX")
    assert flaky("^TNX") == 4.18
    assert flaky("^TNX") == 4.18
    assert attempts == ["^TNX", "^TNX"]


def test_the_key_is_the_arguments_as_passed_without_defaults():
    """Positional and keyword spellings are one key; an omitted default is not."""
    fetch, calls = _counted(ttl=60)
    fetch("AAPL", "2020-01-01")
    fetch("AAPL", start="2020-01-01")
    fetch(ticker="AAPL", start="2020-01-01")
    assert len(calls) == 1
    fetch("AAPL")
    fetch("AAPL", None)  # the same request, spelled out: a second entry
    assert calls[1:] == [("AAPL", None), ("AAPL", None)]


def test_an_entry_expires_its_ttl_after_it_was_written(clock):
    fetch, calls = _counted(ttl=60)
    fetch("AAPL")
    clock[0] += 59
    fetch("AAPL")  # reading does not extend it
    assert len(calls) == 1
    clock[0] += 1
    fetch("AAPL")
    assert len(calls) == 2


def test_max_entries_evicts_the_least_recently_used():
    fetch, calls = _counted(ttl=60, max_entries=2)
    fetch("A")
    fetch("B")
    fetch("A")  # A is now the most recently used
    fetch("C")  # evicts B
    fetch("A")
    assert [ticker for ticker, _ in calls] == ["A", "B", "C"]
    fetch("B")
    assert [ticker for ticker, _ in calls] == ["A", "B", "C", "B"]


def test_every_hit_gets_its_own_copy():
    @cache.data(ttl=60)
    def frame(ticker):
        return pd.DataFrame({"close": [1.0, 2.0]})

    first = frame("AAPL")
    first.drop(columns="close", inplace=True)  # the caller who computed it
    second = frame("AAPL")
    second["close"] = 0.0  # a caller served from the cache
    assert frame("AAPL")["close"].tolist() == [1.0, 2.0]


def test_a_resource_is_the_one_object():
    made = []

    @cache.resource()
    def client(name):
        made.append(name)
        return object()

    assert client("bucket") is client("bucket")
    assert made == ["bucket"]


def test_concurrent_calls_for_one_key_compute_it_once():
    """``parallel.gather`` runs fetchers in threads; one key is one request."""
    started = threading.Event()
    calls = []

    @cache.data(ttl=60)
    def slow(ticker):
        calls.append(ticker)
        started.set()
        time.sleep(0.05)
        return ticker.lower()

    results = []
    threads = [threading.Thread(target=lambda: results.append(slow("AAPL"))) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert started.is_set()
    assert calls == ["AAPL"]
    assert results == ["aapl"] * 8


def test_different_keys_compute_side_by_side():
    """The per-key lock is per key: one slow ticker does not hold up another."""
    inside = threading.Barrier(2, timeout=2)

    @cache.data(ttl=60)
    def fetch(ticker):
        inside.wait()  # both must be inside at once, or this times out
        return ticker

    threads = [threading.Thread(target=fetch, args=(t,)) for t in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not inside.broken


def test_a_waiter_retries_after_the_first_computation_raises():
    attempts = []
    first_inside = threading.Event()
    release = threading.Event()

    @cache.data(ttl=60)
    def flaky(ticker):
        attempts.append(ticker)
        if len(attempts) == 1:
            first_inside.set()
            release.wait(2)
            raise RuntimeError("outage")
        return "ok"

    errors, results = [], []

    def call():
        try:
            results.append(flaky("AAPL"))
        except RuntimeError as error:
            errors.append(error)

    first = threading.Thread(target=call)
    first.start()
    first_inside.wait(2)
    second = threading.Thread(target=call)
    second.start()
    release.set()
    first.join()
    second.join()
    assert len(errors) == 1 and results == ["ok"]
    assert attempts == ["AAPL", "AAPL"]


def test_clear_and_clear_all_forget_entries():
    fetch, calls = _counted(ttl=60)
    other, other_calls = _counted(ttl=60)
    fetch("A")
    other("A")
    fetch.clear()
    fetch("A")
    other("A")
    assert (len(calls), len(other_calls)) == (2, 1)
    cache.clear_all()
    fetch("A")
    other("A")
    assert (len(calls), len(other_calls)) == (3, 2)


def test_an_unhashable_argument_is_refused_by_name():
    fetch, _ = _counted(ttl=60)
    with pytest.raises(TypeError, match="hashable"):
        fetch(["AAPL", "MSFT"])


def test_the_cached_function_keeps_its_name_and_docstring():
    @cache.data(ttl=60)
    def fetch_something(ticker: str) -> str:
        """Fetches something."""
        return ticker

    assert fetch_something.__name__ == "fetch_something"
    assert fetch_something.__doc__ == "Fetches something."
