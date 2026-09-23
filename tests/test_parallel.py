"""``app/data/parallel.py`` — running independent fetches together."""

from __future__ import annotations

import threading
import time

import pytest

from app.data import parallel


def test_results_come_back_by_name():
    assert parallel.gather({"a": lambda: 1, "b": lambda: "two"}) == {"a": 1, "b": "two"}


def test_nothing_to_run_is_an_empty_answer():
    assert parallel.gather({}) == {}


def test_the_calls_actually_overlap():
    """Four quarter-second waits in well under a second is the whole point."""
    started = time.perf_counter()
    parallel.gather({str(i): lambda: time.sleep(0.25) for i in range(4)})
    assert time.perf_counter() - started < 0.75


def test_a_failure_is_raised_not_swallowed():
    def boom():
        raise ValueError("no")

    with pytest.raises(ValueError, match="no"):
        parallel.gather({"ok": lambda: 1, "bad": boom})


def test_the_first_failure_in_submission_order_wins():
    """The same exception a sequential run would have hit first, whichever
    thread happened to finish first."""
    def slow_first():
        time.sleep(0.1)
        raise KeyError("first")

    def fast_second():
        raise ValueError("second")

    with pytest.raises(KeyError):
        parallel.gather({"a": slow_first, "b": fast_second})


def test_every_call_finishes_before_a_failure_is_raised():
    """No request is left running behind the exception."""
    finished = threading.Event()

    def slow():
        time.sleep(0.1)
        finished.set()

    def boom():
        raise ValueError("no")

    with pytest.raises(ValueError):
        parallel.gather({"bad": boom, "slow": slow})
    assert finished.is_set()


def test_a_tolerated_failure_is_the_callers_to_wrap():
    """Nothing here decides what may fail — a caller that tolerates one wraps it."""
    def tolerant():
        try:
            raise ValueError("no")
        except ValueError:
            return None

    assert parallel.gather({"ok": lambda: 1, "tolerated": tolerant}) == {
        "ok": 1, "tolerated": None
    }
