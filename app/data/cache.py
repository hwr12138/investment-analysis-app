"""The data layer's in-process caches.

Every fetcher in ``app/data`` worth remembering is decorated with :func:`data`
(a value — a frame, a dict, a tuple) or :func:`resource` (a shared client
object). Both live in this process only: Cloud Run runs one worker, so an
instance has one cache, and each instance starts cold.

They replaced ``st.cache_data`` and ``st.cache_resource`` and keep the
behaviours the data layer was built on:

* **An exception is never cached.** A call that raises leaves nothing behind,
  and the next call tries again. ``market_data.FetchFailed`` and
  ``options_store.OptionsStoreError`` rely on it: the cached fetcher *raises*,
  and an uncached wrapper turns the raise into ``Sourced.unavailable``. A
  failure *returned* from a cached function is kept for its whole TTL — a
  transient outage remembered for a day.
* **The key is the arguments as passed, with no defaults filled in.** An
  argument passed positionally and the same argument by keyword are one key;
  leaving a default out and spelling it are two. That is why
  ``fetch_statements`` forwards positionally to ``_cached_statements``, and why
  a cached function never grows a keyword default its callers may or may not
  spell. Arguments must be hashable — a tuple, not a list.
* **One computation per key at a time.** ``parallel.gather`` runs fetchers in
  threads; concurrent calls for one key wait for the first and then read its
  result, while calls for different keys run side by side. A computation that
  raises lets the next waiter try for itself.
* **A cached value is a copy.** :func:`data` stores each result pickled and
  hands every hit its own unpickled copy, so a caller that mutates a frame
  cannot change what the next caller reads. :func:`resource` returns the one
  object, which is the point of it.

An entry expires ``ttl`` seconds after it was written, whether or not it was
read since; ``max_entries`` evicts the least recently used. Each cached function
has ``.clear()``, and :func:`clear_all` empties every cache — the test suite
calls it before each test.
"""

from __future__ import annotations

import functools
import inspect
import math
import pickle
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

#: The clock entries expire against. A module attribute so a test can move it.
_clock: Callable[[], float] = time.monotonic

_MISSING = object()

_caches: list[_Cache] = []
_caches_lock = threading.Lock()


class _Cache(Generic[P, R]):
    """One function's entries, behind the function's own signature."""

    def __init__(
        self,
        func: Callable[P, R],
        *,
        ttl: float | None,
        max_entries: int | None,
        copies: bool,
    ) -> None:
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be positive, or None to keep entries until cleared")
        if max_entries is not None and max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._func = func
        self._signature = inspect.signature(func)
        self._ttl = ttl
        self._max_entries = max_entries
        self._copies = copies
        # Least recently used first. Each value is (expires_at, stored).
        self._entries: OrderedDict[tuple, tuple[float, Any]] = OrderedDict()
        # One lock per key being computed, with the number of callers holding
        # or waiting on it, so the table shrinks back once a key is settled.
        self._computing: dict[tuple, tuple[threading.Lock, int]] = {}
        self._lock = threading.Lock()
        functools.update_wrapper(self, func)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        key = self._key(args, kwargs)
        value = self._read(key)
        if value is not _MISSING:
            return value
        lock = self._enter(key)
        try:
            with lock:
                # Whoever held the lock before us may have just written it.
                value = self._read(key)
                if value is not _MISSING:
                    return value
                value = self._func(*args, **kwargs)
                self._write(key, value)
                return value
        finally:
            self._leave(key)

    def clear(self) -> None:
        """Forgets every entry. A computation still running writes its result."""
        with self._lock:
            self._entries.clear()

    # -- keys ----------------------------------------------------------------

    def _key(self, args: tuple, kwargs: dict) -> tuple:
        # bind() maps positional arguments to their parameter names and, unlike
        # apply_defaults(), leaves out every parameter the caller did not pass.
        key = tuple(self._signature.bind(*args, **kwargs).arguments.items())
        try:
            hash(key)
        except TypeError:
            raise TypeError(
                f"{self._func.__qualname__} is cached and takes hashable arguments "
                "only (a tuple, not a list)"
            ) from None
        return key

    # -- entries -------------------------------------------------------------

    def _read(self, key: tuple) -> Any:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return _MISSING
            expires_at, stored = entry
            if _clock() >= expires_at:
                del self._entries[key]
                return _MISSING
            self._entries.move_to_end(key)
        return pickle.loads(stored) if self._copies else stored

    def _write(self, key: tuple, value: Any) -> None:
        stored = pickle.dumps(value, pickle.HIGHEST_PROTOCOL) if self._copies else value
        with self._lock:
            now = _clock()
            for stale in [k for k, (expires_at, _) in self._entries.items() if now >= expires_at]:
                del self._entries[stale]
            self._entries[key] = (math.inf if self._ttl is None else now + self._ttl, stored)
            self._entries.move_to_end(key)
            if self._max_entries is not None:
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)

    # -- one computation per key ---------------------------------------------

    def _enter(self, key: tuple) -> threading.Lock:
        with self._lock:
            lock, holders = self._computing.get(key, (None, 0))
            if lock is None:
                lock = threading.Lock()
            self._computing[key] = (lock, holders + 1)
            return lock

    def _leave(self, key: tuple) -> None:
        with self._lock:
            lock, holders = self._computing[key]
            if holders == 1:
                del self._computing[key]
            else:
                self._computing[key] = (lock, holders - 1)


def _register(cache: _Cache) -> _Cache:
    with _caches_lock:
        _caches.append(cache)
    return cache


def data(
    *, ttl: float, max_entries: int | None = None
) -> Callable[[Callable[P, R]], _Cache[P, R]]:
    """Caches a fetcher's result for ``ttl`` seconds; every hit gets a copy.

    The result must pickle. ``max_entries`` bounds a cache whose entries are
    large (the options store's chains) by evicting the least recently used.
    """

    def decorate(func: Callable[P, R]) -> _Cache[P, R]:
        return _register(_Cache(func, ttl=ttl, max_entries=max_entries, copies=True))

    return decorate


def resource() -> Callable[[Callable[P, R]], _Cache[P, R]]:
    """Caches one shared object per key until cleared, and hands out that object.

    For a client that is costly to build and safe to share — a GCS bucket handle.
    """

    def decorate(func: Callable[P, R]) -> _Cache[P, R]:
        return _register(_Cache(func, ttl=None, max_entries=None, copies=False))

    return decorate


def clear_all() -> None:
    """Empties every cache in the process, values and resources alike."""
    with _caches_lock:
        caches = list(_caches)
    for cache in caches:
        cache.clear()
