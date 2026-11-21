"""One run at a time on a store, because ThetaData's free tier serves one request at a time.

Two runs at once break the store two ways. ThetaData answers the second of two
concurrent requests with "Too many concurrent requests", so a nightly run's
fetches fail behind a backfill's and it stores a partial session. And a run
reads the manifest, works, and writes it back whole, so the one that finishes
second undoes the other: a backfill writing back the manifest it read hours
earlier rolls the nightly run's newest session and screens back.

So every mode holds the store's lease (:data:`~app.options_layout.LOCK`) for its
whole run. It is created only if absent — GCS's ``if_generation_match=0``, an
exclusive create on disk — and renewed every minute from a background thread;
a lease :data:`TTL_SECONDS` past its last renewal belongs to a run that died,
and the next run takes it over. A run that finds the lease held waits for it.

A backfill holds it for hours, so it gives way. A waiting run that is not a
backfill leaves a request (:data:`~app.options_layout.LOCK_WANTED`) and renews it
while it waits; the backfill checks for one between name-months, releases the
lease, and does not take it back while a fresh request stands — the waiting run
takes it and clears the request, and the backfill carries on after it. A
request left by a run that gave up or died goes stale within
:data:`REQUEST_FRESH_SECONDS`. Backfills do not ask each other to give way: a
second one queues behind the first.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone

from app import options_layout as layout
from collector.store import Store

TTL_SECONDS = 600
RENEW_SECONDS = 60
POLL_SECONDS = 10
#: A request is renewed every poll, so one three polls old was left by a run
#: that stopped waiting.
REQUEST_FRESH_SECONDS = 3 * POLL_SECONDS

#: How long a run waits for the store. A backfill gives way within a
#: name-month, a minute or two, so half an hour means a run that does not give
#: way — a recompute, a long catch-up — or one that hung. A backfill queues
#: behind another backfill, which can take most of a day.
DEFAULT_MAX_WAIT_SECONDS = 30 * 60
MAX_WAIT_SECONDS = {"backfill": 20 * 3600}

#: The one mode that holds the store long enough to be asked to give it up.
_YIELDING = "backfill"


class StoreBusy(Exception):
    """The store's lease could not be had in time. Names the run holding it."""


class LeaseLost(StoreBusy):
    """This run's lease expired and another run took the store; nothing more may be written."""


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


def _parse(data: bytes | None) -> dict:
    try:
        return json.loads(data) if data else {}
    except ValueError:
        return {}


def _owner(mode: str) -> str:
    where = os.environ.get("CLOUD_RUN_EXECUTION") or socket.gethostname()
    return f"{mode} on {where} ({uuid.uuid4().hex[:8]})"


class Lease:
    def __init__(
        self, store: Store, mode: str, *, owner: str | None = None,
        clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print, renew: bool = True,
    ):
        self._store = store
        self.mode = mode
        self.owner = owner or _owner(mode)
        self._clock, self._sleep, self._log = clock, sleep, log
        self._renew = renew
        self._token: object | None = None
        self._since: float | None = None
        self._guard = threading.Lock()  # the token, between the renewer and the run
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.lost = False

    # -- what is in the store ------------------------------------------------

    def _read(self, path: str) -> dict:
        # Through the token read: these objects come and go, and GCS's plain read
        # raises when one is deleted between finding it and downloading it.
        current = self._store.read_with_token(path)
        return _parse(current[0] if current else None)

    def holder(self) -> dict:
        """The lease as stored: ``owner``, ``mode``, ``since``, ``expires_at``; empty when free."""
        return self._read(layout.LOCK)

    def _request(self) -> dict | None:
        request = self._read(layout.LOCK_WANTED)
        if request.get("owner") == self.owner:
            return None
        if request.get("at", 0) < self._clock() - REQUEST_FRESH_SECONDS:
            return None
        return request

    def _payload(self) -> bytes:
        now = self._clock()
        return json.dumps({
            "owner": self.owner, "mode": self.mode, "since": _iso(self._since or now),
            "renewed": _iso(now), "expires_at": now + TTL_SECONDS,
        }).encode()

    # -- taking and giving up the store ----------------------------------------

    def ask(self) -> None:
        """Ask a backfill holding the store to give way; renewed on every poll."""
        self._store.write_json(layout.LOCK_WANTED, {"owner": self.owner, "mode": self.mode, "at": self._clock()})

    def _try(self) -> bool:
        self._since = self._clock()
        token = self._store.create(layout.LOCK, self._payload())
        if token is None:
            current = self._store.read_with_token(layout.LOCK)
            if current is None:
                return False  # released in between: the next poll takes it
            data, held = current
            info = _parse(data)
            if info.get("expires_at", 0) > self._clock():
                return False
            if not self._store.delete(layout.LOCK, held):
                return False  # renewed or taken over in between
            self._log(f"Taking over the store from {info.get('owner')}, whose lease ran out at "
                      f"{_iso(info.get('expires_at', 0))}")
            token = self._store.create(layout.LOCK, self._payload())
            if token is None:
                return False
        with self._guard:
            self._token, self.lost = token, False
        return True

    def acquire(self, max_wait: float | None = None) -> None:
        """Take the store, waiting up to ``max_wait`` seconds; :class:`StoreBusy` after that."""
        if max_wait is None:
            max_wait = MAX_WAIT_SECONDS.get(self.mode, DEFAULT_MAX_WAIT_SECONDS)
        deadline = self._clock() + max_wait
        waiting = False
        while True:
            # A backfill stands aside while another run's request is fresh.
            if not (self.mode == _YIELDING and self._request()) and self._try():
                if self.mode != _YIELDING:
                    self._store.delete(layout.LOCK_WANTED)  # served; another waiter asks again
                self._start_renewing()
                return
            if self.mode != _YIELDING:
                self.ask()
            if self._clock() >= deadline:
                self._withdraw()
                held = self.holder()
                raise StoreBusy(
                    f"The store is in use by {held.get('owner', 'another run')} since {held.get('since', '?')}; "
                    f"gave up after {max_wait / 60:.0f} min. The next run catches up."
                )
            if not waiting:
                self._log(f"Waiting for the store, held by {self.holder().get('owner', 'another run')}")
                waiting = True
            self._sleep(POLL_SECONDS)

    def _withdraw(self) -> None:
        if self.mode != _YIELDING and self._read(layout.LOCK_WANTED).get("owner") == self.owner:
            self._store.delete(layout.LOCK_WANTED)

    def should_give_way(self) -> bool:
        """Whether a run is waiting for the store this backfill holds."""
        return self.mode == _YIELDING and self._request() is not None

    def give_way(self) -> None:
        """Release the store to the waiting run and take it back once that run is done."""
        request = self._request() or {}
        self._log(f"  giving the store to {request.get('owner', 'a waiting run')}")
        self.release()
        self.acquire()

    def ensure(self) -> None:
        if self.lost:
            raise LeaseLost(f"{self.owner} lost the store: its lease ran out and another run took over")

    def release(self) -> None:
        self._stop_renewing()
        with self._guard:
            token, self._token = self._token, None
        if token is not None and not self.lost and not self._store.delete(layout.LOCK, token):
            self._log("The store's lease was no longer this run's when it finished")

    # -- renewal -----------------------------------------------------------

    def renew_once(self, store: Store | None = None) -> None:
        with self._guard:
            if self._token is None or self.lost:
                return
            token = (store or self._store).replace(layout.LOCK, self._payload(), self._token)
            if token is None:
                self.lost = True
                self._log(f"{self.owner} lost the store's lease to another run")
            else:
                self._token = token

    def _start_renewing(self) -> None:
        if not self._renew:
            return
        self._stop.clear()
        store = self._store.reopen()

        def loop() -> None:
            while not self._stop.wait(RENEW_SECONDS):
                try:
                    self.renew_once(store)
                except Exception as error:  # noqa: BLE001 — one missed renewal is not fatal; ten are
                    self._log(f"  could not renew the store's lease: {type(error).__name__}")
                if self.lost:
                    return

        self._thread = threading.Thread(target=loop, name="lease-renewal", daemon=True)
        self._thread.start()

    def _stop_renewing(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None


class GuardedStore(Store):
    """The store as a run under ``lease`` sees it: every write first checks the lease is still its."""

    def __init__(self, store: Store, lease: Lease):
        self._store, self._lease = store, lease

    def read_bytes(self, path: str) -> bytes | None:
        return self._store.read_bytes(path)

    def write_bytes(self, path: str, data: bytes) -> None:
        self._lease.ensure()
        self._store.write_bytes(path, data)

    def list(self, prefix: str) -> list[str]:
        return self._store.list(prefix)
