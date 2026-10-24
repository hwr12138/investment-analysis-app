"""ThetaData, wrapped for the free tier's rules and normalised to the stored shape.

Everything the free tier imposes, measured on 2026-10-24, is absorbed here:

* **One request at a time.** A second concurrent call fails "Too many concurrent
  requests"; this wrapper is sequential and retries that error with backoff.
* **At most 365 days per request**, and server time grows with the rows asked
  for (about 1 s a session with expiries ≤65 days out, 2.5 s with every
  expiry). Option ranges are fetched a calendar month at a time so each call
  stays near a minute; closes and rates in ≤365-day pieces.
* **No current-day data** ("Cannot fetch current-day data without specifying an
  expiration"): callers never ask for today.
* **An empty answer is an exception** (``NoDataFoundError``), turned into an
  empty frame here so "nothing that day" is not mistaken for an outage.

The ``thetadata`` package is imported only in :meth:`ThetaSource.connect`, so the
tests drive this class with a fake client and run without the package — which
also means without Python 3.12.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd

from app.logic.options import chain as C
from app.logic.options import universe
from app.logic.options.rates import TREASURY_TENORS

#: Status codes worth another attempt; anything else (a permission or argument
#: error) will fail the same way again.
_RETRYABLE_CODES = frozenset({"UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED", "INTERNAL", "UNKNOWN"})
_MAX_ATTEMPTS = 6


class SourceError(Exception):
    """A ThetaData request failed for good. Carries the request, never the key."""


def _code(error: Exception) -> str:
    code = getattr(error, "code", None)
    try:
        value = code() if callable(code) else code
    except Exception:
        return ""
    return getattr(value, "name", str(value or ""))


def _details(error: Exception) -> str:
    details = getattr(error, "details", None)
    try:
        return str(details() if callable(details) else details or error)
    except Exception:
        return str(error)


def chunks(start: date, end: date, *, months: bool = False, max_days: int = 365) -> list[tuple[date, date]]:
    """Inclusive ``(start, end)`` pieces covering the range: calendar months, or ≤ ``max_days``."""
    out = []
    a = start
    while a <= end:
        if months:
            b = (pd.Timestamp(a) + pd.offsets.MonthEnd(0)).date()
        else:
            b = a + timedelta(days=max_days - 1)
        b = min(b, end)
        out.append((a, b))
        a = b + timedelta(days=1)
    return out


def _session(created: pd.Series) -> pd.Series:
    """The trading day a 17:15 ET EOD report belongs to — its local date."""
    stamps = pd.to_datetime(created)
    return stamps.dt.date


def normalise_chain(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """ThetaData's EOD rows → :data:`app.logic.options.chain.COLUMNS`."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(C.COLUMNS))
    out = pd.DataFrame({
        C.SYMBOL: symbol,
        C.SESSION: _session(raw["created"]),
        C.EXPIRATION: pd.to_datetime(raw["expiration"]).dt.date,
        C.STRIKE: pd.to_numeric(raw["strike"], errors="coerce"),
        C.RIGHT: raw["right"].astype(str).str.upper().str[0],
        C.BID: pd.to_numeric(raw["bid"], errors="coerce"),
        C.ASK: pd.to_numeric(raw["ask"], errors="coerce"),
        C.CLOSE: pd.to_numeric(raw["close"], errors="coerce"),
        C.VOLUME: pd.to_numeric(raw["volume"], errors="coerce").fillna(0).astype("int64"),
        C.BID_SIZE: pd.to_numeric(raw["bid_size"], errors="coerce").fillna(0).astype("int64"),
        C.ASK_SIZE: pd.to_numeric(raw["ask_size"], errors="coerce").fillna(0).astype("int64"),
    })
    return out[out[C.RIGHT].isin([C.CALL, C.PUT])].reset_index(drop=True)


class ThetaSource:
    def __init__(self, client, *, sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print):
        self._client = client
        self._sleep = sleep
        self._log = log
        self.requests = 0

    @classmethod
    def connect(cls, api_key: str, **kwargs) -> ThetaSource:
        from thetadata import ThetaClient

        return cls(ThetaClient(api_key=api_key, dataframe_type="pandas"), **kwargs)

    def _call(self, label: str, fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        for attempt in range(_MAX_ATTEMPTS):
            try:
                self.requests += 1
                out = fn()
                return out if out is not None else pd.DataFrame()
            except Exception as error:  # noqa: BLE001 — classified below
                if type(error).__name__ == "NoDataFoundError":
                    return pd.DataFrame()
                details = _details(error)
                retryable = "Too many concurrent" in details or _code(error) in _RETRYABLE_CODES
                if not retryable or attempt == _MAX_ATTEMPTS - 1:
                    raise SourceError(f"{label}: {_code(error) or type(error).__name__} {details[:200]}") from None
                wait = min(60.0, 5.0 * 2**attempt)
                self._log(f"  retry {attempt + 1} for {label} in {wait:.0f}s ({details[:80]})")
                self._sleep(wait)
        raise AssertionError("unreachable")

    def option_chain(self, symbol: str, start: date, end: date, max_dte: int | None = None) -> pd.DataFrame:
        """Every contract's EOD row for ``symbol`` over the range, all roots merged."""
        frames = []
        for root in universe.INDEX_ROOTS.get(symbol, (symbol,)):
            for a, b in chunks(start, end, months=True):
                kwargs = dict(start_date=a, end_date=b, symbol=root, expiration="*")
                if max_dte is not None:
                    kwargs["max_dte"] = max_dte
                raw = self._call(f"option EOD {root} {a}..{b}", lambda kw=kwargs: self._client.option_history_eod(**kw))
                frames.append(normalise_chain(raw, symbol))
        chain = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(C.COLUMNS))
        # One expiry can be listed under two roots (SPXW and SPX); the first root
        # in INDEX_ROOTS wins.
        key = [C.SESSION, C.EXPIRATION, C.STRIKE, C.RIGHT]
        return chain.drop_duplicates(subset=key, keep="first").reset_index(drop=True)

    def closes(self, symbol: str, start: date, end: date) -> pd.Series:
        """Daily closes indexed by session; an index level for :data:`universe.INDICES`."""
        fetch = self._client.index_history_eod if symbol in universe.INDICES else self._client.stock_history_eod
        parts = []
        for a, b in chunks(start, end):
            raw = self._call(f"EOD {symbol} {a}..{b}", lambda a=a, b=b: fetch(symbol=symbol, start_date=a, end_date=b))
            if not raw.empty:
                parts.append(pd.Series(pd.to_numeric(raw["close"], errors="coerce").to_numpy(),
                                       index=_session(raw["created"]).to_numpy()))
        if not parts:
            return pd.Series(dtype=float)
        series = pd.concat(parts)
        series = series[series > 0]
        return series[~series.index.duplicated(keep="last")].sort_index()

    def treasury(self, start: date, end: date) -> pd.DataFrame:
        """Long frame: ``session``, ``tenor`` (years), ``yield_pct``."""
        rows = []
        for name, tenor in TREASURY_TENORS.items():
            for a, b in chunks(start, end):
                raw = self._call(f"rates {name} {a}..{b}",
                                 lambda a=a, b=b, n=name: self._client.interest_rate_history_eod(symbol=n, start_date=a, end_date=b))
                if raw.empty:
                    continue
                rows.append(pd.DataFrame({
                    "session": pd.to_datetime(raw["created"]).dt.date,
                    "tenor": tenor,
                    "yield_pct": pd.to_numeric(raw["rate"], errors="coerce"),
                }))
        if not rows:
            return pd.DataFrame(columns=["session", "tenor", "yield_pct"])
        return pd.concat(rows, ignore_index=True).dropna()
