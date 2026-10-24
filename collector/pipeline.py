"""The two runs: nightly catch-up, and the one-time history backfill.

Both end every unit of work the same way — chain files, then the closes and
rates it read, then ``metrics/daily.parquet``, then **the manifest last** — so a
run killed at any point leaves a manifest that names only what was finished.

* :func:`run_daily` fetches every session after ``manifest.last_session`` that
  ThetaData has released (none if the job runs before the free tier's delay is
  over; it simply exits), with every expiry, so the newest session's chains can
  become ``latest/chains.parquet`` — the only chain the page reads.
* :func:`run_backfill` walks one symbol one calendar month at a time with
  expiries ≤65 days out (what the 30-day metrics need; about half the server
  time of the full chain) and checkpoints each finished (symbol, month) in the
  manifest, so a rerun after a timeout resumes rather than restarts.

**The session calendar is SPY's closes**: a weekday SPY has no close for is not
a session. A name that fails on a session is recorded in the manifest with the
reason and the others still write; the page lists it as missing rather than
silently dropping it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from app import options_layout as layout
from app.logic.options import chain as C
from app.logic.options import freshness, history, universe
from app.logic.options.rates import CURVE_MAX_STALENESS_DAYS, curve_on
from app.logic.options.tolerances import DEFAULT
from collector.client import SourceError, ThetaSource, chunks
from collector.store import Store

COLLECTOR_VERSION = "1"
CALENDAR_SYMBOL = "SPY"

#: Expiries the backfill asks for. The 30-day bracket and the 1M expiry need
#: no more, and it roughly halves ThetaData's server time against every expiry.
BACKFILL_MAX_DTE = 65

#: Calendar days of closes fetched before the first session, for the 21-session
#: realized vol.
_CLOSES_LEAD_DAYS = 45



def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_manifest(store: Store) -> dict:
    return store.read_json(layout.MANIFEST) or {}


def _merge(store: Store, path: str, new: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    old = store.read_frame(path)
    merged = new if old is None or old.empty else pd.concat([old, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=key, keep="last").sort_values(key).reset_index(drop=True)
    store.write_frame(path, merged)
    return merged


def _closes_frame(closes: dict[str, pd.Series]) -> pd.DataFrame:
    rows = [
        pd.DataFrame({"symbol": symbol, "session": series.index, "close": series.to_numpy()})
        for symbol, series in closes.items()
        if not series.empty
    ]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["symbol", "session", "close"])


def _metrics_row(symbol, chain, session, closes, rates) -> tuple[dict | None, str | None]:
    series = closes.get(symbol, pd.Series(dtype=float))
    if session not in series.index:
        return None, f"no close for {symbol} on {session}"
    curve = curve_on(rates, session)
    if curve is None:
        return None, f"no Treasury curve within {CURVE_MAX_STALENESS_DAYS} days of {session}"
    upto = series[series.index <= session]
    return history.session_metrics(symbol, chain, session, float(series[session]), curve, upto, DEFAULT), None


def _fetch_closes(source: ThetaSource, symbols, start: date, end: date, log) -> tuple[dict[str, pd.Series], dict]:
    closes, errors = {}, {}
    for symbol in sorted(set(symbols) | {CALENDAR_SYMBOL}):
        try:
            closes[symbol] = source.closes(symbol, start, end)
        except SourceError as error:
            errors[symbol] = str(error)
            log(f"  closes failed for {symbol}: {error}")
    return closes, errors


def run_daily(source: ThetaSource, store: Store, symbols, today: date, *, log: Callable[[str], None] = print) -> dict:
    manifest = load_manifest(store)
    expected = freshness.expected_session(today)
    last = date.fromisoformat(manifest["last_session"]) if manifest.get("last_session") else None
    start = last + timedelta(days=1) if last else expected
    if start > expected:
        log(f"Up to date through {last}.")
        return manifest

    closes, close_errors = _fetch_closes(source, symbols, start - timedelta(days=_CLOSES_LEAD_DAYS), expected, log)
    if CALENDAR_SYMBOL in close_errors:
        raise SourceError(f"The session calendar ({CALENDAR_SYMBOL} closes) could not be read")
    sessions = [d for d in closes[CALENDAR_SYMBOL].index if start <= d <= expected]
    if not sessions:
        log(f"No session after {last} has been released yet.")
        return manifest
    rates = source.treasury(start - timedelta(days=CURVE_MAX_STALENESS_DAYS + 3), expected)

    status: dict[str, dict] = {}
    rows, newest_chains = [], []
    for session in sessions:
        for symbol in symbols:
            try:
                chain = source.option_chain(symbol, session, session)
            except SourceError as error:
                status[symbol] = {"session": session.isoformat(), "error": str(error)}
                continue
            if chain.empty:
                status[symbol] = {"session": session.isoformat(), "error": "no option chain"}
                continue
            store.write_frame(layout.chain_path(session, symbol), chain)
            row, problem = _metrics_row(symbol, chain, session, closes, rates)
            if row is not None:
                rows.append(row)
            status[symbol] = {"session": session.isoformat(), "error": problem or close_errors.get(symbol)}
            if session == sessions[-1]:
                newest_chains.append(chain)
        log(f"{session}: {sum(1 for s in status.values() if s['session'] == session.isoformat() and not s['error'])} names")

    if not newest_chains:
        raise SourceError(f"No name returned a chain for {sessions[-1]}; the manifest was not advanced")
    _merge(store, layout.CLOSES, _closes_frame(closes), ["symbol", "session"])
    _merge(store, layout.RATES, rates, ["session", "tenor"])
    if rows:
        _merge(store, layout.METRICS, pd.DataFrame(rows, columns=list(history.METRIC_COLUMNS)), ["symbol", "session"])
    store.write_frame(layout.LATEST_CHAINS, pd.concat(newest_chains, ignore_index=True))
    if today.weekday() == 0 or store.read_bytes(layout.EARNINGS) is None:
        _refresh_earnings(store, log)

    manifest.update(
        metrics_version=history.METRICS_VERSION,
        last_session=sessions[-1].isoformat(),
        symbols=status,
        universe_hash=universe.universe_hash(),
        collector_version=COLLECTOR_VERSION,
        written_at=_now(),
    )
    store.write_json(layout.MANIFEST, manifest)
    return manifest


def run_backfill(
    source: ThetaSource, store: Store, symbols, start: date, end: date, *, log: Callable[[str], None] = print
) -> dict:
    manifest = load_manifest(store)
    progress = manifest.setdefault("backfill", {"start": start.isoformat(), "end": end.isoformat(), "done": []})
    done = set(progress["done"])

    closes, close_errors = _fetch_closes(source, symbols, start - timedelta(days=_CLOSES_LEAD_DAYS), end, log)
    rates = source.treasury(start - timedelta(days=CURVE_MAX_STALENESS_DAYS + 3), end)
    _merge(store, layout.CLOSES, _closes_frame(closes), ["symbol", "session"])
    _merge(store, layout.RATES, rates, ["session", "tenor"])
    errors = manifest.setdefault("backfill_errors", {})

    for symbol in symbols:
        if symbol in close_errors:
            errors[symbol] = close_errors[symbol]
            continue
        for a, b in chunks(start, end, months=True):
            key = f"{symbol}:{a:%Y-%m}"
            if key in done:
                continue
            try:
                raw = source.option_chain(symbol, a, b, max_dte=BACKFILL_MAX_DTE)
            except SourceError as error:
                errors[key] = str(error)
                log(f"  {key} failed: {error}")
                continue
            rows = []
            for session, chain in raw.groupby(C.SESSION):
                store.write_frame(layout.chain_path(session, symbol), chain)
                row, problem = _metrics_row(symbol, chain, session, closes, rates)
                if row is not None:
                    rows.append(row)
                elif problem:
                    errors[f"{symbol}:{session}"] = problem
            if rows:
                _merge(store, layout.METRICS, pd.DataFrame(rows, columns=list(history.METRIC_COLUMNS)), ["symbol", "session"])
            done.add(key)
            errors.pop(key, None)
            progress["done"] = sorted(done)
            manifest.update(universe_hash=universe.universe_hash(), collector_version=COLLECTOR_VERSION, written_at=_now())
            store.write_json(layout.MANIFEST, manifest)
            log(f"{key}: {len(rows)} sessions ({source.requests} requests so far)")
    return manifest


def run_recompute(store: Store, symbols, *, log: Callable[[str], None] = print) -> dict:
    """Rebuilds every metrics row from the stored chains — no ThetaData request.

    For when a metric's definition changes (:data:`history.METRICS_VERSION`):
    the raw chains, closes and rates are all in the store, so the history can be
    restated without paying for the fetch again.
    """
    closes_frame = store.read_frame(layout.CLOSES)
    rates = store.read_frame(layout.RATES)
    if closes_frame is None or rates is None:
        raise SourceError("The store has no closes or rates to recompute from")
    closes = {s: g.set_index("session")["close"].sort_index() for s, g in closes_frame.groupby("symbol")}
    wanted = set(symbols)
    paths = [(p, parsed) for p in store.list(layout.CHAINS) if (parsed := layout.parse_chain_path(p))]
    rows, problems = [], 0
    for path, (session, symbol) in paths:
        if symbol not in wanted:
            continue
        row, problem = _metrics_row(symbol, store.read_frame(path), session, closes, rates)
        if row is not None:
            rows.append(row)
        elif problem:
            problems += 1
    if rows:
        frame = pd.DataFrame(rows, columns=list(history.METRIC_COLUMNS))
        _merge(store, layout.METRICS, frame, ["symbol", "session"])
    manifest = load_manifest(store)
    manifest.update(metrics_version=history.METRICS_VERSION, written_at=_now())
    store.write_json(layout.MANIFEST, manifest)
    log(f"Recomputed {len(rows)} rows from {len(paths)} stored chains ({problems} could not be read)")
    return manifest


def _refresh_earnings(store: Store, log) -> None:
    """Next earnings date for each single stock, from Yahoo. ETFs have none."""
    try:
        import yfinance as yf
    except ImportError:
        log("  yfinance not installed; earnings dates not refreshed")
        return
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    rows = []
    for symbol in sorted(universe.SINGLE_STOCKS):
        try:
            dates = yf.Ticker(symbol).get_earnings_dates(limit=8)
            stamps = pd.to_datetime(dates.index).tz_localize(None) if dates is not None else []
            upcoming = sorted(t for t in stamps if t >= today)
        except Exception as error:  # noqa: BLE001 — a scrape; recorded, never fatal
            log(f"  earnings for {symbol} failed: {type(error).__name__}")
            continue
        rows.append({"symbol": symbol, "next_earnings": upcoming[0].date() if upcoming else None})
    if rows:
        store.write_frame(layout.EARNINGS, pd.DataFrame(rows))
