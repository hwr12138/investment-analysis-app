"""The two runs: nightly catch-up, and the one-time history backfill.

(And the store-only ``recompute`` and ``screens``, and :func:`run_repair`, which
fetches the names a stored session is missing.)

Both end every unit of work the same way — chain files, then the closes and
rates it read, then ``metrics/daily.parquet``, then **the manifest last** — so a
run killed at any point leaves a manifest that names only what was finished.

* :func:`run_daily` fetches every session after ``manifest.last_session`` that
  ThetaData has released (none if the job runs before the free tier's delay is
  over; it simply exits), with every expiry, so the newest session's chains can
  become ``latest/chains.parquet`` — the only chain the page reads.
  It then solves that chain into the Discover and Screen tables
  (:func:`write_screens`), so the web API reads them rather than solving.
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
from app.logic.options import freshness, history, snapshot, universe
from app.logic.options.rates import CURVE_MAX_STALENESS_DAYS, curve_on
from app.logic.options.tolerances import DEFAULT
from collector.client import SourceError, ThetaSource, chunks
from collector.lease import Lease
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


def _stored_metrics(store: Store) -> pd.DataFrame:
    stored = store.read_frame(layout.METRICS)
    return pd.DataFrame(columns=list(history.METRIC_COLUMNS)) if stored is None else stored


def write_screens(
    store: Store, session: date, chains: pd.DataFrame, metrics: pd.DataFrame, rates: pd.DataFrame,
    *, log: Callable[[str], None] = print,
) -> dict:
    """Solves the night's chain into the Discover and Screen tables and stores them.

    Returns the manifest's ``screens`` entry. The caller writes the manifest
    after this, so a reader never sees tables the manifest does not vouch for.
    """
    started = datetime.now(timezone.utc)
    solved = snapshot.solve_screens(chains, metrics, rates, session)
    store.write_frame(layout.DISCOVER, solved.discover)
    store.write_frame(layout.SPREADS, solved.spreads)
    store.write_frame(layout.EXPIRATIONS, solved.listed)
    seconds = (datetime.now(timezone.utc) - started).total_seconds()
    log(f"Screens for {session}: {len(solved.discover)} names priced, {len(solved.unpriced)} not, "
        f"{len(solved.spreads)} spreads ({seconds:.1f}s)")
    return {"session": session.isoformat(), "unpriced": solved.unpriced}


def run_screens(store: Store, *, log: Callable[[str], None] = print) -> dict:
    """Rebuilds the stored screens from the stored night — no ThetaData request.

    For a store a collector without :func:`write_screens` filled, or after a
    change to how a screen is computed. Seconds, where ``recompute`` is minutes.
    """
    manifest = load_manifest(store)
    if not manifest.get("last_session"):
        raise SourceError("The store has no completed daily run to build screens from")
    chains, rates = store.read_frame(layout.LATEST_CHAINS), store.read_frame(layout.RATES)
    if chains is None or rates is None:
        raise SourceError("The store has no latest chain or rates to build screens from")
    session = date.fromisoformat(manifest["last_session"])
    manifest["screens"] = write_screens(store, session, chains, _stored_metrics(store), rates, log=log)
    store.write_json(layout.MANIFEST, manifest)
    return manifest


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
    all_rates = _merge(store, layout.RATES, rates, ["session", "tenor"])
    if rows:
        metrics = _merge(store, layout.METRICS, pd.DataFrame(rows, columns=list(history.METRIC_COLUMNS)), ["symbol", "session"])
    else:
        metrics = _stored_metrics(store)
    latest = pd.concat(newest_chains, ignore_index=True)
    store.write_frame(layout.LATEST_CHAINS, latest)
    screens = write_screens(store, sessions[-1], latest, metrics, all_rates, log=log)
    if _earnings_stale(store, today):
        _refresh_earnings(store, log, today)

    manifest.update(
        metrics_version=history.METRICS_VERSION,
        last_session=sessions[-1].isoformat(),
        screens=screens,
        symbols=status,
        universe_hash=universe.universe_hash(),
        collector_version=COLLECTOR_VERSION,
        written_at=_now(),
    )
    store.write_json(layout.MANIFEST, manifest)
    return manifest


def run_backfill(
    source: ThetaSource, store: Store, symbols, start: date, end: date, *,
    lease: Lease | None = None, log: Callable[[str], None] = print,
) -> dict:
    """Every session from ``start`` to ``end`` for ``symbols``, a calendar month per request.

    Each finished (name, month) is checkpointed in the manifest, so a rerun
    resumes. A backfill takes hours, so under a ``lease`` it gives the store to
    any run waiting for it between name-months (:mod:`collector.lease`), and it
    re-reads the manifest before every write: writing back the one it read at
    its start would undo whatever ran in the meantime.
    """
    done = _backfill_done(store)
    errors: dict[str, str] = {}
    cleared: set[str] = set()

    closes, close_errors = _fetch_closes(source, symbols, start - timedelta(days=_CLOSES_LEAD_DAYS), end, log)
    rates = source.treasury(start - timedelta(days=CURVE_MAX_STALENESS_DAYS + 3), end)
    _merge(store, layout.CLOSES, _closes_frame(closes), ["symbol", "session"])
    _merge(store, layout.RATES, rates, ["session", "tenor"])

    for symbol in symbols:
        if symbol in close_errors:
            errors[symbol] = close_errors[symbol]
            continue
        for a, b in chunks(start, end, months=True):
            key = f"{symbol}:{a:%Y-%m}"
            if key in done:
                continue
            if lease is not None and lease.should_give_way():
                lease.give_way()
                done |= _backfill_done(store)  # another backfill may have had the store meanwhile
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
            cleared.add(key)
            _record_backfill(store, start, end, done, errors, cleared)
            log(f"{key}: {len(rows)} sessions ({source.requests} requests so far)")
    # Once more, so failures after the last finished month are recorded too.
    return _record_backfill(store, start, end, done, errors, cleared)


def _backfill_done(store: Store) -> set[str]:
    return set(load_manifest(store).get("backfill", {}).get("done", []))


def _record_backfill(store: Store, start: date, end: date, done: set[str], errors: dict[str, str], cleared: set[str]) -> dict:
    """Write this backfill's progress into the manifest as it is now, leaving every other key alone."""
    manifest = load_manifest(store)
    progress = manifest.setdefault("backfill", {"start": start.isoformat(), "end": end.isoformat(), "done": []})
    progress["done"] = sorted(set(progress.get("done", [])) | done)
    stored = manifest.setdefault("backfill_errors", {})
    for key in cleared:
        stored.pop(key, None)
    stored.update(errors)
    manifest.update(universe_hash=universe.universe_hash(), collector_version=COLLECTOR_VERSION, written_at=_now())
    store.write_json(layout.MANIFEST, manifest)
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
    latest = store.read_frame(layout.LATEST_CHAINS)
    if manifest.get("last_session") and latest is not None:
        # The Discover table ranks against the history just restated.
        session = date.fromisoformat(manifest["last_session"])
        manifest["screens"] = write_screens(store, session, latest, _stored_metrics(store), rates, log=log)
    manifest.update(metrics_version=history.METRICS_VERSION, written_at=_now())
    store.write_json(layout.MANIFEST, manifest)
    log(f"Recomputed {len(rows)} rows from {len(paths)} stored chains ({problems} could not be read)")
    return manifest


def run_repair(
    source: ThetaSource, store: Store, symbols, sessions, *, log: Callable[[str], None] = print
) -> dict:
    """Fetches again the names a stored session is missing.

    The daily run starts after ``last_session``, so each session is fetched once.
    A run that lands while the free tier is still releasing a session stores only
    the names already out, and no later run asks for the rest: the 01:30 ET runs
    kept 49 of 90 names for 2026-10-27 and 26 for 2026-11-06. This fetches each
    name in ``symbols`` that has no metrics row for a session, as :func:`run_daily`
    would, then rebuilds the screens, whose percentiles read that history.

    It never moves ``last_session`` or ``latest/chains.parquet``: a session after
    ``last_session`` is the daily run's to fetch, and is refused here.
    """
    manifest = load_manifest(store)
    if not manifest.get("last_session"):
        raise SourceError("The store has no completed daily run to repair")
    last = date.fromisoformat(manifest["last_session"])
    sessions = sorted(set(sessions))
    if later := [s.isoformat() for s in sessions if s > last]:
        raise SourceError(f"{', '.join(later)} is after the last stored session, {last}: the daily run fetches it")
    closes_frame, rates = store.read_frame(layout.CLOSES), store.read_frame(layout.RATES)
    if closes_frame is None or rates is None:
        raise SourceError("The store has no closes or rates to repair with")
    closes = {s: g.set_index("session")["close"].sort_index() for s, g in closes_frame.groupby("symbol")}
    calendar = closes.get(CALENDAR_SYMBOL, pd.Series(dtype=float)).index
    if not_sessions := [s.isoformat() for s in sessions if s not in calendar]:
        raise SourceError(f"{', '.join(not_sessions)} has no stored {CALENDAR_SYMBOL} close, so is not a session")

    stored = _stored_metrics(store)
    have = set(zip(stored["symbol"], pd.to_datetime(stored["session"]).dt.date))
    rows, errors = [], {}
    for session in sessions:
        missing = [s for s in symbols if (s, session) not in have]
        log(f"{session}: {len(missing)} of {len(symbols)} names missing")
        for symbol in missing:
            key = f"{symbol}:{session}"
            try:
                chain = source.option_chain(symbol, session, session)
            except SourceError as error:
                errors[key] = str(error)
                continue
            if chain.empty:
                errors[key] = "no option chain"
                continue
            store.write_frame(layout.chain_path(session, symbol), chain)
            row, problem = _metrics_row(symbol, chain, session, closes, rates)
            if row is None:
                errors[key] = problem
                continue
            rows.append(row)

    if rows:
        frame = pd.DataFrame(rows, columns=list(history.METRIC_COLUMNS))
        _merge(store, layout.METRICS, frame, ["symbol", "session"])
        latest = store.read_frame(layout.LATEST_CHAINS)
        if latest is not None:
            manifest["screens"] = write_screens(store, last, latest, _stored_metrics(store), rates, log=log)
    for key, error in errors.items():
        log(f"  {key} still missing: {error}")
    manifest["repair"] = {
        "sessions": [s.isoformat() for s in sessions], "repaired": len(rows), "errors": errors, "at": _now(),
    }
    manifest.update(written_at=_now())
    store.write_json(layout.MANIFEST, manifest)
    log(f"Repaired {len(rows)} name-sessions, {len(errors)} still missing ({source.requests} requests)")
    return manifest


#: Earnings dates are refreshed once they are this old. By age, not by weekday:
#: the job runs Tuesday to Saturday, so a "refresh on Mondays" rule never fired.
EARNINGS_MAX_AGE_DAYS = 6


def _earnings_stale(store: Store, today: date) -> bool:
    frame = store.read_frame(layout.EARNINGS)
    if frame is None or frame.empty or "fetched" not in frame:
        return True
    # Read as either type: a refresh before this was fixed wrote ``fetched`` as a
    # Timestamp, and subtracting that from a date crashed every run from
    # 2026-11-18 on, after its fetch and before its manifest.
    fetched = pd.to_datetime(frame["fetched"]).max().date()
    return (today - fetched).days >= EARNINGS_MAX_AGE_DAYS


def _yahoo_symbol(symbol: str) -> str:
    """Yahoo writes a share class with a hyphen: ``BRK.B`` finds nothing, ``BRK-B`` does."""
    return symbol.replace(".", "-")


def _refresh_earnings(store: Store, log, today: date) -> None:
    """Next earnings date for each single stock, from Yahoo. ETFs have none."""
    try:
        import yfinance as yf
    except ImportError:
        log("  yfinance not installed; earnings dates not refreshed")
        return
    cutoff = pd.Timestamp(today)  # the job's ET date; ``fetched`` stays a date
    rows = []
    for symbol in sorted(universe.SINGLE_STOCKS):
        try:
            dates = yf.Ticker(_yahoo_symbol(symbol)).get_earnings_dates(limit=8)
            stamps = pd.to_datetime(dates.index).tz_localize(None) if dates is not None else []
            upcoming = sorted(t for t in stamps if t >= cutoff)
        except Exception as error:  # noqa: BLE001 — a scrape; recorded, never fatal
            log(f"  earnings for {symbol} failed: {type(error).__name__}")
            continue
        rows.append({"symbol": symbol, "next_earnings": upcoming[0].date() if upcoming else None, "fetched": today})
    if rows:
        store.write_frame(layout.EARNINGS, pd.DataFrame(rows))
