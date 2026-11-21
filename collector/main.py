"""``python -m collector.main daily|backfill|repair|recompute|screens`` — the Cloud Run job's entry point.

The store comes from ``--store`` or ``OPTIONS_STORE_URI`` (``gs://bucket`` or
``file:///abs/path``); the ThetaData key from ``THETADATA_API_KEY``, which on
Cloud Run is mounted from Secret Manager. The key is never logged.

Every mode holds the store's lease for its whole run (:mod:`collector.lease`),
so two runs never share ThetaData or the manifest; one that cannot have the
store in time fails, naming the run that holds it, and writes nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.logic.options import freshness, universe
from collector import pipeline
from collector.client import SourceError, ThetaSource
from collector.lease import GuardedStore, Lease, StoreBusy
from collector.store import open_store


#: Modes that read only the store, never ThetaData, so need no key.
_OFFLINE = ("recompute", "screens")


def _today_et() -> date:
    return datetime.now(ZoneInfo("America/New_York")).date()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="collector")
    parser.add_argument("mode", choices=("daily", "backfill", "repair", "recompute", "screens"))
    parser.add_argument("--start", type=date.fromisoformat, help="backfill: first calendar day")
    parser.add_argument("--end", type=date.fromisoformat, help="backfill: last calendar day (default: last session)")
    parser.add_argument("--symbols", help="comma-separated subset of the universe (default: all)")
    parser.add_argument("--sessions", help="repair: comma-separated sessions to fetch the missing names of")
    parser.add_argument("--store", default=os.environ.get("OPTIONS_STORE_URI"))
    args = parser.parse_args(argv)

    if not args.store:
        parser.error("set --store or OPTIONS_STORE_URI")
    key = os.environ.get("THETADATA_API_KEY")
    if not key and args.mode not in _OFFLINE:
        parser.error("THETADATA_API_KEY is not set")
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else list(universe.UNIVERSE)
    unknown = [s for s in symbols if s not in universe.UNIVERSE]
    if unknown:
        parser.error(f"not in the universe: {', '.join(unknown)}")

    sessions = [date.fromisoformat(s.strip()) for s in args.sessions.split(",")] if args.sessions else []
    if args.mode == "repair" and not sessions:
        parser.error("repair needs --sessions")
    if args.mode == "backfill" and not args.start:
        parser.error("backfill needs --start")

    raw = open_store(args.store)
    lease = Lease(raw, args.mode)
    try:
        lease.acquire()
    except StoreBusy as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    try:
        return _run(args, GuardedStore(raw, lease), lease, key, symbols, sessions)
    except (SourceError, StoreBusy) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    finally:
        lease.release()


def _run(args, store, lease: Lease, key: str | None, symbols: list[str], sessions: list[date]) -> int:
    if args.mode == "recompute":
        pipeline.run_recompute(store, symbols)
        return 0
    if args.mode == "screens":
        pipeline.run_screens(store)
        return 0
    source = ThetaSource.connect(key)
    today = _today_et()
    if args.mode == "daily":
        manifest = pipeline.run_daily(source, store, symbols, today)
        print(f"Done: last session {manifest.get('last_session')}, {source.requests} requests")
    elif args.mode == "repair":
        pipeline.run_repair(source, store, symbols, sessions)
    else:
        end = args.end or freshness.expected_session(today)
        pipeline.run_backfill(source, store, symbols, args.start, end, lease=lease)
        print(f"Backfill done, {source.requests} requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
