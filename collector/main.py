"""``python -m collector.main daily|backfill|recompute|screens`` — the Cloud Run job's entry point.

The store comes from ``--store`` or ``OPTIONS_STORE_URI`` (``gs://bucket`` or
``file:///abs/path``); the ThetaData key from ``THETADATA_API_KEY``, which on
Cloud Run is mounted from Secret Manager. The key is never logged.
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
from collector.store import open_store


#: Modes that read only the store, never ThetaData, so need no key.
_OFFLINE = ("recompute", "screens")


def _today_et() -> date:
    return datetime.now(ZoneInfo("America/New_York")).date()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="collector")
    parser.add_argument("mode", choices=("daily", "backfill", "recompute", "screens"))
    parser.add_argument("--start", type=date.fromisoformat, help="backfill: first calendar day")
    parser.add_argument("--end", type=date.fromisoformat, help="backfill: last calendar day (default: last session)")
    parser.add_argument("--symbols", help="comma-separated subset of the universe (default: all)")
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

    store = open_store(args.store)
    if args.mode in _OFFLINE:
        try:
            if args.mode == "recompute":
                pipeline.run_recompute(store, symbols)
            else:
                pipeline.run_screens(store)
        except SourceError as error:
            print(f"FAILED: {error}", file=sys.stderr)
            return 1
        return 0
    source = ThetaSource.connect(key)
    today = _today_et()
    try:
        if args.mode == "daily":
            manifest = pipeline.run_daily(source, store, symbols, today)
            print(f"Done: last session {manifest.get('last_session')}, {source.requests} requests")
        else:
            if not args.start:
                parser.error("backfill needs --start")
            end = args.end or freshness.expected_session(today)
            pipeline.run_backfill(source, store, symbols, args.start, end)
            print(f"Backfill done, {source.requests} requests")
    except SourceError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
