"""Where the collector writes and the page reads — the bucket's object names.

A contract between two programs that share nothing else at runtime. It sits at
the ``app/`` root, beside ``provenance``, because the data layer's reader and
the collector (which imports the logic layer) both need it and the data layer
may not import logic. Bumping :data:`PREFIX` is how a change to any
stored shape ships: the collector writes the new prefix alongside the old and
the page moves over when it is deployed.

* ``manifest.json`` is written **last** by every run. A reader that finds a
  manifest can trust every object it names; a run that died part-way leaves the
  previous manifest, and the page keeps showing the previous night, saying so.
* ``latest/chains.parquet`` is the newest session's full chain for every name —
  the only chain the page reads.
* ``latest/discover.parquet``, ``latest/spreads.parquet`` and
  ``latest/expirations.parquet`` are that chain solved into the Discover and
  Screen tables (:func:`app.logic.options.snapshot.solve_screens`), for a reader
  that cannot spend ten seconds solving them per request — the web API. The
  manifest's ``screens.session`` names the close they were solved from, so a
  run of an older collector that advanced the chain without them is caught.
* ``metrics/daily.parquet`` is one :data:`~app.logic.options.history.METRIC_COLUMNS`
  row per symbol per session — every history panel reads only this.
* ``chains/session=…/<SYM>.parquet`` keeps every raw chain fetched, so metrics
  can be recomputed when a definition changes without paying for the fetch again.
"""

from __future__ import annotations

from datetime import date

PREFIX = "options/v1"

MANIFEST = f"{PREFIX}/manifest.json"
LATEST_CHAINS = f"{PREFIX}/latest/chains.parquet"
DISCOVER = f"{PREFIX}/latest/discover.parquet"
SPREADS = f"{PREFIX}/latest/spreads.parquet"
EXPIRATIONS = f"{PREFIX}/latest/expirations.parquet"
METRICS = f"{PREFIX}/metrics/daily.parquet"
EARNINGS = f"{PREFIX}/ref/earnings.parquet"
RATES = f"{PREFIX}/ref/rates.parquet"
CLOSES = f"{PREFIX}/ref/closes.parquet"


CHAINS = f"{PREFIX}/chains/"


def chain_path(session: date, symbol: str) -> str:
    return f"{CHAINS}session={session.isoformat()}/{symbol}.parquet"


def parse_chain_path(path: str) -> tuple[date, str] | None:
    """``(session, symbol)`` from a :func:`chain_path`, or ``None`` for anything else."""
    if not path.startswith(CHAINS) or not path.endswith(".parquet"):
        return None
    folder, _, name = path[len(CHAINS):].partition("/")
    if not folder.startswith("session=") or "/" in name:
        return None
    return date.fromisoformat(folder[len("session="):]), name[: -len(".parquet")]
