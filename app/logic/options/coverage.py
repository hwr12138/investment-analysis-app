"""Which names the 🎯 Options page shows, and why each of the others is not on it.

Two kinds of absence, and the page says which: a name the **filters** hide (a
theme left out, earnings too close) is the reader's choice; a name that
**failed** — its fetch errored, or its chain could not be priced — or was **not
collected** at all is listed with its reason, never dropped quietly.

Shared by both front ends: the Streamlit shell filters its own solved
snapshots, the web API filters the collector's stored tables, and both pass the
same theme lookup through :func:`visible`.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import date

import pandas as pd

from app.logic.options import universe

#: The earnings filter's windows, in calendar days after the session. A single
#: stock reporting within the window is hidden: its calls price the event, not
#: the name. ``0`` hides nothing.
EARNINGS_WINDOWS = (0, 7, 14, 21)
DEFAULT_EARNINGS_WINDOW = 21


def hidden_for_earnings(earnings: pd.DataFrame | None, session: date, days: int) -> set[str]:
    """Names whose next earnings date falls ``0..days`` calendar days after ``session``."""
    if not days or earnings is None or earnings.empty:
        return set()
    hidden = set()
    for _, item in earnings.iterrows():
        upcoming = item.get("next_earnings")
        if upcoming is None or pd.isna(upcoming):
            continue
        if 0 <= (pd.Timestamp(upcoming).date() - session).days <= days:
            hidden.add(item["symbol"])
    return hidden


def visible(
    themes: Mapping[str, str], chosen: Collection[str] | None, hidden: Collection[str]
) -> list[str]:
    """The priced names the filters leave on screen, in universe order.

    ``themes`` maps every priced name to its theme; ``chosen`` is the themes
    kept, ``None`` for all of them.
    """
    return [
        symbol for symbol in universe.UNIVERSE
        if symbol in themes and (chosen is None or themes[symbol] in chosen) and symbol not in hidden
    ]


def unavailable(manifest: Mapping, unpriced: Mapping[str, str]) -> dict[str, str]:
    """Collected names that are not on the page, with why, sorted by name.

    A failed fetch is the manifest's record of the run; a chain that came back
    but could not be priced is the solve's (``unpriced``).
    """
    problems = {
        symbol: status.get("error")
        for symbol, status in manifest.get("symbols", {}).items()
        if status.get("error")
    }
    problems.update(unpriced)
    return dict(sorted(problems.items()))


def not_collected(manifest: Mapping) -> list[str]:
    """Universe names the last run has no record of — added since, say."""
    collected = manifest.get("symbols", {})
    return [symbol for symbol in universe.UNIVERSE if symbol not in collected]
