"""What every 🎯 Options tab is handed: the stored night, solved, and filtered."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.logic.options.snapshot import NameSnapshot

#: Plotly colours per theme, cycled; fixed by sorted theme so a theme keeps its
#: colour when the filter hides others.
PALETTE = (
    "#ef5350", "#42a5f5", "#66bb6a", "#ffa726", "#ab47bc", "#26c6da", "#d4e157", "#ec407a",
    "#8d6e63", "#5c6bc0", "#9ccc65", "#ffca28", "#7e57c2", "#29b6f6", "#ff7043", "#78909c",
)


@dataclass(frozen=True)
class OptionsContext:
    session: date
    snapshots: dict[str, NameSnapshot]
    metrics: pd.DataFrame
    #: Names the filters leave on screen, in universe order.
    visible: list[str]


def theme_colours(themes: list[str]) -> dict[str, str]:
    return {theme: PALETTE[i % len(PALETTE)] for i, theme in enumerate(sorted(themes))}
