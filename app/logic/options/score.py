"""The buy-vol score: is buying calls on this name cheap against its own year?

**A judgement, and printed as one.** The page this was specified from showed a
score but not its formula, so this one is ours:

    score = 50 × mean(−z(30d ATM IV), −z(30d IV − 21d realized), −z(30d 25Δ call skew))

each z taken against the name's own trailing trading year. Every component is
inverted so that *higher is cheaper*: implied vol low for this name, implied
low against what the name is actually realizing, and the call wing cheap
against the at-the-money. 50 × a mean z puts a name two standard deviations
cheap on all three at +100.

The components are returned with the score so the page can show which one is
doing the work, and a name with too short a history gets no score rather than
one computed over a few weeks.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.logic.options.tolerances import DEFAULT, OptionsTolerances

SCALE = 50.0

#: (label, how the series is built from a symbol's metrics frame).
COMPONENTS = (
    ("IV level", lambda m: m["atm_iv_30"]),
    ("IV − realized", lambda m: m["atm_iv_30"] - m["rv_21"]),
    ("25Δ call skew", lambda m: m["skew_30"]),
)


@dataclass(frozen=True)
class BuyVolScore:
    score: float
    #: label → the (un-inverted) z-score of today's reading.
    z: dict[str, float]
    sessions: int


def _z(series: pd.Series, tolerances: OptionsTolerances) -> float | None:
    window = series.dropna().sort_index().iloc[-tolerances.history_window:]
    if len(window) < tolerances.min_history_sessions:
        return None
    std = float(window.std(ddof=1))
    if not std > 0:
        return None
    return (float(window.iloc[-1]) - float(window.mean())) / std


def buy_vol_score(metrics: pd.DataFrame, tolerances: OptionsTolerances = DEFAULT) -> BuyVolScore | None:
    """``metrics`` is one symbol's rows indexed by session; ``None`` if any part is unreadable."""
    z = {}
    for label, build in COMPONENTS:
        value = _z(build(metrics), tolerances)
        if value is None:
            return None
        z[label] = value
    sessions = int(metrics["atm_iv_30"].dropna().iloc[-tolerances.history_window:].shape[0])
    return BuyVolScore(score=SCALE * -sum(z.values()) / len(z), z=z, sessions=sessions)
