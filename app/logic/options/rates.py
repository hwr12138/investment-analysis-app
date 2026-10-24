"""The discount curve an option is priced off, from the Treasury curve.

The collector stores ThetaData's daily Treasury yields (``TREASURY_M1`` …
``TREASURY_Y30``) as published: percent, bond-equivalent. A LEAP two years out
discounts at the two-year rate, not the T-bill, which on a 2028 call moves the
implied vol by a visible amount — hence a curve rather than one rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

#: ThetaData's Treasury symbols and their tenors in years.
TREASURY_TENORS = {
    "TREASURY_M1": 1 / 12,
    "TREASURY_M3": 0.25,
    "TREASURY_M6": 0.5,
    "TREASURY_Y1": 1.0,
    "TREASURY_Y2": 2.0,
    "TREASURY_Y3": 3.0,
    "TREASURY_Y5": 5.0,
}


@dataclass(frozen=True)
class RateCurve:
    """Continuously compounded zero rates by tenor, interpolated linearly.

    Beyond either end the nearest tenor's rate is held flat. That is a
    statement about the curve's shape over a few months, not a fill of missing
    data: the shortest tenor is one month and a weekly option still needs a rate.
    """

    tenors: tuple[float, ...]
    rates: tuple[float, ...]

    @classmethod
    def from_yields(cls, yields_pct: dict[float, float]) -> RateCurve:
        """Builds the curve from bond-equivalent yields in percent, keyed by tenor."""
        points = sorted((t, math.log1p(y / 100.0)) for t, y in yields_pct.items() if y == y)
        if not points:
            raise ValueError("A rate curve needs at least one tenor")
        return cls(tuple(t for t, _ in points), tuple(r for _, r in points))

    def rate(self, t: float) -> float:
        return float(np.interp(t, self.tenors, self.rates))

    def discount(self, t: float) -> float:
        return math.exp(-self.rate(t) * t)


#: A Treasury curve older than this on a session is not used for it. ThetaData
#: fills weekends and bond holidays itself, so this only trips on a real gap.
CURVE_MAX_STALENESS_DAYS = 7


def curve_on(rates: pd.DataFrame, session: date) -> RateCurve | None:
    """The curve published on or before ``session``, from the stored long frame.

    ``rates`` has ``session``, ``tenor`` (years) and ``yield_pct`` columns.
    ``None`` when nothing was published within :data:`CURVE_MAX_STALENESS_DAYS`.
    """
    if rates is None or rates.empty:
        return None
    window = rates[(rates["session"] <= session) & (rates["session"] >= session - timedelta(days=CURVE_MAX_STALENESS_DAYS))]
    if window.empty:
        return None
    latest = window[window["session"] == window["session"].max()]
    return RateCurve.from_yields(dict(zip(latest["tenor"], latest["yield_pct"])))
