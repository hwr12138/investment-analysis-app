"""One symbol's vol surface on one session: forwards, smiles, constant maturity.

Each expiry becomes an :class:`Expiry` — its time, discount factor, the forward
read off put–call parity, and an out-of-the-money smile (puts below the forward,
calls above, each solved from its mid). Everything the screens and the history
read is a lookup on those smiles, so the solve happens once per expiry.

Two rules here are what make the numbers comparable over time:

* **Constant maturity by total variance.** A "30-day IV" read straight off the
  nearest listed expiry drifts between ~20 and ~50 days as expiries roll, so a
  year of it partly measures the calendar. :func:`constant_maturity`
  interpolates σ²·t between the two expiries bracketing 30 days, which is linear
  in time for a flat term structure and never crosses a calendar arbitrage.
* **Constant delta, not constant strike.** The 25Δ call is found by
  interpolating the call wing's IV against its own delta, so "25Δ skew" a year
  ago and today refer to the same place on the smile even though spot moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from app.logic.options import chain as C
from app.logic.options.pricing import call_spot_delta, implied_vol
from app.logic.options.rates import RateCurve
from app.logic.options.tolerances import DEFAULT, OptionsTolerances


def year_fraction(session: date, expiration: date) -> float:
    return (pd.Timestamp(expiration) - pd.Timestamp(session)).days / 365.0


def quoted(quotes: pd.DataFrame) -> pd.DataFrame:
    """The two-sided quotes, with their mid. A zero bid is no market, not a price."""
    two_sided = quotes[(quotes[C.BID] > 0) & (quotes[C.ASK] > 0) & (quotes[C.ASK] >= quotes[C.BID])]
    return two_sided.assign(mid=(two_sided[C.BID] + two_sided[C.ASK]) / 2)


def parity_forward(quotes: pd.DataFrame, spot: float, t: float, df: float, n: int) -> float | None:
    """F = K + (C − P)/DF, the median over the ``n`` paired strikes nearest spot."""
    two_sided = quoted(quotes)
    pairs = two_sided.pivot_table(index=C.STRIKE, columns=C.RIGHT, values="mid", aggfunc="first")
    if C.CALL not in pairs or C.PUT not in pairs:
        return None
    pairs = pairs.dropna(subset=[C.CALL, C.PUT])
    if pairs.empty:
        return None
    nearest = pairs.iloc[np.argsort(np.abs(pairs.index.to_numpy() - spot))[:n]]
    return float(np.median(nearest.index.to_numpy() + (nearest[C.CALL] - nearest[C.PUT]).to_numpy() / df))


@dataclass(frozen=True)
class Expiry:
    """One expiry's smile, solved once.

    ``smile`` carries one row per out-of-the-money strike: ``strike``, ``right``,
    ``bid``, ``ask``, ``mid``, ``iv``, and ``call_delta`` (the delta of the call
    at that strike, also for put rows, so the call wing can be read by delta).
    """

    expiration: date
    t: float
    df: float
    forward: float
    spot: float
    smile: pd.DataFrame

    def iv_at_strike(self, strike: float) -> float:
        """Linear in strike inside the quoted range; NaN outside it."""
        strikes = self.smile[C.STRIKE].to_numpy()
        if not strikes.size or strike < strikes.min() or strike > strikes.max():
            return float("nan")
        return float(np.interp(strike, strikes, self.smile["iv"].to_numpy()))

    @property
    def atm_iv(self) -> float:
        return self.iv_at_strike(self.forward)

    def call_iv_at_delta(self, delta: float) -> float:
        """The call wing's IV at a delta, interpolated against delta; NaN outside it."""
        wing = self.smile[self.smile[C.STRIKE] >= self.forward]
        deltas = wing["call_delta"].to_numpy()[::-1]  # ascending as strike falls
        ivs = wing["iv"].to_numpy()[::-1]
        if deltas.size < 2 or delta < deltas.min() or delta > deltas.max():
            return float("nan")
        return float(np.interp(delta, deltas, ivs))


def build_expiry(
    quotes: pd.DataFrame,
    session: date,
    spot: float,
    curve: RateCurve,
    tolerances: OptionsTolerances = DEFAULT,
) -> Expiry | None:
    """Solves one expiry's smile, or ``None`` when it has too few quotes to have one."""
    if quotes.empty:
        return None
    expiration = quotes[C.EXPIRATION].iloc[0]
    t = year_fraction(session, expiration)
    if t <= 0:
        return None
    df = curve.discount(t)
    forward = parity_forward(quotes, spot, t, df, tolerances.forward_strikes)
    if forward is None:
        return None

    two_sided = quoted(quotes)
    otm = two_sided[
        ((two_sided[C.RIGHT] == C.CALL) & (two_sided[C.STRIKE] >= forward))
        | ((two_sided[C.RIGHT] == C.PUT) & (two_sided[C.STRIKE] < forward))
    ].sort_values(C.STRIKE)
    iv = implied_vol(otm["mid"].to_numpy(), forward, otm[C.STRIKE].to_numpy(), t, df,
                     (otm[C.RIGHT] == C.CALL).to_numpy())
    otm = otm.assign(iv=iv).dropna(subset=["iv"])
    if len(otm) < tolerances.min_otm_quotes:
        return None
    otm = otm.assign(call_delta=call_spot_delta(forward, otm[C.STRIKE].to_numpy(), t, otm["iv"].to_numpy(), df, spot))
    columns = [C.STRIKE, C.RIGHT, C.BID, C.ASK, "mid", "iv", "call_delta"]
    return Expiry(expiration, t, df, forward, spot, otm[columns].reset_index(drop=True))


def build_surface(
    quotes: pd.DataFrame,
    session: date,
    spot: float,
    curve: RateCurve,
    tolerances: OptionsTolerances = DEFAULT,
    *,
    max_days: int | None = None,
) -> list[Expiry]:
    """Every expiry of one symbol on one session that has a smile, nearest first."""
    out = []
    for expiration, group in quotes.groupby(C.EXPIRATION, sort=True):
        if max_days is not None and year_fraction(session, expiration) * 365 > max_days:
            continue
        expiry = build_expiry(group, session, spot, curve, tolerances)
        if expiry is not None:
            out.append(expiry)
    return out


def constant_maturity(
    surface: list[Expiry], read, tolerances: OptionsTolerances = DEFAULT, *, variance: bool = True
) -> float:
    """A figure read at a fixed maturity by interpolating between two expiries.

    ``read`` maps an :class:`Expiry` to a vol (its ATM IV, its 25Δ call IV …),
    interpolated in total variance; with ``variance=False`` it may be any figure
    (a ratio), interpolated linearly in time.
    Needs an expiry on each side of the target inside the bracket window; with
    only one side it is NaN rather than the nearest expiry's value, because the
    nearest expiry is precisely the drifting number this exists to avoid.
    """
    target = tolerances.constant_maturity_days / 365.0
    lo_bound = tolerances.min_bracket_days / 365.0
    hi_bound = tolerances.max_bracket_days / 365.0
    points = [(e.t, read(e)) for e in surface if lo_bound <= e.t <= hi_bound]
    points = [(t, v) for t, v in points if v == v]
    below = [p for p in points if p[0] <= target]
    above = [p for p in points if p[0] > target]
    if not below or not above:
        return float("nan")
    (t1, v1), (t2, v2) = below[-1], above[0]
    weight = (target - t1) / (t2 - t1)
    if not variance:
        return float((1 - weight) * v1 + weight * v2)
    total = (1 - weight) * v1 * v1 * t1 + weight * v2 * v2 * t2
    return float(np.sqrt(total / target))
