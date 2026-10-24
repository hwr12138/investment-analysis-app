"""The snapshot screens: the 25Δ call, the replacement-cost grid, the +k SD spread.

Each reads one symbol's latest chain and reproduces a panel of the page the
feature was specified from. The definitions below were checked against that
page's GLD figures for 2026-10-23 — strike 383 at 3.23 against 3.25 shown, every
grid cell within a few hundredths of a point, the 398/399 spread at 5.7× — so
where a choice looks arbitrary it is the one that matched.

* **Annualised cost** is ``cost % × 252 ÷ trading days to expiry``, simple, not
  compounded.
* **The +k SD target is lognormal**, ``S·e^{kσ√t}``, with σ the expiry's ATM IV.
  The linear ``S·(1 + kσ√t)`` reads +8.5% for GLD where the page showed +9.0%,
  and picks the 396/397 spread instead of 398/399.
* **A spread's price is the difference of its legs' mids**; its bid is floored
  at zero. GLD's 398/399 showed bid 0.00, ask 0.35 and price 0.15 — the mid of a
  floored bid would have been 0.18.

The payout is shown at the ask beside the mid on purpose. At the mid GLD's
spread pays 5.7×; at the ask, the price a buyer actually pays, 1.9×. A screen
ranked on mids alone rewards wide markets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

from app.logic.options import chain as C
from app.logic.options.pricing import black_price, call_spot_delta
from app.logic.options.surface import Expiry, quoted
from app.logic.options.tolerances import DEFAULT, OptionsTolerances

#: The page's replacement-grid rows: upside of the strike over spot.
GRID_UPSIDES = (0.04, 0.09, 0.14, 0.19, 0.29, 0.41)

#: Columns of the grid, as (label, target calendar days, standard monthlies
#: only). 6M and 12M take whatever is listed nearest — far out, a fund's board
#: is mostly quarter-end expiries (GLD's 2027-03-31, 2027-09-30). The two
#: longest listed expiries are appended as LEAPS columns.
GRID_TENORS = (("1M", 30, True), ("3M", 91, True), ("6M", 182, False), ("12M", 365, False))

#: In Spread mode a grid cell is the call spread from the row's strike to this
#: many points of spot higher — +4% to +14%, +9% to +19%.
GRID_SPREAD_WIDTH = 0.10


def trading_days(session: date, expiration: date) -> int:
    """Weekdays between the session and expiry — 20 from Fri 10/23 to Fri 11/20.

    Exchange holidays are not removed; across a month that overstates by at most
    one day, and it is the count that reproduced the page's annualised figures.
    """
    return int(np.busday_count(pd.Timestamp(session).date(), pd.Timestamp(expiration).date())) or 1


def is_monthly(expiration: date) -> bool:
    """A standard monthly expiry: the third Friday, or the Thursday before it on a holiday."""
    expiration = pd.Timestamp(expiration)
    return 15 <= expiration.day <= 21 and expiration.weekday() in (3, 4)


def nearest_expiry(
    surface: list[Expiry], days: int, window: tuple[int, int] | None = None, *, monthly: bool = False
) -> Expiry | None:
    """The expiry nearest ``days`` out, optionally only among standard monthlies.

    The "1M" point is always a monthly. A weekly a few days nearer 30 days is
    often listed on a sparse strike grid — SPX's 11/23 weekly had 25-point
    strikes with a stale 7900 quote — while the monthly beside it is the most
    densely quoted expiry on the board, and it is the same date for every name.
    """
    candidates = [e for e in surface if (window is None or window[0] <= e.t * 365 <= window[1])
                  and (not monthly or is_monthly(e.expiration))]
    if not candidates:
        return None
    return min(candidates, key=lambda e: abs(e.t * 365 - days))


def _wide(bid: float, ask: float, tolerances: OptionsTolerances) -> bool:
    mid = (bid + ask) / 2
    return bid <= 0 or mid <= 0 or (ask - bid) / mid > tolerances.wide_quote_ratio


@dataclass(frozen=True)
class DeltaCall:
    expiration: date
    strike: float
    delta: float
    iv: float
    bid: float
    ask: float
    cost: float
    cost_pct: float
    otm: float
    annualised: float
    wide: bool


def delta_call(
    expiry: Expiry,
    calls: pd.DataFrame,
    session: date,
    target: float = 0.25,
    tolerances: OptionsTolerances = DEFAULT,
) -> DeltaCall | None:
    """The listed call whose delta is nearest ``target``, priced at its mid.

    Each strike's delta uses the smile's IV *at that strike*, so a call that is
    itself unquoted on one side still gets the delta its neighbours imply — but
    it is only chosen if it has a two-sided quote to cost it at.
    """
    two_sided = quoted(calls[calls[C.RIGHT] == C.CALL])
    if two_sided.empty:
        return None
    strikes = two_sided[C.STRIKE].to_numpy()
    ivs = np.array([expiry.iv_at_strike(k) for k in strikes])
    keep = ~np.isnan(ivs)
    if not keep.any():
        return None
    two_sided, strikes, ivs = two_sided[keep], strikes[keep], ivs[keep]
    deltas = call_spot_delta(expiry.forward, strikes, expiry.t, ivs, expiry.df, expiry.spot)
    i = int(np.argmin(np.abs(deltas - target)))
    row = two_sided.iloc[i]
    cost = float(row["mid"])
    cost_pct = cost / expiry.spot
    return DeltaCall(
        expiration=expiry.expiration,
        strike=float(row[C.STRIKE]),
        delta=float(deltas[i]),
        iv=float(ivs[i]),
        bid=float(row[C.BID]),
        ask=float(row[C.ASK]),
        cost=cost,
        cost_pct=cost_pct,
        otm=float(row[C.STRIKE]) / expiry.spot - 1,
        annualised=cost_pct * 252 / trading_days(session, expiry.expiration),
        wide=_wide(float(row[C.BID]), float(row[C.ASK]), tolerances),
    )


def grid_expiries(surface: list[Expiry]) -> list[tuple[str, Expiry]]:
    """The grid's columns: the expiry nearest each tenor, then the two longest."""
    columns: list[tuple[str, Expiry]] = []
    used = set()
    for label, days, monthly in GRID_TENORS:
        expiry = nearest_expiry(surface, days, monthly=monthly)
        if expiry is not None and expiry.expiration not in used:
            columns.append((label, expiry))
            used.add(expiry.expiration)
    for expiry in surface[-2:]:
        if expiry.expiration not in used:
            columns.append((pd.Timestamp(expiry.expiration).strftime("%b%y").upper(), expiry))
            used.add(expiry.expiration)
    return columns


@dataclass(frozen=True)
class GridCell:
    cost_pct: float | None
    annualised: float | None
    wide: bool


def _call_mid_at(calls: pd.DataFrame, strike: float) -> tuple[float, bool] | None:
    """A call's mid at ``strike``, linear between listed strikes; ``None`` outside them.

    The second element says whether either bracketing quote is one-sided or wide.
    """
    calls = calls.sort_values(C.STRIKE)
    strikes = calls[C.STRIKE].to_numpy()
    if not strikes.size or strike < strikes.min() or strike > strikes.max():
        return None
    mids = calls["mid"].to_numpy()
    j = int(np.searchsorted(strikes, strike))
    near = calls.iloc[[max(j - 1, 0), min(j, len(calls) - 1)]]
    return float(np.interp(strike, strikes, mids)), bool(near["wide"].any())


def replacement_grid(
    calls_by_expiry: dict[date, pd.DataFrame],
    columns: list[tuple[str, Expiry]],
    session: date,
    spot: float,
    *,
    spread: bool = False,
    upsides: tuple[float, ...] = GRID_UPSIDES,
    tolerances: OptionsTolerances = DEFAULT,
) -> dict[float, dict[str, GridCell]]:
    """Premium as a % of spot for each (upside, expiry); ``None`` where unquoted."""
    grid: dict[float, dict[str, GridCell]] = {u: {} for u in upsides}
    for label, expiry in columns:
        calls = quoted(calls_by_expiry.get(expiry.expiration, pd.DataFrame(columns=list(C.COLUMNS))))
        calls = calls[calls[C.RIGHT] == C.CALL]
        calls = calls.assign(wide=[_wide(b, a, tolerances) for b, a in zip(calls[C.BID], calls[C.ASK])])
        days = trading_days(session, expiry.expiration)
        for upside in upsides:
            long_leg = _call_mid_at(calls, spot * (1 + upside))
            short_leg = _call_mid_at(calls, spot * (1 + upside + GRID_SPREAD_WIDTH)) if spread else (0.0, False)
            if long_leg is None or short_leg is None:
                grid[upside][label] = GridCell(None, None, False)
                continue
            cost_pct = (long_leg[0] - short_leg[0]) / spot
            grid[upside][label] = GridCell(cost_pct, cost_pct * 252 / days, long_leg[1] or short_leg[1])
    return grid


@dataclass(frozen=True)
class SdSpread:
    expiration: date
    sd: float
    atm_iv: float
    move: float
    long_strike: float
    short_strike: float
    width: float
    price: float
    bid: float
    ask: float
    payout_mid: float
    payout_ask: float | None
    fair_price: float
    fair_payout: float

    @property
    def reward_risk(self) -> float:
        """Mid payout ÷ the payout a flat smile at ATM IV would give: 1.0 is fair."""
        return self.payout_mid / self.fair_payout


def smile_reward_risk(expiry: Expiry, sd: float = 1.0, tolerances: OptionsTolerances = DEFAULT) -> float:
    """Reward:risk at the +``sd`` target read off the smile, not off two quotes.

    A one-strike call spread is a digital, whose price is ``−∂C/∂K``:

        D = DF · (N(d2) − F·φ(d1)·√t · ∂σ/∂K)    at the smile's σ(K)

    against ``DF · N(d2)`` at the ATM IV for a flat smile; each is turned into a
    payout ``(1 − D) ÷ D`` as :func:`sd_spread` does, and this is their ratio.
    σ(K) and ∂σ/∂K come from a straight line through the ``slope_quotes``
    tightly quoted calls nearest the target, so no single stale mid sets the
    figure. :func:`sd_spread` prices the two quotes a reader would trade; this
    prices the same bet from the smile around them, which is what a history can
    compare day to day. On GLD the two-quote version ranged 0.33×–4× over a
    year, mostly as a tick on a 0.15 spread.
    """
    atm = expiry.atm_iv
    if atm != atm:
        return float("nan")
    t, f, df = expiry.t, expiry.forward, expiry.df
    target = expiry.spot * math.exp(sd * atm * math.sqrt(t))
    wing = expiry.smile[(expiry.smile[C.RIGHT] == C.CALL)]
    wing = wing[[not _wide(b, a, tolerances) for b, a in zip(wing[C.BID], wing[C.ASK])]]
    if len(wing) < tolerances.min_slope_quotes:
        return float("nan")
    near = wing.iloc[np.argsort(np.abs(wing[C.STRIKE].to_numpy() - target))[: tolerances.slope_quotes]]
    if not near[C.STRIKE].min() <= target <= near[C.STRIKE].max():
        return float("nan")  # the target sits beyond the quoted wing
    slope, intercept = np.polyfit(near[C.STRIKE].to_numpy(), near["iv"].to_numpy(), 1)
    vol = intercept + slope * target

    def digital(sigma: float, dsigma_dk: float) -> float:
        d1 = (math.log(f / target) + 0.5 * sigma * sigma * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
        return df * (norm.cdf(d2) - f * norm.pdf(d1) * math.sqrt(t) * dsigma_dk)

    market, flat = digital(vol, slope), digital(atm, 0.0)
    if not (0 < market < 1 and 0 < flat < 1):
        return float("nan")
    return ((1 - market) / market) / ((1 - flat) / flat)


def sd_spread(
    expiry: Expiry, calls: pd.DataFrame, sd: float = 1.0, tolerances: OptionsTolerances = DEFAULT
) -> SdSpread | None:
    """The one-strike-wide call spread that pays in full at a +``sd`` move.

    The short strike is the highest tightly quoted strike at or below the
    target, the long strike the next tightly quoted one down, so the whole width
    is earned if the target is reached. A wide quote is skipped rather than
    priced: a spread is the difference of two mids, and one stale mid in it —
    SPX's 7900 call at 15.0/29.3 beside 7950 at 18.7/19.9 — makes the
    difference noise. ``fair_*`` prices the same two strikes at the ATM IV, no
    skew, so :attr:`SdSpread.reward_risk` reads how the market prices the
    spread against a flat smile.
    """
    atm = expiry.atm_iv
    if atm != atm:
        return None
    target = expiry.spot * math.exp(sd * atm * math.sqrt(expiry.t))
    two_sided = quoted(calls[calls[C.RIGHT] == C.CALL]).sort_values(C.STRIKE)
    tight = [not _wide(b, a, tolerances) for b, a in zip(two_sided[C.BID], two_sided[C.ASK])]
    two_sided = two_sided[tight]
    below = two_sided[two_sided[C.STRIKE] <= target]
    if len(below) < 2:
        return None
    short_leg, long_leg = below.iloc[-1], below.iloc[-2]
    width = float(short_leg[C.STRIKE] - long_leg[C.STRIKE])
    price = float(long_leg["mid"] - short_leg["mid"])
    if price <= 0 or price >= width:
        return None
    bid = max(0.0, float(long_leg[C.BID] - short_leg[C.ASK]))
    ask = float(long_leg[C.ASK] - short_leg[C.BID])
    fair = float(
        black_price(expiry.forward, long_leg[C.STRIKE], expiry.t, atm, expiry.df, True)
        - black_price(expiry.forward, short_leg[C.STRIKE], expiry.t, atm, expiry.df, True)
    )
    return SdSpread(
        expiration=expiry.expiration,
        sd=sd,
        atm_iv=atm,
        move=target / expiry.spot - 1,
        long_strike=float(long_leg[C.STRIKE]),
        short_strike=float(short_leg[C.STRIKE]),
        width=width,
        price=price,
        bid=bid,
        ask=ask,
        payout_mid=(width - price) / price,
        payout_ask=(width - ask) / ask if 0 < ask < width else None,
        fair_price=fair,
        fair_payout=(width - fair) / fair if 0 < fair < width else float("nan"),
    )
