"""The latest session across every name — what the Discover and Screen tabs draw.

The page reads the newest stored chain for the whole universe once, and this
module turns it into one :class:`NameSnapshot` per name (spot, solved surface,
quotes by expiry) and the two cross-sectional tables. Solving ninety-odd
surfaces takes a second or two; the view caches the result per stored night,
so it is paid once per day, not per rerun.

A name is in a table only when every figure its row needs could be read; the
names that could not are returned beside the table with the reason, so the view
can list them rather than let them vanish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from app.logic.options import chain as C
from app.logic.options import screens, universe
from app.logic.options.history import own_history_stats
from app.logic.options.rates import curve_on
from app.logic.options.score import buy_vol_score
from app.logic.options.surface import Expiry, build_surface
from app.logic.options.tolerances import DEFAULT, OptionsTolerances


@dataclass(frozen=True)
class NameSnapshot:
    symbol: str
    theme: str
    session: date
    spot: float
    surface: list[Expiry]
    quotes: dict[date, pd.DataFrame] = field(repr=False)

    def expiry(self, expiration: date) -> Expiry | None:
        return next((e for e in self.surface if e.expiration == expiration), None)


def build(
    chains: pd.DataFrame,
    metrics: pd.DataFrame,
    rates: pd.DataFrame,
    session: date,
    tolerances: OptionsTolerances = DEFAULT,
) -> tuple[dict[str, NameSnapshot], dict[str, str]]:
    """Every name's snapshot on ``session``, and why each missing name is missing."""
    curve = curve_on(rates, session)
    if curve is None:
        return {}, {symbol: f"no Treasury curve for {session}" for symbol in chains[C.SYMBOL].unique()}
    spots = metrics[metrics["session"] == session].set_index("symbol")["spot"]
    out, missing = {}, {}
    for symbol, quotes in chains[chains[C.SESSION] == session].groupby(C.SYMBOL):
        if symbol not in spots.index or not spots[symbol] > 0:
            missing[symbol] = "no close for the session"
            continue
        surface = build_surface(quotes, session, float(spots[symbol]), curve, tolerances)
        if not surface:
            missing[symbol] = "no expiry had enough two-sided quotes"
            continue
        out[symbol] = NameSnapshot(
            symbol=symbol,
            theme=universe.UNIVERSE.get(symbol, "Other"),
            session=session,
            spot=float(spots[symbol]),
            surface=surface,
            quotes={e: g for e, g in quotes.groupby(C.EXPIRATION)},
        )
    return out, missing


def _percentile(series: pd.Series, tolerances: OptionsTolerances) -> float | None:
    stats = own_history_stats(series, tolerances)
    return None if stats is None else stats.percentile


def discover_table(
    snapshots: dict[str, NameSnapshot], metrics: pd.DataFrame, tolerances: OptionsTolerances = DEFAULT
) -> pd.DataFrame:
    """One row per name: buy-vol score, own-history percentiles, and the 1M 25Δ call."""
    rows = []
    by_symbol = {s: g.set_index("session").sort_index() for s, g in metrics.groupby("symbol")}
    for symbol, snap in snapshots.items():
        history = by_symbol.get(symbol)
        score = buy_vol_score(history, tolerances) if history is not None else None
        one_month = screens.nearest_expiry(snap.surface, 30, tolerances.one_month_days, monthly=True)
        call = (
            screens.delta_call(one_month, snap.quotes[one_month.expiration], snap.session, 0.25, tolerances)
            if one_month is not None
            else None
        )
        rows.append({
            "symbol": symbol,
            "theme": snap.theme,
            "spot": snap.spot,
            "score": score.score if score else None,
            **{f"z {label}": (score.z[label] if score else None) for label in ("IV level", "IV − realized", "25Δ call skew")},
            "skew_pct": _percentile(history["skew_30"], tolerances) if history is not None else None,
            "iv_pct": _percentile(history["atm_iv_30"], tolerances) if history is not None else None,
            "rr_pct": _percentile(history["reward_risk"], tolerances) if history is not None else None,
            "expiration": call.expiration if call else None,
            "strike": call.strike if call else None,
            "delta": call.delta if call else None,
            "iv": call.iv if call else None,
            "cost": call.cost if call else None,
            "cost_pct": call.cost_pct if call else None,
            "annualised": call.annualised if call else None,
            "otm": call.otm if call else None,
            "wide": call.wide if call else None,
        })
    return pd.DataFrame(rows)


def spread_expirations(snapshots: dict[str, NameSnapshot], min_share: float = 0.5) -> list[date]:
    """Expiries listed for at least ``min_share`` of the names, nearest first."""
    counts: dict[date, int] = {}
    for snap in snapshots.values():
        for expiry in snap.surface:
            counts[expiry.expiration] = counts.get(expiry.expiration, 0) + 1
    floor = max(1, int(min_share * len(snapshots)))
    return sorted(e for e, n in counts.items() if n >= floor)


def spread_table(
    snapshots: dict[str, NameSnapshot], expiration: date, sd: float, tolerances: OptionsTolerances = DEFAULT
) -> pd.DataFrame:
    """The +``sd`` spread at ``expiration`` for every name that lists it."""
    rows = []
    for symbol, snap in snapshots.items():
        expiry = snap.expiry(expiration)
        if expiry is None:
            continue
        spread = screens.sd_spread(expiry, snap.quotes[expiration], sd, tolerances)
        if spread is None:
            continue
        rows.append({
            "symbol": symbol,
            "theme": snap.theme,
            "move": spread.move,
            "spread": f"{spread.long_strike:g}/{spread.short_strike:g}C",
            "price": spread.price,
            "width": spread.width,
            "bid": spread.bid,
            "ask": spread.ask,
            "payout_mid": spread.payout_mid,
            "payout_ask": spread.payout_ask,
            "reward_risk": spread.reward_risk,
        })
    return pd.DataFrame(rows)
