"""The latest session across every name — what the Discover and Screen tabs draw.

The newest stored chain for the whole universe is turned into one
:class:`NameSnapshot` per name (spot, solved surface, quotes by expiry) and the
two cross-sectional tables. Solving every surface took about ten seconds at
ninety names, so it is done once per stored night: the nightly collector solves it
with :func:`solve_screens` and stores the tables (``latest/discover.parquet``
and its neighbours), which the web API reads instead of solving — a request
cannot wait ten seconds.

A name is in a table only when every figure its row needs could be read; the
names that could not are returned beside the table with the reason, so the view
can list them rather than let them vanish.
"""

from __future__ import annotations

from collections.abc import Collection
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

#: The Screen tab's targets: a call spread that pays in full at a +k SD move.
SD_TARGETS = (1.0, 2.0)

#: The Screen tab opens on the first expiry at least this many calendar days
#: out: past the weeklies, where a one-strike spread is mostly noise.
DEFAULT_SPREAD_DAYS = 56

DISCOVER_COLUMNS = (
    "symbol", "theme", "spot", "score", "z IV level", "z IV − realized", "z 25Δ call skew",
    "skew_pct", "iv_pct", "rr_pct",
    "expiration", "strike", "delta", "iv", "cost", "cost_pct", "annualised", "otm", "wide",
)

SPREAD_COLUMNS = (
    "symbol", "theme", "move", "spread", "long_strike", "short_strike", "price", "width", "bid", "ask",
    "payout_mid", "payout_ask", "reward_risk",
)


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
    *,
    symbols: Collection[str] | None = None,
) -> tuple[dict[str, NameSnapshot], dict[str, str]]:
    """Every name's snapshot on ``session``, and why each missing name is missing.

    ``symbols`` solves only those names — one, for a page that shows one.
    """
    rows = chains[chains[C.SESSION] == session]
    if symbols is not None:
        rows = rows[rows[C.SYMBOL].isin(list(symbols))]
    curve = curve_on(rates, session)
    if curve is None:
        return {}, {symbol: f"no Treasury curve for {session}" for symbol in rows[C.SYMBOL].unique()}
    spots = metrics[metrics["session"] == session].set_index("symbol")["spot"]
    out, missing = {}, {}
    for symbol, quotes in rows.groupby(C.SYMBOL):
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
    return pd.DataFrame(rows, columns=list(DISCOVER_COLUMNS))


def listed_expirations(snapshots: dict[str, NameSnapshot]) -> pd.DataFrame:
    """One ``(symbol, expiration)`` row per expiry each name's surface solved."""
    rows = [(symbol, e.expiration) for symbol, snap in snapshots.items() for e in snap.surface]
    return pd.DataFrame(rows, columns=["symbol", "expiration"])


def common_expirations(listed: pd.DataFrame, min_share: float = 0.5) -> list[date]:
    """Expiries listed for at least ``min_share`` of the names in ``listed``, nearest first."""
    names = listed["symbol"].nunique()
    if not names:
        return []
    floor = max(1, int(min_share * names))
    counts = listed.drop_duplicates().groupby("expiration")["symbol"].nunique()
    return sorted(e for e, n in counts.items() if n >= floor)


def default_spread_expiry(choices: list[date], session: date) -> date | None:
    """The first choice at least :data:`DEFAULT_SPREAD_DAYS` out, else the furthest."""
    for expiration in choices:
        if (expiration - session).days >= DEFAULT_SPREAD_DAYS:
            return expiration
    return choices[-1] if choices else None


def _spread_row(snap: NameSnapshot, expiry: Expiry, sd: float, tolerances: OptionsTolerances) -> dict | None:
    spread = screens.sd_spread(expiry, snap.quotes[expiry.expiration], sd, tolerances)
    if spread is None:
        return None
    return {
        "symbol": snap.symbol,
        "theme": snap.theme,
        "move": spread.move,
        "spread": f"{spread.long_strike:g}/{spread.short_strike:g}C",
        "long_strike": spread.long_strike,
        "short_strike": spread.short_strike,
        "price": spread.price,
        "width": spread.width,
        "bid": spread.bid,
        "ask": spread.ask,
        "payout_mid": spread.payout_mid,
        "payout_ask": spread.payout_ask,
        "reward_risk": spread.reward_risk,
    }


def spread_table(
    snapshots: dict[str, NameSnapshot], expiration: date, sd: float, tolerances: OptionsTolerances = DEFAULT
) -> pd.DataFrame:
    """The +``sd`` spread at ``expiration`` for every name that lists it."""
    rows = []
    for snap in snapshots.values():
        expiry = snap.expiry(expiration)
        if expiry is not None and (row := _spread_row(snap, expiry, sd, tolerances)) is not None:
            rows.append(row)
    return pd.DataFrame(rows, columns=list(SPREAD_COLUMNS))


def every_spread(
    snapshots: dict[str, NameSnapshot], sds: tuple[float, ...] = SD_TARGETS, tolerances: OptionsTolerances = DEFAULT
) -> pd.DataFrame:
    """:func:`spread_table` at every expiry any name lists and every target, in one frame.

    Two more columns, ``expiration`` and ``sd``, say which choice a row answers,
    so a reader can select any expiry and target without solving a surface.
    """
    rows = [
        row | {"expiration": expiry.expiration, "sd": sd}
        for snap in snapshots.values()
        for expiry in snap.surface
        for sd in sds
        if (row := _spread_row(snap, expiry, sd, tolerances)) is not None
    ]
    return pd.DataFrame(rows, columns=[*SPREAD_COLUMNS, "expiration", "sd"])


@dataclass(frozen=True)
class Screens:
    """Every cross-sectional table for one stored night — what the collector writes."""

    session: date
    discover: pd.DataFrame
    spreads: pd.DataFrame
    listed: pd.DataFrame
    #: Names with a chain that could not be priced, and why.
    unpriced: dict[str, str]


def solve_screens(
    chains: pd.DataFrame,
    metrics: pd.DataFrame,
    rates: pd.DataFrame,
    session: date,
    tolerances: OptionsTolerances = DEFAULT,
) -> Screens:
    """:func:`build`, then each table the Discover and Screen tabs read."""
    snaps, unpriced = build(chains, metrics, rates, session, tolerances)
    return Screens(
        session=session,
        discover=discover_table(snaps, metrics, tolerances),
        spreads=every_spread(snaps, tolerances=tolerances),
        listed=listed_expirations(snaps),
        unpriced=unpriced,
    )
