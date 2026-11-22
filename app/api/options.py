"""🎯 Options over HTTP: one stored night, read rather than solved.

Solving every name's surface takes about ten seconds, longer than a request
can wait on a cold instance, so the nightly collector solves the night once
(:func:`app.logic.options.snapshot.solve_screens`) and stores the tables; the
overview, Discover and Screen routes read those. Evaluate and History draw one
name each, and solve or read just that one when asked.

The filters travel with every route that lists names — ``themes`` (repeated,
omitted for every theme) and ``earnings_window`` (calendar days; 0 hides
nothing) — so a URL names the whole request, as the other pages' do.

**Never silently stale.** Every response names the session it is from, the
overview says how many sessions behind the market that is, and stored tables
solved from an older close than the manifest names are refused, not served:
an older collector that advanced the chain without them would otherwise leave
yesterday's screens under today's date.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query, status

from app import options_layout as layout
from app.api.contract import Contract, finite_or_none
from app.data import options_store
from app.logic.options import coverage, freshness, screens, snapshot, universe
from app.logic.options.history import own_history_stats, reward_risk_quartile
from app.logic.options.score import SCALE
from app.logic.options.tolerances import DEFAULT

router = APIRouter(prefix="/options", tags=["options"])

Themes = Annotated[
    list[str] | None,
    Query(max_length=64, description="Themes to keep; omit for every theme."),
]
EarningsWindow = Annotated[
    int,
    Query(
        ge=0,
        le=90,
        description="Hide single stocks reporting within this many calendar days of the session; 0 hides none.",
    ),
]
Symbol = Annotated[str, Path(pattern=r"^[A-Za-z]{1,6}$")]


def _today_et() -> date:
    return datetime.now(ZoneInfo("America/New_York")).date()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class Name(Contract):
    symbol: str
    theme: str
    #: Single stocks only, and only when the collector's scrape found one.
    next_earnings: date | None


class Unavailable(Contract):
    symbol: str
    reason: str


class ThemeGroup(Contract):
    theme: str
    symbols: list[str]


class OptionsOverview(Contract):
    #: The close every figure on the page is from.
    session: date
    written_at: datetime | None
    #: Weekdays after ``session`` the market has closed; above one, the
    #: collector has stopped and the page must say so.
    sessions_behind: int
    #: Themes of the priced names, sorted.
    themes: list[str]
    #: Every priced name, in universe order.
    names: list[Name]
    #: What the filters leave on screen, in universe order.
    visible: list[str]
    #: Names the theme filter keeps that the earnings window hides.
    hidden_for_earnings: list[str]
    #: ``False`` when the earnings dates could not be read, so none is hidden.
    earnings_known: bool
    #: Collected names that are not on the page, with why.
    unavailable: list[Unavailable]
    #: Universe names the last run has no record of.
    not_collected: list[str]
    universe: list[ThemeGroup]
    #: Stored sessions a score or percentile needs.
    min_history_sessions: int
    score_scale: float


class OneMonthCall(Contract):
    """The listed call nearest 25Δ on the standard monthly nearest 30 days."""

    expiration: date
    strike: float
    delta: float
    iv: float
    #: Mid of the 17:15 ET NBBO.
    cost: float
    cost_pct: float
    #: ``cost_pct`` × 252 ÷ weekdays to expiry.
    annualised: float
    otm: float
    #: The bid/ask is one-sided or wider than half its mid.
    wide: bool


class DiscoverRow(Contract):
    symbol: str
    theme: str
    spot: float
    #: ``null`` under ``min_history_sessions`` of stored history, like each
    #: z-score and percentile.
    score: float | None
    z_iv: float | None
    z_iv_rv: float | None
    z_skew: float | None
    #: Percentiles of the name's own trailing year, 0–1.
    skew_pct: float | None
    iv_pct: float | None
    rr_pct: float | None
    call: OneMonthCall | None


class OptionsDiscover(Contract):
    session: date
    #: The visible names, highest score first, unscored last.
    rows: list[DiscoverRow]


class Spread(Contract):
    symbol: str
    theme: str
    #: The +k SD target over spot: e^(k·σ·√t) − 1 at the expiry's ATM IV.
    move: float
    long_strike: float
    short_strike: float
    #: The legs' mid difference.
    price: float
    width: float
    #: Floored at zero.
    bid: float
    ask: float
    payout_mid: float
    #: ``null`` where the ask is zero or the whole width.
    payout_ask: float | None
    #: Mid payout ÷ the payout at a flat smile; 1.0 is fair.
    reward_risk: float | None


class OptionsScreen(Contract):
    session: date
    #: The targets the collector solves, in SDs; ``sd`` is one of them.
    sds: list[float]
    sd: float
    #: Expiries listed for at least half the visible names, nearest first.
    expirations: list[date]
    #: The expiry shown: the one asked for when it is a choice, else the
    #: first at least eight weeks out. ``null`` when there is no choice.
    expiration: date | None
    #: Highest payout at the ask first.
    rows: list[Spread]
    visible_count: int


class GridColumn(Contract):
    label: str
    expiration: date


class GridCell(Contract):
    #: Premium as a fraction of spot; ``null`` outside the quoted strikes.
    cost_pct: float | None
    annualised: float | None
    #: A bracketing quote is one-sided or wider than half its mid.
    wide: bool


class GridRow(Contract):
    #: The long strike over spot.
    upside: float
    #: One per column, in ``columns`` order.
    cells: list[GridCell]


class OptionsEvaluate(Contract):
    symbol: str
    theme: str
    session: date
    spot: float
    columns: list[GridColumn]
    #: The call at spot × (1 + upside).
    single: list[GridRow]
    #: The call spread from that strike to ``spread_width`` of spot higher.
    spread: list[GridRow]
    spread_width: float


class HistoryPoint(Contract):
    session: date
    reward_risk: float | None
    atm_iv: float | None
    #: 25Δ call IV − ATM IV, at 30 days.
    skew: float | None


class RewardRiskStats(Contract):
    latest: float
    as_of: date
    #: Share of the window at or below ``latest``, 0–1.
    percentile: float
    quartile: str
    median: float
    q25: float
    q75: float
    low: float
    high: float
    sessions: int


class OptionsHistory(Contract):
    symbol: str
    theme: str
    session: date
    #: The trailing trading year, oldest first.
    points: list[HistoryPoint]
    #: ``null`` until ``needed`` sessions are stored.
    stats: RewardRiskStats | None
    stored: int
    needed: int


# ---------------------------------------------------------------------------
# Reading the store
# ---------------------------------------------------------------------------


def _unavailable(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail)


def _night() -> tuple[dict, date]:
    manifest = options_store.fetch_manifest()
    if not manifest.get("last_session"):
        raise _unavailable("The options collector has not completed a daily run yet.")
    return manifest, date.fromisoformat(manifest["last_session"])


def _screens(manifest: dict, session: date) -> dict:
    """The manifest's record of the stored tables, refused unless it is this night's."""
    stored = manifest.get("screens") or {}
    if stored.get("session") != session.isoformat():
        built = f" (the stored ones are from {stored['session']})" if stored.get("session") else ""
        raise _unavailable(
            f"The Discover and Screen tables have not been built for the {session} close{built}. "
            "The nightly collector builds them; `python -m collector.main screens` builds them "
            "from the stored night."
        )
    return stored


def _earnings() -> pd.DataFrame | None:
    try:
        return options_store.fetch_frame(layout.EARNINGS)
    except options_store.OptionsStoreError:
        return None


def _check_themes(themes: list[str] | None) -> set[str] | None:
    if themes is None:
        return None
    unknown = sorted(set(themes) - set(universe.themes()))
    if unknown:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Not a theme: {', '.join(unknown)}."
        )
    return set(themes)


def _filtered(
    discover: pd.DataFrame, earnings: pd.DataFrame | None, session: date,
    themes: set[str] | None, window: int,
) -> tuple[list[str], list[str]]:
    """The visible names, and the ones the earnings window took out of the themes kept."""
    names = coverage.themes_of(discover["symbol"])
    hidden = coverage.hidden_for_earnings(earnings, session, window)
    kept = coverage.visible(names, themes, ())
    return coverage.visible(names, themes, hidden), [s for s in kept if s in hidden]


def _known_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    if symbol not in universe.UNIVERSE:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{symbol} is not in the options universe.")
    return symbol


def _date_or_none(value) -> date | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/overview")
def overview(
    themes: Themes = None, earnings_window: EarningsWindow = coverage.DEFAULT_EARNINGS_WINDOW
) -> OptionsOverview:
    """What the stored night holds, how old it is, and which names the filters leave."""
    chosen = _check_themes(themes)
    manifest, session = _night()
    stored = _screens(manifest, session)
    discover = options_store.fetch_frame(layout.DISCOVER)
    earnings = _earnings()
    visible, hidden = _filtered(discover, earnings, session, chosen, earnings_window)

    next_earnings = (
        {} if earnings is None or earnings.empty
        else dict(zip(earnings["symbol"], earnings["next_earnings"]))
    )
    themes_of = coverage.themes_of(discover["symbol"])
    try:
        written_at = datetime.fromisoformat(manifest.get("written_at", ""))
    except ValueError:
        written_at = None
    return OptionsOverview(
        session=session,
        written_at=written_at,
        sessions_behind=freshness.sessions_behind(session, _today_et()),
        themes=sorted(set(themes_of.values())),
        names=[
            Name(symbol=s, theme=theme, next_earnings=_date_or_none(next_earnings.get(s)))
            for s, theme in themes_of.items()
        ],
        visible=visible,
        hidden_for_earnings=hidden,
        earnings_known=earnings is not None,
        unavailable=[
            Unavailable(symbol=s, reason=str(reason))
            for s, reason in coverage.unavailable(manifest, stored.get("unpriced", {})).items()
        ],
        not_collected=coverage.not_collected(manifest),
        universe=[
            ThemeGroup(theme=t, symbols=[s for s, th in universe.UNIVERSE.items() if th == t])
            for t in universe.themes()
        ],
        min_history_sessions=DEFAULT.min_history_sessions,
        score_scale=SCALE,
    )


def _call(row: pd.Series) -> OneMonthCall | None:
    if finite_or_none(row["strike"]) is None:
        return None
    return OneMonthCall(
        expiration=_date_or_none(row["expiration"]),
        strike=float(row["strike"]),
        delta=float(row["delta"]),
        iv=float(row["iv"]),
        cost=float(row["cost"]),
        cost_pct=float(row["cost_pct"]),
        annualised=float(row["annualised"]),
        otm=float(row["otm"]),
        wide=bool(row["wide"]),
    )


@router.get("/discover")
def discover(
    themes: Themes = None, earnings_window: EarningsWindow = coverage.DEFAULT_EARNINGS_WINDOW
) -> OptionsDiscover:
    """Each visible name's buy-vol score, own-year percentiles and one-month 25Δ call."""
    chosen = _check_themes(themes)
    manifest, session = _night()
    _screens(manifest, session)
    table = options_store.fetch_frame(layout.DISCOVER)
    visible, _ = _filtered(table, _earnings(), session, chosen, earnings_window)
    shown = table[table["symbol"].isin(visible)].astype({"score": float})
    shown = shown.sort_values("score", ascending=False, na_position="last")
    return OptionsDiscover(
        session=session,
        rows=[
            DiscoverRow(
                symbol=row["symbol"],
                theme=universe.UNIVERSE[row["symbol"]],
                spot=float(row["spot"]),
                score=finite_or_none(row["score"]),
                z_iv=finite_or_none(row["z IV level"]),
                z_iv_rv=finite_or_none(row["z IV − realized"]),
                z_skew=finite_or_none(row["z 25Δ call skew"]),
                skew_pct=finite_or_none(row["skew_pct"]),
                iv_pct=finite_or_none(row["iv_pct"]),
                rr_pct=finite_or_none(row["rr_pct"]),
                call=_call(row),
            )
            for _, row in shown.iterrows()
        ],
    )


@router.get("/screen")
def screen(
    themes: Themes = None,
    earnings_window: EarningsWindow = coverage.DEFAULT_EARNINGS_WINDOW,
    sd: Annotated[float, Query(description="The move the spread pays in full at, in SDs.")] = 1.0,
    expiration: Annotated[date | None, Query()] = None,
) -> OptionsScreen:
    """The one-strike call spread that pays in full at a +k SD move, for every visible name."""
    chosen = _check_themes(themes)
    if sd not in snapshot.SD_TARGETS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"sd must be one of {', '.join(f'{t:g}' for t in snapshot.SD_TARGETS)}.",
        )
    manifest, session = _night()
    _screens(manifest, session)
    visible, _ = _filtered(
        options_store.fetch_frame(layout.DISCOVER), _earnings(), session, chosen, earnings_window
    )
    listed = options_store.fetch_frame(layout.EXPIRATIONS)
    choices = snapshot.common_expirations(listed[listed["symbol"].isin(visible)])
    if expiration not in choices:
        expiration = snapshot.default_spread_expiry(choices, session)

    spreads = options_store.fetch_frame(layout.SPREADS)
    rows = spreads[
        (spreads["expiration"] == expiration) & (spreads["sd"] == sd) & spreads["symbol"].isin(visible)
    ].astype({"payout_ask": float, "reward_risk": float})
    rows = rows.sort_values("payout_ask", ascending=False, na_position="last")
    return OptionsScreen(
        session=session,
        sds=list(snapshot.SD_TARGETS),
        sd=sd,
        expirations=choices,
        expiration=expiration,
        rows=[
            Spread(
                symbol=row["symbol"],
                theme=universe.UNIVERSE[row["symbol"]],
                move=float(row["move"]),
                long_strike=float(row["long_strike"]),
                short_strike=float(row["short_strike"]),
                price=float(row["price"]),
                width=float(row["width"]),
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                payout_mid=float(row["payout_mid"]),
                payout_ask=finite_or_none(row["payout_ask"]),
                reward_risk=finite_or_none(row["reward_risk"]),
            )
            for _, row in rows.iterrows()
        ],
        visible_count=len(visible),
    )


def _grid_rows(grid: dict, columns: list[tuple[str, object]]) -> list[GridRow]:
    return [
        GridRow(
            upside=upside,
            cells=[
                GridCell(
                    cost_pct=finite_or_none(cells[label].cost_pct),
                    annualised=finite_or_none(cells[label].annualised),
                    wide=cells[label].wide,
                )
                for label, _ in columns
            ],
        )
        for upside, cells in grid.items()
    ]


@router.get("/evaluate/{symbol}")
def evaluate(symbol: Symbol) -> OptionsEvaluate:
    """What replacing stock with calls costs on one name, by strike and expiry."""
    symbol = _known_symbol(symbol)
    manifest, session = _night()
    snaps, missing = snapshot.build(
        options_store.fetch_frame(layout.LATEST_CHAINS),
        options_store.fetch_frame(layout.METRICS),
        options_store.fetch_frame(layout.RATES),
        session,
        symbols=[symbol],
    )
    if symbol not in snaps:
        reason = missing.get(symbol) or coverage.unavailable(manifest, {}).get(symbol) or "no chain is stored"
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"{symbol} could not be priced on {session}: {reason}."
        )
    snap = snaps[symbol]
    columns = screens.grid_expiries(snap.surface)
    single = screens.replacement_grid(snap.quotes, columns, session, snap.spot)
    spread = screens.replacement_grid(snap.quotes, columns, session, snap.spot, spread=True)
    return OptionsEvaluate(
        symbol=symbol,
        theme=snap.theme,
        session=session,
        spot=snap.spot,
        columns=[GridColumn(label=label, expiration=e.expiration) for label, e in columns],
        single=_grid_rows(single, columns),
        spread=_grid_rows(spread, columns),
        spread_width=screens.GRID_SPREAD_WIDTH,
    )


@router.get("/history/{symbol}")
def history(symbol: Symbol) -> OptionsHistory:
    """One name's reward:risk over its trailing year, at constant 30 days, with ATM IV and skew."""
    symbol = _known_symbol(symbol)
    _, session = _night()
    metrics = options_store.fetch_frame(layout.METRICS)
    rows = metrics[metrics["symbol"] == symbol].set_index("session").sort_index()
    rows = rows.iloc[-DEFAULT.history_window:]
    stats = own_history_stats(rows["reward_risk"])
    return OptionsHistory(
        symbol=symbol,
        theme=universe.UNIVERSE[symbol],
        session=session,
        points=[
            HistoryPoint(
                session=_date_or_none(day),
                reward_risk=finite_or_none(row["reward_risk"]),
                atm_iv=finite_or_none(row["atm_iv_30"]),
                skew=finite_or_none(row["skew_30"]),
            )
            for day, row in rows.iterrows()
        ],
        stats=None if stats is None else RewardRiskStats(
            latest=stats.latest,
            as_of=stats.as_of,
            percentile=stats.percentile,
            quartile=reward_risk_quartile(stats.percentile),
            median=stats.median,
            q25=stats.q25,
            q75=stats.q75,
            low=stats.low,
            high=stats.high,
            sessions=stats.sessions,
        ),
        stored=int(rows["reward_risk"].notna().sum()),
        needed=DEFAULT.min_history_sessions,
    )
