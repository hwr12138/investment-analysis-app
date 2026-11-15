"""🔎 Stock Research over HTTP: one route per tab, each calling its own loader in ``dataset``.

The page is a profile and six tabs — 📊 Summary, 📄 Financials, ⚖️ Valuation
and Growth, 👤 Insiders and 🏛️ Holdings here, and the 💰 DCF in
:mod:`app.api.dcf`, which loads through the DCF engine rather than the snapshot.

Every route starts from :func:`app.logic.research.dataset.load_research`, the
snapshot a search builds. Its fetches are cached in the data layer, so
after the first route for a company the snapshot costs ~10 ms to rebuild, and
nothing here keeps one of its own. The frontend asks for the profile first and
for a tab only once it has arrived, so a cold search fetches the snapshot once
rather than once per concurrent request.

The one cache that lives here is :func:`_dcf_point_cached`, for the same reason
the Summary view keeps one: each point of the chart's DCF line is a full model
run, and a window revisited should not pay for them again. A failure that might
pass on a retry is raised by the loader and so never cached.

Numbers go out as numbers and the frontend formats them, except where the logic
layer already decided the text: the key statistics (``StatTile.display``) and
the statement cells, whose units and decimals are the statement layout's.
"""

from __future__ import annotations

import functools
from datetime import date
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Path, Query

from app.api.contract import Contract, finite_or_none
from app.data import sharadar_client
from app.logic import statements as statement_layouts
from app.logic.dcf import model as dcf_model
from app.logic.dcf import validation as dcf_validation
from app.logic.research import comparison, growth, holdings, insiders, summary, valuation
from app.logic.research import dataset as research
from app.provenance import Origin, Sourced

router = APIRouter(prefix="/research", tags=["research"])

Ticker = Annotated[str, Path(pattern=r"^[A-Za-z0-9.\-]{1,10}$")]
Period = Literal[summary.PERIODS]  # type: ignore[valid-type]
OriginName = Literal[tuple(o.value for o in Origin)]  # type: ignore[valid-type]
StatementKey = Literal[tuple(spec.key for spec in statement_layouts.STATEMENTS.values())]  # type: ignore[valid-type]
UnitName = Literal[tuple(statement_layouts.UNITS)]  # type: ignore[valid-type]
PeriodKind = Literal["annual", "quarterly"]

Cell = str | float | int | None


def _today() -> date:
    """The day every loader is keyed on. A function so a test can fix it."""
    return date.today()


def _snapshot(ticker: str) -> research.ResearchDataset:
    return research.load_research(ticker.strip().upper(), today=_today())


# ---------------------------------------------------------------------------
# Shapes shared by several tabs
# ---------------------------------------------------------------------------


class Provenance(Contract):
    """Where a figure came from — the columns of every tab's sources panel."""

    origin: OriginName
    source: str
    formula: str | None = None
    as_of: date | None = None
    detail: str = ""


class Frame(Contract):
    """A small table as it is printed: named columns, then rows of cells.

    For the reconciliation panels, whose columns are the logic layer's own and
    are read rather than computed on.
    """

    columns: list[str]
    rows: list[list[Cell]]


def as_provenance(sourced: Sourced) -> Provenance:
    return Provenance(
        origin=sourced.origin.value,
        source=sourced.source,
        formula=sourced.formula,
        as_of=sourced.as_of,
        detail=sourced.detail,
    )


def _cell(value: object) -> Cell:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp, date)):
        return pd.Timestamp(value).date().isoformat()
    if pd.api.types.is_bool(value):
        return "yes" if value else "no"
    if isinstance(value, int):
        return value
    if isinstance(value, float) or hasattr(value, "dtype"):
        number = float(value)  # numpy scalars
        if number.is_integer() and abs(number) < 2**53:
            return int(number)
        return finite_or_none(number)
    return str(value)


def as_frame(frame: pd.DataFrame) -> Frame:
    return Frame(
        columns=[str(c) for c in frame.columns],
        rows=[[_cell(v) for v in row] for row in frame.itertuples(index=False)],
    )


def _day(value: object) -> date | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date()


def _number(value: object) -> float | None:
    if value is None or (not isinstance(value, (int, float)) and pd.isna(value)):
        return None
    return finite_or_none(float(value))


# ---------------------------------------------------------------------------
# Profile — the page header
# ---------------------------------------------------------------------------


class Profile(Contract):
    ticker: str
    name: str
    exchange: str
    sector: str
    industry: str
    currency: str
    sec_filings: str | None
    website: str | None
    #: The day the snapshot was taken; every tab is keyed on it.
    loaded_at: date
    #: The company's last ``DAILY`` session — what the peer universe is priced on.
    market_session: date | None
    last_close: float | None
    last_session: date | None


@router.get("/{ticker}")
def profile(ticker: Ticker) -> Profile:
    """The header, and the snapshot every tab after it reads from the cache.

    404 when Sharadar has no record or no price history for the ticker.
    """
    dataset = _snapshot(ticker)
    info = dataset.profile

    def text(field: str) -> str:
        return str(info.get(field) or "").strip()

    closes = (
        dataset.ohlcv["Close"].dropna()
        if "Close" in dataset.ohlcv.columns
        else pd.Series(dtype="float64")
    )
    return Profile(
        ticker=dataset.ticker,
        name=dataset.name,
        exchange=text("exchange"),
        sector=text("sector"),
        industry=text("industry"),
        currency=dataset.currency,
        sec_filings=text("secfilings") or None,
        website=text("companysite") or None,
        loaded_at=dataset.loaded_at,
        market_session=dataset.market_session,
        last_close=finite_or_none(closes.iloc[-1]) if not closes.empty else None,
        last_session=_day(closes.index[-1]) if not closes.empty else None,
    )


# ---------------------------------------------------------------------------
# 📊 Summary
# ---------------------------------------------------------------------------


class Candles(Contract):
    sessions: list[date]
    open: list[float | None]
    high: list[float | None]
    low: list[float | None]
    close: list[float | None]
    volume: list[float | None]


class Stat(Contract):
    label: str
    #: Formatted by the logic layer; ``None`` exactly when the figure is unavailable.
    display: str | None
    provenance: Provenance


class Summary(Contract):
    ticker: str
    currency: str
    period: Period
    periods: list[str]
    candles: Candles
    change: float | None
    change_reason: str
    change_formula: str | None
    #: The 4×4 grid as columns of statistic labels, in ``summary.STAT_COLUMNS`` order.
    grid: list[list[str]]
    #: Every statistic in build order, each with its provenance.
    stats: list[Stat]


@router.get("/{ticker}/summary")
def summary_tab(ticker: Ticker, period: Period = "1Y") -> Summary:
    """The price window and the key statistics. A window is a slice of the
    snapshot's full history, so changing it fetches nothing."""
    dataset = _snapshot(ticker)
    window = summary.slice_period(dataset.ohlcv, period, dataset.loaded_at)

    def column(name: str) -> list[float | None]:
        if name not in window.columns:
            return [None] * len(window)
        return [finite_or_none(v) for v in window[name]]

    change = summary.period_change(window)
    return Summary(
        ticker=dataset.ticker,
        currency=dataset.currency,
        period=period,
        periods=list(summary.PERIODS),
        candles=Candles(
            sessions=[d.date() for d in pd.DatetimeIndex(window.index)],
            open=column("Open"),
            high=column("High"),
            low=column("Low"),
            close=column("Close"),
            volume=column("Volume"),
        ),
        change=finite_or_none(change.value) if change.ok else None,
        change_reason="" if change.ok else change.detail,
        change_formula=change.formula,
        grid=[list(labels) for labels in summary.STAT_COLUMNS],
        stats=[
            Stat(
                label=tile.label,
                display=tile.display,
                provenance=Provenance(
                    origin=tile.origin.value,
                    source=tile.source,
                    formula=tile.formula,
                    as_of=tile.as_of,
                    detail=tile.detail,
                ),
            )
            for tile in dataset.stats
        ],
    )


class DcfStep(Contract):
    session: date
    value: float | None
    label: str
    #: Why there is no value at this step, when there is none.
    reason: str


class DcfLine(Contract):
    period: Period
    #: Where the line begins: the window's start, or ``max_years`` back.
    start: date | None
    max_years: int
    steps: list[DcfStep]
    #: Set, with ``steps`` empty, when there is no line at all.
    reason: str


@functools.lru_cache(maxsize=2048)
def _dcf_point_cached(ticker: str, as_of: date | None, today: date | None) -> research.DcfPoint:
    """One filing's value. ``today`` keys only today's point, which moves daily.

    An exception is not cached, and the loader raises for what a retry might
    fix, so an outage is retried on the next request rather than remembered.
    """
    return research.load_dcf_point(ticker, as_of)


def _dcf_point(ticker: str, as_of: date | None, today: date) -> research.DcfPoint:
    try:
        return _dcf_point_cached(ticker, as_of, today if as_of is None else None)
    except (
        sharadar_client.SharadarError,
        dcf_model.DefaultUnavailable,
        dcf_validation.UnsupportedCurrencyError,
    ) as error:
        return research.DcfPoint(as_of, None, f"{error} Retried on the next load.")


@router.get("/{ticker}/dcf-line")
def dcf_line(ticker: Ticker, period: Period = "1Y") -> DcfLine:
    """The DCF tab's default headline, re-struck at each filing in the window.

    Separate from the summary because it is slow — a full model run per filing,
    about a second each cold — and the candles should never wait on it. Points
    are cached one by one, so a wider window pays only for the filings it adds.
    """
    dataset = _snapshot(ticker)
    window = summary.slice_period(dataset.ohlcv, period, dataset.loaded_at)
    empty = dict(period=period, max_years=summary.DCF_LINE_MAX_YEARS, steps=[], start=None)
    if window.empty:
        return DcfLine(**empty, reason="No price history in this window.")
    try:
        filings = research.load_dcf_filings(dataset.ticker)
    except sharadar_client.SharadarError as error:
        return DcfLine(**empty, reason=f"The filing dates could not be fetched — {error}")

    dates, planned = summary.dcf_line_steps(
        {f.filed: f.period for f in filings}, pd.DatetimeIndex(window.index)
    )
    current = _dcf_point(dataset.ticker, None, dataset.loaded_at)
    points = [
        current if step.as_of is None else _dcf_point(dataset.ticker, step.as_of, dataset.loaded_at)
        for step in planned
    ]
    if all(point.value is None for point in points):
        return DcfLine(**empty, reason=current.reason or points[0].reason)
    return DcfLine(
        period=period,
        start=dates.start,
        max_years=summary.DCF_LINE_MAX_YEARS,
        steps=[
            DcfStep(
                session=step.drawn_at,
                value=finite_or_none(point.value),
                label=step.label,
                reason=point.reason,
            )
            for step, point in zip(planned, points)
        ],
        reason="",
    )


# ---------------------------------------------------------------------------
# 📄 Financials
# ---------------------------------------------------------------------------


class StatementChoice(Contract):
    key: StatementKey
    label: str


class PeriodHeader(Contract):
    key: str
    label: str
    sublabel: str
    is_latest: bool
    report_date: date | None
    publish_date: date | None
    restated: str


class StatementCell(Contract):
    #: Formatted in the table's units by the statement layout.
    text: str
    missing: bool
    zero_filled: bool
    reason: str


class StatementRow(Contract):
    key: str
    label: str
    emphasis: str
    indent: int
    origin: OriginName
    formula: str
    note: str
    #: The reported subtotal and the lines above it disagree.
    flagged: bool
    flag_detail: str
    cells: list[StatementCell]


class StatementSection(Contract):
    title: str
    note: str
    rows: list[StatementRow]


class Reconciliation(Contract):
    row_label: str
    formula: str
    status: str
    periods_checked: int
    worst_gap: float | None
    worst_period: str | None
    tolerance: float
    detail: str


class Financials(Contract):
    ticker: str
    currency: str
    statements: list[StatementChoice]
    units: list[UnitName]
    statement: StatementKey
    period: PeriodKind
    unit: UnitName
    show_empty: bool
    #: False when Sharadar returned no fundamentals at either dimension.
    available: bool
    label: str
    dimension: str
    unit_label: str
    unit_suffix: str
    periods: list[PeriodHeader]
    sections: list[StatementSection]
    reported_rows: int
    hidden_rows: int
    checks: list[Reconciliation]
    filings: Frame


@router.get("/{ticker}/financials")
def financials(
    ticker: Ticker,
    statement: StatementKey = "PL",
    period: PeriodKind = "annual",
    unit: Annotated[UnitName, Query()] = "Millions",
    show_empty: bool = False,
) -> Financials:
    """One statement, laid out and reconciled by ``logic.statements``.

    The balance sheet's leading column is the latest quarter, the others' the
    trailing twelve months.
    """
    dataset = _snapshot(ticker)
    data = research.load_financials(dataset.ticker)
    spec = next(s for s in statement_layouts.STATEMENTS.values() if s.key == statement)
    annual = period == "annual"
    available = bool(data.annual or data.quarterly)
    # No fundamentals at either dimension is said rather than laid out: the TTM
    # column alone would draw a table that looks complete and is not.
    table = statement_layouts.StatementTable(
        statement=spec.key, label=spec.label, dimension="", currency=dataset.currency,
        units=statement_layouts.UNITS[unit],
    ) if not available else statement_layouts.build_statement(
        spec,
        periodic=data.annual if annual else data.quarterly,
        latest=data.quarterly if spec.latest_source == "MRQ" else dataset.statements_ttm,
        annual=annual,
        currency=dataset.currency,
        units=statement_layouts.UNITS[unit],
        max_periods=None,
        show_empty=show_empty,
    )
    filings = table.meta.copy()
    return Financials(
        ticker=dataset.ticker,
        currency=dataset.currency,
        statements=[StatementChoice(key=s.key, label=s.label) for s in statement_layouts.STATEMENTS.values()],
        units=list(statement_layouts.UNITS),
        statement=statement,
        period=period,
        unit=unit,
        show_empty=show_empty,
        available=available,
        label=table.label,
        dimension=table.dimension,
        unit_label=table.units.label,
        unit_suffix=table.units.suffix,
        periods=[
            PeriodHeader(
                key=p.key,
                label=p.label,
                sublabel=p.sublabel,
                is_latest=p.is_latest,
                report_date=p.report_date,
                publish_date=p.publish_date,
                restated=p.restated,
            )
            for p in table.periods
        ],
        sections=[
            StatementSection(
                title=section.title,
                note=section.note,
                rows=[
                    StatementRow(
                        key=row.key,
                        label=row.label,
                        emphasis=row.emphasis.value,
                        indent=row.indent,
                        origin=row.origin.value,
                        formula=row.formula,
                        note=row.note,
                        flagged=row.flagged,
                        flag_detail=row.reconcile.detail if row.reconcile else "",
                        cells=[
                            StatementCell(
                                text=cell.text,
                                missing=cell.missing,
                                zero_filled=cell.zero_filled,
                                reason=cell.reason,
                            )
                            for cell in row.cells
                        ],
                    )
                    for row in section.rows
                ],
            )
            for section in table.sections
        ],
        reported_rows=table.reported_rows,
        hidden_rows=table.hidden_rows,
        checks=[
            Reconciliation(
                row_label=r.row_label,
                formula=r.formula,
                status=r.status,
                periods_checked=r.periods_checked,
                worst_gap=finite_or_none(r.worst_gap),
                worst_period=r.worst_period,
                tolerance=r.tolerance,
                detail=r.detail,
            )
            for r in table.diagnostics
        ],
        filings=as_frame(filings),
    )


# ---------------------------------------------------------------------------
# ⚖️ Valuation and Growth
# ---------------------------------------------------------------------------


class ComparisonRow(Contract):
    label: str
    definition: str
    #: Formatted as a percentage rather than a multiple.
    percent: bool
    origin: OriginName
    current: float | None
    current_reason: str
    history_median: float | None
    history_observations: int
    #: What the observations count — "sessions" for a multiple, "quarters" for a rate.
    history_unit: str
    history_reason: str
    #: The share of the company's own five years below today's reading.
    percentile: float | None
    industry_median: float | None
    peer_count: int
    industry_reason: str
    #: A ratio for a multiple (0.25 is 25% dearer), percentage points for a rate.
    gap: float | None
    verdict: str
    favourable: bool | None


class Relative(Contract):
    ticker: str
    name: str
    kind: Literal["valuation", "growth"]
    #: False when the peer universe could not be loaded at all.
    universe_loaded: bool
    industry: str
    peers: int
    priced: date | None
    rows: list[ComparisonRow]


def _relative(ticker: str, kind: Literal["valuation", "growth"]) -> Relative:
    dataset = _snapshot(ticker)
    universe = research.load_peers(dataset.market_session, today=dataset.loaded_at)
    industry = comparison.industry_of(universe, dataset.ticker)
    if kind == "valuation":
        built = valuation.build_comparison(
            daily=dataset.daily,
            statements_ttm=dataset.statements_ttm,
            peer_universe=universe,
            industry=industry,
            today=dataset.loaded_at,
        )
        read = valuation.verdict
    else:
        built = growth.build_comparison(
            statements_ttm=dataset.statements_ttm,
            peer_universe=universe,
            industry=industry,
            today=dataset.loaded_at,
        )
        read = growth.verdict

    def row(c: comparison.Comparison) -> ComparisonRow:
        reading = read(c)
        return ComparisonRow(
            label=c.metric.label,
            definition=c.metric.definition,
            percent=bool(getattr(c.metric, "percent", True)),
            origin=c.origin.value,
            current=finite_or_none(c.current),
            current_reason=c.current_reason,
            history_median=finite_or_none(c.history_median),
            history_observations=c.history_observations,
            history_unit=c.history_unit,
            history_reason=c.history_reason,
            percentile=finite_or_none(c.percentile),
            industry_median=finite_or_none(c.industry_median),
            peer_count=c.peer_count,
            industry_reason=c.industry_reason,
            gap=finite_or_none(reading.gap),
            verdict=reading.word,
            favourable=reading.favourable,
        )

    return Relative(
        ticker=dataset.ticker,
        name=dataset.name,
        kind=kind,
        universe_loaded=not universe.empty,
        industry=industry,
        peers=len(comparison.industry_peers(universe, industry)),
        priced=comparison.priced_on(universe),
        rows=[row(c) for c in built],
    )


@router.get("/{ticker}/valuation")
def valuation_tab(ticker: Ticker) -> Relative:
    """Each multiple now, against its own five years and its industry's median."""
    return _relative(ticker, "valuation")


@router.get("/{ticker}/growth")
def growth_tab(ticker: Ticker) -> Relative:
    """Each year-on-year rate now, against its own five years and its industry's."""
    return _relative(ticker, "growth")


# ---------------------------------------------------------------------------
# 👤 Insiders
# ---------------------------------------------------------------------------


class InsiderWindow(Contract):
    months: int
    available: bool
    reason: str
    net: float | None
    bought: float | None
    sold: float | None
    buy_trades: int
    sell_trades: int
    participants: int
    top_owner: str
    concentration: float | None
    #: One filer is most of the gross dollars, so the net is close to one decision.
    concentrated: bool


class PriceLine(Contract):
    sessions: list[date]
    close: list[float | None]


class Trade(Contract):
    session: date
    code: Literal["P", "S"]
    owner: str
    role: str
    shares: float | None
    #: Signed dollars: shares × the filed price.
    value: float
    #: The price paid on the chart's split-adjusted basis.
    price: float | None
    #: The price on the form, which differs from ``price`` only across a split.
    filed_price: float | None
    filed: date | None


class QuarterFlow(Contract):
    label: str
    bought: float
    sold: float
    net: float
    trades: int
    owners: int
    #: The quarter is still open.
    partial: bool


class Cluster(Contract):
    start: date
    end: date
    buyers: int
    trades: int
    value: float
    owners: list[str]


class RosterRow(Contract):
    owner: str
    role: str
    shares: float | None
    as_of: date | None
    last_code: str


class Insiders(Contract):
    ticker: str
    currency: str
    #: False when nobody filed a form 3, 4 or 5 in the window — an answer, not a gap.
    available: bool
    lookback_years: int
    windows: list[InsiderWindow]
    prices: PriceLine
    trades: list[Trade]
    quarters: list[QuarterFlow]
    clusters: list[Cluster]
    cluster_min_buyers: int
    cluster_min_value: float
    cluster_window_days: int
    roster: list[RosterRow]
    roster_reason: str
    roster_as_of: date | None
    reconciliation: Frame
    filed_rows: int
    shown_rows: int


@router.get("/{ticker}/insiders")
def insiders_tab(ticker: Ticker) -> Insiders:
    """Open-market form 4 activity — ``P`` and ``S`` only — and every row excluded."""
    dataset = _snapshot(ticker)
    frame = research.load_insiders(dataset.ticker, today=dataset.loaded_at)
    tolerances = insiders.InsiderTolerances()
    today = dataset.loaded_at
    criteria = dict(
        lookback_years=tolerances.lookback_years,
        cluster_min_buyers=tolerances.cluster_min_buyers,
        cluster_min_value=tolerances.cluster_min_value,
        cluster_window_days=tolerances.cluster_window_days,
    )
    if not insiders.usable(frame):
        return Insiders(
            ticker=dataset.ticker, currency=dataset.currency, available=False, windows=[],
            prices=PriceLine(sessions=[], close=[]), trades=[], quarters=[], clusters=[],
            roster=[], roster_reason="", roster_as_of=None,
            reconciliation=Frame(columns=[], rows=[]), filed_rows=0, shown_rows=0, **criteria,
        )

    cutoff = pd.Timestamp(today) - pd.DateOffset(years=tolerances.lookback_years)
    closes = dataset.ohlcv["Close"] if "Close" in dataset.ohlcv.columns else pd.Series(dtype=float)
    closes = closes[closes.index >= cutoff]

    trades = insiders.open_market_trades(frame, dataset.actions, tolerances=tolerances, today=today)
    quarters = insiders.periodic_flow(frame, tolerances=tolerances, today=today)
    roster = insiders.roster(frame, tolerances=tolerances, today=today)
    panel = insiders.reconciliation(frame, tolerances=tolerances, today=today)
    counted = panel[panel["Category"] == "Open market, priced"]

    return Insiders(
        ticker=dataset.ticker,
        currency=dataset.currency,
        available=True,
        windows=[
            InsiderWindow(
                months=w.months,
                available=w.ok,
                reason="" if w.ok else w.net.detail,
                net=finite_or_none(w.net.value),
                bought=finite_or_none(w.bought.value),
                sold=finite_or_none(w.sold.value),
                buy_trades=w.buy_trades,
                sell_trades=w.sell_trades,
                participants=w.participants,
                top_owner=w.top_owner,
                concentration=finite_or_none(w.concentration),
                concentrated=bool(w.concentration_note),
            )
            for w in insiders.trailing_windows(frame, tolerances=tolerances, today=today)
        ],
        prices=PriceLine(
            sessions=[d.date() for d in pd.DatetimeIndex(closes.index)],
            close=[finite_or_none(v) for v in closes],
        ),
        trades=[
            Trade(
                session=row.when.date(),
                code=row.transactioncode,
                owner=str(row.ownername),
                role=str(row.role),
                shares=_number(row.transactionshares),
                value=float(row.flow),
                price=_number(row.plotted),
                filed_price=_number(row.transactionpricepershare),
                filed=_day(getattr(row, "filingdate", None)),
            )
            for row in trades.itertuples()
        ],
        quarters=[
            QuarterFlow(
                label=pd.Period(row.period, freq="Q").strftime("%Y Q%q"),
                bought=float(row.bought),
                sold=float(row.sold),
                net=float(row.net),
                trades=int(row.trades),
                owners=int(row.owners),
                partial=bool(row.partial),
            )
            for row in quarters.itertuples()
        ],
        clusters=[
            Cluster(
                start=c.start, end=c.end, buyers=c.buyers, trades=c.trades,
                value=c.value, owners=list(c.owners),
            )
            for c in insiders.clusters(frame, tolerances=tolerances, today=today)
        ],
        roster=[
            RosterRow(
                owner=str(row.owner),
                role=str(row.role),
                shares=_number(row.shares),
                as_of=_day(row.as_of),
                last_code=str(row.last_code),
            )
            for row in roster.rows.itertuples()
        ],
        roster_reason=roster.reason,
        roster_as_of=roster.as_of,
        reconciliation=as_frame(panel),
        filed_rows=len(frame),
        shown_rows=int(counted["Rows"].iloc[0]) if not counted.empty else 0,
        **criteria,
    )


# ---------------------------------------------------------------------------
# 🏛️ Holdings
# ---------------------------------------------------------------------------


class Holder(Contract):
    name: str
    is_index: bool
    index_house: str
    units: float
    #: Quarter-on-quarter change in shares; ``None`` for a new position.
    change: float | None
    owned: float | None
    value: float
    #: Shares held each quarter in ``periods``; ``None`` is a quarter it filed nothing.
    series: list[float | None]


class OwnershipPoint(Contract):
    quarter: date
    owned: float | None
    units: float | None
    holders: float | None


class ChurnCounts(Contract):
    available: bool
    reason: str
    opened: int
    closed: int
    added: int
    trimmed: int
    unchanged: int


class FilerFamily(Contract):
    label: str
    entities: int


class Holdings(Contract):
    ticker: str
    currency: str
    #: False when no 13F position is on file; ``reason`` says why.
    available: bool
    reason: str
    as_of: date | None
    deadline: date | None
    #: Days since the filing deadline; negative while managers are still filing.
    days_old: int | None
    filing_lag_days: int
    owned: float | None
    units: float | None
    filers: float | None
    top_n: int
    top_share: float | None
    hhi: float | None
    material_change: float
    churn: ChurnCounts
    holders: list[Holder]
    periods: list[date]
    total_value: float
    total_units: float
    total_change: float | None
    filer_count: int
    ownership: list[OwnershipPoint]
    reconciliation: Frame
    families: list[FilerFamily]
    index_managers: list[str]


@router.get("/{ticker}/holdings")
def holdings_tab(ticker: Ticker) -> Holdings:
    """Who filed a 13F position, as at the latest quarter on file — never today."""
    dataset = _snapshot(ticker)
    data = research.load_holdings(dataset.ticker, today=dataset.loaded_at)
    tolerances = holdings.HoldingsTolerances()
    built = holdings.build(
        summary=data.summary,
        current=data.current,
        history=data.history,
        names=data.names,
        actions=dataset.actions,
        statements=data.derived,
        tolerances=tolerances,
        today=dataset.loaded_at,
    )
    table = built.holders
    latest = built.latest
    return Holdings(
        ticker=dataset.ticker,
        currency=dataset.currency,
        available=built.ok,
        reason="" if built.ok else table.reason,
        as_of=_day(built.as_of),
        deadline=built.deadline,
        days_old=built.days_old(dataset.loaded_at),
        filing_lag_days=holdings.FILING_LAG_DAYS,
        owned=finite_or_none(latest.get("owned")),
        units=finite_or_none(latest.get("units")),
        filers=finite_or_none(latest.get("holders")),
        top_n=tolerances.concentration_top_n,
        top_share=finite_or_none(built.top_share.value),
        hhi=finite_or_none(built.hhi.value),
        material_change=tolerances.material_change,
        churn=ChurnCounts(
            available=built.churn.ok,
            reason=built.churn.reason,
            opened=built.churn.opened,
            closed=built.churn.closed,
            added=built.churn.added,
            trimmed=built.churn.trimmed,
            unchanged=built.churn.unchanged,
        ),
        holders=[
            Holder(
                name=h.name,
                is_index=h.is_index,
                index_house=h.index_house,
                units=float(h.units),
                change=finite_or_none(h.change),
                owned=finite_or_none(h.owned),
                value=float(h.value),
                series=[finite_or_none(v) for v in h.series],
            )
            for h in table.rows
        ],
        periods=[_day(p) for p in table.periods],
        total_value=float(table.total_value),
        total_units=float(table.total_units),
        total_change=finite_or_none(table.total_change),
        filer_count=table.holders,
        ownership=[
            OwnershipPoint(
                quarter=_day(row.quarter),
                owned=_number(row.owned),
                units=_number(row.units),
                holders=_number(row.holders),
            )
            for row in built.ownership.itertuples()
        ],
        reconciliation=as_frame(built.reconciliation),
        families=[
            FilerFamily(label=label, entities=len(members))
            for label, members in holdings.FILER_FAMILIES.items()
        ],
        index_managers=sorted(set(holdings.INDEX_MANAGERS.values())),
    )
