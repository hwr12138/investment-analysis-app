"""Financial statements as sectioned, reconciled tables.

Turns the four wide frames a :class:`~app.data.sharadar_client.StatementSet`
carries into something an analyst reads down rather than across: labelled
sections, indented detail lines, bolded subtotals, and — where the arithmetic is
checkable — a reconciliation of each reported subtotal against the lines above
it.

Pure. No Streamlit, no network, no global state. The view layer supplies the
frames and renders the result; every number and every string in the output is
decided here.

Three properties are load-bearing and easy to break:

* **A subtotal shows what the company reported**, never the sum of the rows
  above it. Where both exist the two are compared and the row is flagged when
  they disagree — a table that always adds up because it was made to add up
  tells the reader nothing.
* **Missing stays missing.** Every derived row is plain arithmetic over numpy
  arrays, so one absent component makes the whole row absent for that period.
  ``np.nansum`` and friends are banned here; see :func:`_derive`.
* **Units apply to money and share counts only.** Scaling a whole frame by a
  million is how EPS becomes 0.0 and a margin becomes 0.0%; :class:`Unit`
  exists so that cannot happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Mapping

import numpy as np
import pandas as pd

from app.data import sharadar_client
from app.provenance import Origin

_STATEMENT_KEYS = ("PL", "BS", "CF", "DERIVED")

#: Metadata columns, in the order the provenance panel shows them. A subset of
#: :data:`sharadar_client.META_COLUMNS`; the rest are dropped.
_META_OUT = (
    "Fiscal Year", "Fiscal Period", "Report Date", "Publish Date", "Restated",
)


# ---------------------------------------------------------------------------
# Specification vocabulary
# ---------------------------------------------------------------------------


class Emphasis(str, Enum):
    """How much visual weight a row carries."""

    DETAIL = "detail"
    SUBTOTAL = "subtotal"
    TOTAL = "total"


class Unit(str, Enum):
    """What kind of quantity a row holds.

    Drives formatting and, critically, whether the millions/billions divisor
    applies at all. Only ``CURRENCY`` and ``SHARES`` are scaled.
    """

    CURRENCY = "currency"
    SHARES = "shares"
    PER_SHARE = "per_share"
    PERCENT = "percent"
    RATIO = "ratio"


@dataclass(frozen=True)
class FromCaption:
    """Read a Sharadar caption.

    Which of the four frames it lives on is not declared — the builder works
    off one merged frame, so :data:`sharadar_client.SF1_LAYOUT` stays the single
    source of truth for the partition. That is also what lets the cash-flow
    statement open with Net Income, which Sharadar publishes on the income
    statement.
    """

    caption: str


@dataclass(frozen=True)
class Sum:
    """Signed sum of other rows in the same statement.

    ``minus`` is not decoration: ``netincnci`` and ``prefdivis`` are the two SF1
    income-statement lines the client does *not* negate on ingest, so they
    arrive as positive magnitudes and genuinely subtract.
    """

    plus: tuple[str, ...]
    minus: tuple[str, ...] = ()


@dataclass(frozen=True)
class Residual:
    """A plug: ``base`` less the components already shown above it.

    Sharadar publishes subtotals whose detail lines do not account for them —
    ``opex`` is not ``sgna + rnd`` — because the remainder has no indicator of
    its own. Without these rows a section visibly fails to add up; with them the
    remainder is named and marked as this app's arithmetic, not a filed line.
    """

    base: str
    less: tuple[str, ...]


@dataclass(frozen=True)
class Ratio:
    """One row divided by another — per-share figures, rates, margins.

    A zero denominator yields *missing*, not infinity. ``scale`` exists for the
    one case where the sign has to flip: tax expense is negative downstream of
    ingest, but an effective tax rate reads as a positive percentage.

    ``require_positive_denominator`` is for rates that are meaningless when the
    base is negative — an effective tax rate on a pretax *loss* is arithmetic
    without a meaning, so it reports missing instead. Same guard
    ``dcf.financials.effective_tax_rate`` applies.
    """

    numerator: str
    denominator: str
    scale: float = 1.0
    require_positive_denominator: bool = False


@dataclass(frozen=True)
class Product:
    """Rows multiplied together.

    Exists for share counts: Sharadar's ``sharesbas`` is the count on the filing
    cover, and ``sharefactor`` corrects it for ADRs representing a fraction or
    multiple of an underlying share. Their product is what Sharadar itself
    multiplies by price to get market capitalisation, and it is what
    ``sharadar_client.common_shares_frame`` already computes.
    """

    factors: tuple[str, ...]


Source = FromCaption | Sum | Residual | Ratio | Product


@dataclass(frozen=True)
class Check:
    """Does this reported row equal this expression over the other rows?

    Lives on the row being checked rather than in a side registry, so a row and
    its identity cannot drift apart.

    ``expected`` is any :data:`Source`, so a check can verify a division as
    readily as a sum — which matters for the vendor's per-share metrics, where
    the interesting question is *which* share count was used.
    """

    expected: Source
    note: str = ""


@dataclass(frozen=True)
class RowSpec:
    """One line of a statement."""

    key: str
    label: str
    source: Source
    emphasis: Emphasis = Emphasis.DETAIL
    unit: Unit = Unit.CURRENCY
    indent: int = 0
    check: Check | None = None
    #: ``None`` infers from the source kind: reported for a caption, derived
    #: for arithmetic. Set explicitly only where the inference would lie.
    origin: Origin | None = None
    note: str = ""

    @property
    def inferred_origin(self) -> Origin:
        if self.origin is not None:
            return self.origin
        return Origin.REPORTED if isinstance(self.source, FromCaption) else Origin.DERIVED


@dataclass(frozen=True)
class SectionSpec:
    title: str
    rows: tuple[RowSpec, ...]
    note: str = ""


@dataclass(frozen=True)
class StatementSpec:
    """One statement's full layout."""

    key: str
    label: str
    #: Header for the leading column.
    latest_label: str
    #: Which fetched dimension feeds it — ``"MRT"`` for flows, ``"MRQ"`` for the
    #: balance sheet, which has no twelve-month reading.
    latest_source: str
    sections: tuple[SectionSpec, ...]

    def rows(self):
        for section in self.sections:
            yield from section.rows


# ---------------------------------------------------------------------------
# Units and tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitScale:
    divisor: float
    suffix: str
    decimals: int
    label: str


UNITS: dict[str, UnitScale] = {
    "As reported": UnitScale(1.0, "", 0, "as reported"),
    "Millions": UnitScale(1e6, " (m)", 1, "millions"),
    "Billions": UnitScale(1e9, " (bn)", 2, "billions"),
}


@dataclass(frozen=True)
class StatementTolerances:
    """Every tolerance this module applies, in one place.

    Mirrors ``dcf.validation.Tolerances`` — same discipline, different owner.
    Deliberately *not* imported from there: those thresholds gate whether a
    valuation is trustworthy, these gate whether a display row grows a marker,
    and sharing them would mean a change made for a valuation reason silently
    changed what the research page flags.

    Relative tolerances are fractions of the larger of the reported and the
    computed figure.
    """

    income_statement: float = 0.005
    balance_sheet: float = 0.001
    cash_flow: float = 0.001
    derived: float = 0.010
    #: Absolute floor, in units of the reporting currency. Gaps below this are
    #: rounding in the filing rather than disagreement — and without it a
    #: subtotal that is genuinely zero has no meaningful relative scale, so a
    #: half-unit difference would read as infinitely wrong.
    absolute_floor: float = 1.0
    #: The same idea for rows that are not money: per-share figures, rates and
    #: multiples. One currency unit is the right floor for a figure reported in
    #: whole dollars and a useless one for a per-share figure of about 9, where
    #: it would let an 11% error pass unflagged. Sized instead for the precision
    #: the vendor publishes these at — Sharadar's own rounding on a per-share
    #: metric runs to about 0.0005, so this is an order of magnitude above the
    #: noise and still tight enough to catch a wrong divisor on a figure of well
    #: under a currency unit.
    unscaled_floor: float = 0.001


STATEMENT_TOLERANCES = StatementTolerances()

_TOLERANCE_FIELD = {
    "PL": "income_statement",
    "BS": "balance_sheet",
    "CF": "cash_flow",
    "DERIVED": "derived",
}

#: What a cell shows when the figure was never reported. Deliberately not "0".
MISSING_TEXT = "—"


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodColumn:
    """One column of the table."""

    key: str
    label: str
    sublabel: str
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    report_date: date | None = None
    publish_date: date | None = None
    restated: str = ""
    is_latest: bool = False


@dataclass(frozen=True)
class Cell:
    """One figure, already formatted."""

    text: str
    #: Unscaled and unformatted, for export and for tests.
    value: float | None
    missing: bool
    #: Reported as exactly 0 on an indicator Sharadar zero-fills, which may mean
    #: "absent from the filing" rather than "nil". The two cannot be told apart.
    zero_filled: bool = False
    #: Why a derived row could not be computed for this period.
    reason: str = ""


@dataclass(frozen=True)
class ReconcileResult:
    row_key: str
    row_label: str
    status: str  # "ok" | "gap" | "not evaluated"
    formula: str
    periods_checked: int
    tolerance: float
    worst_period: str | None = None
    worst_gap: float | None = None
    worst_relative: float | None = None
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.status == "gap"


@dataclass(frozen=True)
class RenderedRow:
    key: str
    label: str
    emphasis: Emphasis
    unit: Unit
    indent: int
    origin: Origin
    formula: str
    note: str
    cells: tuple[Cell, ...]
    reconcile: ReconcileResult | None = None

    @property
    def flagged(self) -> bool:
        return self.reconcile is not None and self.reconcile.failed


@dataclass(frozen=True)
class RenderedSection:
    title: str
    note: str
    rows: tuple[RenderedRow, ...]


@dataclass(frozen=True)
class StatementTable:
    statement: str
    label: str
    dimension: str
    currency: str
    units: UnitScale
    periods: tuple[PeriodColumn, ...] = ()
    sections: tuple[RenderedSection, ...] = ()
    reported_rows: int = 0
    hidden_rows: int = 0
    diagnostics: tuple[ReconcileResult, ...] = ()
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def empty(self) -> bool:
        return not self.periods or not self.sections

    @property
    def flagged_rows(self) -> tuple[str, ...]:
        return tuple(
            row.key
            for section in self.sections
            for row in section.rows
            if row.flagged
        )


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------


def _text(value: object) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return str(value)


def _period_key(frame: pd.DataFrame) -> pd.Index:
    """A stable join key for one statement frame.

    ``_order_statement_rows`` de-duplicates on fiscal year and period, so this
    is unique within a frame. The report date is folded in so a frame that
    somehow carried two rows for one period does not silently collapse them.
    """
    columns = [frame.get(name) for name in ("Fiscal Year", "Fiscal Period", "Report Date")]
    parts = [
        [_text(value) for value in (column if column is not None else [None] * len(frame))]
        for column in columns
    ]
    return pd.Index([f"{a}|{b}|{c}" for a, b, c in zip(*parts)])


def _wide(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The four statement frames merged into one row per period.

    They come from one SF1 row apiece — ``_sf1_frames`` builds a single metadata
    block and concatenates it to each — so they are row-aligned by construction.
    They are still joined on an explicit period key rather than by position: a
    future change to ``_order_statement_rows`` that dropped a row from one frame
    and not another would otherwise shift every figure by a period, silently and
    undetectably.

    Merging is also what lets the cash-flow spec name an income-statement
    caption, and the derived spec reconcile EBITDA against EBIT and D&A.
    Captions are unique across all four frames, so nothing collides.
    """
    present = [
        frames[key]
        for key in _STATEMENT_KEYS
        if key in frames and frames[key] is not None and not frames[key].empty
    ]
    if not present:
        return pd.DataFrame()

    bodies = []
    for frame in present:
        columns = [c for c in frame.columns if c not in sharadar_client.META_COLUMNS]
        body = frame[columns].copy()
        body.index = _period_key(frame)
        bodies.append(body)

    wide = pd.concat(bodies, axis=1)
    wide = wide.loc[:, ~wide.columns.duplicated()]

    meta_columns = [c for c in _META_OUT if c in present[0].columns]
    meta = present[0][meta_columns].copy()
    meta.index = _period_key(present[0])
    meta = meta.reindex(wide.index)

    combined = pd.concat([meta, wide], axis=1)
    if "Report Date" in combined.columns:
        combined = combined.sort_values("Report Date")
    return combined


def _column_from_row(
    row: pd.Series, key: str, *, annual: bool, label: str | None = None,
    sublabel: str | None = None, is_latest: bool = False,
) -> PeriodColumn:
    """One :class:`PeriodColumn` from a merged-frame row.

    ``Fiscal Year`` is a nullable ``Int64``, so it is checked for null before
    ``int()``; a row with no fiscal year falls back to its report date rather
    than raising.
    """
    raw_year = row.get("Fiscal Year")
    year = None if raw_year is None or pd.isna(raw_year) else int(raw_year)
    period = _text(row.get("Fiscal Period")) or None
    report = row.get("Report Date")
    report_date = None if report is None or pd.isna(report) else pd.Timestamp(report).date()
    publish = row.get("Publish Date")
    publish_date = None if publish is None or pd.isna(publish) else pd.Timestamp(publish).date()
    report_text = report_date.isoformat() if report_date else ""

    if label is None:
        if year is None:
            label = report_text or key
        elif annual:
            label = f"FY{year}"
        else:
            label = f"FY{year} {period}" if period else f"FY{year}"

    return PeriodColumn(
        key=key,
        label=label,
        sublabel=report_text if sublabel is None else sublabel,
        fiscal_year=year,
        fiscal_period=period,
        report_date=report_date,
        publish_date=publish_date,
        restated=_text(row.get("Restated")),
        is_latest=is_latest,
    )


def _period_columns(
    spec: StatementSpec,
    periodic: pd.DataFrame,
    latest: pd.DataFrame,
    *,
    annual: bool,
    max_periods: int | None,
) -> tuple[list[PeriodColumn], pd.DataFrame]:
    """Columns in display order — leading column, then periods descending.

    The leading column is suppressed for the quarterly balance sheet, and only
    there: it is drawn from ``MRQ``, so against quarterly periods it *is* the
    first periodic column. Under quarterly income statement or cash flow the
    trailing-twelve-month figure is a genuinely different aggregate and stays.

    The test is the dimension, not the period stamp. Sharadar labels the ``MRT``
    row with the same fiscal period and report date as the quarter it ends on,
    so comparing period keys would wrongly drop every quarterly TTM column.
    """
    columns: list[PeriodColumn] = []
    rows: list[pd.Series] = []

    ordered = list(range(len(periodic)))[::-1]
    if max_periods is not None:
        ordered = ordered[:max_periods]

    duplicates_first = spec.latest_source == "MRQ" and not annual and bool(ordered)

    if not latest.empty:
        row = latest.iloc[-1]
        key = _text(latest.index[-1])
        if not duplicates_first:
            if spec.latest_source == "MRQ":
                fy = row.get("Fiscal Year")
                stamp = "" if fy is None or pd.isna(fy) else f"FY{int(fy)} {_text(row.get('Fiscal Period'))}"
                report = _text(row.get("Report Date"))
                sublabel = " · ".join(bit for bit in (stamp.strip(), report) if bit)
            else:
                report = _text(row.get("Report Date"))
                sublabel = f"to {report}" if report else ""
            columns.append(
                _column_from_row(
                    row, key, annual=annual, label=spec.latest_label,
                    sublabel=sublabel, is_latest=True,
                )
            )
            rows.append(row)

    for index in ordered:
        row = periodic.iloc[index]
        columns.append(
            _column_from_row(row, _text(periodic.index[index]), annual=annual)
        )
        rows.append(row)

    frame = (
        pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()
    )
    return columns, frame


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _derive(source: Source, values: Mapping[str, np.ndarray]) -> np.ndarray:
    """Arithmetic over the rows a derived row names.

    Plain numpy arithmetic, which **propagates NaN**. Never ``np.nansum``: that
    treats an unreported component as zero and produces a total that looks
    entirely reasonable and is wrong. A row whose input is missing is missing.
    This is the single most important function in the module.
    """
    if isinstance(source, Residual):
        result = values[source.base].copy()
        for key in source.less:
            result = result - values[key]
        return result

    if isinstance(source, Product):
        result = values[source.factors[0]].copy()
        for key in source.factors[1:]:
            result = result * values[key]
        return result

    if isinstance(source, Ratio):
        numerator = values[source.numerator]
        denominator = values[source.denominator]
        usable = denominator > 0 if source.require_positive_denominator else denominator != 0
        # A zero denominator is missing, never infinity. NaN on either side
        # fails `usable` or propagates through the division, as it must.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(usable, numerator / denominator, np.nan) * source.scale

    result = np.zeros_like(values[source.plus[0]])
    for key in source.plus:
        result = result + values[key]
    for key in source.minus:
        result = result - values[key]
    return result


def _dependencies(source: Source) -> tuple[str, ...]:
    if isinstance(source, Residual):
        return (source.base, *source.less)
    if isinstance(source, Sum):
        return (*source.plus, *source.minus)
    if isinstance(source, Ratio):
        return (source.numerator, source.denominator)
    if isinstance(source, Product):
        return source.factors
    return ()


def _evaluate(
    spec: StatementSpec, frame: pd.DataFrame, periods: int
) -> tuple[dict[str, np.ndarray], dict[str, Origin]]:
    """Every row's values, keyed by row key.

    Two phases, because a plug row is displayed *above* the subtotal it plugs
    off — ``opex_other`` sits above ``opex``, ``ncfo_other`` above ``ncfo``. So
    reported rows resolve first, in any order, and derived rows then evaluate in
    declaration order. The invariant a spec must satisfy is therefore that a
    derived row may name any reported row, or a derived row declared earlier.
    """
    blank = np.full(periods, np.nan)
    values: dict[str, np.ndarray] = {}
    origins: dict[str, Origin] = {}

    for row in spec.rows():
        if not isinstance(row.source, FromCaption):
            continue
        caption = row.source.caption
        if caption in frame.columns:
            values[row.key] = pd.to_numeric(frame[caption], errors="coerce").to_numpy(
                dtype="float64"
            )
            origins[row.key] = row.inferred_origin
        else:
            # Sharadar returned no such indicator at all — distinct from
            # returning it empty, and worth saying so in the sources panel.
            values[row.key] = blank.copy()
            origins[row.key] = Origin.UNAVAILABLE

    for row in spec.rows():
        if isinstance(row.source, FromCaption):
            continue
        values[row.key] = _derive(row.source, values)
        origins[row.key] = row.inferred_origin

    return values, origins


def _reasons(
    row: RowSpec,
    values: Mapping[str, np.ndarray],
    labels: Mapping[str, str],
    periods: int,
) -> list[str]:
    """Per-period explanation of why a derived row has no figure.

    A blank cell under the missing-stays-missing rule is correct but a dead end
    for the reader, so the components that were absent are named. This matters
    concretely: ``sbcomp`` is outside ``ZERO_FILLED_INDICATORS``, so a company
    with no share-based compensation line reports nothing rather than zero, and
    Other Operating Activities vanishes for it.
    """
    keys = _dependencies(row.source)
    if not keys:
        return [""] * periods

    out = []
    for index in range(periods):
        if not np.isnan(values[row.key][index]):
            out.append("")
            continue
        absent = [labels[key] for key in keys if np.isnan(values[key][index])]
        out.append(
            f"not computed — {', '.join(absent)} not reported" if absent else ""
        )
    return out


def _formula(row: RowSpec, labels: Mapping[str, str]) -> str:
    """The arithmetic behind a row, in the labels the reader sees.

    An unknown key falls back to the key itself rather than raising:
    :func:`validate_spec` is the guard against a dangling reference, and a
    tooltip is not worth taking the page down for.
    """
    def name(key: str) -> str:
        return labels.get(key, key)

    source = row.source
    if isinstance(source, Residual):
        return " − ".join([name(source.base), *(name(k) for k in source.less)])
    if isinstance(source, Product):
        return " × ".join(name(key) for key in source.factors)
    if isinstance(source, Ratio):
        sign = "−" if source.scale < 0 else ""
        text = f"{sign}{name(source.numerator)} ÷ {name(source.denominator)}"
        if source.require_positive_denominator:
            text += f", where {name(source.denominator)} is positive"
        return text
    if isinstance(source, Sum):
        text = " + ".join(name(key) for key in source.plus)
        for key in source.minus:
            text += f" − {name(key)}"
        return text
    return ""


def _check_formula(check: Check, labels: Mapping[str, str]) -> str:
    """The identity being tested, in the row labels the reader sees."""
    return _formula(RowSpec("", "", check.expected), labels)


def _floor_for(unit: Unit, tolerances: StatementTolerances) -> float:
    """The absolute floor appropriate to what the row measures.

    Money and share counts are reported in whole units and scale into the
    millions; per-share figures, rates and multiples do not. One floor for both
    would be either far too loose for the second group or far too tight for the
    first.
    """
    if unit in (Unit.CURRENCY, Unit.SHARES):
        return tolerances.absolute_floor
    return tolerances.unscaled_floor


def _reconcile(
    row: RowSpec,
    values: Mapping[str, np.ndarray],
    labels: Mapping[str, str],
    columns: list[PeriodColumn],
    tolerance: float,
    floor: float,
) -> ReconcileResult:
    """Reported subtotal against the rows that should make it up.

    A check whose inputs are absent reports *not evaluated* rather than passing,
    mirroring the contract the DCF validation layer holds itself to.
    """
    check = row.check
    assert check is not None
    formula = _check_formula(check, labels)

    expected = _derive(check.expected, values)
    actual = values[row.key]
    usable = ~(np.isnan(expected) | np.isnan(actual))

    if not usable.any():
        return ReconcileResult(
            row_key=row.key, row_label=row.label, status="not evaluated",
            formula=formula, periods_checked=0, tolerance=tolerance,
            detail="No period has both the reported figure and every component.",
        )

    gap = np.abs(actual - expected)
    scale = np.maximum(np.abs(actual), np.abs(expected))
    limit = np.maximum(tolerance * scale, floor)
    breach = usable & (gap > limit)

    worst_index = int(np.nanargmax(np.where(usable, gap, np.nan)))
    worst_gap = float(gap[worst_index])
    worst_scale = float(scale[worst_index])
    worst_relative = worst_gap / worst_scale if worst_scale else None

    if not breach.any():
        return ReconcileResult(
            row_key=row.key, row_label=row.label, status="ok", formula=formula,
            periods_checked=int(usable.sum()), tolerance=tolerance,
            worst_period=columns[worst_index].label, worst_gap=worst_gap,
            worst_relative=worst_relative, detail=check.note,
        )

    worst_index = int(np.nanargmax(np.where(breach, gap, np.nan)))
    worst_gap = float(gap[worst_index])
    worst_scale = float(scale[worst_index])
    worst_relative = worst_gap / worst_scale if worst_scale else None
    detail = (
        f"Reported {actual[worst_index]:,.0f} vs computed "
        f"{expected[worst_index]:,.0f} in {columns[worst_index].label}"
    )
    if worst_relative is not None:
        detail += f" — {worst_relative:.1%} apart, tolerance {tolerance:.1%}"
    if check.note:
        detail += f". {check.note}"
    return ReconcileResult(
        row_key=row.key, row_label=row.label, status="gap", formula=formula,
        periods_checked=int(usable.sum()), tolerance=tolerance,
        worst_period=columns[worst_index].label, worst_gap=worst_gap,
        worst_relative=worst_relative, detail=detail,
    )


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def format_value(value: float, unit: Unit, units: UnitScale) -> str:
    """One figure as the reader sees it.

    Only money and share counts take the units divisor. Scaling a whole frame
    uniformly — which is what a naive implementation does — turns EPS into 0.0
    and a 46% margin into 0.0%.
    """
    # Negating a reported zero on ingest leaves -0.0, which prints as "-0" and
    # reads as a rounded-away negative rather than the nil it is.
    if value == 0.0:
        value = 0.0
    if unit in (Unit.CURRENCY, Unit.SHARES):
        return f"{value / units.divisor:,.{units.decimals}f}"
    if unit is Unit.PERCENT:
        return f"{value:.1%}"
    return f"{value:,.2f}"


def _cells(
    row: RowSpec,
    values: Mapping[str, np.ndarray],
    reasons: list[str],
    units: UnitScale,
    zero_filled: bool,
) -> tuple[Cell, ...]:
    series = values[row.key]
    cells = []
    for index in range(len(series)):
        value = float(series[index])
        if np.isnan(value):
            cells.append(
                Cell(text=MISSING_TEXT, value=None, missing=True, reason=reasons[index])
            )
            continue
        cells.append(
            Cell(
                text=format_value(value, row.unit, units),
                value=value,
                missing=False,
                zero_filled=zero_filled and value == 0.0,
            )
        )
    return tuple(cells)


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def build_statement(
    spec: StatementSpec,
    *,
    periodic: Mapping[str, pd.DataFrame],
    latest: Mapping[str, pd.DataFrame] | None = None,
    annual: bool = True,
    currency: str = "",
    units: UnitScale | None = None,
    max_periods: int | None = 8,
    show_empty: bool = False,
    tolerances: StatementTolerances = STATEMENT_TOLERANCES,
) -> StatementTable:
    """One statement, laid out and reconciled.

    ``periodic`` is the ``MRY`` or ``MRQ`` frame set; ``latest`` feeds the
    leading column and is the ``MRT`` set for flows, the ``MRQ`` set for the
    balance sheet. Returns an empty :class:`StatementTable` rather than raising
    when nothing came back, so the caller can say so.
    """
    units = units or UNITS["Millions"]
    dimension = ("MRY" if annual else "MRQ")
    empty = StatementTable(
        statement=spec.key, label=spec.label, dimension=dimension,
        currency=currency, units=units,
    )

    periodic_wide = _wide(periodic)
    latest_wide = _wide(latest or {})
    if periodic_wide.empty and latest_wide.empty:
        return empty

    columns, frame = _period_columns(
        spec, periodic_wide, latest_wide, annual=annual, max_periods=max_periods
    )
    if not columns:
        return empty

    values, origins = _evaluate(spec, frame, len(columns))
    labels = {row.key: row.label for row in spec.rows()}
    tolerance = getattr(tolerances, _TOLERANCE_FIELD[spec.key])

    sections: list[RenderedSection] = []
    diagnostics: list[ReconcileResult] = []
    reported = 0
    hidden = 0

    for section in spec.sections:
        rendered: list[RenderedRow] = []
        for row in section.rows:
            reasons = _reasons(row, values, labels, len(columns))
            zero_filled = isinstance(row.source, FromCaption) and sharadar_client.is_zero_filled(
                row.source.caption
            )
            cells = _cells(row, values, reasons, units, zero_filled)

            result = None
            if row.check is not None:
                result = _reconcile(
                    row, values, labels, columns, tolerance,
                    _floor_for(row.unit, tolerances),
                )
                diagnostics.append(result)

            if all(cell.missing for cell in cells):
                hidden += 1
                if not show_empty:
                    continue
            else:
                reported += 1

            rendered.append(
                RenderedRow(
                    key=row.key,
                    label=row.label,
                    emphasis=row.emphasis,
                    unit=row.unit,
                    indent=row.indent,
                    origin=origins[row.key],
                    formula=_formula(row, labels),
                    note=row.note,
                    cells=cells,
                    reconcile=result,
                )
            )
        if rendered:
            sections.append(
                RenderedSection(title=section.title, note=section.note, rows=tuple(rendered))
            )

    meta_columns = [c for c in _META_OUT if c in frame.columns]
    meta = frame[meta_columns].copy() if meta_columns else pd.DataFrame()
    if not meta.empty:
        meta.insert(0, "Column", [column.label for column in columns])

    return StatementTable(
        statement=spec.key,
        label=spec.label,
        dimension=dimension,
        currency=currency,
        units=units,
        periods=tuple(columns),
        sections=tuple(sections),
        reported_rows=reported,
        hidden_rows=hidden,
        diagnostics=tuple(diagnostics),
        meta=meta,
    )


def validate_spec(spec: StatementSpec) -> list[str]:
    """Structural problems with a statement spec, as human-readable strings.

    Exercised by the tests rather than at import: a caption typo or a dangling
    key should fail a named test, not break the page at start-up with a
    traceback.
    """
    problems: list[str] = []
    captions = set(sharadar_client.SF1_CAPTIONS.values()) | {
        sharadar_client.MINORITY_INTEREST_CAPTION
    }

    seen: set[str] = set()
    reported: set[str] = {
        row.key for row in spec.rows() if isinstance(row.source, FromCaption)
    }

    for row in spec.rows():
        if row.key in seen:
            problems.append(f"{spec.key}: duplicate row key {row.key!r}")
        if isinstance(row.source, FromCaption) and row.source.caption not in captions:
            problems.append(
                f"{spec.key}.{row.key}: no Sharadar caption {row.source.caption!r}"
            )
        for named in _dependencies(row.source):
            if named not in reported and named not in seen:
                problems.append(
                    f"{spec.key}.{row.key}: names {named!r}, which is neither a "
                    f"reported row nor declared earlier"
                )
        if row.check is not None:
            for named in _dependencies(row.check.expected):
                if named not in reported and named not in seen:
                    problems.append(
                        f"{spec.key}.{row.key}: check names {named!r}, which is "
                        f"neither a reported row nor declared earlier"
                    )
        seen.add(row.key)

    # A subtotal that a residual in the same statement plugs off cannot also be
    # checked against that residual: the identity would hold by construction and
    # the check would be decoration.
    plugged: dict[str, set[str]] = {}
    for row in spec.rows():
        if isinstance(row.source, Residual):
            plugged.setdefault(row.source.base, set()).add(row.key)
    for row in spec.rows():
        if row.check is None:
            continue
        named = set(_dependencies(row.check.expected))
        overlap = named & plugged.get(row.key, set())
        if overlap:
            problems.append(
                f"{spec.key}.{row.key}: check names {sorted(overlap)}, which "
                f"plug off this very row — the identity cannot fail"
            )

    return problems


# ---------------------------------------------------------------------------
# The specs
# ---------------------------------------------------------------------------
#
# Captions below are Sharadar's own, from ``sharadar_client.SF1_LAYOUT``. Labels
# are ours, and are allowed to read better than the vendor's.
#
# Costs arrive negative — ``sharadar_client.SIGN_FLIPPED`` negates them on
# ingest — so most statement identities are plain sums: gross profit is revenue
# *plus* cost of revenue. The exceptions are ``netincnci`` and ``prefdivis``,
# which Sharadar publishes as positive magnitudes and the client leaves alone.


INCOME_STATEMENT = StatementSpec(
    key="PL",
    label="Income statement",
    latest_label="TTM",
    latest_source="MRT",
    sections=(
        SectionSpec("Revenues", (
            RowSpec("revenue", "Revenues", FromCaption("Revenues")),
            RowSpec("cor", "Cost of Revenue", FromCaption("Cost of Revenue"), indent=1),
            RowSpec("gp", "Gross Profit", FromCaption("Gross Profit"),
                    emphasis=Emphasis.SUBTOTAL, check=Check(Sum(("revenue", "cor")))),
        )),
        SectionSpec("Operating Expenses & Income", (
            RowSpec("sgna", "Selling, General & Administrative",
                    FromCaption("Selling General and Administrative Expense"), indent=1),
            RowSpec("rnd", "Research & Development",
                    FromCaption("Research and Development Expense"), indent=1),
            RowSpec("opex_other", "Other Operating Expenses",
                    Residual("opex", ("sgna", "rnd")), indent=1),
            RowSpec("opex", "Total Operating Expenses", FromCaption("Operating Expenses"),
                    emphasis=Emphasis.SUBTOTAL),
            RowSpec("opinc", "Operating Income", FromCaption("Operating Income"),
                    emphasis=Emphasis.SUBTOTAL, check=Check(Sum(("gp", "opex")))),
        )),
        SectionSpec("Earnings from Continuing Operations", (
            RowSpec("nonop", "Non-Operating Income", Residual("ebit", ("opinc",)),
                    indent=1,
                    note="EBIT less operating income — what Sharadar counts before "
                         "interest and tax but not as operating."),
            RowSpec("ebit", "Earnings Before Interest & Taxes (EBIT)",
                    FromCaption("Earning Before Interest & Taxes (EBIT)"),
                    emphasis=Emphasis.SUBTOTAL),
            RowSpec("intexp", "Interest Expense", FromCaption("Interest Expense"), indent=1),
            RowSpec("ebt", "Earnings before Tax", FromCaption("Earnings before Tax"),
                    emphasis=Emphasis.SUBTOTAL, check=Check(Sum(("ebit", "intexp")))),
            RowSpec("taxexp", "Income Tax Expense", FromCaption("Income Tax Expense"),
                    indent=1),
            RowSpec("consolinc", "Consolidated Income", FromCaption("Consolidated Income"),
                    emphasis=Emphasis.SUBTOTAL, check=Check(Sum(("ebt", "taxexp")))),
        )),
        SectionSpec("Net Income", (
            RowSpec("netincnci", "Net Income to Non-Controlling Interests",
                    FromCaption("Net Income to Non-Controlling Interests"), indent=1),
            RowSpec("netinc", "Net Income", FromCaption("Net Income"),
                    emphasis=Emphasis.TOTAL,
                    check=Check(Sum(("consolinc",), ("netincnci",)),
                                note="Sharadar publishes non-controlling income as a "
                                     "positive magnitude, so it subtracts here.")),
            RowSpec("prefdivis", "Preferred Dividends",
                    FromCaption("Preferred Dividends Income Statement Impact"), indent=1),
            RowSpec("netinccmn", "Net Income to Common Stockholders",
                    FromCaption("Net Income Common Stock"), emphasis=Emphasis.TOTAL,
                    check=Check(Sum(("netinc",), ("prefdivis",)))),
        )),
        SectionSpec(
            "Supplemental Items",
            (
                RowSpec("sps", "Revenue per Share", FromCaption("Sales per Share"),
                        unit=Unit.PER_SHARE),
                RowSpec("netincdis", "Net Income (Loss) from Discontinued Operations",
                        FromCaption("Net Loss Income from Discontinued Operations")),
                RowSpec("netinc_continuing", "Net Income from Continuing Operations",
                        Sum(("netinccmn",), ("netincdis",)),
                        note="Income to common stockholders less the discontinued "
                             "portion. This is as close to a normalised earnings "
                             "figure as SF1 allows: Sharadar breaks out no other "
                             "non-recurring items."),
                RowSpec("eps", "Basic EPS", FromCaption("Earnings per Basic Share"),
                        unit=Unit.PER_SHARE),
                RowSpec("eps_continuing", "Basic EPS, Continuing Operations",
                        Ratio("netinc_continuing", "shareswa"), unit=Unit.PER_SHARE),
                RowSpec("shareswa", "Basic Weighted Average Shares Outstanding",
                        FromCaption("Weighted Average Shares"), unit=Unit.SHARES),
                RowSpec("epsdil", "Diluted EPS",
                        FromCaption("Earnings per Diluted Share"), unit=Unit.PER_SHARE),
                RowSpec("epsdil_continuing", "Diluted EPS, Continuing Operations",
                        Ratio("netinc_continuing", "shareswadil"), unit=Unit.PER_SHARE),
                RowSpec("shareswadil", "Diluted Weighted Average Shares Outstanding",
                        FromCaption("Weighted Average Shares Diluted"),
                        unit=Unit.SHARES),
                RowSpec("dps", "Dividends per Share",
                        FromCaption("Dividends per Basic Common Share"),
                        unit=Unit.PER_SHARE),
                RowSpec("payoutratio", "Payout Ratio", FromCaption("Payout Ratio"),
                        unit=Unit.PERCENT),
                RowSpec("ebitda", "EBITDA",
                        FromCaption("Earnings Before Interest Taxes & Depreciation "
                                    "Amortization (EBITDA)")),
                RowSpec("effective_tax_rate", "Effective Tax Rate",
                        # Tax expense is negative downstream of ingest; a rate reads
                        # as a positive percentage. A pretax loss makes the rate
                        # meaningless rather than merely negative, so it is withheld.
                        Ratio("taxexp", "ebt", scale=-1.0,
                              require_positive_denominator=True),
                        unit=Unit.PERCENT),
            ),
            note="EBIT appears above, under Earnings from Continuing Operations.",
        ),
    ),
)


BALANCE_SHEET = StatementSpec(
    key="BS",
    label="Balance sheet",
    # A balance sheet is a point-in-time snapshot, so there is no such thing as
    # a trailing-twelve-month one. Sharadar's MRT dimension simply repeats the
    # latest quarter-end balance; reading MRQ instead gives the same figures
    # with a period label that can be checked.
    latest_label="Latest (MRQ)",
    latest_source="MRQ",
    sections=(
        SectionSpec("Cash & Short-Term Investments", (
            RowSpec("cashneq", "Cash and Equivalents", FromCaption("Cash and Equivalents"),
                    indent=1),
            RowSpec("investmentsc", "Short-Term Investments",
                    FromCaption("Investments Current"), indent=1),
            RowSpec("cash_sti", "Cash & Short-Term Investments",
                    Sum(("cashneq", "investmentsc")), emphasis=Emphasis.SUBTOTAL),
        )),
        SectionSpec("Receivables & Current Assets", (
            RowSpec("receivables", "Trade and Non-Trade Receivables",
                    FromCaption("Trade and Non-Trade Receivables"), indent=1),
            RowSpec("inventory", "Inventory", FromCaption("Inventory"), indent=1),
            RowSpec("assetsc_other", "Other Current Assets",
                    Residual("assetsc",
                             ("cashneq", "investmentsc", "receivables", "inventory")),
                    indent=1),
            RowSpec("assetsc", "Total Current Assets", FromCaption("Current Assets"),
                    emphasis=Emphasis.SUBTOTAL),
        )),
        SectionSpec("Long-Term Assets", (
            RowSpec("investmentsnc", "Investments, Non-Current",
                    FromCaption("Investments Non-Current"), indent=1),
            RowSpec("ppnenet", "Property, Plant & Equipment, Net",
                    FromCaption("Property Plant & Equipment Net"), indent=1),
            RowSpec("intangibles", "Goodwill and Intangible Assets",
                    FromCaption("Goodwill and Intangible Assets"), indent=1),
            RowSpec("taxassets", "Tax Assets", FromCaption("Tax Assets"), indent=1),
            RowSpec("assetsnc_other", "Other Non-Current Assets",
                    Residual("assetsnc",
                             ("investmentsnc", "ppnenet", "intangibles", "taxassets")),
                    indent=1),
            RowSpec("assetsnc", "Total Non-Current Assets",
                    FromCaption("Assets Non-Current"), emphasis=Emphasis.SUBTOTAL),
            RowSpec("assets", "Total Assets", FromCaption("Total Assets"),
                    emphasis=Emphasis.TOTAL, check=Check(Sum(("assetsc", "assetsnc")))),
        )),
        SectionSpec("Current Liabilities", (
            RowSpec("payables", "Trade and Non-Trade Payables",
                    FromCaption("Trade and Non-Trade Payables"), indent=1),
            RowSpec("deferredrev", "Deferred Revenue", FromCaption("Deferred Revenue"),
                    indent=1),
            RowSpec("debtc", "Debt, Current", FromCaption("Debt Current"), indent=1),
            RowSpec("liabilitiesc_other", "Other Current Liabilities",
                    Residual("liabilitiesc", ("payables", "deferredrev", "debtc")),
                    indent=1),
            RowSpec("liabilitiesc", "Total Current Liabilities",
                    FromCaption("Current Liabilities"), emphasis=Emphasis.SUBTOTAL),
        )),
        SectionSpec("Long-Term Liabilities", (
            RowSpec("debtnc", "Debt, Non-Current", FromCaption("Debt Non-Current"),
                    indent=1),
            RowSpec("taxliabilities", "Tax Liabilities", FromCaption("Tax Liabilities"),
                    indent=1),
            RowSpec("liabilitiesnc_other", "Other Non-Current Liabilities",
                    Residual("liabilitiesnc", ("debtnc", "taxliabilities")), indent=1),
            RowSpec("liabilitiesnc", "Total Non-Current Liabilities",
                    FromCaption("Liabilities Non-Current"), emphasis=Emphasis.SUBTOTAL),
            RowSpec("liabilities", "Total Liabilities", FromCaption("Total Liabilities"),
                    emphasis=Emphasis.TOTAL,
                    check=Check(Sum(("liabilitiesc", "liabilitiesnc")))),
        )),
        SectionSpec("Common Equity", (
            RowSpec("retearn", "Accumulated Retained Earnings (Deficit)",
                    FromCaption("Accumulated Retained Earnings (Deficit)"), indent=1),
            RowSpec("accoci", "Accumulated Other Comprehensive Income",
                    FromCaption("Accumulated Other Comprehensive Income"), indent=1),
            RowSpec("equity_other", "Contributed Capital & Other",
                    Residual("equity", ("retearn", "accoci")), indent=1),
        )),
        SectionSpec("Stockholders' Equity", (
            RowSpec("equity", "Shareholders' Equity Attributable to Parent",
                    FromCaption("Shareholders Equity Attributable to Parent"),
                    emphasis=Emphasis.SUBTOTAL),
            RowSpec("nci", "Minority Interest",
                    FromCaption(sharadar_client.MINORITY_INTEREST_CAPTION),
                    indent=1, origin=Origin.DERIVED,
                    note="SF1 has no balance-sheet minority-interest line. The client "
                         "recovers it as total assets less liabilities less parent "
                         "equity, which is what Sharadar's own definition implies."),
            RowSpec("equity_total", "Total Equity", Sum(("equity", "nci")),
                    emphasis=Emphasis.TOTAL),
        )),
        SectionSpec(
            "Liabilities & Equity",
            (
                RowSpec("liab_and_equity", "Total Liabilities & Equity",
                        Sum(("liabilities", "equity_total")), emphasis=Emphasis.TOTAL,
                        note="Equals total assets by construction, not by coincidence: "
                             "minority interest is derived as exactly that residual, so "
                             "this identity cannot fail and is shown rather than checked."),
            ),
        ),
        SectionSpec("Supplemental Items", (
            RowSpec("sharesbas", "Total Shares Outstanding on Filing Date",
                    FromCaption("Shares (Basic)"), unit=Unit.SHARES,
                    note="The count stated on the cover of the 10-K or 10-Q, "
                         "adjusted for splits."),
            RowSpec("sharefactor", "Share Factor", FromCaption("Share Factor"),
                    unit=Unit.RATIO,
                    note="Corrects for ADRs representing a fraction or multiple of "
                         "an underlying share, and for multi-class structures. 1.0 "
                         "for most US companies."),
            RowSpec("shares_out", "Total Common Shares Outstanding",
                    Product(("sharesbas", "sharefactor")), unit=Unit.SHARES),
            RowSpec("cash_per_share", "Cash per Share",
                    Ratio("cash_sti", "shares_out"), unit=Unit.PER_SHARE),
            RowSpec("bvps", "Book Value per Share",
                    FromCaption("Book Value per Share"), unit=Unit.PER_SHARE),
            RowSpec("tangible_bv", "Tangible Book Value",
                    Sum(("equity",), ("intangibles",)),
                    note="Parent equity less goodwill and intangibles. Distinct from "
                         "Sharadar's Tangible Asset Value, which nets intangibles off "
                         "total assets rather than off equity."),
            RowSpec("tangible_bvps", "Tangible Book Value per Share",
                    Ratio("tangible_bv", "shares_out"), unit=Unit.PER_SHARE),
            RowSpec("debt", "Total Debt", FromCaption("Total Debt"),
                    check=Check(Sum(("debtc", "debtnc")))),
            RowSpec("net_debt", "Net Debt", Sum(("debt",), ("cash_sti",))),
            RowSpec("investments", "Investments, Total", FromCaption("Investments"),
                    check=Check(Sum(("investmentsc", "investmentsnc")))),
            RowSpec("deposits", "Deposit Liabilities", FromCaption("Deposit Liabilities"),
                    note="Sharadar does not classify deposits as current or "
                         "non-current, so they are shown here rather than netted into "
                         "either bucket's residual. For a deposit-taking institution "
                         "Other Current Liabilities will therefore be large."),
        )),
    ),
)


CASH_FLOW = StatementSpec(
    key="CF",
    label="Cash flow",
    latest_label="TTM",
    latest_source="MRT",
    sections=(
        SectionSpec("Net Income", (
            # An income-statement caption. The merged frame makes it a plain
            # lookup; see `_wide`.
            RowSpec("netinc", "Net Income", FromCaption("Net Income"),
                    emphasis=Emphasis.SUBTOTAL),
        )),
        SectionSpec("Cash Flow from Operating Activities", (
            RowSpec("depamor", "Depreciation, Amortization & Accretion",
                    FromCaption("Depreciation Amortization & Accretion"), indent=1),
            RowSpec("sbcomp", "Share-Based Compensation",
                    FromCaption("Share Based Compensation"), indent=1),
            RowSpec("ncfo_other", "Other Operating Activities",
                    Residual("ncfo", ("netinc", "depamor", "sbcomp")), indent=1),
            RowSpec("ncfo", "Net Cash Flow from Operations",
                    FromCaption("Net Cash Flow from Operations"),
                    emphasis=Emphasis.SUBTOTAL),
        )),
        SectionSpec("Cash Flow from Investing Activities", (
            RowSpec("capex", "Capital Expenditure", FromCaption("Capital Expenditure"),
                    indent=1),
            RowSpec("ncfbus", "Business Acquisitions & Disposals",
                    FromCaption("Net Cash Flow - Business Acquisitions and Disposals"),
                    indent=1),
            RowSpec("ncfinv", "Investment Acquisitions & Disposals",
                    FromCaption("Net Cash Flow - Investment Acquisitions and Disposals"),
                    indent=1),
            RowSpec("ncfi_other", "Other Investing Activities",
                    Residual("ncfi", ("capex", "ncfbus", "ncfinv")), indent=1),
            RowSpec("ncfi", "Net Cash Flow from Investing",
                    FromCaption("Net Cash Flow from Investing"),
                    emphasis=Emphasis.SUBTOTAL),
        )),
        SectionSpec("Cash Flow from Financing Activities", (
            RowSpec("ncfdebt", "Issuance (Repayment) of Debt Securities",
                    FromCaption("Issuance (Repayment) of Debt Securities"), indent=1),
            RowSpec("ncfcommon", "Issuance (Purchase) of Equity Shares",
                    FromCaption("Issuance (Purchase) of Equity Shares"), indent=1),
            RowSpec("ncfdiv", "Payment of Dividends & Other Cash Distributions",
                    FromCaption("Payment of Dividends & Other Cash Distributions"),
                    indent=1),
            RowSpec("ncff_other", "Other Financing Activities",
                    Residual("ncff", ("ncfdebt", "ncfcommon", "ncfdiv")), indent=1),
            RowSpec("ncff", "Net Cash Flow from Financing",
                    FromCaption("Net Cash Flow from Financing"),
                    emphasis=Emphasis.SUBTOTAL),
        )),
        SectionSpec("Change in Cash & Equivalents", (
            RowSpec("ncfx", "Effect of Exchange Rate Changes on Cash",
                    FromCaption("Effect of Exchange Rate Changes on Cash"), indent=1),
            RowSpec("ncf", "Net Change in Cash & Equivalents",
                    FromCaption("Net Cash Flow / Change in Cash & Cash Equivalents"),
                    emphasis=Emphasis.TOTAL,
                    check=Check(Sum(("ncfo", "ncfi", "ncff", "ncfx")))),
        )),
        SectionSpec(
            "Supplemental Items",
            (
                RowSpec("fcf", "Free Cash Flow (Levered)",
                        FromCaption("Free Cash Flow"),
                        note="Operating cash flow less capital expenditure. Levered: "
                             "operating cash flow is already net of interest paid.",
                        check=Check(Sum(("ncfo", "capex")),
                                    note="Sharadar's own definition; capital "
                                         "expenditure is already signed negative.")),
                # The divisor of the per-share row below, spelled out here
                # rather than a tab away. `shareswa` is an income-statement
                # caption and `sharefactor` a metrics one; the merged frame
                # makes both plain lookups.
                RowSpec("shareswa", "Basic Weighted Average Shares Outstanding",
                        FromCaption("Weighted Average Shares"), unit=Unit.SHARES),
                RowSpec("sharefactor", "Share Factor", FromCaption("Share Factor"),
                        unit=Unit.RATIO,
                        note="1.0 for most companies. An ADR representing a "
                             "fraction or multiple of an underlying share carries "
                             "the ratio here — TSM's is 0.2, five ordinary shares "
                             "to the ADR."),
                RowSpec("shareswa_adj",
                        "Basic Weighted Average Shares, Share-Factor Adjusted",
                        Product(("shareswa", "sharefactor")), unit=Unit.SHARES),
                RowSpec("fcfps", "Free Cash Flow per Share (Levered)",
                        FromCaption("Free Cash Flow per Share"),
                        unit=Unit.PER_SHARE,
                        # Sharadar documents no divisor for `fcfps`, so the
                        # identity is checked rather than asserted. The share
                        # factor is not optional detail: without it every US
                        # company ties and every ADR is out by the ADR ratio —
                        # TSM by a factor of five.
                        check=Check(Ratio("fcf", "shareswa_adj"),
                                    note="Levered free cash flow — operating cash "
                                         "flow is already net of interest paid — "
                                         "over share-factor-adjusted basic "
                                         "weighted average shares.")),
            ),
            note="SF1 carries no cash-interest-paid or cash-tax-paid disclosure, so "
                 "unlevered free cash flow cannot be built here without assuming a "
                 "tax rate — the DCF tab computes it properly. Net capital "
                 "expenditure and net debt issuance are already shown above.",
        ),
    ),
)


DERIVED_FIGURES = StatementSpec(
    key="DERIVED",
    label="Derived figures",
    latest_label="TTM",
    latest_source="MRT",
    sections=(
        SectionSpec("Earnings & Cash Flow", (
            RowSpec("ebit", "Earnings Before Interest & Taxes (EBIT)",
                    FromCaption("Earning Before Interest & Taxes (EBIT)"), indent=1),
            RowSpec("depamor", "Depreciation, Amortization & Accretion",
                    FromCaption("Depreciation Amortization & Accretion"), indent=1),
            RowSpec("ebitda", "EBITDA",
                    FromCaption("Earnings Before Interest Taxes & Depreciation "
                                "Amortization (EBITDA)"),
                    emphasis=Emphasis.SUBTOTAL, check=Check(Sum(("ebit", "depamor")))),
            RowSpec("fcf", "Free Cash Flow", FromCaption("Free Cash Flow")),
            RowSpec("workingcapital", "Working Capital", FromCaption("Working Capital")),
            RowSpec("invcap", "Invested Capital", FromCaption("Invested Capital")),
            RowSpec("tangibles", "Tangible Asset Value",
                    FromCaption("Tangible Asset Value")),
        )),
        SectionSpec(
            "Capital & Market Value",
            (
                RowSpec("sharesbas", "Shares (Basic)", FromCaption("Shares (Basic)"),
                        unit=Unit.SHARES),
                RowSpec("sharefactor", "Share Factor", FromCaption("Share Factor"),
                        unit=Unit.RATIO),
                RowSpec("price", "Share Price (Adjusted Close)",
                        FromCaption("Share Price (Adjusted Close)"), unit=Unit.PER_SHARE),
                # Raw, not millions. The millions conversion in the client is on
                # the DAILY table only; SF1 publishes these unscaled.
                RowSpec("marketcap", "Market Capitalization",
                        FromCaption("Market Capitalization")),
                RowSpec("ev", "Enterprise Value", FromCaption("Enterprise Value")),
            ),
            note="Point-in-time as of the period end, not a period aggregate. Under "
                 "the TTM column these are the latest period end.",
        ),
        SectionSpec("Margins & Returns", (
            RowSpec("grossmargin", "Gross Margin", FromCaption("Gross Margin"),
                    unit=Unit.PERCENT),
            RowSpec("ebitdamargin", "EBITDA Margin", FromCaption("EBITDA Margin"),
                    unit=Unit.PERCENT),
            RowSpec("netmargin", "Profit Margin", FromCaption("Profit Margin"),
                    unit=Unit.PERCENT),
            RowSpec("ros", "Return on Sales", FromCaption("Return on Sales"),
                    unit=Unit.PERCENT),
            RowSpec("roe", "Return on Average Equity",
                    FromCaption("Return on Average Equity"), unit=Unit.PERCENT),
            RowSpec("roa", "Return on Average Assets",
                    FromCaption("Return on Average Assets"), unit=Unit.PERCENT),
            RowSpec("roic", "Return on Invested Capital",
                    FromCaption("Return on Invested Capital"), unit=Unit.PERCENT),
            RowSpec("assetturnover", "Asset Turnover", FromCaption("Asset Turnover"),
                    unit=Unit.RATIO),
            RowSpec("currentratio", "Current Ratio", FromCaption("Current Ratio"),
                    unit=Unit.RATIO),
            RowSpec("de", "Debt to Equity Ratio", FromCaption("Debt to Equity Ratio"),
                    unit=Unit.RATIO),
        )),
        SectionSpec("Multiples", (
            RowSpec("evebit", "EV / EBIT", FromCaption("Enterprise Value over EBIT"),
                    unit=Unit.RATIO),
            RowSpec("evebitda", "EV / EBITDA",
                    FromCaption("Enterprise Value over EBITDA"), unit=Unit.RATIO),
            RowSpec("pe1", "Price / Earnings", FromCaption("Price to Earnings Ratio"),
                    unit=Unit.RATIO),
            RowSpec("pb", "Price / Book", FromCaption("Price to Book Value"),
                    unit=Unit.RATIO),
            RowSpec("ps1", "Price / Sales", FromCaption("Price to Sales Ratio"),
                    unit=Unit.RATIO),
        )),
        SectionSpec("Per Share & Distributions", (
            RowSpec("bvps", "Book Value per Share", FromCaption("Book Value per Share"),
                    unit=Unit.PER_SHARE),
            RowSpec("fcfps", "Free Cash Flow per Share",
                    FromCaption("Free Cash Flow per Share"), unit=Unit.PER_SHARE),
            RowSpec("sps", "Sales per Share", FromCaption("Sales per Share"),
                    unit=Unit.PER_SHARE),
            RowSpec("divyield", "Dividend Yield", FromCaption("Dividend Yield"),
                    unit=Unit.PERCENT),
            RowSpec("payoutratio", "Payout Ratio", FromCaption("Payout Ratio"),
                    unit=Unit.PERCENT),
        )),
        SectionSpec("Reference", (
            RowSpec("fxusd", "Foreign Currency to USD Exchange Rate",
                    FromCaption("Foreign Currency to USD Exchange Rate"),
                    unit=Unit.RATIO),
        )),
    ),
)


#: The four views, in the order an analyst reads them.
STATEMENTS: dict[str, StatementSpec] = {
    spec.label: spec
    for spec in (INCOME_STATEMENT, BALANCE_SHEET, CASH_FLOW, DERIVED_FIGURES)
}
