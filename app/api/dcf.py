"""🔎 Stock Research's 💰 DCF tab over HTTP: the assumption panel, the valuation, the export.

The tab is one request: the reader's overrides go up, and the resolved panel and
the valued run come back together, because every number in the panel can move
when one input does — a cost of debt nobody typed follows the risk-free rate.
:func:`app.logic.dcf.inputs.resolve` assembles the run from the overrides and
:func:`~app.logic.dcf.inputs.evaluate` values it; they are the functions the
Streamlit tab calls, so an untouched panel is the same run on both.

Two more routes sit beside it: a what-if price solved against the same run, and
the workbook or CSV export, which needs the reader's acknowledgements as well.

A company the model refuses — a bank, an unconvertible currency, a schema that
does not resolve, no annual history — is a normal response carrying
``blocked``, so the tab can say why; so is a run the model cannot build. A
ticker Sharadar does not know is a 404 and a failed Sharadar request a 502, as
on every route. An override the panel does not offer is a 422 naming it.

Like the rest of the API this caches nothing of its own. The statements, prices
and rates behind ``load_company`` are cached in the data layer, so a warm
request pays for the normalisation and the arithmetic only.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Response, status
from pydantic import Field as PydanticField

from app.api.contract import Contract, finite_or_none
from app.api.research import Frame, Provenance, Ticker, as_frame, as_provenance
from app.data import market_data
from app.logic.dcf import export as dcf_export
from app.logic.dcf import financials as fin
from app.logic.dcf import inputs as dcf_inputs
from app.logic.dcf import model as dcf_model
from app.logic.dcf import reverse
from app.logic.dcf import validation as val
from app.logic.dcf import valuation as dcf
from app.logic.dcf.schema import SchemaResolutionError
from app.provenance import Sourced

router = APIRouter(prefix="/research", tags=["dcf"])

DriverKey = Literal[dcf_inputs.DRIVER_KEYS]  # type: ignore[valid-type]
TargetKey = Literal[tuple(key for key, _ in dcf_inputs.TARGETS)]  # type: ignore[valid-type]
Method = Literal[tuple(dcf_inputs.HEADLINE_METHODS)]  # type: ignore[valid-type]
Frequency = Literal[dcf_inputs.BETA_FREQUENCIES]  # type: ignore[valid-type]
Rate = Annotated[float, PydanticField(ge=-1.0, le=5.0)]


def _today() -> date:
    """The valuation date. A function so a test can fix it."""
    return date.today()


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


class DcfOverrides(Contract):
    """What the reader changed; every field left out takes its default.

    Rates are fractions, not percentages. ``target_on`` turns a driver's year-N
    target on or off and ``target_values`` moves where it lands.
    """

    years: Annotated[int, PydanticField(ge=5, le=15)] | None = None
    bases: dict[DriverKey, str] = {}
    values: dict[DriverKey, Rate] = {}
    fade: bool | None = None
    hold: Annotated[int, PydanticField(ge=1, le=14)] | None = None
    target_on: dict[TargetKey, bool] = {}
    target_values: dict[TargetKey, Rate] = {}
    acquisitions: bool | None = None

    risk_free: Rate | None = None
    erp: Rate | None = None
    beta_frequency: Frequency | None = None
    blume: bool | None = None
    beta: Annotated[float, PydanticField(ge=-5.0, le=10.0)] | None = None
    cost_of_debt: Rate | None = None
    marginal_tax: Annotated[float, PydanticField(ge=0.0, le=1.0)] | None = None
    share_basis: str | None = None

    terminal_growth: Rate | None = None
    ronic_basis: str | None = None
    ronic: Annotated[float, PydanticField(gt=0.0, le=10.0)] | None = None
    terminal_at_marginal: bool | None = None
    ebita_proxy: bool | None = None
    exit_multiple: Annotated[float, PydanticField(gt=0.0, le=1000.0)] | None = None
    headline_method: Method | None = None
    mid_year: bool | None = None

    include_sti: bool | None = None
    cash_floor: Annotated[float, PydanticField(ge=0.0)] | None = None
    sbc: float | None = None
    non_operating: float | None = None

    def logic(self) -> dcf_inputs.Overrides:
        return dcf_inputs.Overrides(**self.model_dump())


class WhatIf(Contract):
    overrides: DcfOverrides = DcfOverrides()
    price: Annotated[float, PydanticField(gt=0.0)]


class ExportRequest(Contract):
    overrides: DcfOverrides = DcfOverrides()
    #: Issue key to when it was acknowledged, as the tab stamps it.
    acknowledged: dict[str, str] = {}
    format: Literal["xlsx", "csv"] = "xlsx"


# ---------------------------------------------------------------------------
# The response — the company, the panel, the valuation
# ---------------------------------------------------------------------------


class Blocked(Contract):
    """Why there is no valuation, drawn in place of one."""

    title: str
    message: str
    detail: str = ""


class Company(Contract):
    ticker: str
    name: str
    sector: str | None
    currency: str
    valuation_currency: str
    converted: bool
    #: Units of the valuation currency per unit of the reporting currency.
    fx_rate: float | None
    fx_provenance: Provenance | None
    fiscal_year_end_month: int | None
    last_price: float | None
    price_date: date | None
    shares_date: date | None
    asreported: bool
    template: str
    years_retrieved: int
    share_factor: float


class Note(Contract):
    level: Literal["caption", "info", "warning"]
    text: str


class Input(Contract):
    """One numeric input: the value in use, the untouched value, and their provenance."""

    value: float | None
    #: ``None`` where nothing could pre-populate it — the reader must supply it.
    default: float | None
    provenance: Provenance
    overridden: bool
    notes: list[Note]


class Driver(Contract):
    key: DriverKey
    label: str
    bases: list[str]
    default_basis: str
    basis: str
    input: Input
    periods: list[str]
    manual: bool


class Target(Contract):
    key: TargetKey
    label: str
    default_on: bool
    on: bool
    seed: float
    input: Input | None
    notes: list[Note]


class AcquisitionSpend(Contract):
    on: bool
    share: float
    years_used: int


class Regression(Contract):
    beta: float
    raw_beta: float
    r_squared: float
    observations: int
    index_ticker: str
    standard_error: float | None
    interval_low: float | None
    interval_high: float | None


class Wacc(Contract):
    wacc: float
    cost_of_equity: float
    cost_of_debt: float
    after_tax_cost_of_debt: float
    equity_weight: float
    debt_weight: float
    formula: str


class NamedValue(Contract):
    name: str
    value: float | None


class BaseFcff(Contract):
    method_a: float | None
    method_b: float | None
    divergence: float | None
    inputs: list[NamedValue]
    notes: list[str]


class Panel(Contract):
    """Every input, top to bottom, as the Streamlit panel draws them."""

    base_label: str
    base_detail: str
    base_is_ttm: bool
    stocks_as_of: date | None
    flows_as_of: date | None

    years: int
    years_range: list[int]
    drivers: list[Driver]
    fade: bool
    hold: int
    targets: list[Target]
    acquisitions: AcquisitionSpend | None
    capex_estimate: str | None

    risk_free: Input
    erp: Input
    beta_frequency: Frequency
    beta_frequencies: list[str]
    blume: bool
    beta: Input
    regression: Regression | None
    cost_of_debt: Input
    implied_cost_of_debt: float | None
    marginal_tax: Input

    share_bases: list[str]
    share_basis: str
    shares: float | None
    share_label: str
    share_notes: list[Note]

    wacc: Wacc | None
    equity_value: float
    debt_value: float

    terminal_growth: Input
    ronic_bases: list[str]
    ronic_basis: str
    #: ``value`` ``None`` means the terminal year earns the WACC.
    ronic: Input
    incremental_return: float | None
    incremental_return_note: str
    terminal_at_marginal: bool
    ebita_proxy: bool
    exit_multiple: Input
    headline_method: Method
    mid_year: bool
    elapsed: float
    elapsed_note: Note

    include_sti: bool
    cash_floor: float
    cash: float | None
    short_term_investments: float | None
    sbc: Input
    non_operating: float
    long_term_investments: float | None

    base_fcff: BaseFcff
    etr: float | None
    etr_years: int
    delta_nwc_span: str


class ProjectionRow(Contract):
    label: str
    years_after: float
    discount_exponent: float
    revenue: float
    revenue_growth: float
    operating_margin: float
    ebit: float
    tax_rate: float
    nopat: float
    dna: float
    capex: float
    delta_nwc: float
    fcff: float
    discount_factor: float
    present_value: float


class TerminalMethod(Contract):
    method: Method
    label: str
    headline: bool
    terminal_value: float
    present_value: float
    enterprise_value: float
    equity_value: float
    value_per_share: float
    share_of_ev: float | None
    discount_exponent: float
    #: This method restated in the other method's units.
    implies: str


class TerminalYear(Contract):
    revenue: float
    growth: float
    operating_margin: float
    ebit: float
    amortisation: float
    nopat: float
    tax_rate: float
    ronic: float
    reinvestment_rate: float
    reinvestment: float
    fcff: float


class BridgeLine(Contract):
    label: str
    amount: float
    reported: bool
    note: str
    kind: str


class Bridge(Contract):
    lines: list[BridgeLine]
    shares: float
    share_basis: str
    value_per_share: float


class Sensitivity(Contract):
    method: Method
    column_label: str
    waccs: list[float]
    columns: list[float]
    headings: list[str]
    values: list[list[float | None]]


class IssueValue(Contract):
    label: str
    value: str


class Issue(Contract):
    key: str
    severity: Literal["blocking", "critical", "warning"]
    check: str
    headline: str
    concept: str
    message: str
    period: str | None
    values: list[IssueValue]
    tolerance: str | None
    issue_type: str


class Check(Contract):
    name: str
    status: Literal["pass", "fail", "not evaluated"]
    detail: str
    severity: str | None


class Solve(Contract):
    method: Method
    label: str
    status: str
    growth: float | None
    roots: list[float]
    reason: str
    #: The banner the solve needs, if any — ``""`` for an ordinary answer.
    level: Literal["", "info", "warning", "error"]
    message: str
    formula: str


class RevenuePath(Contract):
    name: str
    kind: Literal["price", "forecast", "history"]
    method: Method | None
    values: list[float]


class RevenueChart(Contract):
    history_labels: list[str]
    reported: list[float | None]
    anchor: str
    base_revenue: float
    forward_labels: list[str]
    paths: list[RevenuePath]


class GrowthHistory(Contract):
    ok: bool
    cagr: float | None
    arithmetic_mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    first_period: str
    last_period: str
    intervals: int
    observations: int
    detail: str


class ScanRow(Contract):
    growth: float
    gordon: float | None
    exit_multiple: float | None


class Implied(Contract):
    forecast_growth: float
    faded: bool
    hold: int
    years: int
    history: GrowthHistory
    solves: list[Solve]
    chart: RevenueChart | None
    scan: list[ScanRow]


class Valuation(Contract):
    headline_method: Method
    value_per_share: float
    upside: float | None
    wacc: float
    mid_year: bool
    pv_explicit: float
    projection: list[ProjectionRow]
    methods: list[TerminalMethod]
    terminal_year: TerminalYear | None
    bridge: Bridge
    sensitivity: Sensitivity
    issues: list[Issue]
    checks: list[Check]
    implied: Implied


class RatioYear(Contract):
    period: str
    values: list[float | None]


class Filing(Contract):
    fiscal_year: str
    period: str
    report_date: date | None
    publish_date: date | None
    restated: bool
    link: str | None


class History(Contract):
    """The ratios the drivers come from, and the filing behind each year."""

    ratio_keys: list[str]
    ratio_labels: list[str]
    ratios: list[RatioYear]
    #: Years each driver could use — a missing component excludes a year.
    usable: list[NamedValue]
    revenue: list[float | None]
    ebit: list[float | None]
    filings: list[Filing]


class Reference(Contract):
    schema_map: Frame
    ambiguous: bool
    tolerances: Frame


class Dcf(Contract):
    ticker: str
    valuation_date: date
    company: Company | None
    blocked: Blocked | None
    panel: Panel | None
    valuation: Valuation | None
    history: History | None
    reference: Reference | None


class WhatIfSolves(Contract):
    price: float
    solves: list[Solve]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class _Refused(Exception):
    def __init__(self, blocked: Blocked):
        self.blocked = blocked


def _load(ticker: str) -> dcf_model.CompanyDataset:
    """The company, or why the model refuses it — raised as :class:`_Refused`."""
    try:
        dataset = dcf_model.load_company(ticker)
    except val.UnsupportedSectorError as error:
        raise _Refused(Blocked(title="Unsupported sector", message=str(error))) from None
    except val.UnsupportedCurrencyError as error:
        raise _Refused(
            Blocked(title="No exchange rate for the reporting currency", message=str(error))
        ) from None
    except SchemaResolutionError as error:
        raise _Refused(Blocked(
            title="Could not resolve a required financial concept",
            message=str(error),
            detail=f"Columns Sharadar did return on the {error.statement} statement: "
                   f"{', '.join(error.available[:40])}",
        )) from None
    if dataset.annual.empty:
        raise _Refused(Blocked(
            title="No usable statements",
            message=f"Sharadar returned no annual fundamentals for {ticker}, so there is no "
                    "history to build a forecast on.",
        ))
    return dataset


def _resolve(
    dataset: dcf_model.CompanyDataset, overrides: DcfOverrides, today: date
) -> dcf_inputs.Assumptions:
    chosen = overrides.logic()
    try:
        frequency, blume, years = dcf_inputs.beta_settings(chosen)
        beta = dcf_model.regress_company_beta(
            dataset, frequency=frequency, blume=blume, years=years
        )
        return dcf_inputs.resolve(
            dataset,
            valuation_date=today,
            risk_free=market_data.fetch_10y_treasury_rate(),
            erp=market_data.fetch_erp(),
            beta=beta,
            overrides=chosen,
        )
    except dcf_inputs.InvalidOverride as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from None


# ---------------------------------------------------------------------------
# Shaping
# ---------------------------------------------------------------------------


def _number(value) -> float | None:
    if value is None or (not isinstance(value, (int, float)) and pd.isna(value)):
        return None
    return finite_or_none(float(value))


def _notes(notes) -> list[Note]:
    return [Note(level=n.level, text=n.text) for n in notes]


def _input(field: dcf_inputs.Field) -> Input:
    return Input(
        value=_number(field.value),
        default=_number(field.default),
        provenance=as_provenance(field.sourced),
        overridden=field.overridden,
        notes=_notes(field.notes),
    )


def _company(dataset: dcf_model.CompanyDataset) -> Company:
    fx: Sourced | None = dataset.fx_rate if dataset.is_converted else None
    return Company(
        ticker=dataset.ticker,
        name=dataset.name,
        sector=dataset.sector_name,
        currency=dataset.currency,
        valuation_currency=dataset.valuation_currency,
        converted=dataset.is_converted,
        fx_rate=_number(fx.value) if fx is not None else None,
        fx_provenance=as_provenance(fx) if fx is not None else None,
        fiscal_year_end_month=dataset.fiscal_year_end_month,
        last_price=_number(dataset.last_price),
        price_date=dataset.price_date,
        shares_date=dataset.shares_point_in_time_date,
        asreported=dataset.asreported,
        template=dataset.template or "",
        years_retrieved=len(dataset.annual),
        share_factor=float(dataset.share_factor),
    )


def _panel(a: dcf_inputs.Assumptions) -> Panel:
    stocks = a.base.stocks
    regressed = a.beta_outcome.result
    interval = regressed.confidence_interval if regressed is not None else None
    base_fcff = a.base_fcff
    return Panel(
        base_label=a.base.label,
        base_detail=a.base.detail,
        base_is_ttm=a.base.is_ttm,
        stocks_as_of=a.base.stocks_as_of,
        flows_as_of=a.base.flows_as_of,
        years=a.years,
        years_range=list(dcf_inputs.YEARS_RANGE),
        drivers=[
            Driver(
                key=d.key, label=d.label, bases=list(d.bases), default_basis=d.default_basis,
                basis=d.basis, input=_input(d.input), periods=[str(p) for p in d.periods],
                manual=d.manual,
            )
            for d in a.drivers_in
        ],
        fade=a.fade,
        hold=a.hold,
        targets=[
            Target(
                key=t.key, label=t.label, default_on=t.default_on, on=t.on, seed=t.seed,
                input=_input(t.input) if t.input is not None else None, notes=_notes(t.notes),
            )
            for t in a.targets
        ],
        acquisitions=AcquisitionSpend(
            on=a.acquisitions.on, share=float(a.acquisitions.spend.value),
            years_used=a.acquisitions.spend.years_used,
        ) if a.acquisitions else None,
        capex_estimate=a.capex_estimate,
        risk_free=_input(a.risk_free),
        erp=_input(a.erp),
        beta_frequency=a.beta_frequency,
        beta_frequencies=list(dcf_inputs.BETA_FREQUENCIES),
        blume=a.blume,
        beta=_input(a.beta),
        regression=Regression(
            beta=regressed.beta, raw_beta=regressed.raw_beta, r_squared=regressed.r_squared,
            observations=regressed.observations, index_ticker=regressed.index_ticker,
            standard_error=_number(regressed.standard_error),
            interval_low=_number(interval[0]) if interval else None,
            interval_high=_number(interval[1]) if interval else None,
        ) if regressed is not None else None,
        cost_of_debt=_input(a.cost_of_debt),
        implied_cost_of_debt=_number(a.implied_cost_of_debt),
        marginal_tax=_input(a.marginal_tax),
        share_bases=list(dcf_inputs.SHARE_BASES),
        share_basis=a.share_basis_choice,
        shares=_number(a.shares),
        share_label=a.share_basis,
        share_notes=_notes(a.share_notes),
        wacc=Wacc(
            wacc=a.wacc.wacc, cost_of_equity=a.wacc.cost_of_equity,
            cost_of_debt=a.wacc.cost_of_debt, after_tax_cost_of_debt=a.wacc.after_tax_cost_of_debt,
            equity_weight=a.wacc.equity_weight, debt_weight=a.wacc.debt_weight,
            formula=a.wacc.formula,
        ) if a.wacc is not None else None,
        equity_value=float(a.equity_value),
        debt_value=float(a.debt_value),
        terminal_growth=_input(a.terminal_growth),
        ronic_bases=list(a.ronic_bases),
        ronic_basis=a.ronic_basis,
        ronic=_input(a.ronic),
        incremental_return=_number(a.incremental_return),
        incremental_return_note=a.incremental_return_note,
        terminal_at_marginal=a.terminal_at_marginal,
        ebita_proxy=a.ebita_proxy,
        exit_multiple=_input(a.exit_multiple),
        headline_method=a.headline_method,
        mid_year=a.mid_year,
        elapsed=float(a.elapsed),
        elapsed_note=Note(level=a.elapsed_note.level, text=a.elapsed_note.text),
        include_sti=a.include_sti,
        cash_floor=float(a.cash_floor),
        cash=_number(stocks.get("cash")),
        short_term_investments=_number(stocks.get("short_term_investments")),
        sbc=_input(a.sbc),
        non_operating=float(a.non_operating),
        long_term_investments=_number(stocks.get("long_term_investments")),
        base_fcff=BaseFcff(
            method_a=_number(base_fcff.method_a),
            method_b=_number(base_fcff.method_b),
            divergence=_number(base_fcff.divergence),
            inputs=[NamedValue(name=k, value=_number(v)) for k, v in base_fcff.inputs.items()],
            notes=list(base_fcff.notes),
        ),
        etr=_number(a.etr),
        etr_years=a.etr_years,
        delta_nwc_span=a.delta_nwc_span,
    )


def _solve(method: str, outcome: reverse.ReverseDcfResult) -> Solve:
    level, message = dcf_inputs.reverse_verdict(outcome)
    return Solve(
        method=method,
        label=dcf_inputs.REVERSE_LABELS[method],
        status=outcome.status.value,
        growth=_number(outcome.growth) if outcome.status is reverse.Status.SOLVED else None,
        roots=[float(r) for r in outcome.roots if _number(r) is not None],
        reason=outcome.reason,
        level=level,
        message=message,
        formula=outcome.formula,
    )


def _issue(issue: val.Issue) -> Issue:
    return Issue(
        key=issue.key,
        severity=issue.severity.value,
        check=issue.check,
        headline=issue.headline(),
        concept=issue.concept,
        message=issue.message,
        period=issue.period,
        values=[IssueValue(label=k, value=str(v)) for k, v in issue.values.items()],
        tolerance=issue.tolerance,
        issue_type=issue.issue_type,
    )


def _growth_history(history: fin.GrowthHistory) -> GrowthHistory:
    return GrowthHistory(
        ok=history.ok,
        cagr=_number(history.cagr),
        arithmetic_mean=_number(history.arithmetic_mean),
        median=_number(history.median),
        minimum=_number(history.minimum),
        maximum=_number(history.maximum),
        first_period=str(history.first_period),
        last_period=str(history.last_period),
        intervals=history.intervals,
        observations=history.observations,
        detail=history.detail,
    )


def _valuation(
    dataset: dcf_model.CompanyDataset,
    inputs: dict,
    evaluation: dcf_inputs.Evaluation,
) -> Valuation:
    result = evaluation.result
    bridge = result.headline_bridge
    steady = result.terminal_values["gordon"].terminal_year
    grid = evaluation.sensitivity
    history = fin.revenue_growth_history(dataset.annual)
    chart = dcf_inputs.revenue_chart(dataset, result, evaluation.reverse, history, inputs)
    drivers: dcf.DriverPath = inputs["drivers"]
    return Valuation(
        headline_method=result.headline_method,
        value_per_share=bridge.value_per_share,
        upside=_number(bridge.upside_vs(dataset.last_price)) if dataset.last_price else None,
        wacc=result.wacc.wacc,
        mid_year=result.projection.mid_year,
        pv_explicit=result.projection.pv_explicit,
        projection=[
            ProjectionRow(
                label=row.label, years_after=row.cumulative_time,
                discount_exponent=row.discount_exponent, revenue=row.revenue,
                revenue_growth=row.revenue_growth, operating_margin=row.operating_margin,
                ebit=row.ebit, tax_rate=row.tax_rate, nopat=row.nopat, dna=row.dna,
                capex=row.capex, delta_nwc=row.delta_nwc, fcff=row.fcff,
                discount_factor=row.discount_factor, present_value=row.present_value,
            )
            for row in result.projection.rows
        ],
        methods=[
            TerminalMethod(
                method=method,
                label=terminal.method,
                headline=method == result.headline_method,
                terminal_value=terminal.value,
                present_value=terminal.present_value,
                enterprise_value=result.enterprise_values[method],
                equity_value=result.bridges[method].equity_value,
                value_per_share=result.bridges[method].value_per_share,
                share_of_ev=_number(terminal.share_of(result.enterprise_values[method])),
                discount_exponent=terminal.discount_exponent,
                implies=dcf_inputs.cross_implied(terminal),
            )
            for method, terminal in result.terminal_values.items()
        ],
        terminal_year=TerminalYear(
            revenue=steady.revenue, growth=steady.growth,
            operating_margin=steady.operating_margin, ebit=steady.ebit,
            amortisation=steady.amortisation, nopat=steady.nopat, tax_rate=steady.tax_rate,
            ronic=steady.ronic, reinvestment_rate=steady.reinvestment_rate,
            reinvestment=steady.reinvestment, fcff=steady.fcff,
        ) if steady is not None else None,
        bridge=Bridge(
            lines=[
                BridgeLine(
                    label=line.label, amount=line.amount, reported=line.reported,
                    note=line.note, kind=line.kind,
                )
                for line in bridge.lines
            ],
            shares=bridge.shares,
            share_basis=bridge.share_basis,
            value_per_share=bridge.value_per_share,
        ),
        sensitivity=Sensitivity(
            method=grid.method,
            column_label=grid.column_label,
            waccs=list(grid.waccs),
            columns=list(grid.columns),
            headings=[grid.column_heading(c) for c in grid.columns],
            values=[[_number(v) for v in row] for row in grid.values],
        ),
        issues=[_issue(i) for i in evaluation.issues],
        checks=[
            Check(
                name=c.name, status=c.status, detail=c.detail,
                severity=c.severity.value if c.severity else None,
            )
            for c in evaluation.check_results
        ],
        implied=Implied(
            forecast_growth=drivers.revenue_growth[0],
            faded=inputs["growth_fade_to"] is not None,
            hold=inputs["growth_hold"],
            years=drivers.years,
            history=_growth_history(history),
            solves=[_solve(m, evaluation.reverse[m]) for m in dcf_inputs.REVERSE_METHODS],
            chart=RevenueChart(
                history_labels=list(chart.history_labels),
                reported=[_number(v) for v in chart.reported],
                anchor=chart.anchor,
                base_revenue=chart.base_revenue,
                forward_labels=list(chart.forward_labels),
                paths=[
                    RevenuePath(name=p.name, kind=p.kind, method=p.method, values=list(p.values))
                    for p in chart.paths
                ],
            ) if chart is not None else None,
            scan=[
                ScanRow(
                    growth=g, gordon=_number(values["gordon"]),
                    exit_multiple=_number(values["exit_multiple"]),
                )
                for g, values in dcf_inputs.scan_rows(evaluation.reverse)
            ],
        ),
    )


def _history(dataset: dcf_model.CompanyDataset) -> History:
    ratios = dataset.ratios
    keys = [str(c) for c in ratios.columns]
    annual = dataset.annual
    return History(
        ratio_keys=keys,
        ratio_labels=[fin.RATIO_LABELS.get(k, k) for k in keys],
        ratios=[
            RatioYear(period=str(period), values=[_number(v) for v in row])
            for period, row in zip(ratios.index, ratios.itertuples(index=False))
        ],
        usable=[
            NamedValue(name=fin.DRIVER_LABELS[k], value=float(ratios[k].notna().sum()))
            for k in fin.DRIVER_KEYS if k in ratios.columns
        ],
        revenue=[_number(v) for v in annual["revenue"]] if "revenue" in annual else [],
        ebit=[_number(v) for v in annual["ebit"]] if "ebit" in annual else [],
        filings=[
            Filing(
                fiscal_year=f.fiscal_year, period=f.period, report_date=f.report_date,
                publish_date=f.publish_date, restated=f.restated, link=f.link,
            )
            for f in dcf_inputs.filings(dataset)
        ],
    )


def _reference(dataset: dcf_model.CompanyDataset, tolerances: val.Tolerances) -> Reference:
    return Reference(
        schema_map=as_frame(dataset.schema.as_frame()),
        ambiguous=bool(dataset.schema.ambiguous),
        tolerances=as_frame(tolerances.describe()),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/{ticker}/dcf")
def dcf_tab(ticker: Ticker, overrides: DcfOverrides | None = None) -> Dcf:
    """The panel, defaulted wherever the reader left it, and the valuation it makes.

    Every check runs on every request; a BLOCKING one, or a model that cannot
    be built, returns the panel with ``blocked`` and no valuation, so the reader
    can undo whatever caused it.
    """
    ticker = ticker.strip().upper()
    today = _today()
    shell = dict(ticker=ticker, valuation_date=today)
    try:
        dataset = _load(ticker)
    except _Refused as refused:
        return Dcf(**shell, company=None, blocked=refused.blocked, panel=None, valuation=None,
                   history=None, reference=None)

    resolved = _resolve(dataset, overrides or DcfOverrides(), today)
    common = dict(
        **shell,
        company=_company(dataset),
        panel=_panel(resolved),
        history=_history(dataset),
        reference=_reference(dataset, val.DEFAULT_TOLERANCES),
    )
    if resolved.blocked is not None:
        blocked = resolved.blocked
        return Dcf(**common, valuation=None, blocked=Blocked(
            title=blocked.title, message=blocked.message, detail=blocked.detail,
        ))

    try:
        evaluation = dcf_inputs.evaluate(dataset, resolved.inputs)
    except dcf.TerminalValueError as error:
        return Dcf(**common, valuation=None, blocked=Blocked(
            title="Terminal value cannot be computed",
            message=str(error),
            detail="Lower the terminal growth rate below the WACC, or raise the WACC.",
        ))
    except dcf.ProjectionError as error:
        return Dcf(**common, valuation=None, blocked=Blocked(
            title="The projection cannot be built", message=str(error),
        ))
    if evaluation.blocking:
        issue = evaluation.blocking[0]
        return Dcf(**common, valuation=None, blocked=Blocked(
            title=issue.check,
            message=issue.message,
            detail="; ".join(f"{k}: {v}" for k, v in issue.values.items()),
        ))
    return Dcf(**common, blocked=None, valuation=_valuation(dataset, resolved.inputs, evaluation))


@router.post("/{ticker}/dcf/what-if")
def dcf_what_if(ticker: Ticker, request: WhatIf) -> WhatIfSolves:
    """The year-1 growth another price implies, against the run the panel describes."""
    ticker = ticker.strip().upper()
    try:
        dataset = _load(ticker)
    except _Refused as refused:
        raise HTTPException(status.HTTP_409_CONFLICT, refused.blocked.message) from None
    resolved = _resolve(dataset, request.overrides, _today())
    if resolved.inputs is None:
        raise HTTPException(status.HTTP_409_CONFLICT, resolved.blocked.message)
    return WhatIfSolves(
        price=request.price,
        solves=[
            _solve(method, dcf_inputs.solve_price(dataset, resolved.inputs, request.price, method))
            for method in dcf_inputs.REVERSE_METHODS
        ],
    )


_MEDIA = {
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "csv": "text/csv",
}


@router.post(
    "/{ticker}/dcf/export",
    response_class=Response,
    responses={200: {"content": {media: {} for media in _MEDIA.values()}}},
)
def dcf_export_file(ticker: Ticker, request: ExportRequest) -> Response:
    """The workbook or CSV of the run the panel describes, stamped with the acknowledgements."""
    ticker = ticker.strip().upper()
    today = _today()
    try:
        dataset = _load(ticker)
    except _Refused as refused:
        raise HTTPException(status.HTTP_409_CONFLICT, refused.blocked.message) from None
    resolved = _resolve(dataset, request.overrides, today)
    if resolved.inputs is None:
        raise HTTPException(status.HTTP_409_CONFLICT, resolved.blocked.message)
    try:
        evaluation = dcf_inputs.evaluate(dataset, resolved.inputs)
    except (dcf.TerminalValueError, dcf.ProjectionError) as error:
        raise HTTPException(status.HTTP_409_CONFLICT, str(error)) from None

    # Only this run's issues: an acknowledgement of one an earlier input raised
    # would otherwise be counted in the export as if it were on screen.
    keys = {issue.key for issue in evaluation.issues}
    shared = dict(
        dataset=dataset,
        result=evaluation.result,
        assumptions=list(resolved.records),
        issues=evaluation.issues,
        check_results=evaluation.check_results,
        acknowledged={k: v for k, v in request.acknowledged.items() if k in keys},
        valuation_date=today,
    )
    if request.format == "xlsx":
        content: bytes | str = dcf_export.build_workbook(**shared, tolerances=val.DEFAULT_TOLERANCES)
    else:
        content = dcf_export.build_csv(**shared)
    name = f"{dataset.ticker}_dcf_{today.isoformat()}.{request.format}"
    return Response(
        content=content,
        media_type=_MEDIA[request.format],
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
