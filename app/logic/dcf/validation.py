"""Validation checks, severity classification, and issue types.

The primary safety mechanism. Three tiers:

``BLOCKING``
    The computation is genuinely impossible, so nothing is rendered. Reserved
    for absent data — never for data that merely looks wrong.
``CRITICAL``
    A number was computed but a core input is suspect. Renders a large,
    persistent red banner and marks the headline figure.
``WARNING``
    Worth knowing, not necessarily wrong. Amber banner.

Every check names itself, both disagreeing figures, the tolerance it breached,
and the period it applies to. Checks aggregate: six failures produce six
issues, not the first one.

A check that cannot run because its inputs are absent reports ``not evaluated``
rather than quietly passing — a green tick that only means "we didn't look" is
exactly the kind of silent absorption this module exists to prevent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import numpy as np
import pandas as pd


class Severity(str, Enum):
    BLOCKING = "blocking"
    CRITICAL = "critical"
    WARNING = "warning"


# ---------------------------------------------------------------------------
# Tolerances — every threshold in the model lives here
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerances:
    """Every tolerance in one place, documented, defaulting tight.

    Relative tolerances are fractions of the larger figure being compared.
    """

    #: Total assets vs total liabilities + equity.
    balance_sheet: float = 0.001
    #: CFO + CFI + CFF + FX vs net change in cash.
    cash_flow_tie: float = 0.001
    #: Four quarters summed vs the reported fiscal year.
    quarters_vs_fy: float = 0.005
    #: The API's TTM vs a manual four-quarter sum.
    ttm_vs_manual: float = 0.005
    #: Revenue + cost of revenue + operating expenses vs reported EBIT.
    ebit_reconciliation: float = 0.001
    #: Reported total debt vs the sum of its current and non-current parts.
    total_debt: float = 0.001
    #: Agreement between independent reads on the *same* share-count concept.
    share_count: float = 0.01
    #: Gap between the point-in-time and weighted-average bases, which measure
    #: different things and legitimately differ at any company buying back stock.
    share_basis_gap: float = 0.10
    #: Model-implied market cap vs Sharadar's reported market cap.
    market_cap: float = 0.02
    #: FCFF method A vs method B.
    fcff_methods: float = 0.05

    #: Calendar days before a price is treated as stale.
    price_staleness_days: int = 5
    #: Calendar days before a point-in-time share count is treated as stale.
    share_staleness_days: int = 120

    #: Fiscal years of history below which the forecast basis is thin.
    min_history_years: int = 10
    #: Plausible effective tax rate band.
    etr_band: tuple[float, float] = (0.0, 0.50)
    #: Plausible equity risk premium band.
    erp_band: tuple[float, float] = (0.02, 0.10)
    #: R² below which a regressed beta is noise.
    beta_r2_floor: float = 0.10
    #: Terminal value share of enterprise value above which the valuation is
    #: mostly assumption rather than forecast.
    terminal_value_share: float = 0.75
    #: SBC as a share of revenue above which the add-back choice matters a lot.
    sbc_share_of_revenue: float = 0.05
    #: Long-term debt share above which book value is likely to diverge from
    #: market value if rates have moved.
    debt_share_of_capital: float = 0.20

    def describe(self) -> pd.DataFrame:
        """Tabular view for the validation panel and the export."""
        rows = [
            ("Balance sheet balances", f"{self.balance_sheet:.3%} of total assets"),
            ("Cash flow ties", f"{self.cash_flow_tie:.3%} of net change in cash"),
            ("Quarters sum to fiscal year", f"{self.quarters_vs_fy:.3%}"),
            ("API TTM vs manual TTM", f"{self.ttm_vs_manual:.3%}"),
            ("EBIT reconciliation", f"{self.ebit_reconciliation:.3%}"),
            ("Total debt vs components", f"{self.total_debt:.3%}"),
            ("Share count agreement", f"{self.share_count:.2%}"),
            ("Market cap agreement", f"{self.market_cap:.2%}"),
            ("FCFF method A vs B", f"{self.fcff_methods:.2%}"),
            ("Price staleness", f"{self.price_staleness_days} days"),
            ("Share count staleness", f"{self.share_staleness_days} days"),
            ("Minimum history", f"{self.min_history_years} fiscal years"),
            ("Effective tax rate band", f"{self.etr_band[0]:.0%}–{self.etr_band[1]:.0%}"),
            ("Equity risk premium band", f"{self.erp_band[0]:.0%}–{self.erp_band[1]:.0%}"),
            ("Beta R² floor", f"{self.beta_r2_floor:.2f}"),
            ("Terminal value share of EV", f"{self.terminal_value_share:.0%}"),
            ("SBC share of revenue", f"{self.sbc_share_of_revenue:.0%}"),
        ]
        return pd.DataFrame(rows, columns=["Tolerance", "Default"])


DEFAULT_TOLERANCES = Tolerances()


# ---------------------------------------------------------------------------
# Issue types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Issue:
    """One validation failure, with everything needed to diagnose it.

    ``key`` is stable across reruns for the same ticker and period — it is the
    handle the per-check acknowledgement is stored against.
    """

    key: str
    severity: Severity
    check: str
    concept: str
    company: str
    message: str
    period: str | None = None
    values: dict[str, str] = field(default_factory=dict)
    tolerance: str | None = None
    issue_type: str = "Issue"

    def headline(self) -> str:
        where = f" · {self.period}" if self.period else ""
        return f"{self.check}{where}"

    def detail_lines(self) -> list[str]:
        lines = [self.message]
        for label, value in self.values.items():
            lines.append(f"{label}: {value}")
        if self.tolerance:
            lines.append(f"Tolerance: {self.tolerance}")
        return lines


@dataclass(frozen=True)
class ReconciliationIssue(Issue):
    """Two figures that should agree, do not."""

    issue_type: str = "ReconciliationIssue"


@dataclass(frozen=True)
class ImplausibleValueIssue(Issue):
    """A figure computed cleanly but landing outside a credible range."""

    issue_type: str = "ImplausibleValueIssue"


@dataclass(frozen=True)
class InsufficientHistoryIssue(Issue):
    """Fewer usable historical periods than the forecast basis wants."""

    issue_type: str = "InsufficientHistoryIssue"


@dataclass(frozen=True)
class StaleDataIssue(Issue):
    """Data that is real but too old to value against."""

    issue_type: str = "StaleDataIssue"


class UnsupportedSectorError(Exception):
    """Banks and insurers need a different model entirely. BLOCKING."""

    def __init__(self, ticker: str, sector: str, template: str) -> None:
        self.ticker = ticker
        self.sector = sector
        self.template = template
        super().__init__(
            f"{ticker} is a financial-sector company (sector: {sector or 'unknown'}, "
            f"classified as: {template or 'unknown'}). Banks and insurers do not have "
            "the operating/financing split an unlevered FCFF model depends on — "
            "interest is revenue, not a financing cost, and capital structure is "
            "regulated rather than chosen. A DCF built this way would produce a "
            "meaningless number, so none is shown."
        )


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check, for the pass/fail list in the validation panel."""

    name: str
    status: str  # "pass" | "fail" | "not evaluated"
    detail: str = ""
    severity: Severity | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finite(value) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def relative_gap(a: float, b: float) -> float:
    """Absolute difference scaled by the larger magnitude."""
    scale = max(abs(a), abs(b))
    if scale == 0:
        return 0.0
    return abs(a - b) / scale


def _money(value: float) -> str:
    if not _finite(value):
        return "not reported"
    return f"{value:,.0f}"


def _fy(label) -> str:
    return f"FY{label}" if isinstance(label, (int, np.integer)) else str(label)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class ValidationContext:
    """Everything the checks need. Fields left ``None`` mark a check unevaluable."""

    ticker: str
    company: str

    annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    annual_meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    quarterly: pd.DataFrame = field(default_factory=pd.DataFrame)
    api_ttm: pd.Series | None = None
    manual_ttm: pd.Series | None = None

    currencies_seen: tuple[str, ...] = ()
    asreported_flags: tuple[bool, ...] = ()

    price_date: date | None = None
    last_price: float | None = None
    reported_market_cap: float | None = None
    valuation_date: date | None = None

    shares_point_in_time: float | None = None
    shares_point_in_time_date: date | None = None
    shares_from_prices: float | None = None
    shares_weighted_diluted: float | None = None
    shares_eps_implied: float | None = None

    fcff_method_a: float | None = None
    fcff_method_b: float | None = None

    effective_tax_rate: float | None = None
    beta_r_squared: float | None = None
    beta_observations: int | None = None
    equity_risk_premium: float | None = None

    terminal_value_share: float | None = None
    terminal_growth: float | None = None
    risk_free_rate: float | None = None

    sbc: float | None = None
    latest_revenue: float | None = None
    debt_weight: float | None = None
    long_term_debt: float | None = None

    balance_sheet_delta_nwc: float | None = None

    #: ``(concept label, resolved column)`` for base-period items reported as
    #: exactly zero, which Sharadar also uses to mean "absent from the filing".
    zero_filled: tuple[tuple[str, str], ...] = ()


# ---------------------------------------------------------------------------
# CRITICAL checks
# ---------------------------------------------------------------------------


def check_balance_sheet_balances(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Total Assets = Total Liabilities + Total Equity + Minority Interest.

    Sharadar's equity figure is attributable to the parent, so non-controlling
    interests sit outside it and have to be added back for the identity to
    hold. Because the client derives minority interest as exactly this residual,
    the check can only fail when one of the three totals is itself wrong — which
    is what it is for.
    """
    name = "Balance sheet balances"
    frame = ctx.annual
    required = {"total_assets", "total_liabilities", "total_equity"}
    if frame.empty or not required.issubset(frame.columns):
        return [], CheckResult(name, "not evaluated", "Balance sheet totals not resolved.")

    issues: list[Issue] = []
    checked = 0
    for label, row in frame.iterrows():
        assets, liabilities, equity = (
            row["total_assets"],
            row["total_liabilities"],
            row["total_equity"],
        )
        if not all(_finite(v) for v in (assets, liabilities, equity)):
            continue
        nci = row.get("minority_interest")
        claims = float(liabilities) + float(equity) + (float(nci) if _finite(nci) else 0.0)
        checked += 1
        if relative_gap(assets, claims) > tol.balance_sheet:
            issues.append(
                ReconciliationIssue(
                    key=f"balance_sheet:{label}",
                    severity=Severity.CRITICAL,
                    check=name,
                    concept="Total Assets = Total Liabilities + Total Equity + Minority Interest",
                    company=ctx.ticker,
                    period=_fy(label),
                    message="The balance sheet does not balance for this period.",
                    values={
                        "Total assets": _money(assets),
                        "Liabilities + equity + NCI": _money(claims),
                        "Minority interest": _money(nci),
                        "Difference": _money(float(assets) - claims),
                    },
                    tolerance=f"{tol.balance_sheet:.3%} of total assets",
                )
            )
    if not checked:
        return [], CheckResult(name, "not evaluated", "No period had all three totals.")
    return issues, _result(name, issues, f"{checked} periods checked", Severity.CRITICAL)


def check_cash_flow_ties(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """CFO + CFI + CFF + FX = Net Change in Cash.

    The FX line is legitimately absent for companies with no translation
    exposure. When it is missing and the identity ties without it, that is
    recorded as a pass with a note; when it is missing and the identity does
    *not* tie, the gap is reported as exactly that rather than assumed to be FX.
    """
    name = "Cash flow statement ties"
    frame = ctx.annual
    required = {"cfo", "cfi", "cff", "net_change_in_cash"}
    if frame.empty or not required.issubset(frame.columns):
        return [], CheckResult(name, "not evaluated", "Cash flow subtotals not resolved.")

    issues: list[Issue] = []
    checked = 0
    fx_missing = 0
    for label, row in frame.iterrows():
        cfo, cfi, cff, net = (
            row["cfo"], row["cfi"], row["cff"], row["net_change_in_cash"],
        )
        if not all(_finite(v) for v in (cfo, cfi, cff, net)):
            continue
        fx = row.get("fx_effect")
        has_fx = _finite(fx)
        if not has_fx:
            fx_missing += 1
        total = cfo + cfi + cff + (float(fx) if has_fx else 0.0)
        checked += 1
        if relative_gap(total, net) > tol.cash_flow_tie:
            issues.append(
                ReconciliationIssue(
                    key=f"cash_flow_tie:{label}",
                    severity=Severity.CRITICAL,
                    check=name,
                    concept="CFO + CFI + CFF + FX = Net Change in Cash",
                    company=ctx.ticker,
                    period=_fy(label),
                    message=(
                        "The cash flow statement does not tie."
                        + (
                            " No FX translation line was reported, so the gap is "
                            "unexplained rather than attributable to currency effects."
                            if not has_fx
                            else ""
                        )
                    ),
                    values={
                        "CFO + CFI + CFF" + (" + FX" if has_fx else ""): _money(total),
                        "Net change in cash": _money(net),
                        "Difference": _money(total - net),
                        "FX translation": _money(fx) if has_fx else "not reported",
                    },
                    tolerance=f"{tol.cash_flow_tie:.3%}",
                )
            )
    if not checked:
        return [], CheckResult(name, "not evaluated", "No period had all four subtotals.")
    detail = f"{checked} periods checked"
    if fx_missing:
        detail += f"; {fx_missing} without an FX line, which tie without one"
    return issues, _result(name, issues, detail, Severity.CRITICAL)


def check_quarters_sum_to_year(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The four quarters of a fiscal year sum to the reported fiscal year."""
    name = "Quarters sum to fiscal year"
    if ctx.quarterly.empty or ctx.annual.empty or "revenue" not in ctx.quarterly.columns:
        return [], CheckResult(name, "not evaluated", "Quarterly data unavailable.")

    quarterly = ctx.quarterly.copy()
    fiscal_years = [
        int(str(label).split()[0].replace("FY", ""))
        for label in quarterly.index
        if str(label).startswith("FY")
    ]
    if not fiscal_years:
        return [], CheckResult(name, "not evaluated", "Quarterly labels unrecognised.")
    quarterly = quarterly.assign(_fy=fiscal_years)

    issues: list[Issue] = []
    checked = 0
    for fiscal_year, group in quarterly.groupby("_fy"):
        if len(group) != 4 or fiscal_year not in ctx.annual.index:
            continue
        annual_revenue = ctx.annual.loc[fiscal_year, "revenue"]
        quarter_sum = group["revenue"].sum(skipna=False)
        if not (_finite(annual_revenue) and _finite(quarter_sum)):
            continue
        checked += 1
        if relative_gap(quarter_sum, annual_revenue) > tol.quarters_vs_fy:
            issues.append(
                ReconciliationIssue(
                    key=f"quarters_sum:{fiscal_year}",
                    severity=Severity.CRITICAL,
                    check=name,
                    concept="Σ quarterly revenue = fiscal year revenue",
                    company=ctx.ticker,
                    period=f"FY{fiscal_year}",
                    message="The four reported quarters do not sum to the reported year.",
                    values={
                        "Sum of quarters": _money(quarter_sum),
                        "Reported fiscal year": _money(annual_revenue),
                        "Difference": _money(quarter_sum - annual_revenue),
                    },
                    tolerance=f"{tol.quarters_vs_fy:.3%}",
                )
            )
    if not checked:
        return [], CheckResult(name, "not evaluated", "No fiscal year had all four quarters.")
    return issues, _result(name, issues, f"{checked} fiscal years checked", Severity.CRITICAL)


def check_api_ttm_matches_manual(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The API's native TTM against a hand-summed four quarters."""
    name = "API TTM matches manual four-quarter sum"
    if ctx.api_ttm is None or ctx.manual_ttm is None:
        return [], CheckResult(
            name, "not evaluated", "Fewer than four quarters available to sum."
        )

    issues: list[Issue] = []
    checked = 0
    for key in ("revenue", "ebit", "cfo", "net_income"):
        api_value = ctx.api_ttm.get(key)
        manual_value = ctx.manual_ttm.get(key)
        if not (_finite(api_value) and _finite(manual_value)):
            continue
        checked += 1
        if relative_gap(float(api_value), float(manual_value)) > tol.ttm_vs_manual:
            issues.append(
                ReconciliationIssue(
                    key=f"ttm_manual:{key}",
                    severity=Severity.CRITICAL,
                    check=name,
                    concept=f"TTM {key}",
                    company=ctx.ticker,
                    period="TTM",
                    message=(
                        "The API's trailing-twelve-month figure disagrees with the sum "
                        "of the last four reported quarters."
                    ),
                    values={
                        "API TTM": _money(api_value),
                        "Manual four-quarter sum": _money(manual_value),
                        "Difference": _money(float(api_value) - float(manual_value)),
                    },
                    tolerance=f"{tol.ttm_vs_manual:.3%}",
                )
            )
    if not checked:
        return [], CheckResult(name, "not evaluated", "No comparable TTM line items.")
    return issues, _result(name, issues, f"{checked} line items checked", Severity.CRITICAL)


def check_ebit_reconciles(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Revenue + cost of revenue + operating expenses = reported operating income.

    Costs arrive negative — Sharadar publishes them positive and the client
    negates them — so the identity is additive.
    """
    name = "Computed EBIT reconciles to reported operating income"
    frame = ctx.annual
    required = {"revenue", "cost_of_revenue", "operating_expenses", "ebit"}
    if frame.empty or not required.issubset(frame.columns):
        return [], CheckResult(
            name, "not evaluated", "Revenue, cost of revenue or operating expenses not reported."
        )

    issues: list[Issue] = []
    checked = 0
    for label, row in frame.iterrows():
        parts = (row["revenue"], row["cost_of_revenue"], row["operating_expenses"])
        reported = row["ebit"]
        if not all(_finite(v) for v in (*parts, reported)):
            continue
        checked += 1
        computed = sum(float(p) for p in parts)
        if relative_gap(computed, float(reported)) > tol.ebit_reconciliation:
            issues.append(
                ReconciliationIssue(
                    key=f"ebit_reconciliation:{label}",
                    severity=Severity.CRITICAL,
                    check=name,
                    concept="Revenue + Cost of Revenue + Operating Expenses = EBIT",
                    company=ctx.ticker,
                    period=_fy(label),
                    message="Operating income does not reconcile to its components.",
                    values={
                        "Computed": _money(computed),
                        "Reported operating income": _money(reported),
                        "Difference": _money(computed - float(reported)),
                    },
                    tolerance=f"{tol.ebit_reconciliation:.3%}",
                )
            )
    if not checked:
        return [], CheckResult(name, "not evaluated", "No period had every component.")
    return issues, _result(name, issues, f"{checked} periods checked", Severity.CRITICAL)


def check_total_debt_reconciles(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The reported total debt line equals current plus non-current debt.

    The equity bridge deducts debt built from the two maturity components, so a
    disagreement with the publisher's own aggregate changes the valuation
    directly.
    """
    name = "Total debt equals sum of components"
    frame = ctx.annual
    if frame.empty or "derived_total_debt" not in frame.columns:
        return [], CheckResult(name, "not evaluated", "No aggregate debt line reported.")

    issues: list[Issue] = []
    checked = 0
    for label, row in frame.iterrows():
        aggregate = row.get("derived_total_debt")
        current = row.get("short_term_debt")
        non_current = row.get("long_term_debt")
        if not _finite(aggregate) or not (_finite(current) or _finite(non_current)):
            continue
        components = (float(current) if _finite(current) else 0.0) + (
            float(non_current) if _finite(non_current) else 0.0
        )
        checked += 1
        if relative_gap(float(aggregate), components) > tol.total_debt:
            issues.append(
                ReconciliationIssue(
                    key=f"total_debt:{label}",
                    severity=Severity.CRITICAL,
                    check=name,
                    concept="Total Debt = Debt Current + Debt Non-Current",
                    company=ctx.ticker,
                    period=_fy(label),
                    message="The total debt line does not equal the sum of its parts.",
                    values={
                        "Reported total debt": _money(aggregate),
                        "Current + non-current": _money(components),
                        "Current debt": _money(current),
                        "Non-current debt": _money(non_current),
                        "Difference": _money(float(aggregate) - components),
                    },
                    tolerance=f"{tol.total_debt:.3%}",
                )
            )
    if not checked:
        return [], CheckResult(name, "not evaluated", "No period reported both.")
    return issues, _result(name, issues, f"{checked} periods checked", Severity.CRITICAL)


def _compare_share_sources(
    sources: dict[str, float],
    *,
    name: str,
    concept: str,
    key_prefix: str,
    ticker: str,
    tolerance: float,
    severity: Severity,
    message: str,
) -> tuple[list[Issue], CheckResult]:
    if len(sources) < 2:
        return [], CheckResult(
            name, "not evaluated", "Fewer than two independent sources available."
        )

    labels = list(sources)
    issues: list[Issue] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            if relative_gap(sources[a], sources[b]) > tolerance:
                issues.append(
                    ReconciliationIssue(
                        key=f"{key_prefix}:{i}{j}",
                        severity=severity,
                        check=name,
                        concept=concept,
                        company=ticker,
                        period=None,
                        message=message,
                        values={
                            a: _money(sources[a]),
                            b: _money(sources[b]),
                            "Divergence": f"{relative_gap(sources[a], sources[b]):.2%}",
                        },
                        tolerance=f"{tolerance:.2%}",
                    )
                )
    return issues, _result(name, issues, f"{len(sources)} sources compared", severity)


def check_share_counts_agree(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The three independent reads on *point-in-time* shares outstanding.

    The filed cover-page count, the count backed out of the daily market cap,
    and the count implied by dividing reported market cap by price are all
    compared. They measure the same thing by different routes, so a material
    disagreement means one of the routes is wrong.

    Only like is compared with like. A point-in-time count and a weighted-average
    count *should* differ at any company buying back stock, so they are compared
    separately and much more loosely in
    :func:`check_share_basis_difference`.
    """
    sources: dict[str, float] = {}
    if _finite(ctx.shares_point_in_time):
        sources["Shares outstanding endpoint"] = float(ctx.shares_point_in_time)
    if _finite(ctx.shares_from_prices):
        sources["Prices endpoint"] = float(ctx.shares_from_prices)
    if _finite(ctx.reported_market_cap) and _finite(ctx.last_price) and ctx.last_price:
        sources["Market cap ÷ price"] = float(ctx.reported_market_cap) / float(ctx.last_price)

    return _compare_share_sources(
        sources,
        name="Point-in-time share counts agree",
        concept="Common shares outstanding",
        key_prefix="share_count_pit",
        ticker=ctx.ticker,
        tolerance=tol.share_count,
        severity=Severity.CRITICAL,
        message=(
            "Two independent point-in-time share counts disagree. A scaling error "
            "here moves value per share by the same factor."
        ),
    )


def check_weighted_share_counts_agree(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Reported weighted diluted shares against net income ÷ diluted EPS."""
    sources: dict[str, float] = {}
    if _finite(ctx.shares_weighted_diluted):
        sources["Weighted shares endpoint (diluted)"] = float(ctx.shares_weighted_diluted)
    if _finite(ctx.shares_eps_implied):
        sources["Net income ÷ diluted EPS"] = float(ctx.shares_eps_implied)

    return _compare_share_sources(
        sources,
        name="Weighted-average share counts agree",
        concept="Weighted-average diluted shares",
        key_prefix="share_count_wtd",
        ticker=ctx.ticker,
        tolerance=tol.share_count,
        severity=Severity.CRITICAL,
        message=(
            "The reported weighted diluted share count disagrees with the count "
            "implied by dividing net income by diluted EPS."
        ),
    )


def check_share_basis_difference(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Point-in-time versus weighted-average diluted shares.

    These measure different things and a gap is expected — buybacks shrink the
    point-in-time count below the year's average, options and RSUs push the
    diluted count above it. Only a large gap is worth surfacing, because it
    means the choice of denominator materially moves value per share.
    """
    name = "Point-in-time vs weighted-average share basis"
    if not (_finite(ctx.shares_point_in_time) and _finite(ctx.shares_weighted_diluted)):
        return [], CheckResult(name, "not evaluated", "Both bases are not available.")

    point_in_time = float(ctx.shares_point_in_time)
    weighted = float(ctx.shares_weighted_diluted)
    gap = relative_gap(point_in_time, weighted)

    if gap > tol.share_basis_gap:
        return [
            ImplausibleValueIssue(
                key="share_basis_gap",
                severity=Severity.WARNING,
                check=name,
                concept="Share count basis",
                company=ctx.ticker,
                period=None,
                message=(
                    "The two share bases are far apart, so which one denominates value "
                    "per share moves the answer materially. Buybacks, a large issuance, "
                    "or heavy option dilution will all do this."
                ),
                values={
                    "Point-in-time outstanding": _money(point_in_time),
                    "Weighted-average diluted": _money(weighted),
                    "Difference": f"{gap:.2%}",
                },
                tolerance=f"{tol.share_basis_gap:.0%}",
            )
        ], CheckResult(name, "fail", f"{gap:.2%} apart", Severity.WARNING)
    return [], CheckResult(name, "pass", f"{gap:.2%} apart")


def check_price_is_current(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The price series reaches up to the valuation date.

    A ticker change can terminate a price series while fundamentals keep
    updating. Valuing off a months-old price silently would be a serious
    failure.
    """
    name = "Price series is current"
    if ctx.price_date is None or ctx.valuation_date is None:
        return [], CheckResult(name, "not evaluated", "No price date available.")

    age = (ctx.valuation_date - ctx.price_date).days
    if age > tol.price_staleness_days:
        return [
            StaleDataIssue(
                key="price_stale",
                severity=Severity.CRITICAL,
                check=name,
                concept="Last closing price",
                company=ctx.ticker,
                period=ctx.price_date.isoformat(),
                message=(
                    "The most recent price is older than the staleness threshold. A "
                    "ticker change can terminate a price series while the "
                    "fundamentals keep updating."
                ),
                values={
                    "Last price date": ctx.price_date.isoformat(),
                    "Valuation date": ctx.valuation_date.isoformat(),
                    "Age": f"{age} days",
                    "Last price": f"{ctx.last_price:,.2f}" if _finite(ctx.last_price) else "n/a",
                },
                tolerance=f"{tol.price_staleness_days} days",
            )
        ], CheckResult(name, "fail", f"{age} days old", Severity.CRITICAL)

    return [], CheckResult(name, "pass", f"Last price {ctx.price_date.isoformat()}")


def check_market_cap_agrees(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Price × shares against the market cap Sharadar reports."""
    name = "Model-implied market cap matches reported"
    if not (
        _finite(ctx.last_price)
        and _finite(ctx.shares_point_in_time)
        and _finite(ctx.reported_market_cap)
    ):
        return [], CheckResult(name, "not evaluated", "Price, shares or market cap missing.")

    implied = float(ctx.last_price) * float(ctx.shares_point_in_time)
    reported = float(ctx.reported_market_cap)
    if relative_gap(implied, reported) > tol.market_cap:
        return [
            ReconciliationIssue(
                key="market_cap",
                severity=Severity.CRITICAL,
                check=name,
                concept="Market capitalisation",
                company=ctx.ticker,
                period=ctx.price_date.isoformat() if ctx.price_date else None,
                message=(
                    "Price × shares outstanding does not match the reported market cap, "
                    "which points at a bad price or a bad share count."
                ),
                values={
                    "Price × shares": _money(implied),
                    "Reported market cap": _money(reported),
                    "Difference": _money(implied - reported),
                },
                tolerance=f"{tol.market_cap:.2%}",
            )
        ], CheckResult(name, "fail", "", Severity.CRITICAL)
    return [], CheckResult(name, "pass", f"within {tol.market_cap:.2%}")


def check_fcff_methods_agree(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """FCFF built from the cash flow statement against FCFF built from the P&L."""
    name = "FCFF method A agrees with method B"
    if not (_finite(ctx.fcff_method_a) and _finite(ctx.fcff_method_b)):
        return [], CheckResult(
            name, "not evaluated", "One of the two methods could not be computed."
        )

    gap = relative_gap(float(ctx.fcff_method_a), float(ctx.fcff_method_b))
    if gap > tol.fcff_methods:
        return [
            ReconciliationIssue(
                key="fcff_methods",
                severity=Severity.CRITICAL,
                check=name,
                concept="Free cash flow to the firm",
                company=ctx.ticker,
                period="Base period",
                message=(
                    "The two independent FCFF builds diverge. Check the working capital "
                    "movement, the capex basis, and whether interest sits in operating "
                    "or financing cash flow."
                ),
                values={
                    "Method A (cash flow build)": _money(ctx.fcff_method_a),
                    "Method B (income statement build)": _money(ctx.fcff_method_b),
                    "Divergence": f"{gap:.2%}",
                },
                tolerance=f"{tol.fcff_methods:.2%}",
            )
        ], CheckResult(name, "fail", f"{gap:.2%} apart", Severity.CRITICAL)
    return [], CheckResult(name, "pass", f"{gap:.2%} apart")


def check_currency_consistent(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """One reporting currency across every statement in the run."""
    name = "Reporting currency consistent"
    currencies = tuple(c for c in ctx.currencies_seen if c)
    if not currencies:
        return [], CheckResult(name, "not evaluated", "No currency reported.")
    if len(set(currencies)) > 1:
        return [
            ReconciliationIssue(
                key="currency_mixed",
                severity=Severity.CRITICAL,
                check=name,
                concept="Reporting currency",
                company=ctx.ticker,
                period=None,
                message=(
                    "More than one reporting currency appeared in this model run. "
                    "Mixing currencies makes every ratio and every total meaningless."
                ),
                values={"Currencies seen": ", ".join(sorted(set(currencies)))},
                tolerance="exactly one currency",
            )
        ], CheckResult(name, "fail", "", Severity.CRITICAL)
    return [], CheckResult(name, "pass", currencies[0])


def check_asreported_consistent(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Restated and as-filed figures are never mixed within one run."""
    name = "Restated/as-reported basis consistent"
    flags = tuple(ctx.asreported_flags)
    if not flags:
        return [], CheckResult(name, "not evaluated", "No basis recorded.")
    if len(set(flags)) > 1:
        return [
            ReconciliationIssue(
                key="asreported_mixed",
                severity=Severity.CRITICAL,
                check=name,
                concept="asreported",
                company=ctx.ticker,
                period=None,
                message=(
                    "Restated and as-filed statements were mixed in one model run, "
                    "which puts a silent discontinuity in the middle of the series."
                ),
                values={"Bases seen": ", ".join(str(f) for f in sorted(set(flags)))},
                tolerance="a single basis",
            )
        ], CheckResult(name, "fail", "", Severity.CRITICAL)
    return [], CheckResult(
        name, "pass", "as-reported" if flags[0] else "restated (default)"
    )


# ---------------------------------------------------------------------------
# WARNING checks
# ---------------------------------------------------------------------------


def check_sufficient_history(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Enough usable fiscal years to derive forecast drivers from.

    Zero usable years is BLOCKING — there is nothing to build a forecast on.
    """
    name = "Sufficient usable history"
    usable = 0
    if not ctx.annual.empty and "revenue" in ctx.annual.columns:
        usable = int(ctx.annual["revenue"].notna().sum())

    if usable == 0:
        return [
            InsufficientHistoryIssue(
                key="history_none",
                severity=Severity.BLOCKING,
                check=name,
                concept="Historical revenue",
                company=ctx.ticker,
                period=None,
                message="No fiscal year has a usable revenue figure, so no forecast basis exists.",
                values={"Usable fiscal years": "0"},
                tolerance=f"at least {tol.min_history_years} preferred",
            )
        ], CheckResult(name, "fail", "0 usable years", Severity.BLOCKING)

    if usable < tol.min_history_years:
        return [
            InsufficientHistoryIssue(
                key="history_short",
                severity=Severity.WARNING,
                check=name,
                concept="Historical revenue",
                company=ctx.ticker,
                period=None,
                message=(
                    "Fewer fiscal years than preferred. Averaged drivers rest on a "
                    "short sample and one unusual year moves them a lot."
                ),
                values={"Usable fiscal years": str(usable)},
                tolerance=f"{tol.min_history_years} years",
            )
        ], CheckResult(name, "fail", f"{usable} years", Severity.WARNING)
    return [], CheckResult(name, "pass", f"{usable} fiscal years")


def check_etr_plausible(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Effective tax rate inside a credible band.

    One-off settlements and valuation-allowance releases produce genuine
    outliers, so this flags rather than corrects.
    """
    name = "Effective tax rate plausible"
    if not _finite(ctx.effective_tax_rate):
        return [], CheckResult(
            name, "not evaluated", "No profitable year to compute a rate from."
        )
    low, high = tol.etr_band
    rate = float(ctx.effective_tax_rate)
    if not low <= rate <= high:
        return [
            ImplausibleValueIssue(
                key="etr_band",
                severity=Severity.WARNING,
                check=name,
                concept="Effective tax rate",
                company=ctx.ticker,
                period=None,
                message=(
                    "The effective tax rate is outside the usual band. That can be "
                    "entirely genuine — a settlement or a valuation allowance release "
                    "will do it — but it should be a deliberate choice."
                ),
                values={"Effective tax rate": f"{rate:.2%}"},
                tolerance=f"{low:.0%}–{high:.0%}",
            )
        ], CheckResult(name, "fail", f"{rate:.2%}", Severity.WARNING)
    return [], CheckResult(name, "pass", f"{rate:.2%}")


def check_beta_quality(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """R² above the floor — below it, the beta is noise."""
    name = "Beta regression explanatory power"
    if not _finite(ctx.beta_r_squared):
        return [], CheckResult(name, "not evaluated", "No regression was run.")
    r2 = float(ctx.beta_r_squared)
    if r2 < tol.beta_r2_floor:
        return [
            ImplausibleValueIssue(
                key="beta_r2",
                severity=Severity.WARNING,
                check=name,
                concept="Beta R²",
                company=ctx.ticker,
                period=None,
                message=(
                    "The index explains little of this stock's variation, so the "
                    "regressed beta carries a wide error band. Consider an industry "
                    "beta instead."
                ),
                values={
                    "R²": f"{r2:.3f}",
                    "Observations": str(ctx.beta_observations or "n/a"),
                },
                tolerance=f"R² ≥ {tol.beta_r2_floor:.2f}",
            )
        ], CheckResult(name, "fail", f"R² {r2:.3f}", Severity.WARNING)
    return [], CheckResult(name, "pass", f"R² {r2:.3f}")


def check_erp_plausible(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Equity risk premium inside a credible band."""
    name = "Equity risk premium plausible"
    if not _finite(ctx.equity_risk_premium):
        return [], CheckResult(name, "not evaluated", "No ERP available.")
    low, high = tol.erp_band
    erp = float(ctx.equity_risk_premium)
    if not low <= erp <= high:
        return [
            ImplausibleValueIssue(
                key="erp_band",
                severity=Severity.WARNING,
                check=name,
                concept="Equity risk premium",
                company=ctx.ticker,
                period=None,
                message="The equity risk premium in use is outside the plausible band.",
                values={"Equity risk premium": f"{erp:.2%}"},
                tolerance=f"{low:.0%}–{high:.0%}",
            )
        ], CheckResult(name, "fail", f"{erp:.2%}", Severity.WARNING)
    return [], CheckResult(name, "pass", f"{erp:.2%}")


def check_terminal_value_share(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """How much of enterprise value is terminal value."""
    name = "Terminal value share of enterprise value"
    if not _finite(ctx.terminal_value_share):
        return [], CheckResult(name, "not evaluated", "No valuation computed yet.")
    share = float(ctx.terminal_value_share)
    if share > tol.terminal_value_share:
        return [
            ImplausibleValueIssue(
                key="tv_share",
                severity=Severity.WARNING,
                check=name,
                concept="Terminal value",
                company=ctx.ticker,
                period=None,
                message=(
                    "Most of the value sits beyond the explicit forecast, so the answer "
                    "is driven by the terminal assumptions rather than the projection."
                ),
                values={"Terminal value share": f"{share:.1%}"},
                tolerance=f"≤ {tol.terminal_value_share:.0%}",
            )
        ], CheckResult(name, "fail", f"{share:.1%}", Severity.WARNING)
    return [], CheckResult(name, "pass", f"{share:.1%}")


def check_terminal_growth_reasonable(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Terminal growth against the risk-free rate and long-run GDP growth."""
    from app.logic.dcf.valuation import LONG_RUN_NOMINAL_GDP_GROWTH

    name = "Terminal growth below the economy's"
    if not _finite(ctx.terminal_growth):
        return [], CheckResult(name, "not evaluated", "No terminal growth set.")

    growth = float(ctx.terminal_growth)
    ceiling = LONG_RUN_NOMINAL_GDP_GROWTH
    if _finite(ctx.risk_free_rate):
        ceiling = max(ceiling, float(ctx.risk_free_rate))

    if growth > ceiling:
        return [
            ImplausibleValueIssue(
                key="terminal_growth",
                severity=Severity.WARNING,
                check=name,
                concept="Terminal growth rate",
                company=ctx.ticker,
                period=None,
                message=(
                    "Perpetual growth above long-run nominal GDP growth implies the "
                    "company eventually becomes the whole economy."
                ),
                values={
                    "Terminal growth": f"{growth:.2%}",
                    "Long-run nominal GDP growth": f"{LONG_RUN_NOMINAL_GDP_GROWTH:.2%}",
                    "Risk-free rate": f"{ctx.risk_free_rate:.2%}"
                    if _finite(ctx.risk_free_rate)
                    else "n/a",
                },
                tolerance=f"≤ {ceiling:.2%}",
            )
        ], CheckResult(name, "fail", f"{growth:.2%}", Severity.WARNING)
    return [], CheckResult(name, "pass", f"{growth:.2%}")


def check_sbc_share(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """SBC relative to revenue — where it is large, the treatment matters."""
    name = "Stock-based compensation share of revenue"
    if not (_finite(ctx.sbc) and _finite(ctx.latest_revenue)) or not ctx.latest_revenue:
        return [], CheckResult(
            name, "not evaluated", "Sharadar reports no stock-based compensation figure."
        )
    share = float(ctx.sbc) / float(ctx.latest_revenue)
    if share > tol.sbc_share_of_revenue:
        return [
            ImplausibleValueIssue(
                key="sbc_share",
                severity=Severity.WARNING,
                check=name,
                concept="Stock-based compensation",
                company=ctx.ticker,
                period=None,
                message=(
                    "Stock-based compensation is large relative to revenue, so adding it "
                    "back as non-cash and deducting it as a real cost give materially "
                    "different answers. Make that choice deliberately."
                ),
                values={
                    "SBC": _money(ctx.sbc),
                    "Revenue": _money(ctx.latest_revenue),
                    "SBC ÷ revenue": f"{share:.2%}",
                },
                tolerance=f"≤ {tol.sbc_share_of_revenue:.0%}",
            )
        ], CheckResult(name, "fail", f"{share:.2%}", Severity.WARNING)
    return [], CheckResult(name, "pass", f"{share:.2%}")


def check_zero_filled_indicators(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Base-period line items reported as exactly zero.

    Sharadar writes 0, not null, where a line item is absent from the filing, so
    "the company reported nothing here" and "the company reported nil" are the
    same value in the data. Neither this check nor anything upstream can tell
    them apart, so rather than substitute a guess the affected items are named
    and the user decides.

    This is a warning, not an error: a debt-free company genuinely does report
    zero interest expense, and most of these zeros are real.
    """
    name = "Zero-valued items may be undisclosed rather than nil"
    if not ctx.zero_filled:
        return [], CheckResult(name, "pass", "No base-period item is exactly zero.")

    labels = [label for label, _ in ctx.zero_filled]
    return [
        ReconciliationIssue(
            key="zero_filled",
            severity=Severity.WARNING,
            check=name,
            concept="Zero versus not reported",
            company=ctx.ticker,
            period="Base period",
            message=(
                "Sharadar reports these base-period items as 0. That is also how it "
                "records a line item the filing does not contain, so check the "
                "filing before reading any of them as a genuine nil."
            ),
            values={label: f"0 (column: {column})" for label, column in ctx.zero_filled},
            tolerance="exact zero",
        )
    ], CheckResult(
        name, "fail", f"{len(labels)} item(s): {', '.join(labels)}", Severity.WARNING
    )


def check_share_count_currency_of_date(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The point-in-time share count is recent enough to pair with a live price."""
    name = "Share count is current"
    if ctx.shares_point_in_time_date is None or ctx.valuation_date is None:
        return [], CheckResult(name, "not evaluated", "No share count date available.")
    age = (ctx.valuation_date - ctx.shares_point_in_time_date).days
    if age > tol.share_staleness_days:
        return [
            StaleDataIssue(
                key="shares_stale",
                severity=Severity.WARNING,
                check=name,
                concept="Common shares outstanding",
                company=ctx.ticker,
                period=ctx.shares_point_in_time_date.isoformat(),
                message=(
                    "The share count is dated to the last filing cover page while the "
                    "price is today's, so market capitalisation mixes the two. "
                    "Buybacks since that date are not reflected."
                ),
                values={
                    "Share count as of": ctx.shares_point_in_time_date.isoformat(),
                    "Valuation date": ctx.valuation_date.isoformat(),
                    "Age": f"{age} days",
                },
                tolerance=f"{tol.share_staleness_days} days",
            )
        ], CheckResult(name, "fail", f"{age} days old", Severity.WARNING)
    return [], CheckResult(name, "pass", f"{age} days old")


def check_book_debt_proxy(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Book value of debt as a stand-in for market value.

    Book and market value of debt diverge when rates have moved substantially
    since issuance — material for long-dated fixed-rate debt.
    """
    name = "Book value of debt used as market value"
    if not _finite(ctx.debt_weight):
        return [], CheckResult(name, "not evaluated", "No debt weight computed.")
    weight = float(ctx.debt_weight)
    if weight > tol.debt_share_of_capital:
        return [
            ImplausibleValueIssue(
                key="book_debt",
                severity=Severity.WARNING,
                check=name,
                concept="Market value of debt",
                company=ctx.ticker,
                period=None,
                message=(
                    "Debt is a large share of capital and is carried at book value. "
                    "Where rates have moved since issuance, book diverges from market "
                    "value — most for long-dated fixed-rate debt. Note also that "
                    "post-ASC 842 most lease obligations are already on the balance "
                    "sheet, so adding operating leases on top would double-count."
                ),
                values={
                    "Debt weight in capital": f"{weight:.1%}",
                    "Long-term debt": _money(ctx.long_term_debt),
                },
                tolerance=f"flagged above {tol.debt_share_of_capital:.0%}",
            )
        ], CheckResult(name, "fail", f"{weight:.1%} of capital", Severity.WARNING)
    return [], CheckResult(name, "pass", f"{weight:.1%} of capital")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _result(
    name: str, issues: list[Issue], detail: str, severity: Severity
) -> CheckResult:
    if issues:
        return CheckResult(
            name, "fail", f"{len(issues)} failure(s); {detail}", severity
        )
    return CheckResult(name, "pass", detail)


#: Every check, in the order the validation panel lists them.
ALL_CHECKS = (
    check_balance_sheet_balances,
    check_cash_flow_ties,
    check_quarters_sum_to_year,
    check_api_ttm_matches_manual,
    check_ebit_reconciles,
    check_total_debt_reconciles,
    check_share_counts_agree,
    check_weighted_share_counts_agree,
    check_share_basis_difference,
    check_price_is_current,
    check_market_cap_agrees,
    check_fcff_methods_agree,
    check_currency_consistent,
    check_asreported_consistent,
    check_sufficient_history,
    check_etr_plausible,
    check_beta_quality,
    check_erp_plausible,
    check_terminal_value_share,
    check_terminal_growth_reasonable,
    check_sbc_share,
    check_zero_filled_indicators,
    check_share_count_currency_of_date,
    check_book_debt_proxy,
)


def run_all(
    ctx: ValidationContext, tol: Tolerances | None = None
) -> tuple[list[Issue], list[CheckResult]]:
    """Runs every check and aggregates the results.

    Nothing short-circuits: if six checks fail, six issues come back.
    """
    tolerances = tol or DEFAULT_TOLERANCES
    issues: list[Issue] = []
    results: list[CheckResult] = []
    for check in ALL_CHECKS:
        found, result = check(ctx, tolerances)
        issues.extend(found)
        results.append(result)

    order = {Severity.BLOCKING: 0, Severity.CRITICAL: 1, Severity.WARNING: 2}
    issues.sort(key=lambda i: (order[i.severity], i.check, i.period or ""))
    return issues, results


def issues_frame(issues: list[Issue], acknowledged: dict[str, str] | None = None) -> pd.DataFrame:
    """Flat table of issues for the export, stamped with acknowledgements."""
    acks = acknowledged or {}
    rows = []
    for issue in issues:
        rows.append(
            {
                "Severity": issue.severity.value.upper(),
                "Check": issue.check,
                "Issue type": issue.issue_type,
                "Concept": issue.concept,
                "Company": issue.company,
                "Period": issue.period or "",
                "Message": issue.message,
                "Figures": "; ".join(f"{k} = {v}" for k, v in issue.values.items()),
                "Tolerance": issue.tolerance or "",
                "Acknowledged": "yes" if issue.key in acks else "no",
                "Acknowledged at": acks.get(issue.key, ""),
                "Key": issue.key,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Severity", "Check", "Issue type", "Concept", "Company", "Period",
            "Message", "Figures", "Tolerance", "Acknowledged", "Acknowledged at", "Key",
        ],
    )


def results_frame(results: list[CheckResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Check": r.name,
                "Status": r.status,
                "Detail": r.detail,
                "Severity if failed": r.severity.value.upper() if r.severity else "",
            }
            for r in results
        ]
    )
