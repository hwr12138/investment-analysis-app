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
    #: CFO + CFI + CFF + FX vs net change in cash, as a fraction of the largest
    #: of the two sides and |CFO|. Not of the change in cash alone: that is often
    #: small beside the flows that produce it, and a rounding difference of a few
    #: million failed 0.1% of it at companies whose operating cash flow is in the
    #: billions.
    cash_flow_tie: float = 0.001
    #: When Sharadar's net change in cash is missing (reported as exactly 0) and
    #: the balance-sheet cash change stands in for it, a residual up to this
    #: share of |CFO| is a WARNING; above it a flow line — possibly capex — may
    #: be wrong, and it is CRITICAL.
    cash_flow_residual_warning: float = 0.01
    #: At a foreign filer whose FX line is zero-filled, a gap up to this share of
    #: |CFO| is read as the undisclosed FX effect (WARNING); larger stays CRITICAL.
    implied_fx_max: float = 0.10
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
    #: SBC as a share of revenue above which the add-back choice matters a lot.
    sbc_share_of_revenue: float = 0.05
    #: Long-term debt share above which book value is likely to diverge from
    #: market value if rates have moved.
    debt_share_of_capital: float = 0.20

    #: Long-run nominal GDP growth. A perpetual growth rate above it implies the
    #: company eventually becomes the whole economy. The ceiling actually applied
    #: is the lower of this and the risk-free rate, the stricter common rule.
    long_run_nominal_growth: float = 0.04

    # --- Forecast sanity: is the forecast plausible, not just the data -------
    #: Plausible band for the effective tax rate the *forecast* uses, which is
    #: narrower than the historical band: a one-off year is explainable, a
    #: rate carried for the whole horizon has to be sustainable.
    forecast_etr_band: tuple[float, float] = (0.0, 0.40)
    #: Plausible WACC band for a listed, non-financial company.
    wacc_band: tuple[float, float] = (0.05, 0.15)
    #: Minimum gap between WACC and terminal growth. Below it the perpetuity
    #: denominator is so small that the answer is mostly that gap.
    min_wacc_growth_spread: float = 0.03
    #: Plausible band for the beta in use.
    beta_band: tuple[float, float] = (0.3, 2.5)
    #: Points by which forecast revenue growth may exceed the realised CAGR.
    growth_above_realised: float = 0.05
    #: Year-1 FCFF against the base period's, grown one year.
    jump_off: float = 0.25
    #: Gordon against exit-multiple value per share. Wide on purpose: the exit
    #: multiple sits well above Gordon at most tech companies — even at the
    #: company's own RONIC, a perpetuity cannot rise above NOPAT ÷ (WACC − g) —
    #: and a threshold that fired on all of them would say nothing.
    method_divergence: float = 0.35
    #: How far the final forecast year's capex ÷ D&A may fall below the
    #: company's own five-year ratio before the asset base reads as shrinking.
    #: Compared to the company's history as well as to 1.0: asset-light and
    #: acquisitive companies depreciate acquired intangibles capex never
    #: replaces, and capex at or above D&A shrinks nothing.
    capex_to_dna_shortfall: float = 0.20
    #: Upside or downside beyond which an input error is likelier than an
    #: opportunity (WARNING), and upside beyond which the headline is marked
    #: CRITICAL — red banner until acknowledged. No matching downside tier:
    #: below −100% needs negative equity, which the negative-outcomes check
    #: already raises.
    extreme_upside: float = 0.60
    extreme_upside_critical: float = 2.00
    #: Derived minority interest as a share of parent equity above which its
    #: book value, rather than a market value, materially moves the bridge.
    nci_share_of_equity: float = 0.10
    #: Minority interest below this share of total assets is treated as nil
    #: when cross-checking it against NCI income.
    nci_materiality: float = 0.001
    #: Acquisition spend as a share of capex above which reported growth is
    #: partly bought, and the capex driver leaves the cost of buying it out —
    #: provided it is also material against revenue. A company with almost no
    #: capex (SHOP) clears the capex test on any acquisition at all.
    acquisitions_share_of_capex: float = 0.50
    acquisitions_share_of_revenue: float = 0.02
    #: Above this share of the five-year total, one year's acquisitions are
    #: named as a single deal rather than a pattern.
    acquisitions_dominant_year: float = 0.60
    #: Preferred dividends below this share of net income are immaterial — and
    #: Sharadar's line holds other items for some filers (Alphabet has no
    #: preferred shares, yet reads $86m).
    preferred_dividend_materiality: float = 0.005
    #: Months after the spread table's as-of date before it is stale. Damodaran
    #: updates each January; fifteen months allows for a late refresh.
    spread_table_max_age_months: int = 15
    #: Points an *assumed* margin may sit outside the reported range (annual
    #: history plus the TTM margin) before it is flagged.
    margin_outside_history: float = 0.05

    # --- Currency conversion -------------------------------------------------
    #: Spot rate against the rate Sharadar published with the latest statements.
    #: A gap this large is an inverted quote or the wrong pair, not a move.
    fx_spot_vs_reported: float = 0.15
    #: Annualised change in the reporting currency's USD value below which the
    #: spot-rate assumption is not credible (a hyperinflationary currency), and
    #: below which it is worth a warning.
    fx_depreciation_block: float = -0.10
    fx_depreciation_warning: float = -0.03

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
            ("SBC share of revenue", f"{self.sbc_share_of_revenue:.0%}"),
            ("Book debt flagged above", f"{self.debt_share_of_capital:.0%} of capital"),
            (
                "Terminal growth ceiling",
                f"lower of {self.long_run_nominal_growth:.0%} and the risk-free rate",
            ),
            (
                "Forecast tax rate band",
                f"{self.forecast_etr_band[0]:.0%}–{self.forecast_etr_band[1]:.0%}",
            ),
            ("WACC band", f"{self.wacc_band[0]:.0%}–{self.wacc_band[1]:.0%}"),
            ("Minimum WACC − g spread", f"{self.min_wacc_growth_spread:.0%}"),
            ("Beta band", f"{self.beta_band[0]:.1f}–{self.beta_band[1]:.1f}"),
            ("Forecast growth above realised CAGR", f"{self.growth_above_realised:.0%} points"),
            ("Year-1 FCFF vs base period", f"{self.jump_off:.0%}"),
            ("Gordon vs exit-multiple value", f"{self.method_divergence:.0%}"),
            (
                "Extreme upside / downside",
                f"±{self.extreme_upside:.0%} (warning); above +{self.extreme_upside_critical:.0%} "
                "(critical)",
            ),
            ("Minority interest share of equity", f"{self.nci_share_of_equity:.0%}"),
            (
                "Acquisitions share of capex and of revenue",
                f"{self.acquisitions_share_of_capex:.0%} and "
                f"{self.acquisitions_share_of_revenue:.0%}",
            ),
            ("Assumed margin outside reported range", f"{self.margin_outside_history:.0%} points"),
            ("One year dominates acquisitions", f"above {self.acquisitions_dominant_year:.0%}"),
            ("Preferred dividends material", f"above {self.preferred_dividend_materiality:.1%} of net income"),
            ("Credit spread table stale", f"after {self.spread_table_max_age_months} months"),
            ("Cash-flow residual (missing net change)", f"{self.cash_flow_residual_warning:.0%} of |CFO|"),
            ("Implied FX effect (foreign filer)", f"{self.implied_fx_max:.0%} of |CFO|"),
            (
                "Final capex ÷ D&A below 1.0 and its 5-year ratio",
                f"by more than {self.capex_to_dna_shortfall:.2f}",
            ),
            ("FX spot vs Sharadar's rate", f"{self.fx_spot_vs_reported:.0%}"),
            (
                "Currency depreciation",
                f"warn below {self.fx_depreciation_warning:.0%} a year, block below "
                f"{self.fx_depreciation_block:.0%}",
            ),
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


class UnsupportedCurrencyError(Exception):
    """Statements in a currency no exchange rate could be found for. BLOCKING.

    A foreign reporter is converted to the price's currency at spot. When no
    spot rate can be fetched, the only alternatives are a guessed rate or a
    value in the wrong units, and both are worse than no value.
    """

    def __init__(
        self, ticker: str, statement_currency: str, price_currency: str, reason: str = ""
    ) -> None:
        self.ticker = ticker
        self.statement_currency = statement_currency
        self.price_currency = price_currency
        self.reason = reason
        super().__init__(
            f"{ticker} reports its financial statements in {statement_currency}, but its "
            f"price is quoted in {price_currency}, and no {statement_currency}/"
            f"{price_currency} exchange rate could be fetched"
            + (f" ({reason})" if reason else "")
            + ". No rate is substituted, so no value is shown rather than one in the "
            "wrong units."
        )


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


#: Worst-first, for sorting and for taking the max severity across a check.
SEVERITY_ORDER = {Severity.BLOCKING: 0, Severity.CRITICAL: 1, Severity.WARNING: 2}


def _period_severity(
    ctx: "ValidationContext", label, severity: Severity = Severity.CRITICAL
) -> tuple[Severity, str]:
    """Downgrades a per-period failure the model cannot possibly have used.

    A reconciliation break in a fiscal year that sits outside the window the
    drivers are averaged over is real and still worth reporting — it says
    something about the feed — but it cannot move the number on screen, so
    raising it at the same severity as a break in the base period drowns the
    ones that can. Correctness of the *displayed valuation* is what CRITICAL
    means here.

    ``ctx.periods_in_use`` of ``None`` means the window is unknown, in which
    case nothing is downgraded: the conservative reading is that any period
    might matter.
    """
    if ctx.periods_in_use is None or str(label) in ctx.periods_in_use:
        return severity, ""
    return Severity.WARNING, (
        " This period sits outside the window the model draws its drivers from, "
        "so it does not feed the valuation shown."
    )


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

    #: The currency the valuation's monetary inputs are in after any
    #: conversion, and the price's quote currency. They must agree.
    statement_currency: str | None = None
    price_currency: str | None = None
    #: For a converted company: the currency it reports in, the spot rate used
    #: (USD per unit of it) and its date, Sharadar's published ``fxusd`` for the
    #: latest period (units per USD — the opposite quote), and ``(date, fxusd)``
    #: over up to the last five years, oldest first.
    reporting_currency: str | None = None
    fx_spot: float | None = None
    fx_spot_as_of: date | None = None
    fx_reported: float | None = None
    fx_history: tuple[tuple[date, float], ...] = ()
    #: A non-US filer, whatever it reports in — may be on IFRS.
    foreign_filer: bool = False
    #: One flag per statement set fetched for this run (annual, quarterly, TTM).
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
    #: Base-period SBC, the one item method A adds back and method B does not.
    base_sbc: float | None = None

    effective_tax_rate: float | None = None
    beta_r_squared: float | None = None
    beta_observations: int | None = None
    equity_risk_premium: float | None = None

    terminal_growth: float | None = None
    risk_free_rate: float | None = None
    #: The rate discounting the forecast.
    wacc: float | None = None

    sbc: float | None = None
    latest_revenue: float | None = None
    debt_weight: float | None = None
    long_term_debt: float | None = None

    balance_sheet_delta_nwc: float | None = None

    # --- Forecast and outcome, for the forecast-sanity checks ---------------
    #: The driver values the projection actually used, one per forecast year.
    forecast_tax_rates: tuple[float, ...] = ()
    forecast_growth: tuple[float, ...] = ()
    #: Five-year revenue CAGR — the default growth basis's own window — and
    #: the reported operating-margin range: every fiscal year plus the TTM
    #: margin, which is reported data rather than an assumption.
    realised_cagr: float | None = None
    #: The margins the user *assumed*, by name: the target when one is set, and
    #: the starting margin when it was typed or taken from a non-TTM basis. An
    #: untouched TTM start is reported data and is not listed.
    assumed_margins: dict[str, float] = field(default_factory=dict)
    #: The basis the year-1 growth driver was taken from.
    growth_basis: str | None = None
    historical_margin_range: tuple[float, float] | None = None
    beta: float | None = None
    cost_of_debt: float | None = None
    cost_of_equity: float | None = None
    equity_weight: float | None = None
    #: Return on new invested capital in the terminal year.
    ronic: float | None = None
    #: Final forecast year's capex and D&A, and whether it still grows.
    final_capex: float | None = None
    final_dna: float | None = None
    final_growth: float | None = None
    #: The lowest perpetual growth the exit multiple could be justified by, and
    #: the return on new capital it implies at the terminal growth rate.
    exit_implied_growth: float | None = None
    exit_implied_ronic: float | None = None
    #: Year-1 growth the market price implies, per terminal method.
    implied_growth: dict[str, float | None] = field(default_factory=dict)
    #: Σ capex ÷ Σ D&A over the last five fiscal years.
    historical_capex_to_dna: float | None = None
    gordon_terminal_value: float | None = None
    gordon_value_per_share: float | None = None
    exit_value_per_share: float | None = None
    enterprise_values: dict[str, float] = field(default_factory=dict)
    headline_equity_value: float | None = None
    upside: float | None = None
    #: Year-1 projected FCFF, for the jump-off check.
    year_one_fcff: float | None = None
    #: Σ PV of the explicit forecast's FCFF — below zero, the whole value rests
    #: on the terminal year.
    pv_explicit: float | None = None

    #: Latest annual minority interest, parent equity, total assets, and NCI
    #: income over the recent years, for the minority interest check.
    net_income_nci: tuple[float, ...] = ()
    preferred_dividends: float | None = None
    #: Five-year acquisition spend as a share of capex and of revenue, and
    #: whether the user has folded it into the capex driver.
    acquisitions_to_capex: float | None = None
    acquisitions_to_revenue: float | None = None
    #: ``(period, spend)`` over the same five years, oldest first.
    acquisitions_by_year: tuple[tuple[str, float], ...] = ()
    #: Acquisitions across the TTM growth comparison's eight quarters ÷ TTM
    #: revenue, and the two TTM periods compared.
    ttm_window_acquisitions: float | None = None
    ttm_window_periods: tuple[str, ...] = ()
    #: Base-period net income, for preferred-dividend materiality.
    base_net_income: float | None = None
    #: The spread table's as-of date when the synthetic rating set the cost of
    #: debt; ``None`` when a yield was typed.
    spread_table_as_of: date | None = None
    acquisitions_in_capex: bool = False

    #: ``(concept label, resolved column)`` for base-period items reported as
    #: exactly zero, which Sharadar also uses to mean "absent from the filing".
    zero_filled: tuple[tuple[str, str], ...] = ()

    #: True when ``minority_interest`` in ``annual`` is the client's residual
    #: (assets − liabilities − equity) rather than a reported figure.
    minority_interest_derived: bool = True

    #: Period labels the model actually draws on — the driver averaging windows,
    #: the effective-tax-rate window, and the base period. A per-period
    #: reconciliation failure outside this set is reported at WARNING rather than
    #: CRITICAL by :func:`_period_severity`. ``None`` means the window is not
    #: known and nothing is downgraded.
    periods_in_use: frozenset[str] | None = None


# ---------------------------------------------------------------------------
# CRITICAL checks
# ---------------------------------------------------------------------------


def check_balance_sheet_balances(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Total Assets = Total Liabilities + Total Equity (+ Minority Interest).

    Sharadar's equity figure is attributable to the parent, and SF1 has no NCI
    line, so the client *derives* minority interest as exactly this residual.
    Wherever that happened the identity holds by construction and testing it
    proves nothing — an error in liabilities or equity simply becomes "minority
    interest". Those periods are therefore not counted as passes; only a period
    whose NCI came from somewhere else is tested, and the derived figure is
    judged on its own plausibility by :func:`check_minority_interest_plausible`.
    """
    name = "Balance sheet balances"
    frame = ctx.annual
    required = {"total_assets", "total_liabilities", "total_equity"}
    if frame.empty or not required.issubset(frame.columns):
        return [], CheckResult(name, "not evaluated", "Balance sheet totals not resolved.")

    issues: list[Issue] = []
    checked = 0
    by_construction = 0
    for label, row in frame.iterrows():
        assets, liabilities, equity = (
            row["total_assets"],
            row["total_liabilities"],
            row["total_equity"],
        )
        if not all(_finite(v) for v in (assets, liabilities, equity)):
            continue
        nci = row.get("minority_interest")
        if _finite(nci) and ctx.minority_interest_derived:
            by_construction += 1
            continue
        claims = float(liabilities) + float(equity) + (float(nci) if _finite(nci) else 0.0)
        checked += 1
        if relative_gap(assets, claims) > tol.balance_sheet:
            severity, scope_note = _period_severity(ctx, label)
            issues.append(
                ReconciliationIssue(
                    key=f"balance_sheet:{label}",
                    severity=severity,
                    check=name,
                    concept="Total Assets = Total Liabilities + Total Equity + Minority Interest",
                    company=ctx.ticker,
                    period=_fy(label),
                    message="The balance sheet does not balance for this period." + scope_note,
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
        detail = "No period had all three totals."
        if by_construction:
            detail = (
                f"Minority interest is derived as the balance-sheet residual in all "
                f"{by_construction} period(s), so the identity holds by construction; "
                "see the minority interest check."
            )
        return [], CheckResult(name, "not evaluated", detail)
    return issues, _result(name, issues, f"{checked} periods checked", Severity.CRITICAL)


def check_cash_flow_ties(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """CFO + CFI + CFF + FX = Net Change in Cash, under either presentation.

    Two conventions appear in the same feed. The GAAP presentation folds the FX
    translation line into the reported net change in cash; some vintages instead
    report the net change *before* it. Sharadar switches between them mid-series
    — Microsoft's rows change over at FY2006 — so testing only the first
    convention reports thirteen reconciliation failures against a history that
    reconciles exactly.

    Both are therefore tried, and which one a period used is recorded rather
    than absorbed. The FX line is also legitimately absent for companies with no
    translation exposure, in which case the two conventions coincide. Only a
    period that ties under *neither* is a genuine break, and that is the one
    reported — with both gaps shown, so it is clear the convention is not the
    explanation.

    Each gap is scaled by the largest of the two sides *and* |CFO|. The change
    in cash is routinely small next to the flows that net to it, and scaling by
    it alone failed rounding-sized gaps at large companies.

    Two Sharadar feed defects, diagnosed on live data and confirmed against the
    SEC filings, are handled rather than reported as breaks — neither line
    feeds the valuation:

    (a) **A missing net change in cash** (TSM). Sharadar reports it as exactly
        0 while the flows do not net to zero. Missing stays missing: the
        change in balance-sheet cash between the two year ends is compared
        instead, and the substitution is recorded. A residual within the tie
        tolerance passes; one up to ``cash_flow_residual_warning`` of |CFO| is
        a WARNING; beyond that a flow line may be wrong and it stays CRITICAL.
        With either balance missing, the period is not evaluated.
    (b) **A zero-filled FX line at a foreign filer** (SAP). When the FX line is
        0 or absent and the net change (reported, or substituted under (a))
        agrees with the balance sheet, the gap is read as the undisclosed FX
        effect and reported as a WARNING with the implied amount — up to
        ``implied_fx_max`` of |CFO|. A domestic filer's gap stays CRITICAL: for
        it, a broken flow line is likelier than a missing FX figure.
    """
    name = "Cash flow statement ties"
    frame = ctx.annual
    required = {"cfo", "cfi", "cff", "net_change_in_cash"}
    if frame.empty or not required.issubset(frame.columns):
        return [], CheckResult(name, "not evaluated", "Cash flow subtotals not resolved.")

    issues: list[Issue] = []
    checked = 0
    fx_missing = 0
    fx_excluded = 0
    substituted_ties = 0
    implied_fx_periods = 0
    unevaluable = 0
    bs_change = _cash_change(frame)
    for label, row in frame.iterrows():
        cfo, cfi, cff, net = (
            row["cfo"], row["cfi"], row["cff"], row["net_change_in_cash"],
        )
        if not all(_finite(v) for v in (cfo, cfi, cff, net)):
            continue
        fx = row.get("fx_effect")
        has_fx = _finite(fx)
        fx_value = float(fx) if has_fx else 0.0
        core = float(cfo) + float(cfi) + float(cff)
        flows = core + fx_value
        cfo_scale = abs(float(cfo))
        change = bs_change.get(label)

        # (a) A net change of exactly 0 against flows that do not net to zero
        # is a missing figure: compare the balance sheet instead, or nothing.
        substituted = float(net) == 0.0 and _tie_gap(flows, 0.0, cfo_scale) > tol.cash_flow_tie
        if substituted and not _finite(change):
            unevaluable += 1
            continue
        comparator = float(change) if substituted else float(net)
        checked += 1
        if not has_fx:
            fx_missing += 1

        if _tie_gap(flows, comparator, cfo_scale) <= tol.cash_flow_tie:
            substituted_ties += substituted
            continue
        if has_fx and fx_value != 0.0 and _tie_gap(core, comparator, cfo_scale) <= tol.cash_flow_tie:
            # The publisher reported its own total before the FX line for this
            # period. The statement is internally consistent; only the
            # presentation differs.
            fx_excluded += 1
            continue

        common_values = {
            "CFO + CFI + CFF": _money(core),
            "FX translation": _money(fx) if has_fx else "not reported",
            "Net change in cash (reported)": _money(net),
            "Change in balance-sheet cash": _money(change),
        }

        # (b) A zero-filled FX line at a foreign filer, where the net change
        # agrees with the balance sheet: the gap is the undisclosed FX effect.
        fx_absent = not has_fx or fx_value == 0.0
        agrees_with_balance_sheet = substituted or (
            _finite(change) and _tie_gap(comparator, float(change), cfo_scale) <= tol.cash_flow_tie
        )
        implied_fx = comparator - core
        if (
            ctx.foreign_filer and fx_absent and agrees_with_balance_sheet and cfo_scale
            and abs(implied_fx) <= tol.implied_fx_max * cfo_scale
        ):
            implied_fx_periods += 1
            issues.append(
                ReconciliationIssue(
                    key=f"cash_flow_tie:{label}",
                    severity=Severity.WARNING,
                    check=name,
                    concept="CFO + CFI + CFF + FX = Net Change in Cash",
                    company=ctx.ticker,
                    period=_fy(label),
                    message=(
                        "The flows do not tie, but the FX line is "
                        + ("not reported" if not has_fx else "0 — Sharadar's zero-fill")
                        + " and the "
                        + ("balance-sheet cash change (Sharadar's net change is missing) "
                           if substituted else "reported net change ")
                        + "agrees with the balance sheet, so the gap is read as the "
                        "undisclosed FX effect of a foreign filer. It does not feed the "
                        "valuation; check it against the filing's FX line."
                    ),
                    values={
                        **common_values,
                        "Implied FX effect": _money(implied_fx),
                        "Implied FX ÷ |CFO|": f"{abs(implied_fx) / cfo_scale:.2%}",
                    },
                    tolerance=(
                        f"implied FX ≤ {tol.implied_fx_max:.0%} of |CFO| at a foreign filer"
                    ),
                )
            )
            continue

        residual = abs(flows - comparator)
        residual_share = residual / cfo_scale if cfo_scale else float("inf")
        if substituted and residual_share <= tol.cash_flow_residual_warning:
            issues.append(
                ReconciliationIssue(
                    key=f"cash_flow_tie:{label}",
                    severity=Severity.WARNING,
                    check=name,
                    concept="CFO + CFI + CFF + FX = Change in balance-sheet cash",
                    company=ctx.ticker,
                    period=_fy(label),
                    message=(
                        "Sharadar reports the net change in cash as 0 — missing, not nil — "
                        "so the change in balance-sheet cash stands in for it. The flows "
                        "miss it by a small residual."
                    ),
                    values={
                        **common_values,
                        "Residual": _money(flows - comparator),
                        "Residual ÷ |CFO|": f"{residual_share:.2%}",
                    },
                    tolerance=f"residual ≤ {tol.cash_flow_residual_warning:.0%} of |CFO|",
                )
            )
            continue

        severity, scope_note = _period_severity(ctx, label)
        issues.append(
            ReconciliationIssue(
                key=f"cash_flow_tie:{label}",
                severity=severity,
                check=name,
                concept="CFO + CFI + CFF + FX = Net Change in Cash",
                company=ctx.ticker,
                period=_fy(label),
                message=(
                    (
                        "Sharadar reports the net change in cash as 0 — missing, not nil — "
                        "and the flows miss the change in balance-sheet cash by more than a "
                        "rounding residual, so a flow line, possibly capex, may be wrong."
                        if substituted
                        else "The cash flow statement does not tie under either "
                             "presentation of the FX line."
                    )
                    + (
                        " No FX translation line was reported, so the gap is "
                        "unexplained rather than attributable to currency effects."
                        if not has_fx and not substituted
                        else ""
                    )
                    + (
                        " The FX line is exactly 0, which is also what Sharadar writes "
                        "for an FX effect the feed does not carry; at a domestic filer a "
                        "broken flow line is the likelier explanation."
                        if has_fx and fx_value == 0.0 and not ctx.foreign_filer
                        else ""
                    )
                    + scope_note
                ),
                values={
                    **common_values,
                    "Difference including FX": _money(flows - comparator),
                    "Difference excluding FX": _money(core - comparator),
                    "Difference ÷ |CFO|": f"{residual_share:.2%}" if cfo_scale else "n/a",
                },
                tolerance=f"{tol.cash_flow_tie:.3%}",
            )
        )
    if not checked:
        return [], CheckResult(
            name, "not evaluated",
            "No period had all four subtotals"
            + (f"; {unevaluable} with a missing net change and no balance sheet to compare"
               if unevaluable else "")
            + ".",
        )
    detail = f"{checked} periods checked"
    if fx_missing:
        detail += f"; {fx_missing} without an FX line"
    if fx_excluded:
        detail += (
            f"; {fx_excluded} whose reported total excludes the FX line and tie without it"
        )
    if substituted_ties:
        detail += (
            f"; {substituted_ties} with a missing net change that tie to the balance-sheet "
            "cash change instead"
        )
    if implied_fx_periods:
        detail += f"; {implied_fx_periods} explained by an undisclosed FX effect"
    if unevaluable:
        detail += f"; {unevaluable} not evaluated (net change missing, no balance sheet)"
    return issues, _result(name, issues, detail, Severity.CRITICAL)


def _cash_change(frame: pd.DataFrame) -> pd.Series:
    """Change in balance-sheet cash from the previous fiscal year's end.

    ``NaN`` where either balance is missing or the previous row is not the
    immediately preceding fiscal year — a change across a gap is not the
    change the year's cash flow statement describes.
    """
    if "cash" not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    change = pd.to_numeric(frame["cash"], errors="coerce").diff()
    labels = list(frame.index)
    for position, label in enumerate(labels):
        previous = labels[position - 1] if position else None
        consecutive = (
            previous is not None
            and isinstance(label, (int, np.integer))
            and isinstance(previous, (int, np.integer))
            and int(label) - int(previous) == 1
        )
        if not consecutive:
            change.iloc[position] = np.nan
    return change


def _tie_gap(flows: float, net_change: float, cfo_scale: float) -> float:
    """|flows − net change| over the largest of the two and |CFO|."""
    scale = max(abs(flows), abs(net_change), cfo_scale)
    if scale == 0:
        return 0.0
    return abs(flows - net_change) / scale


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
            severity, scope_note = _period_severity(ctx, fiscal_year)
            issues.append(
                ReconciliationIssue(
                    key=f"quarters_sum:{fiscal_year}",
                    severity=severity,
                    check=name,
                    concept="Σ quarterly revenue = fiscal year revenue",
                    company=ctx.ticker,
                    period=f"FY{fiscal_year}",
                    message=(
                        "The four reported quarters do not sum to the reported year."
                        + scope_note
                    ),
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
            severity, scope_note = _period_severity(ctx, label)
            issues.append(
                ReconciliationIssue(
                    key=f"ebit_reconciliation:{label}",
                    severity=severity,
                    check=name,
                    concept="Revenue + Cost of Revenue + Operating Expenses = EBIT",
                    company=ctx.ticker,
                    period=_fy(label),
                    message=(
                        "Operating income does not reconcile to its components." + scope_note
                    ),
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
            severity, scope_note = _period_severity(ctx, label)
            issues.append(
                ReconciliationIssue(
                    key=f"total_debt:{label}",
                    severity=severity,
                    check=name,
                    concept="Total Debt = Debt Current + Debt Non-Current",
                    company=ctx.ticker,
                    period=_fy(label),
                    message=(
                        "The total debt line does not equal the sum of its parts." + scope_note
                    ),
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
    """FCFF from the cash flow statement against FCFF from the P&L, reconciled.

    The two builds differ for known reasons before anything is wrong: method A
    starts from operating cash flow, which has added stock-based compensation
    back, while method B starts from operating income, which has expensed it.
    SBC is therefore taken out of method A before the two are compared, and
    what is left — deferred taxes, impairments, gains on disposal, working
    capital measured differently — is the unexplained residual this reports.

    A WARNING, not CRITICAL: neither build feeds the projection, which runs off
    the forecast drivers. The check that protects the valuation is
    :func:`check_jump_off`. Raising a red banner that fires routinely at every
    software company teaches readers to acknowledge without reading.
    """
    name = "FCFF method A agrees with method B"
    if not (_finite(ctx.fcff_method_a) and _finite(ctx.fcff_method_b)):
        return [], CheckResult(
            name, "not evaluated", "One of the two methods could not be computed."
        )

    method_a = float(ctx.fcff_method_a)
    method_b = float(ctx.fcff_method_b)
    sbc = float(ctx.base_sbc) if _finite(ctx.base_sbc) else None
    reconciled_a = method_a - sbc if sbc is not None else method_a
    gap = relative_gap(reconciled_a, method_b)
    if gap > tol.fcff_methods:
        values = {"Method A (cash flow build)": _money(method_a)}
        if sbc is not None:
            values["Less: SBC added back in operating cash flow"] = _money(-sbc)
            values["Method A after SBC"] = _money(reconciled_a)
        values.update(
            {
                "Method B (income statement build)": _money(method_b),
                "Unexplained residual": _money(reconciled_a - method_b),
                "Divergence": f"{gap:.2%}",
            }
        )
        return [
            ReconciliationIssue(
                key="fcff_methods",
                severity=Severity.WARNING,
                check=name,
                concept="Free cash flow to the firm",
                company=ctx.ticker,
                period="Base period",
                message=(
                    "After taking SBC out of the cash-flow build, the two FCFF builds still "
                    "diverge. Deferred taxes, impairments, disposal gains, and working "
                    "capital moved by acquisitions or FX all sit in operating cash flow and "
                    "not in operating income. Neither build feeds the projection."
                    + (
                        ""
                        if sbc is not None
                        else " No SBC figure was available, so it is not reconciled out."
                    )
                ),
                values=values,
                tolerance=f"{tol.fcff_methods:.2%}",
            )
        ], CheckResult(name, "fail", f"{gap:.2%} residual", Severity.WARNING)
    return [], CheckResult(
        name, "pass", f"{gap:.2%} apart" + (" after SBC" if sbc is not None else "")
    )


def check_currency_consistent(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Every monetary input to the valuation is in the price's currency.

    A foreign reporter's statements are converted at spot before they meet the
    price, so this is the check that the conversion actually happened: what
    reaches the projection and the bridge must be in the same currency as the
    price per share it is compared with.
    """
    name = "Valuation inputs in the price's currency"
    if not ctx.statement_currency or not ctx.price_currency:
        return [], CheckResult(name, "not evaluated", "A currency was not reported.")
    if ctx.statement_currency != ctx.price_currency:
        return [
            ReconciliationIssue(
                key="currency_mixed",
                severity=Severity.CRITICAL,
                check=name,
                concept="Currency of the valuation inputs",
                company=ctx.ticker,
                period=None,
                message=(
                    "The valuation's monetary inputs are in one currency and the price in "
                    "another, so value per share and upside are in the wrong units."
                ),
                values={
                    "Valuation inputs": ctx.statement_currency,
                    "Price": ctx.price_currency,
                },
                tolerance="one currency",
            )
        ], CheckResult(name, "fail", "", Severity.CRITICAL)
    detail = ctx.statement_currency
    if ctx.reporting_currency and ctx.reporting_currency != ctx.statement_currency:
        detail += f", converted from {ctx.reporting_currency}"
    return [], CheckResult(name, "pass", detail)


def check_fx_matches_reported(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The spot rate against the rate Sharadar published with the statements.

    Sharadar's ``fxusd`` is quoted as units of the reporting currency per US
    dollar (TSM ≈ 31 TWD), the inverse of the spot quote used here (USD per
    unit). A gap beyond the tolerance is far more likely an inverted quote or
    the wrong currency pair than a real move, and every monetary input is
    scaled by the rate.
    """
    name = "Spot FX rate agrees with Sharadar's"
    if not ctx.reporting_currency or ctx.reporting_currency == ctx.price_currency:
        return [], CheckResult(name, "not evaluated", "No conversion needed.")
    if not (_finite(ctx.fx_spot) and _finite(ctx.fx_reported)) or not ctx.fx_reported:
        return [], CheckResult(name, "not evaluated", "Sharadar published no rate.")
    spot = float(ctx.fx_spot)
    reported = 1.0 / float(ctx.fx_reported)
    gap = relative_gap(spot, reported)
    if gap > tol.fx_spot_vs_reported:
        return _flag(
            ReconciliationIssue, ctx=ctx, key="fx_vs_reported", severity=Severity.CRITICAL,
            name=name, concept="Exchange rate",
            message=(
                "The spot rate and the rate Sharadar published with the latest statements "
                "are far apart. That is usually an inverted quote or the wrong currency "
                "pair, and every converted figure moves with it."
            ),
            values={
                f"Spot (USD per {ctx.reporting_currency})": f"{spot:.6g}",
                f"Sharadar (1 ÷ {ctx.fx_reported:g} {ctx.reporting_currency} per USD)":
                    f"{reported:.6g}",
                "Gap": f"{gap:.1%}",
            },
            tolerance=f"≤ {tol.fx_spot_vs_reported:.0%}",
            detail=f"{gap:.1%} apart",
        )
    return [], CheckResult(name, "pass", f"{gap:.1%} apart")


def check_fx_spot_assumption(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """A converted valuation assumes today's spot rate holds for good.

    Always a warning when a conversion happened: the drivers come from the
    reporting-currency history, and the whole forecast is translated at one
    rate. A currency that moves changes the dollar value one for one.
    """
    name = "Spot rate assumed to hold"
    if not ctx.reporting_currency or ctx.reporting_currency == ctx.price_currency:
        return [], CheckResult(name, "not evaluated", "No conversion needed.")
    rate = f"{ctx.fx_spot:.6g}" if _finite(ctx.fx_spot) else "n/a"
    return _flag(
        ImplausibleValueIssue, ctx=ctx, key="fx_spot_assumption", severity=Severity.WARNING,
        name=name, concept="Exchange rate",
        message=(
            f"Revenue, margins and reinvestment are forecast in {ctx.reporting_currency} "
            "and converted to US dollars once, at today's spot rate, which the valuation "
            "then assumes holds forever. The value moves one for one with the rate."
        ),
        values={
            f"Spot rate (USD per {ctx.reporting_currency})": rate,
            "As of": ctx.fx_spot_as_of.isoformat() if ctx.fx_spot_as_of else "n/a",
        },
        tolerance="always noted for a converted company",
        detail=f"{rate} USD per {ctx.reporting_currency}",
    )


def check_currency_depreciation(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """How fast the reporting currency has lost value against the dollar.

    Measured from Sharadar's own ``fxusd`` over up to the last five years, so no
    list of "problem currencies" is hardcoded. A currency falling faster than
    the blocking threshold makes the spot-rate assumption not credible — a
    hyperinflationary currency's statements are not a forecast basis — and is
    BLOCKING; a milder fall is a warning.
    """
    name = "Reporting currency stable against the dollar"
    if not ctx.reporting_currency or ctx.reporting_currency == ctx.price_currency:
        return [], CheckResult(name, "not evaluated", "No conversion needed.")
    points = [(d, float(v)) for d, v in ctx.fx_history if _finite(v) and float(v) > 0]
    if len(points) < 2:
        return [], CheckResult(name, "not evaluated", "Fewer than two published rates.")
    (first_date, first), (last_date, last) = points[0], points[-1]
    years = (last_date - first_date).days / 365.0
    if years <= 0:
        return [], CheckResult(name, "not evaluated", "The published rates share a date.")
    # fxusd is units per USD, so the currency's dollar value moves as first ÷ last.
    change = (first / last) ** (1.0 / years) - 1.0
    if change >= tol.fx_depreciation_warning:
        return [], CheckResult(name, "pass", f"{change:+.1%} a year")
    blocking = change < tol.fx_depreciation_block
    severity = Severity.BLOCKING if blocking else Severity.WARNING
    return _flag(
        ImplausibleValueIssue, ctx=ctx, key="fx_depreciation", severity=severity,
        name=name, concept="Exchange rate",
        message=(
            f"The {ctx.reporting_currency} has lost value against the dollar at "
            f"{-change:.1%} a year. "
            + (
                "At that pace a valuation that assumes today's spot rate holds is not "
                "credible, so none is shown."
                if blocking
                else "A valuation that holds today's spot rate forever overstates the "
                     "dollar value of later years' cash flows if the trend continues."
            )
        ),
        values={
            "Published rates": f"{first:g} → {last:g} {ctx.reporting_currency} per USD",
            "Period": f"{first_date.isoformat()} to {last_date.isoformat()}",
            "Annualised change in value": f"{change:+.1%}",
        },
        tolerance=(
            f"warn below {tol.fx_depreciation_warning:.0%} a year, block below "
            f"{tol.fx_depreciation_block:.0%}"
        ),
        detail=f"{change:+.1%} a year",
    )


def check_foreign_accounting(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """A non-US filer may report under IFRS, which moves three things the model reads."""
    name = "Accounting standard of a foreign filer"
    if not ctx.foreign_filer:
        return [], CheckResult(name, "not evaluated", "A US filer.")
    return _flag(
        ImplausibleValueIssue, ctx=ctx, key="foreign_accounting", severity=Severity.WARNING,
        name=name, concept="Accounting standard",
        message=(
            "This is a foreign filer and may report under IFRS — many report under US GAAP "
            "instead, so check the basis of preparation in the filing. If it is IFRS: under "
            "IFRS 16 lease costs "
            "sit below EBITDA and EBIT (as depreciation and interest), so margins and the "
            "exit multiple's EBITDA are not comparable with US GAAP peers, and lease "
            "liabilities may sit inside debt. Interest paid may also be classified in "
            "financing cash flow, in which case method A's interest add-back "
            "double-counts."
        ),
        values={"Filer": "foreign"},
        tolerance="noted for every foreign filer",
        detail="may be IFRS",
    )


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
    """The historical effective tax rate inside a credible band.

    This is the median rate the base-period FCFF cross-check uses. The rate the
    *forecast* carries is the user's driver, and is judged separately — and more
    tightly — by :func:`check_forecast_tax_rate`. One-off settlements and
    valuation-allowance releases produce genuine outliers, so this flags rather
    than corrects.
    """
    name = "Historical effective tax rate plausible"
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


def check_explicit_forecast_adds_value(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The explicit forecast is worth something on its own.

    Replaces the terminal-value-share warning, which fired on every company
    under a ten-year exit-multiple model and so said nothing. The share is still
    shown beside each method; this raises only the case that changes how the
    answer should be read — a forecast whose years consume more cash than they
    generate in total, so every dollar of value comes after year N. It is the
    same condition as a terminal share above 100%.
    """
    name = "Explicit forecast adds value"
    if not _finite(ctx.pv_explicit):
        return [], CheckResult(name, "not evaluated", "No valuation computed yet.")
    total = float(ctx.pv_explicit)
    if total < 0:
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="explicit_forecast_negative",
            severity=Severity.WARNING, name=name, concept="Explicit forecast",
            message=(
                "Taken together, the forecast years consume more cash than they generate, "
                "so the entire value rests on what happens after year N — the terminal "
                "value is more than 100% of enterprise value."
            ),
            values={"Σ PV of forecast FCFF": _money(total)},
            tolerance="≥ 0",
            detail=_money(total),
        )
    return [], CheckResult(name, "pass", _money(total))


def _growth_ceiling(ctx: ValidationContext, tol: Tolerances) -> tuple[float, str]:
    """The highest perpetual growth rate worth underwriting, and why.

    The lower of long-run nominal GDP growth and the risk-free rate: a company
    cannot outgrow the economy forever, and the risk-free rate is the market's
    own long-run nominal growth expectation.
    """
    ceiling = tol.long_run_nominal_growth
    basis = f"long-run nominal GDP growth of {ceiling:.2%}"
    if _finite(ctx.risk_free_rate) and float(ctx.risk_free_rate) < ceiling:
        ceiling = float(ctx.risk_free_rate)
        basis = f"the risk-free rate of {ceiling:.2%}"
    return ceiling, basis


def check_terminal_growth_reasonable(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Terminal growth against the risk-free rate and long-run GDP growth."""
    name = "Terminal growth below the economy's"
    if not _finite(ctx.terminal_growth):
        return [], CheckResult(name, "not evaluated", "No terminal growth set.")

    growth = float(ctx.terminal_growth)
    ceiling, basis = _growth_ceiling(ctx, tol)

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
                    "Perpetual growth above long-run nominal growth implies the company "
                    "eventually becomes the whole economy."
                ),
                values={
                    "Terminal growth": f"{growth:.2%}",
                    "Long-run nominal GDP growth": f"{tol.long_run_nominal_growth:.2%}",
                    "Risk-free rate": f"{ctx.risk_free_rate:.2%}"
                    if _finite(ctx.risk_free_rate)
                    else "n/a",
                },
                tolerance=f"≤ {ceiling:.2%} ({basis})",
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
                    "Stock-based compensation is large relative to revenue. It is already "
                    "expensed here — it sits inside operating income — so this model's FCFF "
                    "is lower than the free cash flow most analysts and data vendors quote, "
                    "which add SBC back as non-cash. Compare like with like: their FCF less "
                    "SBC is this model's figure."
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
# Data plausibility the reconciliations cannot see
# ---------------------------------------------------------------------------


def _flag(
    issue_type: type[Issue],
    *,
    ctx: ValidationContext,
    key: str,
    severity: Severity,
    name: str,
    concept: str,
    message: str,
    values: dict[str, str],
    tolerance: str,
    detail: str,
) -> tuple[list[Issue], CheckResult]:
    """One issue and its failing result — the shape every single-figure check returns."""
    return [
        issue_type(
            key=key,
            severity=severity,
            check=name,
            concept=concept,
            company=ctx.ticker,
            period=None,
            message=message,
            values=values,
            tolerance=tolerance,
        )
    ], CheckResult(name, "fail", detail, severity)


def check_minority_interest_plausible(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The derived minority interest, judged on its own terms.

    Because it is the balance-sheet residual, any error in liabilities or equity
    lands here — and is then deducted in the equity bridge as if it were a real
    claim. Three things expose it: a negative figure, which no real
    non-controlling stake can be; a large one, which the bridge takes at book
    rather than market value; and a disagreement with NCI *income*, since a
    stake that earns nothing, or income with no stake behind it, is more likely
    mezzanine equity or a data error than a minority shareholder.
    """
    name = "Minority interest plausible"
    frame = ctx.annual
    if frame.empty or "minority_interest" not in frame.columns:
        return [], CheckResult(name, "not evaluated", "No minority interest figure.")
    latest = frame["minority_interest"].dropna()
    if latest.empty:
        return [], CheckResult(name, "not evaluated", "No minority interest figure.")

    label = latest.index[-1]
    nci = float(latest.iloc[-1])
    row = frame.loc[label]
    assets = row.get("total_assets")
    equity = row.get("total_equity")
    material = _finite(assets) and abs(nci) > tol.nci_materiality * abs(float(assets))

    problems: list[str] = []
    if material and nci < 0:
        problems.append(
            "It is negative, which no real non-controlling stake can be: liabilities or "
            "equity is misreported, and the bridge is *adding* the difference as value."
        )
    if _finite(equity) and float(equity) > 0 and nci > tol.nci_share_of_equity * float(equity):
        problems.append(
            "It is large relative to parent equity and is carried at book value; the "
            "cash flows it has a claim on are in FCFF at 100%, so its market value is "
            "what the bridge should deduct."
        )
    income = [v for v in ctx.net_income_nci if _finite(v)]
    if income:
        earns = any(abs(v) > 0 for v in income)
        if material and nci > 0 and not earns:
            problems.append(
                "The balance sheet shows a stake but no NCI income was reported in recent "
                "years — often mezzanine or temporary equity rather than a minority holder."
            )
        if not material and earns:
            problems.append(
                "NCI income was reported but the balance sheet shows no stake, so the "
                "residual has probably absorbed it into equity or liabilities."
            )

    if not problems:
        return [], CheckResult(name, "pass", f"{_money(nci)} at {_fy(label)}")
    return [
        ImplausibleValueIssue(
            key="minority_interest",
            severity=Severity.WARNING,
            check=name,
            concept="Minority interest (derived as assets − liabilities − equity)",
            company=ctx.ticker,
            period=_fy(label),
            message=" ".join(problems),
            values={
                "Minority interest": _money(nci),
                "Parent equity": _money(equity),
                "Total assets": _money(assets),
                "NCI income, recent years": ", ".join(_money(v) for v in income) or "not reported",
            },
            tolerance=(
                f"non-negative; ≤ {tol.nci_share_of_equity:.0%} of equity; consistent with NCI "
                f"income (material above {tol.nci_materiality:.1%} of assets)"
            ),
        )
    ], CheckResult(name, "fail", f"{len(problems)} concern(s)", Severity.WARNING)


def check_preferred_equity(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Material preferred dividends, with no preferred equity in the bridge.

    SF1 has no preferred balance, so the bridge's preferred line is always
    "not reported". Sharadar's preferred-dividend line is the only evidence of a
    senior claim — and it holds other items at some filers, so only a figure
    material against net income is raised, and the message says to check.
    """
    name = "Preferred equity accounted for"
    if not _finite(ctx.preferred_dividends):
        return [], CheckResult(name, "not evaluated", "No preferred dividend line reported.")
    dividends = float(ctx.preferred_dividends)
    if dividends == 0.0:
        return [], CheckResult(name, "pass", "No preferred dividends")
    if not _finite(ctx.base_net_income) or float(ctx.base_net_income) == 0.0:
        return [], CheckResult(
            name, "not evaluated", "No net income to judge the preferred dividends against."
        )
    share = abs(dividends) / abs(float(ctx.base_net_income))
    if share <= tol.preferred_dividend_materiality:
        return [], CheckResult(name, "pass", f"{_money(dividends)}, {share:.2%} of net income")
    return _flag(
        ImplausibleValueIssue, ctx=ctx, key="preferred_equity", severity=Severity.WARNING,
        name=name, concept="Preferred equity",
        message=(
            f"Sharadar reports preferred dividends of {_money(dividends)}. This line can "
            "also hold other items, so check the filing for preferred shares outstanding "
            "before adjusting the bridge — if there are some, their value is not deducted "
            "and value per share is overstated by it."
        ),
        values={
            "Preferred dividends (base period)": _money(dividends),
            "Share of net income": f"{share:.2%}",
        },
        tolerance=f"≤ {tol.preferred_dividend_materiality:.1%} of net income",
        detail=f"{_money(dividends)}, {share:.2%} of net income",
    )


def check_acquisitions(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Material acquisition spend that the reinvestment driver leaves out.

    Reported revenue growth includes what was bought, and capex excludes what
    it cost, so a serial acquirer's history projects acquired growth for free.
    Material means against capex *and* against revenue: a company that spends
    almost nothing on capex clears the capex test on any purchase at all.
    """
    name = "Acquisition spend in reinvestment"
    if not (_finite(ctx.acquisitions_to_capex) and _finite(ctx.acquisitions_to_revenue)):
        return [], CheckResult(
            name, "not evaluated", "Acquisitions, capex or revenue not reported."
        )
    of_capex = float(ctx.acquisitions_to_capex)
    of_revenue = float(ctx.acquisitions_to_revenue)
    material = (
        of_capex > tol.acquisitions_share_of_capex
        and of_revenue > tol.acquisitions_share_of_revenue
    )
    detail = f"{of_capex:.0%} of capex, {of_revenue:.1%} of revenue"
    concentration = _acquisition_concentration(ctx.acquisitions_by_year, tol)
    if material and not ctx.acquisitions_in_capex:
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="acquisitions", severity=Severity.WARNING,
            name=name, concept="Business acquisitions",
            message=(
                "Acquisition spend is material next to both capex and revenue. The growth "
                "driver includes the revenue those deals bought, but the capex driver "
                "leaves out what they cost, so FCFF is overstated. Fold acquisitions into "
                "reinvestment, or lower the growth driver to organic growth."
                + (f" {concentration}." if concentration else "")
            ),
            values={
                "Acquisitions ÷ capex (5 years)": f"{of_capex:.0%}",
                "Acquisitions ÷ revenue (5 years)": f"{of_revenue:.1%}",
                **({"By year": concentration} if concentration else {}),
            },
            tolerance=(
                f"not both above {tol.acquisitions_share_of_capex:.0%} of capex and "
                f"{tol.acquisitions_share_of_revenue:.0%} of revenue"
            ),
            detail=detail,
        )
    return [], CheckResult(
        name, "pass",
        detail + (" (included in reinvestment)" if ctx.acquisitions_in_capex else ""),
    )


# ---------------------------------------------------------------------------
# Forecast sanity — is the forecast plausible, not just the data
# ---------------------------------------------------------------------------


def _acquisition_concentration(
    by_year: tuple[tuple[str, float], ...], tol: Tolerances
) -> str:
    """Whether one deal or a pattern: the largest year's share of the total."""
    spends = [(label, float(v)) for label, v in by_year if _finite(v)]
    total = sum(v for _, v in spends)
    if not spends or total <= 0:
        return ""
    label, largest = max(spends, key=lambda item: item[1])
    share = largest / total
    if share > tol.acquisitions_dominant_year:
        return f"One year ({label}) accounts for {share:.0%} of the 5-year total"
    active = sum(1 for _, v in spends if v > 0)
    return f"Spread across {active} years; the largest ({label}) is {share:.0%} of the total"


def check_ttm_acquired_growth(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Whether the trailing growth rate itself may include acquired revenue.

    The five-year acquisitions check says a company buys growth; this says
    whether the *year-1 growth the forecast starts from* spans a deal — the
    two trailing-twelve-month periods it compares. The growth default is not
    changed; the reader is told.
    """
    from app.logic.dcf.financials import TTM_GROWTH_BASIS

    name = "Trailing growth may include acquired revenue"
    if ctx.growth_basis != TTM_GROWTH_BASIS:
        return [], CheckResult(name, "not evaluated", "The growth basis is not TTM.")
    if not _finite(ctx.ttm_window_acquisitions):
        return [], CheckResult(
            name, "not evaluated", "Acquisitions over the TTM window are not reported."
        )
    share = float(ctx.ttm_window_acquisitions)
    window = " and ".join(ctx.ttm_window_periods)
    if share > tol.acquisitions_share_of_revenue:
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="ttm_acquired_growth",
            severity=Severity.WARNING, name=name, concept="Revenue growth",
            message=(
                "Acquisitions over the eight quarters the TTM growth rate compares are "
                "material against revenue, so part of the year-1 growth may have been "
                "bought rather than earned — and will not recur organically. Consider a "
                "history basis for year-1 growth, or check the deal's revenue in the filing."
            ),
            values={
                "Acquisitions ÷ TTM revenue (8 quarters)": f"{share:.1%}",
                "TTM periods compared": window or "n/a",
            },
            tolerance=f"≤ {tol.acquisitions_share_of_revenue:.0%} of TTM revenue",
            detail=f"{share:.1%} of TTM revenue",
        )
    return [], CheckResult(name, "pass", f"{share:.1%} of TTM revenue")


def check_spread_table_current(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The hardcoded credit-spread table is not out of date.

    Only evaluated when the synthetic rating actually set the cost of debt —
    a typed yield does not read the table.
    """
    name = "Credit spread table is current"
    if ctx.spread_table_as_of is None or ctx.valuation_date is None:
        return [], CheckResult(
            name, "not evaluated", "The cost of debt did not come from the spread table."
        )
    as_of, today = ctx.spread_table_as_of, ctx.valuation_date
    months = (today.year - as_of.year) * 12 + (today.month - as_of.month) - (
        1 if today.day < as_of.day else 0
    )
    shifted = as_of.month - 1 + tol.spread_table_max_age_months
    cutoff = date(as_of.year + shifted // 12, shifted % 12 + 1, min(as_of.day, 28))
    if today > cutoff:
        return _flag(
            StaleDataIssue, ctx=ctx, key="spread_table_stale", severity=Severity.WARNING,
            name=name, concept="Cost of debt",
            message=(
                "The synthetic-rating spreads that set the cost of debt are more than "
                f"{tol.spread_table_max_age_months} months old. Refresh SYNTHETIC_RATINGS "
                "in app/logic/dcf/wacc.py from Damodaran's January update."
            ),
            values={"Table as of": as_of.isoformat(), "Age": f"{months} months"},
            tolerance=f"≤ {tol.spread_table_max_age_months} months",
            detail=f"{months} months old",
        )
    return [], CheckResult(name, "pass", f"{months} months old")


def check_forecast_tax_rate(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The tax rate the projection actually uses, not the historical median."""
    name = "Forecast tax rate plausible"
    rates = [float(r) for r in ctx.forecast_tax_rates if _finite(r)]
    if not rates:
        return [], CheckResult(name, "not evaluated", "No forecast has been run.")
    low, high = tol.forecast_etr_band
    outside = [r for r in rates if not low <= r <= high]
    if outside:
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="forecast_etr", severity=Severity.WARNING,
            name=name, concept="Forecast effective tax rate",
            message=(
                "The forecast carries a tax rate outside the sustainable band for the whole "
                "horizon. A one-off historical rate is explainable; projecting it is a claim."
            ),
            values={"Forecast tax rate(s)": ", ".join(f"{r:.2%}" for r in sorted(set(outside)))},
            tolerance=f"{low:.0%}–{high:.0%}",
            detail=", ".join(f"{r:.2%}" for r in sorted(set(outside))),
        )
    return [], CheckResult(name, "pass", f"{min(rates):.2%}–{max(rates):.2%}")


def check_wacc_sanity(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """WACC in a credible band, and far enough above terminal growth."""
    name = "WACC plausible"
    if not _finite(ctx.wacc):
        return [], CheckResult(name, "not evaluated", "No WACC computed.")
    wacc = float(ctx.wacc)
    low, high = tol.wacc_band
    problems: list[str] = []
    if not low <= wacc <= high:
        problems.append(f"The WACC of {wacc:.2%} is outside {low:.0%}–{high:.0%}.")
    spread = None
    if _finite(ctx.terminal_growth):
        spread = wacc - float(ctx.terminal_growth)
        if spread < tol.min_wacc_growth_spread:
            problems.append(
                f"WACC − g is only {spread:.2%}, so the perpetuity is mostly that gap: a "
                "tenth of a point on either input moves the terminal value by several percent."
            )
    if problems:
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="wacc_sanity", severity=Severity.WARNING,
            name=name, concept="Weighted average cost of capital",
            message=" ".join(problems),
            values={
                "WACC": f"{wacc:.2%}",
                "WACC − terminal growth": f"{spread:.2%}" if spread is not None else "n/a",
            },
            tolerance=(
                f"{low:.0%}–{high:.0%}; WACC − g ≥ {tol.min_wacc_growth_spread:.0%}"
            ),
            detail=f"{wacc:.2%}",
        )
    return [], CheckResult(name, "pass", f"{wacc:.2%}")


def check_cost_of_debt_sanity(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Risk-free ≤ cost of debt ≤ cost of equity.

    Lending to a company costs more than lending to the government and less than
    owning it. Only meaningful when debt carries weight in the WACC.
    """
    name = "Cost of debt between risk-free and cost of equity"
    if not (_finite(ctx.cost_of_debt) and _finite(ctx.risk_free_rate) and _finite(ctx.cost_of_equity)):
        return [], CheckResult(name, "not evaluated", "A rate is missing.")
    if _finite(ctx.debt_weight) and float(ctx.debt_weight) == 0.0:
        return [], CheckResult(name, "not evaluated", "No debt in the capital structure.")
    kd, rf, ke = float(ctx.cost_of_debt), float(ctx.risk_free_rate), float(ctx.cost_of_equity)
    if not rf <= kd <= ke:
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="kd_sanity", severity=Severity.WARNING,
            name=name, concept="Cost of debt",
            message=(
                "The cost of debt sits below the risk-free rate or above the cost of "
                "equity. Neither is a price a lender would charge."
            ),
            values={
                "Risk-free rate": f"{rf:.2%}",
                "Cost of debt": f"{kd:.2%}",
                "Cost of equity": f"{ke:.2%}",
            },
            tolerance="risk-free ≤ Kd ≤ Ke",
            detail=f"Kd {kd:.2%}",
        )
    return [], CheckResult(name, "pass", f"{rf:.2%} ≤ {kd:.2%} ≤ {ke:.2%}")


def check_beta_range(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The beta in use — regressed or typed — inside a credible band."""
    name = "Beta in a plausible range"
    if not _finite(ctx.beta):
        return [], CheckResult(name, "not evaluated", "No beta in use.")
    beta = float(ctx.beta)
    low, high = tol.beta_band
    if not low <= beta <= high:
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="beta_range", severity=Severity.WARNING,
            name=name, concept="Beta",
            message=(
                "The beta in use is outside the range almost every listed operating "
                "company falls in. A single-stock regression has a wide error band; an "
                "industry beta, unlevered and relevered, is the usual remedy."
            ),
            values={"Beta": f"{beta:.2f}"},
            tolerance=f"{low:.1f}–{high:.1f}",
            detail=f"β {beta:.2f}",
        )
    return [], CheckResult(name, "pass", f"β {beta:.2f}")


def check_growth_vs_history(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Year-1 revenue growth against the five-year compound rate.

    The five-year window is the default growth basis's own, so the comparison
    is like for like — not against a CAGR stretching back decades, which says
    little about a company today.
    """
    from app.logic.dcf.financials import TTM_GROWTH_BASIS

    name = "Forecast growth against realised growth"
    growth = [float(g) for g in ctx.forecast_growth if _finite(g)]
    if not growth or not _finite(ctx.realised_cagr):
        return [], CheckResult(name, "not evaluated", "No five-year CAGR or no forecast.")
    start, cagr = growth[0], float(ctx.realised_cagr)
    if start > cagr + tol.growth_above_realised:
        trailing = ctx.growth_basis == TTM_GROWTH_BASIS
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="growth_vs_history", severity=Severity.WARNING,
            name=name, concept="Revenue growth",
            message=(
                "Growth over the last twelve months has accelerated well above the "
                "five-year trend, and the forecast starts from the faster rate. Holding it "
                "is the forecast's biggest claim — check it is a new trend, not one strong "
                "year."
                if trailing
                else "The forecast starts well above the rate the company has compounded "
                     "revenue at over the last five years. That can be right — an "
                     "inflection, a new product — but it is the forecast's biggest claim "
                     "and should be a deliberate one."
            ),
            values={
                "Year-1 forecast growth": f"{start:.2%}",
                "Basis": ctx.growth_basis or "n/a",
                "Five-year CAGR": f"{cagr:.2%}",
            },
            tolerance=f"≤ five-year CAGR + {tol.growth_above_realised:.0%} points",
            detail=f"{start:.2%} vs {cagr:.2%}",
        )
    return [], CheckResult(name, "pass", f"{start:.2%} vs {cagr:.2%} five-year CAGR")


def check_margin_vs_history(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Assumed operating margins against the range the company has reported.

    The range includes the TTM margin: a record trailing margin is reported
    data, and flagging it for being a record said nothing. What is tested is
    what the user *assumed* — a target margin, or a starting margin typed by
    hand or taken from a history basis — with a tolerance either side.
    """
    name = "Assumed margin inside the reported range"
    if ctx.historical_margin_range is None:
        return [], CheckResult(name, "not evaluated", "No margin history.")
    assumed = {k: float(v) for k, v in ctx.assumed_margins.items() if _finite(v)}
    if not assumed:
        return [], CheckResult(name, "not evaluated", "No margin was assumed.")
    low, high = ctx.historical_margin_range
    slack = tol.margin_outside_history
    outside = {k: v for k, v in assumed.items() if not low - slack <= v <= high + slack}
    if outside:
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="margin_vs_history", severity=Severity.WARNING,
            name=name, concept="Operating margin",
            message=(
                "An assumed margin sits well outside anything the company has reported, "
                "trailing twelve months included. Margin expansion is a claim about the "
                "business, not a default."
            ),
            values={
                **{k: f"{v:.2%}" for k, v in outside.items()},
                "Reported range (annual + TTM)": f"{low:.2%} – {high:.2%}",
            },
            tolerance=f"within the reported range ± {slack:.0%} points",
            detail=", ".join(f"{k} {v:.2%}" for k, v in outside.items()),
        )
    return [], CheckResult(
        name, "pass", ", ".join(f"{k} {v:.2%}" for k, v in assumed.items())
    )


def check_jump_off(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Year-1 FCFF against the base period's, grown one year.

    The check that protects the valuation rather than the data: year 1 is built
    purely from averaged drivers, so a company whose margins or reinvestment
    have moved since the averaging window starts its forecast somewhere it is
    not. The base figure is the income-statement build, which is on the same
    footing as the projection.
    """
    name = "Year-1 FCFF follows on from the base period"
    growth = [float(g) for g in ctx.forecast_growth if _finite(g)]
    if not (_finite(ctx.year_one_fcff) and _finite(ctx.fcff_method_b)) or not growth:
        return [], CheckResult(name, "not evaluated", "No base-period FCFF to compare with.")
    expected = float(ctx.fcff_method_b) * (1.0 + growth[0])
    actual = float(ctx.year_one_fcff)
    gap = relative_gap(actual, expected)
    if gap > tol.jump_off:
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="jump_off", severity=Severity.WARNING,
            name=name, concept="Free cash flow to the firm",
            message=(
                "Year 1 of the forecast is far from where the base period leaves off. The "
                "drivers are averages over past years, so a margin or reinvestment rate "
                "that has since moved puts the whole projection on a different footing "
                "from today's business."
            ),
            values={
                "Base-period FCFF (method B) × (1 + year-1 growth)": _money(expected),
                "Year-1 projected FCFF": _money(actual),
                "Gap": f"{gap:.1%}",
            },
            tolerance=f"±{tol.jump_off:.0%}",
            detail=f"{gap:.1%} apart",
        )
    return [], CheckResult(name, "pass", f"{gap:.1%} apart")


def check_terminal_reinvestment(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The terminal year's return on new capital, and reinvestment going into it."""
    name = "Terminal reinvestment consistent with growth"
    if not _finite(ctx.ronic) or not _finite(ctx.wacc):
        return [], CheckResult(name, "not evaluated", "No terminal year computed.")
    ronic, wacc = float(ctx.ronic), float(ctx.wacc)
    # (short label for the summary line, sentence for the issue)
    problems: list[tuple[str, str]] = []
    if _finite(ctx.terminal_growth) and float(ctx.terminal_growth) > 0 and ronic < wacc:
        problems.append((
            f"RONIC {ronic:.2%} below WACC",
            f"New capital earns {ronic:.2%} in perpetuity against a {wacc:.2%} cost, so "
            "every dollar of terminal growth destroys value.",
        ))
    capex_ratio = None
    if (
        _finite(ctx.final_capex) and _finite(ctx.final_dna) and _finite(ctx.final_growth)
        and float(ctx.final_dna) > 0 and _finite(ctx.historical_capex_to_dna)
    ):
        capex_ratio = float(ctx.final_capex) / float(ctx.final_dna)
        history = float(ctx.historical_capex_to_dna)
        # Capex at or above D&A replaces everything D&A wears out, so the asset
        # base cannot be shrinking whatever the history says — and with capex
        # normalised from a build-out, the history is the build-out: MSFT's
        # final year spends 1.75× D&A against 2.28× over its last five years.
        if (
            float(ctx.final_growth) > 0
            and capex_ratio < 1.0
            and capex_ratio < history - tol.capex_to_dna_shortfall
        ):
            problems.append((
                "capex ÷ D&A below its own history",
                f"The final forecast year still grows but spends {capex_ratio:.2f}× D&A on "
                f"capex — less than it depreciates, and less than the {history:.2f}× of the "
                "last five years — so the asset base reads as shrinking while revenue "
                "rises. (Capex below D&A alone is normal where D&A includes amortisation of "
                "acquired intangibles, which capex never replaces — the comparison is with "
                "the company's own ratio for that reason.)",
            ))
    if problems:
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="terminal_reinvestment",
            severity=Severity.WARNING, name=name, concept="Reinvestment",
            message=" ".join(sentence for _, sentence in problems),
            values={
                "RONIC": f"{ronic:.2%}",
                "WACC": f"{wacc:.2%}",
                "Final-year capex ÷ D&A": f"{capex_ratio:.2f}" if capex_ratio is not None
                else "n/a",
                "Five-year capex ÷ D&A": f"{ctx.historical_capex_to_dna:.2f}"
                if _finite(ctx.historical_capex_to_dna) else "n/a",
            },
            tolerance=(
                "RONIC ≥ WACC when g > 0; while growing, final capex ÷ D&A at or above "
                f"1.00 or within {tol.capex_to_dna_shortfall:.2f} of its five-year ratio"
            ),
            detail="; ".join(label for label, _ in problems),
        )
    return [], CheckResult(name, "pass", f"RONIC {ronic:.2%}")


def check_exit_implied_growth(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """The growth an exit multiple needs, even if growth were free.

    A multiple earned during a high-growth period, applied after the forecast
    has cut growth to its perpetual rate, prices in growth the model no longer
    has. This inverts the multiple assuming *no* reinvestment at all — the most
    generous reading — so a rate above the ceiling is a multiple too rich for
    the terminal assumptions under any return on capital.
    """
    name = "Exit multiple consistent with perpetual growth"
    if not _finite(ctx.exit_implied_growth):
        return [], CheckResult(name, "not evaluated", "No exit multiple implied growth.")
    growth = float(ctx.exit_implied_growth)
    ceiling, basis = _growth_ceiling(ctx, tol)
    if growth > ceiling:
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="exit_implied_growth",
            severity=Severity.WARNING, name=name, concept="Exit multiple",
            message=(
                "Even if growth needed no reinvestment, the exit multiple only makes "
                "sense with perpetual growth above the ceiling. The multiple is carrying "
                "more than the terminal assumptions support — usually because it was "
                "earned while the company was growing faster than the forecast now has it. "
                "A lower multiple, of the kind a mature company in the sector trades on, is "
                "the usual fix."
            ),
            values={"Growth the multiple requires": f"{growth:.2%}"},
            tolerance=f"≤ {ceiling:.2%} ({basis})",
            detail=f"{growth:.2%}",
        )
    return [], CheckResult(name, "pass", f"{growth:.2%} required")


def check_method_divergence(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Gordon growth and the exit multiple, compared per share."""
    name = "Terminal value methods agree"
    if not (_finite(ctx.gordon_value_per_share) and _finite(ctx.exit_value_per_share)):
        return [], CheckResult(name, "not evaluated", "Both methods did not produce a value.")
    gordon, exit_value = float(ctx.gordon_value_per_share), float(ctx.exit_value_per_share)
    gap = relative_gap(gordon, exit_value)
    if gap > tol.method_divergence:
        values = {
            "Gordon growth value per share": f"{gordon:,.2f}",
            "Exit multiple value per share": f"{exit_value:,.2f}",
            "Gap": f"{gap:.1%}",
        }
        if _finite(ctx.exit_implied_growth):
            values["Growth the multiple needs with no reinvestment"] = (
                f"{ctx.exit_implied_growth:.2%}"
            )
        values["RONIC the multiple implies"] = (
            f"{ctx.exit_implied_ronic:.1%}" if _finite(ctx.exit_implied_ronic)
            else "none reaches it"
        )
        if _finite(ctx.ronic):
            values["RONIC in the Gordon value"] = f"{ctx.ronic:.1%}"
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="method_divergence",
            severity=Severity.WARNING, name=name, concept="Terminal value",
            message=(
                "The two terminal value methods disagree materially. The usual cause: the "
                "exit multiple prices in value creation beyond year N — growth earning more "
                "than its cost of capital — that the Gordon value, at its return on new "
                "capital, excludes. The figures below say how much the multiple assumes."
                if exit_value > gordon
                else "The two terminal value methods disagree materially, with the exit "
                     "multiple below the Gordon value — the multiple is pricing the company "
                     "for less than its own terminal assumptions deliver."
            ),
            values=values,
            tolerance=f"≤ {tol.method_divergence:.0%}",
            detail=f"{gap:.1%} apart",
        )
    return [], CheckResult(name, "pass", f"{gap:.1%} apart")


def check_negative_outcomes(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """A negative terminal value, enterprise value, or equity value."""
    name = "Valuation outputs are positive"
    if not ctx.enterprise_values:
        return [], CheckResult(name, "not evaluated", "No valuation computed yet.")
    negatives: dict[str, str] = {}
    if _finite(ctx.gordon_terminal_value) and float(ctx.gordon_terminal_value) < 0:
        negatives["Gordon terminal value"] = _money(ctx.gordon_terminal_value)
    for method, value in ctx.enterprise_values.items():
        if _finite(value) and float(value) < 0:
            negatives[f"Enterprise value ({method.replace('_', ' ')})"] = _money(value)
    if _finite(ctx.headline_equity_value) and float(ctx.headline_equity_value) < 0:
        negatives["Headline equity value"] = _money(ctx.headline_equity_value)
    if negatives:
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="negative_outcome",
            severity=Severity.CRITICAL, name=name, concept="Valuation output",
            message=(
                "The model produced a negative value. A negative terminal cash flow grown "
                "in perpetuity, or debt exceeding the enterprise value, is not a price — "
                "the inputs describe a business that destroys value forever."
            ),
            values=negatives,
            tolerance="≥ 0",
            detail=", ".join(negatives),
        )
    return [], CheckResult(name, "pass", "All positive")


def check_equity_weight(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """Equity carries weight in the WACC.

    A missing price or point-in-time share count makes the market value of
    equity zero, and the WACC then collapses to the after-tax cost of debt with
    nothing else to say so.
    """
    name = "Equity carries weight in the WACC"
    if not _finite(ctx.equity_weight):
        return [], CheckResult(name, "not evaluated", "No WACC computed.")
    weight = float(ctx.equity_weight)
    if weight <= 0:
        return _flag(
            ReconciliationIssue, ctx=ctx, key="equity_weight", severity=Severity.CRITICAL,
            name=name, concept="Market value of equity",
            message=(
                "The equity weight is zero, so the WACC is just the after-tax cost of "
                "debt. The price or the point-in-time share count is missing."
            ),
            values={"Equity weight": f"{weight:.1%}"},
            tolerance="> 0",
            detail=f"{weight:.1%}",
        )
    return [], CheckResult(name, "pass", f"{weight:.1%}")


def check_extreme_result(
    ctx: ValidationContext, tol: Tolerances
) -> tuple[list[Issue], CheckResult]:
    """An upside or downside large enough that an input error is the likelier story."""
    name = "Result within a plausible distance of the price"
    if not _finite(ctx.upside):
        return [], CheckResult(name, "not evaluated", "No price to compare with.")
    upside = float(ctx.upside)
    if abs(upside) > tol.extreme_upside:
        critical = upside > tol.extreme_upside_critical
        values = {"Upside / downside": f"{upside:+.1%}"}
        if _finite(ctx.exit_implied_growth):
            values["Growth the exit multiple needs with no reinvestment"] = (
                f"{ctx.exit_implied_growth:.2%}"
            )
        return _flag(
            ImplausibleValueIssue, ctx=ctx, key="extreme_result",
            severity=Severity.CRITICAL if critical else Severity.WARNING,
            name=name, concept="Value per share",
            message=(
                "The value is far from the market price. Markets are rarely that wrong "
                "about a liquid stock, so check the inputs before reading it as an "
                "opportunity — the reverse DCF below shows the year-1 growth the price "
                "implies under each method, which is usually where the disagreement sits."
            ),
            values={
                **values,
                **{
                    f"Implied year-1 growth ({label})": (
                        f"{ctx.implied_growth[method]:.2%}"
                        if _finite(ctx.implied_growth.get(method)) else "no single rate"
                    )
                    for method, label in (
                        ("gordon", "Gordon"), ("exit_multiple", "exit multiple")
                    )
                    if method in ctx.implied_growth
                },
            },
            tolerance=(
                f"within ±{tol.extreme_upside:.0%}; critical above "
                f"+{tol.extreme_upside_critical:.0%}"
            ),
            detail=f"{upside:+.1%}",
        )
    return [], CheckResult(name, "pass", f"{upside:+.1%}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _result(
    name: str, issues: list[Issue], detail: str, severity: Severity
) -> CheckResult:
    """``severity`` is the tier the check reports at *when nothing is scoped down*.

    Per-period checks can downgrade individual failures that fall outside the
    window the model uses, so the roll-up takes the worst severity actually
    raised rather than the check's nominal one — otherwise the summary line
    would read CRITICAL while every issue under it was a warning.
    """
    if issues:
        worst = min(issues, key=lambda i: SEVERITY_ORDER[i.severity]).severity
        return CheckResult(
            name, "fail", f"{len(issues)} failure(s); {detail}", worst
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
    check_fx_matches_reported,
    check_fx_spot_assumption,
    check_currency_depreciation,
    check_foreign_accounting,
    check_asreported_consistent,
    check_sufficient_history,
    check_etr_plausible,
    check_beta_quality,
    check_erp_plausible,
    check_terminal_growth_reasonable,
    check_sbc_share,
    check_zero_filled_indicators,
    check_share_count_currency_of_date,
    check_book_debt_proxy,
    check_minority_interest_plausible,
    check_preferred_equity,
    check_acquisitions,
    check_ttm_acquired_growth,
    check_spread_table_current,
    check_forecast_tax_rate,
    check_wacc_sanity,
    check_cost_of_debt_sanity,
    check_beta_range,
    check_growth_vs_history,
    check_margin_vs_history,
    check_jump_off,
    check_terminal_reinvestment,
    check_exit_implied_growth,
    check_method_divergence,
    check_negative_outcomes,
    check_explicit_forecast_adds_value,
    check_equity_weight,
    check_extreme_result,
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

    issues.sort(key=lambda i: (SEVERITY_ORDER[i.severity], i.check, i.period or ""))
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
