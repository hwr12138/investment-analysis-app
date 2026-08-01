"""FCFF construction, projection, terminal value, and the equity bridge.

Pure — no Streamlit, no network, no globals. Numbers in, dataclasses out, so
every formula in this module is unit-testable against a hand-calculated fixture
and batch screening is a loop over these functions rather than a rewrite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

#: Long-run nominal GDP growth. A terminal growth rate above this implies the
#: company eventually becomes the entire economy, which is worth flagging.
LONG_RUN_NOMINAL_GDP_GROWTH = 0.04


class TerminalValueError(Exception):
    """The terminal value cannot be computed as specified."""


class ProjectionError(Exception):
    """The projection inputs are internally inconsistent."""


# ---------------------------------------------------------------------------
# Free cash flow to the firm
# ---------------------------------------------------------------------------


def fcff_from_cash_flow(
    cfo: float, interest_expense: float, tax_rate: float, capex: float
) -> float:
    """Method A — the cash flow statement build.

        FCFF = CFO + Interest Expense × (1 − tax rate) − CapEx

    The interest add-back assumes interest paid sits in operating cash flow.
    That holds under US GAAP; an IFRS filer may classify it in financing, in
    which case adding it back double-counts. Note also that interest *expense*
    is an accrual and interest *paid* is cash — SimFin exposes only the former,
    so that is what this uses and what the UI says it used.
    """
    _require_finite(cfo=cfo, interest_expense=interest_expense, tax_rate=tax_rate, capex=capex)
    return cfo + interest_expense * (1.0 - tax_rate) - capex


def fcff_from_income_statement(
    ebit: float, tax_rate: float, dna: float, capex: float, delta_nwc: float
) -> float:
    """Method B — the income statement build.

        NOPAT = EBIT × (1 − tax rate)
        FCFF  = NOPAT + D&A − CapEx − ΔNWC
    """
    _require_finite(ebit=ebit, tax_rate=tax_rate, dna=dna, capex=capex, delta_nwc=delta_nwc)
    return ebit * (1.0 - tax_rate) + dna - capex - delta_nwc


def _require_finite(**values: float) -> None:
    for name, value in values.items():
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            raise ProjectionError(f"{name} is missing or not a finite number ({value!r}).")


def divergence(a: float, b: float) -> float:
    """Relative gap between two figures, scaled by the larger magnitude.

    Scaling by the larger of the two keeps the measure symmetric and stops a
    near-zero denominator from reporting an infinite divergence.
    """
    scale = max(abs(a), abs(b))
    if scale == 0:
        return 0.0
    return abs(a - b) / scale


# ---------------------------------------------------------------------------
# Stub period
# ---------------------------------------------------------------------------


def stub_fraction(valuation_date: date, next_fiscal_year_end: date) -> float:
    """Fraction of a year from the valuation date to the next fiscal year end.

        stub fraction = days to next fiscal year end ÷ 365

    A valuation struck mid-fiscal-year discounts its first cash flow over less
    than a full year. Getting this wrong is a common and material error, so the
    fraction is computed explicitly and displayed next to the fiscal year end it
    derives from. Returns 1.0 when the valuation date is exactly a year out.
    """
    days = (next_fiscal_year_end - valuation_date).days
    if days <= 0:
        raise ProjectionError(
            f"The next fiscal year end ({next_fiscal_year_end}) is not after the "
            f"valuation date ({valuation_date})."
        )
    return min(days / 365.0, 1.0)


def discount_factor(rate: float, exponent: float) -> float:
    """1 ÷ (1 + rate)^exponent."""
    if rate <= -1:
        raise ProjectionError(f"A discount rate of {rate} is not usable.")
    return 1.0 / ((1.0 + rate) ** exponent)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriverPath:
    """One value per forecast year for each driver.

    Every tuple must be exactly ``forecast_years`` long. A driver that varies by
    year is the point — a single rate repeated is just a special case.
    """

    revenue_growth: tuple[float, ...]
    operating_margin: tuple[float, ...]
    effective_tax_rate: tuple[float, ...]
    capex_pct_revenue: tuple[float, ...]
    dna_pct_revenue: tuple[float, ...]
    dnwc_pct_drevenue: tuple[float, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.revenue_growth),
            len(self.operating_margin),
            len(self.effective_tax_rate),
            len(self.capex_pct_revenue),
            len(self.dna_pct_revenue),
            len(self.dnwc_pct_drevenue),
        }
        if len(lengths) != 1:
            raise ProjectionError(
                "Every driver must have one value per forecast year; got lengths "
                f"{sorted(lengths)}."
            )

    @property
    def years(self) -> int:
        return len(self.revenue_growth)

    @classmethod
    def flat(
        cls,
        years: int,
        *,
        revenue_growth: float,
        operating_margin: float,
        effective_tax_rate: float,
        capex_pct_revenue: float,
        dna_pct_revenue: float,
        dnwc_pct_drevenue: float,
    ) -> "DriverPath":
        """Repeats a single value for each driver across every forecast year."""
        return cls(
            revenue_growth=(revenue_growth,) * years,
            operating_margin=(operating_margin,) * years,
            effective_tax_rate=(effective_tax_rate,) * years,
            capex_pct_revenue=(capex_pct_revenue,) * years,
            dna_pct_revenue=(dna_pct_revenue,) * years,
            dnwc_pct_drevenue=(dnwc_pct_drevenue,) * years,
        )


@dataclass(frozen=True)
class ProjectionRow:
    """One forecast year, with every intermediate figure kept for display.

    Full-year figures and the stub-scaled cash flow are both retained so the
    projection table stays arithmetically checkable line by line.
    """

    label: str
    period_fraction: float
    cumulative_time: float
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
    sbc_deduction: float
    fcff_full_year: float
    fcff_period: float
    discount_factor: float
    present_value: float


@dataclass(frozen=True)
class Projection:
    rows: tuple[ProjectionRow, ...]
    base_revenue: float
    stub_fraction: float
    mid_year: bool
    wacc: float

    @property
    def pv_explicit(self) -> float:
        return sum(row.present_value for row in self.rows)

    @property
    def terminal_row(self) -> ProjectionRow:
        return self.rows[-1]

    @property
    def terminal_ebitda(self) -> float:
        row = self.rows[-1]
        return row.ebit + row.dna


def project(
    base_revenue: float,
    drivers: DriverPath,
    *,
    wacc: float,
    stub: float = 1.0,
    mid_year: bool = True,
    sbc: float | None = None,
    deduct_sbc: bool = False,
    sbc_growth_with_revenue: bool = True,
) -> Projection:
    """Projects revenue through to a discounted FCFF for each forecast year.

    The first period is scaled by ``stub``: its cash flow covers only the part
    of the fiscal year that remains, and its discount exponent is the stub
    fraction (or half of it under the mid-year convention). Compounding still
    runs off full-year revenue, so a stub does not permanently shrink the
    forecast.

    ``sbc`` is deducted from FCFF only when ``deduct_sbc`` is set — the standard
    treatment adds SBC back as a non-cash charge (which operating cash flow has
    already done), and deducting it instead treats it as the real economic cost
    it is. Both are defensible; the choice is the user's and is recorded.
    """
    if base_revenue is None or not math.isfinite(base_revenue):
        raise ProjectionError("Base revenue is missing or not a finite number.")
    if drivers.years < 1:
        raise ProjectionError("The forecast needs at least one year.")
    if deduct_sbc and sbc is None:
        raise ProjectionError(
            "SBC deduction was requested but no stock-based compensation figure was supplied."
        )

    rows: list[ProjectionRow] = []
    previous_revenue = base_revenue
    cumulative = 0.0

    for index in range(drivers.years):
        growth = drivers.revenue_growth[index]
        margin = drivers.operating_margin[index]
        tax_rate = drivers.effective_tax_rate[index]

        revenue = previous_revenue * (1.0 + growth)
        delta_revenue = revenue - previous_revenue
        ebit = revenue * margin
        nopat = ebit * (1.0 - tax_rate)
        dna = revenue * drivers.dna_pct_revenue[index]
        capex = revenue * drivers.capex_pct_revenue[index]
        delta_nwc = delta_revenue * drivers.dnwc_pct_drevenue[index]

        sbc_deduction = 0.0
        if deduct_sbc and sbc is not None:
            sbc_deduction = (
                sbc * (revenue / base_revenue) if sbc_growth_with_revenue else sbc
            )

        fcff_full = nopat + dna - capex - delta_nwc - sbc_deduction

        length = stub if index == 0 else 1.0
        cumulative += length
        exponent = cumulative - (length / 2.0 if mid_year else 0.0)
        factor = discount_factor(wacc, exponent)
        fcff_period = fcff_full * length

        rows.append(
            ProjectionRow(
                label=f"Year {index + 1}" + (" (stub)" if index == 0 and stub < 1 else ""),
                period_fraction=length,
                cumulative_time=cumulative,
                discount_exponent=exponent,
                revenue=revenue,
                revenue_growth=growth,
                operating_margin=margin,
                ebit=ebit,
                tax_rate=tax_rate,
                nopat=nopat,
                dna=dna,
                capex=capex,
                delta_nwc=delta_nwc,
                sbc_deduction=sbc_deduction,
                fcff_full_year=fcff_full,
                fcff_period=fcff_period,
                discount_factor=factor,
                present_value=fcff_period * factor,
            )
        )
        previous_revenue = revenue

    return Projection(
        rows=tuple(rows),
        base_revenue=base_revenue,
        stub_fraction=stub,
        mid_year=mid_year,
        wacc=wacc,
    )


# ---------------------------------------------------------------------------
# Terminal value — both methods, independently
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerminalValue:
    """One terminal value method's result and its cross-implied counterpart."""

    method: str
    value: float
    present_value: float
    discount_exponent: float
    formula: str
    implied_exit_multiple: float | None = None
    implied_growth: float | None = None

    def share_of(self, enterprise_value: float) -> float | None:
        if not enterprise_value:
            return None
        return self.present_value / enterprise_value


def gordon_growth_terminal_value(final_fcff: float, growth: float, wacc: float) -> float:
    """TV_n = FCFF_n × (1 + g) ÷ (WACC − g).

    Raises when ``g >= WACC``: the formula has no finite value there, and
    letting a negative denominator through produces a large negative terminal
    value that looks like an answer.
    """
    _require_finite(final_fcff=final_fcff, growth=growth, wacc=wacc)
    if growth >= wacc:
        raise TerminalValueError(
            f"Terminal growth of {growth:.2%} is not below the WACC of {wacc:.2%}. "
            "A perpetuity growing at or above its discount rate has no finite value."
        )
    return final_fcff * (1.0 + growth) / (wacc - growth)


def exit_multiple_terminal_value(final_ebitda: float, multiple: float) -> float:
    """TV_n = EBITDA_n × exit multiple."""
    _require_finite(final_ebitda=final_ebitda, multiple=multiple)
    return final_ebitda * multiple


def implied_exit_multiple(terminal_value: float, final_ebitda: float) -> float | None:
    """What the Gordon terminal value implies about the exit multiple."""
    if not final_ebitda:
        return None
    return terminal_value / final_ebitda


def implied_perpetuity_growth(
    terminal_value: float, final_fcff: float, wacc: float
) -> float | None:
    """What an exit multiple implies about perpetuity growth.

    Inverts the Gordon formula:  g = (TV × WACC − FCFF) ÷ (TV + FCFF).
    """
    denominator = terminal_value + final_fcff
    if denominator == 0:
        return None
    return (terminal_value * wacc - final_fcff) / denominator


def build_terminal_values(
    projection: Projection,
    *,
    terminal_growth: float,
    exit_multiple: float,
) -> dict[str, TerminalValue]:
    """Computes both methods independently, each cross-checked against the other."""
    final = projection.terminal_row
    final_fcff = final.fcff_full_year
    final_ebitda = projection.terminal_ebitda
    wacc = projection.wacc

    # The terminal value sits at the end of the final projected year, and is
    # discounted on the same convention as that year's cash flow.
    exponent = final.discount_exponent
    factor = discount_factor(wacc, exponent)

    gordon = gordon_growth_terminal_value(final_fcff, terminal_growth, wacc)
    exit_value = exit_multiple_terminal_value(final_ebitda, exit_multiple)

    return {
        "gordon": TerminalValue(
            method="Gordon growth",
            value=gordon,
            present_value=gordon * factor,
            discount_exponent=exponent,
            formula=(
                f"TV = FCFF_n {final_fcff:,.0f} × (1 + {terminal_growth:.2%}) ÷ "
                f"({wacc:.2%} − {terminal_growth:.2%})"
            ),
            implied_exit_multiple=implied_exit_multiple(gordon, final_ebitda),
        ),
        "exit_multiple": TerminalValue(
            method="Exit multiple",
            value=exit_value,
            present_value=exit_value * factor,
            discount_exponent=exponent,
            formula=f"TV = EBITDA_n {final_ebitda:,.0f} × {exit_multiple:.2f}×",
            implied_growth=implied_perpetuity_growth(exit_value, final_fcff, wacc),
        ),
    }


# ---------------------------------------------------------------------------
# Equity bridge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BridgeLine:
    """One line of the enterprise-to-equity bridge, shown as its own row.

    ``kind`` distinguishes the opening enterprise value and the closing equity
    value from the adjustments between them, so a reader can add up exactly the
    rows that are meant to add up.
    """

    label: str
    amount: float
    reported: bool
    note: str = ""
    kind: str = "adjustment"  # "start" | "adjustment" | "total"


@dataclass(frozen=True)
class EquityBridge:
    enterprise_value: float
    equity_value: float
    shares: float
    value_per_share: float
    share_basis: str
    lines: tuple[BridgeLine, ...] = field(default_factory=tuple)

    def upside_vs(self, current_price: float) -> float | None:
        if not current_price:
            return None
        return self.value_per_share / current_price - 1.0


def build_equity_bridge(
    *,
    enterprise_value: float,
    total_debt: float | None,
    minority_interest: float | None,
    preferred_equity: float | None,
    cash: float | None,
    short_term_investments: float | None,
    non_operating_assets: float | None,
    shares: float,
    share_basis: str,
    include_short_term_investments: bool = True,
    operating_cash_floor: float = 0.0,
) -> EquityBridge:
    """Bridges enterprise value to equity value and then to a per-share figure.

        Equity = EV − debt − minority interest − preferred equity
                    + cash + short-term investments + non-operating assets

    A component SimFin did not report enters as zero *with its own visible row*
    saying so, rather than being folded silently into the total — the user can
    see that Apple reports no minority interest and judge whether that is right.

    ``operating_cash_floor`` holds back cash the business needs to run, so a
    cash-rich company is not credited with every dollar on the balance sheet as
    if it were distributable.
    """
    if shares is None or shares <= 0 or not math.isfinite(shares):
        raise ProjectionError(
            "A positive share count is required to compute a value per share."
        )

    lines: list[BridgeLine] = [
        BridgeLine(
            "Enterprise value",
            enterprise_value,
            True,
            "Σ PV(FCFF) + PV(terminal value)",
            kind="start",
        )
    ]

    def _subtract(label: str, value: float | None, note: str = "") -> float:
        reported = value is not None and math.isfinite(value)
        amount = float(value) if reported else 0.0
        lines.append(
            BridgeLine(
                label,
                -amount,
                reported,
                note if reported else "Not reported by SimFin — treated as nil",
            )
        )
        return amount

    def _add(label: str, value: float | None, note: str = "") -> float:
        reported = value is not None and math.isfinite(value)
        amount = float(value) if reported else 0.0
        lines.append(
            BridgeLine(
                label,
                amount,
                reported,
                note if reported else "Not reported by SimFin — treated as nil",
            )
        )
        return amount

    equity = enterprise_value
    equity -= _subtract("Less: total debt", total_debt)
    equity -= _subtract("Less: minority interest", minority_interest)
    equity -= _subtract("Less: preferred equity", preferred_equity)

    usable_cash = None
    if cash is not None and math.isfinite(cash):
        usable_cash = max(cash - operating_cash_floor, 0.0)
    equity += _add(
        "Plus: cash & cash equivalents",
        usable_cash,
        f"Net of a {operating_cash_floor:,.0f} operating cash floor"
        if operating_cash_floor
        else "",
    )

    if include_short_term_investments:
        equity += _add("Plus: short-term investments", short_term_investments)
    else:
        lines.append(
            BridgeLine(
                "Short-term investments (excluded)",
                0.0,
                short_term_investments is not None,
                "Treated as operating rather than excess by the excess-cash setting",
            )
        )

    equity += _add("Plus: non-operating assets", non_operating_assets)

    lines.append(BridgeLine("Equity value", equity, True, "", kind="total"))

    return EquityBridge(
        enterprise_value=enterprise_value,
        equity_value=equity,
        shares=shares,
        value_per_share=equity / shares,
        share_basis=share_basis,
        lines=tuple(lines),
    )
