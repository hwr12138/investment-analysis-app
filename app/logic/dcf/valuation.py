"""FCFF construction, projection, terminal value, and the equity bridge.

Pure — no Streamlit, no network, no globals. Numbers in, dataclasses out, so
every formula in this module is unit-testable against a hand-calculated fixture
and batch screening is a loop over these functions rather than a rewrite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

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
    is an accrual and interest *paid* is cash — Sharadar exposes only the former,
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
# Timing
# ---------------------------------------------------------------------------


def elapsed_years(period_end: date, valuation_date: date) -> float:
    """Years from the end of the base period to the valuation date.

        elapsed = days from base-period end to valuation date ÷ 365

    The forecast is anchored to the end of the base period: year *i* is the
    twelve months ending *i* years after it, which is what compounding off a
    trailing-twelve-month or fiscal-year base actually describes. The balance
    sheet the equity bridge reads is dated to that same point, so the model
    values the company as of then and rolls the value forward to the valuation
    date at the WACC — which is exactly what subtracting ``elapsed`` from every
    discount exponent does.

    The alternative — a stub to the *next fiscal year end* — puts the forecast
    on a different calendar from its own revenue whenever the base is a TTM
    that does not end on a fiscal year: the value then drifts through the year
    and jumps when the valuation date crosses the fiscal year end, although
    nothing about the company has changed.
    """
    days = (valuation_date - period_end).days
    if days < 0:
        raise ProjectionError(
            f"The base period ends on {period_end}, after the valuation date "
            f"({valuation_date})."
        )
    return days / 365.0


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
    nwc_pct_revenue: tuple[float, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.revenue_growth),
            len(self.operating_margin),
            len(self.effective_tax_rate),
            len(self.capex_pct_revenue),
            len(self.dna_pct_revenue),
            len(self.nwc_pct_revenue),
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
        nwc_pct_revenue: float,
    ) -> "DriverPath":
        """Repeats a single value for each driver across every forecast year."""
        return cls(
            revenue_growth=(revenue_growth,) * years,
            operating_margin=(operating_margin,) * years,
            effective_tax_rate=(effective_tax_rate,) * years,
            capex_pct_revenue=(capex_pct_revenue,) * years,
            dna_pct_revenue=(dna_pct_revenue,) * years,
            nwc_pct_revenue=(nwc_pct_revenue,) * years,
        )


def growth_path(
    start: float, years: int, fade_to: float | None = None, *, hold: int = 1
) -> tuple[float, ...]:
    """Revenue growth for each forecast year: flat, or held and then faded.

    With ``fade_to`` set, years 1…``hold`` grow at ``start`` and each later
    year *i* grows at

        start + (fade_to − start) × (i − hold) ÷ (N − hold + 1)

    — equal steps sized so the fade would land on ``fade_to`` in year N + 1,
    the terminal year. The final forecast year is therefore one step short of
    the perpetuity rate rather than equal to it, and there is no cliff between
    the explicit forecast and the terminal value. ``hold`` of 1 is a straight
    linear fade from year 1.

    Holding matters for a company still compounding fast: a ten-year forecast
    that starts fading at once gives away most of the growth the reader came
    to underwrite, while one that holds for a few years lets the excess return
    live in the explicit forecast rather than in the terminal value.
    """
    if years < 1:
        raise ProjectionError("The forecast needs at least one year.")
    if fade_to is None:
        return (start,) * years
    if not 1 <= hold <= years:
        raise ProjectionError(
            f"Growth can be held for 1 to {years} years of a {years}-year forecast, "
            f"not {hold}."
        )
    span = years - hold + 1
    return tuple(
        start if year <= hold else start + (fade_to - start) * (year - hold) / span
        for year in range(1, years + 1)
    )


def linear_path(start: float, target: float, years: int) -> tuple[float, ...]:
    """A driver moving linearly from ``start`` in year 1 to ``target`` in year N.

    Used for the operating margin, capex % and D&A %. The terminal year takes
    the final forecast year's margin, so a margin ``target`` is also the
    steady-state margin. Equal to ``start`` throughout when no target is set —
    the path is flat unless someone chooses otherwise.
    """
    if years < 1:
        raise ProjectionError("The forecast needs at least one year.")
    if years == 1:
        return (start,)
    return tuple(start + (target - start) * index / (years - 1) for index in range(years))


@dataclass(frozen=True)
class ProjectionRow:
    """One forecast year, with every intermediate figure kept for display."""

    label: str
    #: Years from the end of the base period to the end of this forecast year.
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
    fcff: float
    discount_factor: float
    present_value: float


@dataclass(frozen=True)
class Projection:
    rows: tuple[ProjectionRow, ...]
    base_revenue: float
    #: Years from the end of the base period to the valuation date.
    elapsed: float
    mid_year: bool
    wacc: float

    @property
    def pv_explicit(self) -> float:
        return sum(row.present_value for row in self.rows)

    @property
    def final_row(self) -> ProjectionRow:
        return self.rows[-1]

    @property
    def final_ebitda(self) -> float:
        row = self.rows[-1]
        return row.ebit + row.dna


def project(
    base_revenue: float,
    drivers: DriverPath,
    *,
    wacc: float,
    elapsed: float = 0.0,
    mid_year: bool = True,
) -> Projection:
    """Projects revenue through to a discounted FCFF for each forecast year.

    Year *i* is the twelve months ending *i* years after the base period, and
    is discounted over ``i − elapsed`` years (``i − 0.5 − elapsed`` under the
    mid-year convention) — the value at the base period's end, rolled forward
    to the valuation date. See :func:`elapsed_years`.

    **Stock-based compensation is expensed, once.** The operating margin is
    GAAP operating income over revenue, and GAAP operating expenses already
    include SBC, so projected EBIT is after it: paying staff in stock is treated
    as the real cost it is. There is deliberately no add-back — that treats
    equity pay as free, which only holds together with a share count that
    forecasts the dilution, and no basis here does.
    """
    if base_revenue is None or not math.isfinite(base_revenue):
        raise ProjectionError("Base revenue is missing or not a finite number.")
    if drivers.years < 1:
        raise ProjectionError("The forecast needs at least one year.")
    _require_finite(elapsed=elapsed)

    rows: list[ProjectionRow] = []
    previous_revenue = base_revenue

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
        # NWC is held at a share of revenue, so it moves with the change in
        # revenue: ΔNWC = NWC % × Δrevenue. A negative share releases cash.
        delta_nwc = delta_revenue * drivers.nwc_pct_revenue[index]

        fcff = nopat + dna - capex - delta_nwc

        end_of_year = float(index + 1)
        exponent = end_of_year - (0.5 if mid_year else 0.0) - elapsed
        factor = discount_factor(wacc, exponent)

        rows.append(
            ProjectionRow(
                label=f"Year {index + 1}",
                cumulative_time=end_of_year,
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
                fcff=fcff,
                discount_factor=factor,
                present_value=fcff * factor,
            )
        )
        previous_revenue = revenue

    return Projection(
        rows=tuple(rows),
        base_revenue=base_revenue,
        elapsed=elapsed,
        mid_year=mid_year,
        wacc=wacc,
    )


def incremental_return_on_capital(projection: Projection, base_nopat: float | None) -> float | None:
    """The return the explicit forecast earns on the capital it reinvests.

        (NOPAT_N − NOPAT_base) ÷ Σ (capex − D&A + ΔNWC) over years 1…N

    A diagnostic, not an input: it shows how much excess return the forecast
    carries before the terminal year drops to its RONIC. ``None`` — "not
    meaningful" — when the forecast reinvests nothing net, since growth that
    costs nothing has no finite return, or when the base NOPAT is unknown.
    """
    if base_nopat is None or not math.isfinite(base_nopat):
        return None
    reinvested = sum(row.capex - row.dna + row.delta_nwc for row in projection.rows)
    if reinvested <= 0:
        return None
    return (projection.final_row.nopat - base_nopat) / reinvested


# ---------------------------------------------------------------------------
# Terminal value — both methods, independently
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerminalYear:
    """Year N + 1, normalised to the steady state the perpetuity assumes.

    The final forecast year cannot simply be grown at g. Its reinvestment —
    capex over D&A, working capital — is sized for the *forecast* growth rate,
    so capitalising it at a 2.5% perpetuity understates a fast grower's
    terminal cash flow and overstates one that reinvests less than it
    depreciates (growth for free is an infinite return on new capital).
    Reinvestment here comes from the value-driver identity instead:

        reinvestment = g ÷ RONIC × NOPAT_{N+1}

    so growing at g costs exactly what earning RONIC on new capital requires.
    RONIC equal to the WACC means growth beyond the forecast creates no value.

    ``amortisation`` is the EBITA-proxy add-back, zero when it is off — see
    :func:`terminal_year`.
    """

    revenue: float
    growth: float
    operating_margin: float
    ebit: float
    tax_rate: float
    amortisation: float
    nopat: float
    ronic: float
    reinvestment_rate: float
    reinvestment: float
    fcff: float


def terminal_year(
    projection: Projection,
    *,
    growth: float,
    ronic: float,
    tax_rate: float | None = None,
    ebita_proxy: bool = True,
) -> TerminalYear:
    """Builds year N + 1 from the final forecast year's margin.

    ``tax_rate`` of ``None`` carries the final forecast year's effective rate
    into perpetuity; passing the marginal rate lets a low effective rate —
    credits, deferrals, a foreign mix — lapse at the terminal year rather than
    last forever.

    ``ebita_proxy`` adds back amortisation before NOPAT:

        A_N     = max(D&A_N − capex_N, 0)
        A_{N+1} = A_N × (1 + g)
        NOPAT_{N+1} = (EBIT_{N+1} + A_{N+1}) × (1 − t)

    The forecast years add D&A back and deduct capex; a terminal year built
    from EBIT alone keeps the amortisation charge but loses the add-back, so an
    acquirer depreciating acquired intangibles well above its capex (AVGO: D&A
    16% of revenue, capex 1%) saw FCFF halve from year N to N + 1. The excess of
    D&A over capex is amortisation that genuinely runs off. No cap is needed:
    with acquisitions folded into reinvestment they sit in capex and the gap
    shrinks by itself, and under-investment is what the capex ÷ D&A check is
    for.
    """
    _require_finite(growth=growth, ronic=ronic)
    if ronic <= 0:
        raise TerminalValueError(
            f"A return on new invested capital of {ronic:.2%} is not usable: growth "
            "would have to be bought with unlimited reinvestment."
        )
    final = projection.final_row
    tax = final.tax_rate if tax_rate is None else tax_rate
    _require_finite(terminal_tax_rate=tax)

    revenue = final.revenue * (1.0 + growth)
    ebit = revenue * final.operating_margin
    amortisation = max(final.dna - final.capex, 0.0) * (1.0 + growth) if ebita_proxy else 0.0
    nopat = (ebit + amortisation) * (1.0 - tax)
    reinvestment_rate = growth / ronic
    reinvestment = nopat * reinvestment_rate
    return TerminalYear(
        revenue=revenue,
        growth=growth,
        operating_margin=final.operating_margin,
        ebit=ebit,
        tax_rate=tax,
        amortisation=amortisation,
        nopat=nopat,
        ronic=ronic,
        reinvestment_rate=reinvestment_rate,
        reinvestment=reinvestment,
        fcff=nopat - reinvestment,
    )


@dataclass(frozen=True)
class TerminalValue:
    """One terminal value method's result and its cross-implied counterparts."""

    method: str
    value: float
    present_value: float
    discount_exponent: float
    formula: str
    implied_exit_multiple: float | None = None
    implied_growth: float | None = None
    implied_ronic: float | None = None
    terminal_year: TerminalYear | None = None

    def share_of(self, enterprise_value: float) -> float | None:
        if not enterprise_value:
            return None
        return self.present_value / enterprise_value


def gordon_growth_terminal_value(next_fcff: float, growth: float, wacc: float) -> float:
    """TV_N = FCFF_{N+1} ÷ (WACC − g).

    Raises when ``g >= WACC``: the formula has no finite value there, and
    letting a negative denominator through produces a large negative terminal
    value that looks like an answer.
    """
    _require_finite(next_fcff=next_fcff, growth=growth, wacc=wacc)
    if growth >= wacc:
        raise TerminalValueError(
            f"Terminal growth of {growth:.2%} is not below the WACC of {wacc:.2%}. "
            "A perpetuity growing at or above its discount rate has no finite value."
        )
    return next_fcff / (wacc - growth)


def exit_multiple_terminal_value(final_ebitda: float, multiple: float) -> float:
    """TV_N = EBITDA_N × exit multiple."""
    _require_finite(final_ebitda=final_ebitda, multiple=multiple)
    return final_ebitda * multiple


def implied_exit_multiple(terminal_value: float, final_ebitda: float) -> float | None:
    """What a terminal value implies about the exit multiple."""
    if not final_ebitda:
        return None
    return terminal_value / final_ebitda


def growth_required(terminal_value: float, nopat_before_growth: float, wacc: float) -> float | None:
    """The lowest perpetual growth rate that could justify a terminal value.

    Solves TV = NOPAT × (1 + g) ÷ (WACC − g) for g — NOPAT growing with no
    reinvestment at all, the most generous assumption there is. Any real
    reinvestment raises the rate needed, so a multiple that requires more than
    a plausible perpetual rate *even on these terms* is simply too high.

        g = (TV × WACC − NOPAT) ÷ (TV + NOPAT)
    """
    denominator = terminal_value + nopat_before_growth
    if denominator == 0:
        return None
    return (terminal_value * wacc - nopat_before_growth) / denominator


def implied_ronic(
    terminal_value: float, next_nopat: float, growth: float, wacc: float
) -> float | None:
    """The return on new capital a terminal value implies at growth g.

    Inverts TV = NOPAT_{N+1} × (1 − g ÷ RONIC) ÷ (WACC − g):

        RONIC = g × NOPAT_{N+1} ÷ (NOPAT_{N+1} − TV × (WACC − g))

    ``None`` when growth is not positive (reinvestment is then not what the
    value turns on) or when the terminal value exceeds what NOPAT growing at g
    with *no* reinvestment is worth — no return on capital, however high,
    reaches it.
    """
    if growth <= 0 or growth >= wacc:
        return None
    denominator = next_nopat - terminal_value * (wacc - growth)
    if denominator <= 0:
        return None
    return growth * next_nopat / denominator


def build_terminal_values(
    projection: Projection,
    *,
    terminal_growth: float,
    exit_multiple: float,
    ronic: float | None = None,
    terminal_tax_rate: float | None = None,
    ebita_proxy: bool = True,
) -> dict[str, TerminalValue]:
    """Computes both methods independently, each cross-checked against the other.

    ``ronic`` of ``None`` sets the return on new capital equal to the WACC.
    ``ebita_proxy`` is passed to :func:`terminal_year`; it moves the Gordon value
    and, because the exit row's implied RONIC and zero-reinvestment growth read
    NOPAT_{N+1}, those too — never the exit value itself.

    The two are discounted differently on purpose. The Gordon value is a
    perpetuity of cash flows arriving on the same convention as the forecast's,
    so under mid-year discounting it is worth ``(1 + WACC)^0.5`` more at year N
    and discounting it from the final row's mid-year exponent is algebraically
    the same thing. An exit multiple is a sale price struck at the *end* of
    year N, so it is discounted over the full N years whatever the convention —
    discounting it from mid-year inflates it by ``(1 + WACC)^0.5``.
    """
    final = projection.final_row
    wacc = projection.wacc
    return_on_new_capital = wacc if ronic is None else ronic

    steady = terminal_year(
        projection,
        growth=terminal_growth,
        ronic=return_on_new_capital,
        tax_rate=terminal_tax_rate,
        ebita_proxy=ebita_proxy,
    )
    final_ebitda = projection.final_ebitda

    gordon = gordon_growth_terminal_value(steady.fcff, terminal_growth, wacc)
    gordon_exponent = final.discount_exponent

    exit_value = exit_multiple_terminal_value(final_ebitda, exit_multiple)
    exit_exponent = final.cumulative_time - projection.elapsed

    return {
        "gordon": TerminalValue(
            method="Gordon growth",
            value=gordon,
            present_value=gordon * discount_factor(wacc, gordon_exponent),
            discount_exponent=gordon_exponent,
            formula=(
                (
                    f"NOPAT_N+1 = (EBIT {steady.ebit:,.0f} + amortisation "
                    f"{steady.amortisation:,.0f}) × (1 − {steady.tax_rate:.2%}); "
                    if ebita_proxy else ""
                )
                + f"TV = NOPAT_N+1 {steady.nopat:,.0f} × (1 − {terminal_growth:.2%} ÷ "
                f"RONIC {return_on_new_capital:.2%}) ÷ ({wacc:.2%} − {terminal_growth:.2%})"
            ),
            implied_exit_multiple=implied_exit_multiple(gordon, final_ebitda),
            implied_ronic=return_on_new_capital,
            terminal_year=steady,
        ),
        "exit_multiple": TerminalValue(
            method="Exit multiple",
            value=exit_value,
            present_value=exit_value * discount_factor(wacc, exit_exponent),
            discount_exponent=exit_exponent,
            formula=f"TV = EBITDA_N {final_ebitda:,.0f} × {exit_multiple:.2f}×",
            implied_growth=growth_required(
                exit_value, steady.nopat / (1.0 + terminal_growth), wacc
            ),
            implied_ronic=implied_ronic(exit_value, steady.nopat, terminal_growth, wacc),
            terminal_year=steady,
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

    A component Sharadar did not report enters as zero *with its own visible row*
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
                note if reported else "Not reported by Sharadar — treated as nil",
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
                note if reported else "Not reported by Sharadar — treated as nil",
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
