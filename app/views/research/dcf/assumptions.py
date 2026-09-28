"""The assumptions panel: every input the model takes, each with its provenance.

:func:`collect` draws the panel top to bottom and returns the assumption set the
package ``__init__`` fingerprints and runs. Each block of the panel is its own
function, in the order it draws; they share only the list of export records.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import NamedTuple

import pandas as pd
import streamlit as st

from app.data import market_data
from app.logic.dcf import export as dcf_export
from app.logic.dcf import financials as fin
from app.logic.dcf import model as dcf_model
from app.logic.dcf import validation as val
from app.logic.dcf import valuation as dcf
from app.logic.dcf import wacc as wacc_lib
from app.provenance import Origin, Sourced
from app.views.research.dcf import common

FINRA_FIXED_INCOME_URL = "https://www.finra.org/finra-data/fixed-income"

COST_OF_DEBT_GUIDANCE = f"""
**Where to find this:** FINRA publishes TRACE-reported corporate bond trades and yields, free, at
<{FINRA_FIXED_INCOME_URL}>. Open **Corporate & Agency Bonds**, search by **issuer name** rather
than CUSIP — CUSIPs are per-issue and a large issuer will have dozens, so name search is faster
and less error-prone — then pick an issue **maturing about ten years out** and read its **Yield to
Maturity** off the trade detail. That is the number to type in the field above. The company's most
recent 10-K debt footnote lists every outstanding issue with coupon and maturity, which tells you
what to look for before you search.

**Which bond:** a **recently traded, senior unsecured** issue. Avoid secured, subordinated,
convertible, and thinly-traded issues — none of them price the company's general cost of
borrowing.

**Which maturity:** match the tenor of your risk-free rate. If you're using the 10Y Treasury in
CAPM, use a bond with roughly 10 years remaining to maturity. Consistency between the two matters
more than hitting exactly 10 years. If the company's debt is concentrated at a different tenor,
match its weighted-average maturity and use the same-tenor Treasury.

**Use yield to maturity, not the coupon.** The coupon is what the company agreed to pay at
issuance; YTM is what the market charges today. A DCF wants today's cost.

**No public bonds?** The field already starts from the synthetic rating approach — EBIT ÷ interest
expense mapped to a rating band, plus that band's spread over the risk-free rate — which is an
estimate. A traded bond's yield is better evidence where one exists.
"""


#: Headline terminal value method, label to engine key. The first entry is what
#: the radio selects by default, and it is ordered to match
#: ``dcf_model.DEFAULT_HEADLINE_METHOD`` so the UI and the engine agree.
_HEADLINE_METHODS = {
    "Gordon growth": "gordon",
    "Exit multiple": "exit_multiple",
}

#: Share count bases, label to what they divide by. The first is the default.
_SHARE_BASES = (
    "Point-in-time + dilution",
    "Diluted weighted-average (TTM)",
    "Point-in-time shares outstanding",
)

#: Pre-populated defaults that are conventions rather than measurements. Each is
#: labelled an estimate while untouched, never "user" — the user did not set it.
DEFAULT_TERMINAL_GROWTH = 0.025
DEFAULT_MARGINAL_TAX = 0.25


def _untouched(widget_value: float, default_value: float, decimals: int) -> bool:
    """True when a widget still holds its pre-populated default.

    Defaults are rounded before they reach the widget, so the comparison has to
    happen at the same precision — otherwise every untouched field claims to be
    a user override.
    """
    return abs(round(widget_value, decimals) - round(default_value, decimals)) < 10**-decimals / 2


def _seed_driver_value(
    state, widget_key: str, seed_key: str, basis: str, default_pct: float
) -> None:
    """Puts the basis's default into the value widget whenever the basis changes.

    The value is otherwise left alone, so a figure typed by hand survives every
    rerun that does not move the basis. Seeding only once — the old behaviour —
    left the first basis's value in place after the selector changed, and
    :func:`_untouched` then read the stale number as a user override of the new
    default. ``seed_key`` carries the ``dcf_input_`` prefix so a change of
    company clears it with everything else.
    """
    if widget_key not in state or state.get(seed_key) != basis:
        state[widget_key] = default_pct
        state[seed_key] = basis


def _driver_input(
    label: str,
    key: str,
    ratios: pd.DataFrame,
    driver_key: str,
    *,
    bases: tuple[str, ...] = fin.BASES,
    default_basis: str = "5-year average",
    trailing: fin.DriverDefault | None = None,
    trailing_is_default: bool = True,
    also: tuple[fin.DriverDefault, ...] = (),
) -> tuple[float | None, Sourced, tuple[str, ...], str]:
    """One driver: a basis selector, an editable value, and its provenance.

    ``trailing`` is the driver's leading precomputed basis — the trailing-twelve-
    month ratio, or for growth the lower of TTM and the five-year CAGR. When it
    can be computed and ``trailing_is_default``, it leads the basis list and is
    the default; otherwise it is offered after the history bases. ``also`` are
    further precomputed bases offered right after it; when the leading one is
    unavailable, the first of these that is available becomes the default. When
    a basis cannot be computed it is left out, and a caption says why the
    default fell back — never silently.

    Also returns the period labels the default was averaged over, so the
    validation layer can tell which fiscal years actually reach the model, and
    the basis in use.
    """
    precomputed = {d.basis: d for d in (trailing, *also) if d is not None and d.ok}
    offered_extra = tuple(d.basis for d in also if d.ok)
    if trailing is not None and trailing.ok:
        if trailing_is_default:
            bases = (trailing.basis,) + offered_extra + bases
            default_basis = trailing.basis
        else:
            bases = bases + (trailing.basis,) + offered_extra
    else:
        bases = offered_extra + bases
        if trailing_is_default and offered_extra:
            default_basis = offered_extra[0]
    basis = st.selectbox(
        f"{label} — basis",
        bases,
        index=bases.index(default_basis),
        key=f"dcf_input_{key}_basis",
    )
    if trailing is not None and not trailing.ok and trailing_is_default:
        st.caption(
            f"{trailing.basis} is unavailable — {trailing.detail} Defaulting to "
            f"{default_basis} instead."
        )
    if basis in precomputed:
        default = precomputed[basis]
    else:
        default = fin.driver_default(ratios, driver_key, basis)

    if not default.ok:
        common.provenance(
            Sourced.unavailable(source=f"Historical {label.lower()}", reason=default.detail)
        )
        st.warning(
            f"No usable history for {label.lower()} — {default.detail} There is nothing "
            "to pre-populate from, so this starts empty and must be entered by hand."
        )
        manual = st.number_input(
            f"{label} (%)", value=0.0, step=0.5, key=f"dcf_input_{key}_manual"
        )
        return (
            manual / 100.0,
            Sourced.user(manual / 100.0, "No historical basis available."),
            (),
            basis,
        )

    widget_key = f"dcf_input_{key}_value"
    default_pct = round(default.value * 100, 4)
    _seed_driver_value(
        st.session_state, widget_key, f"dcf_input_{key}_seeded_from", basis, default_pct
    )

    value = st.number_input(f"{label} (%)", step=0.25, key=widget_key, format="%.4f")
    derived = Sourced.derived(
        value=default.value,
        source=f"{basis} of reported figures",
        formula=f"{fin.DRIVER_LABELS[driver_key]} over {default.years_used} period(s): "
        f"{', '.join(str(p) for p in default.periods[-6:])}",
    )

    user_value = value / 100.0
    if _untouched(value, default_pct, 4):
        sourced = derived
    else:
        sourced = derived.with_user_value(user_value)
    common.provenance(sourced)
    if default.detail:
        st.caption(f"⚠️ {default.detail}")
    return user_value, sourced, default.periods, basis


class _Discount(NamedTuple):
    """The discount-rate inputs, before they are weighted into a WACC."""

    risk_free: float
    erp: float
    beta: float
    beta_outcome: dcf_model.BetaOutcome
    cost_of_debt: float
    marginal_tax: float
    #: The spread table's date when the synthetic rating set the cost of debt,
    #: ``None`` when a yield was typed or no rating could be formed — the
    #: staleness check only applies when the table is actually in use.
    spread_table_as_of: date | None


class _Drivers(NamedTuple):
    """The six driver values as entered, before the growth and margin paths are shaped."""

    years: int
    values: dict[str, float]
    periods_in_use: set[str]
    fade: bool
    hold: int
    #: Year-N targets by driver key — margin, capex %, D&A % — or ``None``
    #: where the path is flat.
    targets: dict[str, float | None]
    growth_basis: str
    #: The margins the user assumed rather than read off the TTM, by name.
    assumed_margins: dict[str, float]
    acquisitions_in_capex: bool
    #: Acquisition spend ÷ revenue added to every year's capex when folded in.
    acquisitions_share: float


class _Terminal(NamedTuple):
    terminal_growth: float
    exit_multiple: float
    headline_method: str
    mid_year: bool
    elapsed: float
    ronic: float | None
    terminal_tax_rate: float | None
    ebita_proxy: bool
    #: Beside the RONIC input, filled once the forecast exists — the explicit
    #: forecast's own incremental return, for comparison.
    incremental_return_slot: object


class _Bridge(NamedTuple):
    include_sti: bool
    cash_floor: float
    sbc: float | None
    non_operating: float


#: Sharadar publishes a single interest expense line rather than the gross/net
#: pair, so there is no basis to choose between.
_INTEREST_BASIS = "gross"


def collect(dataset: dcf_model.CompanyDataset, valuation_date: date) -> dict | None:
    """Renders every input and returns the assembled assumption set.

    One section per block of the panel, top to bottom, in the order they draw.
    They share only ``records``, the export's Assumptions sheet, which each one
    appends to. ``None`` means a blocking screen has been drawn in place of the
    rest of the page.

    Every key of the returned dict that the valuation depends on must also be in
    ``_assumption_fingerprint`` in the package ``__init__`` — a key missing
    there leaves the previous run's numbers on screen looking current.
    """
    st.divider()
    st.subheader("Assumptions")
    st.caption(
        "Every default is derived from reported history and shows where it came from. "
        "Nothing here is invented — where a basis does not exist, the field says so."
    )

    base = dcf_model.base_period(dataset)
    records: list[dcf_export.AssumptionRecord] = []

    entered = _forecast_drivers(dataset, base, records)
    driver_values, periods_in_use = entered.values, entered.periods_in_use

    st.markdown("#### Discount rate")
    discount = st.container(border=True)
    with discount:
        rates = _discount_inputs(dataset, base, records)

    counted = _share_count(dataset, records)
    if counted is None:
        return None
    shares, share_basis = counted

    wacc_result = _weighted_cost(dataset, base, rates, discount)
    if wacc_result is None:
        return None

    terminal = _terminal_and_conventions(
        dataset, base, valuation_date, wacc_result, rates.marginal_tax, records
    )
    bridge = _cash_sbc_and_bridge(dataset, base, records)

    # The growth path needs the terminal rate to fade toward, so it is shaped
    # only once both are known. The margin, capex and D&A move to their targets
    # where one is set — capex has one by default, its five-year average; the
    # tax rate and working capital stay flat.
    fade_to = terminal.terminal_growth if entered.fade else None

    def path(key: str) -> tuple[float, ...]:
        start = driver_values[key]
        target = entered.targets.get(key)
        return dcf.linear_path(start, start if target is None else target, entered.years)

    drivers = dcf.DriverPath(
        revenue_growth=dcf.growth_path(
            driver_values["revenue_growth"], entered.years, fade_to, hold=entered.hold
        ),
        operating_margin=path("operating_margin"),
        # Acquisition spend, when folded in, rides on top of every year's capex.
        capex_pct_revenue=tuple(c + entered.acquisitions_share for c in path("capex_pct_revenue")),
        dna_pct_revenue=path("dna_pct_revenue"),
        effective_tax_rate=(driver_values["effective_tax_rate"],) * entered.years,
        nwc_pct_revenue=(driver_values["nwc_pct_revenue"],) * entered.years,
    )
    _show_incremental_return(terminal, base, drivers, wacc_result)

    # --- Effective tax rate for FCFF -------------------------------------
    etr_series = fin.effective_tax_rate(dataset.annual).dropna()
    etr_window = etr_series.tail(5)
    etr = float(etr_window.median()) if not etr_series.empty else None
    periods_in_use.update(str(p) for p in etr_window.index)
    # The base period is what everything is projected from, so it is in use by
    # definition even when no driver window happens to reach it.
    if not dataset.annual.empty:
        periods_in_use.add(str(dataset.annual.index[-1]))
    delta_nwc, delta_nwc_span = dcf_model.base_period_delta_nwc(dataset, base)

    base_fcff = dcf_model.base_period_fcff(
        dataset,
        base,
        tax_rate=etr if etr is not None else driver_values["effective_tax_rate"],
        interest_basis=_INTEREST_BASIS,
        delta_nwc=delta_nwc,
    )
    _render_fcff_panel(dataset, base, base_fcff, etr, etr_series, delta_nwc_span)

    return {
        "base": base,
        "drivers": drivers,
        "wacc": wacc_result,
        "terminal_growth": terminal.terminal_growth,
        "exit_multiple": terminal.exit_multiple,
        "elapsed": terminal.elapsed,
        "mid_year": terminal.mid_year,
        "ronic": terminal.ronic,
        "terminal_tax_rate": terminal.terminal_tax_rate,
        "ebita_proxy": terminal.ebita_proxy,
        "spread_table_as_of": rates.spread_table_as_of,
        "growth_fade_to": fade_to,
        "growth_hold": entered.hold,
        "growth_basis": entered.growth_basis,
        "assumed_margins": entered.assumed_margins,
        "acquisitions_in_capex": entered.acquisitions_in_capex,
        "shares": shares,
        "share_basis": share_basis,
        "include_sti": bridge.include_sti,
        "cash_floor": bridge.cash_floor,
        "non_operating": bridge.non_operating or None,
        "base_fcff": base_fcff,
        "headline_method": terminal.headline_method,
        "sbc": bridge.sbc,
        "etr": etr,
        "erp": rates.erp,
        "beta": rates.beta,
        "risk_free": rates.risk_free,
        "beta_outcome": rates.beta_outcome,
        "records": records,
        "periods_in_use": frozenset(periods_in_use),
        "tolerances": val.DEFAULT_TOLERANCES,
    }


def _forecast_drivers(
    dataset: dcf_model.CompanyDataset,
    base: dcf_model.BasePeriod,
    records: list[dcf_export.AssumptionRecord],
) -> _Drivers:
    """The six drivers, the shape of the growth and margin paths, and the periods used."""
    st.markdown("#### Forecast drivers")
    with st.container(border=True):
        st.caption(
            f"Base period: **{base.label}** — {base.detail} "
            f"Balance sheet as of **{base.stocks_as_of}**."
        )
        years = st.slider(
            "Forecast years", 5, 15, 10, key="dcf_input_years",
            help="Ten years by default, so a company still growing well above the "
                 "terminal rate has room to compound before it fades. The excess return "
                 "lives in the explicit forecast; the terminal year earns only its RONIC.",
        )

        driver_values: dict[str, float] = {}
        # Every fiscal year any driver averaged over. Reconciliation failures
        # outside this set cannot reach the valuation, and the validation layer
        # scopes their severity accordingly.
        periods_in_use: set[str] = set()
        ttm_growth = fin.ttm_revenue_growth(dataset.ttm_history, dataset.ttm_history_meta)
        lower_growth = fin.lower_of_growth(
            ttm_growth, fin.driver_default(dataset.ratios, "revenue_growth", "5-year CAGR")
        )
        trailing = {
            # Year-1 growth defaults to the lower of the trailing rate and the
            # five-year CAGR; the trailing rate on its own stays selectable.
            "revenue_growth": lower_growth,
            **{
                key: fin.base_period_driver(
                    key, base.flows, base.stocks, is_ttm=base.is_ttm, label=base.label
                )
                for key in (
                    "operating_margin", "capex_pct_revenue", "dna_pct_revenue", "nwc_pct_revenue"
                )
            },
        }
        growth_basis = ""
        assumed_margins: dict[str, float] = {}
        grid = st.columns(3)
        specs = [
            ("Revenue growth", "growth", "revenue_growth"),
            ("Operating margin", "margin", "operating_margin"),
            ("Effective tax rate", "etr", "effective_tax_rate"),
            ("CapEx % of revenue", "capex", "capex_pct_revenue"),
            ("D&A % of revenue", "dna", "dna_pct_revenue"),
            ("NWC % of revenue", "nwc", "nwc_pct_revenue"),
        ]
        starting = {
            "revenue_growth": "Year-1 revenue growth",
            "operating_margin": "Starting operating margin",
            "capex_pct_revenue": "Starting CapEx % of revenue",
            "dna_pct_revenue": "Starting D&A % of revenue",
        }
        for index, (label, key, driver_key) in enumerate(specs):
            with grid[index % 3]:
                growth = driver_key == "revenue_growth"
                value, sourced, periods, basis = _driver_input(
                    starting.get(driver_key, label),
                    key,
                    dataset.ratios,
                    driver_key,
                    bases=fin.GROWTH_BASES if growth else fin.BASES,
                    default_basis="5-year CAGR" if growth else "5-year average",
                    trailing=trailing.get(driver_key),
                    # Working capital is a level that swings with timing, so a
                    # single balance sheet is offered but the average leads.
                    trailing_is_default=driver_key != "nwc_pct_revenue",
                    also=(ttm_growth,) if growth else (),
                )
                if growth:
                    growth_basis = _effective_growth_basis(basis, lower_growth, ttm_growth)
                if driver_key == "operating_margin" and value is not None and (
                    basis != fin.TTM_BASIS or sourced.origin is Origin.USER
                ):
                    assumed_margins["Starting margin"] = value
                if driver_key == "capex_pct_revenue":
                    st.caption(
                        "Cash capex only. Assets obtained under finance leases — material "
                        "at some hyperscalers — are excluded, and their principal "
                        "repayments sit in financing cash flow, which FCFF does not see."
                    )
                if driver_key == "nwc_pct_revenue":
                    st.caption(
                        "Working capital is held at this share of revenue, so it moves "
                        "with each year's change in revenue. A negative share — customers "
                        "funding the business — releases cash as revenue grows."
                    )
                driver_values[driver_key] = value if value is not None else 0.0
                periods_in_use.update(periods)
                records.append(
                    dcf_export.AssumptionRecord("Forecast drivers", label, value, sourced)
                )

        path_left, path_right = st.columns(2)
        with path_left:
            fade = st.toggle(
                "Fade revenue growth toward terminal growth",
                value=True,
                key="dcf_input_fade",
                help="Year-1 growth is held for the years below, then steps linearly toward "
                     "the perpetuity rate, landing on it in the terminal year — no cliff "
                     "between year N and the terminal value. Off holds the rate flat for "
                     "every forecast year.",
            )
            hold = st.number_input(
                "Years to hold year-1 growth",
                min_value=1,
                max_value=years - 1,
                value=min(3, years - 1),
                step=1,
                key="dcf_input_hold",
                disabled=not fade,
            )
            hold = int(min(hold, years - 1))
        records.append(
            dcf_export.AssumptionRecord(
                "Forecast drivers",
                "Growth path",
                f"held {hold} years, then faded to terminal growth" if fade else "flat",
                Sourced.user(float(hold if fade else 0), "Shape of the revenue growth path."),
            )
        )

        with path_right:
            targets = {
                "operating_margin": _driver_target(
                    "operating margin", "margin", driver_values["operating_margin"], years,
                    records,
                ),
                "capex_pct_revenue": _driver_target(
                    "capex % of revenue", "capex", driver_values["capex_pct_revenue"], years,
                    records,
                    normal=fin.normal_capex(dataset.ratios),
                    caption="D&A stays at its starting ratio, as the margin does: the "
                            "margin already deducts today's D&A, so move the two together "
                            "or cash flow moves by the whole change in D&A rather than by "
                            "its tax shield.",
                ),
                "dna_pct_revenue": _driver_target(
                    "D&A % of revenue", "dna", driver_values["dna_pct_revenue"], years, records,
                ),
            }
        target_margin = targets["operating_margin"]
        if target_margin is not None:
            assumed_margins["Target margin"] = target_margin

        acquisitions_in_capex, acquisitions_share = _acquisitions_toggle(dataset, records)

        if dataset.capex_result.origin == "estimate":
            st.warning(
                f"**CapEx is an estimate.** {dataset.capex_result.detail} "
                f"Formula in use: `{dataset.capex_result.formula}`"
            )

    return _Drivers(
        years, driver_values, periods_in_use, fade, hold, targets, growth_basis,
        assumed_margins, acquisitions_in_capex, acquisitions_share,
    )


#: Session keys for each target toggle. The margin's predate the other two and
#: keep their names so an open session is not reset.
_TARGET_KEYS = {
    "margin": ("dcf_input_target_margin_on", "dcf_input_target_margin"),
    "capex": ("dcf_input_target_capex_on", "dcf_input_target_capex"),
    "dna": ("dcf_input_target_dna_on", "dcf_input_target_dna"),
}


def _effective_growth_basis(
    basis: str, lower: fin.DriverDefault, trailing: fin.DriverDefault
) -> str:
    """The basis year-1 growth actually came from.

    "Lower of TTM and 5-year CAGR" resolves to one of the two, and the checks
    need to know which: the TTM-window acquisitions warning and the
    growth-acceleration wording apply only when the trailing rate is in use.
    """
    if basis != fin.LOWER_GROWTH_BASIS or not lower.ok:
        return basis
    if trailing.ok and lower.value == trailing.value:
        return fin.TTM_GROWTH_BASIS
    return "5-year CAGR"


def _driver_target(
    label: str,
    key: str,
    start: float,
    years: int,
    records: list[dcf_export.AssumptionRecord],
    *,
    normal: fin.DriverDefault | None = None,
    caption: str = "",
) -> float | None:
    """An optional year-N target the driver moves to, linearly from year 1.

    ``normal`` is the level history says the driver returns to — only capex
    has one (:func:`fin.normal_capex`). Where it can be computed the target is
    on by default and starts there, derived rather than typed; where it cannot,
    the toggle starts off and says why. The margin and D&A have none: whether a
    margin expands is a claim about the business, not something history
    supplies, so their paths stay flat unless someone chooses a target.
    Returns ``None`` when no target is set.
    """
    toggle_key, value_key = _TARGET_KEYS[key]
    normalises = normal is not None and normal.ok
    use_target = st.toggle(
        f"Move the {label} to a target",
        value=normalises,
        key=toggle_key,
        help=f"The {label} moves linearly from its starting value in year 1 to the "
             "target in year N; the terminal year keeps the final year's value.",
    )
    if normalises:
        st.caption(
            f"On by default: the {label} reaches its own {normal.basis.lower()} by year "
            f"{years}, so neither today's build-out nor a trough is held forever."
        )
    elif normal is not None:
        st.caption(
            f"No {normal.basis.lower()} to normalise to — {normal.detail} The starting "
            "value is held unless you set a target."
        )
    if caption:
        st.caption(caption)
    if not use_target:
        records.append(
            dcf_export.AssumptionRecord(
                "Forecast drivers", f"Target {label}", "none — flat",
                Sourced.user(start, "No target: the starting value is held."),
            )
        )
        return None
    seed = normal.value if normalises else start
    if value_key not in st.session_state:
        st.session_state[value_key] = round(seed * 100, 4)
    target_pct = st.number_input(
        f"Target {label} in year {years} (%)",
        step=0.5,
        key=value_key,
        format="%.4f",
    )
    target = target_pct / 100.0
    reached = f"Reached in year {years}, linearly from year 1."
    if normalises:
        sourced = Sourced.derived(
            value=normal.value,
            source=f"{normal.basis} of reported figures",
            formula=f"{fin.DRIVER_LABELS[normal.key]} over {normal.years_used} period(s): "
            f"{', '.join(str(p) for p in normal.periods)}. {reached}",
        )
        if not _untouched(target_pct, round(normal.value * 100, 4), 4):
            sourced = sourced.with_user_value(target)
        common.provenance(sourced)
        if normal.detail:
            st.caption(f"⚠️ {normal.detail}")
    else:
        sourced = Sourced.user(target, reached)
    records.append(
        dcf_export.AssumptionRecord("Forecast drivers", f"Target {label}", target, sourced)
    )
    return target


def _acquisitions_toggle(
    dataset: dcf_model.CompanyDataset,
    records: list[dcf_export.AssumptionRecord],
) -> tuple[bool, float]:
    """Folds acquisition spend into reinvestment, for a company that buys growth.

    Reported revenue growth includes what acquisitions bought; capex excludes
    what they cost. Adding the historical acquisition rate to every year's
    capex makes the forecast pay for the growth it projects. Returns whether it
    is on and the share of revenue added.
    """
    spend = fin.driver_default(dataset.ratios, fin.ACQUISITIONS_KEY, "5-year average")
    if not spend.ok or spend.value <= 0:
        return False, 0.0
    include = st.toggle(
        "Include acquisitions in reinvestment",
        value=False,
        key="dcf_input_acquisitions",
        help="Adds the 5-year average of acquisition spend ÷ revenue to the capex "
             "driver in every forecast year.",
    )
    st.caption(
        f"Acquisitions averaged **{spend.value:.2%}** of revenue over "
        f"{spend.years_used} year(s)"
        + (" — added to capex." if include else ", not in capex.")
    )
    records.append(
        dcf_export.AssumptionRecord(
            "Forecast drivers",
            "Acquisitions added to capex",
            spend.value if include else 0.0,
            Sourced.derived(spend.value, "5-year average of reported history",
                            "Business acquisitions ÷ revenue"),
        )
    )
    return include, spend.value if include else 0.0


def _discount_inputs(
    dataset: dcf_model.CompanyDataset,
    base: dcf_model.BasePeriod,
    records: list[dcf_export.AssumptionRecord],
) -> _Discount:
    """The three columns of the discount-rate panel, and the cost-of-debt guidance.

    Drawn inside the panel's container; :func:`_weighted_cost` adds the WACC
    beneath them once the share count is known.
    """
    left, middle, right = st.columns(3)

    with left:
        st.markdown("**Risk-free rate**")
        treasury = market_data.fetch_10y_treasury_rate()
        if treasury.ok:
            default_rf = treasury.value * 100
        else:
            default_rf = 0.0
            st.warning(
                f"The 10-year Treasury rate could not be fetched — {treasury.detail}. "
                "Enter it manually; nothing has been substituted."
            )
        rf_pct = st.number_input(
            "Risk-free rate (%)",
            value=float(default_rf),
            step=0.05,
            key="dcf_input_rf",
            help="Should match the tenor of the bond behind your cost of debt.",
        )
        risk_free = rf_pct / 100.0
        rf_sourced = (
            treasury
            if treasury.ok and _untouched(rf_pct, default_rf, 4)
            else Sourced.user(risk_free, "Manually entered risk-free rate.")
        )
        common.provenance(rf_sourced)
        records.append(
            dcf_export.AssumptionRecord("Discount rate", "Risk-free rate", risk_free, rf_sourced)
        )

        st.markdown("**Equity risk premium**")
        erp_sourced_default = market_data.fetch_erp()
        if erp_sourced_default.ok:
            default_erp = erp_sourced_default.value * 100
        else:
            default_erp = 0.0
            st.warning(
                f"The equity risk premium scrape failed — {erp_sourced_default.detail}. "
                "Enter it manually; no default has been substituted."
            )
        erp_pct = st.number_input(
            "Equity risk premium (%)", value=float(default_erp), step=0.05, key="dcf_input_erp"
        )
        erp = erp_pct / 100.0
        erp_sourced = (
            erp_sourced_default
            if erp_sourced_default.ok and _untouched(erp_pct, default_erp, 4)
            else Sourced.user(erp, "Manually entered equity risk premium.")
        )
        common.provenance(erp_sourced)
        records.append(
            dcf_export.AssumptionRecord("Discount rate", "Equity risk premium", erp, erp_sourced)
        )

    with middle:
        st.markdown("**Beta**")
        frequency = st.radio(
            "Regression window",
            ["monthly", "weekly"],
            horizontal=True,
            key="dcf_input_beta_freq",
            help="5 years of monthly returns by default; 2 years of weekly as the alternative.",
        )
        blume = st.checkbox(
            "Blume adjustment (0.67 × β + 0.33)", value=True, key="dcf_input_blume",
            help="A single stock's regressed beta carries a standard error of 0.2–0.3 "
                 "over five years; pulling it toward the market's 1.0 is the usual "
                 "correction for that noise.",
        )
        beta_outcome = dcf_model.regress_company_beta(
            dataset,
            frequency=frequency,
            blume=blume,
            years=5 if frequency == "monthly" else 2,
        )

        if beta_outcome.result is not None:
            regressed = beta_outcome.result
            beta_default = regressed.beta
            beta_sourced_default = Sourced.derived(
                value=beta_default,
                source=beta_outcome.source,
                formula=regressed.formula,
            )
        else:
            regressed = None
            beta_default = 1.0
            beta_sourced_default = Sourced.unavailable(
                source=beta_outcome.source, reason=beta_outcome.error or "unavailable"
            )
            st.warning(
                f"Beta could not be regressed — {beta_outcome.error} Enter one manually; "
                "the field below starts at the market beta of 1.0, which is a placeholder, "
                "not an estimate for this company."
            )

        if "dcf_input_beta" not in st.session_state:
            st.session_state["dcf_input_beta"] = round(float(beta_default), 4)

        beta_value = st.number_input(
            "Beta (editable)", step=0.05, key="dcf_input_beta", format="%.4f"
        )
        if regressed is not None:
            col_reset, col_stats = st.columns([1, 2])
            with col_reset:
                if st.button("Reset to regressed", key="dcf_beta_reset", width="stretch"):
                    st.session_state["dcf_input_beta"] = round(float(regressed.beta), 4)
                    st.rerun()
            with col_stats:
                st.metric("Regressed β", f"{regressed.beta:.4f}", f"R² {regressed.r_squared:.3f}")
            interval = regressed.confidence_interval
            st.caption(
                f"Raw β {regressed.raw_beta:.4f} · {regressed.observations} {frequency} "
                f"observations vs {regressed.index_ticker}"
                + (
                    f" · standard error {regressed.standard_error:.3f}, 95% interval "
                    f"{interval[0]:.2f}–{interval[1]:.2f}"
                    if interval is not None
                    else ""
                )
            )
            if beta_outcome.price_source_note:
                st.caption(f"⚠️ {beta_outcome.price_source_note}")

        beta_sourced = (
            beta_sourced_default
            if regressed is not None and _untouched(beta_value, regressed.beta, 4)
            else Sourced.user(
                beta_value,
                f"Overrides the regressed β of {regressed.beta:.4f}"
                if regressed
                else "No regression was available.",
            )
        )
        common.provenance(beta_sourced)
        records.append(
            dcf_export.AssumptionRecord("Discount rate", "Beta", beta_value, beta_sourced)
        )

    with right:
        st.markdown("**Cost of debt**")
        synthetic, cause = _synthetic_cost_of_debt(dataset, base, risk_free)
        if synthetic is not None:
            default_kd = synthetic.cost_of_debt * 100
        else:
            default_kd = risk_free * 100
            lookback = wacc_lib.DISCLOSED_INTEREST_LOOKBACK_YEARS
            reason = {
                "undisclosed": "interest expense reads 0 beside material debt, which is how "
                "Sharadar records interest a filing does not break out — it looks "
                f"undisclosed, and none of the last {lookback} fiscal years disclosed it either",
                "missing": "the base period reports no interest expense line, and none of "
                f"the last {lookback} fiscal years disclosed one either",
                "negative": "the base period's interest line is negative — net interest "
                "income, a net finance result rather than the cost of debt — so there is no "
                "coverage ratio to rate",
            }.get(cause, "EBIT or interest expense is not reported for the base period")
            st.warning(
                f"No synthetic rating could be formed — {reason}. The field starts at the "
                "risk-free rate, which is a placeholder rather than an estimate; enter a "
                "yield to maturity."
            )
        # Unrounded, so the risk-free placeholder is exactly the risk-free rate —
        # rounding it to the display precision could land a hair below and trip
        # the "cost of debt below risk-free" check on a figure nobody typed.
        if "dcf_input_kd" not in st.session_state:
            st.session_state["dcf_input_kd"] = default_kd
        cost_of_debt_pct = st.number_input(
            "Cost of debt — yield to maturity (%)",
            step=0.05,
            key="dcf_input_kd",
            format="%.4f",
            help="Starts from a synthetic rating; a traded bond's yield is better. See the "
                 "sourcing guidance below.",
        )
        cost_of_debt = cost_of_debt_pct / 100.0
        from_table = synthetic is not None and _untouched(cost_of_debt_pct, default_kd, 4)
        if from_table:
            kd_sourced = Sourced.estimate(
                cost_of_debt,
                f"Synthetic rating from interest coverage — Damodaran spreads as of "
                f"{wacc_lib.SYNTHETIC_RATINGS_AS_OF}",
                (
                    f"Base-period interest is {'not reported' if cause == 'missing' else 'undisclosed'}, "
                    f"so coverage is taken from {synthetic.coverage_period}, the latest year "
                    f"that disclosed it, and its spread applied to today's risk-free rate. "
                    if synthetic.coverage_period
                    else ""
                )
                + "The spread table drifts year to year and is an estimate; a traded "
                "bond's yield to maturity is better evidence.",
                formula=synthetic.formula,
            )
        else:
            kd_sourced = Sourced.user(
                cost_of_debt, "Manual yield-to-maturity input — Sharadar carries no bond data."
            )
        common.provenance(kd_sourced)
        records.append(
            dcf_export.AssumptionRecord(
                "Discount rate", "Cost of debt (YTM)", cost_of_debt, kd_sourced
            )
        )

        interest_series = fin.interest_expense(dataset.annual)
        debt_series = fin.total_debt(dataset.annual)
        implied = None
        if len(interest_series) and len(debt_series):
            implied = wacc_lib.implied_cost_of_debt(
                float(interest_series.iloc[-1])
                if not pd.isna(interest_series.iloc[-1])
                else None,
                float(debt_series.iloc[-1]) if not pd.isna(debt_series.iloc[-1]) else None,
                float(debt_series.iloc[-2])
                if len(debt_series) > 1 and not pd.isna(debt_series.iloc[-2])
                else None,
            )
        if implied is not None:
            st.metric("Implied average cost of debt", f"{implied:.2%}")
            st.caption(
                "Interest expense ÷ average total debt. This is a **backward-looking "
                "average coupon**, not a current marginal cost — it understates when "
                "rates have risen and overstates when they have fallen. Cross-check only; "
                "it never populates the model."
            )
        else:
            st.info(
                "The implied cost of debt cross-check is unavailable: Sharadar reports no "
                "interest expense for this company."
            )

        marginal_tax_pct = st.number_input(
            "Marginal tax rate (%)",
            value=DEFAULT_MARGINAL_TAX * 100,
            step=0.5,
            key="dcf_input_marginal_tax",
            help=(
                "The interest tax shield is worth the *marginal* rate — what the next "
                "dollar of interest saves — not the historical effective rate that feeds "
                "FCFF. They are separate inputs on purpose. The default is the 21% US "
                "federal rate plus a typical state rate."
            ),
        )
        marginal_tax = marginal_tax_pct / 100.0
        if dataset.is_converted or dataset.foreign_filer:
            st.caption(
                "⚠️ 25% is a US federal-plus-state rate. For a company taxed at home "
                "outside the US, set this to its home country's statutory rate."
            )
        marginal_sourced = (
            Sourced.estimate(
                marginal_tax,
                "Convention: US federal 21% plus state",
                "A combined US statutory rate; a company taxed mostly abroad has a "
                "different one.",
            )
            if _untouched(marginal_tax_pct, DEFAULT_MARGINAL_TAX * 100, 4)
            else Sourced.user(marginal_tax, "Statutory marginal rate for the tax shield.")
        )
        common.provenance(marginal_sourced)
        records.append(
            dcf_export.AssumptionRecord(
                "Discount rate", "Marginal tax rate", marginal_tax, marginal_sourced
            )
        )

    with st.expander("Where to find a cost of debt", expanded=False):
        st.markdown(COST_OF_DEBT_GUIDANCE)
        st.link_button(
            "Open FINRA's fixed income data", FINRA_FIXED_INCOME_URL, width="stretch"
        )

    return _Discount(
        risk_free, erp, beta_value, beta_outcome, cost_of_debt, marginal_tax,
        wacc_lib.SYNTHETIC_RATINGS_AS_OF_DATE if from_table else None,
    )


def _synthetic_cost_of_debt(
    dataset: dcf_model.CompanyDataset, base: dcf_model.BasePeriod, risk_free: float
) -> tuple[wacc_lib.SyntheticCostOfDebt | None, str]:
    """The synthetic-rating cost of debt — see :func:`wacc.resolve_synthetic_cost_of_debt`.

    Coverage is a ratio, so the reporting currency does not matter.
    """
    flows = base.flows
    ebit = flows.get("ebit")
    interest = fin.interest_expense(pd.DataFrame([flows]))
    interest_value = (
        float(interest.iloc[0]) if not interest.empty and not pd.isna(interest.iloc[0]) else None
    )
    debt = fin.total_debt(pd.DataFrame([base.stocks]))
    debt_value = float(debt.iloc[0]) if not debt.empty and not pd.isna(debt.iloc[0]) else None
    annual = dataset.annual
    history: list[tuple[str, float | None, float | None]] = []
    if not annual.empty and "ebit" in annual.columns:
        annual_interest = fin.interest_expense(annual)
        history = [
            (
                f"FY{label}",
                None if pd.isna(annual.loc[label, "ebit"]) else float(annual.loc[label, "ebit"]),
                None if pd.isna(annual_interest.loc[label]) else float(annual_interest.loc[label]),
            )
            for label in annual.index
        ]
    return wacc_lib.resolve_synthetic_cost_of_debt(
        float(ebit) if ebit is not None and not pd.isna(ebit) else None,
        interest_value,
        risk_free,
        debt_value,
        history,
    )


def _share_count(
    dataset: dcf_model.CompanyDataset, records: list[dcf_export.AssumptionRecord]
) -> tuple[float, str] | None:
    """The share count value per share divides by, and a label saying which one.

    The default is today's count plus the dilution the diluted count carries
    over the basic one. The weighted-average count lags a company buying back
    stock by the buyback, and understates value per share by as much; the bare
    point-in-time count ignores options and RSUs altogether.
    """
    share_basis_choice = st.radio(
        "Share count basis for value per share",
        _SHARE_BASES,
        horizontal=True,
        key="dcf_input_share_basis",
        help=(
            "Point-in-time + dilution: the latest cover-page count plus weighted diluted "
            "less weighted basic shares, a proxy for in-the-money options and RSUs. "
            "Sharadar has no treasury-stock-method table, so the proxy is the best "
            "dilution figure available."
        ),
    )
    point_in_time = dataset.shares_point_in_time
    dilution = None
    if dataset.shares_weighted_diluted and dataset.shares_weighted_basic:
        dilution = max(dataset.shares_weighted_diluted - dataset.shares_weighted_basic, 0.0)

    if share_basis_choice == _SHARE_BASES[0] and point_in_time and dilution is not None:
        shares = point_in_time + dilution
        share_basis = (
            f"Point-in-time outstanding (as of {dataset.shares_point_in_time_date}) + "
            f"{dilution:,.0f} dilution ({dataset.shares_weighted_label} diluted − basic)"
        )
    elif share_basis_choice == _SHARE_BASES[1] and dataset.shares_weighted_diluted:
        shares = dataset.shares_weighted_diluted
        share_basis = f"Diluted weighted-average ({dataset.shares_weighted_label})"
    elif point_in_time:
        shares = point_in_time
        share_basis = f"Point-in-time outstanding (as of {dataset.shares_point_in_time_date})"
        if share_basis_choice == _SHARE_BASES[0]:
            st.caption(
                "No weighted basic and diluted pair is reported, so no dilution could be "
                "added; this is the bare point-in-time count."
            )
    elif dataset.shares_weighted_diluted:
        shares = dataset.shares_weighted_diluted
        share_basis = f"Diluted weighted-average ({dataset.shares_weighted_label})"
    else:
        common.render_blocking(
            "No share count available",
            f"Sharadar returned no share count for {dataset.ticker}, so equity value cannot "
            "be divided into a per-share figure.",
        )
        return None

    st.caption(
        f"Using **{shares:,.0f}** shares — {share_basis}. "
        f"Sharadar provides no treasury-stock-method calculation for outstanding options and "
        f"RSUs, and no if-converted treatment of convertibles, so no basis is fully loaded."
        + (
            f" Counts are per ADR: Sharadar's share factor of {dataset.share_factor:g} is "
            "applied to every one."
            if dataset.share_factor != 1.0
            else ""
        )
    )
    records.append(
        dcf_export.AssumptionRecord(
            "Capital structure",
            "Shares used",
            shares,
            Sourced.reported(shares, share_basis, dataset.shares_point_in_time_date),
        )
    )
    return shares, share_basis


def _weighted_cost(
    dataset: dcf_model.CompanyDataset,
    base: dcf_model.BasePeriod,
    rates: _Discount,
    discount,
) -> wacc_lib.WaccResult | None:
    """Weights the discount inputs into a WACC and shows it in the discount panel."""
    total_debt_series = fin.total_debt(pd.DataFrame([base.stocks]))
    debt_value = (
        float(total_debt_series.iloc[0])
        if not total_debt_series.empty and not pd.isna(total_debt_series.iloc[0])
        else 0.0
    )

    equity_value = (dataset.last_price or 0.0) * (dataset.shares_point_in_time or 0.0)
    try:
        wacc_result = wacc_lib.compute_wacc(
            equity_value=equity_value,
            debt_value=debt_value,
            risk_free_rate=rates.risk_free,
            beta=rates.beta,
            equity_risk_premium=rates.erp,
            cost_of_debt=rates.cost_of_debt,
            marginal_tax_rate=rates.marginal_tax,
        )
    except ValueError as error:
        common.render_blocking("The WACC cannot be weighted", str(error))
        return None

    with discount:
        st.markdown("**Weighted average cost of capital**")
        wacc_columns = st.columns(5)
        wacc_columns[0].metric("WACC", f"{wacc_result.wacc:.2%}")
        wacc_columns[1].metric("Cost of equity", f"{wacc_result.cost_of_equity:.2%}")
        wacc_columns[2].metric("After-tax cost of debt", f"{wacc_result.after_tax_cost_of_debt:.2%}")
        wacc_columns[3].metric("Equity weight", f"{wacc_result.equity_weight:.1%}")
        wacc_columns[4].metric("Debt weight", f"{wacc_result.debt_weight:.1%}")
        st.caption(
            f"`{wacc_result.formula}` · E = last close × point-in-time shares "
            f"({common.money(equity_value, dataset.valuation_currency)}) · D = book value of total debt "
            f"({common.money(debt_value, dataset.valuation_currency)}), as of {base.stocks_as_of}."
        )
    return wacc_result


def _terminal_and_conventions(
    dataset: dcf_model.CompanyDataset,
    base: dcf_model.BasePeriod,
    valuation_date: date,
    wacc_result: wacc_lib.WaccResult,
    marginal_tax: float,
    records: list[dcf_export.AssumptionRecord],
) -> _Terminal:
    """Both terminal value inputs, the terminal year, the headline choice, and timing."""
    st.markdown("#### Terminal value and conventions")
    with st.container(border=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            terminal_growth_pct = st.number_input(
                "Perpetuity growth rate (%)",
                value=DEFAULT_TERMINAL_GROWTH * 100,
                step=0.1,
                key="dcf_input_g",
            )
            terminal_growth = terminal_growth_pct / 100.0
            growth_sourced = (
                Sourced.estimate(
                    terminal_growth,
                    "Convention: long-run inflation plus modest real growth",
                    "A default, not a measurement. It should sit at or below the "
                    "risk-free rate, the market's own long-run nominal growth.",
                )
                if _untouched(terminal_growth_pct, DEFAULT_TERMINAL_GROWTH * 100, 4)
                else Sourced.user(terminal_growth, "Gordon growth rate.")
            )
            common.provenance(growth_sourced)
            records.append(
                dcf_export.AssumptionRecord(
                    "Terminal value", "Perpetuity growth", terminal_growth, growth_sourced
                )
            )

            ronic, ronic_sourced = _return_on_new_capital(dataset, wacc_result)
            common.provenance(ronic_sourced)
            incremental_return_slot = st.empty()
            records.append(
                dcf_export.AssumptionRecord(
                    "Terminal value", "Return on new invested capital",
                    wacc_result.wacc if ronic is None else ronic, ronic_sourced,
                )
            )

            terminal_at_marginal = st.toggle(
                "Terminal tax at the marginal rate",
                value=True,
                key="dcf_input_terminal_tax",
                help="Low effective rates from credits, deferrals or a foreign mix rarely "
                     "last forever. On: the terminal year pays the marginal rate. Off: it "
                     "keeps the forecast's effective rate in perpetuity.",
            )
            terminal_tax_rate = marginal_tax if terminal_at_marginal else None

            ebita_proxy = st.toggle(
                "Terminal NOPAT before amortisation (EBITA proxy)",
                value=True,
                key="dcf_input_ebita_proxy",
                help="Adds back max(D&A − capex, 0) from the final forecast year, grown at "
                     "g, before tax in year N + 1 — amortisation of acquired intangibles "
                     "that capex never replaces and that runs off. Off: the terminal year "
                     "starts from EBIT, which keeps the charge but loses the add-back the "
                     "forecast years had.",
            )
            st.caption(
                "Moves the Gordon value, and the exit row's implied RONIC and "
                "zero-reinvestment growth, which read the same NOPAT; never the exit value."
            )
            records.append(
                dcf_export.AssumptionRecord(
                    "Terminal value", "EBITA proxy", "on" if ebita_proxy else "off",
                    Sourced.user(float(ebita_proxy), "Amortisation add-back in year N + 1."),
                )
            )
            records.append(
                dcf_export.AssumptionRecord(
                    "Terminal value", "Terminal tax rate",
                    terminal_tax_rate if terminal_tax_rate is not None else "forecast ETR",
                    Sourced.user(marginal_tax, "Marginal rate from the discount panel.")
                    if terminal_at_marginal
                    else Sourced.user(0.0, "Final forecast year's effective rate."),
                )
            )
        with col_b:
            ev_ebitda = dcf_model.historical_ev_ebitda(dataset)
            if ev_ebitda.ok:
                multiple_default = round(float(ev_ebitda.value), 2)
            else:
                multiple_default = 10.0
                st.warning(
                    f"No historical EV/EBITDA range available — {ev_ebitda.detail} "
                    "The field starts at a generic 10×, which is a placeholder rather than "
                    "this company's own trading range."
                )
            if "dcf_input_exit" not in st.session_state:
                st.session_state["dcf_input_exit"] = float(multiple_default)
            exit_multiple = st.number_input(
                "Exit EV/EBITDA multiple (×)", step=0.25, key="dcf_input_exit"
            )
            exit_sourced = (
                ev_ebitda
                if ev_ebitda.ok and _untouched(exit_multiple, multiple_default, 2)
                else Sourced.user(exit_multiple, "Manually set exit multiple.")
            )
            common.provenance(exit_sourced)
            st.caption(
                "The default is the company's *own* recent multiple, earned while it grew "
                "at its historical rate. By year N the forecast has slowed it toward "
                "terminal growth, so a mature multiple — a peer or sector median — is "
                "often the more consistent choice."
            )
            records.append(
                dcf_export.AssumptionRecord(
                    "Terminal value", "Exit EV/EBITDA multiple", exit_multiple, exit_sourced
                )
            )
        with col_c:
            headline_label = st.radio(
                "Headline method",
                list(_HEADLINE_METHODS),
                key="dcf_input_headline",
                help="Drives the headline value per share and the equity bridge. Both "
                     "methods always compute and both are shown side by side.",
            )
            headline_method = _HEADLINE_METHODS[headline_label]
            mid_year = st.toggle(
                "Mid-year discounting", value=True, key="dcf_input_midyear",
                help="Mid-year: each year's cash flow is discounted from the middle of the "
                     "year. Applies to the Gordon value, a perpetuity of such flows, but "
                     "not to an exit multiple, which is a sale at the end of year N.",
            )

        elapsed = 0.0
        if base.flows_as_of is None:
            st.warning(
                "The base period's end date was not reported, so the forecast is discounted "
                "as if it ended today."
            )
            elapsed_detail = "no base-period end date available"
        else:
            elapsed = dcf.elapsed_years(base.flows_as_of, valuation_date)
            elapsed_detail = (
                f"{(valuation_date - base.flows_as_of).days} days from {base.flows_as_of} "
                f"to {valuation_date} ÷ 365"
            )
            st.caption(
                f"**Timing:** year 1 is the twelve months after the base period ends on "
                f"**{base.flows_as_of}**, the same date as the balance sheet the bridge "
                f"reads. The value is struck there and rolled forward **{elapsed:.4f}** "
                f"years to {valuation_date} at the WACC, so it moves smoothly with the date "
                f"and changes step only when a new quarter is filed."
            )
        records.append(
            dcf_export.AssumptionRecord(
                "Conventions",
                "Years from base period end to valuation date",
                elapsed,
                Sourced.derived(elapsed, "Base period end date", elapsed_detail),
            )
        )

    return _Terminal(
        terminal_growth, exit_multiple, headline_method, mid_year, elapsed, ronic,
        terminal_tax_rate, ebita_proxy, incremental_return_slot,
    )


#: Where the terminal year's return on new capital comes from. The company's
#: own ROIC leads, and is the default, wherever its history supplies one.
_OWN_ROIC = "Company's own ROIC"
_RONIC_AT_WACC = "Equal to the WACC"
_RONIC_MANUAL = "Manual"


def _return_on_new_capital(
    dataset: dcf_model.CompanyDataset, wacc_result: wacc_lib.WaccResult
) -> tuple[float | None, Sourced]:
    """The terminal year's RONIC and its provenance; ``None`` means the WACC.

    ``None`` rather than the WACC's value, so the perpetuity keeps tracking the
    WACC wherever it moves — the sensitivity grid reruns it at other rates. The
    company's own return is used only above the WACC: below it, the default
    would have growth destroy value, which is a claim to make on purpose, so it
    takes the WACC and says why. Growth that destroys value stays available by
    hand.
    """
    wacc = wacc_result.wacc
    own = fin.historical_roic(dataset.annual)
    options = ((_OWN_ROIC,) if own.ok else ()) + (_RONIC_AT_WACC, _RONIC_MANUAL)
    choice = st.radio(
        "Return on new invested capital (RONIC)",
        options,
        key="dcf_input_ronic_basis",
        help="The terminal year reinvests g ÷ RONIC of NOPAT to grow at g. Above the "
             "WACC, growth beyond the forecast creates value; at it, growth neither "
             "creates nor destroys any — the floor for a company without a durable "
             "advantage. However high it goes, the Gordon value rises by at most "
             "WACC ÷ (WACC − g).",
    )
    if not own.ok:
        st.caption(f"No own ROIC to default to — {own.detail}")
    at_wacc = Sourced.estimate(
        wacc, "Set equal to the WACC", "No value is created by growth after the forecast."
    )

    if choice == _OWN_ROIC:
        formula = (
            "NOPAT ÷ average invested capital (equity + debt + minority interest − cash − "
            f"short-term investments), median of {own.years_used} fiscal year(s): "
            f"{', '.join(own.periods)}"
        )
        if own.detail:
            st.caption(f"⚠️ {own.detail}")
        if own.value <= wacc:
            st.caption(
                f"The company's own ROIC, {own.value:.1%}, does not exceed its WACC, so new "
                "capital is taken to earn the WACC: growth that destroys value is not "
                "assumed by default."
            )
            return None, Sourced.estimate(
                wacc, "Set equal to the WACC",
                f"Own ROIC {own.value:.2%} does not exceed the WACC — {formula}.",
            )
        return own.value, Sourced.derived(own.value, f"{own.basis} of reported figures", formula)
    if choice == _RONIC_AT_WACC:
        return None, at_wacc

    if "dcf_input_ronic" not in st.session_state:
        seed = own.value if own.ok and own.value > wacc else wacc
        st.session_state["dcf_input_ronic"] = round(seed * 100, 2)
    ronic_pct = st.number_input(
        "Return on new invested capital (%)", step=0.5, key="dcf_input_ronic", min_value=0.1,
    )
    ronic = ronic_pct / 100.0
    return ronic, Sourced.user(ronic, "Return earned on terminal reinvestment.")


def _show_incremental_return(
    terminal: _Terminal,
    base: dcf_model.BasePeriod,
    drivers: dcf.DriverPath,
    wacc_result: wacc_lib.WaccResult,
) -> None:
    """Fills the slot beside the RONIC input with the forecast's own return.

    Information, not a check: it shows how much excess return the explicit
    forecast carries before the terminal year drops to its RONIC. A projection
    is only arithmetic, so building one here to measure it costs nothing.
    """
    revenue, ebit = base.flows.get("revenue"), base.flows.get("ebit")
    if revenue is None or ebit is None or pd.isna(revenue) or pd.isna(ebit):
        terminal.incremental_return_slot.caption(
            "Forecast incremental return: not available — the base period has no "
            "revenue or operating income."
        )
        return
    projection = dcf.project(
        float(revenue), drivers, wacc=wacc_result.wacc, elapsed=terminal.elapsed,
        mid_year=terminal.mid_year,
    )
    base_nopat = float(ebit) * (1.0 - drivers.effective_tax_rate[0])
    value = dcf.incremental_return_on_capital(projection, base_nopat)
    text = f"{value:.1%}" if value is not None else "not meaningful (no net reinvestment)"
    terminal.incremental_return_slot.caption(
        f"**Forecast incremental return on capital: {text}** — (NOPAT in year "
        f"{drivers.years} − base NOPAT) ÷ Σ (capex − D&A + ΔNWC). The excess return the "
        f"explicit forecast carries before the terminal year earns its RONIC."
    )


def _cash_sbc_and_bridge(
    dataset: dcf_model.CompanyDataset,
    base: dcf_model.BasePeriod,
    records: list[dcf_export.AssumptionRecord],
) -> _Bridge:
    """Excess cash, stock-based compensation, and non-operating assets."""
    stocks = base.stocks
    st.markdown("#### Cash treatment, SBC, and bridge items")
    with st.container(border=True):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            include_sti = st.toggle(
                "Treat short-term investments as excess cash",
                value=True,
                key="dcf_input_sti",
                help="Turning this off treats short-term investments as operating assets.",
            )
            cash_value = stocks.get("cash")
            cash_floor = st.number_input(
                "Operating cash floor to hold back",
                value=0.0,
                step=1e9,
                format="%.0f",
                key="dcf_input_cash_floor",
                help=(
                    "Cash the business needs to run, excluded from the bridge. Materially "
                    "moves the answer for cash-rich companies."
                ),
            )
            short_term = stocks.get("short_term_investments")
            st.caption(
                f"Balance-sheet cash {common.money(cash_value, dataset.valuation_currency)} · short-term "
                f"investments {common.money(short_term, dataset.valuation_currency)} "
                f"(as of {base.stocks_as_of})"
            )

        with col_b:
            st.markdown("**Stock-based compensation**")
            base_sbc = base.flows.get("sbc")
            sbc_available = base_sbc is not None and not pd.isna(base_sbc)

            if not sbc_available:
                st.warning(
                    "**Sharadar reports no stock-based compensation for this company.** The "
                    "field is empty and nothing has been inferred. Enter the figure from the "
                    "cash flow statement in the latest 10-K."
                )

            sbc_input = st.number_input(
                "Stock-based compensation",
                value=float(base_sbc) if sbc_available else float(
                    st.session_state.get("dcf_input_sbc", 0.0)
                ),
                step=1e8,
                format="%.0f",
                key="dcf_input_sbc",
            )
            sbc = sbc_input if sbc_input else None

            if sbc_available:
                sbc_sourced = Sourced.reported(sbc_input, "Sharadar CF: Share Based Compensation")
            elif sbc_input:
                sbc_sourced = Sourced.user(sbc_input, "Entered manually from the filing.")
            else:
                sbc_sourced = Sourced.unavailable(
                    "Sharadar CF: Share Based Compensation",
                    "Not reported and not supplied.",
                )
            common.provenance(sbc_sourced)

            revenue_base = base.flows.get("revenue")
            if sbc and revenue_base:
                st.caption(f"SBC ÷ revenue: **{sbc / float(revenue_base):.2%}**")
            if sbc is not None:
                st.info(
                    "**SBC is expensed, once.** The operating margin is taken from GAAP "
                    "operating income, which already deducts it, so this figure is shown "
                    "for reference and does not move the valuation. Most published FCF "
                    "figures add it back, so this model's FCFF sits below theirs by about "
                    "this amount.",
                    icon="ℹ️",
                )
            records.append(
                dcf_export.AssumptionRecord(
                    "Cash flow", "Stock-based compensation", sbc, sbc_sourced,
                    note="Expensed inside operating income; not added back",
                )
            )

        with col_c:
            st.markdown("**Non-operating assets**")
            lt_investments = stocks.get("long_term_investments")
            reported_lt = lt_investments is not None and not pd.isna(lt_investments)
            if reported_lt:
                st.caption(
                    f"Sharadar reports **{common.money(lt_investments, dataset.valuation_currency)}** of "
                    f"non-current investments (as of {base.stocks_as_of}), at book value. It "
                    "mixes marketable securities with equity-method stakes and restricted "
                    "balances, so it is not added unless you choose to."
                )
            else:
                st.caption(
                    "Sharadar reports no non-current investments line for this company."
                )
            non_operating = st.number_input(
                "Non-operating assets to add",
                value=0.0,
                step=1e8,
                format="%.0f",
                key="dcf_input_nonop",
                help="Long-term and equity-method investments that the operating forecast "
                     "does not already capture. Opt-in: the reported figure is at book and "
                     "mixes operating and non-operating holdings.",
            )
            records.append(
                dcf_export.AssumptionRecord(
                    "Bridge", "Non-operating assets", non_operating,
                    Sourced.user(non_operating, "Opt-in addition to the equity bridge."),
                )
            )

    return _Bridge(include_sti, cash_floor, sbc, non_operating)


def _render_fcff_panel(
    dataset: dcf_model.CompanyDataset,
    base: dcf_model.BasePeriod,
    base_fcff: dcf_model.BaseFcff,
    etr: float | None,
    etr_series: pd.Series,
    delta_nwc_span: str = "",
) -> None:
    st.markdown("#### Base-period free cash flow, computed two independent ways")
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.markdown("**Method A — cash flow statement build**")
            st.code("FCFF = CFO + Interest Expense × (1 − tax rate) − CapEx")
            st.metric("FCFF (method A)", common.money(base_fcff.method_a, dataset.valuation_currency))
        with right:
            st.markdown("**Method B — income statement build**")
            st.code("NOPAT = EBIT × (1 − tax rate)\nFCFF  = NOPAT + D&A − CapEx − ΔNWC")
            st.metric("FCFF (method B)", common.money(base_fcff.method_b, dataset.valuation_currency))

        gap = base_fcff.divergence
        if gap is not None:
            st.caption(f"Divergence between the two builds: **{gap:.2%}**")

        for note in base_fcff.notes:
            st.warning(note)

        inputs_frame = pd.DataFrame(
            [
                {
                    "Input": name,
                    "Value": (
                        f"{value:,.4f}" if name == "Tax rate"
                        else common.money(value, dataset.valuation_currency)
                    ) if value is not None else "not reported",
                }
                for name, value in base_fcff.inputs.items()
            ]
        )
        st.dataframe(inputs_frame, width="stretch", hide_index=True)

        if etr is not None:
            st.caption(
                f"Effective tax rate in use: **{etr:.2%}** — median of the last "
                f"{min(5, len(etr_series))} profitable year(s). Years with a pretax loss are "
                f"excluded because the ratio is meaningless there, not set to zero."
            )
        else:
            st.warning(
                "No profitable year is available to compute an effective tax rate, so the "
                "forecast tax rate driver is used for the base period as well."
            )

        if delta_nwc_span:
            st.caption(f"Change in net working capital measured over — {delta_nwc_span}.")

        st.caption(
            "Method A assumes interest paid sits in operating cash flow, which holds under "
            "US GAAP; an IFRS filer may classify it in financing. Sharadar exposes interest "
            "*expense* (accrual), not interest *paid* (cash), so that is what is used. "
            "The two builds legitimately differ where working capital moves a lot: method B "
            "subtracts the balance-sheet movement, while method A picks up whatever the cash "
            "flow statement booked, including acquisitions and FX translation."
        )
