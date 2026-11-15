"""Every input the DCF takes: its default, the reader's override, and the run they make.

The 💰 DCF tab's assumption panel used to assemble the run inside the Streamlit
view, which left the web app nothing to call. This module is that assembly
without a page. :func:`resolve` takes the dataset, the three market inputs that
have to be fetched (the risk-free rate, the ERP and a regressed beta) and an
:class:`Overrides` naming only what the reader changed; every other input takes
its default *given the choices made*, so a cost of debt nobody typed follows the
risk-free rate it is built on. Both front ends call it, so an untouched panel on
either is one run, and the summary chart's DCF line, :func:`default_value_on`,
is that run once per filing.

:func:`evaluate` then values the run the way the tab draws it: the model, the
reverse DCF under both terminal methods, the sensitivity grid, and every check.
The last section is what the tab reads off a valued run that is more than
formatting — the reverse DCF's revenue paths and verdicts, the cross-check
between the two terminal methods, the filing behind each year — so the API
serves the same reading the Streamlit tab draws.

Each input comes back as a :class:`Field`, holding the value in use, the value
an untouched input holds, its provenance, and the conditional notes about it:
a fallback, a fetch that failed, a basis with no history. The help text that
never changes stays with each front end.

An orchestrator like ``model``, which it sits on. It fetches nothing itself
except in :func:`default_value_on`, which loads a company and its rates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from app.data import market_data
from app.formatting import format_pct
from app.logic.dcf import export as dcf_export
from app.logic.dcf import financials as fin
from app.logic.dcf import model as dcf_model
from app.logic.dcf import reverse
from app.logic.dcf import validation as val
from app.logic.dcf import valuation as dcf
from app.logic.dcf import wacc as wacc_lib
from app.provenance import Origin, Sourced

# ---------------------------------------------------------------------------
# The choices each input offers
# ---------------------------------------------------------------------------

#: The six drivers in panel order: key, the input's label, the export's label.
DRIVERS = (
    ("revenue_growth", "Year-1 revenue growth", "Revenue growth"),
    ("operating_margin", "Starting operating margin", "Operating margin"),
    ("effective_tax_rate", "Effective tax rate", "Effective tax rate"),
    ("capex_pct_revenue", "Starting CapEx % of revenue", "CapEx % of revenue"),
    ("dna_pct_revenue", "Starting D&A % of revenue", "D&A % of revenue"),
    ("nwc_pct_revenue", "NWC % of revenue", "NWC % of revenue"),
)
DRIVER_KEYS = tuple(key for key, _, _ in DRIVERS)

#: The drivers that may move linearly to a year-N target, with their labels.
#: Only capex has a target by default: its own five-year average.
TARGETS = (
    ("operating_margin", "operating margin"),
    ("capex_pct_revenue", "capex % of revenue"),
    ("dna_pct_revenue", "D&A % of revenue"),
)

YEARS_RANGE = (5, 15)

#: Engine key to label, the default first, matching ``DEFAULT_HEADLINE_METHOD``.
HEADLINE_METHODS = {"gordon": "Gordon growth", "exit_multiple": "Exit multiple"}

#: Both are solved; Gordon leads because it is the fundamentals reading — the
#: exit-multiple solve mostly measures today's multiple against its history.
REVERSE_METHODS = ("gordon", "exit_multiple")
REVERSE_LABELS = {"gordon": "Gordon growth (fundamentals)", "exit_multiple": "Exit multiple"}

#: Share count bases. The first is the default.
SHARE_BASES = (
    "Point-in-time + dilution",
    "Diluted weighted-average (TTM)",
    "Point-in-time shares outstanding",
)

BETA_FREQUENCIES = ("monthly", "weekly")
#: Years of returns each regression window covers.
BETA_YEARS = {"monthly": 5, "weekly": 2}

#: Where the terminal year's return on new capital comes from. The company's own
#: ROIC leads, and is the default, wherever its history supplies one.
OWN_ROIC = "Company's own ROIC"
RONIC_AT_WACC = "Equal to the WACC"
RONIC_MANUAL = "Manual"

#: Sharadar publishes one interest expense line, not a gross/net pair.
INTEREST_BASIS = "gross"


class InvalidOverride(ValueError):
    """An override names a choice the panel does not offer, or a value out of range."""


@dataclass(frozen=True)
class Overrides:
    """What the reader changed. Anything left ``None`` (or out of a mapping) is the default.

    Rates are fractions, not percentages. ``target_on`` turns a driver's
    year-N target on or off; ``target_values`` moves where an on target lands,
    which otherwise starts at the five-year average for capex and at the
    starting value for the other two.
    """

    years: int | None = None
    bases: Mapping[str, str] = field(default_factory=dict)
    values: Mapping[str, float] = field(default_factory=dict)
    fade: bool | None = None
    hold: int | None = None
    target_on: Mapping[str, bool] = field(default_factory=dict)
    target_values: Mapping[str, float] = field(default_factory=dict)
    acquisitions: bool | None = None

    risk_free: float | None = None
    erp: float | None = None
    beta_frequency: str | None = None
    blume: bool | None = None
    beta: float | None = None
    cost_of_debt: float | None = None
    marginal_tax: float | None = None
    share_basis: str | None = None

    terminal_growth: float | None = None
    ronic_basis: str | None = None
    ronic: float | None = None
    terminal_at_marginal: bool | None = None
    ebita_proxy: bool | None = None
    exit_multiple: float | None = None
    headline_method: str | None = None
    mid_year: bool | None = None

    include_sti: bool | None = None
    cash_floor: float | None = None
    sbc: float | None = None
    non_operating: float | None = None


def beta_settings(overrides: Overrides) -> tuple[str, bool, int]:
    """The regression a caller has to run before :func:`resolve`: frequency, Blume, years.

    The beta is fetched, so it is the caller's to regress; this keeps both front
    ends regressing the same window for the same choice.
    """
    frequency = overrides.beta_frequency or dcf_model.DEFAULT_BETA_FREQUENCY
    if frequency not in BETA_FREQUENCIES:
        raise InvalidOverride(f"Unknown beta frequency {frequency!r}.")
    blume = dcf_model.DEFAULT_BLUME if overrides.blume is None else overrides.blume
    return frequency, blume, BETA_YEARS[frequency]


# ---------------------------------------------------------------------------
# What resolve returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Note:
    """A conditional remark about an input: ``caption``, ``info`` or ``warning``."""

    level: str
    text: str


@dataclass(frozen=True)
class Field:
    """One numeric input: the value in use, the untouched value, and their provenance.

    ``default`` is ``None`` where nothing could pre-populate the input, and
    ``value`` is ``None`` only where the input itself is optional (a RONIC that
    tracks the WACC, an SBC nobody reported).
    """

    value: float | None
    default: float | None
    sourced: Sourced
    notes: tuple[Note, ...] = ()

    @property
    def overridden(self) -> bool:
        return self.sourced.origin is Origin.USER and self.default is not None


@dataclass(frozen=True)
class Driver:
    """One forecast driver: its bases, the one in use, and the value."""

    key: str
    label: str
    bases: tuple[str, ...]
    default_basis: str
    basis: str
    input: Field
    periods: tuple[str, ...]
    #: True when the basis has no usable history, so the value is typed by hand.
    manual: bool


@dataclass(frozen=True)
class Target:
    """An optional year-N level a driver moves to, linearly from year 1."""

    key: str
    label: str
    default_on: bool
    on: bool
    #: Where an on target starts: the normal level where history has one,
    #: otherwise the driver's starting value.
    seed: float
    input: Field | None
    notes: tuple[Note, ...] = ()


@dataclass(frozen=True)
class Acquisitions:
    """Acquisition spend folded into capex, for a company that buys its growth."""

    spend: fin.DriverDefault
    on: bool

    @property
    def share(self) -> float:
        return float(self.spend.value) if self.on else 0.0


@dataclass(frozen=True)
class Blocked:
    """Why there is no valuation: drawn in place of the answer, never beside one."""

    title: str
    message: str
    detail: str = ""


@dataclass(frozen=True)
class Assumptions:
    """The resolved panel, and the run it makes.

    ``inputs`` is the snapshot the Streamlit tab fingerprints and stores, and
    what :func:`evaluate` and :func:`run_kwargs` read. It is ``None`` when
    ``blocked`` says why the run cannot be assembled.
    """

    base: dcf_model.BasePeriod
    years: int
    drivers_in: tuple[Driver, ...]
    fade: bool
    hold: int
    targets: tuple[Target, ...]
    acquisitions: Acquisitions | None
    capex_estimate: str | None

    risk_free: Field
    erp: Field
    beta_frequency: str
    blume: bool
    beta: Field
    beta_outcome: dcf_model.BetaOutcome
    cost_of_debt: Field
    synthetic: wacc_lib.SyntheticCostOfDebt | None
    implied_cost_of_debt: float | None
    marginal_tax: Field

    share_basis_choice: str
    shares: float | None
    share_basis: str
    share_notes: tuple[Note, ...]
    wacc: wacc_lib.WaccResult | None
    equity_value: float
    debt_value: float

    terminal_growth: Field
    ronic_bases: tuple[str, ...]
    ronic_basis: str
    ronic: Field
    incremental_return: float | None
    incremental_return_note: str
    terminal_at_marginal: bool
    ebita_proxy: bool
    exit_multiple: Field
    headline_method: str
    mid_year: bool
    elapsed: float
    elapsed_note: Note

    include_sti: bool
    cash_floor: float
    sbc: Field
    non_operating: float

    base_fcff: dcf_model.BaseFcff
    etr: float | None
    etr_years: int
    delta_nwc_span: str

    drivers: dcf.DriverPath
    records: tuple[dcf_export.AssumptionRecord, ...]
    inputs: dict | None
    blocked: Blocked | None = None

    def driver(self, key: str) -> Driver:
        return next(d for d in self.drivers_in if d.key == key)


# ---------------------------------------------------------------------------
# Small rules every input follows
# ---------------------------------------------------------------------------


def _changed(value: float | None, default: float | None, places: int) -> bool:
    """True when ``value`` is a real override of ``default`` at ``places`` decimals.

    Both front ends show a default rounded, so a value read back from an
    untouched input differs from it by the rounding. A rate shown to four
    decimals of a percent is six decimals of a fraction.
    """
    if value is None:
        return False
    if default is None:
        return True
    return abs(float(value) - float(default)) >= 0.5 * 10.0**-places + 1e-12


def _choose(choice: str | None, options: tuple[str, ...], default: str, what: str) -> str:
    if choice is None:
        return default
    if choice not in options:
        raise InvalidOverride(f"{what} {choice!r} is not offered; choose one of {', '.join(options)}.")
    return choice


def _flag(choice: bool | None, default: bool) -> bool:
    return default if choice is None else bool(choice)


def effective_growth_basis(
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


# ---------------------------------------------------------------------------
# Forecast drivers
# ---------------------------------------------------------------------------


def _offered(
    dataset: dcf_model.CompanyDataset, base: dcf_model.BasePeriod, key: str
) -> tuple[tuple[str, ...], str, dict[str, fin.DriverDefault], tuple[Note, ...]]:
    """The bases a driver offers, the default, and the bases already computed.

    The leading precomputed basis is the trailing one: the lower of TTM and the
    five-year CAGR for growth, the base period's own ratio for margin, capex,
    D&A and working capital. Where it is available it leads and is the default —
    except for working capital, a level that swings with timing, where the
    five-year average leads and the single balance sheet is offered after it.
    A basis that cannot be computed is left out, and a note says why the
    default fell back — never silently.
    """
    growth = key == "revenue_growth"
    bases = fin.GROWTH_BASES if growth else fin.BASES
    default_basis = "5-year CAGR" if growth else "5-year average"
    also: tuple[fin.DriverDefault, ...] = ()
    trailing: fin.DriverDefault | None = None
    if growth:
        ttm = fin.ttm_revenue_growth(dataset.ttm_history, dataset.ttm_history_meta)
        trailing = fin.lower_of_growth(
            ttm, fin.driver_default(dataset.ratios, key, "5-year CAGR")
        )
        also = (ttm,)
    elif key != "effective_tax_rate":
        trailing = fin.base_period_driver(
            key, base.flows, base.stocks, is_ttm=base.is_ttm, label=base.label
        )
    trailing_is_default = key != "nwc_pct_revenue"

    precomputed = {d.basis: d for d in (trailing, *also) if d is not None and d.ok}
    extra = tuple(d.basis for d in also if d.ok)
    notes: tuple[Note, ...] = ()
    if trailing is not None and trailing.ok:
        if trailing_is_default:
            bases = (trailing.basis,) + extra + bases
            default_basis = trailing.basis
        else:
            bases = bases + (trailing.basis,) + extra
    else:
        bases = extra + bases
        if trailing_is_default and extra:
            default_basis = extra[0]
    if trailing is not None and not trailing.ok and trailing_is_default:
        notes = (Note("caption", f"{trailing.basis} is unavailable — {trailing.detail} "
                                 f"Defaulting to {default_basis} instead."),)
    return bases, default_basis, precomputed, notes


def _resolve_driver(
    dataset: dcf_model.CompanyDataset,
    base: dcf_model.BasePeriod,
    key: str,
    label: str,
    overrides: Overrides,
    strict: bool,
) -> Driver:
    bases, default_basis, precomputed, notes = _offered(dataset, base, key)
    basis = _choose(overrides.bases.get(key), bases, default_basis, f"The {label.lower()} basis")
    default = precomputed.get(basis) or fin.driver_default(dataset.ratios, key, basis)
    override = overrides.values.get(key)

    if not default.ok:
        if strict:
            raise dcf_model.DefaultUnavailable(
                f"No usable history for {fin.DRIVER_LABELS[key]}"
            )
        value = float(override) if override is not None else 0.0
        notes += (Note("warning", f"No usable history for {label.lower()} — {default.detail} "
                                  "There is nothing to pre-populate from, so this starts "
                                  "empty and must be entered by hand."),)
        return Driver(
            key, label, bases, default_basis, basis,
            Field(value, None, Sourced.user(value, "No historical basis available."), notes),
            (), manual=True,
        )

    derived = Sourced.derived(
        value=default.value,
        source=f"{basis} of reported figures",
        formula=f"{fin.DRIVER_LABELS[key]} over {default.years_used} period(s): "
        f"{', '.join(str(p) for p in default.periods[-6:])}",
    )
    if _changed(override, default.value, 6):
        value, sourced = float(override), derived.with_user_value(float(override))
    else:
        value, sourced = float(default.value), derived
    if default.detail:
        notes += (Note("caption", f"⚠️ {default.detail}"),)
    return Driver(
        key, label, bases, default_basis, basis,
        Field(value, float(default.value), sourced, notes), default.periods, manual=False,
    )


def resolve_target(
    key: str,
    label: str,
    start: float,
    years: int,
    *,
    normal: fin.DriverDefault | None,
    on: bool | None,
    value: float | None,
) -> Target:
    """A driver's optional year-N target.

    ``normal`` is the level history says the driver returns to — only capex
    has one (:func:`fin.normal_capex`). Where it can be computed the target is
    on by default and lands there, derived rather than typed; where it cannot,
    it starts off and a note says why. The margin and D&A have none: whether a
    margin expands is a claim about the business, not something history
    supplies, so their paths stay flat unless someone sets a target.
    """
    normalises = normal is not None and normal.ok
    notes: list[Note] = []
    if normalises:
        notes.append(Note("caption", f"On by default: the {label} reaches its own "
                                     f"{normal.basis.lower()} by year {years}, so neither "
                                     "today's build-out nor a trough is held forever."))
    elif normal is not None:
        notes.append(Note("caption", f"No {normal.basis.lower()} to normalise to — "
                                     f"{normal.detail} The starting value is held unless "
                                     "you set a target."))
    is_on = _flag(on, normalises)
    seed = float(normal.value) if normalises else float(start)
    if not is_on:
        return Target(key, label, normalises, False, seed, None, tuple(notes))

    reached = f"Reached in year {years}, linearly from year 1."
    target = float(value) if _changed(value, seed, 6) else seed
    if normalises:
        sourced = Sourced.derived(
            value=normal.value,
            source=f"{normal.basis} of reported figures",
            formula=f"{fin.DRIVER_LABELS[normal.key]} over {normal.years_used} period(s): "
            f"{', '.join(str(p) for p in normal.periods)}. {reached}",
        )
        if _changed(value, seed, 6):
            sourced = sourced.with_user_value(target)
        if normal.detail:
            notes.append(Note("caption", f"⚠️ {normal.detail}"))
    else:
        sourced = Sourced.user(target, reached)
    return Target(
        key, label, normalises, True, seed,
        Field(target, seed if normalises else None, sourced), tuple(notes),
    )


# ---------------------------------------------------------------------------
# Terminal year
# ---------------------------------------------------------------------------


def resolve_ronic(
    annual: pd.DataFrame,
    wacc: float | None,
    *,
    basis: str | None,
    value: float | None,
) -> tuple[tuple[str, ...], str, Field]:
    """The terminal year's RONIC: the bases offered, the one in use, and the value.

    The value is ``None`` where it is the WACC — ``None`` rather than the WACC's
    number, so the perpetuity keeps tracking the WACC wherever it moves (the
    sensitivity grid reruns it at other rates). The company's own return is the
    default only above the WACC: below it, the default would have growth destroy
    value, which is a claim to make on purpose, so it takes the WACC and says
    why. Growth that destroys value stays available by hand.
    """
    own = fin.historical_roic(annual)
    options = ((OWN_ROIC,) if own.ok else ()) + (RONIC_AT_WACC, RONIC_MANUAL)
    choice = _choose(basis, options, options[0], "The RONIC basis")
    notes: list[Note] = []
    if not own.ok:
        notes.append(Note("caption", f"No own ROIC to default to — {own.detail}"))
    if wacc is None:
        return options, choice, Field(
            None, None, Sourced.unavailable("Return on new invested capital", "No WACC."),
            tuple(notes),
        )
    at_wacc = Sourced.estimate(
        wacc, "Set equal to the WACC", "No value is created by growth after the forecast."
    )

    if choice == OWN_ROIC:
        formula = (
            "NOPAT ÷ average invested capital (equity + debt + minority interest − cash − "
            f"short-term investments), median of {own.years_used} fiscal year(s): "
            f"{', '.join(own.periods)}"
        )
        if own.detail:
            notes.append(Note("caption", f"⚠️ {own.detail}"))
        if own.value <= wacc:
            notes.append(Note("caption", f"The company's own ROIC, {own.value:.1%}, does not "
                                         "exceed its WACC, so new capital is taken to earn "
                                         "the WACC: growth that destroys value is not "
                                         "assumed by default."))
            return options, choice, Field(None, None, Sourced.estimate(
                wacc, "Set equal to the WACC",
                f"Own ROIC {own.value:.2%} does not exceed the WACC — {formula}.",
            ), tuple(notes))
        sourced = Sourced.derived(own.value, f"{own.basis} of reported figures", formula)
        return options, choice, Field(float(own.value), float(own.value), sourced, tuple(notes))
    if choice == RONIC_AT_WACC:
        return options, choice, Field(None, None, at_wacc, tuple(notes))

    seed = float(own.value) if own.ok and own.value > wacc else float(wacc)
    ronic = float(value) if value is not None else seed
    if ronic <= 0:
        raise InvalidOverride("The return on new invested capital must be positive.")
    return options, choice, Field(
        ronic, seed, Sourced.user(ronic, "Return earned on terminal reinvestment."), tuple(notes)
    )


# ---------------------------------------------------------------------------
# The panel, top to bottom
# ---------------------------------------------------------------------------


def _share_count(
    dataset: dcf_model.CompanyDataset, choice: str
) -> tuple[float | None, str, tuple[Note, ...]]:
    """The share count value per share divides by, and a label saying which one.

    The default is today's count plus the dilution the diluted count carries
    over the basic one. The weighted-average count lags a company buying back
    stock by the buyback, and understates value per share by as much; the bare
    point-in-time count ignores options and RSUs altogether.
    """
    point_in_time = dataset.shares_point_in_time
    dilution = None
    if dataset.shares_weighted_diluted and dataset.shares_weighted_basic:
        dilution = max(dataset.shares_weighted_diluted - dataset.shares_weighted_basic, 0.0)

    notes: tuple[Note, ...] = ()
    if choice == SHARE_BASES[0] and point_in_time and dilution is not None:
        return point_in_time + dilution, (
            f"Point-in-time outstanding (as of {dataset.shares_point_in_time_date}) + "
            f"{dilution:,.0f} dilution ({dataset.shares_weighted_label} diluted − basic)"
        ), notes
    if choice == SHARE_BASES[1] and dataset.shares_weighted_diluted:
        return dataset.shares_weighted_diluted, (
            f"Diluted weighted-average ({dataset.shares_weighted_label})"
        ), notes
    if point_in_time:
        if choice == SHARE_BASES[0]:
            notes = (Note("caption", "No weighted basic and diluted pair is reported, so no "
                                     "dilution could be added; this is the bare "
                                     "point-in-time count."),)
        return point_in_time, (
            f"Point-in-time outstanding (as of {dataset.shares_point_in_time_date})"
        ), notes
    if dataset.shares_weighted_diluted:
        return dataset.shares_weighted_diluted, (
            f"Diluted weighted-average ({dataset.shares_weighted_label})"
        ), notes
    return None, "", notes


def _implied_cost_of_debt(dataset: dcf_model.CompanyDataset) -> float | None:
    """Interest ÷ average total debt — a backward-looking average coupon, a cross-check only."""
    interest = fin.interest_expense(dataset.annual)
    debt = fin.total_debt(dataset.annual)
    if not len(interest) or not len(debt):
        return None

    def at(series: pd.Series, i: int) -> float | None:
        return None if len(series) < -i or pd.isna(series.iloc[i]) else float(series.iloc[i])

    return wacc_lib.implied_cost_of_debt(at(interest, -1), at(debt, -1), at(debt, -2))


_NO_RATING = {
    "undisclosed": "interest expense reads 0 beside material debt, which is how Sharadar "
                   "records interest a filing does not break out — it looks undisclosed, and "
                   "none of the last {n} fiscal years disclosed it either",
    "missing": "the base period reports no interest expense line, and none of the last {n} "
               "fiscal years disclosed one either",
    "negative": "the base period's interest line is negative — net interest income, a net "
                "finance result rather than the cost of debt — so there is no coverage ratio "
                "to rate",
}


def resolve(
    dataset: dcf_model.CompanyDataset,
    *,
    valuation_date: date,
    risk_free: Sourced,
    erp: Sourced,
    beta: dcf_model.BetaOutcome,
    overrides: Overrides = Overrides(),
    strict: bool = False,
) -> Assumptions:
    """Every input, defaulted where the reader left it, and the run they make.

    ``beta`` must be the regression :func:`beta_settings` names for these
    overrides. ``strict`` is for a value nobody will look at the panel for —
    the summary chart's line of past values: where the tab would put a
    placeholder in front of the reader and ask for a number (no risk-free rate,
    no ERP, no regressed beta, a driver with no history, no share count, a WACC
    that cannot be weighted), it raises :class:`dcf_model.DefaultUnavailable`
    instead, because a line drawn through placeholders would look like a
    measurement. The two placeholders the tab presents as defaults stay: a cost
    of debt at the risk-free rate when no rating can be formed, and a generic
    exit multiple, which never reaches the Gordon headline.
    """
    if strict:
        if not risk_free.ok:
            raise dcf_model.DefaultUnavailable(
                f"No risk-free rate — {risk_free.detail}", transient=True
            )
        if not erp.ok:
            raise dcf_model.DefaultUnavailable(f"No equity risk premium — {erp.detail}", transient=True)
        if beta.result is None:
            raise dcf_model.DefaultUnavailable(f"No beta — {beta.error}")

    base = dcf_model.base_period(dataset)
    records: list[dcf_export.AssumptionRecord] = []

    # --- Forecast drivers ------------------------------------------------
    low, high = YEARS_RANGE
    years = dcf_model.DEFAULT_FORECAST_YEARS if overrides.years is None else int(overrides.years)
    if not low <= years <= high:
        raise InvalidOverride(f"The forecast runs {low} to {high} years, not {years}.")

    if strict:
        missing = []
        for key, label, _ in DRIVERS:
            try:
                _resolve_driver(dataset, base, key, label, overrides, strict=True)
            except dcf_model.DefaultUnavailable:
                missing.append(fin.DRIVER_LABELS[key])
        if missing:
            raise dcf_model.DefaultUnavailable(f"No usable history for {', '.join(missing)}")
    drivers_in = tuple(
        _resolve_driver(dataset, base, key, label, overrides, strict)
        for key, label, _ in DRIVERS
    )
    values = {d.key: float(d.input.value) for d in drivers_in}
    periods_in_use: set[str] = {str(p) for d in drivers_in for p in d.periods}
    for driver, (_, _, record_label) in zip(drivers_in, DRIVERS):
        records.append(dcf_export.AssumptionRecord(
            "Forecast drivers", record_label, driver.input.value, driver.input.sourced
        ))

    growth = drivers_in[0]
    ttm_growth = fin.ttm_revenue_growth(dataset.ttm_history, dataset.ttm_history_meta)
    lower_growth = fin.lower_of_growth(
        ttm_growth, fin.driver_default(dataset.ratios, "revenue_growth", "5-year CAGR")
    )
    growth_basis = effective_growth_basis(growth.basis, lower_growth, ttm_growth)

    margin = drivers_in[1]
    assumed_margins: dict[str, float] = {}
    if margin.basis != fin.TTM_BASIS or margin.input.sourced.origin is Origin.USER:
        assumed_margins["Starting margin"] = values["operating_margin"]

    fade = _flag(overrides.fade, True)
    hold_default = min(dcf_model.DEFAULT_GROWTH_HOLD, years - 1)
    hold = hold_default if overrides.hold is None else int(overrides.hold)
    if not 1 <= hold <= years - 1:
        raise InvalidOverride(f"Year-1 growth can be held 1 to {years - 1} years, not {hold}.")
    records.append(dcf_export.AssumptionRecord(
        "Forecast drivers", "Growth path",
        f"held {hold} years, then faded to terminal growth" if fade else "flat",
        Sourced.user(float(hold if fade else 0), "Shape of the revenue growth path."),
    ))

    normal = fin.normal_capex(dataset.ratios)
    targets = tuple(
        resolve_target(
            key, label, values[key], years,
            normal=normal if key == "capex_pct_revenue" else None,
            on=overrides.target_on.get(key), value=overrides.target_values.get(key),
        )
        for key, label in TARGETS
    )
    for target in targets:
        records.append(
            dcf_export.AssumptionRecord(
                "Forecast drivers", f"Target {target.label}", target.input.value,
                target.input.sourced,
            )
            if target.on else
            dcf_export.AssumptionRecord(
                "Forecast drivers", f"Target {target.label}", "none — flat",
                Sourced.user(values[target.key], "No target: the starting value is held."),
            )
        )
    margin_target = next(t for t in targets if t.key == "operating_margin")
    if margin_target.on:
        assumed_margins["Target margin"] = margin_target.input.value

    spend = fin.driver_default(dataset.ratios, fin.ACQUISITIONS_KEY, "5-year average")
    acquisitions = None
    if spend.ok and spend.value > 0:
        acquisitions = Acquisitions(spend, _flag(overrides.acquisitions, False))
        records.append(dcf_export.AssumptionRecord(
            "Forecast drivers", "Acquisitions added to capex",
            spend.value if acquisitions.on else 0.0,
            Sourced.derived(spend.value, "5-year average of reported history",
                            "Business acquisitions ÷ revenue"),
        ))
    acquisitions_share = acquisitions.share if acquisitions else 0.0
    capex_estimate = (
        f"**CapEx is an estimate.** {dataset.capex_result.detail} "
        f"Formula in use: `{dataset.capex_result.formula}`"
        if dataset.capex_result.origin == "estimate" else None
    )

    # --- Discount rate ----------------------------------------------------
    rf_default = float(risk_free.value) if risk_free.ok else 0.0
    rf_value = float(overrides.risk_free) if _changed(overrides.risk_free, rf_default, 6) else rf_default
    rf_field = Field(
        rf_value, rf_default if risk_free.ok else None,
        risk_free if risk_free.ok and not _changed(overrides.risk_free, rf_default, 6)
        else Sourced.user(rf_value, "Manually entered risk-free rate."),
        () if risk_free.ok else (Note("warning", "The 10-year Treasury rate could not be "
                                                 f"fetched — {risk_free.detail}. Enter it "
                                                 "manually; nothing has been substituted."),),
    )
    records.append(dcf_export.AssumptionRecord(
        "Discount rate", "Risk-free rate", rf_value, rf_field.sourced
    ))

    erp_default = float(erp.value) if erp.ok else 0.0
    erp_value = float(overrides.erp) if _changed(overrides.erp, erp_default, 6) else erp_default
    erp_field = Field(
        erp_value, erp_default if erp.ok else None,
        erp if erp.ok and not _changed(overrides.erp, erp_default, 6)
        else Sourced.user(erp_value, "Manually entered equity risk premium."),
        () if erp.ok else (Note("warning", "The equity risk premium scrape failed — "
                                           f"{erp.detail}. Enter it manually; no default has "
                                           "been substituted."),),
    )
    records.append(dcf_export.AssumptionRecord(
        "Discount rate", "Equity risk premium", erp_value, erp_field.sourced
    ))

    frequency, blume, _ = beta_settings(overrides)
    regressed = beta.result
    beta_default = float(regressed.beta) if regressed is not None else 1.0
    beta_changed = _changed(overrides.beta, beta_default, 4)
    beta_value = float(overrides.beta) if beta_changed else beta_default
    if regressed is not None and not beta_changed:
        beta_sourced = Sourced.derived(beta_default, beta.source, regressed.formula)
    else:
        beta_sourced = Sourced.user(
            beta_value,
            f"Overrides the regressed β of {regressed.beta:.4f}" if regressed
            else "No regression was available.",
        )
    beta_notes: tuple[Note, ...] = ()
    if regressed is None:
        beta_notes += (Note("warning", f"Beta could not be regressed — {beta.error} Enter one "
                                       "manually; the field starts at the market beta of 1.0, "
                                       "which is a placeholder, not an estimate for this "
                                       "company."),)
    elif beta.price_source_note:
        beta_notes += (Note("caption", f"⚠️ {beta.price_source_note}"),)
    beta_field = Field(
        beta_value, beta_default if regressed is not None else None, beta_sourced, beta_notes
    )
    records.append(dcf_export.AssumptionRecord("Discount rate", "Beta", beta_value, beta_sourced))

    synthetic, cause = dcf_model.synthetic_cost_of_debt(dataset, base, rf_value)
    kd_default = synthetic.cost_of_debt if synthetic is not None else rf_value
    kd_changed = _changed(overrides.cost_of_debt, kd_default, 6)
    kd_value = float(overrides.cost_of_debt) if kd_changed else kd_default
    from_table = synthetic is not None and not kd_changed
    kd_notes: tuple[Note, ...] = ()
    if synthetic is None:
        reason = _NO_RATING.get(
            cause, "EBIT or interest expense is not reported for the base period"
        ).format(n=wacc_lib.DISCLOSED_INTEREST_LOOKBACK_YEARS)
        kd_notes = (Note("warning", f"No synthetic rating could be formed — {reason}. The "
                                    "field starts at the risk-free rate, which is a "
                                    "placeholder rather than an estimate; enter a yield to "
                                    "maturity."),)
    if from_table:
        kd_sourced = Sourced.estimate(
            kd_value,
            "Synthetic rating from interest coverage — Damodaran spreads as of "
            f"{wacc_lib.SYNTHETIC_RATINGS_AS_OF}",
            (
                f"Base-period interest is {'not reported' if cause == 'missing' else 'undisclosed'}, "
                f"so coverage is taken from {synthetic.coverage_period}, the latest year that "
                "disclosed it, and its spread applied to today's risk-free rate. "
                if synthetic.coverage_period else ""
            )
            + "The spread table drifts year to year and is an estimate; a traded bond's yield "
            "to maturity is better evidence.",
            formula=synthetic.formula,
        )
    else:
        kd_sourced = Sourced.user(
            kd_value, "Manual yield-to-maturity input — Sharadar carries no bond data."
        )
    kd_field = Field(kd_value, kd_default if synthetic is not None else None, kd_sourced, kd_notes)
    records.append(dcf_export.AssumptionRecord(
        "Discount rate", "Cost of debt (YTM)", kd_value, kd_sourced
    ))

    tax_default = dcf_model.DEFAULT_MARGINAL_TAX
    tax_changed = _changed(overrides.marginal_tax, tax_default, 6)
    marginal_tax = float(overrides.marginal_tax) if tax_changed else tax_default
    tax_field = Field(
        marginal_tax, tax_default,
        Sourced.user(marginal_tax, "Statutory marginal rate for the tax shield.") if tax_changed
        else Sourced.estimate(marginal_tax, "Convention: US federal 21% plus state",
                              "A combined US statutory rate; a company taxed mostly abroad "
                              "has a different one."),
        (Note("caption", "⚠️ 25% is a US federal-plus-state rate. For a company taxed at home "
                         "outside the US, set this to its home country's statutory rate."),)
        if dataset.is_converted or dataset.foreign_filer else (),
    )
    records.append(dcf_export.AssumptionRecord(
        "Discount rate", "Marginal tax rate", marginal_tax, tax_field.sourced
    ))

    # --- Shares and the WACC ---------------------------------------------
    share_choice = _choose(overrides.share_basis, SHARE_BASES, SHARE_BASES[0], "The share basis")
    shares, share_basis, share_notes = _share_count(dataset, share_choice)
    blocked: Blocked | None = None
    if shares is None:
        if strict:
            raise dcf_model.DefaultUnavailable("No share count")
        blocked = Blocked(
            "No share count available",
            f"Sharadar returned no share count for {dataset.ticker}, so equity value cannot "
            "be divided into a per-share figure.",
        )
    else:
        records.append(dcf_export.AssumptionRecord(
            "Capital structure", "Shares used", shares,
            Sourced.reported(shares, share_basis, dataset.shares_point_in_time_date),
        ))

    debt = fin.total_debt(pd.DataFrame([base.stocks]))
    debt_value = float(debt.iloc[0]) if not debt.empty and not pd.isna(debt.iloc[0]) else 0.0
    equity_value = (dataset.last_price or 0.0) * (dataset.shares_point_in_time or 0.0)
    wacc_result = None
    try:
        wacc_result = wacc_lib.compute_wacc(
            equity_value=equity_value,
            debt_value=debt_value,
            risk_free_rate=rf_value,
            beta=beta_value,
            equity_risk_premium=erp_value,
            cost_of_debt=kd_value,
            marginal_tax_rate=marginal_tax,
        )
    except ValueError as error:
        if strict:
            raise dcf_model.DefaultUnavailable(f"The WACC cannot be weighted — {error}") from None
        blocked = blocked or Blocked("The WACC cannot be weighted", str(error))

    # --- Terminal value and conventions ----------------------------------
    g_default = dcf_model.DEFAULT_TERMINAL_GROWTH
    g_changed = _changed(overrides.terminal_growth, g_default, 6)
    terminal_growth = float(overrides.terminal_growth) if g_changed else g_default
    growth_field = Field(
        terminal_growth, g_default,
        Sourced.user(terminal_growth, "Gordon growth rate.") if g_changed
        else Sourced.estimate(terminal_growth, "Convention: long-run inflation plus modest real growth",
                              "A default, not a measurement. It should sit at or below the "
                              "risk-free rate, the market's own long-run nominal growth."),
    )
    records.append(dcf_export.AssumptionRecord(
        "Terminal value", "Perpetuity growth", terminal_growth, growth_field.sourced
    ))

    ronic_bases, ronic_basis, ronic_field = resolve_ronic(
        dataset.annual, wacc_result.wacc if wacc_result else None,
        basis=overrides.ronic_basis, value=overrides.ronic,
    )
    if wacc_result is not None:
        records.append(dcf_export.AssumptionRecord(
            "Terminal value", "Return on new invested capital",
            wacc_result.wacc if ronic_field.value is None else ronic_field.value,
            ronic_field.sourced,
        ))

    terminal_at_marginal = _flag(overrides.terminal_at_marginal, True)
    terminal_tax_rate = marginal_tax if terminal_at_marginal else None
    ebita_proxy = _flag(overrides.ebita_proxy, True)
    records.append(dcf_export.AssumptionRecord(
        "Terminal value", "EBITA proxy", "on" if ebita_proxy else "off",
        Sourced.user(float(ebita_proxy), "Amortisation add-back in year N + 1."),
    ))
    records.append(dcf_export.AssumptionRecord(
        "Terminal value", "Terminal tax rate",
        terminal_tax_rate if terminal_tax_rate is not None else "forecast ETR",
        Sourced.user(marginal_tax, "Marginal rate from the discount panel.")
        if terminal_at_marginal else Sourced.user(0.0, "Final forecast year's effective rate."),
    ))

    ev_ebitda = dcf_model.historical_ev_ebitda(dataset)
    multiple_default = (
        round(float(ev_ebitda.value), 2) if ev_ebitda.ok else dcf_model.PLACEHOLDER_EXIT_MULTIPLE
    )
    multiple_changed = _changed(overrides.exit_multiple, multiple_default, 2)
    exit_multiple = float(overrides.exit_multiple) if multiple_changed else multiple_default
    exit_field = Field(
        exit_multiple, multiple_default,
        ev_ebitda if ev_ebitda.ok and not multiple_changed
        else Sourced.user(exit_multiple, "Manually set exit multiple."),
        () if ev_ebitda.ok else (Note("warning", "No historical EV/EBITDA range available — "
                                                 f"{ev_ebitda.detail} The field starts at a "
                                                 "generic 10×, which is a placeholder rather "
                                                 "than this company's own trading range."),),
    )
    records.append(dcf_export.AssumptionRecord(
        "Terminal value", "Exit EV/EBITDA multiple", exit_multiple, exit_field.sourced
    ))

    headline_method = _choose(
        overrides.headline_method, tuple(HEADLINE_METHODS), dcf_model.DEFAULT_HEADLINE_METHOD,
        "The headline method",
    )
    mid_year = _flag(overrides.mid_year, True)

    if base.flows_as_of is None:
        elapsed = 0.0
        elapsed_detail = "no base-period end date available"
        elapsed_note = Note("warning", "The base period's end date was not reported, so the "
                                       "forecast is discounted as if it ended today.")
    else:
        elapsed = dcf.elapsed_years(base.flows_as_of, valuation_date)
        elapsed_detail = (
            f"{(valuation_date - base.flows_as_of).days} days from {base.flows_as_of} "
            f"to {valuation_date} ÷ 365"
        )
        elapsed_note = Note("caption", f"**Timing:** year 1 is the twelve months after the "
                                       f"base period ends on **{base.flows_as_of}**, the same "
                                       "date as the balance sheet the bridge reads. The value "
                                       f"is struck there and rolled forward **{elapsed:.4f}** "
                                       f"years to {valuation_date} at the WACC, so it moves "
                                       "smoothly with the date and changes step only when a "
                                       "new quarter is filed.")
    records.append(dcf_export.AssumptionRecord(
        "Conventions", "Years from base period end to valuation date", elapsed,
        Sourced.derived(elapsed, "Base period end date", elapsed_detail),
    ))

    # --- Cash, SBC and the bridge ----------------------------------------
    include_sti = _flag(overrides.include_sti, True)
    cash_floor = float(overrides.cash_floor or 0.0)
    if cash_floor < 0:
        raise InvalidOverride("The operating cash floor cannot be negative.")

    reported_sbc = base.flows.get("sbc")
    reported_sbc = (
        float(reported_sbc) if reported_sbc is not None and not pd.isna(reported_sbc) else None
    )
    sbc_changed = _changed(overrides.sbc, reported_sbc, 0)
    sbc_value = float(overrides.sbc) if sbc_changed else reported_sbc
    sbc = sbc_value or None
    if reported_sbc is not None and not sbc_changed:
        sbc_sourced = Sourced.reported(reported_sbc, "Sharadar CF: Share Based Compensation")
    elif sbc:
        sbc_sourced = Sourced.user(sbc, "Entered manually from the filing.")
    else:
        sbc_sourced = Sourced.unavailable(
            "Sharadar CF: Share Based Compensation", "Not reported and not supplied."
        )
    sbc_field = Field(
        sbc, reported_sbc, sbc_sourced,
        () if reported_sbc is not None else (Note("warning", "**Sharadar reports no stock-based "
                                                             "compensation for this company.** "
                                                             "The field is empty and nothing "
                                                             "has been inferred. Enter the "
                                                             "figure from the cash flow "
                                                             "statement in the latest 10-K."),),
    )
    records.append(dcf_export.AssumptionRecord(
        "Cash flow", "Stock-based compensation", sbc, sbc_sourced,
        note="Expensed inside operating income; not added back",
    ))

    non_operating = float(overrides.non_operating or 0.0)
    records.append(dcf_export.AssumptionRecord(
        "Bridge", "Non-operating assets", non_operating,
        Sourced.user(non_operating, "Opt-in addition to the equity bridge."),
    ))

    # --- The driver paths ------------------------------------------------
    # The growth path fades toward the terminal rate, so it is shaped only once
    # both are known. The margin, capex and D&A move to their targets where one
    # is set; the tax rate and working capital stay flat.
    fade_to = terminal_growth if fade else None

    def path(key: str) -> tuple[float, ...]:
        start = values[key]
        target = next((t for t in targets if t.key == key), None)
        end = target.input.value if target is not None and target.on else start
        return dcf.linear_path(start, end, years)

    drivers = dcf.DriverPath(
        revenue_growth=dcf.growth_path(values["revenue_growth"], years, fade_to, hold=hold),
        operating_margin=path("operating_margin"),
        # Acquisition spend, when folded in, rides on top of every year's capex.
        capex_pct_revenue=tuple(c + acquisitions_share for c in path("capex_pct_revenue")),
        dna_pct_revenue=path("dna_pct_revenue"),
        effective_tax_rate=(values["effective_tax_rate"],) * years,
        nwc_pct_revenue=(values["nwc_pct_revenue"],) * years,
    )

    incremental, incremental_note = _incremental_return(
        base, drivers, wacc_result, elapsed, mid_year
    )

    # --- Base-period FCFF ------------------------------------------------
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
        dataset, base,
        tax_rate=etr if etr is not None else values["effective_tax_rate"],
        interest_basis=INTEREST_BASIS,
        delta_nwc=delta_nwc,
    )

    inputs = None if blocked else {
        "base": base,
        "drivers": drivers,
        "wacc": wacc_result,
        "terminal_growth": terminal_growth,
        "exit_multiple": exit_multiple,
        "elapsed": elapsed,
        "mid_year": mid_year,
        "ronic": ronic_field.value,
        "terminal_tax_rate": terminal_tax_rate,
        "ebita_proxy": ebita_proxy,
        "spread_table_as_of": wacc_lib.SYNTHETIC_RATINGS_AS_OF_DATE if from_table else None,
        "growth_fade_to": fade_to,
        "growth_hold": hold,
        "growth_basis": growth_basis,
        "assumed_margins": assumed_margins,
        "acquisitions_in_capex": bool(acquisitions and acquisitions.on),
        "shares": shares,
        "share_basis": share_basis,
        "include_sti": include_sti,
        "cash_floor": cash_floor,
        "non_operating": non_operating or None,
        "base_fcff": base_fcff,
        "headline_method": headline_method,
        "sbc": sbc,
        "etr": etr,
        "erp": erp_value,
        "beta": beta_value,
        "risk_free": rf_value,
        "beta_outcome": beta,
        "records": records,
        "periods_in_use": frozenset(periods_in_use),
        "tolerances": val.DEFAULT_TOLERANCES,
        "valuation_date": valuation_date,
    }

    return Assumptions(
        base=base, years=years, drivers_in=drivers_in, fade=fade, hold=hold, targets=targets,
        acquisitions=acquisitions, capex_estimate=capex_estimate,
        risk_free=rf_field, erp=erp_field, beta_frequency=frequency, blume=blume,
        beta=beta_field, beta_outcome=beta, cost_of_debt=kd_field, synthetic=synthetic,
        implied_cost_of_debt=_implied_cost_of_debt(dataset), marginal_tax=tax_field,
        share_basis_choice=share_choice, shares=shares, share_basis=share_basis,
        share_notes=share_notes, wacc=wacc_result, equity_value=equity_value,
        debt_value=debt_value, terminal_growth=growth_field, ronic_bases=ronic_bases,
        ronic_basis=ronic_basis, ronic=ronic_field, incremental_return=incremental,
        incremental_return_note=incremental_note, terminal_at_marginal=terminal_at_marginal,
        ebita_proxy=ebita_proxy, exit_multiple=exit_field, headline_method=headline_method,
        mid_year=mid_year, elapsed=elapsed, elapsed_note=elapsed_note,
        include_sti=include_sti, cash_floor=cash_floor, sbc=sbc_field,
        non_operating=non_operating, base_fcff=base_fcff, etr=etr,
        etr_years=min(5, len(etr_series)), delta_nwc_span=delta_nwc_span,
        drivers=drivers, records=tuple(records), inputs=inputs, blocked=blocked,
    )


def _incremental_return(
    base: dcf_model.BasePeriod,
    drivers: dcf.DriverPath,
    wacc_result: wacc_lib.WaccResult | None,
    elapsed: float,
    mid_year: bool,
) -> tuple[float | None, str]:
    """The explicit forecast's own return on capital, shown beside the RONIC input.

    Information, not a check: the excess return the forecast carries before the
    terminal year drops to its RONIC. A projection is only arithmetic, so
    building one to measure it costs nothing.
    """
    revenue, ebit = base.flows.get("revenue"), base.flows.get("ebit")
    if revenue is None or ebit is None or pd.isna(revenue) or pd.isna(ebit):
        return None, "not available — the base period has no revenue or operating income"
    if wacc_result is None:
        return None, "not available — no WACC"
    projection = dcf.project(
        float(revenue), drivers, wacc=wacc_result.wacc, elapsed=elapsed, mid_year=mid_year
    )
    value = dcf.incremental_return_on_capital(
        projection, float(ebit) * (1.0 - drivers.effective_tax_rate[0])
    )
    if value is None:
        return None, "not meaningful (no net reinvestment)"
    return value, ""


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_kwargs(inputs: Mapping) -> dict:
    """Every ``run_model`` argument the assumption set supplies, bar the drivers.

    One place builds it because three places call the model — the forward run,
    the reverse solve and the what-if price — and the reverse DCF has to invert
    exactly the model on screen: an argument added to one call and forgotten in
    another would solve a different model than the one displayed.
    """
    return dict(
        wacc_result=inputs["wacc"],
        terminal_growth=inputs["terminal_growth"],
        exit_multiple=inputs["exit_multiple"],
        elapsed=inputs["elapsed"],
        mid_year=inputs["mid_year"],
        shares=inputs["shares"],
        share_basis=inputs["share_basis"],
        include_short_term_investments=inputs["include_sti"],
        operating_cash_floor=inputs["cash_floor"],
        non_operating_assets=inputs["non_operating"],
        base_fcff=inputs["base_fcff"],
        headline_method=inputs["headline_method"],
        ronic=inputs["ronic"],
        terminal_tax_rate=inputs["terminal_tax_rate"],
        ebita_proxy=inputs["ebita_proxy"],
    )


def solve_price(
    dataset: dcf_model.CompanyDataset, inputs: Mapping, price: float | None, method: str
) -> reverse.ReverseDcfResult:
    """The year-1 growth that makes this run's value per share ``price``, under ``method``."""
    return dcf_model.solve_implied_growth(
        dataset,
        inputs["base"],
        target_price=price,
        drivers=inputs["drivers"],
        growth_fade_to=inputs["growth_fade_to"],
        growth_hold=inputs["growth_hold"],
        **{**run_kwargs(inputs), "headline_method": method},
    )


@dataclass(frozen=True)
class Evaluation:
    """One valued run, everything the tab draws below the assumption panel."""

    result: dcf_model.ModelResult
    reverse: dict[str, reverse.ReverseDcfResult]
    sensitivity: dcf_model.SensitivityGrid
    issues: list[val.Issue]
    check_results: list[val.CheckResult]

    @property
    def blocking(self) -> list[val.Issue]:
        return [i for i in self.issues if i.severity is val.Severity.BLOCKING]


def evaluate(dataset: dcf_model.CompanyDataset, inputs: Mapping) -> Evaluation:
    """Values the run: the model, the reverse DCF both ways, the grid, the checks.

    Raises :class:`dcf.TerminalValueError` and :class:`dcf.ProjectionError`
    where the model cannot be built; every other failure is an issue or a
    status on a result. The reverse DCF is solved here rather than when it is
    drawn so it describes the same run as everything else, under both methods
    whichever is the headline.
    """
    kwargs = run_kwargs(inputs)
    result = dcf_model.run_model(dataset, inputs["base"], drivers=inputs["drivers"], **kwargs)
    reverse_results = {
        method: solve_price(dataset, inputs, dataset.last_price, method)
        for method in REVERSE_METHODS
    }
    # The range matters more than the point estimate. Every cell is the same
    # model with the discount rate and the headline's terminal input moved.
    sensitivity = dcf_model.sensitivity_grid(
        dataset,
        inputs["base"],
        wacc_result=inputs["wacc"],
        terminal_growth=inputs["terminal_growth"],
        exit_multiple=inputs["exit_multiple"],
        drivers=inputs["drivers"],
        **{
            key: value for key, value in kwargs.items()
            if key not in {"wacc_result", "terminal_growth", "exit_multiple"}
        },
    )
    context = dcf_model.build_validation_context(
        dataset,
        base=inputs["base"],
        base_fcff=inputs["base_fcff"],
        valuation_date=inputs["valuation_date"],
        effective_tax_rate=inputs["etr"],
        beta=inputs["beta_outcome"],
        equity_risk_premium=inputs["erp"],
        result=result,
        terminal_growth=inputs["terminal_growth"],
        risk_free_rate=inputs["risk_free"],
        sbc=inputs["sbc"],
        periods_in_use=inputs["periods_in_use"],
        drivers=inputs["drivers"],
        beta_in_use=inputs["beta"],
        acquisitions_in_capex=inputs["acquisitions_in_capex"],
        growth_basis=inputs["growth_basis"],
        assumed_margins=inputs["assumed_margins"],
        spread_table_as_of=inputs["spread_table_as_of"],
        implied_growth={method: outcome.growth for method, outcome in reverse_results.items()},
    )
    issues, check_results = val.run_all(context, inputs["tolerances"])
    return Evaluation(result, reverse_results, sensitivity, issues, check_results)


# ---------------------------------------------------------------------------
# Reading the reverse DCF
# ---------------------------------------------------------------------------


def reverse_verdict(result: reverse.ReverseDcfResult) -> tuple[str, str]:
    """The banner for a solve outcome, as ``(level, markdown)``; ``("", "")`` needs none.

    Every status has one, so adding a status without a message fails a test
    rather than drawing a blank panel.
    """
    status = result.status

    if status is reverse.Status.SOLVED:
        if result.growth_creates_value is False:
            return "warning", (
                f"**Growth destroys value in this model.** At the implied rate, one more "
                f"percentage point of revenue growth *lowers* value per share by "
                f"**{abs(result.value_per_growth_point):,.2f}**. A *higher* market price "
                f"therefore implies a *lower* growth rate — the usual reading of this number "
                f"is inverted. This happens when each incremental dollar of revenue consumes "
                f"more working capital and capital expenditure than the margin it earns."
            )
        if result.plausible is False:
            return "warning", (
                f"**{format_pct(result.growth)} a year is outside anything a company sustains for "
                f"long.** Read this as *the price is not explained by revenue growth* rather "
                f"than as a forecast — the difference is more likely to sit in the margin, "
                f"the reinvestment rates, or the discount rate."
            )
        return "", ""

    if status is reverse.Status.MULTIPLE_SOLUTIONS:
        return "warning", f"**No single implied rate.** {result.reason}"

    if status is reverse.Status.NOT_EVALUABLE:
        return "error", (
            f"{result.reason}\n\nThis is not a failed search. The terminal growth rate and "
            f"the discount rate are held fixed while revenue growth is solved for, so this "
            f"fails identically at every growth rate — on the first evaluation or not at all. "
            f"Both terminal value methods are always computed, so a Gordon growth input can "
            f"stop an exit-multiple model."
        )

    if status is reverse.Status.NO_TARGET:
        return "info", f"**No market price to invert.** {result.reason}"

    if status in {reverse.Status.TARGET_ABOVE_RANGE, reverse.Status.TARGET_BELOW_RANGE}:
        return "info", f"**No growth rate reproduces this price.** {result.reason}"

    if status is reverse.Status.DEGENERATE:
        return "info", f"**Nothing to solve for.** {result.reason}"

    if status is reverse.Status.RESIDUAL_TOO_LARGE:
        return "error", f"**The solved rate did not check out.** {result.reason}"

    if status is reverse.Status.SEARCH_BROKEN:
        return "error", f"**The search could not complete.** {result.reason}"

    raise AssertionError(f"No message written for reverse DCF status {status!r}.")


def implied_revenue_path(
    base_revenue: float,
    growth: float,
    years: int,
    fade_to: float | None = None,
    hold: int = 1,
) -> list[float]:
    """Revenue compounding from the base period along the forecast's growth shape."""
    path, revenue = [], base_revenue
    for rate in dcf.growth_path(growth, years, fade_to, hold=hold):
        revenue *= 1.0 + rate
        path.append(revenue)
    return path


# ---------------------------------------------------------------------------
# The untouched value, for the summary chart's line
# ---------------------------------------------------------------------------


def default_value(
    dataset: dcf_model.CompanyDataset,
    *,
    valuation_date: date,
    risk_free: Sourced,
    erp: Sourced,
    beta: dcf_model.BetaOutcome,
) -> dcf_model.ModelResult:
    """The untouched panel's headline, run and checked as the tab would.

    Raises :class:`dcf_model.DefaultUnavailable` wherever the tab would draw a
    blocking screen or ask for a number instead — so a caller plotting past
    values leaves a gap there, never a zero.
    """
    assumptions = resolve(
        dataset, valuation_date=valuation_date, risk_free=risk_free, erp=erp, beta=beta,
        strict=True,
    )
    inputs = assumptions.inputs
    try:
        result = dcf_model.run_model(
            dataset, inputs["base"], drivers=inputs["drivers"], **run_kwargs(inputs)
        )
    except (dcf.TerminalValueError, dcf.ProjectionError) as error:
        raise dcf_model.DefaultUnavailable(str(error)) from None
    context = dcf_model.build_validation_context(
        dataset,
        base=inputs["base"],
        base_fcff=inputs["base_fcff"],
        valuation_date=valuation_date,
        effective_tax_rate=inputs["etr"],
        beta=beta,
        equity_risk_premium=inputs["erp"],
        result=result,
        terminal_growth=inputs["terminal_growth"],
        risk_free_rate=inputs["risk_free"],
        sbc=inputs["sbc"],
        periods_in_use=inputs["periods_in_use"],
        drivers=inputs["drivers"],
        beta_in_use=inputs["beta"],
        growth_basis=inputs["growth_basis"],
        assumed_margins=inputs["assumed_margins"],
        spread_table_as_of=inputs["spread_table_as_of"],
    )
    issues, _ = val.run_all(context, val.DEFAULT_TOLERANCES)
    blocking = [i for i in issues if i.severity is val.Severity.BLOCKING]
    if blocking:
        raise dcf_model.DefaultUnavailable(f"{blocking[0].check}: {blocking[0].message}")
    return result


def default_value_on(ticker: str, as_of: date | None = None) -> dcf_model.ModelResult:
    """The DCF tab's untouched headline, as the company stood on ``as_of``.

    ``None`` is today, and reproduces the tab: the restated statements, today's
    rates. A past date reads the as-reported statements filed by then, the
    10-year yield and exchange rate of that day, and a beta regressed to it.
    Two inputs have no history here and are today's: Damodaran's implied ERP,
    which is scraped from his home page, and the synthetic-rating spread table,
    which is hard-coded and refreshed each January.

    Raises what :func:`dcf_model.load_company` raises — a bank, an
    unconvertible currency, an unresolvable schema — and
    :class:`dcf_model.DefaultUnavailable`.
    """
    if as_of is None:
        dataset = dcf_model.load_company(ticker)
        risk_free = market_data.fetch_10y_treasury_rate()
        valuation_date = date.today()
    else:
        dataset = dcf_model.load_company(ticker, asreported=True, as_of=as_of)
        risk_free = market_data.fetch_10y_treasury_rate_on(as_of)
        valuation_date = as_of
    if dataset.annual.empty:
        raise dcf_model.DefaultUnavailable("Nothing had been filed yet")
    frequency, blume, years = beta_settings(Overrides())
    beta = dcf_model.regress_company_beta(
        dataset, frequency=frequency, blume=blume, years=years, as_of=as_of
    )
    return default_value(
        dataset, valuation_date=valuation_date, risk_free=risk_free,
        erp=market_data.fetch_erp(), beta=beta,
    )


# ---------------------------------------------------------------------------
# What the tab reads off a valued run
# ---------------------------------------------------------------------------


def cross_implied(terminal: dcf.TerminalValue) -> str:
    """Restates one terminal value method in the other method's units.

    Gordon growth produces a terminal value, which divided by terminal EBITDA is
    an exit multiple. An exit multiple produces a terminal value, which says
    what perpetual growth it needs even if growth were free, and what return on
    new capital it needs at the terminal growth rate. Each row therefore reports
    a different quantity, and the label says which.
    """
    if terminal.implied_exit_multiple is not None:
        return f"an exit multiple of {terminal.implied_exit_multiple:,.2f}×"
    if terminal.implied_growth is not None:
        text = f"perpetual growth of at least {terminal.implied_growth:.2%}"
        if terminal.implied_ronic is not None:
            text += f"; RONIC of {terminal.implied_ronic:.1%} at the terminal rate"
        elif terminal.terminal_year is not None and terminal.terminal_year.growth > 0:
            text += "; no return on new capital reaches it at the terminal rate"
        return text
    return "—"


@dataclass(frozen=True)
class RevenuePath:
    """One line of the reverse DCF's chart, anchored at the base period."""

    name: str
    #: ``price`` (what the market implies), ``forecast`` or ``history``.
    kind: str
    method: str | None
    rate: float | None
    values: tuple[float, ...]


@dataclass(frozen=True)
class RevenueChart:
    """Reported revenue, then every path the forecast years could take from the base."""

    history_labels: tuple[str, ...]
    #: Reported revenue converted at today's spot rate, like the forecast, so
    #: both sit on one axis without restating history at historical rates.
    reported: tuple[float | None, ...]
    anchor: str
    base_revenue: float
    forward_labels: tuple[str, ...]
    paths: tuple[RevenuePath, ...]


def revenue_chart(
    dataset: dcf_model.CompanyDataset,
    result: dcf_model.ModelResult,
    reverse_results: Mapping[str, reverse.ReverseDcfResult],
    history: fin.GrowthHistory,
    inputs: Mapping,
) -> RevenueChart | None:
    """The paths the price (both ways), the forecast and history imply.

    Forecast years are labelled by offset, not by fiscal year: each is the
    twelve months after the base period, which need not end on a fiscal year.
    Where a solve has no single answer every root is drawn, because losing the
    market's line is the finding and losing the chart would hide it.
    """
    if "revenue" not in dataset.annual.columns:
        return None
    projected = tuple(row.revenue for row in result.projection.rows)
    years = len(projected)
    base_revenue = result.projection.base_revenue
    fade_to, hold = inputs.get("growth_fade_to"), inputs.get("growth_hold", 1)

    paths: list[RevenuePath] = []
    for method in REVERSE_METHODS:
        outcome = reverse_results.get(method)
        if outcome is None:
            continue
        roots = [outcome.growth] if outcome.status is reverse.Status.SOLVED else outcome.roots
        for root in roots:
            paths.append(RevenuePath(
                f"Price, {REVERSE_LABELS[method].lower()} ({format_pct(root)})", "price", method,
                root, tuple(implied_revenue_path(base_revenue, root, years, fade_to, hold)),
            ))
    paths.append(RevenuePath("Your forecast", "forecast", None, None, projected))
    if history.ok:
        paths.append(RevenuePath(
            f"History repeating ({format_pct(history.cagr)} realised)", "history", None,
            history.cagr, tuple(implied_revenue_path(base_revenue, history.cagr, years)),
        ))

    reported = dataset.annual["revenue"] * dataset.fx_rate_value
    return RevenueChart(
        history_labels=tuple(str(i) for i in dataset.annual.index),
        reported=tuple(None if pd.isna(v) else float(v) for v in reported),
        anchor=result.base.label,
        base_revenue=base_revenue,
        forward_labels=tuple(f"+{i}yr" for i in range(1, years + 1)),
        paths=tuple(paths),
    )


def scan_rows(
    reverse_results: Mapping[str, reverse.ReverseDcfResult],
) -> list[tuple[float, dict[str, float | None]]]:
    """Value per share across the growth range, from the scans the solvers already ran.

    Only rates inside the band a company might sustain, so the table is the
    part of the curve a reader would underwrite. Every crossing of the market
    price in a column is a rate the price is consistent with.
    """
    samples = {method: dict(reverse_results[method].samples) for method in REVERSE_METHODS}
    low, high = reverse.PLAUSIBLE_GROWTH
    growths = sorted(
        g for g in set().union(*(set(v) for v in samples.values())) if low <= g <= high
    )
    return [(g, {method: samples[method].get(g) for method in REVERSE_METHODS}) for g in growths]


@dataclass(frozen=True)
class Filing:
    fiscal_year: str
    period: str
    report_date: date | None
    publish_date: date | None
    restated: bool
    link: str | None


def filings(dataset: dcf_model.CompanyDataset) -> list[Filing]:
    """Which filing each fiscal year came from, and whether it was restated.

    The only path from a valuation back to the documents behind it — 📄
    Financials lists the same dates but not the filings. Read off the income
    statement, though any frame would do: the client partitions one SF1 row per
    period into ``PL``/``BS``/``CF``, so all three carry identical metadata.
    """
    frame = dataset.statements_annual.frame("PL")
    if frame.empty or "Fiscal Year" not in frame.columns:
        return []

    def day(value) -> date | None:
        return None if value is None or pd.isna(value) else pd.Timestamp(value).date()

    rows = []
    for _, record in frame.iterrows():
        year = record["Fiscal Year"]
        if pd.isna(year):
            continue
        rows.append(Filing(
            fiscal_year=str(int(year)),
            period=str(record.get("Fiscal Period", "") or ""),
            report_date=day(record.get("Report Date")),
            publish_date=day(record.get("Publish Date")),
            restated=bool(record.get("Restated")),
            link=dataset.filing_link(int(year)),
        ))
    return rows
