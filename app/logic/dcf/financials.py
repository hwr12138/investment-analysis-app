"""Statement normalisation, TTM assembly, and historical ratio derivation.

Turns the raw Sharadar frames into a concept-keyed table the rest of the engine
can read without knowing a single Sharadar indicator name, and derives the
driver ratios the forecast is pre-populated from.

Missing data stays missing. Every series here can contain ``NaN``, and every
consumer is expected to exclude those periods and report how many periods it
actually used. Nothing is zero-filled, interpolated, or carried forward.

Sign conventions. Costs and cash outflows are negative throughout this module.
Sharadar publishes expense lines as positive magnitudes, so
``vocabulary.SIGN_FLIPPED`` negates them at ingest and this module sees the
same convention it always has:

* Income tax is negative for an expense and positive for a benefit. Tax expense
  is therefore its negation, and a genuine benefit stays negative rather than
  being flipped by an ``abs()``.
* Interest expense is negative; interest expense as a positive number is its
  negation.
* Capex is negative for cash spent — Sharadar signs it that way natively, so it
  is *not* among the flipped indicators.

One caveat that does not exist upstream of a Sharadar feed: Sharadar writes 0,
not null, for a line item absent from the filing. Those zeros arrive here as
genuine zeros and cannot be told apart from a real nil, which is why
``validation.check_zero_filled_indicators`` surfaces them to the user rather
than this module guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from app.data.sharadar.vocabulary import StatementSet
from app.logic.dcf.schema import SchemaMap

#: Driver ratios pre-populated from history, in the order the UI shows them.
DRIVER_KEYS = (
    "revenue_growth",
    "operating_margin",
    "effective_tax_rate",
    "capex_pct_revenue",
    "dna_pct_revenue",
    "nwc_pct_revenue",
)

DRIVER_LABELS = {
    "revenue_growth": "Revenue growth",
    "operating_margin": "Operating margin (EBIT ÷ revenue)",
    "effective_tax_rate": "Effective tax rate",
    "capex_pct_revenue": "CapEx as % of revenue",
    "dna_pct_revenue": "D&A as % of revenue",
    "nwc_pct_revenue": "NWC as % of revenue",
}


@dataclass(frozen=True)
class CapexResult:
    """Capex plus how it had to be obtained.

    Sharadar publishes a single capital expenditure line covering the
    acquisition and disposal of long-lived assets, so there is no gross-versus-
    net choice to label. The origin field is kept because a company can still
    have no capex line at all, and because the UI treats an approximation
    differently from a reported figure.
    """

    values: pd.Series
    origin: str  # "reported" | "derived" | "estimate"
    formula: str
    detail: str


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_META_OUT = ("fiscal_period", "report_date", "publish_date", "source_link", "restated",
             "value_check")


def _join_key(frame: pd.DataFrame, annual: bool) -> pd.DataFrame:
    """Builds the key the four statements are merged on.

    Annual data keys on the fiscal year alone rather than on the reported
    period label, so a provider that labels an annual balance sheet ``Q4`` while
    labelling the income statement ``FY`` still merges into one row instead of
    two half-populated ones.
    """
    out = pd.DataFrame(index=pd.RangeIndex(len(frame)))
    out["fyear"] = pd.to_numeric(frame["Fiscal Year"], errors="coerce").astype("Int64").to_numpy()
    if annual:
        out["period"] = "FY"
    else:
        out["period"] = frame["Fiscal Period"].to_numpy()
    return out


def _statement_concepts(
    frame: pd.DataFrame, schema: SchemaMap, statement: str, annual: bool
) -> pd.DataFrame:
    """Extracts the resolved concepts for one statement, keyed for merging."""
    from app.logic.dcf.schema import CONCEPTS

    if frame.empty or not {"Fiscal Year", "Fiscal Period"}.issubset(frame.columns):
        return pd.DataFrame()

    keys = [
        c.key
        for c in CONCEPTS
        if c.statement == statement
        and schema.has(c.key)
        and schema.column(c.key) in frame.columns
    ]
    out = _join_key(frame, annual)
    for key in keys:
        out[key] = pd.to_numeric(frame[schema.column(key)], errors="coerce").to_numpy()
    return out


def _statement_meta(frame: pd.DataFrame, annual: bool) -> pd.DataFrame:
    if frame.empty or "Fiscal Year" not in frame.columns:
        return pd.DataFrame()
    out = _join_key(frame, annual)
    out["fiscal_period"] = frame["Fiscal Period"].to_numpy()
    for target, column, empty in (
        ("report_date", "Report Date", pd.NaT),
        ("publish_date", "Publish Date", pd.NaT),
        ("source_link", "Source", None),
        ("restated", "Restated", None),
        ("value_check", "Value Check", None),
    ):
        if column in frame.columns:
            out[target] = frame[column].to_numpy()
        else:
            out[target] = empty
    return out


def build_concept_table(
    statements: StatementSet, schema: SchemaMap
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merges PL/BS/CF/DERIVED into one concept-keyed table.

    Returns ``(values, meta)``, indexed by fiscal year for annual data and by
    ``"FY2026 Q2"``-style labels for quarterly. Rows carry ``NaN`` wherever a
    line item was not reported.
    """
    annual = statements.period.strip().upper() == "FY"

    parts = [
        _statement_concepts(statements.frame(name), schema, name, annual)
        for name in ("PL", "BS", "CF", "DERIVED")
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(), pd.DataFrame()

    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on=["fyear", "period"], how="outer")

    # Metadata is taken from the income statement where present and filled from
    # the cash flow and balance sheet, which carry the same report dates.
    meta = pd.DataFrame(columns=["fyear", "period", *_META_OUT])
    for name in ("PL", "CF", "BS"):
        part = _statement_meta(statements.frame(name), annual)
        if part.empty:
            continue
        if meta.empty:
            meta = part
        else:
            meta = meta.set_index(["fyear", "period"]).combine_first(
                part.set_index(["fyear", "period"])
            ).reset_index()

    if not meta.empty:
        meta = meta.drop_duplicates(subset=["fyear", "period"], keep="first")
        merged = merged.merge(meta, on=["fyear", "period"], how="left")

    sort_cols = [c for c in ("report_date", "fyear") if c in merged.columns]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)

    labels = [
        int(row.fyear) if annual else f"FY{int(row.fyear)} {row.period}"
        for row in merged.itertuples()
    ]
    index = pd.Index(labels, name="period")

    meta_out = merged[[c for c in _META_OUT if c in merged.columns]].copy()
    meta_out.index = index

    value_cols = [c for c in merged.columns if c not in ("fyear", "period", *_META_OUT)]
    values = merged[value_cols].copy()
    values.index = index
    return values, meta_out


# ---------------------------------------------------------------------------
# Derived line items
# ---------------------------------------------------------------------------


def tax_expense(values: pd.DataFrame) -> pd.Series:
    """Tax expense as a positive number; a genuine tax benefit stays negative."""
    if "tax_expense" not in values.columns:
        return pd.Series(np.nan, index=values.index)
    return -values["tax_expense"]


def interest_expense(values: pd.DataFrame, prefer: str = "gross") -> pd.Series:
    """Interest expense as a positive number.

    Sharadar publishes a single interest line rather than a gross/net pair, so
    ``prefer`` no longer changes the answer; it is retained because the caller
    passes it and a future provider may split the two again. Net interest
    income at a cash-rich company comes back negative, which is real and is
    surfaced rather than flattened by an ``abs()``.
    """
    order = (
        ("interest_expense_net", "interest_expense_gross")
        if prefer == "net"
        else ("interest_expense_gross", "interest_expense_net")
    )
    result = pd.Series(np.nan, index=values.index)
    for key in order:
        if key in values.columns:
            result = result.fillna(-values[key])
    return result


def capex(values: pd.DataFrame) -> CapexResult:
    """Capital expenditure as a positive number, with its provenance.

    Sharadar's capital expenditure line is already net of disposals and already
    signed negative for cash spent, so the only decision left is whether the
    company has one at all.
    """
    reported = values.get("capex_acquisition")

    if reported is not None and reported.notna().any():
        return CapexResult(
            -reported,
            "reported",
            "CapEx = −(Capital Expenditure)",
            "Acquisitions and disposals of fixed and intangible assets, as "
            "reported by Sharadar.",
        )

    return CapexResult(
        pd.Series(np.nan, index=values.index),
        "estimate",
        "CapEx — not reported",
        "Sharadar reports no capital expenditure line for this company in any "
        "period.",
    )


def acquisition_spend(values: pd.DataFrame) -> pd.Series:
    """Cash spent on business acquisitions, net of disposals, as a positive number.

    Sharadar signs the line negative for cash spent, like capex. ``NaN`` where
    it is not reported — missing is not the same as a company that bought
    nothing.
    """
    reported = values.get("acquisitions")
    if reported is None:
        return pd.Series(np.nan, index=values.index)
    return -reported


def total_debt(values: pd.DataFrame) -> pd.Series:
    """Current debt + non-current debt.

    Sharadar folds short-term borrowings and the current portion of long-term
    debt into one current-debt line, so two components rather than three.
    Companies that do not split the maturities leave one component null, so the
    components are added with a fill. A row where *every* component is null
    stays ``NaN`` — that is "not reported", not "no debt".
    """
    components = ["short_term_debt", "long_term_debt"]
    present = [c for c in components if c in values.columns]
    if not present:
        return pd.Series(np.nan, index=values.index)

    frame = values[present]
    summed = frame.sum(axis=1, skipna=True)
    return summed.where(frame.notna().any(axis=1))


def effective_tax_rate(values: pd.DataFrame) -> pd.Series:
    """Tax expense ÷ pretax income, undefined where pretax income is not positive.

    A negative or zero pretax income makes the ratio meaningless — Ford FY2025
    reports a pretax loss and a tax benefit, which would produce a nonsensical
    positive "rate". Those years come back ``NaN`` and are excluded upstream.
    """
    if "pretax_income" not in values.columns:
        return pd.Series(np.nan, index=values.index)
    pretax = values["pretax_income"]
    rate = tax_expense(values) / pretax
    return rate.where(pretax > 0)


def net_working_capital(values: pd.DataFrame) -> pd.Series:
    """(Current assets − cash − STI) − (current liabilities − current debt).

    Cash and debt are stripped out because they are financing, not operations.
    Components that were not reported are treated as absent from the subtraction
    rather than as zero balances — but the two required totals must be present
    or the result is ``NaN``.
    """
    required = ("total_current_assets", "total_current_liabilities")
    if any(k not in values.columns for k in required):
        return pd.Series(np.nan, index=values.index)

    def _optional(key: str) -> pd.Series:
        if key not in values.columns:
            return pd.Series(0.0, index=values.index)
        return values[key].fillna(0.0)

    operating_assets = (
        values["total_current_assets"] - _optional("cash") - _optional("short_term_investments")
    )
    operating_liabilities = (
        values["total_current_liabilities"] - _optional("short_term_debt")
    )
    return operating_assets - operating_liabilities


def delta_nwc(values: pd.DataFrame) -> pd.Series:
    """Year-on-year change in net working capital.

    The earliest period has no prior balance sheet to difference against, so it
    is ``NaN`` — never zero. Zero would read as "working capital did not move",
    which is a different and false claim.
    """
    return net_working_capital(values).diff()


# ---------------------------------------------------------------------------
# Historical driver ratios
# ---------------------------------------------------------------------------


def driver_ratios(values: pd.DataFrame) -> pd.DataFrame:
    """The six forecast drivers computed for every period available.

    Periods missing a component come back ``NaN`` and are excluded downstream;
    the count of periods actually used is reported alongside every default.
    """
    revenue = values.get("revenue", pd.Series(np.nan, index=values.index))
    ebit = values.get("ebit", pd.Series(np.nan, index=values.index))
    dna = values.get("dna", pd.Series(np.nan, index=values.index))

    revenue_positive = revenue.where(revenue > 0)
    capex_values = capex(values).values

    ratios = pd.DataFrame(index=values.index)
    ratios["revenue_growth"] = revenue.pct_change()
    ratios["operating_margin"] = ebit / revenue_positive
    ratios["effective_tax_rate"] = effective_tax_rate(values)
    ratios["capex_pct_revenue"] = capex_values / revenue_positive
    ratios["dna_pct_revenue"] = dna / revenue_positive
    # A ratio of levels, not of year-on-year changes. ΔNWC ÷ Δrevenue divided
    # by a near-zero change in a flat year — DHR's FY2024 read 2,453% on a $15m
    # revenue move — and one such year dominated any average. The projection
    # applies this ratio to each year's change in revenue, which in steady state
    # is the same thing and never divides by a small number. Negative is
    # allowed: a business funded by its customers releases cash as it grows.
    ratios["nwc_pct_revenue"] = net_working_capital(values) / revenue_positive
    # Not a forecast driver on its own — see `ACQUISITIONS_KEY`.
    ratios[ACQUISITIONS_KEY] = acquisition_spend(values) / revenue_positive
    return ratios


#: Acquisition spend as a share of revenue. Kept beside the six drivers rather
#: than among them: it only reaches the forecast when the user folds it into
#: the capex driver, for a company whose reported growth was partly bought.
ACQUISITIONS_KEY = "acquisitions_pct_revenue"

#: Display labels for every column :func:`driver_ratios` returns.
RATIO_LABELS = {**DRIVER_LABELS, ACQUISITIONS_KEY: "Acquisitions as % of revenue"}


@dataclass(frozen=True)
class DriverDefault:
    """A pre-populated forecast driver and the basis it came from."""

    key: str
    value: float | None
    basis: str
    years_used: int
    periods: tuple[str, ...]
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None and not (
            isinstance(self.value, float) and np.isnan(self.value)
        )


#: Bases the user can pick between for every driver default.
BASES = ("Most recent year", "3-year average", "5-year average", "10-year average", "Median (all years)")

#: Bases for revenue growth, which alone among the drivers compounds. The
#: compound rate comes first because it is the one comparable to a forecast
#: that compounds: the arithmetic mean of yearly rates is biased upward
#: whenever growth was uneven (see :func:`revenue_growth_history`).
GROWTH_BASES = ("5-year CAGR", "3-year CAGR", "10-year CAGR") + BASES

_WINDOWS = {
    "3-year average": 3, "5-year average": 5, "10-year average": 10,
    "3-year CAGR": 3, "5-year CAGR": 5, "10-year CAGR": 10,
}


def driver_default(ratios: pd.DataFrame, key: str, basis: str) -> DriverDefault:
    """Computes one driver default on the requested basis.

    Years with a missing component are excluded rather than zero-filled, and the
    number of years actually used is returned so the UI can show it. If nothing
    usable remains, the value is ``None`` and the user must supply it.
    """
    if key not in ratios.columns:
        return DriverDefault(key, None, basis, 0, (), "Ratio could not be computed.")

    series = ratios[key].dropna()
    if series.empty:
        return DriverDefault(
            key, None, basis, 0, (),
            "No period had every component needed for this ratio.",
        )

    window = _WINDOWS.get(basis)
    if basis.endswith("CAGR") and key != "revenue_growth":
        return DriverDefault(
            key, None, basis, 0, (), "A compound rate only applies to revenue growth."
        )
    if basis == "Most recent year":
        used = series.iloc[-1:]
        value = float(used.iloc[0])
    elif basis == "Median (all years)":
        used = series
        value = float(series.median())
    elif basis.endswith("CAGR"):
        # The geometric mean of (1 + g): the one constant rate that compounds
        # to the same place the reported years did.
        used = series.iloc[-window:]
        growth_factors = 1.0 + used
        if (growth_factors <= 0).any():
            return DriverDefault(
                key, None, basis, len(used), tuple(str(p) for p in used.index),
                "A year in the window lost all its revenue, so no compound rate exists.",
            )
        value = float(np.prod(growth_factors) ** (1.0 / len(used)) - 1.0)
    elif window is not None:
        used = series.iloc[-window:]
        value = float(used.mean())
    else:
        used = series
        value = float(series.mean())

    detail = ""
    if window is not None and len(used) < window:
        detail = f"Only {len(used)} of the requested {window} years had usable data."

    return DriverDefault(
        key=key,
        value=value,
        basis=basis,
        years_used=len(used),
        periods=tuple(str(p) for p in used.index),
        detail=detail,
    )


#: The trailing bases. Neither is a history window: each reads the latest
#: trailing-twelve-month period, which is where the business is *now* — the
#: starting point a forecast for a company whose growth or margins are moving
#: fast should jump off from.
TTM_GROWTH_BASIS = "TTM year-over-year"
#: The base period's own ratio — margin, capex %, D&A %, NWC % — for a base
#: that is a trailing twelve months.
TTM_BASIS = "TTM"

#: How far either side of exactly a year the prior TTM period may end and
#: still count as "four quarters earlier" — 52/53-week fiscal calendars move
#: period ends by a few days a year.
_YEAR_WINDOW_DAYS = (345, 385)


def _ttm_pair(ttm: pd.DataFrame, ttm_meta: pd.DataFrame) -> tuple[pd.Series, pd.Series] | str:
    """The latest TTM period and the one ending about a year before it.

    Sharadar's TTM dimension steps a quarter at a time, so the pair spans the
    last eight quarters. Returns the reason instead when there is no such pair.
    """
    if ttm is None or ttm.empty or "revenue" not in ttm.columns or ttm_meta is None \
            or ttm_meta.empty or "report_date" not in ttm_meta.columns:
        return "Sharadar publishes no trailing-twelve-month series for this company."
    frame = pd.DataFrame(
        {
            "label": [str(label) for label in ttm.index],
            "revenue": ttm["revenue"].to_numpy(),
            "acquisitions": (
                ttm["acquisitions"].to_numpy() if "acquisitions" in ttm.columns
                else np.full(len(ttm), np.nan)
            ),
            "date": pd.to_datetime(
                ttm_meta["report_date"].reindex(ttm.index), errors="coerce"
            ).to_numpy(),
        }
    ).dropna(subset=["revenue", "date"]).sort_values("date")
    if len(frame) < 2:
        return "Fewer than two trailing-twelve-month periods are reported."
    latest = frame.iloc[-1]
    gaps = (latest["date"] - frame["date"]).dt.days
    low, high = _YEAR_WINDOW_DAYS
    prior = frame[(gaps >= low) & (gaps <= high)]
    if prior.empty:
        return (
            f"No trailing-twelve-month period ends about a year before "
            f"{latest['date'].date()}, so there is no like-for-like comparison."
        )
    return prior.iloc[-1], latest


def ttm_revenue_growth(ttm: pd.DataFrame, ttm_meta: pd.DataFrame) -> DriverDefault:
    """Latest TTM revenue ÷ the TTM revenue four quarters earlier, less one.

    Read off Sharadar's own trailing-twelve-month dimension, whose rows step a
    quarter at a time, so the comparison period is the one ending about a year
    before the latest. ``None`` with the reason when there is no TTM series (a
    foreign filer may publish none) or no period a year back.
    """
    basis = TTM_GROWTH_BASIS
    pair = _ttm_pair(ttm, ttm_meta)
    if isinstance(pair, str):
        return DriverDefault("revenue_growth", None, basis, 0, (), pair)
    prior, latest = pair
    if prior["revenue"] <= 0:
        return DriverDefault(
            "revenue_growth", None, basis, 0, (),
            "Revenue four quarters earlier was not positive.",
        )
    return DriverDefault(
        key="revenue_growth",
        value=float(latest["revenue"] / prior["revenue"] - 1.0),
        basis=basis,
        years_used=1,
        periods=(prior["label"], latest["label"]),
    )


#: The default year-1 growth basis: whichever of the trailing rate and the
#: five-year compound rate is lower.
LOWER_GROWTH_BASIS = "Lower of TTM YoY and 5-year CAGR"


def lower_of_growth(trailing: DriverDefault, compound: DriverDefault) -> DriverDefault:
    """The lower of TTM year-over-year growth and the five-year CAGR.

    The forecast holds year-1 growth for several years and then fades it over a
    ten-year horizon, so the starting rate compounds hard. A trailing rate far
    above the company's own five-year record — NVDA 83% against 67%, AVGO 49%
    against 22% — took NVDA's revenue to 70× today's by year 10. Starting from
    the lower of the two keeps a recent slowdown (the trailing rate is lower)
    and refuses to extrapolate a recent burst (the compound rate is lower).
    ``detail`` names which one was used and what the other read.

    Unavailable, with the reason, when either input is: the rule needs both,
    and the caller falls back explicitly rather than calling one of them "the
    lower".
    """
    basis = LOWER_GROWTH_BASIS
    for source in (trailing, compound):
        if not source.ok:
            return DriverDefault(
                "revenue_growth", None, basis, 0, (),
                f"{source.basis} is unavailable — {source.detail}",
            )
    chosen, other = (
        (trailing, compound) if trailing.value <= compound.value else (compound, trailing)
    )
    return DriverDefault(
        key="revenue_growth",
        value=chosen.value,
        basis=basis,
        years_used=chosen.years_used,
        periods=chosen.periods,
        detail=(
            f"Using {chosen.basis} ({chosen.value:.2%}), the lower; "
            f"{other.basis} reads {other.value:.2%}."
        ),
    )


def ttm_window_acquisitions(
    ttm: pd.DataFrame, ttm_meta: pd.DataFrame
) -> tuple[float, tuple[str, str]] | None:
    """Acquisition spend over the TTM growth comparison's window ÷ TTM revenue.

    The window is the two trailing-twelve-month periods the growth rate
    compares — the last eight quarters — so this says whether *that* growth
    rate may include bought revenue, unlike the five-year acquisitions check.
    ``None`` when either period's acquisitions line is not reported.
    """
    pair = _ttm_pair(ttm, ttm_meta)
    if isinstance(pair, str):
        return None
    prior, latest = pair
    if pd.isna(prior["acquisitions"]) or pd.isna(latest["acquisitions"]) or latest["revenue"] <= 0:
        return None
    spend = -(float(prior["acquisitions"]) + float(latest["acquisitions"]))
    return spend / float(latest["revenue"]), (prior["label"], latest["label"])


def base_period_driver(
    key: str, flows: pd.Series, stocks: pd.Series, *, is_ttm: bool, label: str
) -> DriverDefault:
    """One driver's ratio in the base period — the TTM basis.

    ``operating_margin``, ``capex_pct_revenue`` and ``dna_pct_revenue`` read the
    base period's flows over its revenue; ``nwc_pct_revenue`` reads the base
    balance sheet's working capital over the TTM revenue. Pairing a TTM margin
    with five-year-average capex and D&A mixes starting points: the margin
    already reflects today's depreciation, and a hyperscaler whose capex has
    doubled with an AI build-out would start its forecast spending half of what
    it spends now.

    Offered only when the base period *is* a trailing twelve months; an
    annual-only filer's base is its last fiscal year, which the "Most recent
    year" basis already covers. ``None`` with the reason otherwise.
    """
    basis = TTM_BASIS
    if not is_ttm:
        return DriverDefault(
            key, None, basis, 0, (),
            "The base period is a fiscal year, not a trailing twelve months.",
        )
    revenue = flows.get("revenue")
    if revenue is None or pd.isna(revenue) or revenue <= 0:
        return DriverDefault(key, None, basis, 0, (), "The base period has no positive revenue.")
    frame = pd.DataFrame([flows])
    if key == "operating_margin":
        numerator = flows.get("ebit")
    elif key == "capex_pct_revenue":
        numerator = capex(frame).values.iloc[0]
    elif key == "dna_pct_revenue":
        numerator = flows.get("dna")
    elif key == "nwc_pct_revenue":
        numerator = net_working_capital(pd.DataFrame([stocks])).iloc[0]
    else:
        return DriverDefault(key, None, basis, 0, (), "No trailing basis for this driver.")
    if numerator is None or pd.isna(numerator):
        return DriverDefault(
            key, None, basis, 0, (),
            f"The base period does not report the {DRIVER_LABELS[key].split(' ')[0]} line.",
        )
    return DriverDefault(key, float(numerator) / float(revenue), basis, 1, (label,))


# ---------------------------------------------------------------------------
# Realised growth, for benchmarking a forecast against
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrowthHistory:
    """Realised revenue growth on the two bases that are not the same thing.

    ``cagr`` is the single constant rate that, compounded across the span,
    reproduces the endpoint. ``arithmetic_mean`` is the average of the
    year-on-year rates. Both are returned because both get quoted, and the
    difference between them is not a rounding detail — it is Jensen's
    inequality, and it grows with how uneven the growth was.
    """

    cagr: float | None
    arithmetic_mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    first_period: str
    last_period: str
    first_value: float | None
    last_value: float | None
    intervals: int      # compounding intervals across the span = periods - 1
    observations: int   # usable year-on-year growth rates
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.cagr is not None


def revenue_growth_history(values: pd.DataFrame) -> GrowthHistory:
    """Compound and arithmetic realised revenue growth.

    A flat forecast is comparable to the **compound** rate: both are one
    constant rate applied across every period. The arithmetic mean of yearly
    rates is a different quantity, biased upward whenever growth was uneven, and
    equal to the compound rate only when every year was identical — revenue of
    100 → 200 → 100 averages +25% a year and compounds at 0%. Quoting the
    average as what a company "has grown at" overstates it by that gap.

    The span is the longest contiguous **trailing** run of periods with reported,
    positive revenue. A gap ends the span rather than being bridged: a CAGR taken
    across missing years silently asserts those years compounded like the
    reported ones. Where no compound rate can be formed the value is ``None``
    with a reason, never a substituted figure.
    """
    empty = GrowthHistory(
        cagr=None, arithmetic_mean=None, median=None, minimum=None, maximum=None,
        first_period="", last_period="", first_value=None, last_value=None,
        intervals=0, observations=0,
    )

    if values.empty or "revenue" not in values.columns:
        return replace(empty, detail="No revenue series was available.")

    revenue = values["revenue"]
    usable = revenue.notna() & (revenue > 0)
    if not usable.any():
        return replace(empty, detail="No period reported positive revenue.")

    # Walk back from the most recent period until the run breaks. `usable` is in
    # statement order, so the trailing run is the one ending at the last period.
    start = len(usable) - 1
    while start > 0 and bool(usable.iloc[start - 1]):
        start -= 1
    span = revenue.iloc[start:]

    first_period, last_period = str(span.index[0]), str(span.index[-1])
    first_value, last_value = float(span.iloc[0]), float(span.iloc[-1])
    intervals = len(span) - 1

    growth = span.pct_change().dropna()
    observations = len(growth)

    common = dict(
        first_period=first_period,
        last_period=last_period,
        first_value=first_value,
        last_value=last_value,
        intervals=intervals,
        observations=observations,
        arithmetic_mean=float(growth.mean()) if observations else None,
        median=float(growth.median()) if observations else None,
        minimum=float(growth.min()) if observations else None,
        maximum=float(growth.max()) if observations else None,
    )

    if intervals < 1:
        return GrowthHistory(
            cagr=None, **common,
            detail=(
                f"Only one period of positive revenue ({last_period}) was usable, "
                "so there is no interval to compound across."
            ),
        )

    detail = ""
    excluded = len(revenue) - len(span)
    if excluded:
        detail = (
            f"{excluded} earlier period(s) excluded: revenue was missing or not "
            f"positive, and the span was cut there rather than bridged."
        )

    return GrowthHistory(
        cagr=(last_value / first_value) ** (1.0 / intervals) - 1.0,
        **common,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Trailing twelve months
# ---------------------------------------------------------------------------


def manual_ttm(quarterly: pd.DataFrame, flow_keys: tuple[str, ...]) -> pd.Series | None:
    """Sums the last four quarters by hand, for cross-checking the API's TTM.

    Returns ``None`` if fewer than four quarters are available — an incomplete
    sum compared against a real TTM would manufacture a divergence that isn't
    there.
    """
    if quarterly.empty or len(quarterly) < 4:
        return None
    window = quarterly.iloc[-4:]
    present = [k for k in flow_keys if k in window.columns]
    summed = window[present].sum(axis=0, skipna=False)
    return summed


#: Line items that accumulate over a period and may be summed across quarters.
FLOW_KEYS = (
    "revenue",
    "cost_of_revenue",
    "gross_profit",
    "operating_expenses",
    "ebit",
    "derived_ebit_reported",
    "interest_expense_gross",
    "pretax_income",
    "tax_expense",
    "net_income",
    "dna",
    "sbc",
    "cfo",
    "capex_acquisition",
    "cfi",
    "cff",
    "fx_effect",
    "net_change_in_cash",
)

#: Balance-sheet items, which are point-in-time and must never be summed.
STOCK_KEYS = (
    "cash",
    "short_term_investments",
    "total_current_assets",
    "total_current_liabilities",
    "short_term_debt",
    "long_term_debt",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "derived_total_debt",
    "minority_interest",
    "long_term_investments",
)
