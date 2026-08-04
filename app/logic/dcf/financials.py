"""Statement normalisation, TTM assembly, and historical ratio derivation.

Turns the raw SimFin frames into a concept-keyed table the rest of the engine
can read without knowing a single SimFin column name, and derives the driver
ratios the forecast is pre-populated from.

Missing data stays missing. Every series here can contain ``NaN``, and every
consumer is expected to exclude those periods and report how many periods it
actually used. Nothing is zero-filled, interpolated, or carried forward.

Sign conventions, confirmed against live responses:

* ``Income Tax (Expense) Benefit, net`` is negative for an expense and positive
  for a benefit (Ford FY2025: +3,668M in a loss year). Tax expense is therefore
  its negation, and a genuine benefit stays negative rather than being flipped
  by an ``abs()``.
* ``Interest Expense, net`` is negative (KO FY2025: -868M); interest expense is
  its negation.
* Capex lines are negative for cash spent (KO FY2025: acquisition -2,112M =
  net change -2,099M less disposals +13M), so capex is their negation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np
import pandas as pd

from app.data.simfin_client import StatementSet
from app.logic.dcf.schema import SchemaMap

#: Driver ratios pre-populated from history, in the order the UI shows them.
DRIVER_KEYS = (
    "revenue_growth",
    "operating_margin",
    "effective_tax_rate",
    "capex_pct_revenue",
    "dna_pct_revenue",
    "dnwc_pct_drevenue",
)

DRIVER_LABELS = {
    "revenue_growth": "Revenue growth",
    "operating_margin": "Operating margin (EBIT ÷ revenue)",
    "effective_tax_rate": "Effective tax rate",
    "capex_pct_revenue": "CapEx as % of revenue",
    "dna_pct_revenue": "D&A as % of revenue",
    "dnwc_pct_drevenue": "ΔNWC as % of change in revenue",
}


@dataclass(frozen=True)
class CapexResult:
    """Capex plus how it had to be obtained.

    SimFin frequently leaves the gross acquisition line empty and populates only
    the net change in fixed assets, which nets out disposals. That is an
    approximation, so it is labelled rather than passed off as reported.
    """

    values: pd.Series
    origin: str  # "reported" | "derived" | "estimate"
    formula: str
    detail: str


@dataclass
class CompanyFinancials:
    """Everything the model needs about one company, in concept space."""

    ticker: str
    name: str
    currency: str
    template: str
    simfin_id: int
    schema: SchemaMap
    fiscal_year_end_month: int | None

    annual: pd.DataFrame          # index: fiscal year, columns: concept keys
    annual_meta: pd.DataFrame     # index: fiscal year, report/publish dates, filing link
    quarterly: pd.DataFrame
    quarterly_meta: pd.DataFrame
    ttm: pd.Series | None = None
    ttm_meta: dict = field(default_factory=dict)

    asreported: bool = False
    sector_name: str | None = None
    industry_name: str | None = None
    sector_code: int | None = None

    @property
    def years(self) -> list[int]:
        return [int(y) for y in self.annual.index.tolist()]

    @property
    def latest_year(self) -> int | None:
        return self.years[-1] if self.years else None

    def latest_balance_sheet_date(self) -> date | None:
        """Report date of the most recent balance sheet, quarterly if available."""
        if not self.quarterly_meta.empty:
            value = self.quarterly_meta["report_date"].iloc[-1]
            return None if pd.isna(value) else pd.Timestamp(value).date()
        if not self.annual_meta.empty:
            value = self.annual_meta["report_date"].iloc[-1]
            return None if pd.isna(value) else pd.Timestamp(value).date()
        return None

    def latest_stock_items(self) -> pd.Series:
        """Most recent balance-sheet row — quarterly if we have one, else annual.

        Balance-sheet items are stocks, not flows, so the newest observation is
        used as-is and labelled with its as-of date rather than averaged.
        """
        if not self.quarterly.empty:
            return self.quarterly.iloc[-1]
        if not self.annual.empty:
            return self.annual.iloc[-1]
        return pd.Series(dtype="float64")


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_META_OUT = ("fiscal_period", "report_date", "publish_date", "source_link", "restated",
             "value_check")


def _join_key(frame: pd.DataFrame, annual: bool) -> pd.DataFrame:
    """Builds the key the four statements are merged on.

    On an annual request the balance sheet labels its rows ``Q4`` while the
    other three label them ``FY`` — the same period under two names, because a
    balance sheet is a snapshot at the close of the fourth quarter. Merging on
    the raw label would split every fiscal year into two half-populated rows, so
    annual data keys on the fiscal year alone.
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


def interest_expense(values: pd.DataFrame, prefer: str = "net") -> pd.Series:
    """Interest expense as a positive number.

    ``prefer`` picks the net or gross line where both are reported. Net interest
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

    Three ways down, most faithful first:

    1. the gross acquisition line, used as reported;
    2. the net change in fixed assets less the reported disposals, which
       reconstructs the gross figure exactly (KO FY2025: -2,099 - 13 = -2,112);
    3. the net change alone — an estimate, because it nets out disposals.
    """
    gross = values.get("capex_acquisition")
    net = values.get("capex_net_change")
    disposals = values.get("capex_disposals")

    if gross is not None and gross.notna().any():
        series = -gross
        if series.notna().all():
            return CapexResult(
                series,
                "reported",
                "CapEx = −(Acquisition of Fixed Assets & Intangibles)",
                "Gross acquisitions as reported by SimFin.",
            )

    if net is not None and disposals is not None and disposals.notna().any():
        series = -(net - disposals)
        return CapexResult(
            series,
            "derived",
            "CapEx = −(Change in Fixed Assets & Intangibles − Disposition of Fixed Assets & Intangibles)",
            "Gross acquisitions reconstructed by adding reported disposals back to the net change.",
        )

    if net is not None and net.notna().any():
        return CapexResult(
            -net,
            "estimate",
            "CapEx ≈ −(Change in Fixed Assets & Intangibles)",
            "SimFin reports no gross acquisition line for this company, so capex is "
            "taken net of asset disposals. It understates gross capital spending by "
            "whatever was sold.",
        )

    return CapexResult(
        pd.Series(np.nan, index=values.index),
        "estimate",
        "CapEx — not reported",
        "SimFin reports no capital expenditure line for this company in any period.",
    )


def total_debt(values: pd.DataFrame) -> pd.Series:
    """Short-term debt + current portion of LTD + long-term debt.

    Companies that do not split the maturities leave one component null, so the
    components are added with a fill. A row where *every* component is null
    stays ``NaN`` — that is "not reported", not "no debt".
    """
    components = ["short_term_debt", "current_portion_ltd", "long_term_debt"]
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
    """(Current assets − cash − STI) − (current liabilities − short-term debt − CPLTD).

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
        values["total_current_liabilities"]
        - _optional("short_term_debt")
        - _optional("current_portion_ltd")
    )
    return operating_assets - operating_liabilities


def delta_nwc(values: pd.DataFrame) -> pd.Series:
    """Year-on-year change in net working capital.

    The earliest period has no prior balance sheet to difference against, so it
    is ``NaN`` — never zero. Zero would read as "working capital did not move",
    which is a different and false claim.
    """
    return net_working_capital(values).diff()


def cash_flow_delta_nwc(values: pd.DataFrame) -> pd.Series:
    """SimFin's own working-capital movement from the cash flow statement.

    Reported as a cash impact (a build in working capital is negative), so it is
    negated to line up with the balance-sheet delta. The two legitimately differ
    — the cash flow version absorbs acquisitions and FX translation that a
    balance-sheet difference does not — which is why both are shown.
    """
    if "cf_change_in_wc" not in values.columns:
        return pd.Series(np.nan, index=values.index)
    return -values["cf_change_in_wc"]


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
    d_revenue = revenue.diff()

    ratios = pd.DataFrame(index=values.index)
    ratios["revenue_growth"] = revenue.pct_change()
    ratios["operating_margin"] = ebit / revenue_positive
    ratios["effective_tax_rate"] = effective_tax_rate(values)
    ratios["capex_pct_revenue"] = capex_values / revenue_positive
    ratios["dna_pct_revenue"] = dna / revenue_positive
    # Only meaningful when revenue actually moved; a flat year divides by ~zero.
    ratios["dnwc_pct_drevenue"] = (delta_nwc(values) / d_revenue).where(d_revenue.abs() > 0)
    return ratios


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

    window = {"3-year average": 3, "5-year average": 5, "10-year average": 10}.get(basis)
    if basis == "Most recent year":
        used = series.iloc[-1:]
        value = float(used.iloc[0])
    elif basis == "Median (all years)":
        used = series
        value = float(series.median())
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
    "interest_expense_gross",
    "interest_expense_net",
    "pretax_income",
    "tax_expense",
    "net_income",
    "dna",
    "sbc",
    "non_cash_items",
    "cf_change_in_wc",
    "cfo",
    "capex_acquisition",
    "capex_net_change",
    "capex_disposals",
    "cfi",
    "cff",
    "fx_effect",
    "net_change_in_cash",
)

#: Balance-sheet items, which are point-in-time and must never be summed.
STOCK_KEYS = (
    "cash",
    "short_term_investments",
    "cash_aggregate",
    "total_current_assets",
    "total_current_liabilities",
    "short_term_debt",
    "current_portion_ltd",
    "long_term_debt",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "total_liabilities_and_equity",
    "minority_interest",
    "preferred_equity",
    "long_term_investments",
)


# ---------------------------------------------------------------------------
# Fiscal calendar
# ---------------------------------------------------------------------------


def fiscal_year_end(
    fiscal_year_end_month: int | None, reference: date, latest_report_date: date | None
) -> date | None:
    """The next fiscal year end on or after ``reference``.

    SimFin gives the fiscal year end as a *month* (``endFy``: 9 for Apple), so
    the day comes from the most recent annual report date where we have one —
    Apple's fiscal year ends on the Saturday nearest 30 September, not always on
    the 30th.
    """
    if not fiscal_year_end_month:
        return None

    day = 30
    if latest_report_date is not None and latest_report_date.month == fiscal_year_end_month:
        day = latest_report_date.day

    for year in (reference.year, reference.year + 1):
        try:
            candidate = date(year, int(fiscal_year_end_month), day)
        except ValueError:  # e.g. 31 February after a day carried over
            candidate = date(year, int(fiscal_year_end_month), 28)
        if candidate >= reference:
            return candidate
    return None
