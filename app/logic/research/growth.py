"""Year-on-year growth, against the industry and against the company's own past.

Pure — no Streamlit, no network. The sibling of
:mod:`app.logic.research.valuation`: same three columns, same peer universe, same
medians, over rates of change instead of levels. Both sit on
:mod:`app.logic.research.comparison`.

**The governing rule is the inverse of the valuation tab's.** There, only a
positive reading counts — a negative P/E is a loss, not a cheap stock. Here a
negative reading is the answer: revenue falling 8% is exactly what the reader
came for. What cannot be allowed is a negative *base*, because the arithmetic
inverts without saying so:

    Intel's trailing diluted EPS went −4.75 to −2.12. The loss halved. The ratio
    of the two prints **−55%**, which reads as collapse.

The same trap fires on JPMorgan's operating cash flow (banks run it negative),
on Intel's EBIT, free cash flow and ROE, and on Apple's working capital, whose
base was −18.6bn. So: **the base period must be positive; the reading may be any
sign.** Everything that fails the gate reports the reason instead of a number.

Two sign conventions follow from the same rule:

* ``capex`` is negative for every company — Sharadar signs it that way and it is
  deliberately not in ``SIGN_FLIPPED`` — so growth is computed on the
  **magnitude**. "+20%" means the company spent more. Apple's −12.38bn to
  −10.04bn correctly reads −18.9%.
* A **zero base** is undefined and excluded: banks report no capex at all, and a
  non-payer has no dividend to grow. A zero *current* against a positive base is
  fine and meaningful — Intel's dividend going 0.125 to nil is a true −100%.

Growth steps once per filing, not once per session, so a five-year history is
about twenty observations rather than 1,255. That is why the tolerances here are
not the valuation tab's, and why the view prints the count beside the median.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.data.sharadar import vocabulary
from app.logic.research import comparison as shared
from app.logic.research.comparison import PRIOR, Comparison
from app.logic.research.formatting import format_pct
from app.provenance import Origin


@dataclass(frozen=True)
class GrowthTolerances:
    """Thresholds governing when a growth comparison may show a number.

    Owned here rather than shared with ``ValuationTolerances``, per the repo rule
    that a tolerance belongs to its consumer — and here that is not a formality.
    ``min_observations`` counts **filed quarters**, where the valuation tab counts
    **trading sessions**; one shared number would be wrong for one of them by a
    factor of sixty.
    """

    #: Below this many usable peers an industry median is not reported.
    min_peers: int = 5

    #: Quarters required before a 5Y median is reported. Five years of filings is
    #: about twenty, so eight is two years of history — enough for a median to
    #: mean something, little enough that a recently listed company still gets one.
    min_observations: int = 8

    #: Years of the company's own history the median is taken over.
    lookback_years: int = 5


@dataclass(frozen=True)
class Metric:
    """One row of the growth table."""

    label: str
    definition: str
    #: The Sharadar SF1 indicator, resolved to its published caption at read
    #: time rather than named as a column here.
    indicator: str
    #: True only for ``capex``, which Sharadar publishes as a negative outflow.
    #: The growth is taken on the magnitude, so a rise means more spending.
    magnitude: bool = False

    @property
    def sign(self) -> float:
        return -1.0 if self.magnitude else 1.0

    def format(self, value: float | None) -> str:
        return format_pct(value)


#: The growth table, in display order — the one place the row order is decided.
#: Top line down through the cash flow statement, then the returns and balance
#: sheet items, then what reaches the shareholder. ``test_research_growth``
#: asserts this agrees with what :func:`build_comparison` produces.
METRICS: tuple[Metric, ...] = (
    Metric(
        label="Revenue Growth (YoY)",
        definition="SF1 `revenue`, trailing twelve months against the same period a year earlier.",
        indicator="revenue",
    ),
    Metric(
        label="EBITDA Growth (YoY)",
        definition="SF1 `ebitda`, trailing twelve months year on year.",
        indicator="ebitda",
    ),
    Metric(
        label="EBIT Growth (YoY)",
        definition="SF1 `ebit`, trailing twelve months year on year.",
        indicator="ebit",
    ),
    Metric(
        label="EPS Diluted Growth (YoY)",
        definition=(
            "SF1 `epsdil`, trailing twelve months year on year. **GAAP** — Sharadar "
            "publishes no adjusted EPS, so there is no non-GAAP counterpart to show. "
            "Basic EPS tracked this within 0.3% on every company checked, which is "
            "why it is not a second row."
        ),
        indicator="epsdil",
    ),
    Metric(
        label="Levered FCF Growth (YoY)",
        definition=(
            "SF1 `fcf` (operating cash flow plus capex, so after interest), "
            "trailing twelve months year on year."
        ),
        indicator="fcf",
    ),
    Metric(
        label="Operating Cash Flow Growth (YoY)",
        definition="SF1 `ncfo`, trailing twelve months year on year.",
        indicator="ncfo",
    ),
    Metric(
        label="ROE Growth (YoY)",
        definition=(
            "SF1 `roe`, trailing twelve months year on year. The *relative* change "
            "in the ratio: 20% to 25% is +25%, not +5 points."
        ),
        indicator="roe",
    ),
    Metric(
        label="Working Capital Growth (YoY)",
        definition=(
            "SF1 `workingcapital` year on year. Frequently unavailable: a company "
            "running negative working capital — Apple among them — has no positive "
            "base to grow from."
        ),
        indicator="workingcapital",
    ),
    Metric(
        label="CAPEX Growth (YoY)",
        definition=(
            "SF1 `capex` year on year, on the **magnitude**: Sharadar signs capex "
            "negative, and a rise here means the company spent more. Banks report "
            "no capex at all, so they have no reading."
        ),
        indicator="capex",
        magnitude=True,
    ),
    Metric(
        label="Dividend Growth (1Y, TTM)",
        definition=(
            "SF1 `dps`, trailing twelve months year on year. A non-payer has no "
            "base to grow from; a payer that stops shows a true −100%."
        ),
        indicator="dps",
    ),
)


# ---------------------------------------------------------------------------
# The company's own history
# ---------------------------------------------------------------------------


def _indicator_history(
    statements_ttm: dict[str, pd.DataFrame], indicator: str
) -> pd.DataFrame:
    """``(Calendar Date, Publish Date, value)`` for one TTM indicator, oldest first.

    Both dates are carried because they do different jobs. The year-on-year
    pairing is on the **calendar date** — Sharadar's normalisation of the period
    to a calendar quarter-end — and the series is indexed on the **publish
    date**, when the growth rate became knowable.

    The pairing key is emphatically *not* the report date. Companies on a 52/53
    week fiscal calendar close a day or two earlier each year: Apple's quarters
    end 2025-06-28 and 2026-06-27, so "exactly twelve months before the latest
    report date" matches nothing and every growth rate reads unavailable.
    ``calendardate`` maps both onto 30 June, which is what it is for — and it is
    the same key :func:`comparison.build_peer_universe` pairs the peers on, so
    the company and its industry are measured over the same periods.

    Statement frames are keyed by Sharadar's published caption rather than by the
    indicator code, so the lookup goes through ``SF1_CAPTIONS``. A caption change
    upstream then loses the row visibly instead of returning a different line.
    """
    caption = vocabulary.SF1_CAPTIONS.get(indicator)
    if not caption:
        return pd.DataFrame()

    for frame in statements_ttm.values():
        if frame.empty or caption not in frame.columns:
            continue
        if not {"Calendar Date", "Publish Date"}.issubset(frame.columns):
            continue
        out = frame[["Calendar Date", "Publish Date", caption]].rename(
            columns={caption: "value"}
        )
        out = out.dropna(subset=["Calendar Date", "Publish Date"])
        return out.sort_values("Calendar Date").reset_index(drop=True)
    return pd.DataFrame()


def growth_history(
    statements_ttm: dict[str, pd.DataFrame], metric: Metric
) -> pd.DataFrame:
    """Every period's year-on-year growth, with the base it was measured from.

    Columns: ``Publish Date``, ``base``, ``now``, ``growth``. ``growth`` is
    ``NaN`` wherever the base is missing or non-positive — the paired values stay
    on the row so the caller can say *which* of those it was.

    The base is the period **exactly twelve months earlier**, matched on
    calendar date. A company that skipped a filing gets no base for that period
    rather than a rate measured over fifteen months.
    """
    history = _indicator_history(statements_ttm, metric.indicator)
    if history.empty:
        return pd.DataFrame(columns=["Publish Date", "base", "now", "growth"])

    base = history[["Calendar Date", "value"]].rename(columns={"value": "base"})
    base["Calendar Date"] = base["Calendar Date"] + pd.DateOffset(months=12)
    paired = history.merge(base, on="Calendar Date", how="left")

    now = pd.to_numeric(paired["value"], errors="coerce") * metric.sign
    prior = pd.to_numeric(paired["base"], errors="coerce") * metric.sign

    return pd.DataFrame(
        {
            "Publish Date": paired["Publish Date"],
            "base": prior,
            "now": now,
            "growth": _rate(now, prior),
        }
    )


def _rate(now: pd.Series, base: pd.Series) -> pd.Series:
    """``now / base − 1``, but only where the base is positive.

    The gate is the whole point of this module. Dividing by a negative base
    silently flips the sign of the answer, so a narrowing loss reads as a
    collapse; the result there is ``NaN`` and the caller reports why.
    """
    usable_base = base.where(base > 0)
    return now / usable_base - 1.0


def growth_series(
    statements_ttm: dict[str, pd.DataFrame], metric: Metric
) -> pd.Series:
    """The company's growth history for one metric, indexed by publish date."""
    history = growth_history(statements_ttm, metric)
    if history.empty:
        return shared.empty_dated()
    return pd.Series(
        history["growth"].to_numpy(), index=pd.DatetimeIndex(history["Publish Date"])
    )


def _usable(series: pd.Series) -> pd.Series:
    """Drops what cannot be plotted or ranked — and nothing else.

    Unlike the valuation tab's namesake this keeps **negative values**: a growth
    rate of −20% is a fact about the company, not a broken reading. Only
    non-finite entries go, which at this point means the periods whose base
    failed the gate in :func:`_rate`.
    """
    if series.empty:
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[numeric.abs().lt(float("inf"))].dropna()


# ---------------------------------------------------------------------------
# The peer cross-section
# ---------------------------------------------------------------------------


def universe_series(universe: pd.DataFrame, metric: Metric) -> pd.Series:
    """Every peer's year-on-year growth for one metric.

    Reads the paired ``<indicator>`` and ``<indicator>_prior`` columns
    :func:`comparison.build_peer_universe` puts on the snapshot, so the whole
    cross-section is one vectorised division with the same base gate the
    company's own series uses.
    """
    now_column, base_column = metric.indicator, f"{metric.indicator}{PRIOR}"
    if not {now_column, base_column}.issubset(universe.columns):
        return pd.Series(dtype="float64")

    now = pd.to_numeric(universe[now_column], errors="coerce") * metric.sign
    base = pd.to_numeric(universe[base_column], errors="coerce") * metric.sign
    return _rate(now, base)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_comparison(
    *,
    statements_ttm: dict[str, pd.DataFrame],
    peer_universe: pd.DataFrame,
    industry: str = "",
    today: date | None = None,
    tolerances: GrowthTolerances | None = None,
) -> list[Comparison]:
    """One :class:`Comparison` per entry in :data:`METRICS`, in that order.

    Keyword-only and pure. Takes no ``daily`` frame — unlike a multiple, a growth
    rate does not move with the share price, so nothing here reads a session.
    """
    tolerances = tolerances or GrowthTolerances()
    today = today or date.today()
    peers = shared.industry_peers(peer_universe, industry)
    cutoff = pd.Timestamp(today) - pd.DateOffset(years=tolerances.lookback_years)

    return [
        _compare(metric, statements_ttm, peers, industry, cutoff, tolerances)
        for metric in METRICS
    ]


def _compare(
    metric: Metric,
    statements_ttm: dict[str, pd.DataFrame],
    peers: pd.DataFrame,
    industry: str,
    cutoff: pd.Timestamp,
    tolerances: GrowthTolerances,
) -> Comparison:
    history = growth_history(statements_ttm, metric)
    series = growth_series(statements_ttm, metric)
    usable = _usable(series)

    current, current_reason = _current(history, metric)
    history_median, observations, percentile, history_reason = shared.summarise_history(
        usable,
        current,
        cutoff,
        min_observations=tolerances.min_observations,
        lookback_years=tolerances.lookback_years,
        unit="quarter",
    )
    values = (
        _usable(universe_series(peers, metric))
        if not peers.empty
        else pd.Series(dtype="float64")
    )
    median, peer_count, industry_reason = shared.industry_median(
        values, peers, industry, min_peers=tolerances.min_peers
    )

    return Comparison(
        metric=metric,
        current=current,
        current_reason=current_reason,
        history_median=history_median,
        history_observations=observations,
        history_unit="quarters",
        history_reason=history_reason,
        percentile=percentile,
        industry_median=median,
        peer_count=peer_count,
        industry_reason=industry_reason,
        # Always derived: every figure here is one reported number divided by
        # another, never a line item read straight off a filing.
        origin=Origin.UNAVAILABLE if current is None else Origin.DERIVED,
        # Percentage points, not a ratio. Growing 14% where the industry grows 5%
        # is "9 points faster"; calling it "180% faster" divides one rate by
        # another and means almost nothing — and breaks entirely when the
        # industry median is negative.
        industry_gap=None if current is None or median is None else current - median,
    )


def _current(history: pd.DataFrame, metric: Metric) -> tuple[float | None, str]:
    """The latest growth rate, or the reason there is not one.

    Four absences are distinguished, because they are four different facts and a
    reader who sees "not reported" deserves to know which: no history at all, no
    prior-year period to compare against, a base of zero, and a negative base.
    """
    if history.empty:
        return None, "No filed history for this measure."

    row = history.iloc[-1]
    base, now, growth = row["base"], row["now"], row["growth"]
    term = "dividend" if metric.indicator == "dps" else metric.indicator

    if pd.isna(now):
        return None, "Not reported in the latest filing."
    if pd.isna(base):
        return None, (
            "No filing twelve months before the latest one, so there is nothing "
            "to measure the year-on-year against."
        )
    if base == 0:
        if metric.indicator == "dps":
            return None, "No dividend a year ago, so there is no growth to report."
        return None, (
            f"Reported {term} was zero a year ago, so a percentage change from it "
            "is undefined."
        )
    if base < 0:
        return None, (
            f"Reported {term} was negative a year ago ({base:,.4g}). A percentage "
            "change from a negative base inverts its sign — a narrowing loss would "
            "read as a collapse — so no figure is shown."
        )
    if pd.isna(growth):
        return None, "Year-on-year change could not be computed."
    return float(growth), ""
