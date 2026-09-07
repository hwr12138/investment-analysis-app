"""Relative valuation: one multiple against its peers and against its own past.

Pure — no Streamlit, no network, no global state. The fetches that feed it live
in :mod:`app.logic.research.dataset`, which calls :func:`build_peer_universe`
once per search and stores the result on the snapshot.

Every metric is answered in three ways, and all three use the **same definition**
so the columns are comparable side by side:

* **Current** — the company's latest reading.
* **5Y median** — the median of its own daily readings over five years.
* **Industry median** — the median across every listed company Sharadar puts in
  the same industry, on the same day.

Two rules govern which numbers are allowed to appear, and both exist because a
valuation multiple stops meaning anything when its denominator goes the wrong
way:

**Only positive readings count.** A negative P/E is a loss, not a cheap stock;
including it would drag an industry median downwards and read as bargain. Of
5,373 tickers with a ``pe`` on a sample day, only 2,874 were positive. The same
rule applied to the dividend yield excludes non-payers, which is what stops the
median yield of most industries collapsing to zero.

**Medians, never means.** A P/E series through a quarter of near-zero earnings
contains readings in the thousands. The median ignores them; a mean would report
them. That is the reason this module never averages anything, and the reason no
winsorisation is needed either.

The three derived multiples are ratios of two whole-dollar aggregates rather than
of per-share figures. Sharadar restates ``dps`` and ``price`` onto the current
share basis, so a per-share ratio has to reason about splits; a ratio of two
dollar totals does not. ``-ncfdiv / marketcap`` was checked against ``dps /
price`` across NVIDIA's 10-for-1 split and agrees to 1e-5.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.data import sharadar_client
from app.logic.research import comparison as shared
from app.logic.research.comparison import Comparison
from app.logic.research.formatting import format_pct, format_ratio
from app.provenance import Origin

#: Where a term in a derived multiple is read from. ``DAILY`` terms are
#: per-session and ``SF1`` terms step at each filing.
DAILY = "daily"
SF1 = "sf1"


@dataclass(frozen=True)
class ValuationTolerances:
    """Thresholds governing when a comparison is allowed to show a number.

    Owned here rather than shared with ``dcf.validation.Tolerances`` or
    ``statements.StatementTolerances``, per the repo's rule that a tolerance
    belongs to its consumer. These gate whether a *peer median* is meaningful;
    those gate whether a valuation is trustworthy and whether a statement row
    grows a marker. Coupling them would let a change made for one reason quietly
    move another.
    """

    #: Below this many usable peers an industry median is not reported. Sharadar's
    #: 145 vendor industries have a median of ~20 members, falling to ~13 once
    #: non-positive readings are excluded, and the smallest hold three or four.
    #: A median of three companies is not a median of anything.
    min_peers: int = 5

    #: Sessions required before a 5Y median is reported. A company public for two
    #: months has a "5-year history" of forty prints.
    min_observations: int = 60

    #: Years of the company's own history the median is taken over.
    lookback_years: int = 5


@dataclass(frozen=True)
class Term:
    """One side of a derived multiple: a column, and where to read it from."""

    source: str
    column: str
    #: Applied before the division. Only the dividend payment uses it, to turn
    #: Sharadar's negative cash outflow into a positive distribution.
    scale: float = 1.0


@dataclass(frozen=True)
class Metric:
    """One row of the comparison table.

    Either ``daily_column`` is set — Sharadar publishes the multiple itself — or
    ``numerator`` and ``denominator`` are, and it is divided here.
    """

    label: str
    definition: str
    daily_column: str | None = None
    numerator: Term | None = None
    denominator: Term | None = None
    #: ``format_ratio`` for a multiple, ``format_pct`` for the yield.
    percent: bool = False
    #: True only for the dividend yield: for every other metric a lower reading
    #: is the cheaper one, and the "vs industry" verdict has to say so.
    higher_is_cheaper: bool = False

    @property
    def vendor(self) -> bool:
        return self.daily_column is not None

    def format(self, value: float | None) -> str:
        return format_pct(value) if self.percent else format_ratio(value)


#: The comparison table, in display order — the one place the row order is
#: decided. The five vendor multiples come first because they are read straight
#: off Sharadar; the three derived ones follow. ``test_valuation`` asserts this
#: agrees with what :func:`build_comparison` produces, so a metric added here
#: without a source, or computed without being listed, fails a test rather than
#: silently appearing or vanishing.
METRICS: tuple[Metric, ...] = (
    Metric(
        label="P/E (TTM)",
        definition="Sharadar DAILY `pe` — price ÷ trailing twelve-month net income to common.",
        daily_column="pe",
    ),
    Metric(
        label="P/S (TTM)",
        definition="Sharadar DAILY `ps` — market cap ÷ trailing twelve-month revenue.",
        daily_column="ps",
    ),
    Metric(
        label="P/B (TTM)",
        definition="Sharadar DAILY `pb` — market cap ÷ shareholders' equity.",
        daily_column="pb",
    ),
    Metric(
        label="EV/EBITDA (TTM)",
        definition="Sharadar DAILY `evebitda` — enterprise value ÷ trailing EBITDA.",
        daily_column="evebitda",
    ),
    Metric(
        label="EV/EBIT (TTM)",
        definition="Sharadar DAILY `evebit` — enterprise value ÷ trailing EBIT.",
        daily_column="evebit",
    ),
    Metric(
        label="EV/Sales (TTM)",
        definition=(
            "Enterprise value (DAILY) ÷ trailing twelve-month revenue (SF1). "
            "Derived: Sharadar publishes no EV/Sales column."
        ),
        numerator=Term(DAILY, "ev"),
        denominator=Term(SF1, "revenue"),
    ),
    Metric(
        label="Price/Cash Flow (TTM)",
        definition=(
            "Market cap (DAILY) ÷ trailing twelve-month net cash from operations "
            "(SF1 `ncfo`). Operating cash flow, not free cash flow."
        ),
        numerator=Term(DAILY, "marketcap"),
        denominator=Term(SF1, "ncfo"),
    ),
    Metric(
        label="Dividend Yield (TTM)",
        definition=(
            "Trailing twelve-month dividends paid (SF1 `ncfdiv`, sign reversed) ÷ "
            "market cap (DAILY). A cash total over a cash total, so splits cancel. "
            "Non-payers are excluded from the medians rather than counted as zero."
        ),
        numerator=Term(SF1, "ncfdiv", scale=-1.0),
        denominator=Term(DAILY, "marketcap"),
        percent=True,
        higher_is_cheaper=True,
    ),
)


def _universe_series(universe: pd.DataFrame, metric: Metric) -> pd.Series:
    """The metric's reading for every company in the universe.

    Positionally indexed, not dated: this is a cross-section of many companies
    on one day, where :func:`company_series` is one company across many days.
    """
    if metric.vendor:
        if metric.daily_column not in universe.columns:
            return pd.Series(dtype="float64")
        return pd.to_numeric(universe[metric.daily_column], errors="coerce")

    numerator, denominator = metric.numerator, metric.denominator
    if numerator.column not in universe.columns or denominator.column not in universe.columns:
        return pd.Series(dtype="float64")
    top = pd.to_numeric(universe[numerator.column], errors="coerce") * numerator.scale
    bottom = pd.to_numeric(universe[denominator.column], errors="coerce")
    return top / bottom.where(bottom != 0)


# ---------------------------------------------------------------------------
# The company's own history
# ---------------------------------------------------------------------------


def _ttm_terms(statements_ttm: dict[str, pd.DataFrame], indicator: str) -> pd.DataFrame:
    """A ``(Publish Date, value)`` history of one TTM indicator, oldest first.

    Keyed on **Publish Date** — the SF1 ``datekey``, when the figure became
    public — and not on the report period. Joining a price series onto the
    period end would value the company using numbers nobody had yet.

    Statement frames are keyed by Sharadar's published caption rather than by
    the indicator code, so the lookup goes through ``SF1_CAPTIONS`` instead of
    naming a column here. A caption change upstream then loses the row visibly
    rather than returning a different line.
    """
    caption = sharadar_client.SF1_CAPTIONS.get(indicator)
    if not caption:
        return pd.DataFrame()

    for frame in statements_ttm.values():
        if frame.empty or caption not in frame.columns:
            continue
        if "Publish Date" not in frame.columns:
            continue
        out = frame[["Publish Date", caption]].rename(columns={caption: indicator})
        out = out.dropna(subset=["Publish Date"])
        return out.sort_values("Publish Date").reset_index(drop=True)
    return pd.DataFrame()


def company_series(
    daily: pd.DataFrame, statements_ttm: dict[str, pd.DataFrame], metric: Metric
) -> pd.Series:
    """The company's daily history of one metric, indexed by date.

    Vendor multiples slice ``daily`` directly. Derived ones step the SF1 term
    forward from each filing date with :func:`pandas.merge_asof`, so every
    session divides by the most recent figure that was actually published by
    then.
    """
    if daily.empty or "date" not in daily.columns:
        return shared.empty_dated()

    frame = daily.sort_values("date")

    if metric.vendor:
        if metric.daily_column not in frame.columns:
            return shared.empty_dated()
        values = pd.to_numeric(frame[metric.daily_column], errors="coerce")
        return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(frame["date"]))

    numerator, denominator = metric.numerator, metric.denominator
    daily_term = numerator if numerator.source == DAILY else denominator
    sf1_term = denominator if numerator.source == DAILY else numerator

    if daily_term.column not in frame.columns:
        return shared.empty_dated()
    history = _ttm_terms(statements_ttm, sf1_term.column)
    if history.empty:
        return shared.empty_dated()

    merged = pd.merge_asof(
        frame[["date", daily_term.column]],
        history,
        left_on="date",
        right_on="Publish Date",
        direction="backward",
    )
    left = pd.to_numeric(merged[daily_term.column], errors="coerce") * daily_term.scale
    right = pd.to_numeric(merged[sf1_term.column], errors="coerce") * sf1_term.scale
    if numerator.source == DAILY:
        values = left / right.where(right != 0)
    else:
        values = right / left.where(left != 0)
    return pd.Series(values.to_numpy(), index=pd.DatetimeIndex(merged["date"]))


def _usable(series: pd.Series) -> pd.Series:
    """Drops everything that cannot be a valuation multiple.

    Non-finite values, and non-positive ones: a negative multiple means the
    denominator went the wrong way, which is information about the business
    rather than about its price.
    """
    if series.empty:
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric[numeric.gt(0) & numeric.lt(float("inf"))].dropna()


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_comparison(
    *,
    daily: pd.DataFrame,
    statements_ttm: dict[str, pd.DataFrame],
    peer_universe: pd.DataFrame,
    industry: str = "",
    today: date | None = None,
    tolerances: ValuationTolerances | None = None,
) -> list[Comparison]:
    """One :class:`Comparison` per entry in :data:`METRICS`, in that order.

    Keyword-only and pure: every argument is a frame the snapshot already holds,
    so this runs on each rerun without touching the network.
    """
    tolerances = tolerances or ValuationTolerances()
    today = today or date.today()
    peers = shared.industry_peers(peer_universe, industry)
    cutoff = pd.Timestamp(today) - pd.DateOffset(years=tolerances.lookback_years)

    return [
        _compare(metric, daily, statements_ttm, peers, industry, cutoff, tolerances)
        for metric in METRICS
    ]



def _compare(
    metric: Metric,
    daily: pd.DataFrame,
    statements_ttm: dict[str, pd.DataFrame],
    peers: pd.DataFrame,
    industry: str,
    cutoff: pd.Timestamp,
    tolerances: ValuationTolerances,
) -> Comparison:
    series = company_series(daily, statements_ttm, metric)
    usable = _usable(series)

    current, current_reason = _current(series, usable, metric)
    history_median, observations, percentile, history_reason = shared.summarise_history(
        usable,
        current,
        cutoff,
        min_observations=tolerances.min_observations,
        lookback_years=tolerances.lookback_years,
        unit="session",
    )
    median, peer_count, industry_reason = _industry(metric, peers, industry, tolerances)

    return Comparison(
        metric=metric,
        current=current,
        current_reason=current_reason,
        history_median=history_median,
        history_observations=observations,
        history_unit="sessions",
        history_reason=history_reason,
        percentile=percentile,
        industry_median=median,
        peer_count=peer_count,
        industry_reason=industry_reason,
        # Vendor multiples are read straight off Sharadar; the ones divided here
        # are arithmetic over reported figures.
        origin=(
            Origin.UNAVAILABLE
            if current is None
            else Origin.REPORTED if metric.vendor else Origin.DERIVED
        ),
        # A ratio, not a difference: "25% dearer than its industry" is how a
        # multiple is read. The growth tab compares in percentage points instead,
        # which is why this is set by the builder rather than by `Comparison`.
        industry_gap=(
            None if current is None or not median else current / median - 1.0
        ),
    )


def _current(
    series: pd.Series, usable: pd.Series, metric: Metric
) -> tuple[float | None, str]:
    """The latest reading, or the reason there is not one.

    A company whose latest print is negative is told apart from one with no
    print at all: "earnings are negative" and "we have no data" are different
    facts and the page says which.
    """
    if series.empty:
        return None, "No history for this multiple."
    latest = series.dropna()
    if latest.empty:
        return None, "No reading in the series."

    value = float(latest.iloc[-1])
    term = "yield" if metric.percent else "multiple"

    # Zero and negative are different facts and get different sentences. A
    # suspended dividend is not "a negative denominator", and saying so of
    # Intel's nil payout would be plainly untrue.
    if value == 0:
        if metric.percent:
            return None, "No dividend paid in the trailing twelve months."
        return None, f"Latest {term} is zero."
    if value < 0:
        return None, (
            f"Latest {term} is {metric.format(value)}. A negative reading means "
            "the denominator is negative — the company is loss-making on this "
            "measure — not that the stock is cheap."
        )
    if usable.empty:
        return None, "No positive reading in the series."
    return value, ""


def _industry(
    metric: Metric, peers: pd.DataFrame, industry: str, tolerances: ValuationTolerances
) -> tuple[float | None, int, str]:
    """The industry median, over the peers whose reading is usable."""
    values = _usable(_universe_series(peers, metric)) if not peers.empty else pd.Series(dtype="float64")
    return shared.industry_median(
        values, peers, industry, min_peers=tolerances.min_peers
    )

