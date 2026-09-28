"""The machinery both ⚖️ Relative Valuation sub-tabs are built on.

Pure — no Streamlit, no network. Shared by :mod:`app.logic.research.valuation`
and :mod:`app.logic.research.growth`, which ask the same question of different
arithmetic: *for this metric, what is the company's reading now, what has it
been over five years, and what is the industry doing?*

What lives here is the part that is genuinely the same — the record the view
renders, the median-and-percentile summary, the cross-sectional universe, and
the industry lookup. What deliberately does **not** live here is each tab's
notion of a *usable* number, because on that point the two tabs are opposites:

* A valuation multiple must be **positive to be shown at all** — a negative P/E
  is a loss, not a cheap stock.
* A growth rate is **meaningful at any sign** — revenue falling 8% is the
  answer, not a failure — but its *base* must be positive, or the arithmetic
  inverts. Intel's diluted EPS went −4.75 to −2.12, halving the loss, and the
  ratio of the two prints −55%.

Each tab also owns its own tolerances, per the repo rule that a threshold
belongs to its consumer: the valuation window is counted in trading sessions and
the growth window in filed quarters, so a shared number would be wrong for one
of them by a factor of sixty.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.formatting import is_missing
from app.provenance import Origin

#: SF1 indicators carried on the universe snapshot for the year-on-year
#: comparison, paired against their ``_prior`` counterparts by
#: :func:`build_peer_universe`.
GROWTH_TERMS = (
    "revenue", "ebitda", "ebit", "epsdil", "fcf",
    "ncfo", "roe", "workingcapital", "capex", "dps",
)

#: Level indicators the valuation multiples divide by. No ``_prior`` twin —
#: nothing asks what last year's revenue bought at last year's price.
LEVEL_TERMS = ("revenue", "ncfo", "ncfdiv")

#: Suffix marking a term's value four quarters earlier.
PRIOR = "_prior"


@dataclass(frozen=True)
class Comparison:
    """One metric, answered three ways, with the reason for each absence.

    ``None`` on any of the three values means *not obtainable*, and the matching
    ``*_reason`` says why. Nothing is substituted — that is the same rule the DCF
    engine's ``Sourced`` enforces, applied to a display record carrying three
    numbers instead of one.

    ``origin`` and ``industry_gap`` are **fields rather than derived properties**
    because the two sub-tabs derive them differently. A vendor-published multiple
    is ``REPORTED`` where a growth rate is always ``DERIVED``; and the gap to the
    industry is a *ratio* for a multiple (25% dearer) but a *difference in
    percentage points* for a growth rate, where dividing one rate by another
    says almost nothing.
    """

    metric: object
    current: float | None = None
    current_reason: str = ""
    history_median: float | None = None
    history_observations: int = 0
    #: What ``history_observations`` counts, already pluralised — "sessions" for
    #: a multiple sampled every trading day, "quarters" for a growth rate that
    #: steps once per filing. The two differ by a factor of sixty, so the view
    #: shows the unit rather than a bare number.
    history_unit: str = "observations"
    history_reason: str = ""
    percentile: float | None = None
    industry_median: float | None = None
    peer_count: int = 0
    industry_reason: str = ""
    origin: Origin = Origin.UNAVAILABLE
    industry_gap: float | None = None


def empty_dated() -> pd.Series:
    """An empty series that is still date-indexed.

    A bare ``pd.Series(dtype="float64")`` carries a ``RangeIndex``, and slicing
    that against a ``Timestamp`` raises rather than returning nothing — so a
    company with no history would crash the tab instead of reporting itself
    unavailable. Every empty series returned by a history builder is dated for
    that reason.
    """
    return pd.Series(dtype="float64", index=pd.DatetimeIndex([]))


def summarise_history(
    series: pd.Series,
    current: float | None,
    cutoff: pd.Timestamp,
    *,
    min_observations: int,
    lookback_years: int,
    unit: str,
) -> tuple[float | None, int, float | None, str]:
    """Median, count, percentile and reason for one metric's own history.

    ``unit`` names what is being counted — "session" for the daily multiples,
    "quarter" for the growth rates, which step only when a company files. The
    count is shown beside the median precisely because those two are three
    orders of magnitude apart and a reader has to know which they are looking at.

    The percentile is the share of the window that sat *below* today's reading,
    with no direction read into it; the view decides whether low means cheap.
    """
    window = series[series.index >= cutoff]
    if len(window) < min_observations:
        return None, len(window), None, (
            f"Only {len(window)} usable {unit}(s) in the last {lookback_years} "
            f"years; {min_observations} needed."
        )

    median = float(window.median())
    percentile = None if current is None else float((window < current).mean())
    return median, len(window), percentile, ""


def industry_median(
    values: pd.Series, peers: pd.DataFrame, industry: str, *, min_peers: int
) -> tuple[float | None, int, str]:
    """The median across the peer group, or the reason there is not one.

    Medians, never means: a metric whose denominator approaches zero prints in
    the thousands, and one such company moves an average but not a median. The
    count travels with the answer because Sharadar's 145 vendor industries hold
    a median of about twenty companies and the smallest hold three — below
    ``min_peers`` the count is reported instead of a number, rather than falling
    back to a broader group the reader did not ask for.
    """
    if not industry:
        return None, 0, "No industry classification for this company."
    if peers.empty:
        return None, 0, f"No peer data loaded for {industry}."

    count = len(values)
    if count < min_peers:
        return None, count, (
            f"Only {count} of {len(peers)} companies in {industry} have a usable "
            f"reading; {min_peers} needed for a median."
        )
    return float(values.median()), count, ""


# ---------------------------------------------------------------------------
# The cross-sectional universe
# ---------------------------------------------------------------------------


def build_peer_universe(
    daily: pd.DataFrame, fundamentals: pd.DataFrame, directory: pd.DataFrame
) -> pd.DataFrame:
    """Joins the three universe snapshots into one row per listed company.

    Carries the five vendor multiples and today's enterprise value and market
    cap from DAILY, the industry label from TICKERS, and from SF1 both the
    latest trailing-twelve-month terms **and** their ``_prior`` twins from four
    quarters earlier, which is what the growth comparison divides.

    Returns an empty frame if the DAILY or directory fetch came back empty.
    Empty is a legitimate outcome that leaves every industry median reporting
    itself unavailable; it is never a reason to fall back to something else.
    """
    if daily.empty or directory.empty or "ticker" not in daily.columns:
        return pd.DataFrame()

    keep = [
        c
        for c in ("ticker", "date", "pe", "ps", "pb", "evebit", "evebitda", "ev", "marketcap")
        if c in daily.columns
    ]
    universe = daily[keep].copy()

    listed = directory[directory.get("isdelisted", "N") == "N"]
    columns = [c for c in ("ticker", "sector", "industry") if c in listed.columns]
    universe = universe.merge(listed[columns], on="ticker", how="inner")

    paired = _pair_year_ago(fundamentals)
    if not paired.empty:
        universe = universe.merge(paired, on="ticker", how="left")

    return universe.reset_index(drop=True)


def _pair_year_ago(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Each ticker's latest SF1 row, with its year-ago values alongside.

    The universe fetch returns every period in an eighteen-month window, so the
    pairing happens here rather than in the client: the base row is the one whose
    ``calendardate`` is *exactly* twelve months before the latest, never merely
    the oldest available. A company that skipped a filing, or listed part-way
    through the window, gets no ``_prior`` values and reports its growth
    unavailable — which is true, rather than a growth rate measured over a period
    nobody chose.
    """
    if fundamentals.empty or not {"ticker", "calendardate"}.issubset(fundamentals.columns):
        return pd.DataFrame()

    # Ordered, not a set union: column order in the joined frame has to be
    # stable, and `set` ordering is not.
    wanted = list(GROWTH_TERMS) + [c for c in LEVEL_TERMS if c not in GROWTH_TERMS]
    terms = [c for c in wanted if c in fundamentals.columns]
    if not terms:
        return pd.DataFrame()

    ordered = fundamentals.sort_values(["ticker", "calendardate"])
    latest = ordered.groupby("ticker", as_index=False).tail(1)[
        ["ticker", "calendardate", *terms]
    ]

    base_keys = latest[["ticker", "calendardate"]].copy()
    base_keys["calendardate"] = base_keys["calendardate"] - pd.DateOffset(months=12)
    prior_terms = [c for c in GROWTH_TERMS if c in fundamentals.columns]
    prior = base_keys.merge(
        ordered[["ticker", "calendardate", *prior_terms]],
        on=["ticker", "calendardate"],
        how="left",
    ).drop(columns=["calendardate"])
    prior = prior.rename(columns={c: f"{c}{PRIOR}" for c in prior_terms})

    return latest.drop(columns=["calendardate"]).merge(prior, on="ticker", how="left")


def industry_peers(universe: pd.DataFrame, industry: str) -> pd.DataFrame:
    """Every company the vendor puts in the same industry."""
    if universe.empty or not industry or "industry" not in universe.columns:
        return pd.DataFrame()
    return universe[universe["industry"] == industry]


def industry_of(peer_universe: pd.DataFrame, ticker: str) -> str:
    """The industry Sharadar puts one ticker in, or ``""``.

    Read from the universe rather than from the company's own profile so the
    company and its peers are always grouped by the same field — a label taken
    from one source and compared against buckets built from another is a
    comparison against the wrong set.
    """
    if peer_universe.empty or "industry" not in peer_universe.columns:
        return ""
    match = peer_universe[peer_universe["ticker"] == ticker.strip().upper()]
    if match.empty:
        return ""
    value = match.iloc[0]["industry"]
    return "" if is_missing(value) else str(value)
