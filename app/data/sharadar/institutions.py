"""``SF3``/``SF3A``/``SF3B``: 13F institutional holdings, summaries and filer names."""

from __future__ import annotations

import pandas as pd

from app.data import cache
from app.data.sharadar import transport
from app.data.sharadar.vocabulary import SF3_SHARES

#: The prefixes SF3A repeats its holder/unit/value triple under, one per
#: security type, in the vendor's own order.
_SF3_PREFIXES = ("shr", "cll", "put", "wnt", "dbt", "prf", "fnd", "und")

#: Columns SF3 and SF3A publish **in thousands** of shares or contracts.
_SF3_THOUSANDS = ("units",) + tuple(f"{p}units" for p in _SF3_PREFIXES)

#: Columns published **in millions** of dollars.
_SF3_MILLIONS = ("value", "totalvalue") + tuple(f"{p}value" for p in _SF3_PREFIXES)

#: Holder counts, and the one percentage. Numeric, but never rescaled.
_SF3_PLAIN = ("percentoftotal",) + tuple(f"{p}holders" for p in _SF3_PREFIXES)

#: SF3A carries 29 columns and nothing reads the warrant, debt, preferred, fund
#: or undefined triples. Projected for the same reason the directory is.
_SF3A_COLUMNS = (
    "date,ticker,name,shrholders,shrunits,shrvalue,"
    "cllholders,cllunits,cllvalue,putholders,putunits,putvalue,"
    "totalvalue,percentoftotal"
)

#: The investor directory, projected to the two columns that resolve an id to a
#: name. Everything else on SF3B is a summary this app does not read.
_SF3B_COLUMNS = "investorid,investorname"


def _normalise_sf3(frame: pd.DataFrame) -> pd.DataFrame:
    """Types SF3/SF3A and converts the vendor's scale to raw units.

    ``units`` arrives in **thousands** of shares and ``value`` in **millions** of
    dollars. Both are rescaled here rather than at the call sites, on exactly the
    argument :func:`_normalise_daily` carries: there is more than one consumer,
    and a missed conversion passes every plausibility check by a factor of a
    thousand. Berkshire's 2018-Q3 filing is the check — ``shrvalue`` 221,020.2
    becomes $221.0bn, which is what it reported.

    Holder counts and ``percentoftotal`` are numeric but not rescaled.

    What is **not** fixed here: the share counts are as filed and are never
    restated for splits, so a series that spans one has a step in it. That is a
    join against corporate actions rather than a column-wise transform, so it
    belongs to ``logic/research/holdings`` — the same division of labour as
    SF2's filed prices.
    """
    if frame.empty:
        return frame

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in _SF3_PLAIN:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in _SF3_THOUSANDS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce") * 1_000.0
    for column in _SF3_MILLIONS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce") * 1_000_000.0

    if "date" not in frame.columns:
        return frame.reset_index(drop=True)
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _fetch_institutional_holders(
    ticker: str, since: str | None = None, investorids: tuple[str, ...] = ()
) -> pd.DataFrame:
    params: dict = {"ticker": ticker, "securitytype": SF3_SHARES}
    if since:
        params["date.gte"] = since
    if investorids:
        params["investorid"] = ",".join(investorids)
    return _normalise_sf3(transport._request_all("SF3", params, ticker=ticker))


def _fetch_institutional_summary(ticker: str) -> pd.DataFrame:
    params = {"ticker": ticker, "qopts.columns": _SF3A_COLUMNS}
    return _normalise_sf3(transport._request_all("SF3A", params, ticker=ticker))


def _fetch_investor_directory(as_of: str) -> pd.DataFrame:
    frame = transport._request_all(
        "SF3B", {"date": as_of, "qopts.columns": _SF3B_COLUMNS}
    )
    if frame.empty or "investorid" not in frame.columns:
        return pd.DataFrame(columns=["investorid", "investorname"])
    for column in ("investorid", "investorname"):
        frame[column] = frame[column].astype("string").str.strip()
    return frame.drop_duplicates(subset=["investorid"]).reset_index(drop=True)


@cache.data(ttl=60 * 60 * 24)
def fetch_institutional_holders(
    ticker: str, since: str | None = None, investorids: tuple[str, ...] = ()
) -> pd.DataFrame:
    """13F share positions in one company: ``ticker investorid date value units``.

    Restricted to ``SHR`` — common stock. Puts and calls are reported by SF3 too
    but belong to a different question, and :func:`fetch_institutional_summary`
    already carries their totals.

    Two filters, and the second is what makes a per-holder history affordable.
    ``since`` bounds the quarters, which is how the current register and the one
    before it are read together — about 12,000 rows for Apple, so churn against
    the prior quarter costs nothing beyond the register itself. ``investorids`` is passed through as SF3's
    comma-separated ``investorid``, so **twenty investors' entire thirteen-year
    history is one request of 945 rows** rather than fifty-three pages of six
    thousand. Without that, per-holder history would not be worth having.

    A day's TTL: 13F data moves four times a year, so even this is generous.

    ``units`` and ``value`` arrive rescaled to raw shares and raw dollars — see
    :func:`_normalise_sf3`. The counts are **as filed** and a split shows up as a
    step; ``logic/research/holdings`` owns that correction.
    """
    return _fetch_institutional_holders(ticker, since, investorids)


@cache.data(ttl=60 * 60 * 24)
def fetch_institutional_summary(ticker: str) -> pd.DataFrame:
    """SF3A: one row per quarter with the whole 13F register summarised.

    Fifty-three quarters back to June 2013 in a single request, which is what
    makes the ownership and breadth history cheap enough to draw without asking.
    Carries holder counts, units and value for shares, calls and puts, plus
    ``totalvalue`` and ``percentoftotal`` — the company's share of all 13F value
    reported that quarter.
    """
    return _fetch_institutional_summary(ticker)


@cache.data(ttl=60 * 60 * 24)
def fetch_investor_directory(as_of: str) -> pd.DataFrame:
    """``investorid`` → ``investorname``, for every 13F filer on one date.

    SF3 identifies a holder only by ``investorid`` — ``BLKROK``, ``VANGRD`` —
    and the names live on SF3B. This is **keyed on a date alone and never on a
    ticker**, so one cache entry serves every company looked up that day; it is
    the same argument the cross-sectional universe section below makes, and the
    same rule applies: a version of this taking a ticker would turn a fixed cost
    into a per-company one.

    Counter-intuitively the unfiltered fetch is also the *faster* one — 8,872
    rows projected to two columns in 3.7s, against 6.6s for the same request
    narrowed to ten ``investorid`` values, because the date is what the table is
    partitioned on.
    """
    return _fetch_investor_directory(as_of)
