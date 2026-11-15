"""What the people closest to the company are doing with its stock.

Pure — no Streamlit, no network. Reads ``SHARADAR/SF2`` as
:func:`app.data.sharadar_client.fetch_insider_transactions` returns it: one row
per line item on an SEC Form 3, 4 or 5.

**This module is mostly a set of exclusions, and that is the point.** SF2 is the
only table the app reads where the obvious arithmetic returns a number with the
*wrong sign* rather than a noisy one. Three separate traps do it, and any one of
them alone is enough:

1. **The codes.** Only ``P`` and ``S`` are decisions taken at a market price.
   Everything else is compensation mechanics or a transfer, and the mechanics
   outnumber the decisions by roughly forty-five to one — in a live 31,724-row
   sample, ``S`` 9,072, ``M`` 5,333, ``A`` 4,364, ``F`` 2,267 against ``P`` 267.
   Summed over every code, TSLA's filings since 2024-09-01 net **−$4,218m**;
   over ``P`` and ``S`` alone they net **+$220m**. The −$3,854m that flips the
   sign is one holder moving stock under a non-market code.
2. **The value column.** ``transactionvalue`` is an unsigned magnitude — 40,334
   shares sold at $115.95 is stored as ``+4676727`` — and it is null on 48% of
   rows besides. Summing it adds every sale to every purchase *and* misses half
   the trades. The flow is :func:`signed_flow`, and nothing here reads
   ``transactionvalue`` except the cross-check in :func:`reconciliation`.
3. **The holdings rows.** 28% of rows report a *position*, not a trade — a Form
   3, or a Form 4 line restating a balance. They carry no code and zero shares,
   so they do not move a total, but they do wreck every count and average taken
   over "transactions".

Two things follow that are easy to get backwards:

* **A 10% owner is labelled, never filtered.** The fix for the sign problem is
  the code set, as the TSLA figures show. Excluding the role as well would drop
  an activist's open-market purchases, which are among the most informative rows
  the table holds. Where one filer dominates a window,
  :class:`InsiderWindow` says so instead.
* **A quiet quarter is a zero, not a gap.** Nothing was unobtainable; the
  insiders traded nothing. Apple went five years without a single open-market
  purchase. A hole in the chart would read as missing data, which is the one
  thing it is not.

The unit throughout is the **transaction date**, never the filing date: a trade
made on 30 June and filed on 2 July belongs to the quarter the decision was
taken in. The client can only *request* on ``filingdate`` — see
``fetch_insider_transactions`` — so the frame arriving here is a superset, and
trimming it is this module's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from app.data.sharadar import vocabulary
from app.provenance import Origin, Sourced

SOURCE = "Sharadar SF2"


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InsiderTolerances:
    """Thresholds governing what this tab will call a signal.

    Owned here and deliberately not shared. ``GrowthTolerances.lookback_years``
    is also 5 and means something else: there, five years of *filings*, about
    twenty observations. Here, five years of *calendar*, over which a mega-cap
    may produce no open-market purchase at all. Sharing the number would be a
    coincidence, and the next person to tune one would silently move the other.
    """

    #: Years of history the charts cover.
    lookback_years: int = 5
    #: Quarters on the net-flow chart. Twenty-one is readable and 67–100%
    #: populated across the companies measured; sixty-one monthly buckets are
    #: 41–89% empty at large caps and 1-of-61 at the small ones.
    periodic_quarters: int = 21
    #: Trailing windows on the headline row, in months.
    trailing_months: tuple[int, ...] = (3, 6, 12)
    #: Distinct owners buying inside :attr:`cluster_window_days` before a run
    #: counts as a cluster. Two people is a coincidence often enough to be
    #: worthless.
    cluster_min_buyers: int = 3
    #: Two business weeks — long enough that independent decisions taken after
    #: one earnings release land together, short enough that a quarter of
    #: unrelated ordinary trades does not.
    cluster_window_days: int = 14
    #: Dollars below which a purchase does not count toward a cluster. A
    #: director topping up $4,000 is not evidence of anything.
    cluster_min_value: float = 25_000.0
    #: Share of a window's *gross* dollars one filer may account for before the
    #: net figure is annotated. TSLA's post-2024-09 total is essentially one
    #: person; a net number that is really one decision has to say so.
    concentration_threshold: float = 0.60
    #: Relative disagreement tolerated between ``|shares × price|`` and the
    #: reported ``transactionvalue`` before a row is counted inconsistent.
    value_check_tolerance: float = 0.01
    #: Months after which an insider's last reported balance is too old to list.
    #: SF2 records no departure event, so an old balance is not evidence the
    #: person still holds anything — or still works there.
    roster_stale_months: int = 18


# ---------------------------------------------------------------------------
# The transaction-code taxonomy
# ---------------------------------------------------------------------------

#: The only codes that are decisions taken at a market price, and the whole
#: subject of this tab. ``P`` is an open-market purchase, ``S`` an open-market
#: sale.
OPEN_MARKET_CODES = frozenset({"P", "S"})

#: Compensation mechanics. The insider chose none of these in the sense a reader
#: means: ``A`` is a grant the board made, ``M`` is exercising an option that was
#: already vesting, ``F`` is shares withheld to pay the tax on the other two.
COMPENSATION_CODES = frozenset({"A", "M", "F"})

#: Transfers with no market price or no counterparty decision: ``G`` gift, ``J``
#: other, ``C`` conversion, ``D`` disposition to the issuer, ``I`` discretionary
#: transaction, ``W`` will or inheritance, ``X`` in-the-money option exercise,
#: ``E`` expiry, ``L`` small acquisition.
TRANSFER_CODES = frozenset({"G", "J", "C", "D", "I", "W", "X", "E", "L"})

#: Every code seen live. **Not closed** — the SEC adds codes and Sharadar passes
#: through whatever the filing carried — so consumers test membership of the set
#: they understand rather than exhausting this one. An unrecognised code lands in
#: the "unclassified" row of :func:`reconciliation`, where it is visible rather
#: than dropped.
KNOWN_CODES = OPEN_MARKET_CODES | COMPENSATION_CODES | TRANSFER_CODES

BUY_CODE = "P"
SELL_CODE = "S"

#: The columns every function here needs. A frame missing one of these is not
#: an SF2 frame, and is treated as empty rather than half-read.
REQUIRED_COLUMNS = (
    "ownername",
    "transactiondate",
    "transactioncode",
    "transactionshares",
    "transactionpricepershare",
    "securityadcode",
)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def usable(frame: pd.DataFrame) -> bool:
    """Whether the frame carries enough of SF2 to read at all."""
    return not frame.empty and set(REQUIRED_COLUMNS).issubset(frame.columns)


def signed_flow(frame: pd.DataFrame) -> pd.Series:
    """Dollars, signed by direction: ``transactionshares × price``.

    Never a sum of ``transactionvalue``. Sharadar publishes that as an unsigned
    magnitude, so adding it up counts every disposal as a purchase, and it is
    null on roughly half of all rows besides.

    ``NaN`` where either factor is missing — a gift or an unpriced grant has no
    dollar flow, and zero is not the answer. :func:`reconciliation` counts those
    rows rather than absorbing them.
    """
    if not usable(frame):
        return pd.Series(dtype="float64")
    shares = pd.to_numeric(frame["transactionshares"], errors="coerce")
    price = pd.to_numeric(frame["transactionpricepershare"], errors="coerce")
    return shares * price


def is_holdings_only(frame: pd.DataFrame) -> pd.Series:
    """The 28% of rows that restate a position instead of reporting a trade.

    A **bare** ``securityadcode`` — ``N`` or ``D`` with no acquired/disposed
    second letter — together with a null ``transactioncode``. On the live sample
    those two conditions mapped onto each other exactly, so **both** are tested
    rather than either alone: a row satisfying one and not the other is a change
    in the feed, and it stays visible in :func:`reconciliation` rather than being
    silently reclassified as a trade.
    """
    if not usable(frame):
        return pd.Series(dtype="bool")
    bare = frame["securityadcode"].astype("string").str.len().eq(1)
    return bare.fillna(False) & frame["transactioncode"].isna()


def is_open_market(frame: pd.DataFrame) -> pd.Series:
    """Rows carrying a decision at a market price: code ``P`` or ``S``."""
    if not usable(frame):
        return pd.Series(dtype="bool")
    return frame["transactioncode"].isin(OPEN_MARKET_CODES)


def is_non_derivative(frame: pd.DataFrame) -> pd.Series:
    """Rows whose balance is in shares rather than option contracts.

    ``securityadcode`` opening with ``N``. A derivative row's
    ``sharesownedfollowingtransaction`` counts contracts, and adding it to a
    share balance overstates the holding, sometimes by multiples.
    """
    if not usable(frame):
        return pd.Series(dtype="bool")
    opening = frame["securityadcode"].astype("string").str[0]
    return opening.eq(vocabulary.SF2_NON_DERIVATIVE).fillna(False)


def split_adjusted_price(frame: pd.DataFrame, actions: pd.DataFrame) -> pd.Series:
    """Filed prices moved onto the price chart's split-adjusted basis.

    **The two disagree, and plotting them together without this is wrong.**
    Sharadar's SEP ``close`` is adjusted for splits; SF2's
    ``transactionpricepershare`` is the price as filed and never restated. Tesla
    is the case that shows it: insiders filed sales at $981–$1,237 in November
    2021, while the chart's close for the same month reads $338–$410, because a
    3:1 split in August 2022 sits between them. Left alone, every pre-split
    marker floats far above the line and implies a collapse that never happened.

    The factor comes from ``ACTIONS`` split rows, which the snapshot already
    carries — one filing's price is divided by the product of every split since.
    A company with no splits gets its filed prices back unchanged.

    Only the *position* needs this. The dollar flow is shares times price and
    both are as-filed, so their product is the real amount and is invariant: this
    is deliberately not applied anywhere in :func:`signed_flow`.

    **Divides**, where the holdings tab's share counts multiply — see
    :func:`app.data.sharadar.vocabulary.split_factors`, which both sit on.
    """
    price = pd.to_numeric(frame["transactionpricepershare"], errors="coerce")
    factor = vocabulary.split_factors(actions, frame["transactiondate"])
    return price / factor


def role_of(frame: pd.DataFrame) -> pd.Series:
    """"Director", "Officer — CFO", "10% owner", or a joined combination.

    The 10% owner is a **label, never a filter**. See the module docstring: the
    code set is what fixes the sign, and dropping the role would discard an
    activist's open-market purchases.
    """
    if frame.empty:
        return pd.Series(dtype="object")

    yes = vocabulary.SF2_FLAG_YES

    def _describe(row) -> str:
        parts: list[str] = []
        if str(row.get("isofficer") or "") == yes:
            title = str(row.get("officertitle") or "").strip()
            parts.append(f"Officer — {title}" if title else "Officer")
        if str(row.get("isdirector") or "") == yes:
            parts.append("Director")
        if str(row.get("istenpercentowner") or "") == yes:
            parts.append("10% owner")
        return " · ".join(parts) if parts else "Insider"

    return frame.apply(_describe, axis=1)


def open_market_trades(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    tolerances: InsiderTolerances | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """Every priced ``P`` and ``S`` in the lookback, ready to plot against the price.

    The filed rows plus four columns: ``when`` (the transaction date), ``flow``
    (signed dollars, :func:`signed_flow`), ``role`` and ``plotted`` — the price
    the insider paid, moved onto the chart's split-adjusted basis by
    :func:`split_adjusted_price`. A marker belongs at the price paid rather than
    that day's close: for a ``P`` or an ``S`` the two agree inside the day's
    range, and snapping to the close would quietly redraw the trade.

    Restricted to the open-market codes for a visual reason as well as an
    arithmetic one: an ``M`` exercise is priced at the strike, often a fraction of
    the market price, so plotting every code would scatter markers far below the
    line and imply a crash that never happened. A trade with no price or no date
    is left out, never placed by guesswork.
    """
    tolerances = tolerances or InsiderTolerances()
    today = today or date.today()
    if not usable(frame):
        return frame.iloc[0:0]
    cutoff = pd.Timestamp(today) - pd.DateOffset(years=tolerances.lookback_years)
    market = frame[is_open_market(frame)].copy()
    market["flow"] = signed_flow(market)
    market["when"] = pd.to_datetime(market["transactiondate"], errors="coerce")
    market = market[market["flow"].notna() & market["when"].notna()]
    market = market[market["when"] >= cutoff]
    if market.empty:
        return market
    market["role"] = role_of(market)
    market["plotted"] = split_adjusted_price(market, actions)
    return market


def _when(frame: pd.DataFrame) -> pd.Series:
    """The transaction date, falling back to the filing date.

    A holdings-only row reports no transaction and so carries no transaction
    date. It still has to sort somewhere, and the filing date is the only date it
    has. Nothing that counts *trades* ever reaches this fallback, because those
    rows are excluded first.
    """
    when = pd.to_datetime(frame["transactiondate"], errors="coerce")
    if "filingdate" in frame.columns:
        when = when.fillna(pd.to_datetime(frame["filingdate"], errors="coerce"))
    return when


def _window(
    frame: pd.DataFrame, months: int, today: date
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Rows transacted inside the trailing window, and the cutoff itself."""
    cutoff = pd.Timestamp(today) - pd.DateOffset(months=months)
    transacted = pd.to_datetime(frame["transactiondate"], errors="coerce")
    inside = transacted.gt(cutoff) & transacted.le(pd.Timestamp(today))
    return frame[inside.fillna(False)], cutoff


def _code_tally(frame: pd.DataFrame) -> str:
    """"A 2, F 1, M 1" — what a window held instead of open-market trades."""
    counts = frame["transactioncode"].dropna().value_counts()
    return ", ".join(f"{code} {int(n)}" for code, n in counts.items())


# ---------------------------------------------------------------------------
# Trailing windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InsiderWindow:
    """One trailing window's open-market activity, with its provenance.

    The three dollar figures are :class:`Sourced` because each is a single number
    that can be genuinely unobtainable for four distinct reasons, and the rule
    against silent fallbacks says which one has to reach the screen. Their origin
    is ``DERIVED``, never ``REPORTED``: no single SF2 column is the answer, and
    the column that looks like it is is the one that gets the sign backwards.
    """

    months: int
    net: Sourced
    bought: Sourced
    sold: Sourced
    buy_trades: int = 0
    sell_trades: int = 0
    buyers: int = 0
    sellers: int = 0
    #: Distinct people, which is **not** ``buyers + sellers``: an officer who
    #: both bought and sold inside the window is one insider, and adding the two
    #: counts them twice.
    participants: int = 0
    top_owner: str = ""
    concentration: float | None = None
    concentration_note: str = ""

    @property
    def ok(self) -> bool:
        return self.net.ok


def _unavailable(months: int, reason: str) -> InsiderWindow:
    absent = Sourced.unavailable(SOURCE, reason)
    return InsiderWindow(months=months, net=absent, bought=absent, sold=absent)


def trailing_flow(
    frame: pd.DataFrame,
    months: int,
    *,
    tolerances: InsiderTolerances | None = None,
    today: date | None = None,
) -> InsiderWindow:
    """Net open-market dollars over the trailing ``months``.

    Four absences are distinguished, because they are four different facts and a
    reader who sees "not reported" deserves to know which: nobody has ever filed;
    filings exist but none inside the window; rows inside the window but every
    one of them compensation or a transfer; open-market rows that carry no price.

    The last has never been observed live — all 4,951 ``P``/``S`` rows sampled
    carried a price — and is kept because "never seen" is not "cannot happen".
    """
    tolerances = tolerances or InsiderTolerances()
    today = today or date.today()

    if not usable(frame):
        return _unavailable(
            months,
            "Sharadar returned no Form 3/4/5 filings for this company. SF2 "
            "coverage starts in 2008; a company listed since then with no rows "
            "has had nobody file.",
        )

    trades = frame[~is_holdings_only(frame)]
    inside, _ = _window(trades, months, today)

    if inside.empty:
        latest = pd.to_datetime(trades["transactiondate"], errors="coerce").max()
        when = "" if pd.isna(latest) else f" The most recent was {latest.date()}."
        return _unavailable(
            months,
            f"No insider reported a transaction in the last {months} months. "
            f"There are {len(trades)} earlier filings.{when}",
        )

    market = inside[is_open_market(inside)]
    if market.empty:
        return _unavailable(
            months,
            f"All {len(inside)} filings in this window are compensation "
            f"mechanics or transfers ({_code_tally(inside)}). None is a decision "
            "to buy or sell at a market price, so there is no net flow.",
        )

    flow = signed_flow(market)
    priced = market[flow.notna()]
    if priced.empty:
        shares = pd.to_numeric(market["transactionshares"], errors="coerce")
        return _unavailable(
            months,
            f"{len(market)} open-market trades in this window report no price "
            "per share, so their dollar value cannot be computed. The share "
            f"counts are {shares[shares > 0].sum():,.0f} bought and "
            f"{-shares[shares < 0].sum():,.0f} sold.",
        )

    values = signed_flow(priced)
    buys = priced[values > 0]
    sells = priced[values < 0]
    bought = float(values[values > 0].sum())
    sold = float(values[values < 0].sum())
    formula = "Σ transaction shares × price per share, codes P and S only"
    as_of = pd.to_datetime(priced["transactiondate"]).max().date()

    gross = values.abs()
    top_owner, concentration, note = "", None, ""
    if gross.sum() > 0:
        by_owner = gross.groupby(priced["ownername"]).sum()
        top_owner = str(by_owner.idxmax())
        concentration = float(by_owner.max() / gross.sum())
        if concentration >= tolerances.concentration_threshold:
            note = (
                f"{top_owner} accounts for {concentration:.0%} of the gross "
                "dollars traded in this window, so the net figure is close to "
                "one person's decision rather than a consensus."
            )

    return InsiderWindow(
        months=months,
        net=Sourced.derived(bought + sold, SOURCE, formula, as_of=as_of),
        bought=Sourced.derived(bought, SOURCE, formula, as_of=as_of),
        sold=Sourced.derived(sold, SOURCE, formula, as_of=as_of),
        buy_trades=len(buys),
        sell_trades=len(sells),
        buyers=int(buys["ownername"].nunique()),
        sellers=int(sells["ownername"].nunique()),
        participants=int(priced["ownername"].nunique()),
        top_owner=top_owner,
        concentration=concentration,
        concentration_note=note,
    )


def trailing_windows(
    frame: pd.DataFrame,
    *,
    tolerances: InsiderTolerances | None = None,
    today: date | None = None,
) -> list[InsiderWindow]:
    """One :class:`InsiderWindow` per entry in ``trailing_months``."""
    tolerances = tolerances or InsiderTolerances()
    return [
        trailing_flow(frame, months, tolerances=tolerances, today=today)
        for months in tolerances.trailing_months
    ]


# ---------------------------------------------------------------------------
# Quarterly flow
# ---------------------------------------------------------------------------

PERIODIC_COLUMNS = ("period", "bought", "sold", "net", "trades", "owners", "partial")


def periodic_flow(
    frame: pd.DataFrame,
    *,
    tolerances: InsiderTolerances | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """Net open-market dollars per calendar quarter, oldest first.

    Quarters with no open-market trade are present **with zeros, not absent**.
    That is not a zero-fill of the kind the invariants forbid: nothing was
    unobtainable, the company's insiders simply traded nothing, and a gap in a
    bar chart reads as missing data. Apple had no open-market purchase at all in
    five years, so this is the common case rather than the edge one.

    Bucketed on ``transactiondate``, never ``filingdate``. Quarterly rather than
    monthly because monthly is 41–89% empty depending on the company, and
    1-of-61 at the smallest measured.

    The bucket holding ``today`` is flagged ``partial`` — it is still open, and a
    short final bar is otherwise read as a collapse in activity.

    Carries no reason string. Every absence here is already named by the
    twelve-month :class:`InsiderWindow`, and a second reason channel would be a
    second place for the two to disagree.
    """
    tolerances = tolerances or InsiderTolerances()
    today = today or date.today()

    end = pd.Period(pd.Timestamp(today), freq="Q")
    periods = pd.period_range(
        end=end, periods=tolerances.periodic_quarters, freq="Q"
    )
    empty = pd.DataFrame(
        {
            "period": periods.to_timestamp(how="end").normalize(),
            "bought": 0.0,
            "sold": 0.0,
            "net": 0.0,
            "trades": 0,
            "owners": 0,
            "partial": [p == end for p in periods],
        }
    )
    if not usable(frame):
        return empty

    market = frame[is_open_market(frame)].copy()
    market["flow"] = signed_flow(market)
    market = market[market["flow"].notna()]
    if market.empty:
        return empty

    transacted = pd.to_datetime(market["transactiondate"], errors="coerce")
    market = market[transacted.notna()]
    if market.empty:
        return empty
    market["bucket"] = pd.to_datetime(
        market["transactiondate"]
    ).dt.to_period("Q")
    market = market[market["bucket"].isin(periods)]
    if market.empty:
        return empty

    grouped = market.groupby("bucket", observed=True)
    tally = pd.DataFrame(
        {
            "bought": grouped["flow"].apply(lambda s: s[s > 0].sum()),
            "sold": grouped["flow"].apply(lambda s: s[s < 0].sum()),
            "net": grouped["flow"].sum(),
            "trades": grouped.size(),
            "owners": grouped["ownername"].nunique(),
        }
    )
    tally.index = tally.index.to_timestamp(how="end").normalize()

    filled = empty.set_index("period")
    for column in ("bought", "sold", "net", "trades", "owners"):
        filled[column] = tally[column].reindex(filled.index).fillna(0)
    filled["trades"] = filled["trades"].astype(int)
    filled["owners"] = filled["owners"].astype(int)
    return filled.reset_index()[list(PERIODIC_COLUMNS)]


# ---------------------------------------------------------------------------
# Cluster buying
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InsiderCluster:
    """Several insiders buying on the open market at about the same time."""

    start: date
    end: date
    buyers: int
    trades: int
    value: float
    owners: tuple[str, ...] = ()


def clusters(
    frame: pd.DataFrame,
    *,
    tolerances: InsiderTolerances | None = None,
    today: date | None = None,
) -> list[InsiderCluster]:
    """Episodes of several insiders buying at once, most recent first.

    **Purchases only.** A sell cluster is a calendar artefact: 10b5-1 plans and
    vesting schedules make executives sell in lockstep on dates chosen months
    earlier, so counting simultaneous sellers measures the plan rather than a
    view.

    Distinct **owners** are counted, not trades — one director buying three times
    in a week is one person's decision. Buys below
    :attr:`InsiderTolerances.cluster_min_value` do not count toward a cluster at
    all: a director topping up $4,000 is not evidence.

    Overlapping windows merge, so one continuous buying episode is one cluster
    rather than a dozen sliding ones.
    """
    tolerances = tolerances or InsiderTolerances()
    today = today or date.today()

    if not usable(frame):
        return []

    buys = frame[frame["transactioncode"].eq(BUY_CODE)].copy()
    buys["flow"] = signed_flow(buys)
    buys = buys[buys["flow"] >= tolerances.cluster_min_value]
    buys["when"] = pd.to_datetime(buys["transactiondate"], errors="coerce")
    cutoff = pd.Timestamp(today) - pd.DateOffset(years=tolerances.lookback_years)
    buys = buys[buys["when"].notna() & buys["when"].gt(cutoff)]
    if buys.empty:
        return []

    buys = buys.sort_values("when")
    span = pd.Timedelta(days=tolerances.cluster_window_days)

    episodes: list[list] = []
    current: list = []
    for _, row in buys.iterrows():
        if current and row["when"] - current[-1]["when"] > span:
            episodes.append(current)
            current = []
        current.append(row)
    if current:
        episodes.append(current)

    found = []
    for episode in episodes:
        owners = {str(row["ownername"]) for row in episode}
        if len(owners) < tolerances.cluster_min_buyers:
            continue
        found.append(
            InsiderCluster(
                start=episode[0]["when"].date(),
                end=episode[-1]["when"].date(),
                buyers=len(owners),
                trades=len(episode),
                value=float(sum(row["flow"] for row in episode)),
                owners=tuple(sorted(owners)),
            )
        )
    return sorted(found, key=lambda c: c.end, reverse=True)


# ---------------------------------------------------------------------------
# The roster
# ---------------------------------------------------------------------------

ROSTER_COLUMNS = ("owner", "role", "shares", "as_of", "last_code")


@dataclass(frozen=True)
class InsiderRoster:
    """Share balances as last reported, which is not the same as held today."""

    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    reason: str = ""
    omitted: int = 0
    as_of: date | None = None

    @property
    def ok(self) -> bool:
        return not self.rows.empty


def roster(
    frame: pd.DataFrame,
    *,
    tolerances: InsiderTolerances | None = None,
    today: date | None = None,
) -> InsiderRoster:
    """Each insider's share balance, as last reported.

    Four rules, and dropping any one of them ships a plausible wrong number:

    * **Non-derivative rows only.** A derivative row's
      ``sharesownedfollowingtransaction`` counts option contracts, not shares.
    * **Latest row per (owner, security title, direct/indirect), then summed
      across those groups.** Each is a genuinely separate holding; summing across
      *filings* within one of them double-counts a running balance.
    * **Stale owners are excluded and counted.** SF2 records no departure event,
      so a balance from 2019 is not evidence that person still holds it — or
      still works there.
    * **The figure is "as last reported", never "currently held".** There is no
      date on which SF2 asserts a current position.
    """
    tolerances = tolerances or InsiderTolerances()
    today = today or date.today()

    if not usable(frame):
        return InsiderRoster(
            reason="Sharadar returned no Form 3/4/5 filings for this company."
        )
    if "sharesownedfollowingtransaction" not in frame.columns:
        return InsiderRoster(reason="SF2 returned no share balances.")

    shares = frame[is_non_derivative(frame)].copy()
    if shares.empty:
        return InsiderRoster(
            reason=(
                "Every filing for this company reports a derivative holding — "
                "options, warrants or units — so there is no share balance to "
                "report. Contract counts are deliberately not added to shares."
            )
        )

    shares["when"] = _when(shares)
    shares = shares[shares["when"].notna()]
    if shares.empty:
        return InsiderRoster(reason="No filing carries a usable date.")

    order = [c for c in ("when", "filingdate", "rownum") if c in shares.columns]
    shares = shares.sort_values(order, kind="stable")

    group = [
        c
        for c in ("ownername", "securitytitle", "directorindirect")
        if c in shares.columns
    ]
    latest = shares.groupby(group, dropna=False, as_index=False).tail(1)

    cutoff = pd.Timestamp(today) - pd.DateOffset(
        months=tolerances.roster_stale_months
    )
    seen = latest.groupby("ownername")["when"].max()
    fresh = set(seen[seen >= cutoff].index)
    omitted = int(len(seen) - len(fresh))
    current = latest[latest["ownername"].isin(fresh)]

    if current.empty:
        return InsiderRoster(
            reason=(
                f"All {omitted} insiders on file last reported before "
                f"{cutoff.date()}. SF2 records no departure, so an older balance "
                "is not evidence anyone still holds it."
            ),
            omitted=omitted,
        )

    roles = role_of(current)
    balances = pd.DataFrame(
        {
            "owner": current["ownername"].astype(str),
            "role": roles.astype(str),
            "shares": pd.to_numeric(
                current["sharesownedfollowingtransaction"], errors="coerce"
            ),
            "as_of": current["when"],
            "last_code": current["transactioncode"].astype("string").fillna("—"),
        }
    )
    rows = (
        balances.groupby("owner", as_index=False)
        .agg(
            role=("role", "first"),
            shares=("shares", "sum"),
            as_of=("as_of", "max"),
            last_code=("last_code", "last"),
        )
        .sort_values("shares", ascending=False, ignore_index=True)
    )

    reason = ""
    if omitted:
        reason = (
            f"{omitted} more insider(s) last filed before {cutoff.date()} and are "
            "not listed. SF2 records no departure, so an old balance is not "
            "evidence that person still holds it — or still works here."
        )
    return InsiderRoster(
        rows=rows[list(ROSTER_COLUMNS)],
        reason=reason,
        omitted=omitted,
        as_of=rows["as_of"].max().date() if not rows.empty else None,
    )


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------

#: Categories, in the order the panel shows them. They **partition the frame
#: exactly**, so the counts sum to the row count.
RECONCILIATION_CATEGORIES = (
    "Open market, priced",
    "Open market, no price",
    "Compensation",
    "Transfer",
    "Holdings only",
    "Outside the window",
    "Unclassified code",
)


def _categorise(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """One category per row. Order matters — the first match wins."""
    holdings = is_holdings_only(frame)
    transacted = pd.to_datetime(frame["transactiondate"], errors="coerce")
    stale = transacted.lt(cutoff).fillna(False)
    market = is_open_market(frame)
    priced = signed_flow(frame).notna()
    code = frame["transactioncode"]

    category = pd.Series("Unclassified code", index=frame.index, dtype="object")
    category[code.isin(TRANSFER_CODES)] = "Transfer"
    category[code.isin(COMPENSATION_CODES)] = "Compensation"
    category[market & ~priced] = "Open market, no price"
    category[market & priced] = "Open market, priced"
    category[~holdings & stale] = "Outside the window"
    category[holdings] = "Holdings only"
    return category


def reconciliation(
    frame: pd.DataFrame,
    *,
    tolerances: InsiderTolerances | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """Every row that came back, and which bucket it fell in.

    This is the tab's answer to "why does Apple look empty". The categories
    partition the frame exactly, so a reader can see that 8 of 16 rows are being
    shown on purpose rather than by accident — and at a real mega-cap, that the
    number is 267 of 31,724.

    Also carries the value cross-check: rows where ``|shares × price|`` and the
    reported ``transactionvalue`` disagree by more than
    :attr:`InsiderTolerances.value_check_tolerance`. The two agreeing on almost
    every live row is what makes the flow arithmetic trustworthy; a disagreement
    is a data-quality fact the reader is entitled to.
    """
    tolerances = tolerances or InsiderTolerances()
    today = today or date.today()

    columns = ["Category", "Rows", "Shares", "Dollars", "Note"]
    if not usable(frame):
        return pd.DataFrame(columns=columns)

    cutoff = pd.Timestamp(today) - pd.DateOffset(years=tolerances.lookback_years)
    category = _categorise(frame, cutoff)
    flow = signed_flow(frame)
    shares = pd.to_numeric(frame["transactionshares"], errors="coerce")

    rows = []
    for name in RECONCILIATION_CATEGORIES:
        mask = category == name
        if not mask.any():
            continue
        rows.append(
            {
                "Category": name,
                "Rows": int(mask.sum()),
                "Shares": float(shares[mask].sum(skipna=True)),
                "Dollars": float(flow[mask].sum(skipna=True)),
                "Note": _CATEGORY_NOTES.get(name, ""),
            }
        )

    mismatched = _value_mismatches(frame, tolerances)
    rows.append(
        {
            "Category": "— of which reported value disagrees",
            "Rows": int(mismatched),
            "Shares": float("nan"),
            "Dollars": float("nan"),
            "Note": (
                "|shares × price| differs from the reported transactionvalue by "
                f"more than {tolerances.value_check_tolerance:.0%}."
            ),
        }
    )
    return pd.DataFrame(rows, columns=columns)


_CATEGORY_NOTES = {
    "Open market, priced": "Counted. Codes P and S — decisions at a market price.",
    "Open market, no price": "Excluded: no price per share, so no dollar value.",
    "Compensation": "Excluded: grants, option exercises, tax withholding.",
    "Transfer": "Excluded: gifts, conversions, inheritances — no market decision.",
    "Holdings only": "Excluded: a position restated, not a trade.",
    "Outside the window": "Older than the chart's five years.",
    "Unclassified code": "A transaction code this app does not recognise.",
}


def _value_mismatches(frame: pd.DataFrame, tolerances: InsiderTolerances) -> int:
    """Rows where the reported magnitude and the computed one disagree."""
    if "transactionvalue" not in frame.columns:
        return 0
    reported = pd.to_numeric(frame["transactionvalue"], errors="coerce")
    computed = signed_flow(frame).abs()
    both = reported.notna() & computed.notna() & reported.gt(0)
    if not both.any():
        return 0
    relative = (computed[both] - reported[both]).abs() / reported[both]
    return int(relative.gt(tolerances.value_check_tolerance).sum())
