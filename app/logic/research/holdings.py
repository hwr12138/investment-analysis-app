"""Who owns the company, and what they did with it last quarter.

Pure — no Streamlit, no network. Reads ``SHARADAR/SF3`` and ``SF3A`` as the
client returns them: one row per (investor, quarter) 13F share position, and one
summary row per quarter.

**Like the insiders module, this is mostly a set of defences**, and for the same
reason: the obvious reading of 13F data is not noisy, it is *categorically
wrong*. Four traps, two of them shaped exactly like ones the insiders tab
already carries.

1. **Filer restructuring.** An asset manager reorganising its filing entities
   produces a position that collapses in one entity and reappears in a sibling.
   Vanguard did this in 2026-Q1: Apple reads ``VANGRD`` 1,426,284,000 →
   953,848,000, a 33% "sale", while sibling entities appear from nothing. It is
   not one company — the same phantom sale lands on **every ticker that manager
   holds**, so a naive "biggest sellers" list is wrong for thousands of companies
   at once. :data:`FILER_FAMILIES` sums a manager's entities before anything is
   ranked, which reads that same position as **−5.9%**, against −1.6% for the
   register as a whole.

   That map is curated and it is the *only* mechanism here, because the obvious
   derived one does not work. Matching filers on a shared leading name token was
   tried and measured: it produced 32 flags on Apple and 33 on Microsoft, almost
   every one spurious — ``BANK OF MARIN`` paired with ``BANK OF AMERICA``,
   ``ONE RIVER ASSET MANAGEMENT`` with ``ONE DAY IN JULY``, ``CAPITAL FUND
   MANAGEMENT`` (a French quant fund) with ``CAPITAL WORLD INVESTORS``. Six
   thousand advisory firms share a few hundred opening words, so the signal is
   swamped. A curated list is narrower and honest; a heuristic here would have
   been neither.
2. **Splits.** SF3's ``units`` are **as filed and never restated**, so a split
   lands in the series as a step nobody took: BlackRock's Tesla position reads
   7,720 then 38,020 across the 2020 five-for-one, which a quarter-on-quarter
   change reports as a **+392% purchase**. Every unit count here is put through
   :func:`split_adjusted_units` first. This is the mirror of the insiders tab's
   filed prices — there the factor divides, here it multiplies — and both sit on
   ``vocabulary.split_factors``.
3. **Ownership is a ratio across two bases.** SF1's ``sharesbas`` *is* restated
   for splits and SF3's ``units`` are not, so dividing them straight gives a
   figure wrong by the split ratio at every pre-split quarter.
   :func:`ownership_history` puts both on today's basis before dividing.
4. **``value`` is a market value, so it moves when the stock does.** A manager
   who traded nothing shows a value change equal to the quarter's return, and a
   "biggest sellers by value" table on a quarter where the price fell 20% names
   six thousand sellers, every one of them fictional. **Every change here is
   computed on units**; value is carried for magnitude and ranking within a
   single quarter, and is never differenced across two. :class:`Holder` carries a
   ``change`` in units and no change in value, on purpose.

Two more things the figures mean but do not say:

* **Reported ownership above 100% is real and is never clamped.** A lent share
  is reported by the lender and again by whoever bought it, and a manager may
  file under several entities. Etsy reads 114.3% on live data. The figure is
  labelled *reported* institutional ownership throughout, because that is what
  it is; clamping it would be the no-silent-fallbacks rule broken in a new
  direction.
* **A gap in a holder's series is not a zero** — the opposite of
  ``insiders.periodic_flow``, deliberately. There, a quiet quarter is a fact:
  nobody traded. Here, a manager absent from a quarter may have sold out, fallen
  under the $100m filing threshold, been acquired, changed filing entity or
  filed late. Those are five different facts and none of them is "held nothing",
  so the series carries ``None`` and the chart draws a break.

One more thing the numbers cannot tell you, so the module does not pretend to:
roughly a third of a large company's reported institutional value sits with a
handful of index managers, whose position changes are fund flows and index
weights rather than opinions. :data:`INDEX_MANAGERS` marks them — but it is a
**curated editorial list, not a derived fact**, it is approximate for houses
running both active and passive money, and the tab shows it rather than hiding
it behind a label.

Finally, what 13F simply does not contain: short positions, holders below the
$100m filing threshold, and anything more recent than the last filing deadline —
which is 45 days after quarter end, so a "current" register can be 135 days old.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from app.data.sharadar import vocabulary
from app.provenance import Origin, Sourced

SOURCE = "Sharadar SF3"

#: Days the SEC allows between quarter end and a 13F landing.
FILING_LAG_DAYS = 45

#: The columns every function here needs from SF3.
REQUIRED_COLUMNS = ("investorid", "date", "units", "value")


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldingsTolerances:
    """Thresholds governing what this tab will call a change.

    Owned here and not shared. ``InsiderTolerances.lookback_years`` is also 5 and
    counts something else — there, five years of calendar over an event feed;
    here, five years of quarterly filings, which is twenty observations. The
    collision is a coincidence and sharing the number would propagate a tuning of
    one into the other.
    """

    #: Holders in the table. Twenty covers about half the register's value at a
    #: mega-cap and nearly all of it at a small one, and is as many rows as a
    #: reader will actually compare.
    top_holders: int = 20
    #: Quarters drawn in each sparkline. Twelve is three years — long enough to
    #: show a trend, short enough that the line is not a smear at table width.
    sparkline_quarters: int = 12
    #: Years of ownership history on the aggregate chart.
    lookback_years: int = 13
    #: Fractional move below which a position counts as unchanged rather than
    #: added or trimmed. Half a percent: 13F counts are rounded and a manager
    #: reinvesting a dividend is not a decision.
    material_change: float = 0.005
    #: Holders in the concentration figure.
    concentration_top_n: int = 10


#: Investor ids whose position changes are mechanical rather than discretionary
#: — index and index-adjacent managers, where a quarter-on-quarter move is fund
#: flow and index weight rather than anyone's view of the company.
#:
#: **This is a curated editorial list and not a fact derived from the data.**
#: Sharadar publishes no manager type. It is deliberately narrow: the houses here
#: are overwhelmingly index-tracking, and firms running both active and passive
#: money in size — Fidelity, Invesco, T. Rowe — are left off precisely because
#: marking them either way would be a claim the list cannot support. Custodial
#: banks and market makers appear in registers too and are not marked, because
#: "not an index fund" is not the same as "an investor with a view".
#:
#: The tab shows this list rather than applying it invisibly.
#: Keys are ``investorid`` values **and** :data:`FILER_FAMILIES` labels, because
#: a collapsed family row is keyed on its label rather than on any one entity.
INDEX_MANAGERS = {
    "BLKROK": "BlackRock",
    "VANGRD": "Vanguard",
    "VANGPM": "Vanguard",
    "Vanguard": "Vanguard",
    "STASTR": "State Street",
    "GEODEC": "Geode",
    "NORTHE": "Northern Trust",
    "CHASCH": "Schwab",
}


#: Managers that file under several entities, and the entities they file under.
#:
#: **Curated, and the only defence against filer restructuring.** A manager
#: reorganising its filings produces a position that collapses in one entity and
#: reappears in siblings, which reads as a large sale and is not one — and
#: because the reorganisation is the manager's, the same phantom sale lands on
#: every company it holds. Summing the family before ranking is what makes the
#: number true: Apple's Vanguard position reads −33.1% as ``VANGRD`` alone and
#: **−5.9%** as the family, against −1.6% for the register as a whole.
#:
#: Deriving this instead was tried and abandoned — see the module docstring. The
#: list is therefore **incomplete by construction**: it covers the managers large
#: enough that a restructuring would distort a table, and any filer not named
#: here stands alone, which is the safe default. The tab shows the map.
FILER_FAMILIES = {
    "Vanguard": (
        "VANGRD", "VANGPM", "VANGPI", "VANFID", "VANMAR", "VANUST",
        "VANGU2", "VANGU3", "VANGU4", "VANGU5", "VANGU6",
    ),
    "Capital Group": ("RESEAR", "WORLDX", "CAPITA", "CAPIT2", "CAPIT5", "SARLXX"),
}

#: ``investorid`` → family label, for the ids that belong to one.
FAMILY_OF = {
    investorid: label
    for label, members in FILER_FAMILIES.items()
    for investorid in members
}


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def usable(frame: pd.DataFrame) -> bool:
    """Whether the frame carries enough of SF3 to read at all."""
    return (
        frame is not None
        and not frame.empty
        and set(REQUIRED_COLUMNS).issubset(frame.columns)
    )


def split_adjusted_units(frame: pd.DataFrame, actions: pd.DataFrame) -> pd.Series:
    """Filed share counts moved onto today's split basis.

    **Multiplies**, where the insiders tab's filed prices divide — both sit on
    :func:`app.data.sharadar.vocabulary.split_factors`. A count filed before a
    five-for-one is a fifth of the shares it represents today, so a series that
    spans one has a step in it that nobody took: BlackRock's Tesla position goes
    7,720 → 38,020 across the 2020 split, and a quarter-on-quarter change reads
    that as a 392% purchase.

    ``value`` needs no such treatment. A split changes the share count and the
    price together, so the dollar value is already continuous.
    """
    units = pd.to_numeric(frame["units"], errors="coerce")
    return units * vocabulary.split_factors(actions, frame["date"])


def quarters(frame: pd.DataFrame) -> list[pd.Timestamp]:
    """Every quarter the frame carries, oldest first."""
    if not usable(frame):
        return []
    found = pd.to_datetime(frame["date"], errors="coerce").dropna().unique()
    return sorted(pd.Timestamp(q) for q in found)


REGISTER_COLUMNS = ("investorid", "name", "units", "value", "family")


def register(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    quarter: pd.Timestamp,
    names: pd.DataFrame | None = None,
    *,
    collapse: bool = True,
) -> pd.DataFrame:
    """One quarter's holders: :data:`REGISTER_COLUMNS`, largest value first.

    ``units`` is split-adjusted. A holder filing twice for one quarter cannot
    happen — SF3's primary key covers it — so nothing is de-duplicated on that.

    ``collapse`` sums the entities of a :data:`FILER_FAMILIES` manager into one
    row, and is **on by default**: every consumer that compares two quarters
    needs it, because a family that reorganised between them otherwise reads as
    one huge sale beside several new positions. The reconciliation panel passes
    ``collapse=False`` to report the register as actually filed.
    """
    empty = pd.DataFrame(columns=list(REGISTER_COLUMNS))
    if not usable(frame):
        return empty

    rows = frame[pd.to_datetime(frame["date"], errors="coerce").eq(quarter)].copy()
    if rows.empty:
        return empty

    rows["units"] = split_adjusted_units(rows, actions)
    rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
    rows["name"] = _resolve_names(rows["investorid"], names)
    rows["family"] = rows["investorid"].map(FAMILY_OF).fillna("")
    rows = rows[list(REGISTER_COLUMNS)].dropna(subset=["units"])

    if collapse:
        rows = _collapse_families(rows)
    return rows.sort_values("value", ascending=False).reset_index(drop=True)


def _collapse_families(rows: pd.DataFrame) -> pd.DataFrame:
    """Sums each :data:`FILER_FAMILIES` manager's entities into a single row.

    The surviving row is keyed on the family label rather than on any one
    entity's id, so it cannot be mistaken for a filing that exists.
    """
    grouped = rows["family"].where(rows["family"].ne(""), rows["investorid"])
    if grouped.equals(rows["investorid"]):
        return rows

    out = (
        rows.assign(_key=grouped)
        .groupby("_key", as_index=False)
        .agg(
            investorid=("investorid", "first"),
            name=("name", "first"),
            units=("units", "sum"),
            value=("value", "sum"),
            family=("family", "first"),
        )
    )
    named = out["family"].ne("")
    out.loc[named, "name"] = out.loc[named, "family"] + " (all filings)"
    out.loc[named, "investorid"] = out.loc[named, "family"]
    return out[list(REGISTER_COLUMNS)]


def _resolve_names(ids: pd.Series, names: pd.DataFrame | None) -> pd.Series:
    """Investor names, falling back to the id when the directory has no entry.

    SF3 identifies a holder only by its id, so an unresolved one shows as
    ``BLKROK`` rather than as a blank — a code the reader can look up beats
    nothing at all.
    """
    ids = ids.astype("string")
    if names is None or names.empty or "investorname" not in names.columns:
        return ids
    lookup = dict(zip(names["investorid"], names["investorname"]))
    return ids.map(lambda i: lookup.get(i) or i).astype("string")


def top_investorids(
    frame: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    tolerances: HoldingsTolerances | None = None,
) -> tuple[str, ...]:
    """The largest holders in the most recent quarter, by value.

    Called between fetches: the per-holder history is one request only because
    SF3 takes a comma-separated ``investorid``, and this is what fills that list
    in. Value rather than units, because units across two share classes are not
    comparable.

    **Ranks collapsed but returns expanded.** A family counts as one holder when
    deciding who is large enough to chart — otherwise Vanguard's eleven entities
    would take half the table — but the ids handed back have to be ones SF3 will
    actually match, and the family label is a name this module invented.
    """
    tolerances = tolerances or HoldingsTolerances()
    periods = quarters(frame)
    if not periods:
        return ()

    latest = register(frame, actions, periods[-1])
    filed = set(frame["investorid"].astype("string").dropna())

    picked: list[str] = []
    for investorid in latest["investorid"].head(tolerances.top_holders):
        members = FILER_FAMILIES.get(str(investorid))
        if members is None:
            picked.append(str(investorid))
        else:
            picked.extend(m for m in members if m in filed)
    return tuple(dict.fromkeys(picked))


# ---------------------------------------------------------------------------
# Ownership over time
# ---------------------------------------------------------------------------

OWNERSHIP_COLUMNS = ("quarter", "units", "shares_out", "owned", "holders")


def shares_outstanding(statements: pd.DataFrame) -> pd.DataFrame:
    """``quarter``/``shares`` from the quarterly DERIVED frame.

    ``Shares (Basic) × Share Factor`` — the same product ``logic/statements``
    calls ``shares_out``, and what Sharadar itself multiplies through for market
    cap. Indexed on **``Calendar Date``**, never ``Report Date``: a company on a
    52/53-week fiscal calendar closes a day or two earlier each year, so an exact
    join on the report date finds nothing. That is the regression the growth tab
    already carries a test for.

    The count is split-adjusted by the vendor, which is why SF3's units have to
    be brought onto the same basis before the two are divided.
    """
    empty = pd.DataFrame(columns=["quarter", "shares"])
    if statements is None or statements.empty:
        return empty

    basic = vocabulary.SF1_CAPTIONS["sharesbas"]
    factor = vocabulary.SF1_CAPTIONS["sharefactor"]
    if "Calendar Date" not in statements.columns or basic not in statements.columns:
        return empty

    shares = pd.to_numeric(statements[basic], errors="coerce")
    if factor in statements.columns:
        shares = shares * pd.to_numeric(statements[factor], errors="coerce").fillna(1.0)

    out = pd.DataFrame(
        {
            "quarter": pd.to_datetime(statements["Calendar Date"], errors="coerce"),
            "shares": shares,
        }
    )
    return (
        out.dropna(subset=["quarter", "shares"])
        .drop_duplicates(subset=["quarter"], keep="last")
        .sort_values("quarter")
        .reset_index(drop=True)
    )


def ownership_history(
    summary: pd.DataFrame,
    statements: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    tolerances: HoldingsTolerances | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """Institutional ownership as a share of the company, quarter by quarter.

    Columns :data:`OWNERSHIP_COLUMNS`. ``owned`` is a fraction and is ``NaN``
    wherever the share count for that quarter is missing — the ratio is not
    guessed, and a quarter with 13F data but no filed share count reports the
    units and leaves the percentage blank.

    Both sides are on today's split basis: SF3A's ``shrunits`` is filed and gets
    the split factor, ``sharesbas`` is already restated.
    """
    tolerances = tolerances or HoldingsTolerances()
    today = today or date.today()

    if summary is None or summary.empty or "date" not in summary.columns:
        return pd.DataFrame(columns=list(OWNERSHIP_COLUMNS))

    frame = summary.copy()
    frame["quarter"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["quarter"])
    if frame.empty:
        return pd.DataFrame(columns=list(OWNERSHIP_COLUMNS))

    cutoff = pd.Timestamp(today) - pd.DateOffset(years=tolerances.lookback_years)
    frame = frame[frame["quarter"] >= cutoff]

    filed = pd.to_numeric(frame.get("shrunits"), errors="coerce")
    frame["units"] = filed * vocabulary.split_factors(actions, frame["quarter"])
    frame["holders"] = pd.to_numeric(frame.get("shrholders"), errors="coerce")

    shares = shares_outstanding(statements)
    merged = frame.merge(shares, on="quarter", how="left")
    merged["shares_out"] = merged["shares"]
    merged["owned"] = merged["units"] / merged["shares_out"]

    return (
        merged[list(OWNERSHIP_COLUMNS)]
        .sort_values("quarter")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# The holder table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Holder:
    """One institution's position, and what it did with it."""

    investorid: str
    name: str
    units: float
    value: float
    prior_units: float | None = None
    change: float | None = None
    owned: float | None = None
    #: Split-adjusted units per quarter over the sparkline window, ``None``
    #: where the holder filed nothing. Absence is never zero and never bridged.
    series: tuple[float | None, ...] = ()
    is_index: bool = False
    index_house: str = ""


@dataclass(frozen=True)
class HolderTable:
    rows: tuple[Holder, ...] = ()
    periods: tuple[pd.Timestamp, ...] = ()
    as_of: pd.Timestamp | None = None
    total_value: float = 0.0
    shown_value: float = 0.0
    #: Units across **every** holder, and how that moved. The table's foot row.
    #: This is the general defence where :data:`FILER_FAMILIES` is the specific
    #: one: any single holder's collapse is read beside the aggregate, so a
    #: reorganisation by a manager the map does not name still cannot be mistaken
    #: for the register emptying out.
    total_units: float = 0.0
    total_change: float | None = None
    holders: int = 0
    reason: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.rows)


def top_holders(
    current: pd.DataFrame,
    history: pd.DataFrame,
    actions: pd.DataFrame,
    names: pd.DataFrame | None = None,
    shares_out: float | None = None,
    *,
    tolerances: HoldingsTolerances | None = None,
) -> HolderTable:
    """The largest holders, their quarter-on-quarter move, and their series.

    ``current`` carries every holder for the last two quarters; ``history``
    carries the top holders across every quarter, which is what the sparklines
    are drawn from. Both are split-adjusted here.

    A holder absent from a quarter contributes ``None`` to its series rather
    than zero. Over a twelve-quarter window that is the common case, not an edge
    one — three of Apple's top twenty have gaps and eight of Crocs' do.
    """
    tolerances = tolerances or HoldingsTolerances()

    if not usable(current):
        return HolderTable(
            reason=(
                "Sharadar returned no 13F positions for this company. SF3 covers "
                "form 13F from June 2013; a company with no rows is held by no "
                "manager large enough to file, which is an answer rather than a "
                "gap."
            )
        )

    periods = quarters(current)
    latest = periods[-1]
    prior = periods[-2] if len(periods) > 1 else None

    now = register(current, actions, latest, names)
    if now.empty:
        return HolderTable(reason=f"No 13F positions reported for {latest.date()}.")
    before = (
        register(current, actions, prior)
        if prior is not None
        else pd.DataFrame(columns=["investorid", "units"])
    )
    prior_units = dict(zip(before["investorid"], before["units"]))

    window = quarters(history)[-tolerances.sparkline_quarters :]
    series = _series_by_holder(history, actions, window)

    total_value = float(now["value"].sum(skipna=True))
    total_units = float(now["units"].sum(skipna=True))
    prior_total = float(before["units"].sum(skipna=True)) if not before.empty else 0.0
    total_change = (
        total_units / prior_total - 1.0 if prior_total > 0 else None
    )
    picked = now.head(tolerances.top_holders)

    rows = []
    for entry in picked.itertuples():
        was = prior_units.get(entry.investorid)
        change = None
        if was is not None and was > 0:
            change = float(entry.units) / float(was) - 1.0
        rows.append(
            Holder(
                investorid=str(entry.investorid),
                name=str(entry.name),
                units=float(entry.units),
                value=float(entry.value),
                prior_units=None if was is None else float(was),
                change=change,
                owned=(
                    float(entry.units) / shares_out
                    if shares_out and shares_out > 0
                    else None
                ),
                series=series.get(str(entry.investorid), (None,) * len(window)),
                is_index=str(entry.investorid) in INDEX_MANAGERS,
                index_house=INDEX_MANAGERS.get(str(entry.investorid), ""),
            )
        )

    return HolderTable(
        rows=tuple(rows),
        periods=tuple(window),
        as_of=latest,
        total_value=total_value,
        shown_value=float(picked["value"].sum(skipna=True)),
        total_units=total_units,
        total_change=total_change,
        holders=len(now),
    )


def _series_by_holder(
    history: pd.DataFrame, actions: pd.DataFrame, window: list[pd.Timestamp]
) -> dict[str, tuple[float | None, ...]]:
    """Split-adjusted units per quarter, per holder, aligned to ``window``.

    Keyed the same way :func:`register` keys its rows — on the family label where
    there is one, summing that family's entities per quarter. Without the sum a
    collapsed row's series is looked up under a label no filing carries and every
    quarter comes back empty, which draws the largest holder on the page as
    having no history at all.
    """
    if not usable(history) or not window:
        return {}
    frame = history.copy()
    frame["units"] = split_adjusted_units(frame, actions)
    frame["quarter"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["key"] = (
        frame["investorid"].map(FAMILY_OF).fillna(frame["investorid"]).astype(str)
    )

    out: dict[str, tuple[float | None, ...]] = {}
    for key, group in frame.groupby("key"):
        by_quarter = group.groupby("quarter")["units"].sum()
        out[str(key)] = tuple(
            None if q not in by_quarter.index or pd.isna(by_quarter[q])
            else float(by_quarter[q])
            for q in window
        )
    return out


# ---------------------------------------------------------------------------
# Churn and concentration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Churn:
    """What the register did between the last two quarters."""

    opened: int = 0
    closed: int = 0
    added: int = 0
    trimmed: int = 0
    unchanged: int = 0
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.reason


def churn(
    current: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    tolerances: HoldingsTolerances | None = None,
) -> Churn:
    """Positions opened, closed, added to, trimmed and left alone.

    The five buckets **partition the union of both quarters' holders exactly**.
    Moves smaller than :attr:`HoldingsTolerances.material_change` count as
    unchanged: 13F counts are rounded, and a manager whose position drifts a
    tenth of a percent has not decided anything.
    """
    tolerances = tolerances or HoldingsTolerances()

    if not usable(current):
        return Churn(reason="No 13F positions for this company.")
    periods = quarters(current)
    if len(periods) < 2:
        return Churn(
            reason=(
                "Only one quarter of 13F data was fetched, so there is nothing "
                "to compare it against."
            )
        )

    now = register(current, actions, periods[-1]).set_index("investorid")["units"]
    before = register(current, actions, periods[-2]).set_index("investorid")["units"]

    opened = now.index.difference(before.index)
    closed = before.index.difference(now.index)
    both = now.index.intersection(before.index)

    added = trimmed = unchanged = 0
    for investorid in both:
        was, is_now = float(before[investorid]), float(now[investorid])
        if was <= 0:
            unchanged += 1
            continue
        move = is_now / was - 1.0
        if move > tolerances.material_change:
            added += 1
        elif move < -tolerances.material_change:
            trimmed += 1
        else:
            unchanged += 1

    return Churn(
        opened=len(opened),
        closed=len(closed),
        added=added,
        trimmed=trimmed,
        unchanged=unchanged,
    )


def concentration(
    current: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    tolerances: HoldingsTolerances | None = None,
) -> tuple[Sourced, Sourced]:
    """Share of reported institutional value held by the largest N, and the HHI.

    Both on **value**, not units: two share classes are not addable, and value is
    what the register reports in one currency.
    """
    tolerances = tolerances or HoldingsTolerances()
    top_n = tolerances.concentration_top_n

    if not usable(current):
        absent = Sourced.unavailable(SOURCE, "No 13F positions for this company.")
        return absent, absent

    periods = quarters(current)
    now = register(current, actions, periods[-1])
    total = float(now["value"].sum(skipna=True))
    if total <= 0:
        absent = Sourced.unavailable(
            SOURCE, "The reported positions carry no value to concentrate."
        )
        return absent, absent

    top = float(now["value"].head(top_n).sum(skipna=True))
    shares = (now["value"] / total).dropna()
    hhi = float((shares**2).sum() * 10_000)
    as_of = periods[-1].date()

    return (
        Sourced.derived(
            top / total,
            SOURCE,
            f"value held by the largest {top_n} ÷ total reported value",
            as_of=as_of,
        ),
        Sourced.derived(
            hhi,
            SOURCE,
            "Herfindahl-Hirschman index over each holder's share of value",
            as_of=as_of,
        ),
    )


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def reconciliation(
    current: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    tolerances: HoldingsTolerances | None = None,
) -> pd.DataFrame:
    """Every holder in the latest quarter, and which band they fell in.

    The bands **partition the register exactly**, so a reader can see that a
    twenty-row table is twenty of six thousand holders on purpose, and how much
    of the value those twenty are.
    """
    tolerances = tolerances or HoldingsTolerances()
    columns = ["Band", "Holders", "Shares", "Value", "Share of value"]
    if not usable(current):
        return pd.DataFrame(columns=columns)

    periods = quarters(current)
    now = register(current, actions, periods[-1])
    total = float(now["value"].sum(skipna=True))
    if total <= 0:
        return pd.DataFrame(columns=columns)

    top = now.head(tolerances.top_holders)
    index_rows = now[now["investorid"].isin(INDEX_MANAGERS)]
    rest = now.iloc[tolerances.top_holders :]

    def band(label: str, rows: pd.DataFrame) -> dict:
        value = float(rows["value"].sum(skipna=True))
        return {
            "Band": label,
            "Holders": len(rows),
            "Shares": float(rows["units"].sum(skipna=True)),
            "Value": value,
            "Share of value": value / total,
        }

    return pd.DataFrame(
        [
            band(f"Shown — the largest {tolerances.top_holders}", top),
            band("Not shown", rest),
            band("— of which index managers (whole register)", index_rows),
        ],
        columns=columns,
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Holdings:
    """Everything the tab draws, built once per load."""

    ownership: pd.DataFrame = field(default_factory=pd.DataFrame)
    holders: HolderTable = field(default_factory=HolderTable)
    churn: Churn = field(default_factory=Churn)
    top_share: Sourced = field(
        default_factory=lambda: Sourced.unavailable(SOURCE, "Not computed.")
    )
    hhi: Sourced = field(
        default_factory=lambda: Sourced.unavailable(SOURCE, "Not computed.")
    )
    reconciliation: pd.DataFrame = field(default_factory=pd.DataFrame)
    as_of: pd.Timestamp | None = None

    @property
    def ok(self) -> bool:
        return self.holders.ok

    @property
    def deadline(self) -> date | None:
        """The last day managers may file the quarter on screen."""
        if self.as_of is None:
            return None
        return (self.as_of + pd.Timedelta(days=FILING_LAG_DAYS)).date()

    def days_old(self, today: date) -> int | None:
        """Days since the filing deadline for the quarter on screen.

        Negative before the deadline, when managers are still filing and the
        register is incomplete rather than merely old.
        """
        if self.deadline is None:
            return None
        return int((pd.Timestamp(today) - pd.Timestamp(self.deadline)).days)

    @property
    def latest(self) -> dict[str, float | None]:
        """The last quarter's ``owned``, ``units`` and ``holders``, ``None`` where absent."""
        if self.ownership.empty:
            return {}
        row = self.ownership.iloc[-1]
        return {
            key: None if pd.isna(row.get(key)) else float(row[key])
            for key in ("owned", "units", "holders")
        }


def build(
    *,
    summary: pd.DataFrame,
    current: pd.DataFrame,
    history: pd.DataFrame,
    names: pd.DataFrame | None = None,
    actions: pd.DataFrame | None = None,
    statements: pd.DataFrame | None = None,
    tolerances: HoldingsTolerances | None = None,
    today: date | None = None,
) -> Holdings:
    """Assemble the tab from the four fetches the view makes.

    Keyword-only and pure, on the model of ``growth.build_comparison``: nothing
    here reaches for a network or a session, so the whole tab is testable by
    handing it five frames.
    """
    tolerances = tolerances or HoldingsTolerances()
    today = today or date.today()
    actions = pd.DataFrame() if actions is None else actions

    owned = ownership_history(
        summary, statements, actions, tolerances=tolerances, today=today
    )
    shares_out = None
    if not owned.empty and owned["shares_out"].notna().any():
        shares_out = float(owned["shares_out"].dropna().iloc[-1])

    table = top_holders(
        current, history, actions, names, shares_out, tolerances=tolerances
    )
    top_share, hhi = concentration(current, actions, tolerances=tolerances)
    return Holdings(
        ownership=owned,
        holders=table,
        churn=churn(current, actions, tolerances=tolerances),
        top_share=top_share,
        hhi=hhi,
        reconciliation=reconciliation(current, actions, tolerances=tolerances),
        as_of=table.as_of,
    )
