"""``app/logic/research/holdings.py`` — the corrections, and why each is there.

The fixture is the eleven-holder register in ``conftest``: six quarters, a 2:1
split between the third and fourth, and a Vanguard filing-entity split between
the fifth and sixth from last. Every number below is hand-checkable.

The rule under test throughout is the same one the insiders module carries: the
obvious read of this data is not noisy, it is **categorically wrong**, and each
section isolates one of the four reasons.

No test here touches the network.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.logic.research import holdings
from app.provenance import Origin
from tests.conftest import sf3_splits

TODAY = date(2026, 8, 5)

#: The six quarters the fixture covers.
Q = [pd.Timestamp(d) for d in (
    "2025-03-31", "2025-06-30", "2025-09-30",
    "2025-12-31", "2026-03-31", "2026-06-30",
)]


@pytest.fixture
def actions():
    return sf3_splits()


def _upto(frame: pd.DataFrame, quarter: pd.Timestamp) -> pd.DataFrame:
    """The register as it stood at ``quarter`` — everything filed up to it."""
    return frame[pd.to_datetime(frame["date"]) <= quarter]


# ---------------------------------------------------------------------------
# Splits — filed counts are never restated, share counts are
# ---------------------------------------------------------------------------


def test_a_split_is_not_reported_as_a_purchase(holders_frame, actions):
    """BlackRock's Tesla position reads 7,720 → 38,020 across the 2020
    five-for-one, which a naive change reports as a +392% purchase. Here the
    fixture's index manager holds exactly the same stake throughout."""
    rows = holders_frame[holders_frame["investorid"] == "BLKROK"].copy()
    filed = pd.to_numeric(rows["units"])
    assert filed.min() == 1_000_000 and filed.max() == 2_000_000  # the step

    adjusted = holdings.split_adjusted_units(rows, actions)
    assert adjusted.nunique() == 1
    assert adjusted.iloc[0] == pytest.approx(2_000_000)


def test_the_adjustment_multiplies_where_the_insider_one_divides(actions):
    """A price filed before a split is divided by it; a share count is
    multiplied. Getting the direction backwards is the whole failure mode."""
    from app.logic.research import insiders

    before = pd.DataFrame(
        {"date": [pd.Timestamp("2025-06-30")], "units": [100.0],
         "transactiondate": [pd.Timestamp("2025-06-30")],
         "transactionpricepershare": [100.0]}
    )
    assert holdings.split_adjusted_units(before, actions).iloc[0] == 200.0
    assert insiders.split_adjusted_price(before, actions).iloc[0] == 50.0


def test_a_company_that_never_split_keeps_its_filed_counts(holders_frame):
    unchanged = holdings.split_adjusted_units(holders_frame, pd.DataFrame())
    assert unchanged.equals(pd.to_numeric(holders_frame["units"]))


def test_value_is_never_split_adjusted(holders_frame, actions):
    """A split changes the count and the price together, so value is already
    continuous. Adjusting it would break a correct number."""
    blk = holders_frame[holders_frame["investorid"] == "BLKROK"]
    assert blk["value"].nunique() == 1


# ---------------------------------------------------------------------------
# Filer restructuring — the trap that lands on every ticker at once
# ---------------------------------------------------------------------------


def test_a_filing_entity_split_is_not_reported_as_a_sale(
    holders_frame, actions, investor_names
):
    """The fixture reproduces Vanguard's 2026-Q1 reorganisation with Vanguard's
    real investorids, so this exercises the shipped ``FILER_FAMILIES`` map
    rather than a test-only stand-in.

    Live, Apple reads −33.1% for ``VANGRD`` alone and −5.9% for the family.
    """
    upto = _upto(holders_frame, Q[4])
    alone_before = holdings.register(upto, actions, Q[3], collapse=False)
    alone_after = holdings.register(upto, actions, Q[4], collapse=False)
    naive = (
        float(alone_after.set_index("investorid").loc["VANGRD", "units"])
        / float(alone_before.set_index("investorid").loc["VANGRD", "units"])
        - 1.0
    )

    family_before = holdings.register(upto, actions, Q[3], investor_names)
    family_after = holdings.register(upto, actions, Q[4], investor_names)
    truth = (
        float(family_after.set_index("investorid").loc["Vanguard", "units"])
        / float(family_before.set_index("investorid").loc["Vanguard", "units"])
        - 1.0
    )

    assert naive == pytest.approx(-2 / 3, abs=0.01)
    assert truth == pytest.approx(-0.0333, abs=0.001)


def test_a_collapsed_family_is_labelled_as_a_sum_not_as_a_filing(
    holders_frame, actions, investor_names
):
    """The surviving row must not look like an entity that filed."""
    now = holdings.register(_upto(holders_frame, Q[4]), actions, Q[4], investor_names)
    row = now[now["investorid"] == "Vanguard"].iloc[0]
    assert row["name"] == "Vanguard (all filings)"
    assert "VANGRD" not in set(now["investorid"])


def test_the_family_map_is_curated_and_never_derived():
    """A leading-token heuristic was tried and measured: 32 flags on Apple,
    almost all spurious — BANK OF MARIN paired with BANK OF AMERICA. Nothing
    here may infer a family from the data."""
    import ast
    import inspect

    source = inspect.getsource(holdings)
    assert isinstance(holdings.FILER_FAMILIES, dict)
    called = {
        node.attr for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
    }
    assert "startswith" not in called
    assert not any("token" in name for name in dir(holdings))


def test_the_aggregate_total_is_carried_beside_the_holders(
    holders_frame, actions, investor_names
):
    """The general defence where the family map is the specific one: a single
    holder's collapse is always read against the whole register."""
    table = holdings.top_holders(
        _upto(holders_frame, Q[4]), holders_frame, actions, investor_names
    )
    assert table.total_units == pytest.approx(5_470_000)
    assert table.total_change == pytest.approx(5_470_000 / 5_460_000 - 1.0)


# ---------------------------------------------------------------------------
# Value is a market value, so it moves when the stock does
# ---------------------------------------------------------------------------


def test_a_holder_who_traded_nothing_is_not_a_seller(actions, investor_names):
    """Halve the price and hold the units: value halves and nothing was sold.
    A change computed on value would name this holder a 50% seller."""
    frame = pd.DataFrame(
        {
            "investorid": ["STEADY", "STEADY"],
            "date": [Q[4], Q[5]],
            "units": [1_000.0, 1_000.0],
            "value": [100_000.0, 50_000.0],
        }
    )
    table = holdings.top_holders(frame, frame, pd.DataFrame(), investor_names)
    assert table.rows[0].change == pytest.approx(0.0)


def test_the_holder_record_carries_no_change_in_value():
    """There is no ``delta_value`` and there must not be one."""
    fields = holdings.Holder.__dataclass_fields__
    assert "change" in fields
    assert not [name for name in fields if "value" in name and "change" in name]


# ---------------------------------------------------------------------------
# Ownership — a ratio across two bases
# ---------------------------------------------------------------------------


def _shares(quarters, count) -> pd.DataFrame:
    from app.data import sharadar_client as sc

    return pd.DataFrame(
        {
            "Calendar Date": quarters,
            sc.SF1_CAPTIONS["sharesbas"]: [count] * len(quarters),
            sc.SF1_CAPTIONS["sharefactor"]: [1.0] * len(quarters),
        }
    )


def test_ownership_puts_units_and_the_share_count_on_one_basis(
    holdings_summary, actions
):
    """SF1 restates ``sharesbas`` for splits and SF3 never restates ``units``,
    so an unadjusted ratio is wrong by the split factor at every pre-split
    quarter — here by exactly two."""
    shares = _shares(Q, 10_000_000.0)
    owned = holdings.ownership_history(holdings_summary, shares, actions, today=TODAY)

    pre = owned[owned["quarter"] == Q[0]].iloc[0]
    post = owned[owned["quarter"] == Q[3]].iloc[0]
    # 2,580 thousand filed pre-split, doubled onto today's basis.
    assert pre["units"] == pytest.approx(5_160_000)
    assert pre["owned"] == pytest.approx(0.516)
    assert post["units"] == pytest.approx(5_460_000)

    naive = 2_580_000 / 10_000_000
    assert pre["owned"] == pytest.approx(naive * 2)


def test_ownership_joins_on_the_calendar_date_not_the_report_date(
    holdings_summary, actions
):
    """A company on a 52/53-week fiscal calendar closes a day or two earlier
    each year, so an exact join on the report date finds nothing and every
    ownership reading goes blank. That shipped once on the growth tab."""
    from app.data import sharadar_client as sc

    shares = _shares(Q, 10_000_000.0)
    shares["Report Date"] = [q - pd.Timedelta(days=3) for q in Q]

    owned = holdings.ownership_history(holdings_summary, shares, actions, today=TODAY)
    assert owned["owned"].notna().all()

    misaligned = shares.rename(columns={"Calendar Date": "unused"}).rename(
        columns={"Report Date": "Calendar Date"}
    )
    assert holdings.ownership_history(
        holdings_summary, misaligned, actions, today=TODAY
    )["owned"].isna().all()


def test_a_quarter_with_no_share_count_reports_units_and_no_percentage(
    holdings_summary, actions
):
    """The ratio is never guessed."""
    shares = _shares(Q[:3], 10_000_000.0)
    owned = holdings.ownership_history(holdings_summary, shares, actions, today=TODAY)
    late = owned[owned["quarter"] == Q[5]].iloc[0]
    assert late["units"] > 0
    assert pd.isna(late["owned"])


def test_ownership_above_one_hundred_percent_is_left_alone(
    holdings_summary, actions
):
    """Lent shares are reported by the lender and by whoever bought them, so
    over-100% is real — Etsy reads 114.3% live. Clamping it would be the
    no-silent-fallbacks rule broken in a new direction."""
    owned = holdings.ownership_history(
        holdings_summary, _shares(Q, 1_000_000.0), actions, today=TODAY
    )
    assert owned["owned"].max() > 1.0


# ---------------------------------------------------------------------------
# The holder table and its sparklines
# ---------------------------------------------------------------------------


def test_a_gap_in_a_holders_history_is_neither_bridged_nor_zeroed(
    holders_frame, actions, investor_names
):
    """The opposite of ``insiders.periodic_flow``, on purpose. A manager absent
    from a quarter may have sold out, dropped below the filing threshold, been
    acquired or filed late — none of which is "held nothing"."""
    table = holdings.top_holders(
        holders_frame, holders_frame, actions, investor_names
    )
    gappy = next(r for r in table.rows if r.investorid == "GAPPY")
    assert gappy.series[1] is None and gappy.series[2] is None
    assert 0.0 not in gappy.series
    assert gappy.series[0] == pytest.approx(300_000)


def test_the_two_tabs_disagree_about_gaps_deliberately(holders_frame, actions):
    """``periodic_flow`` fills a quiet quarter with a zero and this fills it with
    nothing. The two look alike and must not be reconciled: there a quiet quarter
    means the insiders traded, here an absent quarter means the manager did not
    file, and those are not the same claim."""
    from app.logic.research import insiders

    quiet = insiders.periodic_flow(pd.DataFrame(), today=TODAY)
    assert (quiet["net"] == 0).all()

    table = holdings.top_holders(holders_frame, holders_frame, actions)
    gappy = next(r for r in table.rows if r.investorid == "GAPPY")
    assert None in gappy.series


def test_a_collapsed_family_has_a_history_of_its_own(
    holders_frame, actions, investor_names
):
    """The series is keyed the way the row is — on the family label, summing its
    entities. Keyed on a raw investorid instead, the collapsed row is looked up
    under a name no filing carries and the largest holder on the page draws as
    having no history at all. That shipped and was caught against live data:
    Apple's Vanguard row read twelve gaps out of twelve."""
    table = holdings.top_holders(
        holders_frame, holders_frame, actions, investor_names
    )
    vanguard = next(r for r in table.rows if r.investorid == "Vanguard")
    assert None not in vanguard.series[-2:]
    # 400 + 760 thousand filed, on a post-split basis.
    assert vanguard.series[-1] == pytest.approx(1_160_000)


def test_the_series_is_split_adjusted(holders_frame, actions, investor_names):
    """Otherwise every line with a split in view has a step no one took."""
    table = holdings.top_holders(
        holders_frame, holders_frame, actions, investor_names
    )
    index_row = next(r for r in table.rows if r.investorid == "BLKROK")
    assert len(set(index_row.series)) == 1


def test_an_index_manager_is_marked(holders_frame, actions, investor_names):
    table = holdings.top_holders(
        holders_frame, holders_frame, actions, investor_names
    )
    marked = {r.investorid for r in table.rows if r.is_index}
    assert "BLKROK" in marked
    assert "ACTIVE" not in marked


def test_an_unlisted_manager_is_never_assumed_active(
    holders_frame, actions, investor_names
):
    """The list is curated and incomplete; absence from it is not a claim."""
    table = holdings.top_holders(
        holders_frame, holders_frame, actions, investor_names
    )
    unlisted = next(r for r in table.rows if r.investorid == "ACTIVE")
    assert unlisted.is_index is False
    assert unlisted.index_house == ""


def test_an_unresolved_investorid_shows_the_code_not_a_blank(
    holders_frame, actions
):
    table = holdings.top_holders(holders_frame, holders_frame, actions, None)
    assert all(r.name for r in table.rows)


def test_a_company_with_no_filings_says_so():
    table = holdings.top_holders(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )
    assert not table.ok
    assert "June 2013" in table.reason


# ---------------------------------------------------------------------------
# Churn
# ---------------------------------------------------------------------------


def test_the_churn_buckets_partition_the_register(holders_frame, actions):
    """Ten holders in the final quarter once Vanguard's two entities are summed:
    NEWCO opened, EXITED closed, ADDER added, ACTIVE and TRIMMER trimmed, the
    remaining six unchanged."""
    result = holdings.churn(holders_frame, actions)
    assert (result.opened, result.closed) == (1, 1)
    assert (result.added, result.trimmed, result.unchanged) == (1, 2, 6)
    # Opened, added, trimmed and unchanged account for the current quarter
    # exactly; closed is the only bucket counted off the prior one.
    assert result.opened + result.added + result.trimmed + result.unchanged == 10


def test_a_move_under_the_floor_is_unchanged_not_added(holders_frame, actions):
    """A holder drifting 0.2% has not decided anything; 13F counts are rounded."""
    result = holdings.churn(holders_frame, actions)
    strict = holdings.churn(
        holders_frame, actions,
        tolerances=holdings.HoldingsTolerances(material_change=0.0),
    )
    assert result.unchanged == 6
    # STEADY's +0.2% only counts as a decision once the floor is removed.
    assert strict.added == result.added + 1
    assert strict.unchanged == result.unchanged - 1


def test_a_split_between_two_quarters_does_not_make_everyone_a_buyer(
    holders_frame, actions
):
    """Q3 → Q4 straddles the 2:1. Unadjusted every holder doubles and the whole
    register reads as buying; adjusted, only the one that really bought does."""
    upto = _upto(holders_frame, Q[3])
    result = holdings.churn(upto, actions)
    assert result.unchanged == 8
    # ACTIVE is the only genuine buyer that quarter, 800 -> 900 adjusted.
    assert result.added == 1
    assert result.trimmed == 0

    naive = holdings.churn(upto, pd.DataFrame())
    assert naive.added == 9 and naive.unchanged == 0


def test_one_quarter_of_data_says_so_rather_than_reporting_no_change(
    holders_frame, actions
):
    upto = _upto(holders_frame, Q[0])
    result = holdings.churn(upto, actions)
    assert not result.ok
    assert "nothing to compare" in result.reason


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------


def test_concentration_is_taken_on_value(holders_frame, actions):
    top, hhi = holdings.concentration(holders_frame, actions)
    assert top.origin is Origin.DERIVED
    assert 0 < top.value <= 1.0
    assert hhi.value > 0


def test_concentration_says_so_when_there_is_nothing_to_concentrate():
    top, hhi = holdings.concentration(pd.DataFrame(), pd.DataFrame())
    assert top.value is None and hhi.value is None
    assert top.origin is Origin.UNAVAILABLE


# ---------------------------------------------------------------------------
# Reconciliation and tolerances
# ---------------------------------------------------------------------------


def test_the_reconciliation_accounts_for_every_holder(holders_frame, actions):
    small = holdings.HoldingsTolerances(top_holders=3)
    panel = holdings.reconciliation(holders_frame, actions, tolerances=small)
    shown = panel[panel["Band"].str.startswith("Shown")].iloc[0]
    hidden = panel[panel["Band"] == "Not shown"].iloc[0]
    assert shown["Holders"] == 3
    assert shown["Holders"] + hidden["Holders"] == 10


def test_the_tolerances_are_not_the_insider_ones():
    from app.logic.research import insiders

    assert holdings.HoldingsTolerances is not insiders.InsiderTolerances
    assert not hasattr(holdings.HoldingsTolerances(), "cluster_min_buyers")


def test_every_threshold_is_injectable(holders_frame, actions, investor_names):
    tight = holdings.HoldingsTolerances(top_holders=2)
    table = holdings.top_holders(
        holders_frame, holders_frame, actions, investor_names, tolerances=tight
    )
    assert len(table.rows) == 2


def test_the_module_makes_no_request():
    import ast
    import inspect

    source = inspect.getsource(holdings)
    assert "import streamlit" not in source
    called = {
        node.attr for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
    }
    assert not {name for name in called if name.startswith("fetch")}
