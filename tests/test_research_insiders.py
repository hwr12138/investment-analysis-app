"""``app/logic/research/insiders.py`` — the exclusions, and why each one is there.

The fixture is the sixteen-row company in ``conftest``, priced at $100 so every
total is checkable by hand. The rule under test throughout is that **the obvious
arithmetic returns the wrong sign**, and each section isolates one of the three
reasons it does.

No test here touches the network.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.logic.research import insiders
from app.provenance import Origin

TODAY = date(2026, 8, 5)


def _window(frame, months=12, **kwargs):
    return insiders.trailing_flow(frame, months, today=TODAY, **kwargs)


def _frame(rows: list[dict]) -> pd.DataFrame:
    """A minimal SF2 frame, for cases the fixture company does not carry."""
    base = {
        "ownername": "SOMEONE",
        "transactiondate": pd.Timestamp("2026-06-01"),
        "filingdate": pd.Timestamp("2026-06-03"),
        "transactioncode": "P",
        "transactionshares": 1_000.0,
        "transactionpricepershare": 100.0,
        "transactionvalue": 100_000.0,
        "sharesownedfollowingtransaction": 1_000.0,
        "securityadcode": "NA",
        "securitytitle": "Com",
        "directorindirect": "D",
        "isdirector": "Y",
        "isofficer": "N",
        "istenpercentowner": "N",
        "officertitle": None,
        "rownum": 1,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


# ---------------------------------------------------------------------------
# The sign — the reason this module exists
# ---------------------------------------------------------------------------


def test_the_total_over_every_code_has_the_wrong_sign(insider_frame):
    """The whole point. Summed over every transaction code the window reads as
    heavy selling; over open-market codes alone it is net buying.

    This is TSLA in miniature, where the same mistake read -$4,218m against a
    true +$220m, flipped by one holder under a non-market code.
    """
    trades = insider_frame[~insiders.is_holdings_only(insider_frame)]
    inside, _ = insiders._window(trades, 12, TODAY)
    naive = insiders.signed_flow(inside).sum(skipna=True)

    assert naive == pytest.approx(-789_350_000)
    assert _window(insider_frame).net.value == pytest.approx(13_350_000)
    assert naive < 0 < _window(insider_frame).net.value


def test_flow_is_shares_times_price_not_a_sum_of_transaction_value(insider_frame):
    """``transactionvalue`` is unsigned, so summing it counts a sale as a buy.

    Wrong independently of the code filter: over the same window's P and S rows
    it still reads +26,800,000 against a true +13,350,000.
    """
    inside, _ = insiders._window(insider_frame, 12, TODAY)
    market = inside[insiders.is_open_market(inside)]
    assert market["transactionvalue"].sum() == pytest.approx(26_800_000)
    assert insiders.signed_flow(market).sum() == pytest.approx(13_350_000)


def test_a_disposal_is_negative_though_its_reported_value_is_positive(insider_frame):
    sale = insider_frame[
        (insider_frame["ownername"] == "JANE OFFICER")
        & (insider_frame["transactionshares"] == -60_000)
    ]
    assert sale["transactionvalue"].iloc[0] == 6_000_000
    assert insiders.signed_flow(sale).iloc[0] == pytest.approx(-6_000_000)


def test_an_unpriced_row_has_no_flow_rather_than_a_zero_one(insider_frame):
    """A grant with no price has no dollar value. Zero is not the answer."""
    grant = insider_frame[insider_frame["transactioncode"] == "A"]
    assert grant["transactionpricepershare"].isna().all()
    assert insiders.signed_flow(grant).isna().all()


def test_compensation_mechanics_are_excluded_from_the_flow(insider_frame):
    """Grants, exercises and withholding outnumber real purchases 45:1 live."""
    window = _window(insider_frame)
    market = insider_frame[insiders.is_open_market(insider_frame)]
    assert set(market["transactioncode"]) == {"P", "S"}
    assert window.bought.value == pytest.approx(20_050_000)
    assert window.sold.value == pytest.approx(-6_700_000)


def test_the_code_groups_are_disjoint():
    groups = (
        insiders.OPEN_MARKET_CODES,
        insiders.COMPENSATION_CODES,
        insiders.TRANSFER_CODES,
    )
    assert sum(len(g) for g in groups) == len(insiders.KNOWN_CODES)


def test_a_ten_percent_owner_is_labelled_rather_than_filtered(insider_frame):
    """The code set fixes the sign; dropping the role would discard an
    activist's open-market purchases, which are the most informative rows here."""
    holder = insider_frame[insider_frame["ownername"] == "BIGHOLDER LP"]
    assert "10% owner" in insiders.role_of(holder).iloc[0]
    # Still present in the frame — nothing removed it.
    assert not holder.empty


def test_a_single_filer_dominating_the_window_is_flagged(insider_frame):
    window = _window(insider_frame)
    assert window.top_owner == "JANE OFFICER"
    assert window.concentration == pytest.approx(0.6804, abs=1e-3)
    assert "one person's decision" in window.concentration_note


def test_a_diffuse_window_carries_no_concentration_note(insider_frame):
    loose = insiders.InsiderTolerances(concentration_threshold=0.95)
    assert _window(insider_frame, tolerances=loose).concentration_note == ""


# ---------------------------------------------------------------------------
# Holdings-only rows
# ---------------------------------------------------------------------------


def test_a_bare_security_code_is_a_holding_not_a_trade(insider_frame):
    """28% of live rows. A bare ``N`` or ``D`` with no acquired/disposed letter."""
    holdings = insider_frame[insiders.is_holdings_only(insider_frame)]
    assert len(holdings) == 2
    assert set(holdings["securityadcode"]) == {"N", "D"}
    assert holdings["transactioncode"].isna().all()
    assert (holdings["transactionshares"] == 0).all()


def test_holdings_only_rows_never_reach_the_flow(insider_frame):
    window = _window(insider_frame)
    assert window.buy_trades + window.sell_trades == 8


def test_a_row_matching_only_one_holdings_test_is_not_reclassified():
    """Both conditions are tested, not either alone: a row satisfying one and
    not the other is a change in the feed and has to stay visible."""
    odd = _frame([{"securityadcode": "N", "transactioncode": "P"}])
    assert not insiders.is_holdings_only(odd).iloc[0]


# ---------------------------------------------------------------------------
# Splits — filed prices are never restated, the price chart always is
# ---------------------------------------------------------------------------


def _splits(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp(when), "action": "split", "value": ratio}
            for when, ratio in rows
        ]
    )


def test_a_price_filed_before_a_split_is_moved_onto_the_charts_basis():
    """Tesla is the live case: insiders filed November 2021 sales at $981-$1,237
    while SEP's close for the same month reads $338-$410, because a 3:1 split in
    August 2022 sits between them. Plotted raw, every pre-split marker floats
    above the line and implies a collapse that never happened."""
    trades = _frame(
        [
            {"transactiondate": pd.Timestamp("2021-11-10"),
             "transactionpricepershare": 1_050.0},
            {"transactiondate": pd.Timestamp("2022-09-10"),
             "transactionpricepershare": 300.0},
        ]
    )
    adjusted = insiders.split_adjusted_price(
        trades, _splits([("2022-08-25", 3.0)])
    )
    assert adjusted.iloc[0] == pytest.approx(350.0)
    assert adjusted.iloc[1] == pytest.approx(300.0)


def test_successive_splits_compound():
    """Tesla split 5:1 in 2020 and 3:1 in 2022 — a 2019 filing is off by 15."""
    trades = _frame(
        [{"transactiondate": pd.Timestamp("2019-06-01"),
          "transactionpricepershare": 1_500.0}]
    )
    adjusted = insiders.split_adjusted_price(
        trades, _splits([("2020-08-31", 5.0), ("2022-08-25", 3.0)])
    )
    assert adjusted.iloc[0] == pytest.approx(100.0)


def test_a_company_that_never_split_keeps_its_filed_prices(insider_frame):
    unchanged = insiders.split_adjusted_price(insider_frame, pd.DataFrame())
    assert unchanged.equals(
        pd.to_numeric(insider_frame["transactionpricepershare"], errors="coerce")
    )


def test_the_dollar_flow_is_never_split_adjusted(insider_frame):
    """Shares and price are both as-filed, so their product is the real amount
    and is invariant. Adjusting one side would break a correct number."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(insiders.signed_flow).lstrip())
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "split_adjusted_price" not in called


# ---------------------------------------------------------------------------
# Trailing windows
# ---------------------------------------------------------------------------


def test_no_filings_at_all_says_so():
    window = insiders.trailing_flow(pd.DataFrame(), 12, today=TODAY)
    assert window.net.value is None
    assert window.net.origin is Origin.UNAVAILABLE
    assert "coverage starts in 2008" in window.net.detail


def test_filings_but_none_in_the_window_names_the_most_recent(insider_frame):
    window = insiders.trailing_flow(insider_frame, 1, today=date(2026, 12, 31))
    assert window.net.value is None
    assert "No insider reported a transaction in the last 1 months" in window.net.detail
    assert "2026-08-01" in window.net.detail


def test_a_window_of_only_compensation_says_which_codes(insider_frame):
    only_comp = insider_frame[insider_frame["transactioncode"].isin({"A", "M", "F"})]
    window = insiders.trailing_flow(only_comp, 12, today=TODAY)
    assert window.net.value is None
    assert "compensation" in window.net.detail
    assert "M 1" in window.net.detail and "F 1" in window.net.detail


def test_open_market_rows_with_no_price_report_the_share_counts():
    """Never observed live — all 4,951 P/S rows sampled carried a price — but
    "never seen" is not "cannot happen"."""
    unpriced = _frame(
        [
            {"transactionpricepershare": None, "transactionshares": 500.0},
            {"transactionpricepershare": None, "transactionshares": -200.0,
             "transactioncode": "S"},
        ]
    )
    window = insiders.trailing_flow(unpriced, 12, today=TODAY)
    assert window.net.value is None
    assert "report no price per share" in window.net.detail
    assert "500 bought" in window.net.detail and "200 sold" in window.net.detail


def test_the_four_absences_do_not_share_a_reason_string(insider_frame):
    """A reader who sees "not reported" deserves to know which fact it was."""
    reasons = {
        insiders.trailing_flow(pd.DataFrame(), 12, today=TODAY).net.detail,
        insiders.trailing_flow(
            insider_frame, 1, today=date(2026, 12, 31)
        ).net.detail,
        insiders.trailing_flow(
            insider_frame[insider_frame["transactioncode"].isin({"A", "M", "F"})],
            12,
            today=TODAY,
        ).net.detail,
        insiders.trailing_flow(
            _frame([{"transactionpricepershare": None}]), 12, today=TODAY
        ).net.detail,
    }
    assert len(reasons) == 4


def test_the_windows_are_derived_never_reported(insider_frame):
    """No single SF2 column is the answer, and the one that looks like it is is
    the one that gets the sign backwards."""
    window = _window(insider_frame)
    assert window.net.origin is Origin.DERIVED
    assert "shares × price" in window.net.formula


def test_each_trailing_window_covers_its_own_span(insider_frame):
    three, six, twelve = insiders.trailing_windows(insider_frame, today=TODAY)
    assert (three.months, six.months, twelve.months) == (3, 6, 12)
    # April's sale falls outside three months but inside six.
    assert three.sold.value == pytest.approx(-6_500_000)
    assert six.sold.value == pytest.approx(-6_700_000)


# ---------------------------------------------------------------------------
# Quarterly flow
# ---------------------------------------------------------------------------


def test_a_quarter_with_no_open_market_trade_is_a_zero_not_a_gap(insider_frame):
    """Apple went five years without one open-market purchase. A hole in the
    chart would read as missing data, which is the one thing it is not."""
    quarters = insiders.periodic_flow(insider_frame, today=TODAY)
    assert len(quarters) == 21
    assert quarters["net"].notna().all()
    assert int((quarters["net"] != 0).sum()) == 2


def test_the_window_is_twenty_one_quarters_not_sixty_one_months(insider_frame):
    """Monthly is 41-89% empty at large caps and 1-of-61 at the smallest."""
    assert insiders.InsiderTolerances().periodic_quarters == 21
    assert len(insiders.periodic_flow(insider_frame, today=TODAY)) == 21


def test_buckets_are_on_the_transaction_date_not_the_filing_date(insider_frame):
    """LATE FILER traded on 30 June and filed on 2 July. The decision was taken
    in Q2, and bucketing on the filing date would move it to Q3."""
    quarters = insiders.periodic_flow(insider_frame, today=TODAY)
    q2 = quarters[quarters["period"] == pd.Timestamp("2026-06-30")].iloc[0]
    q3 = quarters[quarters["period"] == pd.Timestamp("2026-09-30")].iloc[0]
    assert q2["trades"] == 6
    assert q2["net"] == pytest.approx(19_310_000)
    assert q3["trades"] == 2


def test_the_bucket_the_load_date_falls_in_is_marked_partial(insider_frame):
    quarters = insiders.periodic_flow(insider_frame, today=TODAY)
    assert quarters["partial"].sum() == 1
    assert quarters["partial"].iloc[-1]
    assert quarters["period"].iloc[-1] == pd.Timestamp("2026-09-30")


def test_gross_is_kept_beside_the_net(insider_frame):
    """A quarter netting zero on $50m each way is not a quiet quarter, and one
    bar cannot say which it was."""
    quarters = insiders.periodic_flow(insider_frame, today=TODAY)
    q2 = quarters[quarters["period"] == pd.Timestamp("2026-06-30")].iloc[0]
    assert q2["bought"] == pytest.approx(20_010_000)
    assert q2["sold"] == pytest.approx(-700_000)


def test_a_company_with_no_filings_still_gets_the_full_grid():
    quarters = insiders.periodic_flow(pd.DataFrame(), today=TODAY)
    assert len(quarters) == 21
    assert (quarters["net"] == 0).all()


# ---------------------------------------------------------------------------
# Cluster buying
# ---------------------------------------------------------------------------


def test_three_buyers_inside_the_window_is_a_cluster(insider_frame):
    found = insiders.clusters(insider_frame, today=TODAY)
    assert len(found) == 1
    cluster = found[0]
    assert cluster.buyers == 3
    assert cluster.trades == 3
    assert cluster.start == date(2026, 6, 10)
    assert cluster.end == date(2026, 6, 18)
    assert cluster.value == pytest.approx(20_000_000)


def test_a_buy_below_the_minimum_value_does_not_count(insider_frame):
    """TINY BUYER bought $10,000 inside the window. A director topping up
    $4,000 is not evidence, and neither is this."""
    cluster = insiders.clusters(insider_frame, today=TODAY)[0]
    assert "TINY BUYER" not in cluster.owners


def test_a_buy_outside_the_window_starts_no_cluster(insider_frame):
    """SOLO BUYER is over the value floor but seven weeks late."""
    cluster = insiders.clusters(insider_frame, today=TODAY)[0]
    assert "SOLO BUYER" not in cluster.owners


def test_two_buyers_is_not_a_cluster():
    rows = _frame(
        [
            {"ownername": "A", "transactiondate": pd.Timestamp("2026-06-01"),
             "transactionshares": 1_000.0},
            {"ownername": "B", "transactiondate": pd.Timestamp("2026-06-03"),
             "transactionshares": 1_000.0},
        ]
    )
    assert insiders.clusters(rows, today=TODAY) == []


def test_one_person_buying_three_times_is_not_a_cluster():
    """Distinct owners are counted, not trades: that is one decision."""
    rows = _frame(
        [
            {"ownername": "A", "transactiondate": pd.Timestamp(d),
             "transactionshares": 1_000.0}
            for d in ("2026-06-01", "2026-06-03", "2026-06-05")
        ]
    )
    assert insiders.clusters(rows, today=TODAY) == []


def test_overlapping_windows_merge_into_one_episode():
    """One continuous buying run is one cluster, not a dozen sliding ones."""
    rows = _frame(
        [
            {"ownername": name, "transactiondate": pd.Timestamp(when),
             "transactionshares": 1_000.0}
            for name, when in [
                ("A", "2026-06-01"), ("B", "2026-06-10"),
                ("C", "2026-06-20"), ("D", "2026-06-28"),
            ]
        ]
    )
    found = insiders.clusters(rows, today=TODAY)
    assert len(found) == 1
    assert found[0].buyers == 4


def test_sells_never_form_a_cluster():
    """10b5-1 plans make executives sell in lockstep on dates chosen months
    earlier, so a sell cluster measures the plan rather than a view."""
    rows = _frame(
        [
            {"ownername": name, "transactioncode": "S",
             "transactionshares": -1_000.0,
             "transactiondate": pd.Timestamp("2026-06-0%d" % day)}
            for day, name in enumerate(["A", "B", "C"], start=1)
        ]
    )
    assert insiders.clusters(rows, today=TODAY) == []


# ---------------------------------------------------------------------------
# The roster
# ---------------------------------------------------------------------------


def test_the_latest_balance_per_owner_is_reported(insider_frame):
    result = insiders.roster(insider_frame, today=TODAY)
    assert len(result.rows) == 9
    top = result.rows.iloc[0]
    assert top["owner"] == "BIGHOLDER LP"
    assert top["shares"] == 80_000_000


def test_two_filings_on_one_day_keep_the_later_row_number(insider_frame):
    """Jane sold (rownum 1) and had shares withheld (rownum 3) on 2 July. Only
    the last line in row order carries her end-of-day balance."""
    result = insiders.roster(insider_frame, today=TODAY)
    jane = result.rows[result.rows["owner"] == "JANE OFFICER"].iloc[0]
    assert jane["shares"] == 72_000
    assert jane["last_code"] == "F"


def test_derivative_holdings_are_not_added_to_the_share_count(insider_frame):
    """QUIET HOLDER holds 12,000 shares and 500,000 option contracts. Adding
    the two overstates the holding by a factor of forty."""
    result = insiders.roster(insider_frame, today=TODAY)
    quiet = result.rows[result.rows["owner"] == "QUIET HOLDER"].iloc[0]
    assert quiet["shares"] == 12_000


def test_an_owner_with_only_a_holdings_row_still_appears(insider_frame):
    """A Form 3 reports a real position even though it reports no trade."""
    result = insiders.roster(insider_frame, today=TODAY)
    assert "QUIET HOLDER" in set(result.rows["owner"])


def test_a_stale_owner_is_excluded_and_counted(insider_frame):
    """SF2 records no departure, so a 2021 balance is not evidence anyone still
    holds it — or still works there."""
    result = insiders.roster(insider_frame, today=TODAY)
    assert "OLD TIMER" not in set(result.rows["owner"])
    assert result.omitted == 1
    assert "records no departure" in result.reason


def test_a_roster_of_only_derivative_rows_says_so():
    options = _frame([{"securityadcode": "DA", "securitytitle": "Opt"}])
    result = insiders.roster(options, today=TODAY)
    assert result.rows.empty
    assert "derivative holding" in result.reason


def test_the_roster_carries_a_date_per_owner(insider_frame):
    """"As last reported", never "currently held" — there is no date on which
    SF2 asserts a current position."""
    result = insiders.roster(insider_frame, today=TODAY)
    assert "as_of" in result.rows.columns
    assert result.rows["as_of"].notna().all()


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def test_the_categories_partition_every_row(insider_frame):
    """The tab's answer to "why does Apple look empty": the counts have to sum
    to the row count, or the reader cannot tell filtering from a bug."""
    panel = insiders.reconciliation(insider_frame, today=TODAY)
    counted = panel[panel["Category"].isin(insiders.RECONCILIATION_CATEGORIES)]
    assert counted["Rows"].sum() == len(insider_frame)


def test_the_counted_rows_tie_to_the_headline(insider_frame):
    panel = insiders.reconciliation(insider_frame, today=TODAY)
    priced = panel[panel["Category"] == "Open market, priced"].iloc[0]
    assert priced["Rows"] == 8
    assert priced["Dollars"] == pytest.approx(13_350_000)


def test_an_unrecognised_code_is_surfaced_not_dropped(insider_frame):
    """The SEC adds codes; Sharadar passes through whatever the filing carried."""
    panel = insiders.reconciliation(insider_frame, today=TODAY)
    unknown = panel[panel["Category"] == "Unclassified code"]
    assert len(unknown) == 1
    assert unknown["Rows"].iloc[0] == 1


def test_a_row_whose_reported_value_disagrees_is_counted(insider_frame):
    panel = insiders.reconciliation(insider_frame, today=TODAY)
    check = panel[panel["Category"].str.startswith("— of which")].iloc[0]
    assert check["Rows"] == 1


def test_rows_older_than_the_window_are_shown_as_such(insider_frame):
    panel = insiders.reconciliation(insider_frame, today=TODAY)
    stale = panel[panel["Category"] == "Outside the window"].iloc[0]
    assert stale["Rows"] == 1


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


def test_the_tolerances_are_not_the_growth_ones():
    """Both carry a 5 and it means something different: filed quarters there,
    calendar years here. Sharing it would move one when the other was tuned."""
    from app.logic.research import growth

    assert insiders.InsiderTolerances is not growth.GrowthTolerances
    assert not hasattr(insiders.InsiderTolerances(), "min_observations")


def test_every_threshold_is_injectable(insider_frame):
    strict = insiders.InsiderTolerances(cluster_min_buyers=4)
    assert insiders.clusters(insider_frame, tolerances=strict, today=TODAY) == []


def test_the_module_makes_no_request_and_reads_no_price_series():
    """Insider prices come off the filings themselves, not off SEP. The client
    is imported for its vendor vocabulary only, as ``growth`` imports
    ``SF1_CAPTIONS`` — never for a fetch."""
    import ast
    import inspect

    source = inspect.getsource(insiders)
    assert "import streamlit" not in source

    # Against the parsed tree, not the text: the module docstring names the
    # fetch that feeds it, and a substring check would trip over its own prose.
    called = {
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
    }
    assert not {name for name in called if name.startswith("fetch")}
    assert "ohlcv" not in called


def test_the_plotted_trades_are_priced_open_market_ones_in_the_lookback(insider_frame):
    """What the events chart draws: P and S only, each with a signed flow, a
    role and a price on the chart's basis — the same rows the view plotted."""
    trades = insiders.open_market_trades(insider_frame, pd.DataFrame(), today=TODAY)
    assert set(trades["transactioncode"]) <= {"P", "S"}
    assert trades["flow"].notna().all() and trades["when"].notna().all()
    assert {"role", "plotted"} <= set(trades.columns)
    cutoff = pd.Timestamp(TODAY) - pd.DateOffset(years=insiders.InsiderTolerances().lookback_years)
    assert (trades["when"] >= cutoff).all()
    # No split on file: the plotted price is the filed one.
    assert (trades["plotted"] == pd.to_numeric(trades["transactionpricepershare"])).all()


def test_the_plotted_price_divides_by_the_splits_since(insider_frame):
    split = _splits([("2026-06-01", 2.0)])
    trades = insiders.open_market_trades(insider_frame, split, today=TODAY)
    before = trades["when"] < pd.Timestamp("2026-06-01")
    filed = pd.to_numeric(trades["transactionpricepershare"])
    assert before.any()
    assert (trades.loc[before, "plotted"] == filed[before] / 2.0).all()
    assert (trades.loc[~before, "plotted"] == filed[~before]).all()


def test_no_filings_means_no_trades():
    assert insiders.open_market_trades(pd.DataFrame(), pd.DataFrame(), today=TODAY).empty
