"""Relative valuation: the universe fetches, the medians, and the exclusions.

Covers ``app/logic/research/valuation.py``. Not to be confused with
``test_valuation.py``, which covers the DCF engine's ``app/logic/dcf/valuation.py``
— two different modules that happen to share a name.


The fixture universe in ``conftest`` is eleven tickers in three industries, sized
so every median here is checkable by hand — see the comment above
``UNIVERSE_DAILY_ROWS`` for the shape. Any median moving is either a real change
or a broken exclusion, and the tests say which.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.data import sharadar_client as sc
from app.data.sharadar import prices as sh_prices
from app.data.sharadar import transport as sh_transport
from app.data.sharadar import universe as sh_universe
from app.logic.research import comparison as comparison_lib
from app.logic.research import valuation
from app.provenance import Origin
from tests.conftest import (
    UNIVERSE_SF1_ROWS,
    directory_payload,
    stub_universe,
    universe_daily_payload,
    universe_sf1_payload,
)

CONSUMER = "Consumer Electronics"


def _comparisons(peer_universe, *, industry=CONSUMER, daily=None, statements=None, **kwargs):
    return {
        c.metric.label: c
        for c in valuation.build_comparison(
            daily=pd.DataFrame() if daily is None else daily,
            statements_ttm={} if statements is None else statements,
            peer_universe=peer_universe,
            industry=industry,
            today=date(2026, 8, 5),
            **kwargs,
        )
    }


# ---------------------------------------------------------------------------
# The universe fetches
# ---------------------------------------------------------------------------


def test_universe_daily_rescales_market_cap_and_ev(monkeypatch):
    """The second DAILY call site has to convert millions too.

    ``fetch_daily_metrics`` documents this as a factor-of-a-million bug waiting
    for a second call site; this is that call site.
    """
    stub_universe(monkeypatch)
    frame = sh_universe._fetch_universe_daily("2026-08-05").set_index("ticker")
    assert frame.loc["TEST", "marketcap"] == pytest.approx(2.0e9)
    assert frame.loc["TEST", "ev"] == pytest.approx(2.2e9)
    # Ratios are dimensionless and must not be touched.
    assert frame.loc["TEST", "pe"] == pytest.approx(20.0)


def test_both_daily_fetchers_share_one_conversion():
    """Not a behavioural test — a structural one.

    The per-ticker and universe fetchers must keep going through the same
    normaliser. Two copies of the rescale is how one of them gets fixed alone.
    """
    import inspect

    for fetcher in (sh_prices._fetch_daily_metrics, sh_universe._fetch_universe_daily):
        assert "_normalise_daily" in inspect.getsource(fetcher)


def test_a_restated_period_keeps_the_later_filing(monkeypatch):
    """TEST filed twice for 2026-03-31; the later ``datekey`` supersedes.

    This is a correction of one period, not two periods of history, so exactly
    one row survives — and it has to be the corrected one.
    """
    stub_universe(monkeypatch)
    frame = sh_universe._fetch_universe_fundamentals("2025-01-15")
    latest = frame[
        (frame["ticker"] == "TEST") & (frame["calendardate"] == pd.Timestamp("2026-03-31"))
    ]
    assert len(latest) == 1
    assert latest.iloc[0]["revenue"] == pytest.approx(1.1e9)
    assert latest.iloc[0]["datekey"] == pd.Timestamp("2026-05-01")


def test_the_window_keeps_every_period_not_just_the_latest(monkeypatch):
    """Growth needs both ends of a year-on-year, so the window is not collapsed."""
    stub_universe(monkeypatch)
    frame = sh_universe._fetch_universe_fundamentals("2025-01-15")
    periods = frame[frame["ticker"] == "TEST"]["calendardate"]
    assert sorted(periods.dt.date.astype(str)) == ["2025-03-31", "2026-03-31"]
    # Every ticker in the fixture files twice; only the restatement collapses.
    assert len(frame) == len(UNIVERSE_SF1_ROWS) - 1


def test_universe_requests_project_their_columns(monkeypatch):
    """The directory is 17,800 rows; unprojected it is ~10MB instead of ~2MB."""
    seen: list[dict] = []

    def capture(table, query, **kwargs):
        seen.append(query)
        return directory_payload() if table == "TICKERS" else universe_sf1_payload()

    monkeypatch.setattr(sh_transport, "_get", capture)
    sh_universe._fetch_ticker_directory()
    sh_universe._fetch_universe_fundamentals("2026-02-15")
    assert all("qopts.columns" in query for query in seen)


def test_universe_daily_asks_for_a_date_and_no_ticker(monkeypatch):
    """One request for the whole market is the reason peers are affordable."""
    seen: list[dict] = []

    def capture(table, query, **kwargs):
        seen.append(query)
        return universe_daily_payload()

    monkeypatch.setattr(sh_transport, "_get", capture)
    sh_universe._fetch_universe_daily("2026-08-05")
    assert seen[0]["date"] == "2026-08-05"
    assert "ticker" not in seen[0]


# ---------------------------------------------------------------------------
# Building the peer universe
# ---------------------------------------------------------------------------


def test_delisted_companies_are_dropped(peer_universe):
    """GONE still has a DAILY row and must not reach a median."""
    assert "GONE" not in set(peer_universe["ticker"])
    assert "TEST" in set(peer_universe["ticker"])


def test_the_universe_carries_the_industry_label(peer_universe):
    assert comparison_lib.industry_of(peer_universe, "TEST") == CONSUMER
    assert comparison_lib.industry_of(peer_universe, "ALU1") == "Aluminum"
    assert comparison_lib.industry_of(peer_universe, "NOSUCH") == ""


def test_an_empty_universe_is_empty_rather_than_wrong():
    """A failed fetch leaves the medians unavailable; it never substitutes."""
    assert comparison_lib.build_peer_universe(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ).empty


# ---------------------------------------------------------------------------
# Industry medians
# ---------------------------------------------------------------------------


def test_vendor_multiple_median_is_the_middle_of_the_usable_peers(peer_universe):
    """P/E over [10, 14, 18, 20, 26, 30] — LOSS and GONE excluded — is 19."""
    comparison = _comparisons(peer_universe)["P/E (TTM)"]
    assert comparison.industry_median == pytest.approx(19.0)
    assert comparison.peer_count == 6


def test_negative_multiples_never_reach_a_median(peer_universe):
    """A negative P/E is a loss, not a cheap stock, and would drag the median."""
    comparison = _comparisons(peer_universe)["P/E (TTM)"]
    peers = peer_universe[peer_universe["industry"] == CONSUMER]
    assert (peers["pe"] < 0).any(), "fixture must contain a loss-maker"
    assert comparison.peer_count == len(peers) - 1


def test_derived_multiples_divide_daily_by_the_sf1_term(peer_universe):
    """EV/Sales over [1.0, 1.6, 1.6, 2.0, 2.0, 2.9, 3.4] is 2.0."""
    comparison = _comparisons(peer_universe)["EV/Sales (TTM)"]
    assert comparison.industry_median == pytest.approx(2.0)


def test_the_exclusion_is_per_metric_not_per_company(peer_universe):
    """A loss-maker has no P/E but still has an EV/Sales, and counts for it.

    Dropping the company outright would quietly shrink every peer group to the
    profitable half of an industry, which is a different comparison than the one
    the tab claims to be making.
    """
    comparisons = _comparisons(peer_universe)
    assert comparisons["P/E (TTM)"].peer_count == 6
    assert comparisons["EV/Sales (TTM)"].peer_count == 7
    # ...and the loss-maker is the difference: negative earnings, positive sales.
    loss = peer_universe[peer_universe["ticker"] == "LOSS"].iloc[0]
    assert loss["pe"] < 0 and loss["revenue"] > 0


def test_price_to_cash_flow_uses_operating_cash_flow(peer_universe):
    """Market cap over ncfo: [5, 7.5, 9, 10, 13, 15] is 9.5."""
    comparison = _comparisons(peer_universe)["Price/Cash Flow (TTM)"]
    assert comparison.industry_median == pytest.approx(9.5)


def test_dividend_yield_negates_the_cash_outflow(peer_universe):
    """``ncfdiv`` is negative in the feed; a yield is not."""
    comparison = _comparisons(peer_universe)["Dividend Yield (TTM)"]
    assert comparison.industry_median == pytest.approx(0.02)
    assert comparison.industry_median > 0


def test_non_payers_are_excluded_from_the_yield_median(peer_universe):
    """PEER3 pays nothing. Counting its zero would halve every industry's yield."""
    comparison = _comparisons(peer_universe)["Dividend Yield (TTM)"]
    peers = peer_universe[peer_universe["industry"] == CONSUMER]
    assert (peers["ncfdiv"] == 0).any(), "fixture must contain a non-payer"
    assert comparison.peer_count == 5


def test_a_thin_industry_reports_the_count_rather_than_a_median(peer_universe):
    """Three aluminium companies is not a median of anything."""
    comparison = _comparisons(peer_universe, industry="Aluminum")["P/E (TTM)"]
    assert comparison.industry_median is None
    assert comparison.peer_count == 3
    assert "3" in comparison.industry_reason
    assert "Aluminum" in comparison.industry_reason


def test_the_peer_floor_is_a_tolerance_not_a_literal(peer_universe):
    """Loosening the floor is a knob, and the aluminium median then appears."""
    loose = valuation.ValuationTolerances(min_peers=3)
    comparison = _comparisons(peer_universe, industry="Aluminum", tolerances=loose)
    assert comparison["P/E (TTM)"].industry_median == pytest.approx(13.0)


def test_an_unclassified_company_says_so(peer_universe):
    comparison = _comparisons(peer_universe, industry="")["P/E (TTM)"]
    assert comparison.industry_median is None
    assert "industry classification" in comparison.industry_reason


# ---------------------------------------------------------------------------
# The company's own history
# ---------------------------------------------------------------------------


def _daily_history(**columns) -> pd.DataFrame:
    """400 sessions ending on the fixture date."""
    dates = pd.bdate_range(end="2026-08-05", periods=400)
    return pd.DataFrame({"date": dates, **columns})


def _ttm_frames(caption: str, rows: list[tuple[str, float]]) -> dict:
    """A minimal TTM statement frame carrying one indicator's filing history."""
    return {
        "PL": pd.DataFrame(
            {
                "Publish Date": pd.to_datetime([r[0] for r in rows]),
                caption: [r[1] for r in rows],
            }
        )
    }


def test_history_median_is_taken_over_the_five_year_window(peer_universe):
    daily = _daily_history(pe=[float(i % 40) + 1.0 for i in range(400)])
    comparison = _comparisons(peer_universe, daily=daily)["P/E (TTM)"]
    assert comparison.history_median == pytest.approx(20.5)
    assert comparison.history_observations == 400


def test_a_short_history_reports_how_short(peer_universe):
    daily = pd.DataFrame(
        {"date": pd.bdate_range(end="2026-08-05", periods=10), "pe": [20.0] * 10}
    )
    comparison = _comparisons(peer_universe, daily=daily)["P/E (TTM)"]
    assert comparison.history_median is None
    assert "10 usable session" in comparison.history_reason


def test_the_percentile_says_where_today_sits_in_its_own_range(peer_universe):
    """Today at 40 against a history running 1..40: dearer than nearly all of it."""
    values = [float(i % 40) + 1.0 for i in range(399)] + [40.0]
    comparison = _comparisons(peer_universe, daily=_daily_history(pe=values))["P/E (TTM)"]
    assert comparison.current == pytest.approx(40.0)
    assert comparison.percentile == pytest.approx(0.9750)


def test_a_negative_current_reading_is_told_apart_from_a_missing_one(peer_universe):
    daily = _daily_history(pe=[15.0] * 399 + [-3.0])
    comparison = _comparisons(peer_universe, daily=daily)["P/E (TTM)"]
    assert comparison.current is None
    assert "denominator is negative" in comparison.current_reason
    assert comparison.origin is Origin.UNAVAILABLE


def test_a_suspended_dividend_is_not_called_a_negative_denominator():
    """Intel pays nothing, and ``-1 x 0.0`` is ``-0.0``.

    Reporting that as "the denominator is negative" would be plainly untrue —
    a company that stopped paying is a different fact from a loss-making one.
    """
    caption = sc.SF1_CAPTIONS["ncfdiv"]
    daily = pd.DataFrame(
        {"date": pd.to_datetime(["2026-06-01"]), "marketcap": [2.0e9]}
    )
    statements = {
        "CF": pd.DataFrame(
            {"Publish Date": pd.to_datetime(["2026-05-01"]), caption: [0.0]}
        )
    }
    comparison = _comparisons(
        pd.DataFrame(), industry="", daily=daily, statements=statements
    )["Dividend Yield (TTM)"]
    assert comparison.current is None
    assert comparison.current_reason == "No dividend paid in the trailing twelve months."
    assert "negative" not in comparison.current_reason


def test_a_negative_yield_is_reported_as_a_percentage(peer_universe):
    """The reason quotes the figure, so it has to be in the metric's own units."""
    daily = _daily_history(pe=[15.0] * 399 + [-0.03])
    comparison = _comparisons(peer_universe, daily=daily)["P/E (TTM)"]
    assert "-0.03" in comparison.current_reason


def test_derived_history_steps_at_the_filing_date_not_the_period_end():
    """A price on 1 April may not divide by revenue first published on 1 May.

    This is the whole reason the join keys on ``Publish Date``. Keying it on the
    reported period would value the company with figures nobody had yet.
    """
    caption = sc.SF1_CAPTIONS["revenue"]
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-01", "2026-05-02"]),
            "ev": [2.0e9, 2.0e9],
        }
    )
    statements = _ttm_frames(
        caption, [("2026-01-15", 5.0e8), ("2026-05-01", 1.0e9)]
    )
    metric = {m.label: m for m in valuation.METRICS}["EV/Sales (TTM)"]
    series = valuation.company_series(daily, statements, metric)

    # 1 April still uses the January filing; 2 May uses the May one.
    assert series.iloc[0] == pytest.approx(4.0)
    assert series.iloc[1] == pytest.approx(2.0)


def test_a_price_before_the_first_filing_has_no_derived_reading():
    """No lookahead: there is nothing to divide by yet, so nothing is shown."""
    caption = sc.SF1_CAPTIONS["revenue"]
    daily = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "ev": [2.0e9]})
    metric = {m.label: m for m in valuation.METRICS}["EV/Sales (TTM)"]
    series = valuation.company_series(
        daily, _ttm_frames(caption, [("2026-05-01", 1.0e9)]), metric
    )
    assert series.isna().all()


def test_the_dividend_yield_series_is_a_positive_fraction():
    caption = sc.SF1_CAPTIONS["ncfdiv"]
    daily = pd.DataFrame(
        {"date": pd.to_datetime(["2026-06-01"]), "marketcap": [2.0e9]}
    )
    statements = {
        "CF": pd.DataFrame(
            {
                "Publish Date": pd.to_datetime(["2026-05-01"]),
                caption: [-4.0e7],
            }
        )
    }
    metric = {m.label: m for m in valuation.METRICS}["Dividend Yield (TTM)"]
    series = valuation.company_series(daily, statements, metric)
    assert series.iloc[0] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# The layout declaration
# ---------------------------------------------------------------------------


def test_every_metric_has_a_source_and_a_definition():
    for metric in valuation.METRICS:
        assert metric.label.strip()
        assert metric.definition.strip(), metric.label
        assert metric.vendor ^ (metric.numerator is not None), metric.label
        if not metric.vendor:
            assert metric.denominator is not None, metric.label


def test_the_layout_and_the_comparisons_agree_exactly(peer_universe):
    """A metric computed but not declared, or declared but not computed, fails."""
    produced = [c.metric.label for c in _comparisons(peer_universe).values()]
    assert produced == [m.label for m in valuation.METRICS]
    assert len(set(produced)) == len(produced)


def test_only_the_dividend_yield_reads_the_other_way():
    """Everywhere else a higher number is the dearer one."""
    inverted = [m.label for m in valuation.METRICS if m.higher_is_cheaper]
    assert inverted == ["Dividend Yield (TTM)"]
    assert [m.label for m in valuation.METRICS if m.percent] == inverted


def test_the_premium_is_relative_to_the_industry(peer_universe):
    """TEST's P/E of 20 against an industry median of 19 is ~5% dearer."""
    daily = _daily_history(pe=[20.0] * 400)
    comparison = _comparisons(peer_universe, daily=daily)["P/E (TTM)"]
    assert comparison.industry_gap == pytest.approx(20.0 / 19.0 - 1.0)


def test_no_premium_without_both_sides(peer_universe):
    comparison = _comparisons(peer_universe, industry="")["P/E (TTM)"]
    assert comparison.industry_gap is None


# ---------------------------------------------------------------------------
# The verdict, as both front ends read it
# ---------------------------------------------------------------------------


def _gap(label: str, gap: float | None) -> comparison_lib.Verdict:
    metric = {m.label: m for m in valuation.METRICS}[label]
    return valuation.verdict(comparison_lib.Comparison(metric=metric, industry_gap=gap))


def test_a_premium_is_rich_and_a_discount_cheap():
    assert _gap("P/E (TTM)", 0.5) == comparison_lib.Verdict(0.5, "richer than industry", False)
    assert _gap("P/E (TTM)", -0.3) == comparison_lib.Verdict(-0.3, "cheaper than industry", True)


def test_the_dividend_yield_reads_the_other_way():
    assert _gap("Dividend Yield (TTM)", 0.5).favourable is True


def test_a_gap_inside_the_threshold_is_in_line_and_neutral():
    reading = _gap("P/E (TTM)", 0.05)
    assert reading.word == "in line" and reading.favourable is None
    wide = valuation.ValuationTolerances(material_premium=0.01)
    metric = valuation.METRICS[0]
    assert valuation.verdict(
        comparison_lib.Comparison(metric=metric, industry_gap=0.05), wide
    ).favourable is False


def test_no_median_means_no_verdict():
    assert _gap("P/E (TTM)", None) == comparison_lib.Verdict(None)
