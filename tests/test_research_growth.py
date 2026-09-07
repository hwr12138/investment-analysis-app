"""Year-on-year growth: the base gate, the sign conventions, and the medians.

Covers ``app/logic/research/growth.py``. The fixture universe in ``conftest``
carries two periods for every ticker, chosen so each median below is checkable
by hand and every degenerate base has a case — see the comment above
``UNIVERSE_SF1_ROWS``.

The rule under test throughout is the one that inverts from the valuation tab: a
**negative reading is kept** and a **negative base is not**.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.data import sharadar_client as sc
from app.logic.research import growth
from app.provenance import Origin

CONSUMER = "Consumer Electronics"
BY_LABEL = {m.label: m for m in growth.METRICS}


def _comparisons(peer_universe, *, industry=CONSUMER, statements=None, **kwargs):
    return {
        c.metric.label: c
        for c in growth.build_comparison(
            statements_ttm={} if statements is None else statements,
            peer_universe=peer_universe,
            industry=industry,
            today=date(2026, 8, 5),
            **kwargs,
        )
    }


def _ttm(indicator: str, rows: list[tuple[str, str, float]]) -> dict:
    """A TTM statement frame: ``(calendar date, publish date, value)`` per period."""
    caption = sc.SF1_CAPTIONS[indicator]
    return {
        "PL": pd.DataFrame(
            {
                "Calendar Date": pd.to_datetime([r[0] for r in rows]),
                "Publish Date": pd.to_datetime([r[1] for r in rows]),
                caption: [r[2] for r in rows],
            }
        )
    }


def _quarters(indicator: str, values: list[float], start="2021-06-30") -> dict:
    """A quarterly TTM history long enough to clear ``min_observations``."""
    periods = pd.date_range(start=start, periods=len(values), freq="QE")
    return _ttm(
        indicator,
        [
            (p.date().isoformat(), (p + pd.Timedelta(days=30)).date().isoformat(), v)
            for p, v in zip(periods, values)
        ],
    )


# ---------------------------------------------------------------------------
# The base gate — the reason this module exists
# ---------------------------------------------------------------------------


def test_a_narrowing_loss_is_not_reported_as_a_collapse(peer_universe):
    """Intel's diluted EPS went −4.75 to −2.12: the loss halved.

    The naive ratio prints −55%, which reads as the business falling apart. This
    is the single case the whole base gate exists for.
    """
    statements = _ttm(
        "epsdil",
        [("2025-03-31", "2025-05-01", -4.75), ("2026-03-31", "2026-05-01", -2.12)],
    )
    comparison = _comparisons(peer_universe, statements=statements)[
        "EPS Diluted Growth (YoY)"
    ]
    assert comparison.current is None
    assert "negative a year ago" in comparison.current_reason
    assert "inverts its sign" in comparison.current_reason
    assert comparison.origin is Origin.UNAVAILABLE


def test_a_negative_growth_rate_is_kept(peer_universe):
    """Revenue falling is the answer, not an absence. The inverse of valuation."""
    statements = _ttm(
        "revenue",
        [("2025-03-31", "2025-05-01", 1.0e9), ("2026-03-31", "2026-05-01", 9.2e8)],
    )
    comparison = _comparisons(peer_universe, statements=statements)["Revenue Growth (YoY)"]
    assert comparison.current == pytest.approx(-0.08)


def test_a_zero_base_says_so_rather_than_dividing(peer_universe):
    """Banks report no capex; a percentage change from nothing is undefined."""
    statements = _ttm(
        "capex",
        [("2025-03-31", "2025-05-01", 0.0), ("2026-03-31", "2026-05-01", -5.0e7)],
    )
    comparison = _comparisons(peer_universe, statements=statements)["CAPEX Growth (YoY)"]
    assert comparison.current is None
    assert "zero a year ago" in comparison.current_reason


def test_a_non_payer_is_told_apart_from_a_cut(peer_universe):
    """No dividend a year ago is not the same fact as a dividend eliminated."""
    none_ever = _ttm(
        "dps", [("2025-03-31", "2025-05-01", 0.0), ("2026-03-31", "2026-05-01", 0.0)]
    )
    stopped = _ttm(
        "dps", [("2025-03-31", "2025-05-01", 0.125), ("2026-03-31", "2026-05-01", 0.0)]
    )
    label = "Dividend Growth (1Y, TTM)"

    never = _comparisons(peer_universe, statements=none_ever)[label]
    assert never.current is None
    assert "No dividend a year ago" in never.current_reason

    # A payer that stops is a real −100%, not a gap.
    cut = _comparisons(peer_universe, statements=stopped)[label]
    assert cut.current == pytest.approx(-1.0)


def test_a_missing_year_ago_period_says_so(peer_universe):
    """A company that listed part-way through has no year-on-year, and says it."""
    statements = _ttm("revenue", [("2026-03-31", "2026-05-01", 1.0e9)])
    comparison = _comparisons(peer_universe, statements=statements)["Revenue Growth (YoY)"]
    assert comparison.current is None
    assert "twelve months before" in comparison.current_reason


# ---------------------------------------------------------------------------
# Sign conventions
# ---------------------------------------------------------------------------


def test_capex_growth_is_taken_on_the_magnitude(peer_universe):
    """Sharadar signs capex negative. Spending less must read as a fall.

    Apple went −12.38bn to −10.04bn, which is 18.9% *less* capital spending.
    """
    statements = _ttm(
        "capex",
        [("2025-03-31", "2025-05-01", -12.38e9), ("2026-03-31", "2026-05-01", -10.04e9)],
    )
    comparison = _comparisons(peer_universe, statements=statements)["CAPEX Growth (YoY)"]
    assert comparison.current == pytest.approx(-0.189, abs=1e-3)


def test_rising_capex_reads_as_growth(peer_universe):
    statements = _ttm(
        "capex",
        [("2025-03-31", "2025-05-01", -1.0e8), ("2026-03-31", "2026-05-01", -1.2e8)],
    )
    comparison = _comparisons(peer_universe, statements=statements)["CAPEX Growth (YoY)"]
    assert comparison.current == pytest.approx(0.20)


def test_only_capex_uses_the_magnitude():
    """Every other metric would have its sign silently flipped by it."""
    assert [m.label for m in growth.METRICS if m.magnitude] == ["CAPEX Growth (YoY)"]


# ---------------------------------------------------------------------------
# Pairing the periods
# ---------------------------------------------------------------------------


def test_the_pairing_is_on_the_period_not_the_filing_date():
    """"A year earlier" means the period, and filings do not land on schedule."""
    statements = _ttm(
        "revenue",
        [
            # Filed late, but it is still the 2025-03-31 period.
            ("2025-03-31", "2025-08-20", 1.0e9),
            ("2026-03-31", "2026-05-01", 1.5e9),
        ],
    )
    history = growth.growth_history(statements, BY_LABEL["Revenue Growth (YoY)"])
    assert history["growth"].iloc[-1] == pytest.approx(0.5)


def test_a_52_week_fiscal_calendar_still_pairs():
    """Apple's quarters end 2025-06-28 and 2026-06-27 — a day apart, not a year.

    Pairing on the *report* date matches nothing for any company on a 52/53 week
    calendar, and every growth rate reads unavailable. Sharadar's
    ``calendardate`` maps both onto 30 June, which is why that is the key. This
    is a regression test: the report-date version shipped and was caught only
    against live data.
    """
    caption = sc.SF1_CAPTIONS["revenue"]
    statements = {
        "PL": pd.DataFrame(
            {
                # The drifting fiscal period ends, as Sharadar reports them...
                "Report Date": pd.to_datetime(["2025-06-28", "2026-06-27"]),
                # ...and the normalised calendar quarters they map onto.
                "Calendar Date": pd.to_datetime(["2025-06-30", "2026-06-30"]),
                "Publish Date": pd.to_datetime(["2025-08-01", "2026-07-31"]),
                caption: [4.086e11, 4.668e11],
            }
        )
    }
    history = growth.growth_history(statements, BY_LABEL["Revenue Growth (YoY)"])
    assert history["growth"].iloc[-1] == pytest.approx(0.1424, abs=1e-4)


def test_the_series_is_indexed_on_when_the_figure_became_public():
    """The window and any future chart must carry no lookahead."""
    statements = _ttm(
        "revenue",
        [("2025-03-31", "2025-05-01", 1.0e9), ("2026-03-31", "2026-05-01", 1.1e9)],
    )
    series = growth.growth_series(statements, BY_LABEL["Revenue Growth (YoY)"])
    assert series.index[-1] == pd.Timestamp("2026-05-01")


def test_a_skipped_period_gets_no_base_rather_than_a_stretched_one():
    """Fifteen months of growth reported as twelve would be quietly wrong."""
    statements = _ttm(
        "revenue",
        [("2024-12-31", "2025-02-01", 1.0e9), ("2026-03-31", "2026-05-01", 1.5e9)],
    )
    history = growth.growth_history(statements, BY_LABEL["Revenue Growth (YoY)"])
    assert pd.isna(history["growth"].iloc[-1])


# ---------------------------------------------------------------------------
# The company's own history
# ---------------------------------------------------------------------------


def test_the_five_year_median_counts_quarters_not_sessions(peer_universe):
    """Growth steps once per filing; the unit has to travel with the count."""
    values = [100.0 * (1.1 ** i) for i in range(20)]
    comparison = _comparisons(
        peer_universe, statements=_quarters("revenue", values)
    )["Revenue Growth (YoY)"]
    # Each quarter compounds 10%, so every year-on-year is 1.1**4 - 1.
    assert comparison.history_median == pytest.approx(1.1**4 - 1)
    assert comparison.history_unit == "quarters"
    assert comparison.history_observations == 16  # 20 periods, first 4 have no base


def test_a_short_history_reports_how_short(peer_universe):
    statements = _quarters("revenue", [100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
    comparison = _comparisons(peer_universe, statements=statements)["Revenue Growth (YoY)"]
    assert comparison.history_median is None
    assert "quarter(s)" in comparison.history_reason


def test_the_observation_floor_is_a_tolerance_not_a_literal(peer_universe):
    statements = _quarters("revenue", [100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
    loose = growth.GrowthTolerances(min_observations=2)
    comparison = _comparisons(
        peer_universe, statements=statements, tolerances=loose
    )["Revenue Growth (YoY)"]
    assert comparison.history_median is not None


# ---------------------------------------------------------------------------
# Industry medians
# ---------------------------------------------------------------------------


def test_revenue_growth_median_includes_the_loss_maker(peer_universe):
    """Growth over [−50, −50, +10, +25, +25, +60, +100] is +25%.

    LOSS has a positive revenue base even though it loses money, so it counts
    here — the exclusion is per metric, not per company.
    """
    comparison = _comparisons(peer_universe)["Revenue Growth (YoY)"]
    assert comparison.industry_median == pytest.approx(0.25)
    assert comparison.peer_count == 7


def test_a_negative_base_drops_only_that_metric(peer_universe):
    """LOSS counts toward revenue growth but not toward EPS growth."""
    comparisons = _comparisons(peer_universe)
    assert comparisons["Revenue Growth (YoY)"].peer_count == 7
    assert comparisons["EPS Diluted Growth (YoY)"].peer_count == 6
    loss = peer_universe[peer_universe["ticker"] == "LOSS"].iloc[0]
    assert loss["revenue_prior"] > 0 and loss["epsdil_prior"] < 0


def test_a_zero_base_drops_out_of_the_peer_median(peer_universe):
    """PEER3 had no capex and paid nothing a year ago."""
    comparisons = _comparisons(peer_universe)
    assert comparisons["CAPEX Growth (YoY)"].peer_count == 6
    assert comparisons["Dividend Growth (1Y, TTM)"].peer_count == 5


def test_the_universe_pairs_each_ticker_with_its_own_year_ago_period(peer_universe):
    """TEST's latest revenue is 1.1e9 against a base of 1.0e9."""
    test = peer_universe[peer_universe["ticker"] == "TEST"].iloc[0]
    assert test["revenue"] == pytest.approx(1.1e9)
    assert test["revenue_prior"] == pytest.approx(1.0e9)


def test_a_thin_industry_reports_the_count_rather_than_a_median(peer_universe):
    comparison = _comparisons(peer_universe, industry="Aluminum")["Revenue Growth (YoY)"]
    assert comparison.industry_median is None
    assert comparison.peer_count == 3
    assert "Aluminum" in comparison.industry_reason


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def test_the_gap_to_the_industry_is_in_percentage_points(peer_universe):
    """Not a ratio: dividing one rate by another means little and breaks at zero."""
    statements = _ttm(
        "revenue",
        [("2025-03-31", "2025-05-01", 1.0e9), ("2026-03-31", "2026-05-01", 1.1e9)],
    )
    comparison = _comparisons(peer_universe, statements=statements)["Revenue Growth (YoY)"]
    # 10% against an industry median of 25% is fifteen points behind.
    assert comparison.industry_gap == pytest.approx(0.10 - 0.25)


def test_a_negative_industry_median_still_yields_a_gap(peer_universe):
    """The case a ratio could not express at all."""
    from app.logic.research.comparison import Comparison

    record = Comparison(
        metric=growth.METRICS[0], current=0.05, industry_median=-0.10,
        industry_gap=0.05 - (-0.10),
    )
    assert record.industry_gap == pytest.approx(0.15)


def test_growth_is_always_derived_never_reported(peer_universe):
    """Every figure is one reported number divided by another."""
    statements = _ttm(
        "revenue",
        [("2025-03-31", "2025-05-01", 1.0e9), ("2026-03-31", "2026-05-01", 1.1e9)],
    )
    comparison = _comparisons(peer_universe, statements=statements)["Revenue Growth (YoY)"]
    assert comparison.origin is Origin.DERIVED


def test_every_metric_has_a_definition_and_a_real_indicator():
    for metric in growth.METRICS:
        assert metric.label.strip()
        assert metric.definition.strip(), metric.label
        assert metric.indicator in sc.SF1_CAPTIONS, metric.label


def test_the_layout_and_the_comparisons_agree_exactly(peer_universe):
    produced = [c.metric.label for c in _comparisons(peer_universe).values()]
    assert produced == [m.label for m in growth.METRICS]
    assert len(set(produced)) == len(produced)


def test_the_requested_metrics_are_all_present():
    """The ten agreed rows, with EPS collapsed to one GAAP diluted line."""
    assert [m.label for m in growth.METRICS] == [
        "Revenue Growth (YoY)",
        "EBITDA Growth (YoY)",
        "EBIT Growth (YoY)",
        "EPS Diluted Growth (YoY)",
        "Levered FCF Growth (YoY)",
        "Operating Cash Flow Growth (YoY)",
        "ROE Growth (YoY)",
        "Working Capital Growth (YoY)",
        "CAPEX Growth (YoY)",
        "Dividend Growth (1Y, TTM)",
    ]


def test_the_eps_row_says_it_is_gaap():
    """Sharadar has no adjusted EPS, so the row has to disclose which basis it is."""
    eps = BY_LABEL["EPS Diluted Growth (YoY)"]
    assert "GAAP" in eps.definition
    assert eps.indicator == "epsdil"


def test_growth_reads_no_price_series():
    """A growth rate does not move with the share price, so nothing takes `daily`."""
    import inspect

    assert "daily" not in inspect.signature(growth.build_comparison).parameters
