"""Export contents, acknowledgement stamping, and provenance round-tripping."""

from __future__ import annotations

import io
from datetime import date

import openpyxl
import pandas as pd
import pytest

from app.logic.dcf import export as ex
from app.logic.dcf import validation as val
from app.provenance import Origin, Sourced

from .test_golden import build_model  # shared model builder


@pytest.fixture
def built(statements_payload, tickers_response, monkeypatch):
    return build_model(statements_payload, tickers_response, monkeypatch)


@pytest.fixture
def issues():
    return [
        val.ReconciliationIssue(
            key="balance_sheet:2025",
            severity=val.Severity.CRITICAL,
            check="Balance sheet balances",
            concept="Total Assets = Total Liabilities + Total Equity",
            company="TEST",
            message="The balance sheet does not balance.",
            period="FY2025",
            values={"Total assets": "1,060", "Liabilities + equity": "1,999"},
            tolerance="0.100%",
        ),
        val.ImplausibleValueIssue(
            key="tv_share",
            severity=val.Severity.WARNING,
            check="Terminal value share of enterprise value",
            concept="Terminal value",
            company="TEST",
            message="Mostly assumption.",
            values={"Terminal value share": "88.0%"},
            tolerance="≤ 75%",
        ),
    ]


def test_acknowledged_checks_are_stamped_into_the_export(issues):
    acknowledged = {"tv_share": "2026-07-31T09:00:00"}
    frame = val.issues_frame(issues, acknowledged)
    acked = frame[frame["Key"] == "tv_share"].iloc[0]
    unacked = frame[frame["Key"] == "balance_sheet:2025"].iloc[0]
    assert acked["Acknowledged"] == "yes"
    assert acked["Acknowledged at"] == "2026-07-31T09:00:00"
    assert unacked["Acknowledged"] == "no"


def test_acknowledged_items_remain_listed_rather_than_removed(issues):
    """Acknowledging hides nothing — the row stays, flagged."""
    frame = val.issues_frame(issues, {"tv_share": "now"})
    assert len(frame) == len(issues)
    assert "Terminal value share of enterprise value" in set(frame["Check"])


def test_issue_rows_carry_both_figures_and_the_tolerance(issues):
    frame = val.issues_frame(issues, {})
    row = frame.iloc[0]
    assert "Total assets = 1,060" in row["Figures"]
    assert "Liabilities + equity = 1,999" in row["Figures"]
    assert row["Tolerance"] == "0.100%"
    assert row["Issue type"] == "ReconciliationIssue"


def test_assumption_records_carry_provenance():
    record = ex.AssumptionRecord(
        "Discount rate",
        "Beta",
        1.0716,
        Sourced.derived(1.0716, "Sharadar SEP vs SPY", "OLS slope, n=60", date(2026, 7, 30)),
    )
    row = record.as_row()
    assert row["Provenance"] == "DERIVED"
    assert row["Source"] == "Sharadar SEP vs SPY"
    assert row["Formula / basis"] == "OLS slope, n=60"
    assert row["As of"] == "2026-07-30"


def test_estimates_are_distinguishable_from_reported_figures():
    reported = ex.AssumptionRecord("x", "SBC", 100.0, Sourced.reported(100.0, "Sharadar CF"))
    estimate = ex.AssumptionRecord(
        "x", "SBC", 138.0, Sourced.estimate(138.0, "Sharadar CF: derived", "Upper bound")
    )
    assert reported.as_row()["Provenance"] == "REPORTED"
    assert estimate.as_row()["Provenance"] == "ESTIMATE"
    assert "Upper bound" in estimate.as_row()["Note"]


def test_unavailable_values_never_carry_a_number():
    sourced = Sourced.unavailable("Sharadar CF: Share Based Compensation", "Not reported")
    assert sourced.value is None
    assert not sourced.ok
    assert sourced.origin is Origin.UNAVAILABLE


def test_user_override_records_what_it_replaced():
    derived = Sourced.derived(0.0934, "5-year average", "over 5 periods")
    overridden = derived.with_user_value(0.12)
    assert overridden.origin is Origin.USER
    assert overridden.value == 0.12
    assert "0.0934" in overridden.detail


def test_projection_frame_is_arithmetically_checkable(built):
    frame = ex.projection_frame(built["result"])
    row = frame.iloc[1]  # a full year, not the stub
    expected = (
        row["EBIT"] * (1 - row["Tax rate"])
        + row["D&A"] - row["CapEx"] - row["ΔNWC"] - row["SBC deducted"]
    )
    assert row["FCFF (full year)"] == pytest.approx(expected)
    assert row["PV of FCFF"] == pytest.approx(
        row["FCFF (this period)"] * row["Discount factor"]
    )


def test_bridge_frame_covers_both_terminal_methods(built):
    frame = ex.bridge_frame(built["result"])
    methods = set(frame["Terminal value method"])
    assert methods == {"Gordon growth", "Exit multiple"}
    for method in methods:
        subset = frame[frame["Terminal value method"] == method]
        assert "Value per share" in set(subset["Line"])


def test_terminal_value_frame_carries_cross_implied_figures(built):
    frame = ex.terminal_value_frame(built["result"])
    gordon = frame[frame["Method"] == "Gordon growth"].iloc[0]
    exit_row = frame[frame["Method"] == "Exit multiple"].iloc[0]
    assert gordon["Implied exit multiple"] is not None
    assert exit_row["Implied perpetuity growth"] is not None
    assert set(frame["Headline"]) == {"yes", "no"}


def test_workbook_has_every_sheet_and_opens(built, issues):
    workbook_bytes = ex.build_workbook(
        dataset=built["dataset"],
        result=built["result"],
        assumptions=[
            ex.AssumptionRecord("Terminal value", "Perpetuity growth", 0.02, Sourced.user(0.02))
        ],
        issues=issues,
        check_results=built["check_results"],
        acknowledged={"tv_share": "2026-07-31T09:00:00"},
        tolerances=val.DEFAULT_TOLERANCES,
        valuation_date=date(2026, 7, 31),
    )
    book = openpyxl.load_workbook(io.BytesIO(workbook_bytes))
    assert {
        "Summary", "Assumptions", "Projection", "Terminal Values", "Equity Bridge",
        "Historical Financials", "Historical Ratios", "Validation Results", "Issues",
        "Schema Map", "Tolerances",
    } <= set(book.sheetnames)


def test_workbook_summary_reports_unacknowledged_criticals(built, issues):
    summary = ex.summary_frame(
        built["dataset"], built["result"], issues, {"tv_share": "now"}, date(2026, 7, 31)
    )
    lookup = dict(zip(summary["Item"], summary["Value"]))
    assert lookup["Unacknowledged CRITICAL issues"] == 1
    assert lookup["Acknowledged checks"] == 1
    assert lookup["Ticker"] == "TEST"


def test_csv_export_is_sectioned(built, issues):
    text = ex.build_csv(
        dataset=built["dataset"],
        result=built["result"],
        assumptions=[],
        issues=issues,
        check_results=built["check_results"],
        acknowledged={},
        valuation_date=date(2026, 7, 31),
    )
    for section in ("### Summary", "### Projection", "### Equity Bridge", "### Issues"):
        assert section in text


def test_schema_map_is_exported_for_auditing(built):
    frame = built["dataset"].schema.as_frame()
    assert "Revenue" in set(frame["Concept"])
    resolved = frame[frame["Concept"] == "Revenue"].iloc[0]
    assert resolved["Resolved column"] == "Revenues"
