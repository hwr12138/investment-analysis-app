"""Workbook export.

Writes the whole model to a multi-sheet ``.xlsx`` so it can be audited outside
the app: every assumption with its provenance, the projection line by line, the
equity bridge, all validation results, and which checks were acknowledged.

The point is reproducibility. Anyone with the workbook should be able to rebuild
value per share from the Assumptions and Projection sheets without the app.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

import pandas as pd

from app.logic.dcf import validation as val
from app.provenance import Sourced

if TYPE_CHECKING:
    # Annotations only. Importing `model` at runtime would load the Sharadar
    # client with it, and writing a workbook needs no network.
    from app.logic.dcf.model import CompanyDataset, ModelResult


@dataclass(frozen=True)
class AssumptionRecord:
    """One assumption as it appears in the export."""

    group: str
    name: str
    value: float | str | None
    sourced: Sourced | None = None
    note: str = ""

    def as_row(self) -> dict:
        origin = self.sourced.origin.label if self.sourced else "USER"
        return {
            "Group": self.group,
            "Assumption": self.name,
            "Value": self.value,
            "Provenance": origin,
            "Source": self.sourced.source if self.sourced else "User input",
            "Formula / basis": (self.sourced.formula or "") if self.sourced else "",
            "As of": (
                self.sourced.as_of.isoformat()
                if self.sourced and self.sourced.as_of
                else ""
            ),
            "Note": self.note or (self.sourced.detail if self.sourced else ""),
        }


def projection_frame(result: ModelResult) -> pd.DataFrame:
    """Year-by-year projection, with every intermediate figure and the formula."""
    rows = []
    for row in result.projection.rows:
        rows.append(
            {
                "Period": row.label,
                "Period fraction": row.period_fraction,
                "Revenue growth": row.revenue_growth,
                "Revenue (full year)": row.revenue,
                "Operating margin": row.operating_margin,
                "EBIT": row.ebit,
                "Tax rate": row.tax_rate,
                "NOPAT": row.nopat,
                "D&A": row.dna,
                "CapEx": row.capex,
                "ΔNWC": row.delta_nwc,
                "SBC deducted": row.sbc_deduction,
                "FCFF (full year)": row.fcff_full_year,
                "FCFF (this period)": row.fcff_period,
                "Discount exponent": row.discount_exponent,
                "Discount factor": row.discount_factor,
                "PV of FCFF": row.present_value,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["formula"] = (
        "FCFF = EBIT × (1 − tax) + D&A − CapEx − ΔNWC − SBC deducted; "
        "PV = FCFF (this period) × discount factor; "
        "discount factor = 1 ÷ (1 + WACC)^exponent"
    )
    return frame


def bridge_frame(result: ModelResult) -> pd.DataFrame:
    """Equity bridge for both terminal-value methods, line by line."""
    rows = []
    for method, bridge in result.bridges.items():
        label = result.terminal_values[method].method
        for line in bridge.lines:
            rows.append(
                {
                    "Terminal value method": label,
                    "Line": line.label,
                    "Amount": line.amount,
                    "Kind": line.kind,
                    "Reported by Sharadar": "yes" if line.reported else "no",
                    "Note": line.note,
                }
            )
        rows.append(
            {
                "Terminal value method": label,
                "Line": f"Shares ({bridge.share_basis})",
                "Amount": bridge.shares,
                "Kind": "divisor",
                "Reported by Sharadar": "yes",
                "Note": "Equity value ÷ shares = value per share",
            }
        )
        rows.append(
            {
                "Terminal value method": label,
                "Line": "Value per share",
                "Amount": bridge.value_per_share,
                "Kind": "result",
                "Reported by Sharadar": "no",
                "Note": "Derived",
            }
        )
    return pd.DataFrame(rows)


def terminal_value_frame(result: ModelResult) -> pd.DataFrame:
    rows = []
    for method, terminal in result.terminal_values.items():
        enterprise_value = result.enterprise_values[method]
        rows.append(
            {
                "Method": terminal.method,
                "Terminal value (undiscounted)": terminal.value,
                "Discount exponent": terminal.discount_exponent,
                "PV of terminal value": terminal.present_value,
                "PV of explicit forecast": result.projection.pv_explicit,
                "Enterprise value": enterprise_value,
                "Terminal share of EV": terminal.share_of(enterprise_value),
                "Implied exit multiple": terminal.implied_exit_multiple,
                "Implied perpetuity growth": terminal.implied_growth,
                "Formula": terminal.formula,
                "Headline": "yes" if method == result.headline_method else "no",
            }
        )
    return pd.DataFrame(rows)


def summary_frame(
    dataset: CompanyDataset,
    result: ModelResult,
    issues: list[val.Issue],
    acknowledged: dict[str, str],
    valuation_date: date,
) -> pd.DataFrame:
    bridge = result.headline_bridge
    unacknowledged_critical = [
        i
        for i in issues
        if i.severity is val.Severity.CRITICAL and i.key not in acknowledged
    ]
    rows = [
        ("Ticker", dataset.ticker),
        ("Company", dataset.name),
        ("Reporting currency", dataset.currency),
        ("Sector classification", dataset.template),
        ("Sector (Sharadar)", dataset.sector_name or "not reported"),
        ("Valuation date", valuation_date.isoformat()),
        ("Exported at", datetime.now().isoformat(timespec="seconds")),
        ("Statement basis", "as-reported" if dataset.asreported else "restated (default)"),
        ("Base period", result.base.label),
        ("Balance sheet as of", str(result.base.stocks_as_of or "n/a")),
        ("Last price", dataset.last_price),
        ("Last price date", str(dataset.price_date or "n/a")),
        ("Headline terminal value method", result.headline_terminal.method),
        ("Share basis", bridge.share_basis),
        ("Shares used", bridge.shares),
        ("Enterprise value", result.enterprise_values[result.headline_method]),
        ("Equity value", bridge.equity_value),
        ("Value per share", bridge.value_per_share),
        (
            "Upside vs current price",
            bridge.upside_vs(dataset.last_price) if dataset.last_price else None,
        ),
        ("WACC", result.wacc.wacc),
        ("WACC formula", result.wacc.formula),
        ("Unacknowledged CRITICAL issues", len(unacknowledged_critical)),
        ("Acknowledged checks", len(acknowledged)),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def build_workbook(
    *,
    dataset: CompanyDataset,
    result: ModelResult,
    assumptions: list[AssumptionRecord],
    issues: list[val.Issue],
    check_results: list[val.CheckResult],
    acknowledged: dict[str, str],
    tolerances: val.Tolerances,
    valuation_date: date,
) -> bytes:
    """Assembles the workbook and returns it as bytes for a download button."""
    buffer = io.BytesIO()

    historical = dataset.annual.copy()
    if not dataset.annual_meta.empty:
        historical = historical.join(
            dataset.annual_meta[["report_date", "source_link"]], how="left"
        )

    sheets: dict[str, pd.DataFrame] = {
        "Summary": summary_frame(dataset, result, issues, acknowledged, valuation_date),
        "Assumptions": pd.DataFrame([a.as_row() for a in assumptions]),
        "Projection": projection_frame(result),
        "Terminal Values": terminal_value_frame(result),
        "Equity Bridge": bridge_frame(result),
        "Historical Financials": historical.reset_index(),
        "Historical Ratios": dataset.ratios.reset_index(),
        "Validation Results": val.results_frame(check_results),
        "Issues": val.issues_frame(issues, acknowledged),
        "Schema Map": dataset.schema.as_frame(),
        "Tolerances": tolerances.describe(),
    }

    if dataset.quarterly is not None and not dataset.quarterly.empty:
        sheets["Quarterly"] = dataset.quarterly.reset_index()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            safe = frame.copy()
            for column in safe.columns:
                if pd.api.types.is_datetime64_any_dtype(safe[column]):
                    safe[column] = safe[column].dt.tz_localize(None)
            safe.to_excel(writer, sheet_name=name[:31], index=False)

            worksheet = writer.sheets[name[:31]]
            for index, column in enumerate(safe.columns, start=1):
                width = max(12, min(48, len(str(column)) + 4))
                worksheet.column_dimensions[
                    worksheet.cell(row=1, column=index).column_letter
                ].width = width

    buffer.seek(0)
    return buffer.getvalue()


def build_csv(
    *,
    dataset: CompanyDataset,
    result: ModelResult,
    assumptions: list[AssumptionRecord],
    issues: list[val.Issue],
    check_results: list[val.CheckResult],
    acknowledged: dict[str, str],
    valuation_date: date,
) -> str:
    """A single section-tagged CSV, for anyone who would rather not open Excel."""
    parts: list[str] = []

    def _section(title: str, frame: pd.DataFrame) -> None:
        parts.append(f"### {title}")
        parts.append(frame.to_csv(index=False))
        parts.append("")

    _section("Summary", summary_frame(dataset, result, issues, acknowledged, valuation_date))
    _section("Assumptions", pd.DataFrame([a.as_row() for a in assumptions]))
    _section("Projection", projection_frame(result))
    _section("Terminal Values", terminal_value_frame(result))
    _section("Equity Bridge", bridge_frame(result))
    _section("Validation Results", val.results_frame(check_results))
    _section("Issues", val.issues_frame(issues, acknowledged))
    _section("Schema Map", dataset.schema.as_frame())
    return "\n".join(parts)
