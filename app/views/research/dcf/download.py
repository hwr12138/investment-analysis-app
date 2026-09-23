"""The export: the workbook and CSV, built from the snapshot on screen."""

from __future__ import annotations

from datetime import date

import streamlit as st

from app.logic.dcf import export as dcf_export
from app.logic.dcf import model as dcf_model
from app.logic.dcf import validation as val

def render(
    dataset: dcf_model.CompanyDataset,
    result: dcf_model.ModelResult,
    inputs: dict,
    issues: list[val.Issue],
    check_results: list[val.CheckResult],
    acknowledged: dict[str, str],
    valuation_date: date,
) -> None:
    st.divider()
    st.subheader("Export")
    st.caption(
        "The workbook carries every assumption with its provenance, the projection line by "
        "line, the equity bridge, all validation results, and which checks were acknowledged. "
        "It is built from the same snapshot the page is showing."
    )

    # Acknowledgements change without a recompute, so the export is rebuilt when
    # they move — but not on every unrelated widget interaction.
    cache_key = (id(result), tuple(sorted(acknowledged.items())))
    cached = st.session_state.get("dcf_export_cache")
    if cached is None or cached["key"] != cache_key:
        try:
            workbook = dcf_export.build_workbook(
                dataset=dataset,
                result=result,
                assumptions=inputs["records"],
                issues=issues,
                check_results=check_results,
                acknowledged=acknowledged,
                tolerances=inputs["tolerances"],
                valuation_date=valuation_date,
            )
        except Exception as error:  # noqa: BLE001 - surfaced, never swallowed
            st.error(f"The workbook could not be built: {error}")
            return

        csv_text = dcf_export.build_csv(
            dataset=dataset,
            result=result,
            assumptions=inputs["records"],
            issues=issues,
            check_results=check_results,
            acknowledged=acknowledged,
            valuation_date=valuation_date,
        )
        cached = {"key": cache_key, "workbook": workbook, "csv": csv_text}
        st.session_state["dcf_export_cache"] = cached

    workbook = cached["workbook"]
    csv_text = cached["csv"]

    stamp = valuation_date.isoformat()
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download Excel workbook",
            data=workbook,
            file_name=f"{dataset.ticker}_dcf_{stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    with right:
        st.download_button(
            "Download CSV",
            data=csv_text,
            file_name=f"{dataset.ticker}_dcf_{stamp}.csv",
            mime="text/csv",
            width="stretch",
        )
