"""The severity banners and the validation panel — the checks before the answer."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from app.logic.dcf import financials as fin
from app.logic.dcf import model as dcf_model
from app.logic.dcf import validation as val

def _render_issue(issue: val.Issue, acknowledged: dict[str, str]) -> None:
    with st.container(border=True):
        left, right = st.columns([5, 1.3])
        with left:
            st.markdown(f"**{issue.headline()}**")
            st.markdown(issue.message)
            figures = " · ".join(f"**{k}:** {v}" for k, v in issue.values.items())
            if figures:
                st.markdown(figures)
            footer = [f"`{issue.issue_type}`", f"concept: {issue.concept}"]
            if issue.tolerance:
                footer.append(f"tolerance: {issue.tolerance}")
            st.caption(" · ".join(footer))
        with right:
            if st.button("Acknowledge", key=f"ack_{issue.key}", width="stretch"):
                acknowledged[issue.key] = datetime.now().isoformat(timespec="seconds")
                st.rerun()


def render_banners(issues: list[val.Issue], acknowledged: dict[str, str]) -> None:
    """Red block, then amber block, above everything else. Neither collapses."""
    live = [i for i in issues if i.key not in acknowledged]
    critical = [i for i in live if i.severity is val.Severity.CRITICAL]
    warnings = [i for i in live if i.severity is val.Severity.WARNING]

    if critical:
        with st.container(key="dcf_critical_banner"):
            st.markdown(f"## 🛑 {len(critical)} critical data problem(s)")
            st.markdown(
                "A number was computed, but a core input is suspect. Every figure below "
                "the fold inherits these problems."
            )
            for issue in critical:
                _render_issue(issue, acknowledged)

    if warnings:
        with st.container(key="dcf_warning_banner"):
            st.markdown(f"### ⚠️ {len(warnings)} warning(s)")
            st.markdown("Worth knowing. Not necessarily wrong.")
            for issue in warnings:
                _render_issue(issue, acknowledged)

    if acknowledged:
        acked = [i for i in issues if i.key in acknowledged]
        with st.expander(f"Acknowledged ({len(acked)})", expanded=False):
            st.caption(
                "Acknowledged checks stay listed and are stamped into the export. "
                "They reset when the ticker changes."
            )
            for issue in acked:
                col_a, col_b = st.columns([5, 1.3])
                with col_a:
                    st.markdown(
                        f"**[{issue.severity.value.upper()}] {issue.headline()}** — "
                        f"{issue.message}"
                    )
                    st.caption(f"Acknowledged at {acknowledged[issue.key]}")
                with col_b:
                    if st.button("Un-acknowledge", key=f"unack_{issue.key}", width="stretch"):
                        acknowledged.pop(issue.key, None)
                        st.rerun()

    if not live and not acknowledged:
        st.success("Every validation check passed.")


def render_validation_panel(
    dataset: dcf_model.CompanyDataset,
    check_results: list[val.CheckResult],
    tolerances: val.Tolerances,
) -> None:
    st.divider()
    st.subheader("Validation panel")

    results_frame = val.results_frame(check_results)
    passed = int((results_frame["Status"] == "pass").sum())
    failed = int((results_frame["Status"] == "fail").sum())
    skipped = int((results_frame["Status"] == "not evaluated").sum())
    columns = st.columns(4)
    columns[0].metric("Checks passed", passed)
    columns[1].metric("Checks failed", failed)
    columns[2].metric("Not evaluated", skipped)
    columns[3].metric("Fiscal years retrieved", len(dataset.annual))
    if skipped:
        st.caption(
            "“Not evaluated” means the inputs a check needs were absent — it is not a pass."
        )
    st.dataframe(results_frame, width="stretch", hide_index=True)

    with st.expander("Resolved schema mappings", expanded=False):
        st.caption(
            "Concept-to-column resolution against the columns Sharadar actually returned for "
            "this company. Nothing here is hardcoded at a call site."
        )
        st.dataframe(dataset.schema.as_frame(), width="stretch", hide_index=True)
        if dataset.schema.ambiguous:
            st.warning(
                "Some concepts matched more than one column. The first alias in the "
                "preference order was used; the alternates are listed above."
            )

    with st.expander("Tolerances in use", expanded=False):
        st.dataframe(tolerances.describe(), width="stretch", hide_index=True)

    with st.expander("Sharadar's own derived figures", expanded=False):
        if "value_check" in dataset.annual_meta.columns:
            flags = dataset.annual_meta[["value_check"]].copy()
            flags.insert(0, "Fiscal year", flags.index)
            st.dataframe(flags, width="stretch", hide_index=True)
        if "derived_fcf" in dataset.annual.columns:
            st.caption(
                "Sharadar's derived `Free Cash Flow` is shown for reference only and drives "
                "nothing. It has been observed to exceed operating cash flow for companies "
                "with positive capex, which is not possible, so it is not wired to a check."
            )
            comparison = pd.DataFrame(
                {
                    "Sharadar derived FCF": dataset.annual.get("derived_fcf"),
                    "CFO": dataset.annual.get("cfo"),
                    "CFO − CapEx": dataset.annual.get("cfo") - fin.capex(dataset.annual).values,
                }
            )
            st.dataframe(comparison, width="stretch")
