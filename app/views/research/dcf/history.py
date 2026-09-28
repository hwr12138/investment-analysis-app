"""The historical financials: the ratios the drivers come from, and their filings."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.formatting import format_date
from app.logic.dcf import financials as fin
from app.logic.dcf import model as dcf_model

def render(dataset: dcf_model.CompanyDataset) -> None:
    st.divider()
    st.subheader("Historical financials")

    ratios = dataset.ratios.copy()
    usable = {
        fin.DRIVER_LABELS[key]: int(ratios[key].notna().sum())
        for key in fin.DRIVER_KEYS
        if key in ratios.columns
    }
    st.caption(
        f"{len(dataset.annual)} fiscal years retrieved. Years used per driver: "
        + " · ".join(f"{k}: {v}" for k, v in usable.items())
        + ". Years missing a component are excluded, never zero-filled."
        + (
            f" Figures here are in the reporting currency, {dataset.currency}."
            if dataset.is_converted
            else ""
        )
    )

    display = ratios.copy()
    display.columns = [fin.RATIO_LABELS.get(c, c) for c in display.columns]
    st.dataframe(
        display.style.format("{:.2%}", na_rep="—"), width="stretch"
    )

    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Revenue and EBIT", "Driver ratios"),
    )
    index = [str(i) for i in dataset.annual.index]
    if "revenue" in dataset.annual.columns:
        figure.add_trace(
            go.Bar(x=index, y=dataset.annual["revenue"], name="Revenue"), row=1, col=1
        )
    if "ebit" in dataset.annual.columns:
        figure.add_trace(
            go.Bar(x=index, y=dataset.annual["ebit"], name="EBIT"), row=1, col=1
        )
    for key in ("operating_margin", "capex_pct_revenue", "dna_pct_revenue", "effective_tax_rate"):
        if key in ratios.columns:
            figure.add_trace(
                go.Scatter(
                    x=index, y=ratios[key], mode="lines+markers",
                    name=fin.DRIVER_LABELS[key], connectgaps=False,
                ),
                row=2, col=1,
            )
    figure.update_layout(
        height=650, template="plotly_dark", margin=dict(t=40, b=10, l=10, r=10), barmode="group"
    )
    figure.update_yaxes(tickformat=".0%", row=2, col=1)
    st.plotly_chart(figure, width="stretch")

    st.caption(
        "These are the ratios the drivers are derived from, not the statements they "
        "were derived from. For the line items themselves — all four statements, "
        "sectioned, reconciled, annual or quarterly — see the **📄 Financials** tab."
    )
    _render_filing_provenance(dataset)


def _render_filing_provenance(dataset: dcf_model.CompanyDataset) -> None:
    """Which filing each fiscal year came from, and whether it was restated.

    Kept on this tab rather than left to Financials, which lists the same dates
    but not the filings themselves: this is the only path from a valuation back
    to the documents behind it.

    Read off the income statement, though any frame would do — the client
    partitions one SF1 row per period into ``PL``/``BS``/``CF``, so all three
    carry identical metadata.
    """
    frame = dataset.statements_annual.frame("PL")
    if frame.empty or "Fiscal Year" not in frame.columns:
        return

    with st.expander("The filing behind each year", expanded=False):
        rows = []
        for _, record in frame.iterrows():
            year = record["Fiscal Year"]
            if pd.isna(year):
                continue
            rows.append(
                {
                    "Fiscal year": str(int(year)),
                    "Period": record.get("Fiscal Period", ""),
                    "Report date": format_date(record.get("Report Date"), missing="—"),
                    "Publish date": format_date(record.get("Publish Date"), missing="—"),
                    "Restated": "yes" if record.get("Restated") else "no",
                    "Source filing": dataset.filing_link(int(year)),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Source filing": st.column_config.LinkColumn(
                    "Source filing", display_text="View filing"
                )
            },
        )
        st.caption(
            "Report date is the period end; publish date is when the filing reached "
            "Sharadar. A restated year is one the company later revised — this model "
            "always uses the restated figures, so that the time series the drivers are "
            "averaged over is a consistent one."
        )
