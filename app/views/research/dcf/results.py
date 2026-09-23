"""The answer: the company header, the projection, and the valuation."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from app.formatting import format_pct
from app.logic.dcf import export as dcf_export
from app.logic.dcf import model as dcf_model
from app.logic.dcf import validation as val
from app.logic.dcf import valuation as dcf
from app.views.research.dcf import common

def render_header(dataset: dcf_model.CompanyDataset, valuation_date: date) -> None:
    with st.container(border=True):
        st.subheader(f"{dataset.name} ({dataset.ticker})")
        columns = st.columns(5)
        columns[0].metric("Sector (Sharadar)", dataset.sector_name or "not reported")
        columns[1].metric("Currency", dataset.currency or "not reported")

        fye_month = dataset.fiscal_year_end_month
        fye_label = (
            date(2000, fye_month, 1).strftime("%B") if fye_month else "not reported"
        )
        columns[2].metric("Fiscal year end", fye_label)

        price_label = (
            f"{dataset.last_price:,.2f}" if dataset.last_price is not None else "n/a"
        )
        columns[3].metric(
            "Last close",
            price_label,
            help=f"As of {dataset.price_date}" if dataset.price_date else None,
        )
        columns[4].metric("Market cap", common.money(dataset.reported_market_cap, dataset.currency))

        st.caption(
            f"Price as of {dataset.price_date} · share count as of "
            f"{dataset.shares_point_in_time_date} · statements "
            f"{'as-reported' if dataset.asreported else 'restated'} · "
            f"{dataset.template or 'sector not classified'} · "
            f"{len(dataset.annual)} fiscal years retrieved"
        )


def _cross_implied(terminal: dcf.TerminalValue) -> str:
    """Restates one terminal value method in the other method's units.

    Gordon growth produces a terminal value, which divided by terminal EBITDA is
    an exit multiple. An exit multiple produces a terminal value, which inverted
    through the Gordon formula is a perpetual growth rate. Each row therefore
    reports a different quantity, and the label says which.
    """
    if terminal.implied_exit_multiple is not None:
        return f"an exit multiple of {terminal.implied_exit_multiple:,.2f}×"
    if terminal.implied_growth is not None:
        return f"perpetual growth of {terminal.implied_growth:.2%}"
    return "—"


def render_valuation(
    dataset: dcf_model.CompanyDataset,
    result: dcf_model.ModelResult,
    inputs: dict,
    unacknowledged_critical: list[val.Issue],
) -> None:
    st.divider()
    st.subheader("Valuation")

    marker = ""
    if unacknowledged_critical:
        marker = " ⚠️"
        st.error(
            f"⚠️ **The figures below carry {len(unacknowledged_critical)} unacknowledged "
            f"critical data problem(s).** Read the red banner above before using this number."
        )

    bridge = result.headline_bridge
    upside = bridge.upside_vs(dataset.last_price) if dataset.last_price else None
    columns = st.columns(4)
    columns[0].metric(
        f"Value per share{marker}",
        f"{bridge.value_per_share:,.2f}",
        help=f"Headline method: {result.headline_terminal.method}",
    )
    columns[1].metric("Current price", f"{dataset.last_price:,.2f}" if dataset.last_price else "n/a")
    columns[2].metric("Upside / downside", format_pct(upside) if upside is not None else "n/a")
    columns[3].metric("WACC", f"{result.wacc.wacc:.2%}")

    st.markdown("#### Both terminal value methods, side by side")
    rows = []
    for method, terminal in result.terminal_values.items():
        enterprise_value = result.enterprise_values[method]
        method_bridge = result.bridges[method]
        rows.append(
            {
                "Method": terminal.method + (" (headline)" if method == result.headline_method else ""),
                "Terminal value": common.money(terminal.value, dataset.currency),
                "PV of terminal value": common.money(terminal.present_value, dataset.currency),
                "Enterprise value": common.money(enterprise_value, dataset.currency),
                "Equity value": common.money(method_bridge.equity_value, dataset.currency),
                "Value per share": f"{method_bridge.value_per_share:,.2f}",
                "TV share of EV": format_pct(terminal.share_of(enterprise_value), 1),
                # Each method is cross-checked by restating it in the *other*
                # method's units, so the two columns are deliberately different
                # quantities rather than a like-for-like pair.
                "This method implies": _cross_implied(terminal),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.caption(
        "**Reading the cross-check.** Each method is restated in the other's terms, so the "
        "last column is a different quantity on each row — that is the point of it, not a "
        "misalignment. The Gordon growth row answers *“what exit multiple would my "
        "perpetuity assumption have to be worth?”*, and the exit multiple row answers *“what "
        "perpetual growth rate is my chosen multiple secretly assuming?”* If the exit "
        "multiple row implies growth you would never underwrite, the multiple is too high."
    )

    gordon_vps = result.bridges["gordon"].value_per_share
    exit_vps = result.bridges["exit_multiple"].value_per_share
    if gordon_vps:
        st.caption(
            f"The two methods differ by **{abs(exit_vps / gordon_vps - 1):.1%}** on value per "
            f"share. A gap is informative, not an error."
        )
    st.caption(
        f"Terminal value discounted at exponent {result.headline_terminal.discount_exponent:.4f}, "
        f"the same convention as the final projected year "
        f"({'mid-year' if result.projection.mid_year else 'end-of-year'})."
    )

    st.markdown("#### Equity bridge")
    st.caption(
        f"Every bridge item is taken from the balance sheet as of "
        f"**{result.base.stocks_as_of}**. Sharadar provides no treasury-stock-method calculation "
        f"for outstanding options and RSUs, so the diluted count is not fully loaded."
    )
    bridge_rows = [
        {
            "Line": line.label,
            "Amount": common.money(line.amount, dataset.currency),
            "Source": "reported" if line.reported else "not reported",
            "Note": line.note,
        }
        for line in bridge.lines
    ]
    bridge_rows.append(
        {
            "Line": f"÷ shares — {bridge.share_basis}",
            "Amount": f"{bridge.shares:,.0f}",
            "Source": "reported",
            "Note": "",
        }
    )
    bridge_rows.append(
        {
            "Line": "= Value per share",
            "Amount": f"{bridge.value_per_share:,.2f}",
            "Source": "derived",
            "Note": "",
        }
    )
    st.dataframe(pd.DataFrame(bridge_rows), width="stretch", hide_index=True)


def render_projection(
    dataset: dcf_model.CompanyDataset, result: dcf_model.ModelResult
) -> None:
    st.divider()
    st.subheader("Projection")
    st.caption(
        "FCFF = EBIT × (1 − tax) + D&A − CapEx − ΔNWC − SBC deducted. "
        "PV = FCFF for the period × discount factor."
    )

    frame = dcf_export.projection_frame(result)
    display = frame.copy()
    for column in (
        "Revenue (full year)", "EBIT", "NOPAT", "D&A", "CapEx", "ΔNWC",
        "SBC deducted", "FCFF (full year)", "FCFF (this period)", "PV of FCFF",
    ):
        display[column] = frame[column].map(lambda v: common.money(v, dataset.currency))
    for column in ("Revenue growth", "Operating margin", "Tax rate"):
        display[column] = frame[column].map(lambda v: f"{v:.2%}")
    for column in ("Period fraction", "Discount exponent", "Discount factor"):
        display[column] = frame[column].map(lambda v: f"{v:.4f}")

    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        f"Sum of discounted explicit cash flows: "
        f"**{common.money(result.projection.pv_explicit, dataset.currency)}** · "
        f"PV of terminal value: "
        f"**{common.money(result.headline_terminal.present_value, dataset.currency)}**"
    )
