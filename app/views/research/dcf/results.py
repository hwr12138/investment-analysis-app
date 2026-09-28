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
        currency = dataset.currency or "not reported"
        if dataset.is_converted:
            currency = f"{dataset.currency} → {dataset.valuation_currency}"
        columns[1].metric("Currency", currency)

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
        columns[4].metric("Market cap", common.money(dataset.reported_market_cap, dataset.valuation_currency))

        st.caption(
            f"Price as of {dataset.price_date} · share count as of "
            f"{dataset.shares_point_in_time_date} · statements "
            f"{'as-reported' if dataset.asreported else 'restated'} · "
            f"{dataset.template or 'sector not classified'} · "
            f"{len(dataset.annual)} fiscal years retrieved"
        )
        if dataset.is_converted:
            rate = dataset.fx_rate
            st.caption(
                f"**Reported in {dataset.currency}, valued in {dataset.valuation_currency}.** "
                f"Drivers come from the {dataset.currency} history, so growth and margins "
                f"carry no exchange-rate noise; the base period is converted once at a spot "
                f"rate of **{rate.value:.6g} {dataset.valuation_currency} per "
                f"{dataset.currency}**, which the valuation assumes holds. History tables "
                f"stay in {dataset.currency}; the projection, the bridge and value per share "
                f"are in {dataset.valuation_currency}."
            )
            common.provenance(rate)


def _cross_implied(terminal: dcf.TerminalValue) -> str:
    """Restates one terminal value method in the other method's units.

    Gordon growth produces a terminal value, which divided by terminal EBITDA is
    an exit multiple. An exit multiple produces a terminal value, which says
    what perpetual growth it needs even if growth were free, and what return on
    new capital it needs at the terminal growth rate. Each row therefore reports
    a different quantity, and the label says which.
    """
    if terminal.implied_exit_multiple is not None:
        return f"an exit multiple of {terminal.implied_exit_multiple:,.2f}×"
    if terminal.implied_growth is not None:
        text = f"perpetual growth of at least {terminal.implied_growth:.2%}"
        if terminal.implied_ronic is not None:
            text += f"; RONIC of {terminal.implied_ronic:.1%} at the terminal rate"
        elif terminal.terminal_year is not None and terminal.terminal_year.growth > 0:
            text += "; no return on new capital reaches it at the terminal rate"
        return text
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
                "Terminal value": common.money(terminal.value, dataset.valuation_currency),
                "PV of terminal value": common.money(terminal.present_value, dataset.valuation_currency),
                "Enterprise value": common.money(enterprise_value, dataset.valuation_currency),
                "Equity value": common.money(method_bridge.equity_value, dataset.valuation_currency),
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
        "is my chosen multiple secretly assuming?”* — the perpetual growth it needs even if "
        "growth cost nothing, and the return on new capital it needs at your terminal rate. "
        "If either is something you would never underwrite, the multiple is too high."
    )

    gordon_vps = result.bridges["gordon"].value_per_share
    exit_vps = result.bridges["exit_multiple"].value_per_share
    if gordon_vps:
        st.caption(
            f"The two methods differ by **{abs(exit_vps / gordon_vps - 1):.1%}** on value per "
            f"share. A gap is informative, not an error."
        )
    gordon = result.terminal_values["gordon"]
    exit_multiple = result.terminal_values["exit_multiple"]
    st.caption(
        f"Gordon value discounted at exponent {gordon.discount_exponent:.4f}, the final "
        f"forecast year's own ({'mid-year' if result.projection.mid_year else 'end-of-year'}) "
        f"convention, because it is a perpetuity of flows arriving the same way. Exit value "
        f"discounted at {exit_multiple.discount_exponent:.4f} — the end of year N — because a "
        f"multiple is a sale price struck on that date."
    )

    steady = gordon.terminal_year
    if steady is not None:
        st.markdown("#### Terminal year (N + 1)")
        st.caption(
            "The Gordon value capitalises a normalised year rather than growing the final "
            "forecast year: reinvestment is set to what growing at g costs at the return on "
            "new capital, so reinvestment sized for the forecast's growth is not carried "
            "into perpetuity. With the EBITA proxy on, NOPAT adds back the final year's "
            "D&A in excess of capex, grown at g — amortisation that capex never replaces."
        )
        terminal_rows = [
            ("Revenue", common.money(steady.revenue, dataset.valuation_currency)),
            ("Operating margin", f"{steady.operating_margin:.2%}"),
            ("EBIT", common.money(steady.ebit, dataset.valuation_currency)),
            (
                "Amortisation add-back (EBITA proxy)",
                common.money(steady.amortisation, dataset.valuation_currency)
                if inputs.get("ebita_proxy", True) else "off",
            ),
            ("NOPAT", common.money(steady.nopat, dataset.valuation_currency)),
            ("Tax rate", f"{steady.tax_rate:.2%}"),
            ("Return on new invested capital", f"{steady.ronic:.2%}"),
            ("Reinvestment rate (g ÷ RONIC)", f"{steady.reinvestment_rate:.2%}"),
            ("Reinvestment", common.money(steady.reinvestment, dataset.valuation_currency)),
            ("FCFF", common.money(steady.fcff, dataset.valuation_currency)),
        ]
        st.dataframe(
            pd.DataFrame(terminal_rows, columns=["Item", "Year N + 1"]),
            width="stretch",
            hide_index=True,
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
            "Amount": common.money(line.amount, dataset.valuation_currency),
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
        "FCFF = EBIT × (1 − tax) + D&A − CapEx − ΔNWC. PV = FCFF × discount factor. "
        "EBIT is after stock-based compensation, which is not added back."
    )

    frame = dcf_export.projection_frame(result)
    display = frame.copy()
    for column in (
        "Revenue", "EBIT", "NOPAT", "D&A", "CapEx", "ΔNWC", "FCFF", "PV of FCFF",
    ):
        display[column] = frame[column].map(lambda v: common.money(v, dataset.valuation_currency))
    for column in ("Revenue growth", "Operating margin", "Tax rate"):
        display[column] = frame[column].map(lambda v: f"{v:.2%}")
    for column in ("Years after base period", "Discount exponent", "Discount factor"):
        display[column] = frame[column].map(lambda v: f"{v:.4f}")

    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        f"Sum of discounted explicit cash flows: "
        f"**{common.money(result.projection.pv_explicit, dataset.valuation_currency)}** · "
        f"PV of terminal value: "
        f"**{common.money(result.headline_terminal.present_value, dataset.valuation_currency)}**"
    )


def render_sensitivity(
    dataset: dcf_model.CompanyDataset, grid: dcf_model.SensitivityGrid
) -> None:
    """Headline value per share across WACC × the headline's own terminal input."""
    # Lower case mid-sentence, but Gordon is a name.
    method = "exit multiple" if grid.method == "exit_multiple" else "Gordon growth"
    st.markdown(f"#### Sensitivity — value per share, {method} headline")
    across = (
        "the exit multiple at −20%, −10%, as set, +10% and +20% across the top — terminal "
        "growth barely moves an exit-multiple value, so it is not what the grid varies"
        if grid.method == "exit_multiple"
        else "terminal growth across the top"
    )
    st.caption(
        f"The {method} headline rerun with the WACC down the side and {across}; "
        "everything else as above. The centre cell is the headline. A blank cell is one "
        "where the model cannot be built — a WACC at or below terminal growth, where no "
        "perpetuity exists."
    )
    frame = pd.DataFrame(
        [
            [f"{value:,.2f}" if value is not None else "" for value in row]
            for row in grid.values
        ],
        index=pd.Index([f"{rate:.2%}" for rate in grid.waccs], name="WACC"),
        columns=pd.Index([grid.column_heading(c) for c in grid.columns], name=grid.column_label),
    )
    st.dataframe(frame, width="stretch")
    if dataset.last_price:
        st.caption(f"Last close: **{dataset.last_price:,.2f}**.")

