"""The reverse DCF: the revenue growth the market price implies.

Named for what it shows rather than ``reverse`` so it cannot be confused with
``app.logic.dcf.reverse``, the solver it displays.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.formatting import format_pct
from app.logic.dcf import financials as fin
from app.logic.dcf import inputs as dcf_inputs
from app.logic.dcf import model as dcf_model
from app.logic.dcf import reverse
from app.provenance import Sourced
from app.views.research.dcf import common

#: The banner for a solve outcome, as ``(level, markdown)`` — shared with the
#: web app, so a status without a message fails a test on both.
_reverse_message = dcf_inputs.reverse_verdict

#: Both methods are solved; Gordon leads because it is the fundamentals
#: reading — the exit-multiple solve mostly measures today's multiple against
#: the company's own history.
METHOD_ORDER = dcf_inputs.REVERSE_METHODS
METHOD_LABELS = dcf_inputs.REVERSE_LABELS
_PATH_COLOURS = {"gordon": "#f2a900", "exit_multiple": "#e06666"}


_implied_revenue_path = dcf_inputs.implied_revenue_path


def _path_shape(inputs: dict) -> str:
    """How the solved rate is spread across the forecast, in words."""
    years = inputs["drivers"].years
    if inputs.get("growth_fade_to") is None:
        return f"a flat revenue growth rate across all {years} forecast years"
    hold = inputs.get("growth_hold", 1)
    return (
        f"a year-1 revenue growth rate, held for {hold} year(s) and then faded toward "
        f"the terminal rate over the rest of the {years}-year forecast, exactly as yours is"
    )


def _render_chart(
    dataset: dcf_model.CompanyDataset,
    result: dcf_model.ModelResult,
    reverse_results: dict[str, reverse.ReverseDcfResult],
    history: fin.GrowthHistory,
    inputs: dict,
) -> None:
    """Reported revenue, then the paths the price (both ways), the forecast, and history imply."""
    chart = dcf_inputs.revenue_chart(dataset, result, reverse_results, history, inputs)
    if chart is None:
        return
    axis = [*chart.history_labels, chart.anchor, *chart.forward_labels]

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=list(chart.history_labels), y=list(chart.reported),
            name="Reported revenue", marker_color="rgba(130,150,180,0.55)",
        )
    )
    styles = {"price": ("dash", 3.0), "forecast": ("dot", 2.5), "history": ("longdash", 1.5)}
    colours = {"forecast": "#4fa3f7", "history": "#8fd18f"}
    for path in chart.paths:
        dash, width = styles[path.kind]
        # Anchored at the base period so every path fans out from one point.
        figure.add_trace(
            go.Scatter(
                x=[chart.anchor, *chart.forward_labels],
                y=[chart.base_revenue, *path.values],
                name=path.name, mode="lines+markers",
                line=dict(
                    dash=dash, width=width,
                    color=_PATH_COLOURS[path.method] if path.method else colours[path.kind],
                ),
            )
        )

    figure.update_layout(
        height=420, template="plotly_dark", hovermode="x unified",
        margin=dict(t=30, b=10, l=10, r=10),
        yaxis_title=f"Revenue ({dataset.valuation_currency})",
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(categoryorder="array", categoryarray=axis),
    )
    figure.update_traces(
        hovertemplate="%{fullData.name}<br>%{x}: %{y:,.0f}<extra></extra>"
    )
    st.plotly_chart(figure, width="stretch")

    note = (
        f"Forecast years are labelled by offset: each is the twelve months after the base "
        f"period, **{result.base.label}**, which is where every path starts."
    )
    if result.base.is_ttm:
        note += (
            " The base period is a trailing twelve months, so the step from the last "
            "fiscal-year bar to the anchor is a real discontinuity, not a plotting artefact."
        )
    if dataset.is_converted:
        note += (
            f" Reported bars are converted at today's spot rate, like the forecast, so they "
            f"compare on one axis; the history tables above stay in {dataset.currency}."
        )
    st.caption(note)


def render(
    dataset: dcf_model.CompanyDataset,
    result: dcf_model.ModelResult,
    inputs: dict,
    reverse_results: dict[str, reverse.ReverseDcfResult],
    history: fin.GrowthHistory,
) -> None:
    st.divider()
    st.subheader("Reverse DCF — what the market price implies")

    st.caption(
        f"Everything above holds the assumptions fixed and solves for value. This holds value "
        f"fixed at the market price and solves for the one assumption that would justify it: "
        f"{_path_shape(inputs)}, with the other drivers exactly as shown above. It is solved "
        f"under both terminal methods, whichever one is the headline."
    )

    forecast = inputs["drivers"].revenue_growth[0]
    columns = st.columns(5)
    for column, method in zip(columns, METHOD_ORDER):
        outcome = reverse_results[method]
        solved = outcome.status is reverse.Status.SOLVED
        column.metric(
            f"Implied growth — {METHOD_LABELS[method]}",
            format_pct(outcome.growth) if solved else
            ("no single rate" if outcome.roots else "—"),
            help="The year-1 rate that makes this model's value per share equal the "
                 "market price under this terminal method.",
        )
    columns[2].metric("Your year-1 forecast", format_pct(forecast))
    columns[3].metric(
        f"Realised CAGR ({history.first_period}–{history.last_period})"
        if history.ok else "Realised CAGR",
        format_pct(history.cagr) if history.ok else "n/a",
        help=history.detail or (
            f"Compound annual growth over {history.intervals} interval(s) of reported revenue."
        ),
    )
    columns[4].metric(
        "Market price",
        f"{dataset.last_price:,.2f}" if dataset.last_price else "n/a",
        help=f"Last close as of {dataset.price_date}" if dataset.price_date else None,
    )

    st.caption(
        "**How to read the two.** The Gordon solve is the fundamentals reading: the growth "
        "the business must deliver for the price to be right when growth after the forecast "
        "earns only its cost of capital. The exit-multiple solve holds the terminal multiple "
        "at the company's own recent median, so it mostly measures how far today's multiple "
        "sits from that history rather than what the business must do."
    )

    for method in METHOD_ORDER:
        outcome = reverse_results[method]
        if outcome.status is reverse.Status.SOLVED:
            verdict = (
                f"**{METHOD_LABELS[method]}:** the price implies "
                f"**{format_pct(outcome.growth)}** in year 1"
                + (", then fading" if inputs.get("growth_fade_to") is not None else " a year")
                + f", against your **{format_pct(forecast)}**."
            )
            if history.ok:
                gap = (outcome.growth - history.cagr) * 100
                verdict += (
                    f" That is **{gap:+.1f} points** {'above' if gap >= 0 else 'below'} the "
                    f"{format_pct(history.cagr)} the company compounded over the "
                    f"{history.intervals} years to {history.last_period}."
                )
            st.markdown(verdict)
            common.provenance(
                Sourced.derived(
                    value=outcome.growth,
                    source=(
                        f"Solved against the last close of {dataset.last_price:,.2f}"
                        + (f" ({dataset.price_date})" if dataset.price_date else "")
                        + f", {METHOD_LABELS[method].lower()}"
                    ),
                    formula=outcome.formula,
                )
            )
        level, message = _reverse_message(outcome)
        if message:
            {"warning": st.warning, "error": st.error, "info": st.info}[level](
                f"*{METHOD_LABELS[method]}* — {message}"
            )

    if history.ok and history.arithmetic_mean is not None:
        st.caption(
            f"**Two ways to state past growth, and they are not the same number.** The "
            f"*arithmetic average* of yearly growth rates is "
            f"**{format_pct(history.arithmetic_mean)}**; the *compound* rate is "
            f"**{format_pct(history.cagr)}**, which is the one comparable to a forecast that "
            f"compounds. They diverge whenever growth was uneven, and the average is always "
            f"the higher. Yearly rates ranged "
            f"{format_pct(history.minimum)} to {format_pct(history.maximum)}, median "
            f"{format_pct(history.median)}, over {history.observations} year(s)."
            + (f" {history.detail}" if history.detail else "")
        )

    _render_chart(dataset, result, reverse_results, history, inputs)

    scanned = dcf_inputs.scan_rows(reverse_results)
    if scanned:
        with st.expander("Value per share across the growth range", expanded=False):
            st.caption(
                "The scans the solvers already ran, so this costs nothing extra. Every "
                "crossing of the market price in a column is a rate the price is consistent "
                "with under that method."
            )
            rows = []
            for growth, values in scanned:
                row = {"Year-1 revenue growth": f"{growth:.2%}"}
                for method in METHOD_ORDER:
                    value = values[method]
                    row[METHOD_LABELS[method]] = f"{value:,.2f}" if value is not None else ""
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            if dataset.last_price:
                st.caption(f"Market price: **{dataset.last_price:,.2f}**.")

    _render_whatif(dataset, inputs)


def _render_whatif(dataset: dcf_model.CompanyDataset, inputs: dict) -> None:
    """A second target price, solved against the same assumptions as the run above.

    The assumptions come from the snapshot the page is showing rather than being
    re-read from the widgets, so these answers and the implied rates above are
    always points on the same curves.
    """
    with st.expander("Solve for a different price", expanded=False):
        st.caption(
            "Answers *what growth would justify this price?* — using the same assumptions "
            "as the implied rates above, with only the target price changed."
        )
        default = float(dataset.last_price or 0.0)
        price = st.number_input(
            "Price per share", value=default, min_value=0.0, step=1.0, key="dcf_reverse_price"
        )
        if price <= 0 or abs(price - default) < 1e-12:
            return

        for method in METHOD_ORDER:
            what_if = dcf_inputs.solve_price(dataset, inputs, price, method)
            label = METHOD_LABELS[method]
            if what_if.status is reverse.Status.SOLVED:
                st.markdown(
                    f"**{label}:** a price of **{price:,.2f}** implies year-1 revenue growth "
                    f"of **{format_pct(what_if.growth)}**"
                    + (", fading from there." if inputs.get("growth_fade_to") is not None
                       else " a year, held flat.")
                )
                common.provenance(
                    Sourced.derived(
                        value=what_if.growth,
                        source=f"Solved against a hypothetical price of {price:,.2f}, "
                               f"{label.lower()}",
                        formula=what_if.formula,
                    )
                )
            else:
                level, message = _reverse_message(what_if)
                {"warning": st.warning, "error": st.error, "info": st.info}[level or "info"](
                    f"*{label}* — {message or what_if.reason}"
                )
