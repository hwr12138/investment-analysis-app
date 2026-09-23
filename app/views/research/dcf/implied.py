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
from app.logic.dcf import model as dcf_model
from app.logic.dcf import reverse
from app.provenance import Sourced
from app.views.research.dcf import common

def _reverse_message(result: reverse.ReverseDcfResult) -> tuple[str, str]:
    """The banner for a solve outcome, as ``(level, markdown)``.

    Pure and separate from the rendering so every status can be checked without
    a Streamlit runtime — and so adding a status without a message fails a test
    rather than rendering blank.
    """
    status = result.status

    if status is reverse.Status.SOLVED:
        if result.growth_creates_value is False:
            return "warning", (
                f"**Growth destroys value in this model.** At the implied rate, one more "
                f"percentage point of revenue growth *lowers* value per share by "
                f"**{abs(result.value_per_growth_point):,.2f}**. A *higher* market price "
                f"therefore implies a *lower* growth rate — the usual reading of this number "
                f"is inverted. This happens when each incremental dollar of revenue consumes "
                f"more working capital and capital expenditure than the margin it earns."
            )
        if result.plausible is False:
            return "warning", (
                f"**{format_pct(result.growth)} a year is outside anything a company sustains for "
                f"long.** Read this as *the price is not explained by revenue growth* rather "
                f"than as a forecast — the difference is more likely to sit in the margin, "
                f"the reinvestment rates, or the discount rate."
            )
        return "", ""

    if status is reverse.Status.MULTIPLE_SOLUTIONS:
        return "warning", f"**No single implied rate.** {result.reason}"

    if status is reverse.Status.NOT_EVALUABLE:
        return "error", (
            f"{result.reason}\n\nThis is not a failed search. The terminal growth rate and "
            f"the discount rate are held fixed while revenue growth is solved for, so this "
            f"fails identically at every growth rate — on the first evaluation or not at all. "
            f"Both terminal value methods are always computed, so a Gordon growth input can "
            f"stop an exit-multiple model."
        )

    if status is reverse.Status.NO_TARGET:
        return "info", f"**No market price to invert.** {result.reason}"

    if status in {reverse.Status.TARGET_ABOVE_RANGE, reverse.Status.TARGET_BELOW_RANGE}:
        return "info", f"**No growth rate reproduces this price.** {result.reason}"

    if status is reverse.Status.DEGENERATE:
        return "info", f"**Nothing to solve for.** {result.reason}"

    if status is reverse.Status.RESIDUAL_TOO_LARGE:
        return "error", f"**The solved rate did not check out.** {result.reason}"

    if status is reverse.Status.SEARCH_BROKEN:
        return "error", f"**The search could not complete.** {result.reason}"

    raise AssertionError(f"No message written for reverse DCF status {status!r}.")


def _implied_revenue_path(base_revenue: float, growth: float, years: int) -> list[float]:
    return [base_revenue * (1.0 + growth) ** i for i in range(1, years + 1)]


def _render_chart(
    dataset: dcf_model.CompanyDataset,
    result: dcf_model.ModelResult,
    reverse_result: reverse.ReverseDcfResult,
    history: fin.GrowthHistory,
) -> None:
    """Reported revenue, then the paths the price, the forecast, and history imply."""
    if "revenue" not in dataset.annual.columns:
        return

    projected = [row.revenue for row in result.projection.rows]
    years = len(projected)
    base_revenue = result.projection.base_revenue

    # Forecast years are labelled by offset, not by fiscal year: the first is a
    # stub period, and a fiscal-year label would claim a precision the model
    # does not have.
    history_labels = [str(i) for i in dataset.annual.index]
    forward_labels = [f"+{i}yr" for i in range(1, years + 1)]
    anchor = result.base.label
    axis = history_labels + [anchor] + forward_labels

    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=history_labels, y=dataset.annual["revenue"], name="Reported revenue",
            marker_color="rgba(130,150,180,0.55)",
        )
    )

    def add_path(values: list[float], name: str, dash: str, width: float, color: str) -> None:
        # Anchored at the base period so every path fans out from one point.
        figure.add_trace(
            go.Scatter(
                x=[anchor] + forward_labels, y=[base_revenue] + values, name=name,
                mode="lines+markers", line=dict(dash=dash, width=width, color=color),
            )
        )

    if reverse_result.status is reverse.Status.SOLVED:
        add_path(
            _implied_revenue_path(base_revenue, reverse_result.growth, years),
            f"Implied by the market price ({format_pct(reverse_result.growth)})",
            "dash", 3.5, "#f2a900",
        )
    else:
        # Losing the market's line is the finding; losing the chart hides it.
        for index, root in enumerate(reverse_result.roots):
            add_path(
                _implied_revenue_path(base_revenue, root, years),
                f"Reproduces the price ({format_pct(root)})",
                "dash", 2.5, ["#f2a900", "#e06666"][index % 2],
            )

    add_path(projected, "Your forecast", "dot", 2.5, "#4fa3f7")

    if history.ok:
        add_path(
            _implied_revenue_path(base_revenue, history.cagr, years),
            f"History repeating ({format_pct(history.cagr)} realised)",
            "longdash", 1.5, "#8fd18f",
        )

    figure.update_layout(
        height=420, template="plotly_dark", hovermode="x unified",
        margin=dict(t=30, b=10, l=10, r=10),
        yaxis_title=f"Revenue ({dataset.currency})" if dataset.currency else "Revenue",
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(categoryorder="array", categoryarray=axis),
    )
    figure.update_traces(
        hovertemplate="%{fullData.name}<br>%{x}: %{y:,.0f}<extra></extra>"
    )
    st.plotly_chart(figure, width="stretch")

    note = (
        f"Forecast years are labelled by offset because the first is a stub period of "
        f"{result.projection.stub_fraction:.2f} of a year. All three paths start from the "
        f"base period, **{result.base.label}**."
    )
    if result.base.is_ttm:
        note += (
            " The base period is a trailing twelve months, so the step from the last "
            "fiscal-year bar to the anchor is a real discontinuity, not a plotting artefact."
        )
    st.caption(note)


def render(
    dataset: dcf_model.CompanyDataset,
    result: dcf_model.ModelResult,
    inputs: dict,
    reverse_result: reverse.ReverseDcfResult,
    history: fin.GrowthHistory,
) -> None:
    st.divider()
    st.subheader("Reverse DCF — what the market price implies")

    years = inputs["drivers"].years
    st.caption(
        f"Everything above holds the assumptions fixed and solves for value. This holds value "
        f"fixed at the market price and solves for the one assumption that would justify it: "
        f"a flat revenue growth rate across all {years} forecast years, with the other five "
        f"drivers exactly as shown above."
    )

    solved = reverse_result.status is reverse.Status.SOLVED
    forecast = inputs["drivers"].revenue_growth[0]

    columns = st.columns(4)
    columns[0].metric(
        "Implied revenue growth",
        format_pct(reverse_result.growth) if solved else
        ("no single rate" if reverse_result.roots else "—"),
        help="The flat rate that makes this model's value per share equal the market price.",
    )
    columns[1].metric("Your forecast", format_pct(forecast))
    columns[2].metric(
        f"Realised CAGR ({history.first_period}–{history.last_period})"
        if history.ok else "Realised CAGR",
        format_pct(history.cagr) if history.ok else "n/a",
        help=history.detail or (
            f"Compound annual growth over {history.intervals} interval(s) of reported revenue."
        ),
    )
    columns[3].metric(
        "Market price",
        f"{dataset.last_price:,.2f}" if dataset.last_price else "n/a",
        help=f"Last close as of {dataset.price_date}" if dataset.price_date else None,
    )

    if solved:
        verdict = (
            f"The market is pricing in **{format_pct(reverse_result.growth)}** a year for {years} "
            f"years. You are forecasting **{format_pct(forecast)}**."
        )
        if history.ok:
            gap = (reverse_result.growth - history.cagr) * 100
            verdict += (
                f" The company delivered **{format_pct(history.cagr)}** over the "
                f"{history.intervals} years to {history.last_period} — the market's number is "
                f"**{gap:+.1f} points** {'above' if gap >= 0 else 'below'} what it has "
                f"actually done."
            )
        st.markdown(verdict)
        common.provenance(
            Sourced.derived(
                value=reverse_result.growth,
                source=(
                    f"Solved against the last close of {dataset.last_price:,.2f}"
                    + (f" ({dataset.price_date})" if dataset.price_date else "")
                ),
                formula=reverse_result.formula,
            )
        )

    level, message = _reverse_message(reverse_result)
    if message:
        {"warning": st.warning, "error": st.error, "info": st.info}[level](message)

    if history.ok and history.arithmetic_mean is not None:
        st.caption(
            f"**Two ways to state past growth, and they are not the same number.** The "
            f"forecast driver above is pre-populated from the *arithmetic average* of yearly "
            f"growth rates (**{format_pct(history.arithmetic_mean)}**); the benchmark here is the "
            f"*compound* rate (**{format_pct(history.cagr)}**), which is the one comparable to a "
            f"flat forecast. They diverge whenever growth was uneven. Yearly rates ranged "
            f"{format_pct(history.minimum)} to {format_pct(history.maximum)}, median "
            f"{format_pct(history.median)}, over {history.observations} year(s)."
            + (f" {history.detail}" if history.detail else "")
        )

    _render_chart(dataset, result, reverse_result, history)

    if reverse_result.samples:
        with st.expander("Value per share across the growth range", expanded=False):
            st.caption(
                "The scan the solver already ran, so this costs nothing extra. Every crossing "
                "of the market price in this table is a rate the price is consistent with."
            )
            low, high = reverse.PLAUSIBLE_GROWTH
            rows = [
                {
                    "Revenue growth": f"{growth:.2%}",
                    "Value per share": f"{value:,.2f}",
                    "vs market price": (
                        format_pct(value / dataset.last_price - 1) if dataset.last_price else "n/a"
                    ),
                }
                for growth, value in reverse_result.samples
                if low <= growth <= high
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    _render_whatif(dataset, inputs, reverse_result)


def _render_whatif(
    dataset: dcf_model.CompanyDataset, inputs: dict, reverse_result: reverse.ReverseDcfResult
) -> None:
    """A second target price, solved against the same assumptions as the run above.

    The assumptions come from the snapshot the page is showing rather than being
    re-read from the widgets, so this answer and the implied rate above are
    always two points on one curve.
    """
    with st.expander("Solve for a different price", expanded=False):
        st.caption(
            "Answers *what growth would justify this price?* — using the same assumptions "
            "as the implied rate above, with only the target price changed."
        )
        default = float(dataset.last_price or 0.0)
        price = st.number_input(
            "Price per share", value=default, min_value=0.0, step=1.0, key="dcf_reverse_price"
        )
        if price <= 0 or abs(price - default) < 1e-12:
            return

        what_if = dcf_model.solve_implied_growth(
            dataset,
            inputs["base"],
            target_price=price,
            drivers=inputs["drivers"],
            wacc_result=inputs["wacc"],
            terminal_growth=inputs["terminal_growth"],
            exit_multiple=inputs["exit_multiple"],
            stub=inputs["stub"],
            mid_year=inputs["mid_year"],
            shares=inputs["shares"],
            share_basis=inputs["share_basis"],
            include_short_term_investments=inputs["include_sti"],
            operating_cash_floor=inputs["cash_floor"],
            non_operating_assets=inputs["non_operating"],
            base_fcff=inputs["base_fcff"],
            headline_method=inputs["headline_method"],
            sbc=inputs["sbc"],
            deduct_sbc=inputs["deduct_sbc"],
        )
        if what_if.status is reverse.Status.SOLVED:
            st.markdown(
                f"A price of **{price:,.2f}** implies flat revenue growth of "
                f"**{format_pct(what_if.growth)}** a year."
            )
            common.provenance(
                Sourced.derived(
                    value=what_if.growth,
                    source=f"Solved against a hypothetical price of {price:,.2f}",
                    formula=what_if.formula,
                )
            )
        else:
            level, message = _reverse_message(what_if)
            {"warning": st.warning, "error": st.error, "info": st.info}[level or "info"](
                message or what_if.reason
            )
