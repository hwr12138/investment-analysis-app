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
from app.logic.dcf import valuation as dcf
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


#: Both methods are solved; Gordon leads because it is the fundamentals
#: reading — the exit-multiple solve mostly measures today's multiple against
#: the company's own history.
METHOD_ORDER = ("gordon", "exit_multiple")
METHOD_LABELS = {
    "gordon": "Gordon growth (fundamentals)",
    "exit_multiple": "Exit multiple",
}
_PATH_COLOURS = {"gordon": "#f2a900", "exit_multiple": "#e06666"}


def _implied_revenue_path(
    base_revenue: float,
    growth: float,
    years: int,
    fade_to: float | None = None,
    hold: int = 1,
) -> list[float]:
    """Revenue compounding from the base period along the forecast's growth shape."""
    path, revenue = [], base_revenue
    for rate in dcf.growth_path(growth, years, fade_to, hold=hold):
        revenue *= 1.0 + rate
        path.append(revenue)
    return path


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
    if "revenue" not in dataset.annual.columns:
        return

    projected = [row.revenue for row in result.projection.rows]
    years = len(projected)
    base_revenue = result.projection.base_revenue
    fade_to, hold = inputs.get("growth_fade_to"), inputs.get("growth_hold", 1)

    # Forecast years are labelled by offset, not by fiscal year: each is the
    # twelve months after the base period, which need not end on a fiscal year.
    history_labels = [str(i) for i in dataset.annual.index]
    forward_labels = [f"+{i}yr" for i in range(1, years + 1)]
    anchor = result.base.label
    axis = history_labels + [anchor] + forward_labels

    # Reported bars stay in the reporting currency; the forecast is in the
    # valuation currency. Converting the bars back at the same spot rate puts
    # both on one axis without restating history at historical rates.
    to_valuation = dataset.fx_rate_value
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=history_labels, y=dataset.annual["revenue"] * to_valuation,
            name="Reported revenue", marker_color="rgba(130,150,180,0.55)",
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

    for method in METHOD_ORDER:
        outcome = reverse_results.get(method)
        if outcome is None:
            continue
        # Losing the market's line is the finding; losing the chart hides it,
        # so every root is drawn when there is no single answer.
        roots = [outcome.growth] if outcome.status is reverse.Status.SOLVED else outcome.roots
        for root in roots:
            add_path(
                _implied_revenue_path(base_revenue, root, years, fade_to, hold),
                f"Price, {METHOD_LABELS[method].lower()} ({format_pct(root)})",
                "dash", 3.0, _PATH_COLOURS[method],
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

    samples = {
        method: dict(reverse_results[method].samples) for method in METHOD_ORDER
    }
    if any(samples.values()):
        with st.expander("Value per share across the growth range", expanded=False):
            st.caption(
                "The scans the solvers already ran, so this costs nothing extra. Every "
                "crossing of the market price in a column is a rate the price is consistent "
                "with under that method."
            )
            low, high = reverse.PLAUSIBLE_GROWTH
            growths = sorted(
                g for g in set().union(*(set(v) for v in samples.values())) if low <= g <= high
            )
            rows = []
            for growth in growths:
                row = {"Year-1 revenue growth": f"{growth:.2%}"}
                for method in METHOD_ORDER:
                    value = samples[method].get(growth)
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
            what_if = dcf_model.solve_implied_growth(
                dataset,
                inputs["base"],
                target_price=price,
                drivers=inputs["drivers"],
                growth_fade_to=inputs.get("growth_fade_to"),
                growth_hold=inputs.get("growth_hold", 1),
                **{**common.model_kwargs(inputs), "headline_method": method},
            )
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
