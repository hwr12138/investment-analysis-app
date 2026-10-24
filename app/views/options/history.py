"""📈 History — one name's reward:risk over its trailing year, with ATM IV and skew.

Reads only ``metrics/daily.parquet``: one row a session, each figure at
constant 30-day maturity, so the line moves with the market rather than with
the expiry calendar. The strip above the chart says where today sits in the
year; under half a year of stored history it says how much there is instead.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.formatting import format_date
from app.logic.options.history import own_history_stats
from app.logic.options.tolerances import DEFAULT
from app.views.options.common import OptionsContext


def _quartile(percentile: float) -> str:
    return ("worst quartile", "below median", "above median", "best quartile")[min(int(percentile * 4), 3)]


def render(context: OptionsContext) -> None:
    default = context.visible.index("GLD") if "GLD" in context.visible else 0
    symbol = st.selectbox("Name", context.visible, index=default, key="options_history_symbol")
    rows = context.metrics[context.metrics["symbol"] == symbol].set_index("session").sort_index()
    rows = rows.iloc[-DEFAULT.history_window:]
    stats = own_history_stats(rows["reward_risk"])

    st.subheader(f"Reward:risk — {symbol}")
    if stats is None:
        st.info(
            f"{rows['reward_risk'].notna().sum()} of the {DEFAULT.min_history_sessions} sessions needed for a "
            "percentile are stored. The collector adds one a night."
        )
    else:
        cells = st.columns(5)
        cells[0].metric(format_date(stats.as_of), f"{stats.latest:.2f}×", _quartile(stats.percentile), delta_color="off")
        cells[1].metric("Percentile of own year", f"{stats.percentile:.0%}")
        cells[2].metric("Median", f"{stats.median:.2f}×")
        cells[3].metric("Middle half", f"{stats.q25:.2f}–{stats.q75:.2f}")
        cells[4].metric(f"Range ({stats.sessions} sessions)", f"{stats.low:.2f}–{stats.high:.2f}")

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    if stats is not None:
        figure.add_hrect(y0=stats.q25, y1=stats.q75, fillcolor="rgba(255,200,120,0.10)", line_width=0)
        figure.add_hline(y=stats.median, line_dash="dash", line_color="rgba(160,160,255,0.7)",
                         annotation_text=f"median {stats.median:.2f}×", annotation_position="top left")
    figure.add_trace(go.Scatter(x=rows.index, y=rows["reward_risk"], name="Reward:risk",
                                line=dict(color="#7c83fd", width=2)), secondary_y=False)
    figure.add_trace(go.Scatter(x=rows.index, y=rows["atm_iv_30"] * 100, name="30d ATM IV (%)",
                                line=dict(color="#66bb6a", width=1, dash="dot")), secondary_y=True)
    figure.add_trace(go.Scatter(x=rows.index, y=rows["skew_30"] * 100, name="25Δ call skew (vol pts)",
                                line=dict(color="#d4a017", width=1, dash="dot")), secondary_y=True)
    figure.update_layout(template="plotly_dark", height=460, margin=dict(t=20, b=30, l=40, r=40),
                         legend=dict(orientation="h", y=-0.15))
    figure.update_yaxes(title_text="Reward : risk", secondary_y=False)
    figure.update_yaxes(title_text="Surface (%)", secondary_y=True, showgrid=False)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Reward:risk — the +1 SD call spread's payout ÷ the payout a flat smile at ATM IV gives, read off the "
        "smile at a constant 30 days. Above 1.0× the spread costs less than flat vol says it should."
    )
