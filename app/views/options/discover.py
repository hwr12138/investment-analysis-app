"""🔭 Discover — where calls are cheap against each name's own year.

The scatter puts the buy-vol score against a percentile of the reader's
choosing; the table below it is the one-month 25Δ call on every name, sorted by
the score. A name with under half a year of stored history has no score and
no percentile, and is counted beneath the chart rather than plotted at zero.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.logic.options import snapshot
from app.logic.options.tolerances import DEFAULT
from app.views.options.common import OptionsContext, theme_colours

#: Y-axis choices: label → column of the discover table (a 0–1 percentile).
_Y_AXES = {
    "25Δ call-skew percentile": "skew_pct",
    "30d IV percentile": "iv_pct",
    "Reward:risk percentile": "rr_pct",
}


@st.cache_data(show_spinner=False, max_entries=4)
def _table(night: str, _context: OptionsContext):
    """Cached per stored night. ``_context`` is not hashed; ``night`` names it —
    the session and the identity of the solved snapshots, which the shell's
    resource cache replaces when a new night is written."""
    return snapshot.discover_table(_context.snapshots, _context.metrics)


def render(context: OptionsContext) -> None:
    table = _table(f"{context.session}:{id(context.snapshots)}", context)
    table = table[table["symbol"].isin(context.visible)]
    # Missing figures arrive as None in object columns, which the grid prints as
    # "None"; as floats they are NaN and the cell is left blank.
    numeric = [c for c in table.columns if c not in ("symbol", "theme", "expiration", "wide")]
    table = table.astype({c: float for c in numeric})

    y_label = st.selectbox("Y axis", list(_Y_AXES), key="options_discover_y")
    y = _Y_AXES[y_label]
    plotted = table.dropna(subset=["score", y])
    colours = theme_colours(sorted(set(table["theme"])))
    figure = go.Figure()
    for theme, group in plotted.groupby("theme"):
        figure.add_trace(go.Scatter(
            x=group["score"], y=group[y] * 100, mode="markers+text", name=theme, text=group["symbol"],
            textposition="top center", textfont=dict(size=10), marker=dict(size=9, color=colours[theme]),
            hovertemplate="%{text}<br>score %{x:.0f}<br>" + y_label + " %{y:.0f}<extra></extra>",
        ))
    figure.add_vline(x=0, line_dash="dash", line_color="rgba(160,160,255,0.6)")
    figure.update_layout(
        template="plotly_dark", height=560, margin=dict(t=20, b=40, l=40, r=20),
        xaxis_title="Buy-vol score (higher = calls cheaper than usual)", yaxis_title=y_label,
        yaxis=dict(range=[-3, 103]), legend=dict(font=dict(size=11)),
    )
    st.plotly_chart(figure, width="stretch")
    unplotted = sorted(set(table["symbol"]) - set(plotted["symbol"]))
    if unplotted:
        st.caption(f"Not plotted — under {DEFAULT.min_history_sessions} stored sessions of history, so no score yet: {', '.join(unplotted)}")

    st.subheader("One-month 25Δ calls")
    shown = table.sort_values("score", ascending=False, na_position="last").assign(
        cost_pct=lambda f: f["cost_pct"] * 100, annualised=lambda f: f["annualised"] * 100,
        otm=lambda f: f["otm"] * 100, iv=lambda f: f["iv"] * 100,
        quote=lambda f: f["wide"].map({True: "° wide", False: ""}),
    )
    st.dataframe(
        shown[["symbol", "theme", "spot", "score", "z IV level", "z IV − realized", "z 25Δ call skew",
               "expiration", "strike", "delta", "cost", "cost_pct", "annualised", "otm", "iv", "quote"]],
        hide_index=True, width="stretch",
        column_config={
            "symbol": "Name", "theme": "Theme",
            "spot": st.column_config.NumberColumn("Spot", format="%.2f"),
            "score": st.column_config.NumberColumn("Buy-vol score", format="%.1f"),
            "z IV level": st.column_config.NumberColumn("z IV", format="%.2f", help="Today's 30d ATM IV against its year"),
            "z IV − realized": st.column_config.NumberColumn("z IV−RV", format="%.2f"),
            "z 25Δ call skew": st.column_config.NumberColumn("z skew", format="%.2f"),
            "expiration": st.column_config.DateColumn("Expiry"),
            "strike": st.column_config.NumberColumn("Strike", format="%g"),
            "delta": st.column_config.NumberColumn("Δ", format="%.2f"),
            "cost": st.column_config.NumberColumn("Cost", format="%.2f", help="Mid of the 17:15 ET NBBO"),
            "cost_pct": st.column_config.NumberColumn("Cost %", format="%.2f%%"),
            "annualised": st.column_config.NumberColumn("Ann.", format="%.1f%%", help="Cost % × 252 ÷ trading days"),
            "otm": st.column_config.NumberColumn("OTM", format="%.1f%%"),
            "iv": st.column_config.NumberColumn("IV", format="%.1f%%"),
            "quote": st.column_config.TextColumn("Quote", help="° — the bid/ask is wider than half its mid"),
        },
    )
