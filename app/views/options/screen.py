"""📋 Screen — the call spreads that pay in full at a +k SD move.

Ranked by the payout **at the ask**, with the mid beside it. On a wide market
the two differ by a multiple — GLD's 398/399 for 12/18 paid 5.7× at the mid
and 1.9× at the ask on 2026-10-23 — and a screen ranked on mids rewards exactly
the names a buyer cannot trade at the mid.
"""

from __future__ import annotations

import streamlit as st

from app.logic.options import snapshot
from app.views.options.common import OptionsContext

_SDS = {f"+{sd:g} SD": sd for sd in snapshot.SD_TARGETS}

#: The grid's columns; the stored table also carries each leg's strike.
_SHOWN = ("symbol", "theme", "move", "spread", "price", "width", "bid", "ask",
          "payout_mid", "payout_ask", "reward_risk")


def render(context: OptionsContext) -> None:
    visible = {s: context.snapshots[s] for s in context.visible}
    choices = snapshot.spread_expirations(visible)
    if not choices:
        st.info("No expiry is listed for enough of the names on screen.")
        return
    row = st.columns([1.3, 2, 2.7], vertical_alignment="bottom")
    with row[0]:
        sd_label = st.selectbox("Move", list(_SDS), key="options_screen_sd")
    with row[1]:
        expiration = st.selectbox(
            "Expiry", choices, index=choices.index(snapshot.default_spread_expiry(choices, context.session)),
            key="options_screen_expiry",
            format_func=lambda e: f"{e:%Y-%m-%d} ({(e - context.session).days}d)",
        )
    table = snapshot.spread_table(visible, expiration, _SDS[sd_label])
    st.subheader(f"Call spreads paying in full at {sd_label} · {expiration:%Y-%m-%d}")
    if table.empty:
        st.info("No name has a tightly quoted spread at that target.")
        return
    table = table.astype({"payout_ask": float})  # None → NaN, so the cell is blank, not "None"
    shown = table.sort_values("payout_ask", ascending=False, na_position="last").assign(move=lambda f: f["move"] * 100)
    st.dataframe(
        shown[list(_SHOWN)], hide_index=True, width="stretch",
        column_config={
            "symbol": "Name", "theme": "Theme",
            "move": st.column_config.NumberColumn("Move to target", format="+%.1f%%",
                                                  help="spot × e^(k·σ·√t) at the expiry's ATM IV"),
            "spread": "Spread",
            "price": st.column_config.NumberColumn("Price (mid)", format="%.2f"),
            "width": st.column_config.NumberColumn("Width", format="%.2f"),
            "bid": st.column_config.NumberColumn("Spread bid", format="%.2f", help="Floored at zero"),
            "ask": st.column_config.NumberColumn("Spread ask", format="%.2f"),
            "payout_mid": st.column_config.NumberColumn("Payout at mid", format="%.1f×"),
            "payout_ask": st.column_config.NumberColumn("Payout at ask", format="%.1f×"),
            "reward_risk": st.column_config.NumberColumn("Reward:risk", format="%.2f×",
                                                         help="Mid payout ÷ the payout at a flat smile; 1.0× is fair"),
        },
    )
    st.caption(f"{len(table)} of {len(visible)} names list a tightly quoted spread at this target. "
               "Legs with a bid/ask wider than half their mid are skipped, not priced.")
