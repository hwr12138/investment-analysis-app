"""🧮 Evaluate — what replacing stock with calls costs, by strike and expiry, for one name.

Each cell is the call's mid as a percentage of spot (in Spread mode, the call
spread up to ten points of spot higher). Cells are shaded by the **annualised**
cost, so a two-year LEAP and a one-month call can be compared on one scale.
A strike outside the quoted range shows a dash, never an extrapolation.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from matplotlib import colormaps, colors

from app.logic.options import screens
from app.views.options.common import OptionsContext


_UNQUOTED = "background-color: #3a3d46; color: #b8bcc6"


def _shading(shade: pd.DataFrame) -> pd.DataFrame:
    """CSS per cell: darker green for cheaper per year; neutral grey where unquoted.

    Built by hand rather than with ``Styler.background_gradient``, which paints a
    missing value solid black. The grid widget also fills any cell of a styled
    frame that has no background — or a transparent one — with black, so the
    unquoted cells get an explicit grey.
    """
    values = shade.to_numpy(dtype=float)
    finite = values[~pd.isna(values)]
    css = pd.DataFrame(_UNQUOTED, index=shade.index, columns=shade.columns)
    if not finite.size:
        return css
    scale = colors.Normalize(vmin=finite.min(), vmax=max(finite.max(), finite.min() + 1e-12))
    cmap = colormaps["Greens_r"]
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            if pd.isna(value):
                continue
            fill = cmap(scale(value))
            ink = "#111111" if colors.rgb_to_hsv(fill[:3])[2] > 0.6 else "#f5f5f5"
            css.iat[i, j] = f"background-color: {colors.to_hex(fill)}; color: {ink}"
    return css


def render(context: OptionsContext) -> None:
    row = st.columns([1.2, 1.4, 3], vertical_alignment="bottom")
    with row[0]:
        default = context.visible.index("GLD") if "GLD" in context.visible else 0
        symbol = st.selectbox("Name", context.visible, index=default, key="options_evaluate_symbol")
    with row[1]:
        mode = st.radio("Structure", ("Single", "Spread"), horizontal=True, key="options_evaluate_mode")
    snap = context.snapshots[symbol]
    columns = screens.grid_expiries(snap.surface)
    grid = screens.replacement_grid(snap.quotes, columns, snap.session, snap.spot, spread=mode == "Spread")

    headers = [f"{label} · {expiry.expiration:%Y-%m-%d}" for label, expiry in columns]
    text = pd.DataFrame(index=[f"+{u:.0%}" for u in grid], columns=headers, dtype=object)
    shade = pd.DataFrame(index=text.index, columns=headers, dtype=float)
    for upside, cells in grid.items():
        for (label, _), header in zip(columns, headers):
            cell = cells[label]
            if cell.cost_pct is None:
                text.loc[f"+{upside:.0%}", header] = "—"  # an empty cell reads as a rendering fault
                continue
            text.loc[f"+{upside:.0%}", header] = f"{cell.cost_pct:.2%}" + (" °" if cell.wide else "")
            shade.loc[f"+{upside:.0%}", header] = cell.annualised

    st.subheader(f"Replacement cost — {symbol} at {snap.spot:,.2f}")
    styled = text.style.apply(lambda _: _shading(shade), axis=None).set_properties(
        **{"font-variant-numeric": "tabular-nums"}
    )
    st.dataframe(styled, width="stretch")
    what = ("the call spread from the row's strike to ten points of spot higher" if mode == "Spread"
            else "the call at spot × (1 + row)")
    st.caption(
        f"Each cell is {what}, at its mid, as a % of spot. Darker = cheaper per year of time bought. "
        "° — a bracketing quote is one-sided or wider than half its mid; — outside the quoted strikes."
    )
