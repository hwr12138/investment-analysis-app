"""Tabs that are not built yet.

Each entry names what the tab will show and which Sharadar table will feed it,
so the next person does not have to re-derive where the figures come from.
Order is immaterial here — ``_TAB_ORDER`` in the package ``__init__`` decides
that.

Fetches nothing. A built tab that needs data loads it on first open, through
``components.load_tab``, the way every built tab does.
"""

from __future__ import annotations

import streamlit as st

#: Every tab still to be built, keyed by the label ``_TAB_ORDER`` places it by.
PLACEHOLDERS = {
    "🎯 Earnings": (
        "EPS and revenue history, surprises, and margins by quarter.",
        "`SHARADAR/SF1` at the `ARQ` dimension, which carries real filing dates.",
    ),
    "💵 Dividends": (
        "Full dividend and split history, yield over time, payout ratio.",
        "`SHARADAR/ACTIONS` via `sharadar_client.fetch_actions` — already "
        "fetched for the ex-dividend tile on Summary.",
    ),
    "💰 Profitability": (
        "Margins, ROE, ROIC and turnover, against history.",
        "`SHARADAR/SF1` DERIVED frame — `grossmargin`, `netmargin`, `roe`, "
        "`roic`, `assetturnover`.",
    ),
    "🧭 Peers": (
        "The named companies behind the ⚖️ Valuation tab's industry medians, "
        "ranked — who is dearest, who is cheapest, and where this one sits.",
        "`sharadar_client.fetch_universe_daily` and `fetch_ticker_directory`, "
        "already on the snapshot as `peer_universe` — no further requests.",
    ),
    "🕰️ History": (
        "Long-run total return, drawdowns, and volatility.",
        "`SHARADAR/SEP` `closeadj` — the dividend-adjusted total-return series.",
    ),
}


def render(name: str, what: str, source: str) -> None:
    """A tab that is not built yet."""
    label = name.split(" ", 1)[-1]
    st.subheader(label)
    st.info(f"**Not built yet.** Planned: {what}")
    st.caption(f"Will be sourced from {source}")
