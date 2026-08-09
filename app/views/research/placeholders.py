"""Tabs that are not built yet.

Each entry names what the tab will show and which Sharadar table will feed it,
so the next person does not have to re-derive where the figures come from.
Order is immaterial here — ``_TAB_ORDER`` in the package ``__init__`` decides
that.

Deliberately fetches nothing: ``st.tabs`` renders every tab body on every
rerun, so a placeholder that loaded data would fire on page load and on every
window change.
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
    "⚖️ Valuation": (
        "P/E, P/S, P/B and EV/EBITDA bands over five years, with today's "
        "reading against the company's own history.",
        "`SHARADAR/DAILY` via `sharadar_client.fetch_daily_metrics`.",
    ),
    "🌱 Growth": (
        "Revenue, earnings and free-cash-flow growth, with CAGRs.",
        "`SHARADAR/SF1` annual and TTM.",
    ),
    "💰 Profitability": (
        "Margins, ROE, ROIC and turnover, against history.",
        "`SHARADAR/SF1` DERIVED frame — `grossmargin`, `netmargin`, `roe`, "
        "`roic`, `assetturnover`.",
    ),
    "🧭 Peers": (
        "Comparison against companies in the same industry.",
        "`SHARADAR/TICKERS` on `sicindustry`/`famaindustry`, then `DAILY` for "
        "the multiples.",
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
