"""Streamlit pieces more than one page draws the same way.

Only what is genuinely identical across its callers lives here. Most of a tab's
markup is its own — its CSS travels with it, and two tabs with a function of
the same name (``_render_legend``, ``_render_sources``) usually say different
things in it. Promote something here when a second caller would otherwise copy
it, not before.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import streamlit as st

from app.data import market_data
from app.data.sharadar_client import SharadarError

T = TypeVar("T")


def render_blocking(title: str, body: str, detail: str = "") -> None:
    """A condition that stops the page: what happened, why, and what it means."""
    st.error(f"**{title}**\n\n{body}")
    if detail:
        st.caption(detail)


def load_tab(what: str, loader: Callable[..., T], *args) -> T | None:
    """A Stock Research tab's own fetch, drawn the same way on every tab.

    Each tab loads what it needs the first time it is opened, so each one used
    to be free to say "loading" — or fail — in its own words. This is the one
    way: a spinner naming what is being fetched, and a Sharadar failure drawn as
    a blocking notice in the tab's body. Returns ``None`` after drawing that
    notice; the caller returns without rendering anything else.

    ``loader`` is the tab's cached wrapper, so the spinner shows once per
    company and a reopened tab draws without it.
    """
    try:
        with st.spinner(f"Loading {what}…"):
            return loader(*args)
    except SharadarError as error:
        render_blocking(f"Could not load {what}", str(error))
        return None


def risk_free_rate_input(*, key: str | None = None) -> float:
    """The short risk-free rate as a decimal, pre-filled from ``^IRX`` and editable.

    Fetched once per session, so the field keeps what was typed into it across
    reruns. When the fetch fails the field starts at zero and says so. It never
    starts at a plausible constant, because a reader cannot tell one from a
    quote.
    """
    if "risk_free_rate_sourced" not in st.session_state:
        st.session_state["risk_free_rate_sourced"] = (
            market_data.fetch_current_risk_free_rate()
        )
    sourced = st.session_state["risk_free_rate_sourced"]

    rate_pct = st.number_input(
        "Risk-Free Rate (%)",
        value=sourced.value * 100 if sourced.ok else 0.0,
        step=0.1,
        key=key,
        help="Defaults to the 13-week Treasury bill yield (^IRX) from Yahoo Finance.",
    )
    if sourced.ok:
        st.caption(sourced.describe())
    else:
        st.warning(
            f"The risk-free rate could not be fetched — {sourced.detail}. Enter it "
            "manually; nothing has been substituted."
        )
    return rate_pct / 100.0
