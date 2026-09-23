"""Streamlit pieces more than one page draws the same way.

Only what is genuinely identical across its callers lives here. Most of a tab's
markup is its own — its CSS travels with it, and two tabs with a function of
the same name (``_render_legend``, ``_render_sources``) usually say different
things in it. Promote something here when a second caller would otherwise copy
it, not before.
"""

from __future__ import annotations

import streamlit as st


def render_blocking(title: str, body: str, detail: str = "") -> None:
    """A condition that stops the page: what happened, why, and what it means."""
    st.error(f"**{title}**\n\n{body}")
    if detail:
        st.caption(detail)
