"""What more than one part of the 💰 DCF tab draws: badges, money, blocking screens.

The badge classes are styled by the tab's stylesheet in the package
``__init__``, which is injected once per render before any of these draw.
"""

from __future__ import annotations

import streamlit as st

from app.formatting import format_money
from app.logic.dcf import inputs as dcf_inputs
from app.provenance import Origin, Sourced
from app.views import components

_BADGE_CLASS = {
    Origin.REPORTED: "dcf-reported",
    Origin.DERIVED: "dcf-derived",
    Origin.ESTIMATE: "dcf-estimate",
    Origin.USER: "dcf-user",
    Origin.UNAVAILABLE: "dcf-unavailable",
}


def _badge(origin: Origin) -> str:
    return (
        f'<span class="dcf-badge {_BADGE_CLASS[origin]}">{origin.label}</span>'
    )


def provenance(sourced: Sourced) -> None:
    st.markdown(
        f'{_badge(sourced.origin)} '
        f'<span class="dcf-prov">{sourced.describe(include_origin=False)}</span>',
        unsafe_allow_html=True,
    )


def money(value, currency: str = "") -> str:
    """Whole currency units below a million — a valuation does not quote cents."""
    return format_money(value, currency, digits=0)


def render_blocking(title: str, body: str, detail: str = "") -> None:
    """The shared blocking screen, plus what BLOCKING means on this tab."""
    components.render_blocking(title, body, detail)
    st.info(
        "This is a **BLOCKING** condition: the model cannot be computed, so no valuation "
        "is shown. It is not a judgement that the data looks wrong — it is that the data "
        "needed is absent."
    )


def model_kwargs(inputs: dict) -> dict:
    """Every ``run_model`` argument the assumption set supplies, bar the drivers.

    :func:`app.logic.dcf.inputs.run_kwargs`, which the forward run, the reverse
    solve and the what-if price all go through, so the reverse DCF inverts
    exactly the model on screen.
    """
    return dcf_inputs.run_kwargs(inputs)
