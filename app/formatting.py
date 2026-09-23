"""Number and date formatting shared by every page.

Pure string building — no Streamlit, no network. It sits at the ``app/`` root
beside ``provenance`` and for the same reason: both layers need it. The logic
layer builds a :class:`~app.logic.research.summary.StatTile` with its
``display`` string already formatted, and every view formats what it draws.

Every function answers a missing value — ``None``, ``NaN``, ``±inf``,
``pd.NA`` — with ``"not reported"`` rather than a zero or a blank, which is the
display half of the no-silent-fallbacks rule. The one exception is a table
cell, where :func:`format_date` takes ``missing="—"`` so a column of dates
does not shout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def is_missing(value) -> bool:
    """``None``, ``NaN``, ``±inf`` or ``pd.NA`` — anything that prints as missing."""
    if value is None:
        return True
    if isinstance(value, (int, float, np.floating)):
        return bool(np.isnan(value) or np.isinf(value))
    return bool(pd.isna(value))


def format_money(value, currency: str = "", digits: int = 2) -> str:
    if is_missing(value):
        return "not reported"
    prefix = f"{currency} " if currency else ""
    magnitude = abs(float(value))
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if magnitude >= cutoff:
            return f"{prefix}{float(value) / cutoff:,.2f}{suffix}"
    return f"{prefix}{float(value):,.{digits}f}"


def format_count(value) -> str:
    """Whole units with thousands separators — volumes and share counts."""
    if is_missing(value):
        return "not reported"
    magnitude = abs(float(value))
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= cutoff:
            return f"{float(value) / cutoff:,.2f}{suffix}"
    return f"{float(value):,.0f}"


def format_ratio(value, digits: int = 2) -> str:
    if is_missing(value):
        return "not reported"
    return f"{float(value):,.{digits}f}"


def format_pct(value, digits: int = 2) -> str:
    if is_missing(value):
        return "not reported"
    return f"{float(value):.{digits}%}"


def format_date(value, missing: str = "not reported") -> str:
    if is_missing(value):
        return missing
    return pd.Timestamp(value).date().isoformat()


def format_range(low, high, currency: str = "") -> str:
    """``USD 305.67 – 311.71``. The unit is stated once, on the left."""
    if is_missing(low) or is_missing(high):
        return "not reported"
    return f"{format_money(low, currency)} – {format_money(high)}"
