"""The base of every response model, and the one rule about missing numbers.

Pydantic serialises a NaN as ``null`` — even into a field declared ``float`` —
so a gap in a pandas series would reach the frontend as a ``null`` its
generated type promises can never be there. "Missing stays missing" needs the
opposite: a value that can be absent is declared ``float | None`` and converted
on purpose with :func:`finite_or_none`, and a NaN that reaches any field
unconverted fails the request loudly instead of being renamed in transit.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    """A response model. Rejects NaN and ±infinity in every float field."""

    model_config = ConfigDict(allow_inf_nan=False, frozen=True)


def finite_or_none(value: float | None) -> float | None:
    """``None`` for a missing or non-finite number, the number otherwise."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def finite_list(values: Iterable[float]) -> list[float]:
    """Plain floats for a column the caller has already cleared of gaps."""
    return [float(value) for value in values]
