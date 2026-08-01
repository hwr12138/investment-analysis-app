"""Provenance types shared by the data and logic layers.

Every number the DCF page displays carries one of these so the UI can say where
it came from and the export can record it. The accuracy mandate forbids silent
fallbacks, so a value that could not be obtained is represented as
``value=None`` with ``Origin.UNAVAILABLE`` and a reason — never as a plausible
looking constant.

Lives at ``app/`` root rather than under ``app/data`` or ``app/logic`` because
both layers construct these and neither should import the other.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from enum import Enum


class Origin(str, Enum):
    """Where a number came from. Drives the badge rendered beside it."""

    REPORTED = "reported"      # read straight off a SimFin statement line
    DERIVED = "derived"        # arithmetic over reported figures, formula shown
    ESTIMATE = "estimate"      # an approximation; opt-in and always labelled
    USER = "user"              # typed or overridden by the user
    UNAVAILABLE = "unavailable"  # could not be obtained; never substituted

    @property
    def label(self) -> str:
        return {
            Origin.REPORTED: "REPORTED",
            Origin.DERIVED: "DERIVED",
            Origin.ESTIMATE: "ESTIMATE",
            Origin.USER: "USER",
            Origin.UNAVAILABLE: "UNAVAILABLE",
        }[self]


@dataclass(frozen=True)
class Sourced:
    """A number plus everything needed to trace it back.

    ``value`` is ``None`` exactly when the figure could not be obtained. Callers
    must check :attr:`ok` rather than coercing ``None`` to zero.
    """

    value: float | None
    origin: Origin
    source: str
    formula: str | None = None
    as_of: date | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def is_estimate(self) -> bool:
        return self.origin is Origin.ESTIMATE

    def with_user_value(self, value: float) -> "Sourced":
        """Returns a copy marked as a user override, keeping the original source."""
        return replace(
            self,
            value=value,
            origin=Origin.USER,
            source="User input",
            detail=f"Overrides {self.origin.label.lower()} value {self.value} from {self.source}"
            if self.ok
            else f"Original source unavailable: {self.detail or self.source}",
        )

    def describe(self, include_origin: bool = True) -> str:
        """One-line provenance string for tooltips and the export.

        ``include_origin`` is turned off where the origin is already shown as a
        badge beside the text, so it does not read twice.
        """
        bits = [self.origin.label] if include_origin else []
        bits.append(self.source)
        if self.formula:
            bits.append(self.formula)
        if self.as_of:
            bits.append(f"as of {self.as_of.isoformat()}")
        if self.detail:
            bits.append(self.detail)
        return " · ".join(b for b in bits if b)

    @classmethod
    def reported(cls, value: float, source: str, as_of: date | None = None) -> "Sourced":
        return cls(value=value, origin=Origin.REPORTED, source=source, as_of=as_of)

    @classmethod
    def derived(
        cls, value: float, source: str, formula: str, as_of: date | None = None
    ) -> "Sourced":
        return cls(
            value=value, origin=Origin.DERIVED, source=source, formula=formula, as_of=as_of
        )

    @classmethod
    def estimate(
        cls, value: float, source: str, detail: str, formula: str | None = None,
        as_of: date | None = None,
    ) -> "Sourced":
        return cls(
            value=value,
            origin=Origin.ESTIMATE,
            source=source,
            formula=formula,
            as_of=as_of,
            detail=detail,
        )

    @classmethod
    def user(cls, value: float, detail: str = "") -> "Sourced":
        return cls(value=value, origin=Origin.USER, source="User input", detail=detail)

    @classmethod
    def unavailable(cls, source: str, reason: str) -> "Sourced":
        """No value. The reason is shown to the user; nothing is substituted."""
        return cls(value=None, origin=Origin.UNAVAILABLE, source=source, detail=reason)
