"""Provenance for numbers that can fall back to a default.

Every input to a valuation is either observed — quoted from a live source or
derived from reported financials — or a hardcoded stand-in. Once a stand-in
reaches a discounted cash flow it is indistinguishable from a real figure, so
values carry where they came from and, when defaulted, why. The views render
that alongside the number.

Shared by the data and logic layers: a fetch that failed and a margin the
filings never reported are different events, but both leave a default sitting
in the model, and both deserve to be visible.
"""

from enum import Enum
from typing import NamedTuple, Optional


class Origin(str, Enum):
    """Where a resolved value came from, and if it was defaulted, why."""

    OBSERVED = "observed"
    """Quoted from a live source or derived from reported financials."""

    MISSING_CONFIG = "missing_config"
    """An API key or setting is absent. Permanent until someone fixes it, and
    fixable — worth shouting about rather than quietly degrading."""

    FETCH_FAILED = "fetch_failed"
    """The source was reachable in principle but yielded nothing usable.
    Transient, so the next rerun may well succeed."""

    NOT_APPLICABLE = "not_applicable"
    """The concept doesn't apply to this filer — banks report no working
    capital — so the default is the right answer, not a degradation."""

    INSUFFICIENT_DATA = "insufficient_data"
    """The inputs needed to derive the value weren't in the filings."""

    IMPLAUSIBLE = "implausible"
    """A value was derived but failed a sanity bound. The rejected figure is
    itself signal — a tax rate of 300% means the company booked a loss — so it
    belongs in the detail rather than being discarded."""


# How loudly a view should report each origin. MISSING_CONFIG is an error
# because someone can act on it; NOT_APPLICABLE is informational because
# nothing is wrong.
_SEVERITY = {
    Origin.OBSERVED: None,
    Origin.MISSING_CONFIG: "error",
    Origin.FETCH_FAILED: "warning",
    Origin.IMPLAUSIBLE: "warning",
    Origin.INSUFFICIENT_DATA: "info",
    Origin.NOT_APPLICABLE: "info",
}


class Resolved(NamedTuple):
    """A number paired with an account of where it came from.

    Callers unpack .value for arithmetic, which makes ignoring .is_fallback an
    explicit choice rather than the path of least resistance.
    """

    value: float
    source: str
    origin: Origin = Origin.OBSERVED
    detail: str = ""

    @property
    def is_fallback(self) -> bool:
        return self.origin is not Origin.OBSERVED

    @property
    def severity(self) -> Optional[str]:
        """Name of the st.* method a view should report this with, or None if clean."""
        return _SEVERITY[self.origin]

    @classmethod
    def observed(cls, value: float, source: str) -> "Resolved":
        return cls(float(value), source)

    @classmethod
    def fallback(cls, value: float, origin: Origin, detail: str) -> "Resolved":
        return cls(float(value), "default", origin, detail)
