"""Market rates the pages pre-fill and the reader can change.

Only the short risk-free rate so far, which both portfolio pages start from.
A failed fetch is a ``null`` value with the reason, never a stand-in number:
the page then asks the reader for one.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter

from app.api.contract import Contract, finite_or_none
from app.data import market_data

router = APIRouter(prefix="/rates", tags=["rates"])


class RiskFreeRate(Contract):
    """The 13-week Treasury bill yield, as a decimal, and where it came from."""

    value: float | None
    source: str
    as_of: date | None
    #: Why there is no value. Empty when there is one.
    detail: str


@router.get("/risk-free")
def risk_free_rate() -> RiskFreeRate:
    sourced = market_data.fetch_current_risk_free_rate()
    return RiskFreeRate(
        value=finite_or_none(sourced.value),
        source=sourced.source,
        as_of=sourced.as_of,
        detail=sourced.detail,
    )
