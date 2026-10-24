"""Thresholds the options arithmetic compares against.

Owned here, per the repo rule that a tolerance belongs to its consumer. None of
them is shared with the DCF's or the research tabs' — ``min_history_sessions``
counts trading sessions of option history, which no other tolerance does.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptionsTolerances:
    #: Strikes nearest spot that the put-call parity forward is the median over.
    forward_strikes: int = 6

    #: Fewer out-of-the-money quotes than this and an expiry has no smile.
    min_otm_quotes: int = 5

    #: The constant-maturity point every history series is read at.
    constant_maturity_days: int = 30

    #: Expiries closer than this are excluded from the 30-day bracket: a few days
    #: from expiry, a mid of a cent moves the implied vol by points.
    min_bracket_days: int = 7

    #: Expiries further out than this are not used for the 30-day bracket. The
    #: backfill fetched only up to 65 days, so this must stay at or above it.
    max_bracket_days: int = 100

    #: The "1M" expiry is the one nearest 30 days inside this window.
    one_month_days: tuple[int, int] = (20, 45)

    #: A quote whose (ask − bid) ÷ mid exceeds this is flagged as wide.
    wide_quote_ratio: float = 0.5

    #: Tight call quotes nearest a target that the smile's local slope is fitted
    #: through for the smile-based reward:risk, and the fewest it will fit.
    slope_quotes: int = 7
    min_slope_quotes: int = 4

    #: Sessions in the "own history" window — one trading year.
    history_window: int = 252

    #: Below this many sessions in the window a percentile or z-score is not
    #: shown: half a year of daily points.
    min_history_sessions: int = 120

    #: Sessions in the realized-vol window, the trading-day twin of 30 calendar days.
    realized_vol_sessions: int = 21


DEFAULT = OptionsTolerances()
