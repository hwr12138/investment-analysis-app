"""How far behind the market the stored data is.

The collector runs after ThetaData's free tier releases a session — overnight —
so on a weekday the newest session the page can have is the previous weekday's.
Holidays are not removed, so the day after a market holiday reads one session
behind when it is not; that errs toward saying "behind", never toward hiding it.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np


def expected_session(today: date) -> date:
    """The latest weekday strictly before ``today``."""
    day = today - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def sessions_behind(last_session: date, today: date) -> int:
    """Weekdays after ``last_session`` up to and including the expected one."""
    expected = expected_session(today)
    if last_session >= expected:
        return 0
    return int(np.busday_count(last_session + timedelta(days=1), expected + timedelta(days=1)))
