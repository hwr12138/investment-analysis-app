"""One metrics row per symbol per session, and a symbol against its own year.

The collector calls :func:`session_metrics` once a night for each name (and
once per past session in the backfill); the page reads the accumulated rows and
calls :func:`own_history_stats`. Every series is read at constant maturity
(30 days) and, for skew, at constant delta (25Δ), so the year it is ranked
against measures the market and not the expiry calendar.

**Reward:risk** is the +1 SD call spread's payout divided by the payout a flat
smile at the ATM IV would give, at constant 30-day maturity and read off the
smile (:func:`app.logic.options.screens.smile_reward_risk`). At 1.0× the market
prices that spread as a lognormal at ATM vol does; above it the spread costs
less than flat vol says it should. That is *not* the same as the call wing's IV
sitting below ATM: a one-strike spread is a digital, and a smile falling through
its strikes adds to its price.

The row also keeps the 1M monthly's *quoted* spread (``rr_payout``,
``rr_fair_payout``) — the two quotes a reader would trade, as the Screen tab
shows them. Those are not what the history ranks: off two quotes and an expiry
that drifts from 20 to 45 days, GLD's year ranged 0.33×–4×.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from app.logic.options import chain as C
from app.logic.options import screens
from app.logic.options.rates import RateCurve
from app.logic.options.surface import build_surface, constant_maturity
from app.logic.options.tolerances import DEFAULT, OptionsTolerances

#: The metrics columns, in write order. ``rv_21`` is annualised close-to-close.
METRIC_COLUMNS = (
    "symbol", "session", "spot", "rate_30", "atm_iv_30", "call25_iv_30", "skew_30", "rv_21",
    "rr_expiration", "rr_payout", "rr_fair_payout", "reward_risk", "call25_cost_pct",
)

#: Bump when a metric's definition changes, so stored rows can be recomputed.
METRICS_VERSION = 2


def realized_vol(closes: pd.Series, sessions: int) -> float:
    """Annualised close-to-close vol over the last ``sessions`` returns; NaN if short."""
    closes = closes.dropna()
    if len(closes) < sessions + 1:
        return float("nan")
    returns = np.log(closes.iloc[-(sessions + 1):]).diff().dropna()
    return float(returns.std(ddof=1) * math.sqrt(252))


def session_metrics(
    symbol: str,
    quotes: pd.DataFrame,
    session: date,
    spot: float,
    curve: RateCurve,
    closes: pd.Series,
    tolerances: OptionsTolerances = DEFAULT,
) -> dict:
    """The day's row for one symbol. Every figure that could not be read is NaN."""
    surface = build_surface(quotes, session, spot, curve, tolerances, max_days=tolerances.max_bracket_days)
    atm = constant_maturity(surface, lambda e: e.atm_iv, tolerances)
    call25 = constant_maturity(surface, lambda e: e.call_iv_at_delta(0.25), tolerances)
    row = {
        "symbol": symbol,
        "session": pd.Timestamp(session).date(),
        "spot": spot,
        "rate_30": curve.rate(tolerances.constant_maturity_days / 365.0),
        "atm_iv_30": atm,
        "call25_iv_30": call25,
        "skew_30": call25 - atm,
        "rv_21": realized_vol(closes, tolerances.realized_vol_sessions),
        "rr_expiration": None,
        "rr_payout": float("nan"),
        "rr_fair_payout": float("nan"),
        "reward_risk": constant_maturity(
            surface, lambda e: screens.smile_reward_risk(e, 1.0, tolerances), tolerances, variance=False
        ),
        "call25_cost_pct": float("nan"),
    }
    one_month = screens.nearest_expiry(surface, 30, tolerances.one_month_days, monthly=True)
    if one_month is None:
        return row
    calls = quotes[quotes[C.EXPIRATION] == one_month.expiration]
    spread = screens.sd_spread(one_month, calls, 1.0, tolerances)
    if spread is not None:
        row.update(rr_expiration=one_month.expiration, rr_payout=spread.payout_mid, rr_fair_payout=spread.fair_payout)
    call = screens.delta_call(one_month, calls, session, 0.25, tolerances)
    if call is not None:
        row["call25_cost_pct"] = call.cost_pct
    return row


#: Where the latest reward:risk sits in its own year, by quartile. Higher is
#: better for a buyer, so the bottom quartile is the worst.
REWARD_RISK_QUARTILES = ("worst quartile", "below median", "above median", "best quartile")


def reward_risk_quartile(percentile: float) -> str:
    return REWARD_RISK_QUARTILES[min(int(percentile * 4), 3)]


@dataclass(frozen=True)
class HistoryStats:
    latest: float
    as_of: date
    percentile: float
    median: float
    q25: float
    q75: float
    low: float
    high: float
    sessions: int


def own_history_stats(
    values: pd.Series, tolerances: OptionsTolerances = DEFAULT
) -> HistoryStats | None:
    """The latest value against the trailing trading year of the same series.

    ``values`` is indexed by session. The percentile is the share of the window
    at or below the latest value, so the latest itself counts and a new high is
    100. Fewer than ``min_history_sessions`` usable points returns ``None``: a
    percentile of six weeks is not a statement about a year.
    """
    window = values.dropna().sort_index().iloc[-tolerances.history_window:]
    if len(window) < tolerances.min_history_sessions:
        return None
    latest = float(window.iloc[-1])
    return HistoryStats(
        latest=latest,
        as_of=pd.Timestamp(window.index[-1]).date(),
        percentile=float((window <= latest).mean()),
        median=float(window.median()),
        q25=float(window.quantile(0.25)),
        q75=float(window.quantile(0.75)),
        low=float(window.min()),
        high=float(window.max()),
        sessions=len(window),
    )
