"""A synthetic option chain priced from a known smile, for the 🎯 Options tests.

The repository is public and ThetaData's data is licensed, so no real chain is
committed. This one is generated instead, and that is an advantage: every
quote is the Black-76 price of a smile written down here, so a recovered
forward, IV, delta or constant-maturity vol can be checked against the formula
that made it rather than against a number someone once read off a screen.

The smile is ``σ(k, t) = σ_atm(t) + slope·k`` with ``k = ln(K/F)``, and the ATM
total variance ``σ_atm(t)²·t = A·t + B`` is **linear in t**, so the 30-day
interpolation between two expiries is exact. Mids are left unrounded so the
recovery is exact too; bid and ask straddle the mid by 2% of it (at least a
cent), which on the far wings drives the bid to zero — the one-sided quotes the
real feed is full of, produced here the same way.

Two more quirks are planted: one put near the money is missing (a strike with
only a call, which parity must skip), and the chain's rows are shuffled.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from app.logic.options import chain as C
from app.logic.options.pricing import black_price
from app.logic.options.rates import RateCurve

SESSION = date(2026, 10, 23)  # a Friday
SPOT = 400.0
YIELD_PCT = 4.0
DIVIDEND = 0.01
ATM_A, ATM_B = 0.04, 0.0004
EXPIRATIONS = (
    date(2026, 10, 30), date(2026, 11, 20), date(2026, 11, 23), date(2026, 12, 18), date(2027, 1, 15),
    date(2027, 3, 31), date(2027, 9, 30), date(2028, 12, 15), date(2029, 1, 19),
)
MISSING_PUT = (date(2026, 11, 20), 405.0)


def curve() -> RateCurve:
    return RateCurve.from_yields({1 / 12: YIELD_PCT, 1.0: YIELD_PCT, 5.0: YIELD_PCT})


def year_fraction(expiration: date) -> float:
    return (expiration - SESSION).days / 365.0


def forward(t: float) -> float:
    return SPOT * math.exp((curve().rate(t) - DIVIDEND) * t)


def atm_vol(t: float) -> float:
    return math.sqrt(ATM_A + ATM_B / t)


def smile_vol(strike, t: float, slope: float):
    return atm_vol(t) + slope * np.log(np.asarray(strike) / forward(t))


def strikes() -> np.ndarray:
    return np.unique(np.concatenate([np.arange(200, 300, 5.0), np.arange(300, 481, 1.0), np.arange(485, 641, 5.0)]))


def build_chain(slope: float = -0.10, symbol: str = "TEST", seed: int = 7) -> pd.DataFrame:
    """The full chain; ``slope`` < 0 is the usual equity smile (calls below ATM)."""
    rows = []
    ks = strikes()
    for expiration in EXPIRATIONS:
        t = year_fraction(expiration)
        fwd, df = forward(t), curve().discount(t)
        vols = smile_vol(ks, t, slope)
        for right, is_call in ((C.CALL, True), (C.PUT, False)):
            prices = black_price(fwd, ks, t, vols, df, is_call)
            half = np.maximum(0.01, 0.02 * prices)
            for strike, price, h in zip(ks, prices, half):
                if right == C.PUT and (expiration, strike) == MISSING_PUT:
                    continue
                bid = price - h
                rows.append({
                    C.SYMBOL: symbol, C.SESSION: SESSION, C.EXPIRATION: expiration, C.STRIKE: float(strike),
                    C.RIGHT: right, C.BID: max(bid, 0.0), C.ASK: price + h, C.CLOSE: price, C.VOLUME: 10,
                    C.BID_SIZE: 5 if bid > 0 else 0, C.ASK_SIZE: 5,
                })
    frame = pd.DataFrame(rows, columns=list(C.COLUMNS))
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
