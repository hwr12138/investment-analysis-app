"""Black-76 on the forward — price, implied vol, delta.

Pricing on the forward rather than on spot with a dividend yield is what lets
the page ignore dividends entirely: :func:`app.logic.options.surface.parity_forward`
reads the forward off each expiry's own call–put pairs, so the market's view of
dividends, borrow and rates is already in it. US equity options are American,
but every quote this page reads is out of the money, where the early-exercise
premium is negligible.

Everything here is vectorised over numpy arrays, because the collector solves a
few million vols in a backfill and a scalar root-finder per option was the
slowest part of the prototype.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

#: The implied-vol search interval. A solved vol at either end means the price
#: was not reachable — reported as NaN, not as the bound.
_VOL_LO, _VOL_HI = 1e-4, 5.0
_BISECTIONS = 60


def d1(forward, strike, t, vol):
    forward, strike, t, vol = map(np.asarray, (forward, strike, t, vol))
    return (np.log(forward / strike) + 0.5 * vol * vol * t) / (vol * np.sqrt(t))


def black_price(forward, strike, t, vol, df, is_call):
    """Discounted Black-76 price; ``is_call`` may be an array of booleans."""
    forward, strike, t, vol, df = map(np.asarray, (forward, strike, t, vol, df))
    one = d1(forward, strike, t, vol)
    two = one - vol * np.sqrt(t)
    call = df * (forward * norm.cdf(one) - strike * norm.cdf(two))
    put = df * (strike * norm.cdf(-two) - forward * norm.cdf(-one))
    return np.where(is_call, call, put)


def implied_vol(price, forward, strike, t, df, is_call):
    """The vol that reprices each option, or NaN where no vol does.

    Bisection rather than Newton: vega vanishes on the far wings this page
    reads (a +41% call a month out), where Newton steps leave the interval. Sixty
    halvings of [0.0001, 5] resolve the vol to well under a hundredth of a point.
    """
    price, forward, strike, t, df = np.broadcast_arrays(*map(np.asarray, (price, forward, strike, t, df)))
    is_call = np.broadcast_to(np.asarray(is_call), price.shape)
    price = price.astype(float)
    intrinsic = np.where(is_call, df * np.maximum(forward - strike, 0), df * np.maximum(strike - forward, 0))
    ceiling = np.where(is_call, df * forward, df * strike)
    solvable = (price > intrinsic) & (price < ceiling) & (t > 0) & np.isfinite(price)

    lo = np.full(price.shape, _VOL_LO)
    hi = np.full(price.shape, _VOL_HI)
    for _ in range(_BISECTIONS):
        mid = 0.5 * (lo + hi)
        too_cheap = black_price(forward, strike, t, mid, df, is_call) < price
        lo = np.where(too_cheap, mid, lo)
        hi = np.where(too_cheap, hi, mid)
    vol = 0.5 * (lo + hi)
    at_bound = (vol < _VOL_LO * 1.01) | (vol > _VOL_HI * 0.99)
    return np.where(solvable & ~at_bound, vol, np.nan)


def call_spot_delta(forward, strike, t, vol, df, spot):
    """∂C/∂S for a call priced on the forward: ``DF · F/S · N(d1)``.

    With F = S·e^{(r−q)t} this is the familiar ``e^{−qt}·N(d1)``, without having
    to know q.
    """
    return np.asarray(df) * np.asarray(forward) / np.asarray(spot) * norm.cdf(d1(forward, strike, t, vol))
