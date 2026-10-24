"""Black-76, the rate curve, and the per-expiry surface against a known smile."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from app.logic.options import chain as C
from app.logic.options.pricing import black_price, call_spot_delta, implied_vol
from app.logic.options.rates import RateCurve
from app.logic.options.surface import build_expiry, build_surface, constant_maturity, parity_forward
from tests import options_fixtures as fx


def test_implied_vol_recovers_the_vol_that_priced_it():
    strikes = np.array([300.0, 380.0, 400.0, 420.0, 560.0])
    vols = np.array([0.35, 0.24, 0.21, 0.19, 0.30])
    for is_call in (True, False):
        prices = black_price(402.0, strikes, 0.25, vols, 0.99, is_call)
        assert implied_vol(prices, 402.0, strikes, 0.25, 0.99, is_call) == pytest.approx(vols, abs=1e-6)


def test_an_unreachable_price_is_nan_not_a_bound():
    # Below intrinsic, above the ceiling, and zero: no vol reprices any of them.
    prices = np.array([5.0, 500.0, 0.0])
    out = implied_vol(prices, 410.0, np.array([400.0, 400.0, 400.0]), 0.1, 0.99, True)
    assert np.isnan(out).all()


def test_call_delta_carries_the_dividend_through_the_forward():
    # F = S·e^{(r−q)t}, so DF·F/S·N(d1) must equal e^{−qt}·N(d1).
    s, r, q, t, k, vol = 100.0, 0.04, 0.02, 0.5, 105.0, 0.25
    f, df = s * math.exp((r - q) * t), math.exp(-r * t)
    d1 = (math.log(f / k) + 0.5 * vol * vol * t) / (vol * math.sqrt(t))
    from scipy.stats import norm

    assert call_spot_delta(f, k, t, vol, df, s) == pytest.approx(math.exp(-q * t) * norm.cdf(d1))


def test_rate_curve_converts_percent_yields_to_continuous_and_holds_the_ends_flat():
    curve = RateCurve.from_yields({0.25: 4.0, 2.0: 5.0})
    assert curve.rate(0.25) == pytest.approx(math.log(1.04))
    assert curve.rate(0.01) == pytest.approx(math.log(1.04))
    assert curve.rate(10.0) == pytest.approx(math.log(1.05))
    assert curve.discount(1.0) == pytest.approx(math.exp(-curve.rate(1.0)))


def test_parity_forward_matches_the_forward_that_priced_the_chain():
    chain = fx.build_chain()
    expiration = fx.EXPIRATIONS[2]
    t = fx.year_fraction(expiration)
    quotes = chain[chain[C.EXPIRATION] == expiration]
    got = parity_forward(quotes, fx.SPOT, t, fx.curve().discount(t), 6)
    assert got == pytest.approx(fx.forward(t), rel=1e-9)


def test_parity_skips_a_strike_with_only_a_call():
    expiration, strike = fx.MISSING_PUT
    chain = fx.build_chain()
    quotes = chain[chain[C.EXPIRATION] == expiration]
    assert not ((quotes[C.STRIKE] == strike) & (quotes[C.RIGHT] == C.PUT)).any()
    t = fx.year_fraction(expiration)
    assert parity_forward(quotes, fx.SPOT, t, fx.curve().discount(t), 6) == pytest.approx(fx.forward(t), rel=1e-9)


def test_smile_is_out_of_the_money_only_and_skips_one_sided_quotes():
    chain = fx.build_chain()
    expiration = fx.EXPIRATIONS[1]
    expiry = build_expiry(chain[chain[C.EXPIRATION] == expiration], fx.SESSION, fx.SPOT, fx.curve())
    smile = expiry.smile
    assert (smile.loc[smile[C.RIGHT] == C.CALL, C.STRIKE] >= expiry.forward).all()
    assert (smile.loc[smile[C.RIGHT] == C.PUT, C.STRIKE] < expiry.forward).all()
    assert (smile[C.BID] > 0).all()
    truth = fx.smile_vol(smile[C.STRIKE].to_numpy(), expiry.t, -0.10)
    assert smile["iv"].to_numpy() == pytest.approx(truth, abs=1e-6)


def test_an_expiry_with_too_few_quotes_has_no_smile():
    chain = fx.build_chain()
    quotes = chain[chain[C.EXPIRATION] == fx.EXPIRATIONS[1]].head(6)
    assert build_expiry(quotes, fx.SESSION, fx.SPOT, fx.curve()) is None


def test_atm_iv_reads_the_smile_at_the_forward():
    chain = fx.build_chain()
    surface = build_surface(chain, fx.SESSION, fx.SPOT, fx.curve())
    for expiry in surface[:4]:
        assert expiry.atm_iv == pytest.approx(fx.atm_vol(expiry.t), abs=2e-4)


def test_constant_maturity_interpolates_total_variance_between_the_bracket():
    surface = build_surface(fx.build_chain(), fx.SESSION, fx.SPOT, fx.curve())
    got = constant_maturity(surface, lambda e: e.atm_iv)
    # The fixture's ATM variance is linear in t, so the interpolation is exact.
    assert got == pytest.approx(fx.atm_vol(30 / 365), abs=2e-4)


def test_constant_maturity_needs_an_expiry_on_each_side():
    surface = build_surface(fx.build_chain(), fx.SESSION, fx.SPOT, fx.curve())
    beyond = [e for e in surface if e.t * 365 > 30]
    assert math.isnan(constant_maturity(beyond, lambda e: e.atm_iv))


def test_25_delta_call_iv_is_read_at_constant_delta():
    surface = build_surface(fx.build_chain(), fx.SESSION, fx.SPOT, fx.curve())
    expiry = surface[2]
    t, f, df = expiry.t, expiry.forward, expiry.df

    def delta_error(k):
        return float(call_spot_delta(f, k, t, fx.smile_vol(k, t, -0.10), df, fx.SPOT)) - 0.25

    k25 = brentq(delta_error, f, 2 * f)
    assert expiry.call_iv_at_delta(0.25) == pytest.approx(float(fx.smile_vol(k25, t, -0.10)), abs=5e-4)
