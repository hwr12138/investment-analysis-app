"""The 25Δ call, the replacement grid and the +k SD spread on the synthetic chain."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from app.logic.options import chain as C
from app.logic.options import screens
from app.logic.options.pricing import black_price, call_spot_delta
from app.logic.options.surface import build_surface
from tests import options_fixtures as fx


def _surface(slope=-0.10):
    chain = fx.build_chain(slope=slope)
    return chain, build_surface(chain, fx.SESSION, fx.SPOT, fx.curve())


def _calls(chain, expiration):
    return chain[chain[C.EXPIRATION] == expiration]


def test_trading_days_counts_weekdays_to_expiry():
    assert screens.trading_days(date(2026, 10, 23), date(2026, 11, 20)) == 20


def test_delta_call_picks_the_listed_strike_nearest_25_delta():
    chain, surface = _surface()
    expiry = screens.nearest_expiry(surface, 30, (20, 45), monthly=True)
    assert expiry.expiration == date(2026, 11, 20)
    pick = screens.delta_call(expiry, _calls(chain, expiry.expiration), fx.SESSION)

    ks = fx.strikes()
    ks = ks[ks > expiry.forward]
    true_delta = call_spot_delta(expiry.forward, ks, expiry.t, fx.smile_vol(ks, expiry.t, -0.10), expiry.df, fx.SPOT)
    assert pick.strike == ks[np.argmin(np.abs(true_delta - 0.25))]
    assert pick.delta == pytest.approx(0.25, abs=0.02)
    assert pick.cost_pct == pytest.approx(pick.cost / fx.SPOT)
    assert pick.otm == pytest.approx(pick.strike / fx.SPOT - 1)
    assert pick.annualised == pytest.approx(pick.cost_pct * 252 / 20)


def test_grid_columns_are_the_nearest_expiry_per_tenor_then_the_two_longest():
    _, surface = _surface()
    labels = [(label, e.expiration) for label, e in screens.grid_expiries(surface)]
    assert labels == [
        ("1M", date(2026, 11, 20)), ("3M", date(2027, 1, 15)), ("6M", date(2027, 3, 31)),
        ("12M", date(2027, 9, 30)), ("DEC28", date(2028, 12, 15)), ("JAN29", date(2029, 1, 19)),
    ]


def test_grid_cell_is_the_call_premium_at_spot_times_one_plus_upside():
    chain, surface = _surface()
    columns = screens.grid_expiries(surface)
    by_expiry = {e: g for e, g in chain.groupby(C.EXPIRATION)}
    grid = screens.replacement_grid(by_expiry, columns, fx.SESSION, fx.SPOT)
    expiry = dict(columns)["3M"]
    k = fx.SPOT * 1.04
    truth = float(black_price(expiry.forward, k, expiry.t, fx.smile_vol(k, expiry.t, -0.10), expiry.df, True))
    # Linear between $1 strikes on a smooth price curve: within a few hundredths of a point.
    assert grid[0.04]["3M"].cost_pct == pytest.approx(truth / fx.SPOT, abs=2e-4)


def test_grid_spread_mode_is_long_minus_short_ten_points_higher():
    chain, surface = _surface()
    columns = screens.grid_expiries(surface)
    by_expiry = {e: g for e, g in chain.groupby(C.EXPIRATION)}
    single = screens.replacement_grid(by_expiry, columns, fx.SESSION, fx.SPOT)
    spread = screens.replacement_grid(by_expiry, columns, fx.SESSION, fx.SPOT, spread=True)
    assert spread[0.04]["6M"].cost_pct == pytest.approx(single[0.04]["6M"].cost_pct - single[0.14]["6M"].cost_pct)


def test_a_grid_strike_beyond_the_quoted_range_is_missing_not_extrapolated():
    chain, surface = _surface()
    columns = screens.grid_expiries(surface)
    by_expiry = {e: g for e, g in chain.groupby(C.EXPIRATION)}
    grid = screens.replacement_grid(by_expiry, columns, fx.SESSION, fx.SPOT, upsides=(0.80,))
    assert all(cell.cost_pct is None for cell in grid[0.80].values())


def test_far_wing_cells_with_one_sided_neighbours_are_flagged_wide():
    chain, surface = _surface()
    columns = screens.grid_expiries(surface)
    by_expiry = {e: g for e, g in chain.groupby(C.EXPIRATION)}
    grid = screens.replacement_grid(by_expiry, columns, fx.SESSION, fx.SPOT)
    assert grid[0.41]["1M"].cost_pct is None or grid[0.41]["1M"].wide


def test_sd_spread_target_is_lognormal_not_linear():
    chain, surface = _surface()
    expiry = next(e for e in surface if e.expiration == date(2026, 12, 18))
    spread = screens.sd_spread(expiry, _calls(chain, expiry.expiration), 1.0)
    sigma_root_t = expiry.atm_iv * math.sqrt(expiry.t)
    lognormal, linear = fx.SPOT * math.exp(sigma_root_t), fx.SPOT * (1 + sigma_root_t)
    assert math.floor(lognormal) != math.floor(linear)  # the fixture separates the two
    assert spread.short_strike == math.floor(lognormal)
    assert spread.long_strike == spread.short_strike - 1
    assert spread.move == pytest.approx(math.exp(sigma_root_t) - 1)


def test_sd_spread_prices_and_payouts():
    chain, surface = _surface()
    expiry = next(e for e in surface if e.expiration == date(2026, 12, 18))
    calls = _calls(chain, expiry.expiration)
    s = screens.sd_spread(expiry, calls, 1.0)
    legs = calls[calls[C.RIGHT] == C.CALL].set_index(C.STRIKE)
    long_leg, short_leg = legs.loc[s.long_strike], legs.loc[s.short_strike]
    assert s.price == pytest.approx((long_leg[C.BID] + long_leg[C.ASK]) / 2 - (short_leg[C.BID] + short_leg[C.ASK]) / 2)
    assert s.bid == pytest.approx(max(0.0, long_leg[C.BID] - short_leg[C.ASK]))
    assert s.ask == pytest.approx(long_leg[C.ASK] - short_leg[C.BID])
    assert s.payout_mid == pytest.approx((s.width - s.price) / s.price)
    assert s.payout_ask == pytest.approx((s.width - s.ask) / s.ask)
    assert s.payout_ask < s.payout_mid


def _spread_at(slope):
    chain, surface = _surface(slope)
    expiry = next(e for e in surface if e.expiration == date(2026, 12, 18))
    return screens.sd_spread(expiry, _calls(chain, expiry.expiration), 1.0)


def test_reward_risk_is_one_on_a_flat_smile():
    assert _spread_at(0.0).reward_risk == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("slope", [-0.30, -0.10, 0.10, 0.30])
def test_reward_risk_is_above_one_exactly_when_the_spread_costs_less_than_at_flat_vol(slope):
    # Not the same as "the call wing's vol is below ATM": a narrow spread is a
    # digital, and a smile falling through the strikes *adds* to its price.
    s = _spread_at(slope)
    assert (s.reward_risk > 1.0) == (s.price < s.fair_price)


def test_the_one_month_point_is_the_monthly_even_when_a_weekly_is_nearer_30_days():
    _, surface = _surface()
    # 11/23 (a Monday weekly) is 31 days out, 11/20 (the monthly) 28.
    assert screens.nearest_expiry(surface, 30, (20, 45)).expiration == date(2026, 11, 23)
    assert screens.nearest_expiry(surface, 30, (20, 45), monthly=True).expiration == date(2026, 11, 20)


def test_a_wide_quote_is_never_a_spread_leg():
    chain, surface = _surface()
    expiry = next(e for e in surface if e.expiration == date(2026, 12, 18))
    first = screens.sd_spread(expiry, _calls(chain, expiry.expiration), 1.0)
    calls = _calls(chain, expiry.expiration).copy()
    stale = (calls[C.RIGHT] == C.CALL) & (calls[C.STRIKE] == first.long_strike)
    calls.loc[stale, C.BID] = calls.loc[stale, C.BID] * 0.5  # a stale, wide quote
    second = screens.sd_spread(expiry, calls, 1.0)
    assert second.short_strike == first.short_strike
    assert second.long_strike == first.long_strike - 1


def test_smile_reward_risk_is_one_on_a_flat_smile():
    _, surface = _surface(0.0)
    for expiry in surface[1:5]:
        assert screens.smile_reward_risk(expiry) == pytest.approx(1.0, abs=1e-6)


def test_smile_reward_risk_matches_the_digital_of_the_known_smile():
    from scipy.stats import norm

    _, surface = _surface(-0.10)
    expiry = next(e for e in surface if e.expiration == date(2026, 12, 18))
    t, f, df = expiry.t, expiry.forward, expiry.df
    k = fx.SPOT * math.exp(expiry.atm_iv * math.sqrt(t))
    vol = float(fx.smile_vol(k, t, -0.10))
    dvol = -0.10 / k  # σ = σ_atm + slope·ln(K/F)
    d1 = (math.log(f / k) + 0.5 * vol * vol * t) / (vol * math.sqrt(t))
    market = df * (norm.cdf(d1 - vol * math.sqrt(t)) - f * norm.pdf(d1) * math.sqrt(t) * dvol)
    d1f = (math.log(f / k) + 0.5 * expiry.atm_iv ** 2 * t) / (expiry.atm_iv * math.sqrt(t))
    flat = df * norm.cdf(d1f - expiry.atm_iv * math.sqrt(t))
    expected = ((1 - market) / market) / ((1 - flat) / flat)
    assert screens.smile_reward_risk(expiry) == pytest.approx(expected, rel=2e-3)


def test_smile_reward_risk_is_unread_beyond_the_quoted_wing():
    _, surface = _surface()
    assert math.isnan(screens.smile_reward_risk(surface[1], sd=12.0))
