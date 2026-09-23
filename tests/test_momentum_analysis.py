"""Momentum indicators and the signals read off them, on series built to be obvious."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.logic import momentum_analysis as ma


def _closes(values) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": values}, index=pd.bdate_range("2024-01-01", periods=len(values))
    )


def test_a_price_that_only_rises_has_an_rsi_of_100():
    rsi = ma.calculate_rsi(_closes(np.arange(1.0, 41.0)))
    # The first day has no change, which the gain/loss split reads as zero, so
    # the first full 14-day window closes on row 13.
    assert rsi.iloc[:13].isna().all()
    assert rsi.iloc[13:].eq(100.0).all()


def test_a_price_that_only_falls_has_an_rsi_of_0():
    rsi = ma.calculate_rsi(_closes(np.arange(40.0, 0.0, -1.0)))
    assert rsi.iloc[13:].eq(0.0).all()


def test_a_flat_price_has_no_macd():
    macd, signal, histogram = ma.calculate_macd(_closes([50.0] * 60))
    assert (macd == 0).all() and (signal == 0).all() and (histogram == 0).all()


def test_a_histogram_turning_positive_is_a_bullish_crossover():
    momentum, macro = ma.detect_macd_crossovers(pd.Series([1.0, 1.0]), pd.Series([-0.5, 0.5]))
    assert "Bullish Crossover Just Occurred" in momentum
    assert "Bullish Macro Trend" in macro


def test_a_macd_line_turning_negative_is_a_bearish_zero_cross():
    _, macro = ma.detect_macd_crossovers(pd.Series([0.5, -0.5]), pd.Series([-1.0, -1.0]))
    assert "Bearish Zero Line Cross Just Occurred" in macro


@pytest.mark.parametrize(
    "scan", [ma.detect_rsi_trend, ma.detect_failure_swings], ids=["trend", "swings"]
)
def test_too_short_a_history_says_so(scan):
    assert scan(pd.Series([50.0] * 5)) == "Insufficient Data"


def test_a_steady_rsi_has_no_failure_swing():
    assert "No Swings Detected" in ma.detect_failure_swings(pd.Series(np.linspace(40, 60, 30)))


@pytest.mark.parametrize(
    "prices, indicator, points, expected",
    [
        ((10, 9), (30, 40), "Lows", "Regular Bullish Divergence"),
        ((10, 11), (40, 30), "Lows", "Hidden Bullish Divergence"),
        ((10, 11), (70, 60), "Highs", "Regular Bearish Divergence"),
        ((10, 9), (60, 70), "Highs", "Hidden Bearish Divergence"),
        ((10, 11), (60, 70), "Highs", "Bullish Convergence"),
        ((10, 10), (60, 70), "Highs", "Neutral / Unclear"),
    ],
)
def test_divergence_is_read_off_the_two_slopes(prices, indicator, points, expected):
    state, _ = ma.analyze_divergence(*prices, *indicator, points)
    assert state == expected
