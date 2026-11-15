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


# ---------------------------------------------------------------------------
# Preparing a window, as the API does
# ---------------------------------------------------------------------------


def _ohlcv(periods=120) -> pd.DataFrame:
    close = 100 + np.sin(np.arange(periods) / 7.0) * 5
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close},
        index=pd.bdate_range("2024-01-01", periods=periods),
    )


def test_the_indicators_are_settled_on_the_first_bar_kept():
    full = _ohlcv()
    frame = ma.indicator_frame(full, pd.Timestamp("2024-01-01").date())
    # RSI's 14-day window is the slowest to settle (row 13, as above); nothing
    # before it survives.
    assert frame.index[0] == full.index[13]
    assert not frame.isna().any().any()


def test_trimming_happens_after_the_indicators_not_before():
    full = _ohlcv()
    start = full.index[60].date()
    trimmed = ma.indicator_frame(full, start)
    assert trimmed.index[0] == full.index[60]
    # Computed over the run-up, so the first kept RSI matches the full series.
    assert trimmed["RSI"].iloc[0] == ma.calculate_rsi(full).iloc[60]


def test_an_anchor_snaps_back_to_the_last_session_on_or_before_it():
    frame = ma.indicator_frame(_ohlcv(), pd.Timestamp("2024-01-01").date())
    saturday = pd.Timestamp("2024-03-02")
    point = ma.anchor_point(frame, saturday.date())
    assert point.session == pd.Timestamp("2024-03-01")
    assert point.close == frame.loc["2024-03-01", "Close"]


def test_an_anchor_before_the_frame_is_none_rather_than_moved():
    frame = ma.indicator_frame(_ohlcv(), pd.Timestamp("2024-02-01").date())
    assert ma.anchor_point(frame, pd.Timestamp("2024-01-15").date()) is None
