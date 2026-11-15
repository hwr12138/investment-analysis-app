from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

#: Lookback windows, in calendar days. Sharadar is queried by date range rather
#: than by a named period, and MACD needs ~34 sessions of run-up before its
#: first valid value, so the shortest windows deliberately over-fetch and the
#: chart is trimmed afterwards.
PERIOD_DAYS = {"5D": 7, "1M": 31, "3M": 92, "6M": 183, "1Y": 365, "5Y": 1826}

#: Extra calendar days fetched purely to warm up the indicators.
INDICATOR_WARMUP_DAYS = 90


# --- Preparing a window ---

def indicator_frame(ohlcv: pd.DataFrame, window_start: date) -> pd.DataFrame:
    """OHLCV plus RSI and MACD, computed over the whole fetch, then trimmed to the window.

    Computing before trimming is the point: the first bar kept already has a
    settled RSI and MACD rather than a warm-up artefact. Rows still warming up
    are dropped, never filled.
    """
    frame = ohlcv.copy()
    frame["RSI"] = calculate_rsi(frame)
    frame["MACD"], frame["Signal"], frame["MACD_Hist"] = calculate_macd(frame)
    return frame.dropna().loc[pd.Timestamp(window_start):]


@dataclass(frozen=True)
class AnchorPoint:
    """The close and both indicators at one session, for a divergence line."""

    session: pd.Timestamp
    close: float
    macd: float
    rsi: float


def anchor_point(frame: pd.DataFrame, day: date) -> AnchorPoint | None:
    """The last session on or before ``day``; ``None`` when ``day`` precedes the frame."""
    stamp = pd.Timestamp(day)
    if frame.index.tz is not None:
        stamp = stamp.tz_localize(frame.index.tz)
    session = frame.index.asof(stamp)
    if pd.isna(session):
        return None
    row = frame.loc[session]
    if isinstance(row, pd.DataFrame):  # a duplicated session: the later row wins
        row = row.iloc[-1]
    return AnchorPoint(session, float(row["Close"]), float(row["MACD"]), float(row["RSI"]))


# --- Core Indicator Math ---

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

# --- Advanced RSI Scanners ---

def detect_macd_crossovers(macd_series, hist_series):
    """Detects recent MACD crossovers and current momentum states."""
    if len(hist_series) < 2:
        return "Insufficient Data", "Insufficient Data"
    
    current_macd = macd_series.iloc[-1]
    current_hist = hist_series.iloc[-1]
    prev_hist = hist_series.iloc[-2]
    prev_macd = macd_series.iloc[-2]

    # 1. Immediate Momentum (Signal Crossover via Histogram)
    if prev_hist <= 0 and current_hist > 0:
        momentum_signal = "🟢 Bullish Crossover Just Occurred!"
    elif prev_hist >= 0 and current_hist < 0:
        momentum_signal = "🔴 Bearish Crossover Just Occurred!"
    elif current_hist > 0:
        momentum_signal = "📈 Bullish Momentum (MACD > Signal)"
    else:
        momentum_signal = "📉 Bearish Momentum (MACD < Signal)"

    # 2. Macro Trend (Zero Line Crossover via MACD Line)
    if prev_macd <= 0 and current_macd > 0:
        macro_signal = "🟢 Bullish Zero Line Cross Just Occurred!"
    elif prev_macd >= 0 and current_macd < 0:
        macro_signal = "🔴 Bearish Zero Line Cross Just Occurred!"
    elif current_macd > 0:
        macro_signal = "📈 Bullish Macro Trend (MACD > 0)"
    else:
        macro_signal = "📉 Bearish Macro Trend (MACD < 0)"

    return momentum_signal, macro_signal

def detect_rsi_trend(rsi_series, lookback=60):
    """Classifies the overarching market trend based on RSI bounding boxes."""
    if len(rsi_series) < lookback:
        return "Insufficient Data"
    
    recent_rsi = rsi_series.tail(lookback)
    rsi_max = recent_rsi.max()
    rsi_min = recent_rsi.min()
    
    # Heuristic for Bull/Bear ranges
    if rsi_min >= 35 and rsi_max >= 65:
        return "📈 Confirmed Bull Trend (40-90 Range)"
    elif rsi_max <= 65 and rsi_min <= 35:
        return "📉 Confirmed Bear Trend (10-60 Range)"
    else:
        return "⚖️ Ranging / Shifting Macro Trend"

def detect_failure_swings(rsi_series, lookback=25):
    """Scans the recent RSI array for the 4-point Failure Swing reversal pattern."""
    if len(rsi_series) < lookback:
        return "Insufficient Data"
        
    data = rsi_series.tail(lookback).values
    
    troughs = []
    peaks = []
    
    # Find local minima and maxima
    for i in range(1, len(data)-1):
        if data[i] < data[i-1] and data[i] < data[i+1]:
            troughs.append((i, data[i]))
        if data[i] > data[i-1] and data[i] > data[i+1]:
            peaks.append((i, data[i]))
            
    current_rsi = data[-1]
            
    # Check for Bullish Failure Swing
    if len(troughs) >= 2 and len(peaks) >= 1:
        t2_idx, t2_val = troughs[-1]
        t1_idx, t1_val = troughs[-2]
        # Find peak between the two troughs
        peaks_between = [p for p in peaks if t1_idx < p[0] < t2_idx]
        
        if peaks_between:
            p1_val = peaks_between[-1][1]
            if t1_val < 30 and t2_val > t1_val and current_rsi > p1_val:
                return "🟢 Bullish Failure Swing Active (Buy Signal)"
                
    # Check for Bearish Failure Swing
    if len(peaks) >= 2 and len(troughs) >= 1:
        p2_idx, p2_val = peaks[-1]
        p1_idx, p1_val = peaks[-2]
        # Find trough between the two peaks
        troughs_between = [t for t in troughs if p1_idx < t[0] < p2_idx]
        
        if troughs_between:
            t1_val = troughs_between[-1][1]
            if p1_val > 70 and p2_val < p1_val and current_rsi < t1_val:
                return "🔴 Bearish Failure Swing Active (Sell Signal)"
                
    return "⚪ No Swings Detected"

def analyze_divergence(p1, p2, i1, i2, points_type):
    """
    Evaluates the slope between two points to classify divergence/convergence.
    points_type: "Lows" or "Highs"
    """
    price_delta = p2 - p1
    ind_delta = i2 - i1
    
    if points_type == "Lows":
        if price_delta > 0 and ind_delta > 0:
            return "Bullish Convergence", "Strong Uptrend Confirmed"
        elif price_delta < 0 and ind_delta < 0:
            return "Bearish Convergence", "Strong Downtrend Confirmed"
        elif price_delta < 0 and ind_delta > 0:
            return "Regular Bullish Divergence", "Potential Bottom / Reversal to Upside"
        elif price_delta > 0 and ind_delta < 0:
            return "Hidden Bullish Divergence", "Trend Continuation (Buy the Dip)"
            
    elif points_type == "Highs":
        if price_delta > 0 and ind_delta > 0:
            return "Bullish Convergence", "Strong Uptrend Confirmed"
        elif price_delta < 0 and ind_delta < 0:
            return "Bearish Convergence", "Strong Downtrend Confirmed"
        elif price_delta > 0 and ind_delta < 0:
            return "Regular Bearish Divergence", "Potential Top / Reversal to Downside"
        elif price_delta < 0 and ind_delta > 0:
            return "Hidden Bearish Divergence", "Trend Continuation (Sell the Rally)"
            
    return "Neutral / Unclear", "No clear signal"