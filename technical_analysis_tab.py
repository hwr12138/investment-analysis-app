import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# --- Main Dashboard Render ---

def render_tab():
    st.header("Momentum Analysis")
    
    # Inputs: Ticker and Timeframe
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
    with col2:
        period_map = {"5D": "5d", "1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "5Y": "5y"}
        period_selection = st.selectbox("Select Time Period", list(period_map.keys()), index=4)
        
    # Fetch Data
    if ticker:
        df = yf.download(ticker, period=period_map[period_selection], multi_level_index=False)
        
        if df.empty:
            st.error("No data found for this ticker.")
            return
            
        # Calculate Indicators
        df['RSI'] = calculate_rsi(df)
        df['MACD'], df['Signal'], df['MACD_Hist'] = calculate_macd(df)
        df.dropna(inplace=True) 

        # --- Advanced RSI Analysis Output ---
        st.subheader("Live Momentum Scanner (RSI)")
        rsi_trend_status = detect_rsi_trend(df['RSI'])
        rsi_swing_status = detect_failure_swings(df['RSI'])
        
        rsi_col1, rsi_col2 = st.columns(2)
        with rsi_col1:
            st.info(f"**Macro Trend:**\n\n{rsi_trend_status}")
        with rsi_col2:
            st.info(f"**Failure Swings (Last 25 Periods):**\n\n{rsi_swing_status}")
            
        st.divider()

        # Anchor Date Inputs
        st.subheader("Manual Divergence Analysis")
        c1, c2, c3 = st.columns(3)
        with c1:
            date1 = st.date_input("Anchor Date 1", value=df.index[-20])
        with c2:
            date2 = st.date_input("Anchor Date 2", value=df.index[-1])
        with c3:
            anchor_type = st.radio("What are these points?", ["Lows", "Highs"], horizontal=True, help="Connecting swing lows looks for Bullish setups. Connecting highs looks for Bearish setups.")

        d1 = pd.to_datetime(date1).tz_localize(df.index.tz)
        d2 = pd.to_datetime(date2).tz_localize(df.index.tz)

        # Plotting
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            row_heights=[0.5, 0.25, 0.25], 
                            vertical_spacing=0.05)

        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                     low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1.5), name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], line=dict(color='orange', width=1.5), name='Signal'), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color='gray', name='Histogram'), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1.5), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Draw Divergence Lines
        try:
            p1 = df.loc[df.index.asof(d1)]['Close'].values[0] if isinstance(df.loc[df.index.asof(d1)]['Close'], pd.Series) else df.loc[df.index.asof(d1)]['Close']
            p2 = df.loc[df.index.asof(d2)]['Close'].values[0] if isinstance(df.loc[df.index.asof(d2)]['Close'], pd.Series) else df.loc[df.index.asof(d2)]['Close']
            
            macd1, macd2 = df.loc[df.index.asof(d1)]['MACD'], df.loc[df.index.asof(d2)]['MACD']
            rsi1, rsi2 = df.loc[df.index.asof(d1)]['RSI'], df.loc[df.index.asof(d2)]['RSI']

            fig.add_trace(go.Scatter(x=[d1, d2], y=[p1, p2], mode='lines+markers', line=dict(color='yellow', width=2, dash='dot'), name='Price Trend'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[d1, d2], y=[macd1, macd2], mode='lines+markers', line=dict(color='yellow', width=2, dash='dot'), showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=[d1, d2], y=[rsi1, rsi2], mode='lines+markers', line=dict(color='yellow', width=2, dash='dot'), showlegend=False), row=3, col=1)
            
            macd_state, macd_bias = analyze_divergence(p1, p2, macd1, macd2, anchor_type)
            rsi_state, rsi_bias = analyze_divergence(p1, p2, rsi1, rsi2, anchor_type)

            st.markdown("### Analysis Results")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.success(f"**MACD State:** {macd_state}\n\n**Bias:** {macd_bias}")
            with res_col2:
                st.success(f"**RSI State:** {rsi_state}\n\n**Bias:** {rsi_bias}")

        except Exception as e:
            st.warning("Ensure both anchor dates fall within the selected time period.")

        # Update layout
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template='plotly_dark', margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
