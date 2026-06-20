import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tabs.momentum_analysis_models as ma_models

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
        df['RSI'] = ma_models.calculate_rsi(df)
        df['MACD'], df['Signal'], df['MACD_Hist'] = ma_models.calculate_macd(df)
        df.dropna(inplace=True) 

        # --- Live Scanners UI ---
        st.subheader("Live Indicator Scanners")
        scan_col1, scan_col2 = st.columns(2)
        
        with scan_col1:
            st.markdown("#### MACD")
            macd_mom, macd_mac = ma_models.detect_macd_crossovers(df['MACD'], df['MACD_Hist'])
            st.info(f"**Immediate Momentum:**\n\n{macd_mom}")
            st.info(f"**Macro Trend (Zero Line):**\n\n{macd_mac}")
            
        with scan_col2:
            st.markdown("#### RSI")
            rsi_trend_status = ma_models.detect_rsi_trend(df['RSI'])
            rsi_swing_status = ma_models.detect_failure_swings(df['RSI'])
            st.info(f"**Macro Trend:**\n\n{rsi_trend_status}")
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
            
            macd_state, macd_bias = ma_models.analyze_divergence(p1, p2, macd1, macd2, anchor_type)
            rsi_state, rsi_bias = ma_models.analyze_divergence(p1, p2, rsi1, rsi2, anchor_type)

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
        st.plotly_chart(fig)
