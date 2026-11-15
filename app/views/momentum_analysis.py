import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
from plotly.subplots import make_subplots

from app.data import sharadar_client
from app.logic import momentum_analysis as ma_models

PERIOD_DAYS = ma_models.PERIOD_DAYS

# --- Main Dashboard Render ---

def render_page():
    st.header("Momentum Analysis")

    # Inputs: Ticker and Timeframe
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
    with col2:
        period_selection = st.selectbox("Select Time Period", list(PERIOD_DAYS.keys()), index=4)

    # Fetch Data
    if ticker:
        window_days = PERIOD_DAYS[period_selection]
        end = date.today()
        window_start = end - timedelta(days=window_days)
        try:
            df = sharadar_client.fetch_ohlcv(
                ticker,
                (window_start - timedelta(days=ma_models.INDICATOR_WARMUP_DAYS)).isoformat(),
                end.isoformat(),
            )
        except sharadar_client.SharadarError as error:
            st.error(str(error))
            return

        if df.empty:
            st.error("No data found for this ticker.")
            return

        df = ma_models.indicator_frame(df, window_start)

        if df.empty:
            st.error("Not enough price history for this ticker to compute indicators.")
            return

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
            # Short windows can hold fewer than 20 sessions, so clamp rather
            # than index off the front of the frame.
            date1 = st.date_input("Anchor Date 1", value=df.index[-min(20, len(df))])
        with c2:
            date2 = st.date_input("Anchor Date 2", value=df.index[-1])
        with c3:
            anchor_type = st.radio("What are these points?", ["Lows", "Highs"], horizontal=True, help="Connecting swing lows looks for Bullish setups. Connecting highs looks for Bearish setups.")

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
        a1, a2 = ma_models.anchor_point(df, date1), ma_models.anchor_point(df, date2)
        if a1 is None or a2 is None:
            st.warning("Ensure both anchor dates fall within the selected time period.")
        else:
            xs = [a1.session, a2.session]
            fig.add_trace(go.Scatter(x=xs, y=[a1.close, a2.close], mode='lines+markers', line=dict(color='yellow', width=2, dash='dot'), name='Price Trend'), row=1, col=1)
            fig.add_trace(go.Scatter(x=xs, y=[a1.macd, a2.macd], mode='lines+markers', line=dict(color='yellow', width=2, dash='dot'), showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=xs, y=[a1.rsi, a2.rsi], mode='lines+markers', line=dict(color='yellow', width=2, dash='dot'), showlegend=False), row=3, col=1)

            macd_state, macd_bias = ma_models.analyze_divergence(a1.close, a2.close, a1.macd, a2.macd, anchor_type)
            rsi_state, rsi_bias = ma_models.analyze_divergence(a1.close, a2.close, a1.rsi, a2.rsi, anchor_type)

            st.markdown("### Analysis Results")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.success(f"**MACD State:** {macd_state}\n\n**Bias:** {macd_bias}")
            with res_col2:
                st.success(f"**RSI State:** {rsi_state}\n\n**Bias:** {rsi_bias}")

        # Update layout
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, template='plotly_dark', margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig)
