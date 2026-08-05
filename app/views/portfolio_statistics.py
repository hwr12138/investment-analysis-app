import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date

from app.data import market_data
from app.logic import portfolio_statistics as ps_logic

def render_page():
    st.header("Portfolio Statistics")
    st.write("Calculate comprehensive risk and return metrics for a custom portfolio and view asset correlations.")

    if "default_rf_rate" not in st.session_state:
        st.session_state.default_rf_rate = market_data.fetch_current_risk_free_rate() * 100

    st.markdown("### ⚙️ Portfolio Parameters")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            allocations_input = st.text_input(
                "Enter Tickers and Allocations",
                value="QQQ:55, GLD:30, XLE:15",
                help="Format as TICKER:PERCENTAGE, separated by commas."
            )
            rf_input = st.number_input(
                "Risk-Free Rate (%)",
                value=st.session_state.default_rf_rate,
                step=0.1,
                key="ps_rf_input",
                help="Defaults to the implied rate of the 30-Day Fed Funds Futures (ZQ=F)."
            )
        with col2:
            start_date = st.date_input("Custom Start Date", value=date(2020, 1, 1), min_value=date(1900, 1, 1), key="ps_start")
            end_date = st.date_input("Custom End Date", value=date.today(), min_value=date(1900, 1, 1), key="ps_end")

    run_button = st.button("Calculate Statistics", type="primary", width='stretch')

    if run_button:
        if start_date >= end_date:
            st.error("Please ensure your Start Date is before your End Date.")
            return

        # Parse allocations
        allocations = {}
        try:
            parts = [p.strip() for p in allocations_input.split(',')]
            for part in parts:
                if not part: continue
                ticker, alloc = part.split(':')
                allocations[ticker.strip().upper()] = float(alloc.strip())
        except Exception:
            st.error("Invalid format for allocations. Please use TICKER:PERCENTAGE (e.g. QQQ:55, GLD:30).")
            return

        tickers_list = list(allocations.keys())
        if len(tickers_list) < 1:
            st.error("Please enter at least one valid ticker.")
            return

        total_alloc = sum(allocations.values())
        if abs(total_alloc - 100.0) > 0.01 and abs(total_alloc - 1.0) > 0.001:
            st.error(f"Allocations sum to {total_alloc}, not 100. They have been automatically normalized.")

        rf_rate_decimal = rf_input / 100.0
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # We need historical data stretching back 10 years for the 10-year metrics.
        try:
            earliest_start = min(start_date, end_date.replace(year=end_date.year - 10))
        except ValueError: # handle leap years
            earliest_start = min(start_date, end_date.replace(year=end_date.year - 10, day=28))

        earliest_start_str = earliest_start.strftime('%Y-%m-%d')

        with st.spinner(f"Fetching historical data and computing metrics..."):
            data = ps_logic.fetch_data_and_market(tickers_list, earliest_start_str, end_str)

        if data.empty or not any(t in data.columns for t in tickers_list):
            st.error("Failed to download enough valid data. Please check your tickers.")
            return

        # Filter to valid tickers
        valid_tickers = [t for t in tickers_list if t in data.columns]
        missing = set(tickers_list) - set(valid_tickers)
        if missing:
            st.warning(f"Could not find data for: {', '.join(missing)}")
            allocations = {k: v for k, v in allocations.items() if k in valid_tickers}

        st.divider()
        st.subheader("Asset Relationships (Custom Period)")

        custom_data = data[valid_tickers].loc[start_str:end_str].pct_change().dropna()
        if not custom_data.empty and len(valid_tickers) > 1:
            corr_matrix = custom_data.corr()
            cov_matrix = custom_data.cov() * 251 # Annualized covariance

            col_corr, col_cov = st.columns(2)
            with col_corr:
                fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                                     color_continuous_scale="Peach", zmin=-1, zmax=1,
                                     title="Correlation Matrix")
                fig_corr.update_layout(template='plotly_dark', margin=dict(t=50, b=20, l=20, r=20))
                st.plotly_chart(fig_corr, width='stretch')

            with col_cov:
                fig_cov = px.imshow(cov_matrix, text_auto=".4f", aspect="auto",
                                    color_continuous_scale="Peach",
                                    title="Annualized Covariance Matrix")
                fig_cov.update_layout(template='plotly_dark', margin=dict(t=50, b=20, l=20, r=20))
                st.plotly_chart(fig_cov, width='stretch')
        elif len(valid_tickers) == 1:
            st.info("Need at least two valid assets to compute correlation and covariance matrices.")
        else:
            st.warning("Not enough data to compute correlation/covariance matrices in the given custom period.")

        st.divider()
        st.subheader("Portfolio Performance & Risk Metrics")

        metrics_df = ps_logic.get_comprehensive_metrics(
            data, allocations, start_str, end_str, rf_rate_decimal,
            market_ticker=ps_logic.MARKET_TICKER,
        )

        if metrics_df.empty:
            st.warning("Not enough data to calculate portfolio metrics.")
        else:
            formatted_df = metrics_df.copy()

            def format_metric(val, metric_name):
                if pd.isna(val) or val is None:
                    return "N/A"
                if metric_name in ["Mean Annual Return", "CAGR", "Standard Deviation"]:
                    return f"{val*100:.2f}%"
                return f"{val:.4f}"

            for col in formatted_df.columns:
                formatted_df[col] = [format_metric(val, idx) for idx, val in formatted_df[col].items()]

            st.dataframe(formatted_df, width='stretch')
