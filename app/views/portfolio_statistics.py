import streamlit as st
import plotly.express as px
from datetime import date

from app.data import sharadar_client
from app.formatting import format_pct, format_ratio
from app.logic import portfolio_statistics as ps_logic
from app.views.components import risk_free_rate_input

def render_page():
    st.header("Portfolio Statistics")
    st.write("Calculate comprehensive risk and return metrics for a custom portfolio and view asset correlations.")

    st.markdown("### ⚙️ Portfolio Parameters")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            allocations_input = st.text_input(
                "Enter Tickers and Allocations",
                value="QQQ:55, GLD:30, XLE:15",
                help="Format as TICKER:PERCENTAGE, separated by commas."
            )
            rf_rate_decimal = risk_free_rate_input(key="ps_rf_input")
        with col2:
            start_date = st.date_input("Custom Start Date", value=date(2020, 1, 1), min_value=date(1900, 1, 1), key="ps_start")
            end_date = st.date_input("Custom End Date", value=date.today(), min_value=date(1900, 1, 1), key="ps_end")

    run_button = st.button("Calculate Statistics", type="primary", width='stretch')

    if run_button:
        if start_date >= end_date:
            st.error("Please ensure your Start Date is before your End Date.")
            return

        try:
            allocations = ps_logic.parse_allocations(allocations_input)
        except ValueError:
            st.error("Invalid format for allocations. Please use TICKER:PERCENTAGE (e.g. QQQ:55, GLD:30).")
            return

        tickers_list = list(allocations.keys())
        if len(tickers_list) < 1:
            st.error("Please enter at least one valid ticker.")
            return

        total_alloc = sum(allocations.values())
        if ps_logic.needs_rescaling(total_alloc):
            st.warning(f"Allocations sum to {total_alloc}, not 100. They have been automatically normalized.")

        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # The fetch reaches back far enough for the longest fixed window.
        earliest_start_str = ps_logic.history_start(start_date, end_date).strftime('%Y-%m-%d')

        # The benchmark rides along in the same request; alpha, beta and Treynor
        # are measured against it.
        wanted = tuple(dict.fromkeys(tickers_list + [ps_logic.MARKET_TICKER]))
        try:
            with st.spinner("Fetching historical data and computing metrics..."):
                data = sharadar_client.fetch_price_history(wanted, earliest_start_str, end_str)
        except sharadar_client.SharadarError as error:
            st.error(str(error))
            return

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

        related = ps_logic.relationships(data, valid_tickers, start_str, end_str)
        if related is not None:
            corr_matrix, cov_matrix = related.correlation, related.covariance

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
            if ps_logic.MARKET_TICKER not in data.columns:
                st.warning(
                    f"No {ps_logic.MARKET_TICKER} prices came back, so alpha, beta and "
                    "Treynor are not reported rather than measured against a flat market."
                )
            formatted_df = metrics_df.copy()

            def format_metric(val, metric_name):
                if metric_name in ps_logic.PERCENT_METRICS:
                    return format_pct(val)
                return format_ratio(val, 4)

            for col in formatted_df.columns:
                formatted_df[col] = [format_metric(val, idx) for idx, val in formatted_df[col].items()]

            st.dataframe(formatted_df, width='stretch')
