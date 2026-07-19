import streamlit as st
from datetime import date

from app import charts
from app.data import market_data
from app.logic import portfolio_optimization as po_logic

# --- Main Dashboard Render ---

def render_page():
    st.header("Portfolio Optimization")
    st.write("Determine optimal capital allocations based on Historical Data, Geometric Risk metrics, and the Kelly Criterion.")

    # Initialize the default risk-free rate in session state to avoid fetching on every interaction.
    # The Resolved is memoized whole rather than just its value, so a session
    # running on the 4% fallback keeps saying so instead of pinning it silently.
    if "default_rf_rate" not in st.session_state:
        st.session_state.default_rf_rate = market_data.fetch_current_risk_free_rate()

    rf_rate = st.session_state.default_rf_rate
    if rf_rate.is_fallback:
        # A missing API key reports as an error, a failed call as a warning.
        getattr(st, rf_rate.severity)(
            f"Risk-free rate: standing in the {rf_rate.value:.2%} default — {rf_rate.detail}."
        )

    # --- UI Inputs ---
    st.markdown("### ⚙️ Optimization Parameters")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            tickers_input = st.text_input(
                "Enter Tickers (comma-separated)",
                value="QQQ, GLD, XLE",
                help="Enter valid Yahoo Finance tickers."
            )
            start_date = st.date_input("Start Date", value=date(2020, 1, 1), min_value=date(1900, 1, 1))
            end_date = st.date_input("End Date", value=date.today(), min_value=date(1900, 1, 1))

        with col2:
            rf_input = st.number_input(
                "Risk-Free Rate (%)",
                value=rf_rate.value * 100,
                step=0.1,
                help="Defaults to the implied rate of the 30-Day Fed Funds Futures (ZQ=F)."
            )
            vol_target = st.number_input("Volatility Targeting (Max Vol %)", value=25.0, step=1.0)
            kelly_fraction = st.slider("Kelly Fraction", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

    # Convert settings to usable formats
    tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    rf_rate_decimal = rf_input / 100.0
    vol_target_decimal = vol_target / 100.0
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    run_button = st.button("Run Portfolio Optimization", type="primary", width='stretch')

    # --- Execution Logic ---
    if run_button:
        if start_date >= end_date:
            st.error("Please ensure your Start Date is before your End Date.")
            return

        if len(tickers_list) < 2:
            st.error("Please enter at least two valid tickers.")
            return

        with st.spinner(f"Fetching historical data for {len(tickers_list)} assets..."):
            data = market_data.fetch_historical_data(tickers_list, start_str, end_str)

        if data.empty or data.shape[1] < 2:
            st.error("Failed to download enough valid data. Please check your tickers and date range.")
            return

        # Notify user if some tickers failed
        missing_tickers = [t for t in tickers_list if t not in data.columns]
        if missing_tickers:
            st.warning(f"Data not found for: {', '.join(missing_tickers)}. Proceeding with remaining assets.")

        # Create sub-tabs for the different optimization strategies
        st.divider()
        tab_sharpe, tab_geo, tab_vol, tab_kelly = st.tabs([
            "📊 Maximize Traditional Sharpe",
            "📈 Maximize Geometric Sharpe",
            "🛡️ Volatility Targeting",
            "🎲 Fractional Kelly"
        ])

        # --- 1. Maximize Traditional Sharpe ---
        with tab_sharpe:
            st.subheader("Maximum Traditional Sharpe Portfolio")
            alloc_sharpe, cagr_sharpe, vol_sharpe, sharpe_val = po_logic.maximize_sharpe(data, rf_rate_decimal)

            if alloc_sharpe:
                c1, c2, c3 = st.columns(3)
                c1.metric("Expected CAGR", f"{cagr_sharpe*100:.2f}%")
                c2.metric("Annual Volatility", f"{vol_sharpe*100:.2f}%")
                c3.metric("Max Sharpe Ratio", f"{sharpe_val:.3f}")

                col_pie, col_bar = st.columns([1, 1.2])
                with col_pie:
                    st.plotly_chart(charts.plot_allocation(alloc_sharpe, "Allocation Weights"), width='stretch')
                with col_bar:
                    yearly_rets_sharpe, _ = po_logic.get_portfolio_yearly_returns(data, alloc_sharpe)
                    st.plotly_chart(charts.plot_yearly_returns(yearly_rets_sharpe, "Yearly Portfolio Returns"), width='stretch')
            else:
                st.error("Optimization failed.")

        # --- 2. Maximize Geometric Sharpe ---
        with tab_geo:
            st.subheader("Maximum Geometric Sharpe Portfolio")
            alloc_geo, cagr_geo, vol_geo, sharpe_geo = po_logic.maximize_geometric_sharpe(data, rf_rate_decimal)

            if alloc_geo:
                c1, c2, c3 = st.columns(3)
                c1.metric("Expected CAGR", f"{cagr_geo*100:.2f}%")
                c2.metric("Annual Volatility", f"{vol_geo*100:.2f}%")
                c3.metric("Max Geometric Sharpe", f"{sharpe_geo:.3f}")

                col_pie, col_bar = st.columns([1, 1.2])
                with col_pie:
                    st.plotly_chart(charts.plot_allocation(alloc_geo, "Allocation Weights"), width='stretch')
                with col_bar:
                    yearly_rets_geo, _ = po_logic.get_portfolio_yearly_returns(data, alloc_geo)
                    st.plotly_chart(charts.plot_yearly_returns(yearly_rets_geo, "Yearly Portfolio Returns"), width='stretch')
            else:
                st.error("Optimization failed.")

        # --- 3. Volatility Targeting ---
        with tab_vol:
            st.subheader(f"Max Return (Cap at {vol_target}% Volatility)")
            alloc_vol, ret_vol, port_vol, sharpe_vol = po_logic.volatility_targeting_optimizer(data, vol_target_decimal, rf_rate_decimal)

            if alloc_vol:
                c1, c2, c3 = st.columns(3)
                c1.metric("Expected Annual Return", f"{ret_vol*100:.2f}%")
                c2.metric("Annual Volatility", f"{port_vol*100:.2f}%")
                c3.metric("Sharpe Ratio", f"{sharpe_vol:.3f}")

                col_pie, col_bar = st.columns([1, 1.2])
                with col_pie:
                    st.plotly_chart(charts.plot_allocation(alloc_vol, "Allocation Weights"), width='stretch')
                with col_bar:
                    yearly_rets_vol, _ = po_logic.get_portfolio_yearly_returns(data, alloc_vol)
                    st.plotly_chart(charts.plot_yearly_returns(yearly_rets_vol, "Yearly Portfolio Returns"), width='stretch')
            else:
                st.error("Optimization failed to converge under current volatility constraints.")

        # --- 4. Fractional Kelly ---
        with tab_vol: # (Will be placed in tab_kelly)
            pass

        with tab_kelly:
            st.subheader(f"Fractional Kelly-Criterion ({kelly_fraction:.2f}x)")
            alloc_kelly, ret_kelly, vol_kelly = po_logic.calculate_kelly_allocation(data, rf_rate_decimal, kelly_fraction)

            if alloc_kelly:
                c1, c2, c3 = st.columns(3)
                c1.metric("Expected Annual Return", f"{ret_kelly*100:.2f}%")
                c2.metric("Annual Volatility", f"{vol_kelly*100:.2f}%")
                cash_yield = f"({rf_rate_decimal*100:.1f}% Yield)" if alloc_kelly.get('CASH', 0) > 0 else ""
                c3.metric("Cash Position", f"{alloc_kelly.get('CASH', 0)*100:.1f}%", cash_yield)

                col_pie, col_bar = st.columns([1, 1.2])
                with col_pie:
                    st.plotly_chart(charts.plot_allocation(alloc_kelly, "Allocation Weights (Including Cash)"), width='stretch')
                with col_bar:
                    yearly_rets_kelly, _ = po_logic.get_portfolio_yearly_returns(data, alloc_kelly)
                    st.plotly_chart(charts.plot_yearly_returns(yearly_rets_kelly, "Yearly Returns (Equities Portion)"), width='stretch')
            else:
                st.error("Kelly Optimization failed.")
