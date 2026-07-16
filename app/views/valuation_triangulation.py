import streamlit as st

from app.data import market_data
from app.logic import valuation_triangulation as vt_logic

def render_page():
    st.title("Valuation Triangulation")

    # Add ticker input globally for all tabs
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    with col2:
        st.write("") # For vertical alignment
        st.write("")
        if st.button("Run Valuation", use_container_width=True):
            st.session_state['vt_run'] = True

    if 'vt_run' not in st.session_state:
        st.session_state['vt_run'] = False

    # Create the 5 tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Unlevered DCF",
        "Levered DCF",
        "Comparable Company Analysis",
        "Historical Multiple Reversion",
        "Reverse DCF"
    ])

    with tab1:
        st.header("Unlevered DCF")
        if st.session_state['vt_run']:
            with st.spinner(f"Fetching data for {ticker}..."):
                # Fetch baseline data
                financials = market_data.fetch_simfin_financials(ticker)

                if not financials:
                    st.error("Failed to fetch historical data. Check if Ticker is correct or try again.")
                else:
                    risk_free_rate = market_data.fetch_10y_treasury_rate()
                    beta = market_data.fetch_beta(ticker)
                    erp = market_data.fetch_erp()
                    defaults = vt_logic.calculate_historical_averages(financials)

                    st.subheader("Assumptions (Derived from Historicals)")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        rev_growth = st.number_input("Revenue Growth Rate (%)", value=defaults.get('rev_growth', 0.05) * 100, step=1.0) / 100
                        ebit_margin = st.number_input("Target EBIT Margin (%)", value=defaults.get('ebit_margin', 0.15) * 100, step=1.0) / 100
                    with col2:
                        tax_rate = st.number_input("Tax Rate (%)", value=defaults.get('tax_rate', 0.21) * 100, step=1.0) / 100
                        dna_margin = st.number_input("D&A Margin (%)", value=defaults.get('dna_margin', 0.05) * 100, step=1.0) / 100
                    with col3:
                        capex_margin = st.number_input("CapEx Margin (%)", value=defaults.get('capex_margin', 0.05) * 100, step=1.0) / 100
                        nwc_margin = st.number_input("NWC Change Margin (%)", value=defaults.get('nwc_margin', 0.02) * 100, step=1.0) / 100
                    with col4:
                        cost_of_debt = st.number_input("Cost of Debt (%)", value=defaults.get('cost_of_debt', 0.05) * 100, step=0.1) / 100
                        exit_multiple = st.number_input("Exit Multiple (EV/EBITDA)", value=defaults.get('exit_multiple', 10.0), step=1.0)

                    st.subheader("WACC Inputs")
                    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
                    with w_col1:
                        rfr_input = st.number_input("Risk Free Rate (%)", value=risk_free_rate * 100, step=0.1) / 100
                    with w_col2:
                        beta_input = st.number_input("Beta", value=beta, step=0.1)
                    with w_col3:
                        erp_input = st.number_input("Equity Risk Premium (%)", value=erp * 100, step=0.1) / 100
                    with w_col4:
                        ev_market_cap = financials['shares'] * financials['current_price']
                        latest_debt = financials['total_debt'].iloc[0] if not financials['total_debt'].empty else 0
                        ev_total = ev_market_cap + latest_debt

                        debt_weight_def = latest_debt / ev_total if ev_total > 0 else 0.2
                        debt_weight = st.number_input("Debt Weight (%)", value=debt_weight_def * 100, step=1.0) / 100
                        equity_weight = 1.0 - debt_weight
                        st.caption(f"Equity Weight: {equity_weight*100:.1f}%")

                    wacc = vt_logic.calculate_wacc(rfr_input, beta_input, erp_input, cost_of_debt, tax_rate, debt_weight, equity_weight)
                    st.markdown(f"**Calculated WACC:** {wacc*100:.2f}%")

                    assumptions = {
                        'rev_growth': rev_growth,
                        'ebit_margin': ebit_margin,
                        'tax_rate': tax_rate,
                        'dna_margin': dna_margin,
                        'capex_margin': capex_margin,
                        'nwc_margin': nwc_margin,
                        'exit_multiple': exit_multiple
                    }

                    dcf_results = vt_logic.run_unlevered_dcf(financials, assumptions, wacc)

                    st.subheader("Results")
                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("Implied Share Price", f"${dcf_results['implied_share_price']:.2f}")
                    res_col2.metric("Current Share Price", f"${financials['current_price']:.2f}")

                    if financials['current_price'] > 0:
                        premium_discount = (dcf_results['implied_share_price'] / financials['current_price']) - 1
                        res_col3.metric("Premium / (Discount)", f"{premium_discount*100:.2f}%")

                    st.subheader("Projected Free Cash Flows")
                    st.dataframe(dcf_results['projections'].style.format({
                        'Revenue': '${:,.0f}',
                        'EBIT': '${:,.0f}',
                        'NOPAT': '${:,.0f}',
                        '+ D&A': '${:,.0f}',
                        '- CapEx': '${:,.0f}',
                        '- Change in NWC': '${:,.0f}',
                        'UFCF': '${:,.0f}',
                        'PV of UFCF': '${:,.0f}'
                    }))
        else:
            st.info("Enter a ticker in the 'Valuation Settings' section above and click 'Run Valuation' to see the DCF.")

    with tab2:
        st.header("Levered DCF")
        st.write("Levered DCF methodology goes here.")

    with tab3:
        st.header("Comparable Company Analysis")
        st.write("Comparable Company Analysis methodology goes here.")

    with tab4:
        st.header("Historical Multiple Reversion")
        st.write("Historical Multiple Reversion (Time-Series Analysis) goes here.")

    with tab5:
        st.header("The Reverse DCF")
        st.write("The Reverse DCF (Market-Implied Expectations) goes here.")
