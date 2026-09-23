import streamlit as st
from datetime import date

from app import charts
from app.data import sharadar_client
from app.formatting import format_pct, format_ratio
from app.logic import portfolio_optimization as po_logic
from app.views.components import risk_free_rate_input

# --- Main Dashboard Render ---

def render_page():
    st.header("Portfolio Optimization")
    st.write("Determine optimal capital allocations based on Historical Data, Geometric Risk metrics, and the Kelly Criterion.")

    # --- UI Inputs ---
    st.markdown("### ⚙️ Optimization Parameters")
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            tickers_input = st.text_input(
                "Enter Tickers (comma-separated)",
                value="QQQ, GLD, XLE",
                help="Enter US-listed stock or ETF tickers covered by Sharadar."
            )
            start_date = st.date_input("Start Date", value=date(2020, 1, 1), min_value=date(1900, 1, 1))
            end_date = st.date_input("End Date", value=date.today(), min_value=date(1900, 1, 1))

        with col2:
            rf_rate_decimal = risk_free_rate_input()
            vol_target = st.number_input("Volatility Targeting (Max Vol %)", value=25.0, step=1.0)
            kelly_fraction = st.slider("Kelly Fraction", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

    # Convert settings to usable formats
    tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
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

        try:
            with st.spinner(f"Fetching historical data for {len(tickers_list)} assets..."):
                data = sharadar_client.fetch_price_history(tuple(tickers_list), start_str, end_str)
        except sharadar_client.SharadarError as error:
            st.error(str(error))
            return

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

        with tab_sharpe:
            st.subheader("Maximum Traditional Sharpe Portfolio")
            _render_optimization(
                data,
                po_logic.maximize_sharpe(data, rf_rate_decimal),
                ("Expected CAGR", "Annual Volatility", "Max Sharpe Ratio"),
            )

        with tab_geo:
            st.subheader("Maximum Geometric Sharpe Portfolio")
            _render_optimization(
                data,
                po_logic.maximize_geometric_sharpe(data, rf_rate_decimal),
                ("Expected CAGR", "Annual Volatility", "Max Geometric Sharpe"),
            )

        with tab_vol:
            st.subheader(f"Max Return (Cap at {vol_target}% Volatility)")
            _render_optimization(
                data,
                po_logic.volatility_targeting_optimizer(data, vol_target_decimal, rf_rate_decimal),
                ("Expected Annual Return", "Annual Volatility", "Sharpe Ratio"),
                hint="The cap may be below what any long-only mix of these assets can reach.",
            )

        with tab_kelly:
            st.subheader(f"Fractional Kelly-Criterion ({kelly_fraction:.2f}x)")
            kelly = po_logic.calculate_kelly_allocation(data, rf_rate_decimal, kelly_fraction)
            if not kelly.ok:
                st.error(f"Kelly Optimization failed. {kelly.reason}")
                return

            cash = kelly.allocation.get('CASH', 0)
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected Annual Return", format_pct(kelly.expected_return))
            c2.metric("Annual Volatility", format_pct(kelly.volatility))
            cash_yield = f"({rf_rate_decimal*100:.1f}% Yield)" if cash > 0 else ""
            c3.metric("Cash Position", f"{cash*100:.1f}%", cash_yield)
            _render_charts(
                data, kelly.allocation,
                "Allocation Weights (Including Cash)", "Yearly Returns (Equities Portion)",
            )


def _render_optimization(
    data,
    optimization: po_logic.Optimization,
    labels: tuple[str, str, str],
    *,
    hint: str = "",
) -> None:
    """The three headline figures, the allocation, and its yearly returns."""
    if not optimization.ok:
        st.error(f"Optimization failed. {optimization.reason} {hint}".strip())
        return

    return_label, volatility_label, ratio_label = labels
    c1, c2, c3 = st.columns(3)
    c1.metric(return_label, format_pct(optimization.expected_return))
    c2.metric(volatility_label, format_pct(optimization.volatility))
    c3.metric(ratio_label, format_ratio(optimization.ratio, 3))
    _render_charts(data, optimization.allocation, "Allocation Weights", "Yearly Portfolio Returns")


def _render_charts(data, allocation: dict, pie_title: str, bar_title: str) -> None:
    col_pie, col_bar = st.columns([1, 1.2])
    with col_pie:
        st.plotly_chart(charts.plot_allocation(allocation, pie_title), width='stretch')
    with col_bar:
        yearly_returns, _ = po_logic.get_portfolio_yearly_returns(data, allocation)
        st.plotly_chart(charts.plot_yearly_returns(yearly_returns, bar_title), width='stretch')
