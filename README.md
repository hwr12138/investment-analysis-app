# Quantitative Analysis Application

An interactive Streamlit web application designed for financial engineering, technical analysis, and portfolio optimization.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://investment-analysis-app.streamlit.app/)

## Features

The application is organized into five analytical modules:

* **Valuation Triangulation:** Estimates a company's intrinsic value from multiple angles. The Unlevered DCF is implemented — it pulls historical financials, derives baseline assumptions (growth, margins, capital intensity), computes WACC, and projects five years of free cash flow to an implied share price versus the current price. Levered DCF, Comparable Company Analysis, Historical Multiple Reversion, and Reverse DCF are scaffolded for future work.
* **Portfolio Optimization:** Allocates capital across multiple assets from historical data using four methodologies: Maximum Traditional Sharpe, Maximum Geometric Sharpe, Volatility Targeting, and the Fractional Kelly Criterion. Outputs include expected CAGR, volatility metrics, and visualized yearly returns.
* **Portfolio Statistics:** Computes comprehensive risk and return metrics (Mean Annual Return, CAGR, Standard Deviation, Alpha, Beta, Sharpe, Geometric Sharpe, Sortino, and Treynor) for a custom-weighted portfolio across trailing 1/3/5/10-year and custom periods, alongside correlation and annualized covariance matrices.
* **Black-Scholes Model:** Calculates Call and Put option prices alongside their respective Greeks (Delta, Gamma, Theta, Vega, Rho). Features interactive heatmaps to visualize pricing and PnL across varying spot prices and volatilities.
* **Momentum Analysis:** Fetches live market data to compute MACD and RSI indicators. It includes live indicator scanners (crossovers, trend detection, failure swings) and an interactive Plotly candlestick chart for manual divergence analysis.

## Project structure

```
investment-analysis-app/
├── streamlit_app.py       # Entrypoint: page config, styling, and navigation
├── app/
│   ├── views/             # Streamlit UI for each page (render_page functions)
│   ├── logic/             # Pure calculations, one module per feature
│   ├── data/              # Shared market-data access (yfinance, FMP, ERP, SimFin)
│   └── charts.py          # Reusable Plotly chart helpers
├── assets/style.css       # Custom CSS
└── requirements.txt
```

Each feature is split into a **view** (`app/views/<feature>.py`) that renders the
UI and a **logic** module (`app/logic/<feature>.py`) that holds the math. All
external API calls live in `app/data/market_data.py` so caching and API keys
stay in one place.

To add a new page:

1. Create `app/logic/<feature>.py` for the calculations.
2. Create `app/views/<feature>.py` with a `render_page()` function.
3. Register it in the `st.navigation([...])` list in `streamlit_app.py`.
4. Reuse `app/data/market_data.py` for any data fetching.

## How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
