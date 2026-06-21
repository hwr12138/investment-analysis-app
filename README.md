# Quantitative Analysis Application

An interactive Streamlit web application designed for financial engineering, technical analysis, and portfolio optimization.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://investment-analysis-app.streamlit.app/)

## Features

The application is divided into three main analytical modules:

* **Black-Scholes Model:** Calculates Call and Put option prices alongside their respective Greeks (Delta, Gamma, Theta, Vega, Rho). Features interactive heatmaps to visualize pricing and PnL across varying spot prices and volatilities.
* **Momentum Analysis:** Fetches live market data to compute MACD and RSI indicators. It includes live indicator scanners (crossovers, trend detection, failure swings) and an interactive Plotly candlestick chart for manual divergence analysis.
* **Portfolio Optimization:** Allocates capital across multiple assets based on historical data using three methodologies: Maximum Geometric Sharpe, Volatility Targeting, and the Fractional Kelly Criterion. Outputs include expected CAGR, volatility metrics, and visualized yearly returns.

## How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
