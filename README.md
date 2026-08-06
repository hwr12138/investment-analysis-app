# Quantitative Analysis Application

An interactive Streamlit web application designed for financial engineering, technical analysis, and portfolio optimization.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://investment-analysis-app.streamlit.app/)

## Features

The application is organized into five analytical modules:

* **DCF Valuation:** An unlevered free-cash-flow-to-the-firm model producing an intrinsic value per share for a US-listed equity, built on Sharadar Core US Fundamentals via the Nasdaq Data Link Tables API. Resolves every financial concept against the columns the API actually returns rather than hardcoding names, builds FCFF two independent ways and reconciles them, regresses beta against SPY with its R², and computes both Gordon growth and exit multiple terminal values side by side with each method's implied version of the other. Every displayed number carries a provenance badge (reported / derived / estimated / user) and, where computed, its formula. Twenty-four validation checks run on load and surface as a persistent red (critical) or amber (warning) banner with per-check acknowledgement; banks and insurers are rejected outright rather than valued with a model that does not fit them. The full model exports to a multi-sheet workbook that reproduces value per share outside the app.
* **Portfolio Optimization:** Allocates capital across multiple assets from historical data using four methodologies: Maximum Traditional Sharpe, Maximum Geometric Sharpe, Volatility Targeting, and the Fractional Kelly Criterion. Outputs include expected CAGR, volatility metrics, and visualized yearly returns.
* **Portfolio Statistics:** Computes comprehensive risk and return metrics (Mean Annual Return, CAGR, Standard Deviation, Alpha, Beta, Sharpe, Geometric Sharpe, Sortino, and Treynor) for a custom-weighted portfolio across trailing 1/3/5/10-year and custom periods, alongside correlation and annualized covariance matrices.
* **Black-Scholes Model:** Calculates Call and Put option prices alongside their respective Greeks (Delta, Gamma, Theta, Vega, Rho). Features interactive heatmaps to visualize pricing and PnL across varying spot prices and volatilities.
* **Momentum Analysis:** Fetches end-of-day price history to compute MACD and RSI indicators. It includes live indicator scanners (crossovers, trend detection, failure swings) and an interactive Plotly candlestick chart for manual divergence analysis.

## Project structure

```
investment-analysis-app/
├── streamlit_app.py       # Entrypoint: page config, styling, and navigation
├── app/
│   ├── views/             # Streamlit UI for each page (render_page functions)
│   ├── logic/             # Pure calculations, one module per feature
│   │   └── dcf/           # The DCF engine (see below)
│   ├── data/              # Shared data access (Sharadar, Yahoo Treasury yields, ERP)
│   ├── provenance.py      # Origin/Sourced types shared by the data and logic layers
│   └── charts.py          # Reusable Plotly chart helpers
├── assets/style.css       # Custom CSS
├── tests/                 # pytest suite; all network calls mocked
├── requirements.txt
└── requirements-dev.txt   # adds pytest; not installed on the deployed app
```

Each feature is split into a **view** (`app/views/<feature>.py`) that renders the
UI and a **logic** module (`app/logic/<feature>.py`) that holds the math. All
external API calls live in `app/data/` so caching and API keys stay in one place.

To add a new page:

1. Create `app/logic/<feature>.py` for the calculations.
2. Create `app/views/<feature>.py` with a `render_page()` function.
3. Register it in the `st.navigation([...])` list in `streamlit_app.py`.
4. Reuse `app/data/market_data.py` for any data fetching.

### The DCF engine

The DCF feature is a package rather than the usual single logic module, because
its schema resolution, statement normalisation, validation and arithmetic each
need to be testable on their own:

| Module | Responsibility |
|---|---|
| `app/data/sharadar_client.py` | Nasdaq Data Link auth, retry, pagination, caching, datatable parsing |
| `app/logic/dcf/schema.py` | Concept alias table and runtime column resolution |
| `app/logic/dcf/financials.py` | Statement normalisation, TTM, ratio derivation |
| `app/logic/dcf/wacc.py` | Beta regression and discount rate — **pure** |
| `app/logic/dcf/valuation.py` | FCFF, projection, terminal value, equity bridge — **pure** |
| `app/logic/dcf/validation.py` | Checks, severity model, issue types, tolerances |
| `app/logic/dcf/model.py` | Orchestration: fetch → resolve → normalise → validate → value |
| `app/logic/dcf/export.py` | Multi-sheet workbook and CSV export |

`wacc.py` and `valuation.py` import no Streamlit, make no network calls, and
hold no global state — numbers in, dataclasses out. Every tolerance in the model
lives in one `Tolerances` dataclass in `validation.py`.

### API keys

`SHARADAR_API_KEY` (fundamentals and prices) is the only key the app needs; it
is read from `.streamlit/secrets.toml`, which is gitignored. Treasury yields
come from Yahoo Finance via `yfinance`, which needs no key. Nasdaq Data Link
only accepts its key as a query parameter, so no exception raised by the client
carries the request params or URL — the key never appears in a log line, an
error message, or an export.

## Running the tests

```
$ pip install -r requirements-dev.txt
$ python -m pytest tests/ -v
```

No test touches the network; the fixtures mirror real Sharadar datatable responses.

## How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
