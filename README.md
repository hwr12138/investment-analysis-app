# Equiti

A web application for equity research, options screening, technical analysis and portfolio
construction: a React + Mantine frontend (`web/`, on Firebase Hosting) over a FastAPI service
(`app/api/`, on Cloud Run). `web/README.md` covers running it, regenerating the API types, and
deploying.

## Features

The application is organized into five analytical modules:

* **DCF Valuation** (the 💰 DCF tab on Stock Research)**:** An unlevered free-cash-flow-to-the-firm model producing an intrinsic value per share, built on Sharadar Core US Fundamentals via the Nasdaq Data Link Tables API. Resolves every financial concept against the columns the API actually returns rather than hardcoding names, builds FCFF two independent ways and reconciles them, and regresses beta against SPY (Blume-adjusted, with its standard error). The forecast runs ten years by default from the trailing twelve months: year-1 growth starts from the lower of the trailing-twelve-month rate and the five-year CAGR, is held, then fades to the terminal rate; the margin, capex and D&A start from their TTM shares of revenue and can each move to a target — capex by default to its own five-year average, so a build-out or a trough is not held forever; working capital is held as a share of revenue. The headline is a **Gordon growth** value, whose terminal year adds back amortisation that capex never replaces and reinvests to grow at the company's own five-year median return on invested capital (never below the WACC); an **exit multiple** at the company's own historical EV ÷ (operating income + D&A) is always shown beside it, and the reverse DCF solves the growth the price implies under both. Companies reporting in another currency (TSM, ASML, SAP) are valued in US dollars: drivers come from their own-currency history and the base period is converted once at spot. Every displayed number carries a provenance badge (reported / derived / estimated / user) and, where computed, its formula. Forty-six validation checks — data integrity and forecast sanity alike — run on every change and surface as a persistent red (critical) or amber (warning) banner with per-check acknowledgement; banks and insurers are rejected outright rather than valued with a model that does not fit them. A sensitivity grid, and a multi-sheet workbook that reproduces value per share outside the app, come with every run. The Summary tab's price chart carries the same default value as a step line, re-struck at every filing in the window as the company stood that day — the statements filed by then, that day's price, 10-year yield and beta.
* **Portfolio Optimization:** Allocates capital across multiple assets from historical data using four methodologies: Maximum Traditional Sharpe, Maximum Geometric Sharpe, Volatility Targeting, and the Fractional Kelly Criterion. Outputs include expected CAGR, volatility metrics, and visualized yearly returns.
* **Portfolio Statistics:** Computes comprehensive risk and return metrics (Mean Annual Return, CAGR, Standard Deviation, Alpha, Beta, Sharpe, Geometric Sharpe, Sortino, and Treynor) for a custom-weighted portfolio across trailing 1/3/5/10-year and custom periods, alongside correlation and annualized covariance matrices.
* **Options:** Screens ninety curated underlyings each night for cheap upside calls. The chains come from ThetaData through a separate Cloud Run job (`collector/`), which prices every surface (Black-76 on the put–call-parity forward), solves the Discover and Screen tables, and keeps a constant-maturity history of implied volatility, skew and reward:risk, so the page reads a stored night rather than calling the vendor.
* **Momentum Analysis:** Fetches end-of-day price history to compute MACD and RSI indicators. It includes live indicator scanners (crossovers, trend detection, failure swings) and an interactive Plotly candlestick chart for manual divergence analysis.

## Project structure

```
investment-analysis-app/
├── web/                   # The frontend: React + Mantine, one folder per page
│   └── src/pages/         #   (research, momentum, options, portfolio)
├── app/
│   ├── api/               # FastAPI: one router per page, every route behind sign-in
│   ├── logic/             # Pure calculations, one module per feature
│   │   ├── dcf/           # The DCF engine (see below)
│   │   ├── options/       # Options pricing and screens, shared with the collector
│   │   └── research/      # Stock Research: comparison ← tab modules ← dataset
│   ├── data/              # Shared data access (Sharadar, Yahoo Treasury yields, ERP,
│   │   │                  #   the options store)
│   │   └── sharadar/      # The Sharadar client, one module per table, behind
│   │                      #   sharadar_client.py; vocabulary.py is the pure part
│   ├── provenance.py      # Origin/Sourced types shared by the data and logic layers
│   └── formatting.py      # Number and date formatting shared by the logic layer
├── collector/             # The nightly options job (Cloud Run Jobs); deploy runbook inside
├── deploy/api/            # The API's Dockerfile and Cloud Build config
├── tests/                 # pytest suite; all network calls mocked. test_architecture.py
│                          #   fails the build if the layering below is broken
├── requirements.txt       # What the API image installs
└── requirements-dev.txt   # adds pytest and httpx
```

Each feature is split into a **logic** module (`app/logic/<feature>.py`) that holds
the math, and an **API router** (`app/api/<feature>.py`) that fetches through
`app/data/`, hands the frames to the logic, and shapes the response. The API does
no arithmetic of its own. All external API calls live in `app/data/` so caching
and API keys stay in one place.

To add a new page:

1. Create `app/logic/<feature>.py` for the calculations.
2. Add a router in `app/api/` whose response models derive from
   `app.api.contract.Contract`, and mount it in `app/api/main.py`, which puts
   every route behind sign-in.
3. Run `npm run gen:api` in `web/` to regenerate the TypeScript types.
4. Add the page under `web/src/pages/` and its route and nav entry in
   `web/src/App.tsx`.

The logic module never fetches, which `tests/test_architecture.py` checks.

### The DCF engine

The DCF feature is a package rather than the usual single logic module, because
its schema resolution, statement normalisation, validation and arithmetic each
need to be testable on their own:

| Module | Responsibility |
|---|---|
| `app/data/sharadar_client.py` | Public API of `app/data/sharadar/`: auth, retry, pagination, caching, datatable parsing |
| `app/logic/dcf/schema.py` | Concept alias table and runtime column resolution |
| `app/logic/dcf/financials.py` | Statement normalisation, TTM, ratio derivation |
| `app/logic/dcf/wacc.py` | Beta regression and discount rate — **pure** |
| `app/logic/dcf/valuation.py` | FCFF, projection, terminal value, equity bridge — **pure** |
| `app/logic/dcf/reverse.py` | Backsolving a driver from a target price — **pure** |
| `app/logic/dcf/validation.py` | Checks, severity model, issue types, tolerances |
| `app/logic/dcf/model.py` | Orchestration: fetch → resolve → normalise → validate → value |
| `app/logic/dcf/inputs.py` | The assumption panel as data: defaults, the reader's overrides, the valued run |
| `app/logic/dcf/export.py` | Multi-sheet workbook and CSV export |

`wacc.py` and `valuation.py` import no Streamlit, make no network calls, and
hold no global state — numbers in, dataclasses out. Every tolerance in the model
lives in one `Tolerances` dataclass in `validation.py`.

### Financial statements

| Module | Responsibility |
|---|---|
| `app/logic/statements.py` | The four SF1 statement layouts and their evaluator — **pure** |

Declares each statement as sections of rows: reported line items, derived
"Other …" residuals where Sharadar's detail does not account for its own
subtotal, and reconciliations of each reported subtotal against the lines above
it. Subtotals always show what the company filed; where the two disagree the row
is flagged rather than quietly corrected. Its thresholds live in one
`StatementTolerances` dataclass, separate from the model's on purpose.

### API keys

`SHARADAR_API_KEY` (fundamentals and prices) is the only key the app needs. Locally
it is read from the environment or `.streamlit/secrets.toml` (gitignored; the data
layer still reads it through Streamlit's secrets); on Cloud Run it comes from
Secret Manager. Treasury yields
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

From the repository root, install the requirements and start the API with sign-in off:

```
$ pip install -r requirements-dev.txt
$ EQUITI_AUTH=off python3 -m uvicorn app.api.main:app --port 8600 --reload --reload-dir app
```

Then, in a second terminal, the frontend:

```
$ cd web && npm install && npm run dev
```

Open <http://localhost:5173>. `web/README.md` has the rest, sign-in and deploys included.
