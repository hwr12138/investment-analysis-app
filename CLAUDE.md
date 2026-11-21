# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements-dev.txt          # requirements.txt (what the API image installs) + pytest, httpx
python -m pytest tests/ -q                   # full suite (~1,300 tests, no network; CI runs it too)
python -m pytest tests/test_valuation.py -q  # one file
python -m pytest tests/test_wacc.py::test_market_value_of_equity -q  # one test
python -m pytest tests/ -k "fcff" -q         # by name
```

There is no linter, formatter, or pytest config file in this repo. The web frontend
(`web/`) has a TypeScript type check: `cd web && npm run typecheck`.

`.claude/launch.json` defines `api` (uvicorn on 8600, sign-in off) and `web` (Vite on 5173,
proxying `/api` to 8600); start both to preview the app, rather than through Bash.

**Merging to `main` deploys.** `.github/workflows/deploy.yml` runs the tests (`tests.yml`, which
pull requests also run), then deploys only what the merge changed: the API to Cloud Run, the
collector's image to its Cloud Run job, the web app to Firebase Hosting, in that order. It signs
in through Workload Identity Federation as a deployer account that trusts only that workflow on
`main`; `deploy/README.md` has the setup. A deploy is a merge, so there is no separate release
step to remember — and a change that should not ship yet should not be merged.

`SHARADAR_API_KEY` (fundamentals, prices) comes from the environment or
`.streamlit/secrets.toml` at the repository root (gitignored; `app/data/settings.py` reads it
with `tomllib`, and the directory keeps the name the Streamlit app gave it), and on Cloud Run
from Secret Manager. Treasury yields come from Yahoo Finance via `yfinance` (`^IRX` short rate, `^TNX`
10-year) and need no key. Tests never need either.

The 🎯 Options page reads a store written by a separate job, `collector/` (Python ≥ 3.12, its
own `requirements.txt`; deploy runbook in `collector/README.md`). Point the page at a store with
`OPTIONS_STORE_URI` (env or secrets: `gs://bucket` or `file:///abs/path`), or `OPTIONS_BUCKET` plus
a `[gcp_service_account]` table. `THETADATA_API_KEY` is the collector's, never the page's.

## Architecture

A web app: React + Mantine in `web/` on Firebase Hosting, over a FastAPI layer in `app/api/` on
Cloud Run, over the logic and data layers. It replaced a Streamlit front end, removed in November
2026 once every page had been ported, and Streamlit itself went with it: nothing imports it and
`requirements.txt` does not install it (`test_architecture.py` checks the first).
`web/README.md` covers running, regenerating the TypeScript types, and deploying. The rules that
matter:

- `app/api/` is the top layer: it fetches through `app/data/` and hands frames to `app/logic/`,
  and it does no arithmetic of its own. Logic a page needs goes in the logic module, where it is
  tested without a request (as `momentum_analysis.indicator_frame`,
  `portfolio_statistics.relationships`, the options filters in `options/coverage.py`, Stock
  Research's `verdict`s, `summary.dcf_line_steps`, `insiders.open_market_trades` and the whole 💰
  DCF assumption panel, `dcf/inputs.py`, did when they moved out of the Streamlit views).
  `test_architecture.py` forbids anything below the API from importing it.
- The API caches nothing of its own: the data layer's caches make a warm loader cost ~0.01 s. The
  one exception is `research.py`'s DCF-line points (`functools.lru_cache`, one entry per filing, so
  an exception — an outage — is never remembered). The research page fetches the profile first and
  mounts a tab only once it is in, so a cold search assembles the snapshot once, not once per request.
- Every response model derives from `app.api.contract.Contract`, which **rejects NaN**. Pydantic
  would otherwise serialise a NaN as `null` even into a `float` field. A number that can be
  missing is typed `float | None` and converted with `finite_or_none`.
- Every data route carries `auth.require_user` (a test checks the routes). Sign-in is a Firebase
  ID token plus an email allowlist, and `EQUITI_AUTH=off` is ignored on Cloud Run.
- The data layer caches through `app/data/cache.py` — `@cache.data(ttl=...)` for values,
  `@cache.resource()` for a shared client — in process, so each Cloud Run instance starts cold.
  It replaced `st.cache_data` and kept its rules, each of which something relies on: **an
  exception is never cached**; **the key is the arguments as passed, defaults not filled in**; **one
  computation per key at a time** (concurrent misses wait for the first, other keys run alongside);
  and **every hit is an unpickled copy**, so a caller mutating a frame cannot change the next one's.
  `tests/test_cache.py` pins each. `cache.clear_all()` empties every cache (the tests' autouse
  fixture). Keys and settings come from
  `app/data/settings.py`: the environment first, then `.streamlit/secrets.toml`.

Each feature is three Python layers under the page that draws it (`web/src/pages/<feature>/`):

- `app/api/<feature>.py` — the routes and their response models; no arithmetic
- `app/logic/<feature>.py` — the math: no network, no cache
- `app/data/` — every external API call, so caching and keys stay in one place

A feature earns a **package** in place of a logic module when it grows several independently
testable parts — `app/logic/dcf/`, `app/logic/research/` and `app/logic/options/` are the cases.
Reach for it when the single module would otherwise hold pieces that have no reason to know
about each other, not at a line count.

`app/provenance.py` sits at the `app/` root because both the data and logic layers construct its
types and neither may import the other. `app/formatting.py` sits beside it and imports no layer
either: the logic layer builds display strings with it (`StatTile`, the DCF panel's notes).

**The layering is enforced, not just described.** `tests/test_architecture.py` parses `app/` into
an import graph and fails if anything imports Streamlit; if a logic module reaches `requests`,
`yfinance`, or the data layer's caches and settings (`app.data.cache`, `app.data.settings`) — even
transitively, which is how it actually breaks — other than the three orchestrators
(`logic/research/dataset.py`, `logic/dcf/model.py`, `logic/dcf/inputs.py`); if data imports
logic; if anything below the API imports it; if the DCF arithmetic reaches `app.data` at all; or
if any module uses another's `_private` name. A logic module that needs a Sharadar constant imports
`app.data.sharadar.vocabulary`, never `sharadar_client`.

To add a page: create the logic module; add a router in `app/api/` whose models derive from
`Contract`, fetching through `app/data/` — `sharadar_client` or `market_data` — never from the
logic module, and mount it in `app/api/main.py`; run `npm run gen:api` in `web/`; then add the
page under `web/src/pages/` with its route and nav entry in `web/src/App.tsx`.

To add a **tab** to Stock Research: create `app/logic/research/<tab>.py` for the arithmetic, with
a loader beside the others in `dataset.py` if it fetches; add its route to `app/api/research.py`;
then add its key to `TABS` in `web/src/pages/research/common.ts`, and its label and panel to
`ResearchPage.tsx`. Only built tabs go in the strip: there are no "not built yet" stubs.

A tab that compares metrics against peers and against history should be a **sub-tab of ⚖️
Valuation** rather than a new top-level one: declare its metrics and its own usable/base rule,
and reuse `logic/research/comparison.py`, the API's `Relative` response and `ValuationTab.tsx`'s
grid for everything else. That is what Growth is, and it is why there is no top-level 🌱 Growth tab.

### The Stock Research page

The one feature whose logic is a package rather than a module, because it carries six tabs:

```
web/src/pages/research/  ResearchPage.tsx = the shell; one component per tab beside it,
                         and dcf/ for the 💰 DCF tab's parts
app/api/research.py      one route per tab (the DCF's are in app/api/dcf.py)
app/logic/research/      comparison ← summary, valuation, growth,
                         insiders, holdings ← dataset
```

`ResearchPage.tsx` owns the ticker box, the profile header and `TAB_LABELS` — **the one place tab
order is decided** — and mounts **only the open tab** (`keepMounted={false}`), so only it fetches
(see *Lazy tabs* under UI conventions). The tab, like every choice inside one, is in the URL.

On the logic side the dependencies run one way — `comparison` is shared, each
tab module owns its arithmetic, and `dataset` sits on top because it holds every network-touching
loader: `load_research` (the snapshot — only what 📊 Summary draws) and one loader per tab beside
it (`load_financials`, `load_insiders`, `load_peers`, `load_holdings`). Its calls into
`summary.build_key_stats` and `comparison.build_peer_universe` are the only upward edges, and both
exist for the same reason: the result is cached per company per day, so anything a tab would
otherwise recompute on every request is precomputed there.

`logic/research/comparison.py` holds what the two ⚖️ sub-tabs genuinely share — the `Comparison`
record, the median-and-percentile summary, the peer universe join, the industry lookup. What it
deliberately does **not** hold is either tab's notion of a *usable* number or its `Tolerances`,
because on both points the two are opposites; see the ⚖️ section below.

Two modules deliberately stay outside the package. `app/logic/statements.py` declares the four
SF1 statement layouts (sections, rows, subtotals, reconciliations) and evaluates them into a
rendered table; the financials tab is its only consumer today, and it sits outside so a second
one needs no move. `app/logic/dcf/` stays independently reusable so batch screening remains an
extension rather than a rewrite. Both are pure.

**Statements are shown in exactly one place — the 📄 Financials tab.** The DCF tab points there
and keeps only the per-year SEC filing links (`inputs.filings`), which Financials does not show.
Do not add a second statement table anywhere: the Streamlit DCF tab once carried one, which scaled
every row by the units divisor and turned EPS and margins into `0.00`, and `logic/statements.py`
carries a `Unit` enum precisely so a layout cannot repeat that.

### The ⚖️ Relative Valuation tab and its two sub-tabs

`Valuation` asks what the market pays for the company; `Growth` asks how fast the thing being
paid for moves. Both answer per metric in the same three columns — current, the company's own 5Y
median, the industry median — over one peer universe, and a metric that cannot fill all three is
not shown. On the web the two are one table read two ways: `ValuationTab.tsx` switches with
`view=growth` in the URL, draws the shared grid, and words each one's verdict and legend. The
`/valuation` and `/growth` routes both read the peer universe through `dataset.load_peers`,
cached per day, so whichever opens second costs nothing.

**Trailing-only, deliberately.** Forward multiples were dropped because yfinance supplies forward
EPS and revenue only per-ticker by scrape, so forward *industry* medians would cost thousands of
scrapes and a forward *5Y history* needs point-in-time consensus no source here has. Neither
sub-tab makes a yfinance call.

**The two sub-tabs' central rule is inverted, and that is why they cannot share `_usable` or
`_current`:**

- *Valuation* — **only positive readings count**. A negative P/E is a loss, not a cheap stock; a
  zero yield is a non-payer.
- *Growth* — **a negative reading is the answer** (revenue falling 8% is what the reader came
  for), but **the base must be positive**. Dividing by a negative base inverts the sign without
  saying so: Intel's diluted EPS went −4.75 to −2.12, halving the loss, and the ratio prints
  −55%. The same trap fires on JPM's operating cash flow, on Intel's EBIT/FCF/ROE, and on Apple's
  working capital.

Shared by both: the exclusion is applied **per metric, not per company** — a loss-maker still has
an EV/Sales and a revenue growth rate, and belongs in both medians. **Medians, never means**,
because a denominator approaching zero prints in the thousands.

Three more things Growth has to get right, each verified against live data:

- **`capex` is negative for every company and zero for banks**, so growth is taken on the
  **magnitude** — "+20%" means more spending. Apple's −12.38bn → −10.04bn reads −18.9%.
- **Pair periods on `Calendar Date`, never `Report Date`.** Companies on a 52/53-week fiscal
  calendar close a day or two earlier each year — Apple's quarters end 2025-06-28 and 2026-06-27
  — so an exact twelve-month match on the report date finds nothing and *every* growth rate reads
  unavailable. This shipped once and was caught only against live data; there is a regression
  test for it.
- **The 5Y window counts filed quarters, not sessions** (~20 against ~1,255), which is why
  `GrowthTolerances.min_observations` is 8 where the valuation tab's is 60, and why the view
  prints the unit beside the count. The gap to the industry is likewise in **percentage points**,
  not a ratio: dividing one rate by another says little and breaks when the median is negative.

### The DCF engine

`app/logic/dcf/` is a package rather than the usual single logic module so schema resolution,
normalisation, validation, and arithmetic are each independently testable. The pipeline is
**fetch → resolve → normalise → validate → value**, assembled in `model.py`:

| Module | Responsibility |
|---|---|
| `app/data/sharadar_client.py` | The client's public API — re-exports `app/data/sharadar/`: `transport` (auth, retry, pagination, parsing), `vocabulary` (pure), and one module per table |
| `schema.py` | Concept alias table, runtime column resolution |
| `financials.py` | Statement normalisation, TTM, ratio derivation |
| `wacc.py` | Beta regression, discount rate — **pure** |
| `valuation.py` | FCFF, projection, terminal value, equity bridge — **pure** |
| `reverse.py` | Backsolving a driver from a target price — **pure**, DCF-agnostic |
| `validation.py` | Checks, severity model, issue types, tolerances |
| `model.py` | Orchestration: the load, the run, the sensitivity grid |
| `inputs.py` | The assumption panel as data — defaults, overrides, provenance, notes — and everything the tab draws from a run |
| `export.py` | Multi-sheet `.xlsx` and CSV export |

`wacc.py`, `valuation.py`, and `reverse.py` import no Streamlit, make no network calls, and hold
no global state. Keep them that way — it is what makes batch screening an extension rather than a
rewrite. `reverse.py` takes a *callable* from driver value to value per share; putting the solver
inside `valuation.py` would create an import cycle with `model.py`.

`model.load_company()` is the only network-touching entry point and is called once per ticker;
`model.run_model()` runs on every assumption change. Never move a fetch into `run_model`.

**`inputs.py` is the panel without the widgets.** `resolve(dataset, overrides=...)` takes only what
the reader changed (`Overrides`, every field `None` by default) and returns `Assumptions`: each
input's value, its default, its `Sourced` and its notes, plus the `inputs` dict `run_model` takes.
An input left alone is *no override at all*, so it follows its default when the default moves — an
untyped cost of debt follows the risk-free rate, a driver value follows its basis — and a value
typed back to the default at the precision shown (`_changed`) counts as untouched. A choice the
panel does not offer raises `InvalidOverride` (the API's 422). A run that cannot be built
(no share count, a WACC that raises) still returns the whole panel with `blocked` set, so the
reader can undo whatever caused it. `evaluate` runs the model, both reverse solves, the grid and
the checks; `reverse_verdict`, `revenue_chart`, `scan_rows`, `cross_implied` and `filings` are what
the tab draws from the result. It is an orchestrator only because `default_value_on`
fetches the company and that day's rates; `resolve` and `evaluate` take what they are handed.

### The 🎯 Options page and its collector

The one page that fetches nothing from a vendor. ThetaData's free tier — the source — serves
**one request at a time** at ~1–2.5 s of server time per session, refuses the current day, and has
no open interest or greeks, so ~120 names cannot be fetched while a page loads. Instead:

```
Cloud Scheduler → Cloud Run job  collector/  (thetadata, python 3.12) ─┐ imports only
                                                                        │ app.logic.options
                    private GCS bucket  options/v1/…  ◄─────────────────┘ + app.options_layout
                         ▲
FastAPI   ── app/data/options_store.py → app/api/options.py   (reads the stored screens)
```

- **`app/options_layout.py` names every object** and sits at the `app/` root because the data
  layer's reader and the collector (which imports logic) both need it. `manifest.json` is written
  **last** by every run, so a reader can trust whatever a manifest names; `latest/chains.parquet`
  is the only chain the page reads; `metrics/daily.parquet` (one row per name per session) is all
  the history panels read. Bump `PREFIX` to change a stored shape.
- **The collector also solves the screens.** Solving every surface took ~12 s at 90 names, too
  slow for a request. So every daily run (and
  `recompute`, and the offline `screens` mode) writes `latest/discover|spreads|expirations.parquet`
  through `snapshot.solve_screens` and records `screens.session` in the manifest. The API
  **refuses tables solved from an older close than `last_session`** (503), because an older
  collector that advanced the chain alone would otherwise leave yesterday's screens under
  today's date. Evaluate still solves its one name from `latest/chains.parquet` (~0.3 s), and the
  filters run server-side (`coverage.py`), so the Screen's expiry choices count only the names on
  screen. On Cloud Run the reader authenticates as the service's own account: there is no secrets
  file there, so no `[gcp_service_account]` table, and the fallback to that account is explicit.
- **`app/logic/options/` is pure and shared**: no Streamlit, no `thetadata`, no `google` —
  `test_architecture.py` forbids both in logic, and checks `collector/*` reaches only
  `app.logic.options` and `app.options_layout`. The collector's image copies just those.
- **Never silently stale.** The page always names the close it shows, and warns once the store is
  more than one session behind (`freshness.sessions_behind`). A name that failed is listed with
  its reason (from the manifest), never dropped.
- **The arithmetic was checked against the page it reproduces** (GLD, 2026-10-23: strike 383 at
  3.23 vs 3.25 shown, every replacement-grid cell within ~0.02 pts, the 398/399 +1 SD spread at
  5.7× with ask 0.35, 1y 25Δ skew percentile 22 vs ~21). The choices that matched, each a test:
  Black-76 on the **put–call-parity forward** (no dividend estimate); **mids** of the 17:15 ET
  NBBO, a zero bid is no market; annualised = cost % × 252 ÷ weekdays; the +k SD target is
  **lognormal** `S·e^{kσ√t}` (linear picks 396/397); a spread's price is the leg-mid difference
  with the bid floored at zero.
- **The "1M" point is the standard monthly**, not the nearest expiry: SPX's 11/23 weekly sat on
  25-point strikes with a stale 7900 quote (15.0/29.3) and read reward:risk 4.1×; the monthly
  reads 1.19×. **Spread legs skip wide quotes** (`wide_quote_ratio`) for the same reason.
- **History is at constant maturity and constant delta**: 30-day by total-variance interpolation
  between the bracketing expiries, 25Δ read against the call wing's own delta. Off the nearest
  listed expiry, a year of "1M IV" partly measures the calendar.
- **Reward:risk** = the +1 SD spread's payout ÷ the payout at a flat smile (ATM IV). It is *not*
  "call IV below ATM": a one-strike spread is a digital, and a smile falling through its strikes
  adds to its price — a slope of −0.10 reads 0.99×. **Two readings, on purpose**: the Screen tab
  prices the two quotes a reader would trade (`screens.sd_spread`); the history and percentiles read
  it off the smile at constant 30 days (`screens.smile_reward_risk`: the digital
  `DF·(N(d2) − F·φ(d1)·√t·∂σ/∂K)` from a line through the 7 tight quotes nearest the target). The
  two-quote version on the drifting 1M monthly ranged GLD 0.33×–4× in a year; the smile version
  0.95×–1.38×. A sustained high (SPY ~2.5× in March 2026's sell-off) is a regime, not a glitch —
  every expiry read the same.
- **A definition change is a recompute, not a refetch**: bump `history.METRICS_VERSION` and run
  `python -m collector.main recompute`, which rebuilds `metrics/daily.parquet` from the stored
  chains (~0.1 s a row). **Buy-vol score**
  = 50 × mean(−z IV, −z(IV − RV21), −z skew) against the name's own year — our definition, shown
  on the page. Both need `min_history_sessions` (120) or show nothing.
- **The free tier releases a session gradually overnight**. Since November 2026 it has finished
  between 01:30 and 07:00 ET, and a run inside that window stores only part of the session. The
  daily run never fetches a stored session again, so `repair --sessions …` fetches the names a
  session is missing (Oct 27 and Nov 6, 2026 were stored with 49 and 26 of 90).
- **One run at a time** (`collector/lease.py`). ThetaData serves one request at a time and every
  run rewrites the manifest, so every mode holds `options/lock.json` (outside `PREFIX`: the limit is
  the account's) for its whole run — an exclusive create, renewed every minute, taken over ten
  minutes after a dead run's last renewal — and writes through `GuardedStore`, which refuses once
  the lease is lost. A backfill holds it for hours, so it **gives way**: a waiting run leaves
  `options/lock-wanted.json`, the backfill releases between name-months, and it **re-reads the
  manifest before every write** (it once wrote back the one it read at its start, rolling a nightly
  run's newest session and screens back).
- **The universe is curated** (`universe.py`: ETFs and single stocks, with theme labels as reading
  aids) — a printed judgement like `FILER_FAMILIES`. A stock must be in `SINGLE_STOCKS` too, or it
  gets no earnings date and the earnings filter passes it silently. `OPTION_ROOTS` maps a name to
  the roots its options list under where they differ: `BRK.B` lists as `BRKB` (and Yahoo wants
  `BRK-B`, so the earnings refresh asks for that); SPX, while it was followed, was `SPXW` and
  `SPX`, merged with `SPXW` winning an overlap. **Tickers are reused**: `SPCX` was a SPAC fund
  until 2026-04-06 and SpaceX from 2026-06-12, so a new name's backfill starts at its listing.
- **Cboe vol indices are not an input.** Yahoo still has history for only VIX/VXN/VXD/GVZ/OVX, and
  GVZ sits 0.5–5.3 vol points above GLD's 30d ATM IV with a moving gap (it includes the wings).
- The repo is public and ThetaData's data is licensed, so **no real chain is committed**:
  `tests/options_fixtures.py` prices a chain from a known smile, which also makes every recovered
  forward, IV and 30-day point checkable exactly.

## Invariants

These are load-bearing across the DCF code. Violating them silently is the failure mode the
architecture exists to prevent.

**No silent fallbacks.** A number that could not be obtained is `Sourced(value=None,
Origin.UNAVAILABLE, reason)` — never a plausible-looking constant, never zero. Callers check
`.ok` rather than coercing `None`. Every displayed figure carries an `Origin`
(reported / derived / estimate / user / unavailable) and, where computed, its formula.

**Missing stays missing.** Series in `financials.py` can contain `NaN`; consumers exclude those
periods and report how many they actually used. Nothing is zero-filled, interpolated, or carried
forward.

**SF2 signs the shares, not the value.** Insider dollar flow is
`transactionshares × transactionpricepershare` — computed in `logic/research/insiders.py` and
nowhere else. `transactionvalue` is an unsigned magnitude *and* null on 48% of rows, so summing
it counts every sale as a purchase and misses half the trades. Nor may every transaction code be
summed: only `P` and `S` are decisions at a market price, and compensation mechanics outnumber
open-market purchases forty-five to one. Both mistakes return the **wrong sign**, not a noisy
number — TSLA's trailing year reads −$5,988m over all codes and +$22,635m over `transactionvalue`
against a true **+$880m**. A third trap sits beside them: SEP's close is split-adjusted and SF2's
filed price never is, so plotting the two together needs `insiders.split_adjusted_price` or every
pre-split marker floats above the line. SF2 is deliberately **not** in `SIGN_FLIPPED` — that set
negates positive magnitudes, and `transactionshares` is already correct.

**SF3 publishes thousands and millions, and never restates for splits.** The client
rescales `units` and `value` to raw shares and raw dollars in `_normalise_sf3`, for the same
reason `_DAILY_MILLIONS` exists. What it cannot fix column-wise is the split: SF3's share
counts are **as filed**, so BlackRock's Tesla position reads 7,720 then 38,020 across the 2020
five-for-one and a quarter-on-quarter change calls that a **+392% purchase**. SF1's `sharesbas`
*is* restated, so an ownership ratio across the two is wrong by the split factor at every
earlier quarter. `sharadar_client.split_factors` is the shared bridge and the two consumers use
it in **opposite directions** — `insiders.split_adjusted_price` divides a filed price,
`holdings.split_adjusted_units` multiplies a filed count.

**Two more things about 13F that are judgements, not data, and are therefore printed.**
`holdings.FILER_FAMILIES` sums a manager's filing entities before anything is ranked: Vanguard's
2026-Q1 reorganisation reads −33% on Apple as one entity and −5.9% as the family, and because
the reorganisation is the manager's, the same phantom sale lands on *every* company it holds.
Deriving that map by matching filer names was tried and measured — 32 flags on Apple, almost all
spurious, pairing BANK OF MARIN with BANK OF AMERICA — so it is curated, incomplete by
construction, and shown to the reader. `holdings.INDEX_MANAGERS` is the same kind of claim.
Alongside them: SF3 `value` is a *market* value, so it moves when the price does and is never
differenced across quarters; reported ownership above 100% is real (Etsy reads 114.3%) and is
never clamped; and a gap in a holder's series is **not** a zero, which is the deliberate inverse
of `insiders.periodic_flow`.

**Costs are negative** everywhere downstream of ingest. `sharadar_client.SIGN_FLIPPED` negates
the indicators Sharadar publishes as positive magnitudes (`cor`, `opex`, `sgna`, `rnd`, `taxexp`,
`intexp`). `capex` is *not* in that set — Sharadar already signs it negative.

**No hardcoded Sharadar column names at call sites.** Every line item is declared as a `Concept`
in `schema.py` with an ordered alias list and resolved against the columns that actually came
back. A required concept that resolves to nothing raises `SchemaResolutionError` (blocking); an
optional one is recorded as "not reported".

**Severity has three tiers.** `BLOCKING` (computation impossible — nothing renders; reserved for
absent data, never for data that merely looks wrong), `CRITICAL` (red persistent banner),
`WARNING` (amber). Checks aggregate: six failures produce six issues. A check whose inputs are
absent reports *not evaluated* rather than passing.

**Every tolerance lives in a `Tolerances` dataclass owned by its consumer** — `dcf/validation.py`
`Tolerances` for the model, `logic/statements.py` `StatementTolerances` for the statements tab,
and `ValuationTolerances`/`GrowthTolerances` for the two ⚖️ sub-tabs. Never a bare literal at a
comparison site. They are deliberately not shared: the model's thresholds gate whether a
valuation is trustworthy, the statement ones gate whether a display row grows a marker, and the
comparison ones gate whether a peer group is big enough to have a median. The last two are the
sharpest case — `min_observations` counts trading sessions for the multiples and filed quarters
for the growth rates, so one shared number would be wrong for one of them by a factor of sixty.

**Banks and insurers are rejected** before anything is computed, on SIC code (6000–6999), not on
Sharadar's vendor `sector` label. That gate is the DCF's, not the research page's:
`logic/research/dataset.py` deliberately bypasses it so the page renders JPM, and the ⚖️
Valuation tab compares banks on P/B, which is the point of the metric. Peer *grouping*, unlike
the gate, uses the vendor `industry` label — it is the classification a reader recognises, and
the two disagree often enough that which one is in use has to be a deliberate choice.

**Non-USD reporters are converted, once, at spot.** Drivers come from the reporting-currency
history, so growth and margins carry no FX noise; `BasePeriod.converted` puts every monetary
line of the base period into USD at the spot rate `market_data.fetch_fx_rate` returns, and
everything downstream — the projection, the bridge, the WACC's debt, both base-period FCFF
builds — is in USD. History tables stay in the reporting currency and say so. Two facts about
the data, both confirmed live: **Sharadar's `fxusd` is units of the reporting currency per
dollar** (TSM ≈ 31, EUR ≈ 0.85 — the inverse of the spot quote; `revenueusd = revenue ÷ fxusd`),
and **DAILY `ev`/`marketcap` are in USD**, so the historical exit multiple divides USD EV by
EBITDA converted at each period's own `fxusd`. No spot rate is still BLOCKING
(`validation.UnsupportedCurrencyError`); a rate is never substituted. ADRs keep `sharefactor` on
**every** share count compared with the price — the weighted counts and EPS are per underlying
share.

**One BLOCKING check rests on plausibility, not absence:** `check_currency_depreciation` blocks
when the reporting currency has lost more than 10% a year against the dollar (measured from
Sharadar's own `fxusd`, so no currency list is hardcoded), because the spot-rate assumption is
then not credible. Every other plausibility check stops at CRITICAL.

**SBC is expensed once, inside EBIT, and cannot be added back.** The margin driver is GAAP
operating income, which already deducts it; the add-back option was removed rather than left to
double as a way to treat equity pay as free.

**The two terminal values are discounted differently on purpose.** Gordon is a perpetuity of
flows on the forecast's own convention, so it takes the final row's (mid-year) exponent; an exit
multiple is a sale at the *end* of year N. Both sit at the base period's end rolled forward by
`elapsed` — the forecast is anchored to the base period, never to the next fiscal year end,
which made the value jump overnight when a fiscal year closed. **Gordon is the default
headline** (`model.DEFAULT_HEADLINE_METHOD`, the owner's choice, reversing round 2's exit
multiple). The exit multiple is always shown beside it, and the reverse DCF solves both.

**The terminal RONIC defaults to the company's own return on invested capital**
(`financials.historical_roic`): NOPAT ÷ average invested capital, where invested capital is
equity + debt + minority interest − cash − STI (the same cash the bridge adds back; goodwill
stays in, so bought growth is charged for), median of the last five fiscal years. At or below the
WACC it takes the WACC and says so — value-destroying growth is a claim to make by hand — and it
is passed as `ronic=None` then, so the perpetuity keeps tracking the WACC across the sensitivity
grid. Its reach is **bounded**: the Gordon value is NOPAT × (1 − g ÷ RONIC) ÷ (WACC − g), so even
an infinite RONIC lifts it only by WACC ÷ (WACC − g), about +36% at a 9.5% WACC. Measured live,
moving from RONIC = WACC lifted the headline 7–16% where ROIC clears the WACC (MSFT 43%, AAPL
89%) and left most of the book 14–70% below its price: the rest of the gap is growth for longer
than the ten-year fade, or the discount rate — not the terminal return.

**The terminal-value-share warning is gone, on purpose.** Under a ten-year exit-multiple model
it fired on every company and said nothing. The share is still shown beside each method; the
check that replaced it, "Explicit forecast adds value", fires only when the forecast years are
worth less than zero in total. Upside beyond `Tolerances.extreme_upside_critical` (+200%) is
CRITICAL, so an implausible headline is marked red rather than shown quietly.

**The cash-flow tie scales its gap by the largest of the two sides *and* |CFO|**, not by the
change in cash alone, which is often small beside the flows that net to it. It handles two
Sharadar feed defects, both confirmed against SEC filings, rather than reporting them as breaks:
a **net change in cash of exactly 0** where the filing reports one (TSM) is treated as missing
and the balance-sheet cash change compared instead; and a **zero-filled FX line at a foreign
filer** (SAP), where the net change agrees with the balance sheet, is read as the undisclosed FX
effect and raised at WARNING with the implied amount. The thresholds
(`cash_flow_residual_warning`, `implied_fx_max`) are in `Tolerances`; a domestic filer's gap stays
CRITICAL. **Known data caveat:** SAP's Sharadar capex runs 11–16% below its filed
"PP&E, intangibles and other non-current assets" line (it looks like PP&E only) — about 0.3% of
revenue, left as is.

**Year-1 growth defaults to the lower of TTM year-over-year and the five-year CAGR**
(`financials.lower_of_growth`). The rate is held for several years and faded over ten, so it
compounds hard: a trailing burst far above the five-year record (NVDA 83% vs 67%, AVGO 49% vs 22%)
took NVDA's year-10 revenue to 70× today's and its headline to 42× its price. The lower rate keeps
a recent slowdown and refuses to extrapolate a recent burst; the trailing rate stays selectable,
and the checks read the rate actually used (`_effective_growth_basis`).

**The forecast starts from today.** The margin, capex % and D&A % default to the base period's
own TTM ratios (`financials.base_period_driver`), because a TTM margin already reflects today's
depreciation and a hyperscaler's AI-era capex is roughly double its five-year average. Each can
move linearly to a year-N target. **Capex has one by default: its own five-year average**
(`financials.normal_capex`), symmetric, so neither a build-out nor a trough is held forever —
held flat, AMZN's 21.8% TTM capex made every forecast year's FCFF negative, and TSM, off its peak,
reverts *up*. D&A is deliberately not moved with it: the margin already deducts today's D&A, so
holding both holds EBITDA's share, and moving D&A alone moves cash flow by the whole change rather
than by its tax shield. Under the Gordon headline the normalisation moves only the explicit years
(and the EBITA add-back): the terminal year reinvests g ÷ RONIC of NOPAT whatever year N's capex
was. The capex ÷ D&A check therefore fires only below 1.0 *and* below the company's own ratio —
capex at or above D&A shrinks nothing, and against a build-out's history the normalised path
would otherwise read as under-investment. Working capital is **NWC as a % of revenue** (a level
ratio, default five-year average; ΔNWC = ratio × Δrevenue), never ΔNWC ÷ Δrevenue, which divided
by a near-zero change in a flat year — DHR's FY2024 read 2,453% and wrecked its whole valuation.

**The terminal year adds back amortisation (EBITA proxy, on by default):**
`NOPAT_{N+1} = (EBIT_{N+1} + max(D&A_N − capex_N, 0) × (1 + g)) × (1 − t)`. Without it an acquirer
whose D&A is mostly acquired-intangible amortisation (AVGO) saw FCFF halve between year N and
N + 1. It moves the Gordon value and the exit row's implied figures, never the exit value.

**The credit-spread table is hardcoded and dated, on purpose — refresh it each January.**
`wacc.SYNTHETIC_RATINGS` is Damodaran's large non-financial-firm table. To refresh: open
<https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.html>, check the "Data used
is as of" line, copy the fifteen spreads (the coverage bands rarely change), and update
`SYNTHETIC_RATINGS_AS_OF` and `SYNTHETIC_RATINGS_AS_OF_DATE` to match; `test_wacc.py` pins the
values, so update it in the same change. "Credit spread table is current" warns once the table is
more than `Tolerances.spread_table_max_age_months` (15) old, but only when the table actually set
the cost of debt. Where the base period's interest is zero beside debt or not reported at all, the
latest of the last three fiscal years that disclosed it supplies the coverage.

**Failed market-data fetches are never cached.** Each cached function in
`app/data/market_data.py` raises `FetchFailed` (`app.data.cache` does not store an exception) and
an uncached wrapper turns it into `Sourced.unavailable`. Returning the failure from the cached
function cached a transient outage for a day. `options_store.OptionsStoreError` works the same way.

**The API key never appears in an exception, log line, or export.** Nasdaq only accepts it as a
query parameter, so errors raised from `sharadar_client` carry the table, ticker, and status —
never the params or full URL. A secrets file that will not parse is reported by its path alone,
because a TOML parser's message can quote the line it stopped on.

Two more Sharadar quirks the client absorbs: `DAILY` reports market cap and EV **in millions**
while `SF1` reports them raw (a missed conversion passes every sanity check by a factor of a
million), and rows arrive **unsorted** with truncation at 10,000 rows behind an opaque
`meta.next_cursor_id` cursor.

One cache rule it works around too: **`app.data.cache` keys on the arguments as passed and does
not apply defaults** (the rule `st.cache_data` had, kept when the cache replaced it), so
`f(t, annual=True)` and `f(t, annual=True, ttm=False)` are separate entries for the same request;
a positional argument and the same one by keyword are one entry. `fetch_statements` is therefore a plain wrapper that
forwards **positionally** to the cached `_cached_statements`; the cache never sits directly on a
function with keyword defaults. Before that, one search fetched `MRY` and `MRQ` twice each,
because the research page spelled all three keywords and the DCF spelled two.

## UI conventions

### Lazy tabs

**Only the open tab runs.** Stock Research's strip is Mantine `Tabs` with `keepMounted={false}`,
so only the open tab's component is mounted and only it queries the API; the ⚖️ views and the 💰
DCF's parts are chosen the same way, each by a URL parameter.

Everything about loading follows from it:

- **Search fetches only what 📊 Summary draws** (`load_research`, ~9 requests, ~4s cold). Every
  other tab fetches its own **the first time it is opened**, through its own route and its own
  loader in `dataset.py`. The data layer's cache is keyed on the company, so nothing needs
  clearing when the ticker moves.
- **📊 Summary's one fetch of its own is the chart's DCF line** (`/dcf-line`), and it never holds
  up the candles: they draw first and the line joins them once it is valued. Each filing's value
  is cached on its own (`_dcf_point_cached` in `app/api/research.py`), so a window pays only for
  the filings it adds — 1Y is six runs, 5Y twenty-two — and the line reaches back at most
  `DCF_LINE_MAX_YEARS` (5). Measured cold on AAPL: candles 2.7s, a 5Y line 8.1s; once a company's
  as-reported statements are cached, a further filing costs ~0.1s.
- **The requests inside a loader go out together**, through `app/data/parallel.gather`, which
  runs them in a thread pool and re-raises the first failure in submission order. A tolerated
  fetch is wrapped in `_tolerant` *before* it is handed over. The ticker's `TICKERS` record goes
  first and alone, because the statement fetches read it through the cache, which computes one
  key once: concurrent misses would sit in worker threads queued behind its one request.
- **One cache entry per underlying request, shared across tabs.** `_fetch_statements` reads the
  *cached* `fetch_ticker_meta`, price routing reads the cached `fetch_price_tables`, and
  `fetch_ohlcv`, the DCF's `fetch_prices` and the benchmark's `fetch_index_prices` all slice one
  cached full-history `_price_rows` — the last so a beta at each of twenty filings is one request. A
  bare `_fetch_*` call inside another fetcher silently re-requests: that was six of the 23
  requests a search used to make. `test_stock_research.py` counts requests per table to keep it
  that way.
- **The selected tab is a viewing preference**, like the price window: a search keeps the URL's
  other parameters, so a reader on ⚖️ Relative Valuation who searches another ticker lands on its
  Relative Valuation.

Measured on the live API: search 27.6s → 3.8s; Financials ~1.2s and Insiders ~1.4s on first
open; the DCF makes **no** requests once Financials has been opened; Valuation's peer universe
~6s for the first company of the day and nothing after; 🏛️ Holdings 12–24s — still the slowest,
because its five requests form a forced chain (summary → quarter's detail → top holders' history)
and SF3 answers slowly.

The peer universe costs a fixed amount, not one request per comparable. `DAILY` filtered on a
date with no ticker returns every listed company in a single page — measured at 5,446 rows in
~1.1s. All three universe fetches (`fetch_universe_daily`, `fetch_universe_fundamentals`,
`fetch_ticker_directory`) are keyed on a date alone, never on a ticker, and `load_peers` is keyed
on the company's last market session, so **one cache entry serves every company searched that
day**. Keep it that way: a universe fetch that took a ticker would turn a fixed cost back into a
per-company one.

`fetch_universe_fundamentals` reaches back **eighteen months** (`UNIVERSE_LOOKBACK_DAYS`) and
returns *every* period in the window rather than one row per ticker, because the growth
comparison needs both ends of a year-on-year. 36,199 rows over four pages in ~4.8s, and it is a
superset of what the multiples need, so widening it replaced a second fetch rather than adding
one. The only collapse it still does is the **restatement**: two filings for the same
`calendardate` keep the later `datekey`.

### The DCF tab

The tab is one `POST /api/research/{ticker}/dcf` carrying only the reader's overrides, answered
with the resolved panel and the valued run together (warm, ~0.2 s), plus `/dcf/what-if` and
`/dcf/export`; the export filters the reader's acknowledgements to the run's own issues.

The DCF tab has **no Compute gate**. It loads and values whatever ticker the page has searched,
as soon as the tab is opened, and an assumption change is applied on the spot. Nothing on that tab
is ever drawn as current from a run that does not match the controls on screen: the overrides are
the query key, so while the next run loads the last one stays on screen dimmed, headline included,
and a refused input (a 422) leaves it dimmed beside an Undo. Statements are always the
**restated** (`MR`) series; `dcf_model.load_company` still takes `asreported`, but the tab does
not offer the choice. The overrides and acknowledgements live in the tab's component, keyed by
ticker, so a new search starts from the defaults.

Severity banners sit above everything else on the tab. The governing rule for that tab: it must be
impossible to look at the output and not notice something is wrong.

**The untouched assumption set is `inputs.resolve` with no overrides** — the API's panel and the
summary chart's DCF line (`inputs.default_value`, which resolves with `strict=True`) both call it,
so there is one set of defaults; `test_api_dcf.py` asserts an untouched request returns
`default_value`'s headline.

**A past value is struck as the company stood that day** (`inputs.default_value_on(ticker, as_of)`).
`load_company(asreported=True, as_of=...)` keeps only the rows filed by then — `AR` only, because
under `MR` Sharadar's `datekey` is the report period and a restatement moves it — with that day's
close, 10-year yield (`fetch_10y_treasury_rate_on`), exchange rate (`fetch_fx_rate_on`) and a beta
regressed to it. Damodaran's ERP and the spread table have no history here, so today's stand in,
and the chart says so. Sharadar restates **share counts for splits even under `AR`** (NVDA's
October 2023 filing reads 24.7bn, the post-split count), so a past value per share is already on
the basis of the split-adjusted candles — but the *traded* close before a split is not, so a past
dataset quotes the split-adjusted close instead; the traded one overstated NVDA's pre-split market
cap tenfold. Where the tab would ask the reader for a number (no rate, no beta, a driver with no
history) or would block, the point is a gap (`DefaultUnavailable`), never a placeholder; a failed
fetch raises instead (`transient=True`), so the cache in front of it never remembers an outage.

## Tests

No test touches the network. `tests/conftest.py` builds payloads mirroring the real
`SHARADAR/SF1` datatable envelope, deliberately reproducing the quirks the client exists to
absorb — positive-magnitude expense lines, negative capex, unsorted rows, and column order that
differs from the layout declaration so the tests prove the client zips on names. The fixture
company balances, ties, and reconciles exactly, so **any check firing against it is firing
wrongly**.

`tests/test_golden.py` snapshots one full model run. If a refactor moves those numbers, either
the refactor is wrong or the snapshot needs updating with a stated reason.

Tests take the network away by patching **`app.data.sharadar.transport._get`** — every table
module calls it through the module attribute, so that one patch reaches all of them. An autouse
fixture in `conftest.py` calls `cache.clear_all()` before every test, because the private fetchers
read each other through the cached public ones and a company cached by one test would otherwise
answer for the next. Another points `settings.SECRETS_FILE` at nothing, so no test can pass on
the strength of a developer's real key; a test about the file writes its own. A test that
needs a client internal (`_fetch_statements`, `_sf1_frames`, …) imports the submodule that owns
it as `sh_<module>`; `sharadar_client` re-exports only the public API.

API tests (`test_api_*.py`) call the routes through FastAPI's `TestClient`, over the fixture
company or with a loader monkeypatched, and parse every body strictly, so a NaN or a missing field
fails the test rather than reaching the page as `null`.

The ⚖️ Relative Valuation tab is tested per-module, which is the layout the rest should move to —
`test_research_valuation.py` and `test_research_growth.py`. Both carry the `research_` prefix
because `test_valuation.py` was already taken by the **DCF** engine's `logic/dcf/valuation.py`:
two unrelated modules share the name `valuation.py`, and the test files have to disambiguate what
the modules themselves do not.

👤 Insiders and 🏛️ Holdings follow the same layout (`test_research_insiders.py`,
`test_research_holdings.py`). The fixture behind the first is a sixteen-row company in
`conftest.py` priced at $100, built so that the naive total is **negative** and the correct one
**positive** — that one assertion is the file's reason for existing. Every other row breaks
exactly one thing: a restatement that duplicates a trade, a holdings-only pair carrying a bare
`N` and a bare `D`, a quarter-straddling filing, an unpriced grant, a stale owner, a reported
value that disagrees with shares times price, and a transaction code nobody has classified.

The 🏛️ Holdings fixture is an eleven-holder register over six quarters with a 2:1 split in the
middle and Vanguard's **real** investorids for the two entities that reorganise — so the shipped
`FILER_FAMILIES` and `INDEX_MANAGERS` constants are what the tests exercise, not test-only
stand-ins. Its headline assertions are that a split reads as no purchase, that the entity split
reads −67% alone and −3.3% as a family, and that a holder's missing quarters stay missing.

One older file still maps onto the *pre-package* layout: `test_stock_research.py` covers all of
`app/logic/research/` except the comparison and ownership tabs. Split it along its existing section
headers when it is next touched substantially.

`tests/conftest.py` also carries a miniature eleven-ticker universe for the peer medians, two
periods per ticker so the year-on-year pairing has both ends. It is
sized so every median is checkable by hand and every exclusion rule has something to exclude: a
loss-maker **whose loss narrows** (the case that prints backwards without the base gate), a
non-payer, a company with zero capex, a delisted company DAILY still returns, a ticker that filed
twice for the same period, and an industry too small to have a median. `stub_universe` routes by
table name so tests call the real fetchers — the millions conversion and the restatement dedup
then run for real rather than being reimplemented in the fixture and proved against themselves.
