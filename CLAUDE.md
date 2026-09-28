# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements-dev.txt          # requirements.txt + pytest; deployed app installs only requirements.txt
streamlit run streamlit_app.py               # run the app
python -m pytest tests/ -q                   # full suite (~1,135 tests, no network; CI runs it too)
python -m pytest tests/test_valuation.py -q  # one file
python -m pytest tests/test_wacc.py::test_market_value_of_equity -q  # one test
python -m pytest tests/ -k "fcff" -q         # by name
```

There is no linter, formatter, type checker, or pytest config file in this repo.

`.claude/launch.json` defines a `streamlit` configuration on port 8502, with `autoPort` so a
second session falls back to a free port instead of colliding — start the app for preview
with that rather than Bash.

`SHARADAR_API_KEY` (fundamentals, prices) comes from `.streamlit/secrets.toml`, which is
gitignored. Treasury yields come from Yahoo Finance via `yfinance` (`^IRX` short rate, `^TNX`
10-year) and need no key. Tests never need either.

## Architecture

Streamlit multi-page app. `streamlit_app.py` sets page config, loads `assets/style.css`, and
registers pages in `st.navigation([...])`. Each feature is three layers:

- `app/views/<feature>.py` — a `render_page()` function; all Streamlit calls live here
- `app/logic/<feature>.py` — the math, no Streamlit
- `app/data/` — every external API call, so caching and keys stay in one place

A feature earns a **package** in place of either module when it grows several independently
testable parts — `app/logic/dcf/` and both halves of `app/views/research/` are the cases.
Reach for it when the single module would otherwise hold pieces that have no reason to know
about each other, not at a line count.

`app/provenance.py` sits at the `app/` root because both the data and logic layers construct its
types and neither may import the other. `app/formatting.py` sits beside it for the same reason:
the logic layer builds display strings with it (`StatTile`) and every view formats with it.
`app/views/components.py` holds the few Streamlit pieces more than one page draws identically.

**The layering is enforced, not just described.** `tests/test_architecture.py` parses `app/` into
an import graph and fails if a logic module reaches Streamlit, `requests` or `yfinance` — even
transitively, which is how it actually breaks — other than the two orchestrators
(`logic/research/dataset.py`, `logic/dcf/model.py`); if data imports logic or views; if logic
imports views; if the DCF arithmetic reaches `app.data` at all; or if any module uses another's
`_private` name. A logic module that needs a Sharadar constant imports
`app.data.sharadar.vocabulary`, never `sharadar_client`.

To add a page: create the logic module, create the view with `render_page()`, register it in
`st.navigation`, and fetch through `app/data/` — `sharadar_client` or `market_data` — from the
view, never from the logic module.

To add a **tab** to Stock Research: create `app/logic/research/<tab>.py` for the arithmetic and
`app/views/research/<tab>.py` with a `render` function and its own CSS, then add the label to
`_TAB_ORDER` and dispatch it in `views/research/__init__.py` — a test fails if a label has no
body. Only built tabs go in the strip: there are no "not built yet" stubs.

A tab that compares metrics against peers and against history should be a **sub-tab of ⚖️
Valuation** rather than a new top-level one: declare its metrics and its own usable/base rule,
and reuse `logic/research/comparison.py` and `views/research/valuation/table.py` for everything
else. That is what Growth is, and it is why there is no top-level 🌱 Growth tab.

### The Stock Research page

The one feature that is a package on both layers rather than a pair of modules, because it
carries six tabs — one of which is itself a package, holding two sub-tabs:

```
app/views/research/            __init__.py = the shell; one module per tab beside it
app/views/research/valuation/  the ⚖️ tab: shell + table + multiples + growth
app/logic/research/            comparison ← summary, valuation, growth,
                               insiders, holdings ← dataset
```

`views/research/__init__.py` owns the ticker box, the search gate, the session snapshot and
`_TAB_ORDER` — **the one place tab order is decided** — and draws the strip with
`st.tabs(..., key="research_tab", on_change="rerun")`, rendering **only the tab whose `.open` is
true** (see *Lazy tabs* under UI conventions). Each tab module exposes a `render` function taking
exactly what that tab needs: the snapshot for the ones that read it, the bare ticker for `dcf`,
which loads through the DCF engine instead. There is no shared signature, because the dispatch is
explicit and a uniform one would hide that the DCF reads nothing from the snapshot. Each tab
also owns its CSS, so the rules travel with the markup they style.

On the logic side the dependencies run one way — `comparison` is shared, each
tab module owns its arithmetic, and `dataset` sits on top because it holds every network-touching
loader: `load_research` (the snapshot — only what 📊 Summary draws) and one loader per tab beside
it (`load_financials`, `load_insiders`, `load_peers`, `load_holdings`). Its calls into
`summary.build_key_stats` and `comparison.build_peer_universe` are the only upward edges, and both
exist for the same reason: the result is cached per company per day, so anything a tab would
otherwise recompute on every rerun is precomputed there.

`logic/research/comparison.py` holds what the two ⚖️ sub-tabs genuinely share — the `Comparison`
record, the median-and-percentile summary, the peer universe join, the industry lookup. What it
deliberately does **not** hold is either tab's notion of a *usable* number or its `Tolerances`,
because on both points the two are opposites; see the ⚖️ section below.

Two modules deliberately stay outside both packages. `app/logic/statements.py` declares the four
SF1 statement layouts (sections, rows, subtotals, reconciliations) and evaluates them into a
rendered table; the financials tab is its only consumer today, and it sits outside so a second
one needs no move. `app/logic/dcf/` stays independently reusable so batch screening remains an
extension rather than a rewrite. Both are pure.

**Statements are shown in exactly one place — the 📄 Financials tab.** The DCF tab used to carry
a raw viewer of its own and now links there instead; it keeps only `_render_filing_provenance`,
the per-year SEC filing links, which Financials does not show. Do not add a second statement
table anywhere: the removed one scaled every row by the units divisor, which turned EPS and
margins into `0.00`, and `logic/statements.py` carries a `Unit` enum precisely so a layout
cannot repeat that.

`views/research/dcf/` is a package: `__init__` is the shell (the load, the two caches, the
fingerprint, the session lifecycle and the draw order), beside `assumptions` (every input, one
function per block of the panel), `checks` (banners, validation panel), `history`, `results`
(header, projection, valuation), `implied` (the reverse DCF), `download` and `common`. The names
avoid `valuation`, `reverse` and `export` on purpose — `app/logic/dcf/` already has modules by
those names. Functions that cross modules are public inside the package; the rest stay private.

### The ⚖️ Valuation tab and its two sub-tabs

`Valuation` asks what the market pays for the company; `Growth` asks how fast the thing being
paid for moves. Both answer per metric in the same three columns — current, the company's own 5Y
median, the industry median — over one peer universe, and a metric that cannot fill all three is
not shown. The package shell injects the one `.val-*` stylesheet and draws the sub-tab strip;
`table.py` owns the grid and the four cells the bodies render identically. The shell loads the
peer universe once (`dataset.load_peers`) and hands it to both bodies, which fetch nothing. **This
is the only nested `st.tabs` in the app**, and it is stateful like the outer strip, so only the
open sub-tab's body is built.

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
| `model.py` | Orchestration |
| `export.py` | Multi-sheet `.xlsx` and CSV export |

`wacc.py`, `valuation.py`, and `reverse.py` import no Streamlit, make no network calls, and hold
no global state. Keep them that way — it is what makes batch screening an extension rather than a
rewrite. `reverse.py` takes a *callable* from driver value to value per share; putting the solver
inside `valuation.py` would create an import cycle with `model.py`.

`model.load_company()` is the only network-touching entry point and is called once per ticker;
`model.run_model()` runs on every assumption change. Never move a fetch into `run_model`.

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
which made the value jump overnight when a fiscal year closed. The exit multiple is the default
headline; Gordon at RONIC = WACC is always shown beside it, and the reverse DCF solves both.

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
move linearly to an optional year-N target. Working capital is **NWC as a % of revenue** (a level
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

**Failed market-data fetches are never cached.** Each `st.cache_data` function in
`app/data/market_data.py` raises `FetchFailed` (Streamlit does not cache exceptions) and an
uncached wrapper turns it into `Sourced.unavailable`. Returning the failure from the cached
function cached a transient outage for a day.

**The API key never appears in an exception, log line, or export.** Nasdaq only accepts it as a
query parameter, so errors raised from `sharadar_client` carry the table, ticker, and status —
never the params or full URL.

Two more Sharadar quirks the client absorbs: `DAILY` reports market cap and EV **in millions**
while `SF1` reports them raw (a missed conversion passes every sanity check by a factor of a
million), and rows arrive **unsorted** with truncation at 10,000 rows behind an opaque
`meta.next_cursor_id` cursor.

One Streamlit quirk it absorbs too: **`st.cache_data` keys on the arguments as written at the
call site and does not apply defaults**, so `f(t, annual=True)` and `f(t, annual=True, ttm=False)`
are separate entries for the same request. `fetch_statements` is therefore a plain wrapper that
forwards **positionally** to the cached `_cached_statements`; the cache never sits directly on a
function with keyword defaults. Before that, one search fetched `MRY` and `MRQ` twice each,
because the research page spelled all three keywords and the DCF spelled two.

## UI conventions

### Lazy tabs

**Only the open tab runs.** Both tab strips are `st.tabs(..., key=..., on_change="rerun")`:
switching tabs reruns the script, and each `TabContainer.open` says which body is on screen, so
the shell skips every other one. That needs **Streamlit ≥ 1.57**, pinned in `requirements.txt`
— an older build returns `None` from `.open` and the page would render no tab at all.

Everything about loading follows from it:

- **Search fetches only what 📊 Summary draws** (`load_research`, ~9 requests, ~4s cold). Every
  other tab fetches its own **the first time it is opened**, through a `_load_cached(ticker,
  as_of)` wrapper in its own module and `components.load_tab`, which draws the same spinner and
  the same blocking notice on failure for every tab. No tab has a load button, and none keeps its
  payload in session state — the cache is keyed on the company, so there is nothing to clear
  when the ticker moves.
- **The requests inside a loader go out together**, through `app/data/parallel.gather`, which
  carries the Streamlit run context into its worker threads and re-raises the first failure in
  submission order. A tolerated fetch is wrapped in `_tolerant` *before* it is handed over. The
  ticker's `TICKERS` record goes first and alone, because the statement fetches read it through
  the cache and concurrent misses would each pay for it.
- **One cache entry per underlying request, shared across tabs.** `_fetch_statements` reads the
  *cached* `fetch_ticker_meta`, price routing reads the cached `fetch_price_tables`, and
  `fetch_ohlcv` and the DCF's `fetch_prices` both slice one cached full-history `_price_rows`. A
  bare `_fetch_*` call inside another fetcher silently re-requests: that was six of the 23
  requests a search used to make. `test_stock_research.py` counts requests per table to keep it
  that way.
- **The selected tab is a viewing preference**, like the price window: the page's
  `_session_reset` leaves `research_tab` alone, so a reader on ⚖️ Valuation who searches another
  ticker lands on its Valuation.

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

The DCF tab has **no Compute gate**. It loads and values whatever ticker the page has searched,
as soon as the tab is opened, and an assumption change is applied on the spot. Nothing on that tab
is ever drawn from a run that does not match the controls on screen — which is why the staleness
notice it used to carry is gone rather than merely unused. Statements are always the **restated**
(`MR`) series; `dcf_model.load_company` still takes `asreported`, but the page does not offer the
choice.

Two caches are what make that affordable, and both are load-bearing:

- `_load_company_cached` — the fetch and normalisation, per ticker.
- `_needs_run` against `_assumption_fingerprint` — the model, the checks and the workbook, per
  distinct assumption set. A field missing from that tuple is a **correctness** bug: the previous
  run's numbers would stay on screen looking current, with nothing to say so.

Session state is keyed `dcf_*`. The tab clears its own acknowledgements and assumption overrides
via `_session_reset` when it notices the page's ticker has moved; the research page's Search does
**not** reach across and do it, so there is one place that decides when a model is stale.

Severity banners sit above everything and do not collapse. The governing rule for that tab: it
must be impossible to look at the output and not notice something is wrong.

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
fixture in `conftest.py` clears `st.cache_data` before every test, because the private fetchers
read each other through the cached public ones and a company cached by one test would otherwise
answer for the next. A test that
needs a client internal (`_fetch_statements`, `_sf1_frames`, …) imports the submodule that owns
it as `sh_<module>`; `sharadar_client` re-exports only the public API.

View-level tests (`test_page_controls.py`, `test_acknowledgements.py`, `test_research_page_controls.py`)
import the view modules directly and monkeypatch `page.st.session_state` with a dict subclass,
since there is no Streamlit runtime.

The ⚖️ Valuation tab is tested per-module, which is the layout the rest should move to —
`test_research_valuation.py` and `test_research_growth.py` for the logic, and
`..._controls.py` for each view. All six carry the `research_` prefix because
`test_valuation.py` was already taken by the **DCF** engine's `logic/dcf/valuation.py`: two
unrelated modules share the name `valuation.py`, and the test files have to disambiguate what the
modules themselves do not. `test_research_valuation_controls.py` covers the package shell and the
shared grid as well as the Valuation body; `test_research_growth_controls.py` covers only what
Growth words differently.

👤 Insiders and 🏛️ Holdings follow the same split. The fixture behind it is a sixteen-row company in
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

The two older files still map onto the *pre-package* layout:
`test_stock_research.py` covers all of `app/logic/research/` except the comparison tabs, and
`test_research_page_controls.py` covers the shell plus three tab modules. Split them along their
existing section headers when either is next touched substantially.

`tests/conftest.py` also carries a miniature eleven-ticker universe for the peer medians, two
periods per ticker so the year-on-year pairing has both ends. It is
sized so every median is checkable by hand and every exclusion rule has something to exclude: a
loss-maker **whose loss narrows** (the case that prints backwards without the base gate), a
non-payer, a company with zero capex, a delisted company DAILY still returns, a ticker that filed
twice for the same period, and an industry too small to have a median. `stub_universe` routes by
table name so tests call the real fetchers — the millions conversion and the restatement dedup
then run for real rather than being reimplemented in the fixture and proved against themselves.
