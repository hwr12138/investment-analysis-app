# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements-dev.txt          # requirements.txt + pytest; deployed app installs only requirements.txt
streamlit run streamlit_app.py               # run the app
python -m pytest tests/ -q                   # full suite (~350 tests, no network)
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
types and neither may import the other.

To add a page: create the logic module, create the view with `render_page()`, register it in
`st.navigation`, and reuse `app/data/market_data.py` for fetching.

To add a **tab** to Stock Research: create `app/logic/research/<tab>.py` for the arithmetic and
`app/views/research/<tab>.py` with a `render` function and its own CSS, then add the label to
`_TAB_ORDER` and dispatch it in `views/research/__init__.py`. Remove its entry from
`placeholders.PLACEHOLDERS` in the same commit — a test fails if the two disagree.

### The Stock Research page

The one feature that is a package on both layers rather than a pair of modules, because it
carries ten tabs:

```
app/views/research/     __init__.py = the shell; one module per tab beside it
app/logic/research/     formatting ← summary, valuation ← dataset
```

`views/research/__init__.py` owns the ticker box, the search gate, the session snapshot and
`_TAB_ORDER` — **the one place tab order is decided**. Each tab module exposes a `render`
function taking exactly what that tab needs: the snapshot for the ones that read it, the bare
ticker for `dcf`, which loads its own. There is no shared signature, because the dispatch is
explicit and a uniform one would hide that the DCF reads nothing from the snapshot. Each tab
also owns its CSS, so the rules travel with the markup they style.

On the logic side the dependencies run one way — `formatting` is shared by every tab, each tab
module owns its arithmetic, and `dataset` sits on top because it assembles the snapshot. Its
calls into `summary.build_key_stats` and `valuation.build_peer_universe` are the only upward
edges, and both exist for the same reason: the snapshot is assembled once per search, so
anything the tabs would otherwise recompute on every rerun is precomputed there.

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

`views/research/dcf.py` is by far the largest tab body and is the module to split next if it
grows again — assumptions, valuation, reverse and export are each separable.

**The ⚖️ Valuation tab is trailing-only, and deliberately so.** Every metric is answered three
ways — current, the company's own 5Y median, the industry median — and a metric that cannot fill
all three columns is not shown. Forward multiples were dropped for exactly that reason: yfinance
supplies forward EPS and revenue only per-ticker by scrape, so forward *industry* medians would
cost thousands of scrapes, and a forward *5Y history* needs point-in-time consensus no source
here has. That tab makes no yfinance call at all. Two rules govern the numbers: **only positive
readings count** (a negative P/E is a loss, not a cheap stock; a zero yield is a non-payer), and
the exclusion is applied **per metric, not per company** — a loss-maker still has an EV/Sales and
still belongs in that median. **Medians, never means**, because a multiple whose denominator
approaches zero prints in the thousands.

### The DCF engine

`app/logic/dcf/` is a package rather than the usual single logic module so schema resolution,
normalisation, validation, and arithmetic are each independently testable. The pipeline is
**fetch → resolve → normalise → validate → value**, assembled in `model.py`:

| Module | Responsibility |
|---|---|
| `app/data/sharadar_client.py` | Nasdaq Data Link auth, retry, pagination, caching, datatable parsing |
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
`logic/research/valuation.py` `ValuationTolerances` for the peer medians. Never a bare literal at
a comparison site. The three are deliberately not shared: the model's thresholds gate whether a
valuation is trustworthy, the statement ones gate whether a display row grows a marker, and the
valuation ones gate whether a peer group is big enough to have a median. Coupling them would let
a change made for one reason silently move another.

**Banks and insurers are rejected** before anything is computed, on SIC code (6000–6999), not on
Sharadar's vendor `sector` label. That gate is the DCF's, not the research page's:
`logic/research/dataset.py` deliberately bypasses it so the page renders JPM, and the ⚖️
Valuation tab compares banks on P/B, which is the point of the metric. Peer *grouping*, unlike
the gate, uses the vendor `industry` label — it is the classification a reader recognises, and
the two disagree often enough that which one is in use has to be a deliberate choice.

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

The DCF tab has **no Compute gate**. It loads and values whatever ticker the page has searched,
as soon as its body renders, and an assumption change is applied on the spot. Nothing on that tab
is ever drawn from a run that does not match the controls on screen — which is why the staleness
notice it used to carry is gone rather than merely unused. Statements are always the **restated**
(`MR`) series; `dcf_model.load_company` still takes `asreported`, but the page does not offer the
choice.

Two caches are what make that affordable, and both are load-bearing:

- `_load_company_cached` — the fetch and normalisation, per ticker.
- `_needs_run` against `_assumption_fingerprint` — the model, the checks and the workbook, per
  distinct assumption set. A field missing from that tuple is a **correctness** bug: the previous
  run's numbers would stay on screen looking current, with nothing to say so.

The cost is real and was accepted deliberately: `st.tabs` renders every tab body on every rerun
and reports **nothing to Python about which tab is open**, so the DCF's statements are fetched
once per company searched whether or not anyone opens the tab. It is the same constraint that
makes the placeholder tabs render nothing and `logic/research/dataset.py` fetch eagerly for every
tab — "load when opened" is not expressible until the tab strip becomes a control that reports
its selection. A tab whose data is expensive enough gates behind its own button until then.

Peer comparison was expected to be where that bill came due, on the assumption that comparables
cost one `DAILY` request each. **They do not.** `DAILY` filtered on a date with no ticker returns
every listed company in a single page — measured at 5,446 rows in ~1.1s — so the cost is fixed.
All three universe fetches (`fetch_universe_daily`, `fetch_universe_fundamentals`,
`fetch_ticker_directory`) are keyed on a date alone, never on a ticker, so **one cache entry
serves every company searched that day** and the second search pays nothing. Keep it that way: a
universe fetch that took a ticker would turn a fixed cost back into a per-company one.

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

View-level tests (`test_page_controls.py`, `test_acknowledgements.py`, `test_research_page_controls.py`)
import the view modules directly and monkeypatch `page.st.session_state` with a dict subclass,
since there is no Streamlit runtime.

The ⚖️ Valuation tab is the first to be tested per-module — `test_research_valuation.py` for the
logic and `test_research_valuation_controls.py` for the view — which is the layout the rest should
move to. Both carry the `research_` prefix because `test_valuation.py` was already taken by the
**DCF** engine's `logic/dcf/valuation.py`: two unrelated modules share the name `valuation.py`,
and the test files have to disambiguate what the modules themselves do not. The
two older files still map onto the *pre-package* layout: `test_stock_research.py` covers all of
`app/logic/research/` except valuation, and `test_research_page_controls.py` covers the shell
plus three tab modules. Split them along their existing section headers when either is next
touched substantially.

`tests/conftest.py` also carries a miniature eleven-ticker universe for the peer medians. It is
sized so every median is checkable by hand and every exclusion rule has something to exclude: a
loss-maker, a non-payer, a delisted company DAILY still returns, a ticker that filed twice in the
window, and an industry too small to have a median. `stub_universe` routes by table name so tests
call the real fetchers — the millions conversion and the filing-date dedup then run for real
rather than being reimplemented in the fixture and proved against themselves.
