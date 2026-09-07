"""Shared fixtures.

Payloads here mirror the shape a real ``SHARADAR/SF1`` datatable response has —
a ``{"datatable": {"columns": [{"name", "type"}], "data": [[...]]}, "meta": {}}``
envelope with one wide row per period — so the parsing and schema tests run
against what the API actually returns rather than a convenient invention.

Two properties of the real feed are reproduced deliberately, because the client
exists to absorb them:

* expense lines (``cor``, ``opex``, ``taxexp``, ``intexp``) arrive as **positive
  magnitudes**, while ``capex`` arrives **negative**;
* rows come back in **no particular order**.

The figures describe a well-behaved company that balances, ties and reconciles
exactly, so any check firing against this fixture is firing wrongly.

No test in this suite touches the network.
"""

from __future__ import annotations

import pytest

_EDGAR = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000999"

#: Every SF1 column the client reads, in the order Sharadar returns them
#: (alphabetical after the key columns). Order is deliberately *not* the order
#: ``SF1_LAYOUT`` declares, so the tests prove the client zips on names.
#:
#: All 91 indicators in ``sharadar_client.SF1_LAYOUT`` are present. The statement
#: views read line items the model never resolves a concept for, so a fixture
#: carrying only the model's subset would leave most of a rendered statement
#: untested.
SF1_COLUMNS = [
    "ticker", "dimension", "calendardate", "datekey", "reportperiod",
    "fiscalperiod", "lastupdated",
    "accoci", "assets", "assetsc", "assetsnc", "assetturnover", "bvps",
    "capex", "cashneq", "consolinc", "cor", "currentratio", "de", "debt",
    "debtc", "debtnc", "deferredrev", "depamor", "deposits", "divyield", "dps",
    "ebit", "ebitda", "ebitdamargin", "ebt", "eps", "epsdil", "equity", "ev",
    "evebit", "evebitda", "fcf", "fcfps", "fxusd", "gp", "grossmargin",
    "intangibles", "intexp", "invcap", "inventory", "investments",
    "investmentsc", "investmentsnc", "liabilities", "liabilitiesc",
    "liabilitiesnc", "marketcap", "ncf", "ncfbus", "ncfcommon", "ncfdebt",
    "ncfdiv", "ncff", "ncfi", "ncfinv", "ncfo", "ncfx", "netinc", "netinccmn",
    "netincdis", "netincnci", "netmargin", "opex", "opinc", "payables",
    "payoutratio", "pb", "pe1", "ppnenet", "prefdivis", "price", "ps1",
    "receivables", "retearn",
    "revenue", "rnd", "roa", "roe", "roic", "ros", "sbcomp", "sgna",
    "sharefactor", "sharesbas", "shareswa", "shareswadil", "sps", "tangibles",
    "taxassets", "taxexp", "taxliabilities", "workingcapital",
]

#: Shares consistent with the EPS figures below, so the share-count
#: cross-checks tie out: 1,000 basic and 1,010 diluted.
_BASIC_SHARES = 1000.0
_DILUTED_SHARES = 1010.0


def _sf1_row(
    *,
    fiscal_period: str,
    calendardate: str,
    reportperiod: str,
    datekey: str,
    dimension: str = "MRY",
    revenue: float,
    cor: float,
    opex: float,
    tax: float,
    dna: float,
    cfo: float,
    capex: float,
    cfi: float,
    cff: float,
    net_change: float,
    cash: float,
    sti: float,
    tca: float,
    tcl: float,
    debt_c: float,
    debt_nc: float,
    assets: float,
    liabilities: float,
) -> list:
    """One SF1 row, with the derivable figures actually derived.

    Deriving rather than hardcoding gross profit, operating income and the like
    keeps the fixture internally consistent: a test that changes revenue cannot
    silently leave the reconciliation checks comparing stale constants.

    The detail lines added for the statement views are split off their own
    subtotal by fixed fractions that deliberately **do not** sum to it, so every
    "Other …" residual row the statement specs derive has real work to do. A
    fixture whose components added up exactly would let a broken residual pass
    as zero.
    """
    gp = revenue - cor
    opinc = gp - opex
    intexp = 0.0
    ebt = opinc - intexp
    netinc = ebt - tax
    # Equity is the plug that makes the balance sheet balance, and the fixture
    # company has no non-controlling interests.
    equity = assets - liabilities
    ebitda = opinc + dna
    fcf = cfo + capex
    assets_nc = assets - tca
    liabilities_nc = liabilities - tcl

    # Operating expenses: SG&A and R&D leave 15% unattributed.
    sgna = round(opex * 0.55, 6)
    rnd = round(opex * 0.30, 6)

    # Current assets beyond cash and short-term investments; 30% unattributed.
    current_rest = tca - cash - sti
    receivables = round(current_rest * 0.40, 6)
    inventory = round(current_rest * 0.30, 6)

    # Non-current assets; `investmentsnc` is nil, so 25% is unattributed.
    ppnenet = round(assets_nc * 0.45, 6)
    intangibles = round(assets_nc * 0.25, 6)
    taxassets = round(assets_nc * 0.05, 6)

    # Current liabilities beyond current debt; 30% unattributed.
    current_liab_rest = tcl - debt_c
    payables = round(current_liab_rest * 0.50, 6)
    deferredrev = round(current_liab_rest * 0.20, 6)

    # Non-current liabilities beyond non-current debt; half unattributed.
    taxliabilities = round((liabilities_nc - debt_nc) * 0.50, 6)

    # Common equity. AOCI is negative, as it usually is in a real filing.
    retearn = round(equity * 0.70, 6)
    accoci = round(equity * -0.05, 6)

    # Financing and investing detail, again leaving a residual in each.
    ncfdiv = round(cff * 0.40, 6)
    ncfcommon = round(cff * 0.35, 6)
    ncfdebt = round(cff * 0.10, 6)
    ncfbus = round((cfi - capex) * 0.30, 6)
    ncfinv = round((cfi - capex) * 0.50, 6)

    eps = round(netinc / _BASIC_SHARES, 6)
    dps = round(-ncfdiv / _BASIC_SHARES, 6)
    price = round(eps * 20.0, 6)
    marketcap = round(price * _BASIC_SHARES, 6)
    debt = debt_c + debt_nc
    ev = round(marketcap + debt - cash, 6)
    invcap = debt + equity

    values = {
        "ticker": "TEST",
        "dimension": dimension,
        "calendardate": calendardate,
        "datekey": datekey,
        "reportperiod": reportperiod,
        "fiscalperiod": fiscal_period,
        "lastupdated": datekey,
        # --- Income statement ---------------------------------------------
        "revenue": revenue,
        "cor": cor,
        "gp": gp,
        "sgna": sgna,
        "rnd": rnd,
        "opex": opex,
        "opinc": opinc,
        "ebit": netinc + tax + intexp,
        "intexp": intexp,
        "ebt": ebt,
        "taxexp": tax,
        "consolinc": netinc,
        "netincnci": 0.0,
        "netinc": netinc,
        "prefdivis": 0.0,
        "netinccmn": netinc,
        "netincdis": 0.0,
        "eps": eps,
        "epsdil": round(netinc / _DILUTED_SHARES, 6),
        "shareswa": _BASIC_SHARES,
        "shareswadil": _DILUTED_SHARES,
        "dps": dps,
        # --- Balance sheet -------------------------------------------------
        "cashneq": cash,
        "investmentsc": sti,
        "receivables": receivables,
        "inventory": inventory,
        "assetsc": tca,
        "investmentsnc": 0.0,
        "ppnenet": ppnenet,
        "intangibles": intangibles,
        "taxassets": taxassets,
        "investments": sti,
        "assetsnc": assets_nc,
        "assets": assets,
        "payables": payables,
        "deferredrev": deferredrev,
        "debtc": debt_c,
        "liabilitiesc": tcl,
        "debtnc": debt_nc,
        "deposits": 0.0,
        "taxliabilities": taxliabilities,
        "liabilitiesnc": liabilities_nc,
        "liabilities": liabilities,
        "debt": debt,
        "retearn": retearn,
        "accoci": accoci,
        "equity": equity,
        # --- Cash flow ------------------------------------------------------
        "depamor": dna,
        "sbcomp": 5.0,
        "ncfo": cfo,
        "capex": capex,
        "ncfbus": ncfbus,
        "ncfinv": ncfinv,
        "ncfi": cfi,
        "ncfdebt": ncfdebt,
        "ncfcommon": ncfcommon,
        "ncfdiv": ncfdiv,
        "ncff": cff,
        "ncfx": 0.0,
        "ncf": net_change,
        # --- Derived ---------------------------------------------------------
        "ebitda": ebitda,
        "fcf": fcf,
        "workingcapital": tca - tcl,
        "invcap": invcap,
        "tangibles": assets - intangibles,
        "sharesbas": _BASIC_SHARES,
        "sharefactor": 1.0,
        "price": price,
        "marketcap": marketcap,
        "ev": ev,
        "evebit": round(ev / opinc, 6),
        "evebitda": round(ev / ebitda, 6),
        "pe1": round(marketcap / netinc, 6),
        "pb": round(marketcap / equity, 6),
        "ps1": round(marketcap / revenue, 6),
        "grossmargin": round(gp / revenue, 6),
        "ebitdamargin": round(ebitda / revenue, 6),
        "netmargin": round(netinc / revenue, 6),
        "ros": round(opinc / revenue, 6),
        "roe": round(netinc / equity, 6),
        "roa": round(netinc / assets, 6),
        "roic": round(opinc / invcap, 6),
        "currentratio": round(tca / tcl, 6),
        "de": round(liabilities / equity, 6),
        "bvps": round(equity / _BASIC_SHARES, 6),
        "fcfps": round(fcf / _BASIC_SHARES, 6),
        "sps": round(revenue / _BASIC_SHARES, 6),
        "divyield": round(dps / price, 6),
        "payoutratio": round(dps / eps, 6),
        "assetturnover": round(revenue / assets, 6),
        "fxusd": 1.0,
    }
    return [values[name] for name in SF1_COLUMNS]


#: Three fiscal years, oldest last — Sharadar does not sort, and the client is
#: responsible for putting them in order.
_ANNUAL_ROWS = [
    _sf1_row(
        fiscal_period="2025-FY", calendardate="2025-12-31",
        reportperiod="2025-09-30", datekey="2025-10-31",
        revenue=1200, cor=700, opex=190, tax=60, dna=72,
        cfo=340, capex=-100, cfi=-120, cff=-200, net_change=20,
        cash=140, sti=70, tca=480, tcl=360, debt_c=50, debt_nc=220,
        assets=1060, liabilities=700,
    ),
    _sf1_row(
        fiscal_period="2023-FY", calendardate="2023-12-31",
        reportperiod="2023-09-30", datekey="2023-11-03",
        revenue=1000, cor=600, opex=150, tax=48, dna=60,
        cfo=280, capex=-80, cfi=-100, cff=-160, net_change=20,
        cash=100, sti=50, tca=400, tcl=300, debt_c=40, debt_nc=200,
        assets=900, liabilities=600,
    ),
    _sf1_row(
        fiscal_period="2024-FY", calendardate="2024-12-31",
        reportperiod="2024-09-30", datekey="2024-11-01",
        revenue=1100, cor=650, opex=170, tax=54, dna=66,
        cfo=310, capex=-90, cfi=-110, cff=-180, net_change=20,
        cash=120, sti=60, tca=440, tcl=330, debt_c=45, debt_nc=210,
        assets=980, liabilities=650,
    ),
]


#: Four quarters plus the trailing-twelve-month row they sum to, so the
#: statement views can be exercised at every dimension the page requests. The
#: TTM flows are the four quarters added up; its balance sheet is the most
#: recent quarter's, which is what Sharadar's ``MRT`` dimension actually
#: carries and why the balance sheet's leading column is labelled by quarter
#: rather than as a twelve-month figure.
_QUARTER_INPUTS = [
    dict(fiscal_period="2025-Q3", calendardate="2025-09-30",
         reportperiod="2025-09-30", datekey="2025-10-28",
         revenue=280, cor=163, opex=44, tax=14, dna=17,
         cfo=79, capex=-23, cfi=-28, cff=-47, net_change=4,
         cash=128, sti=64, tca=452, tcl=338, debt_c=47, debt_nc=214,
         assets=1010, liabilities=668),
    dict(fiscal_period="2025-Q4", calendardate="2025-12-31",
         reportperiod="2025-12-31", datekey="2026-01-27",
         revenue=310, cor=180, opex=49, tax=15, dna=18,
         cfo=87, capex=-26, cfi=-31, cff=-52, net_change=4,
         cash=134, sti=67, tca=466, tcl=349, debt_c=48, debt_nc=217,
         assets=1035, liabilities=684),
    dict(fiscal_period="2026-Q1", calendardate="2026-03-31",
         reportperiod="2026-03-31", datekey="2026-04-28",
         revenue=295, cor=172, opex=47, tax=15, dna=18,
         cfo=83, capex=-25, cfi=-30, cff=-50, net_change=3,
         cash=140, sti=70, tca=480, tcl=360, debt_c=50, debt_nc=220,
         assets=1060, liabilities=700),
    dict(fiscal_period="2026-Q2", calendardate="2026-06-30",
         reportperiod="2026-06-30", datekey="2026-07-28",
         revenue=325, cor=190, opex=51, tax=16, dna=19,
         cfo=91, capex=-27, cfi=-33, cff=-55, net_change=3,
         cash=146, sti=73, tca=497, tcl=372, debt_c=52, debt_nc=228,
         assets=1092, liabilities=722),
]

#: Newest first, so the client's sort is exercised here too.
_QUARTERLY_ROWS = [
    _sf1_row(dimension="MRQ", **inputs) for inputs in reversed(_QUARTER_INPUTS)
]

_FLOWS = ("revenue", "cor", "opex", "tax", "dna", "cfo", "capex", "cfi", "cff",
          "net_change")

#: The TTM row: flows summed over the four quarters, balances taken from the
#: latest one.
_TTM_ROW = _sf1_row(
    dimension="MRT",
    fiscal_period=_QUARTER_INPUTS[-1]["fiscal_period"],
    calendardate=_QUARTER_INPUTS[-1]["calendardate"],
    reportperiod=_QUARTER_INPUTS[-1]["reportperiod"],
    datekey=_QUARTER_INPUTS[-1]["datekey"],
    **{key: sum(q[key] for q in _QUARTER_INPUTS) for key in _FLOWS},
    **{key: _QUARTER_INPUTS[-1][key]
       for key in ("cash", "sti", "tca", "tcl", "debt_c", "debt_nc",
                   "assets", "liabilities")},
)


def blank(row: list, *names: str) -> list:
    """A copy of ``row`` with the named indicators returned as nulls.

    Sharadar sends a JSON null for an indicator a company does not report, which
    is not the same as sending 0 — and for the indicators outside
    ``ZERO_FILLED_INDICATORS`` it is the only signal there is. Blanking rather
    than zeroing is what lets a test prove a residual row goes missing instead
    of quietly computing against an assumed zero.
    """
    out = list(row)
    for name in names:
        out[SF1_COLUMNS.index(name)] = None
    return out


def sf1_payload(rows: list | None = None, *, cursor: str | None = None) -> dict:
    """Wraps rows in the datatable envelope the Tables API returns."""
    return {
        "datatable": {
            "data": rows if rows is not None else _ANNUAL_ROWS,
            "columns": [{"name": name, "type": "text"} for name in SF1_COLUMNS],
        },
        "meta": {"next_cursor_id": cursor},
    }


#: What ``SHARADAR/TICKERS?table=SF1`` returns for the fixture company.
TICKERS_COLUMNS = [
    "table", "permaticker", "ticker", "name", "exchange", "isdelisted",
    "category", "siccode", "sicsector", "sicindustry", "sector", "industry",
    "currency", "location", "secfilings",
]

TICKERS_ROW = [
    "SF1", 999, "TEST", "TESTCO INC", "NASDAQ", "N", "Domestic Common Stock",
    3571, "Manufacturing", "Electronic Computers", "Technology",
    "Consumer Electronics", "USD", "California; U.S.A", _EDGAR,
]


def tickers_payload(row: list | None = None) -> dict:
    return {
        "datatable": {
            "data": [row if row is not None else TICKERS_ROW],
            "columns": [{"name": name, "type": "text"} for name in TICKERS_COLUMNS],
        },
        "meta": {"next_cursor_id": None},
    }


@pytest.fixture
def annual_rows() -> list:
    """The three annual SF1 rows, unsorted, as lists of values."""
    return list(_ANNUAL_ROWS)


@pytest.fixture
def statements_payload() -> dict:
    """A complete annual ``SHARADAR/SF1`` response for the fixture company."""
    return sf1_payload()


@pytest.fixture
def sparse_statements_payload() -> dict:
    """The same company, but three line items it does not report at all.

    ``sbcomp``, ``rnd`` and ``intangibles`` are blanked. None of the three is in
    ``ZERO_FILLED_INDICATORS``, so Sharadar leaves them null rather than zero,
    and each one is a component of a different residual row. That makes this the
    fixture for "missing stays missing": the residuals that depend on them must
    go missing, and the ones that do not must still compute.
    """
    return sf1_payload([blank(row, "sbcomp", "rnd", "intangibles")
                        for row in _ANNUAL_ROWS])


@pytest.fixture
def quarterly_statements_payload() -> dict:
    """Four ``MRQ`` quarters, newest first."""
    return sf1_payload(list(_QUARTERLY_ROWS))


@pytest.fixture
def ttm_statements_payload() -> dict:
    """The single ``MRT`` row covering those four quarters."""
    return sf1_payload([list(_TTM_ROW)])


@pytest.fixture
def tickers_response() -> dict:
    """A complete ``SHARADAR/TICKERS`` response for the fixture company."""
    return tickers_payload()


@pytest.fixture
def financial_tickers_response() -> dict:
    """The same, but for a bank — SIC 6021, which the sector gate must reject."""
    row = list(TICKERS_ROW)
    row[TICKERS_COLUMNS.index("siccode")] = 6021
    row[TICKERS_COLUMNS.index("sicsector")] = "Finance Insurance And Real Estate"
    row[TICKERS_COLUMNS.index("sicindustry")] = "National Commercial Banks"
    row[TICKERS_COLUMNS.index("sector")] = "Financial Services"
    row[TICKERS_COLUMNS.index("industry")] = "Banks - Diversified"
    return tickers_payload(row)


# ---------------------------------------------------------------------------
# Corporate actions
# ---------------------------------------------------------------------------

#: ``SHARADAR/ACTIONS`` columns. As with SF1, the order here is deliberately not
#: the order the client reads them in.
ACTIONS_COLUMNS = ["action", "value", "date", "ticker", "name", "contraticker", "contraname"]

#: Rows in the order the API actually returns them: **newest first**, the
#: reverse of every other Sharadar table. The contra columns carry the literal
#: string ``"N/A"`` rather than a null, which is the other quirk the client
#: absorbs. Two dividends, one split, one ticker change with a real contra.
ACTIONS_ROWS = [
    ["dividend", 0.27, "2025-11-10", "TEST", "TESTCO INC", "N/A", "N/A"],
    ["dividend", 0.25, "2025-05-12", "TEST", "TESTCO INC", "N/A", "N/A"],
    ["split", 4.0, "2020-08-31", "TEST", "TESTCO INC", "N/A", "N/A"],
    ["tickerchangeto", None, "2014-03-27", "TEST", "TESTCO INC", "OLDT", "OLDCO INC"],
]


def actions_payload(rows: list | None = None, *, cursor: str | None = None) -> dict:
    """Wraps action rows in the datatable envelope."""
    return {
        "datatable": {
            "data": rows if rows is not None else ACTIONS_ROWS,
            "columns": [{"name": name, "type": "text"} for name in ACTIONS_COLUMNS],
        },
        "meta": {"next_cursor_id": cursor},
    }


@pytest.fixture
def actions_response() -> dict:
    """A complete ``SHARADAR/ACTIONS`` response for the fixture company."""
    return actions_payload()


# ---------------------------------------------------------------------------
# The cross-sectional universe
# ---------------------------------------------------------------------------
#
# A miniature market for the Valuation tab's peer medians. Eleven tickers in
# three industries, built so every median below is checkable by hand and every
# exclusion rule has something to exclude:
#
#   Consumer Electronics — TEST and five peers, all usable, plus LOSS (negative
#     multiples) and GONE (delisted). Six survivors, so the medians are the mean
#     of the third and fourth values and a broken exclusion moves them visibly.
#   Aluminum — three tickers, below ``min_peers``, so the floor has a case.
#
#: ``SHARADAR/DAILY`` columns. Order is deliberately not the order the client
#: reads them in, and ``marketcap``/``ev`` are **in millions** as the real table
#: publishes them — the conversion is what these rows exist to catch.
UNIVERSE_DAILY_COLUMNS = [
    "ticker", "date", "lastupdated", "ev", "evebit", "evebitda",
    "marketcap", "pb", "pe", "ps",
]

_UNIVERSE_DATE = "2026-08-05"

#: Unsorted, as the API returns them.
UNIVERSE_DAILY_ROWS = [
    ["PEER3", _UNIVERSE_DATE, _UNIVERSE_DATE, 2000.0, 14.0, 11.0, 1800.0, 3.5, 18.0, 1.8],
    ["GONE", _UNIVERSE_DATE, _UNIVERSE_DATE, 100.0, 2.0, 2.0, 90.0, 0.5, 999.0, 0.2],
    ["TEST", _UNIVERSE_DATE, _UNIVERSE_DATE, 2200.0, 15.0, 12.0, 2000.0, 4.0, 20.0, 2.0],
    ["PEER5", _UNIVERSE_DATE, _UNIVERSE_DATE, 3400.0, 24.0, 19.0, 3000.0, 6.0, 30.0, 3.0],
    ["LOSS", _UNIVERSE_DATE, _UNIVERSE_DATE, 800.0, -6.0, -8.0, 700.0, 1.2, -5.0, 0.7],
    ["PEER1", _UNIVERSE_DATE, _UNIVERSE_DATE, 1100.0, 9.0, 7.0, 1000.0, 2.0, 10.0, 1.0],
    ["PEER4", _UNIVERSE_DATE, _UNIVERSE_DATE, 2900.0, 20.0, 16.0, 2600.0, 5.0, 26.0, 2.6],
    ["PEER2", _UNIVERSE_DATE, _UNIVERSE_DATE, 1600.0, 12.0, 9.0, 1500.0, 3.0, 14.0, 1.5],
    ["ALU1", _UNIVERSE_DATE, _UNIVERSE_DATE, 550.0, 8.0, 6.0, 500.0, 1.5, 11.0, 0.9],
    ["ALU2", _UNIVERSE_DATE, _UNIVERSE_DATE, 660.0, 9.0, 7.0, 600.0, 1.8, 13.0, 1.1],
    ["ALU3", _UNIVERSE_DATE, _UNIVERSE_DATE, 770.0, 10.0, 8.0, 700.0, 2.1, 15.0, 1.3],
]

#: ``SHARADAR/SF1`` at ``dimension=ART`` with the universe column projection.
#:
#: ``ncfdiv`` is negative — a cash outflow — which is why the yield negates it,
#: and ``capex`` is negative for the same reason, which is why growth takes its
#: magnitude. PEER3 pays nothing, so its zero has to be excluded rather than
#: counted as a yield or a growth rate of nought.
UNIVERSE_SF1_COLUMNS = [
    "ticker", "calendardate", "datekey",
    "revenue", "ncfo", "ncfdiv", "ebitda", "ebit", "epsdil",
    "fcf", "roe", "workingcapital", "capex", "dps",
]

#: The period every ``_prior`` value is paired against, and the one the levels
#: are read from. Twelve months apart exactly, which is what the pairing looks
#: for; a fixture whose two periods were fifteen months apart would silently
#: exercise nothing.
_SF1_BASE_PERIOD = "2025-03-31"
_SF1_LATEST_PERIOD = "2026-03-31"

#: Two periods per ticker, so the year-on-year pairing has both ends, and every
#: degenerate base the growth gate exists to catch:
#:
#:   LOSS   — negative bases on ebitda/ebit/epsdil/fcf/roe/workingcapital, with
#:            the loss *narrowing* (epsdil −4.00 → −2.00). The naive ratio prints
#:            −50%, which reads as collapse when the company in fact improved.
#:            Its revenue base is positive, so it still counts toward revenue
#:            growth — the exclusion is per metric, not per company.
#:   PEER3  — zero capex and zero dividend in the base period, the bank and the
#:            non-payer. A zero base is undefined; a zero *current* is not.
#:   PEER1  — every metric halved, so the medians have something below zero.
#:   TEST   — filed **twice for the same period**. The later ``datekey``
#:            supersedes; the earlier carries wrong figures, so a broken
#:            restatement dedup moves every number on the page.
UNIVERSE_SF1_ROWS = [
    # ticker, calendardate, datekey, revenue, ncfo, ncfdiv, ebitda, ebit, epsdil, fcf, roe, workingcapital, capex, dps
    ["TEST", _SF1_BASE_PERIOD, "2025-05-02", 1.0e9, 1.6e8, -3.0e7, 2.0e8, 1.5e8, 1.00, 1.0e8, 0.20, 5.0e7, -1.0e8, 0.20],
    ["TEST", _SF1_LATEST_PERIOD, "2026-04-20", 9.9e8, 1.0e8, -1.0e7, 1.0e8, 1.0e8, 0.90, 1.0e8, 0.10, 1.0e7, -1.0e8, 0.10],
    ["TEST", _SF1_LATEST_PERIOD, "2026-05-01", 1.1e9, 2.0e8, -4.0e7, 2.4e8, 1.8e8, 1.20, 1.2e8, 0.25, 6.0e7, -1.2e8, 0.22],
    ["PEER1", _SF1_BASE_PERIOD, "2025-05-05", 2.2e9, 4.0e8, -2.0e7, 4.0e8, 3.0e8, 2.00, 2.0e8, 0.40, 1.0e8, -2.0e8, 0.10],
    ["PEER1", _SF1_LATEST_PERIOD, "2026-05-04", 1.1e9, 2.0e8, -2.0e7, 2.0e8, 1.5e8, 1.00, 1.0e8, 0.20, 5.0e7, -1.0e8, 0.10],
    ["PEER2", _SF1_BASE_PERIOD, "2025-05-07", 8.0e8, 1.0e8, -3.0e7, 1.0e8, 8.0e7, 0.80, 5.0e7, 0.10, 4.0e7, -5.0e7, 0.15],
    ["PEER2", _SF1_LATEST_PERIOD, "2026-05-06", 1.0e9, 2.0e8, -3.0e7, 1.5e8, 1.2e8, 1.00, 7.5e7, 0.15, 5.0e7, -7.5e7, 0.18],
    ["PEER3", _SF1_BASE_PERIOD, "2025-05-08", 5.0e8, 1.0e8, 0.0, 1.0e8, 8.0e7, 0.50, 5.0e7, 0.10, 2.5e7, 0.0, 0.0],
    ["PEER3", _SF1_LATEST_PERIOD, "2026-05-07", 1.0e9, 2.0e8, 0.0, 2.0e8, 1.6e8, 1.00, 1.0e8, 0.20, 5.0e7, -5.0e7, 0.0],
    ["PEER4", _SF1_BASE_PERIOD, "2025-05-09", 8.0e8, 1.6e8, -5.2e7, 1.6e8, 1.2e8, 0.80, 8.0e7, 0.16, 4.0e7, -8.0e7, 0.40],
    ["PEER4", _SF1_LATEST_PERIOD, "2026-05-08", 1.0e9, 2.0e8, -5.2e7, 2.0e8, 1.5e8, 1.00, 1.0e8, 0.20, 5.0e7, -1.0e8, 0.44],
    ["PEER5", _SF1_BASE_PERIOD, "2025-05-12", 6.25e8, 1.0e8, -1.2e8, 1.0e8, 8.0e7, 0.50, 5.0e7, 0.10, 2.5e7, -5.0e7, 0.50],
    ["PEER5", _SF1_LATEST_PERIOD, "2026-05-11", 1.0e9, 2.0e8, -1.2e8, 2.0e8, 1.6e8, 1.00, 1.0e8, 0.20, 5.0e7, -1.0e8, 0.60],
    ["LOSS", _SF1_BASE_PERIOD, "2025-05-13", 1.0e9, -5.0e7, 0.0, -5.0e7, -1.0e8, -4.00, -8.0e7, -0.20, -2.0e7, -4.0e7, 0.0],
    ["LOSS", _SF1_LATEST_PERIOD, "2026-05-12", 5.0e8, -1.0e8, 0.0, -2.5e7, -5.0e7, -2.00, -4.0e7, -0.10, -1.0e7, -2.0e7, 0.0],
    ["ALU1", _SF1_BASE_PERIOD, "2025-05-14", 3.2e8, 4.0e7, -4.0e6, 5.0e7, 4.0e7, 0.40, 3.0e7, 0.08, 2.0e7, -2.0e7, 0.08],
    ["ALU1", _SF1_LATEST_PERIOD, "2026-05-13", 4.0e8, 5.0e7, -5.0e6, 6.0e7, 5.0e7, 0.50, 3.5e7, 0.10, 2.5e7, -2.5e7, 0.10],
    ["ALU2", _SF1_BASE_PERIOD, "2025-05-15", 3.2e8, 5.0e7, -5.0e6, 5.0e7, 4.0e7, 0.40, 3.0e7, 0.08, 2.0e7, -2.0e7, 0.08],
    ["ALU2", _SF1_LATEST_PERIOD, "2026-05-14", 4.0e8, 6.0e7, -6.0e6, 6.0e7, 5.0e7, 0.50, 3.5e7, 0.10, 2.5e7, -2.5e7, 0.10],
    ["ALU3", _SF1_BASE_PERIOD, "2025-05-16", 3.2e8, 6.0e7, -6.0e6, 5.0e7, 4.0e7, 0.40, 3.0e7, 0.08, 2.0e7, -2.0e7, 0.08],
    ["ALU3", _SF1_LATEST_PERIOD, "2026-05-15", 4.0e8, 7.0e7, -7.0e6, 6.0e7, 5.0e7, 0.50, 3.5e7, 0.10, 2.5e7, -2.5e7, 0.10],
]

#: ``SHARADAR/TICKERS`` under the directory projection. GONE is delisted and has
#: to drop out even though DAILY still carries a row for it.
DIRECTORY_COLUMNS = [
    "ticker", "name", "sector", "industry", "scalemarketcap", "isdelisted",
]

DIRECTORY_ROWS = [
    ["TEST", "TESTCO INC", "Technology", "Consumer Electronics", "4 - Mid", "N"],
    ["PEER1", "PEER ONE", "Technology", "Consumer Electronics", "4 - Mid", "N"],
    ["PEER2", "PEER TWO", "Technology", "Consumer Electronics", "4 - Mid", "N"],
    ["PEER3", "PEER THREE", "Technology", "Consumer Electronics", "4 - Mid", "N"],
    ["PEER4", "PEER FOUR", "Technology", "Consumer Electronics", "4 - Mid", "N"],
    ["PEER5", "PEER FIVE", "Technology", "Consumer Electronics", "4 - Mid", "N"],
    ["LOSS", "LOSSMAKER INC", "Technology", "Consumer Electronics", "3 - Small", "N"],
    ["GONE", "DELISTED INC", "Technology", "Consumer Electronics", "2 - Micro", "Y"],
    ["ALU1", "ALUMINIUM ONE", "Basic Materials", "Aluminum", "3 - Small", "N"],
    ["ALU2", "ALUMINIUM TWO", "Basic Materials", "Aluminum", "3 - Small", "N"],
    ["ALU3", "ALUMINIUM THREE", "Basic Materials", "Aluminum", "3 - Small", "N"],
]


def _envelope(columns: list, rows: list, cursor: str | None = None) -> dict:
    return {
        "datatable": {
            "data": rows,
            "columns": [{"name": name, "type": "text"} for name in columns],
        },
        "meta": {"next_cursor_id": cursor},
    }


def universe_daily_payload(rows: list | None = None, *, cursor: str | None = None) -> dict:
    rows = UNIVERSE_DAILY_ROWS if rows is None else rows
    return _envelope(UNIVERSE_DAILY_COLUMNS, rows, cursor)


def universe_sf1_payload(rows: list | None = None, *, cursor: str | None = None) -> dict:
    rows = UNIVERSE_SF1_ROWS if rows is None else rows
    return _envelope(UNIVERSE_SF1_COLUMNS, rows, cursor)


def directory_payload(rows: list | None = None, *, cursor: str | None = None) -> dict:
    rows = DIRECTORY_ROWS if rows is None else rows
    return _envelope(DIRECTORY_COLUMNS, rows, cursor)


def stub_universe(monkeypatch) -> None:
    """Routes the three universe requests to the fixture payloads by table.

    Dispatching on the table name rather than returning one payload for
    everything is what lets a test call the real fetchers — the millions
    conversion and the filing-date dedup then run for real instead of being
    reimplemented in the fixture and proved against themselves.
    """
    from app.data import sharadar_client as sc

    def route(table, query, **kwargs):
        if table == "DAILY":
            return universe_daily_payload()
        if table == "SF1":
            return universe_sf1_payload()
        if table == "TICKERS":
            return directory_payload()
        raise AssertionError(f"unexpected table {table}")

    monkeypatch.setattr(sc, "_get", route)


@pytest.fixture
def peer_universe(monkeypatch):
    """The joined universe frame, as ``ResearchDataset.peer_universe`` holds it."""
    from app.data import sharadar_client as sc
    from app.logic.research import comparison

    stub_universe(monkeypatch)
    return comparison.build_peer_universe(
        sc._fetch_universe_daily("2026-08-05"),
        sc._fetch_universe_fundamentals("2026-02-15"),
        sc._fetch_ticker_directory(),
    )
