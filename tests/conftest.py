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
SF1_COLUMNS = [
    "ticker", "dimension", "calendardate", "datekey", "reportperiod",
    "fiscalperiod", "lastupdated",
    "assets", "assetsc", "assetsnc", "capex", "cashneq", "consolinc", "cor",
    "debt", "debtc", "debtnc", "depamor", "ebit", "ebitda", "ebt", "eps",
    "epsdil", "equity", "fcf", "gp", "intexp", "investmentsc", "investmentsnc",
    "liabilities", "liabilitiesc", "ncf", "ncff", "ncfi", "ncfo", "ncfx",
    "netinc", "opex", "opinc", "revenue", "sbcomp", "sharefactor", "sharesbas",
    "shareswa", "shareswadil", "taxexp", "workingcapital",
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
    """
    gp = revenue - cor
    opinc = gp - opex
    intexp = 0.0
    ebt = opinc - intexp
    netinc = ebt - tax
    # Equity is the plug that makes the balance sheet balance, and the fixture
    # company has no non-controlling interests.
    equity = assets - liabilities
    values = {
        "ticker": "TEST",
        "dimension": dimension,
        "calendardate": calendardate,
        "datekey": datekey,
        "reportperiod": reportperiod,
        "fiscalperiod": fiscal_period,
        "lastupdated": datekey,
        "assets": assets,
        "assetsc": tca,
        "assetsnc": assets - tca,
        "capex": capex,
        "cashneq": cash,
        "consolinc": netinc,
        "cor": cor,
        "debt": debt_c + debt_nc,
        "debtc": debt_c,
        "debtnc": debt_nc,
        "depamor": dna,
        "ebit": netinc + tax + intexp,
        "ebitda": opinc + dna,
        "ebt": ebt,
        "eps": round(netinc / _BASIC_SHARES, 6),
        "epsdil": round(netinc / _DILUTED_SHARES, 6),
        "equity": equity,
        "fcf": cfo + capex,
        "gp": gp,
        "intexp": intexp,
        "investmentsc": sti,
        "investmentsnc": 0.0,
        "liabilities": liabilities,
        "liabilitiesc": tcl,
        "ncf": net_change,
        "ncff": cff,
        "ncfi": cfi,
        "ncfo": cfo,
        "ncfx": 0.0,
        "netinc": netinc,
        "opex": opex,
        "opinc": opinc,
        "revenue": revenue,
        "sbcomp": 5.0,
        "sharefactor": 1.0,
        "sharesbas": _BASIC_SHARES,
        "shareswa": _BASIC_SHARES,
        "shareswadil": _DILUTED_SHARES,
        "taxexp": tax,
        "workingcapital": tca - tcl,
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
