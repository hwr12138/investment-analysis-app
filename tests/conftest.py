"""Shared fixtures.

All payloads here are trimmed copies of real SimFin responses recorded while
building this page, so the parsing and schema tests run against the shapes the
API actually returns rather than the shapes its OpenAPI definition describes.

No test in this suite touches the network.
"""

from __future__ import annotations

import pytest

# Real AAPL column arrays, trimmed to the concepts the model resolves.
AAPL_PL_COLUMNS = [
    "Fiscal Period", "Fiscal Year", "Report Date", "Publish Date", "Restated", "Source",
    "TTM", "Value Check", "Data Model", "Revenue", "Cost of revenue", "Gross Profit",
    "Operating Expenses", "Operating Income (Loss)", "Interest Expense, net",
    "Interest Expense", "Pretax Income (Loss)", "Income Tax (Expense) Benefit, net",
    "Net Income",
]

AAPL_BS_COLUMNS = [
    "Fiscal Period", "Fiscal Year", "Report Date", "Publish Date", "Restated", "Source",
    "TTM", "Value Check", "Data Model", "Cash, Cash Equivalents & Short Term Investments",
    "Cash & Cash Equivalents", "Short Term Investments", "Total Current Assets",
    "Total Current Liabilities", "Short Term Debt", "Current Portion of Long Term Debt",
    "Long Term Debt", "Total Assets", "Total Liabilities", "Total Equity",
    "Total Liabilities & Equity", "Minority Interest", "Preferred Equity",
    "Long Term Investments",
]

AAPL_CF_COLUMNS = [
    "Fiscal Period", "Fiscal Year", "Report Date", "Publish Date", "Restated", "Source",
    "TTM", "Value Check", "Data Model", "Net Income", "Depreciation & Amortization",
    "Non-Cash Items", "Stock-Based Compensation", "Change in Working Capital",
    "Cash from Operating Activities", "Change in Fixed Assets & Intangibles",
    "Disposition of Fixed Assets & Intangibles", "Acquisition of Fixed Assets & Intangibles",
    "Cash from Investing Activities", "Cash from Financing Activities",
    "Effect of Foreign Exchange Rates", "Net Changes in Cash",
]

AAPL_DERIVED_COLUMNS = [
    "Fiscal Period", "Fiscal Year", "Report Date", "TTM", "Restated", "Data Model",
    "EBITDA", "Earnings Per Share, Basic", "Earnings Per Share, Diluted",
    "Free Cash Flow", "Total Debt", "Operating Margin",
]

_SOURCE = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/index.htm"


def _pl_row(fyear, report_date, publish_date, revenue, cor, opex, ebit, pretax, tax, net):
    return [
        "FY", fyear, report_date, publish_date, 1, _SOURCE, 0, 1, 0.0,
        revenue, cor, revenue + cor, opex, ebit, None, None, pretax, tax, net,
    ]


def _bs_row(fyear, report_date, publish_date, cash, sti, tca, tcl, std, ltd, ta, tl, te):
    return [
        # The balance sheet labels the annual period Q4 while the other
        # statements label it FY — a real quirk the merge has to absorb.
        "Q4", fyear, report_date, publish_date, 1, _SOURCE, 0, 1, 0.0,
        cash + sti, cash, sti, tca, tcl, std, None, ltd, ta, tl, te, ta, None, None, None,
    ]


def _cf_row(fyear, report_date, publish_date, net, dna, cfo, capex_net, cfi, cff, change):
    return [
        "FY", fyear, report_date, publish_date, 1, _SOURCE, 0, 1, 0.0,
        net, dna, dna, None, -1000, cfo, capex_net, None, None, cfi, cff, None, change,
    ]


def _derived_row(fyear, report_date, ebitda, eps_basic, eps_diluted, fcf, debt, margin):
    return [
        "FY", fyear, report_date, 0, 1, 0.0,
        ebitda, eps_basic, eps_diluted, fcf, debt, margin,
    ]


@pytest.fixture
def statements_payload() -> list:
    """Three fiscal years of a well-behaved GENERAL-template company.

    Balances, ties, and reconciles exactly, so any check that fires against this
    fixture is firing wrongly.
    """
    return [
        {
            "template": "GENERAL",
            "name": "TESTCO INC",
            "id": 999,
            "ticker": "TEST",
            "currency": "USD",
            "isin": "US0000000000",
            "statements": [
                {
                    "statement": "PL",
                    "columns": AAPL_PL_COLUMNS,
                    "data": [
                        _pl_row(2023, "2023-09-30", "2023-11-03",
                                1000, -600, -150, 250, 240, -48, 192),
                        _pl_row(2024, "2024-09-30", "2024-11-01",
                                1100, -650, -170, 280, 270, -54, 216),
                        _pl_row(2025, "2025-09-30", "2025-10-31",
                                1200, -700, -190, 310, 300, -60, 240),
                    ],
                },
                {
                    "statement": "BS",
                    "columns": AAPL_BS_COLUMNS,
                    "data": [
                        _bs_row(2023, "2023-09-30", "2023-11-03",
                                100, 50, 400, 300, 40, 200, 900, 600, 300),
                        _bs_row(2024, "2024-09-30", "2024-11-01",
                                120, 60, 440, 330, 45, 210, 980, 650, 330),
                        _bs_row(2025, "2025-09-30", "2025-10-31",
                                140, 70, 480, 360, 50, 220, 1060, 700, 360),
                    ],
                },
                {
                    "statement": "CF",
                    "columns": AAPL_CF_COLUMNS,
                    "data": [
                        _cf_row(2023, "2023-09-30", "2023-11-03",
                                192, 60, 280, -80, -100, -160, 20),
                        _cf_row(2024, "2024-09-30", "2024-11-01",
                                216, 66, 310, -90, -110, -180, 20),
                        _cf_row(2025, "2025-09-30", "2025-10-31",
                                240, 72, 340, -100, -120, -200, 20),
                    ],
                },
                {
                    # EPS is consistent with 1,000 basic and 1,010 diluted
                    # shares, so the share-count cross-checks tie out.
                    "statement": "DERIVED",
                    "columns": AAPL_DERIVED_COLUMNS,
                    "data": [
                        _derived_row(2023, "2023-09-30", 310, 0.192, 0.190099, 200, 240, 0.25),
                        _derived_row(2024, "2024-09-30", 346, 0.216, 0.213861, 220, 255, 0.2545),
                        _derived_row(2025, "2025-09-30", 382, 0.240, 0.237624, 240, 270, 0.2583),
                    ],
                },
            ],
        }
    ]


@pytest.fixture
def detailed_columns() -> list:
    """A ``details=true`` columns array, where each entry is an object."""
    return [
        {"caption": name, "negMult": 1, "order": index, "sumType": 0,
         "displayLevel": 0, "internalId": index, "parentId": None,
         "dexId": None, "format": "int", "displayRow": True}
        for index, name in enumerate(AAPL_PL_COLUMNS)
    ]
