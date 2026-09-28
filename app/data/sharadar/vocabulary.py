"""Sharadar's vocabulary: what its columns, codes and conventions mean.

Everything here is **pure** — no Streamlit, no ``requests``, no I/O — which is
the whole reason it is a module of its own. The logic layer needs to know that
``cor`` is signed positive, what SF1 calls ``sharesbas``, which ``securityadcode``
means a holdings line and how splits compound, and before this split it could
only learn those things by importing the module that also held the HTTP client.
That loaded Streamlit and the network stack into every "pure" logic module.
``tests/test_architecture.py`` now holds the line.

Published here because it is the vendor's vocabulary. What any of it *means* to
a tab — which insider codes count as a decision, which multiples are usable —
belongs in the logic layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

#: Columns that are metadata rather than financial line items. Everything else
#: is coerced to numeric; these keep their reported type.
META_COLUMNS = frozenset(
    {
        "Fiscal Period",
        "Fiscal Year",
        "Report Date",
        "Publish Date",
        "Restated",
        "Source",
        "TTM",
        "Value Check",
        "Data Model",
        "Date",
        "Calendar Date",
        "Dimension",
    }
)

#: SF1 indicators signed as positive magnitudes that the engine expects
#: negative. ``capex`` is NOT here — Sharadar already signs it negative
#: (AAPL FY2025: -12,715M, and their own ``fcf`` = ``ncfo + capex``).
SIGN_FLIPPED = frozenset({"cor", "opex", "sgna", "rnd", "taxexp", "intexp"})

#: Indicators Sharadar documents as "where this item is not contained on the
#: company consolidated financial statements … the value of 0 is used". A zero
#: in any of these may mean *not disclosed* rather than *nil*, which matters
#: concretely: AAPL's ``intexp`` is 0, so the FCFF interest add-back vanishes
#: and the implied cost of debt reads 0%.
ZERO_FILLED_INDICATORS = frozenset(
    {
        "accoci", "capex", "cashneq", "cor", "debt", "debtc", "debtnc",
        "deferredrev", "depamor", "deposits", "divyield", "dps", "intangibles",
        "intexp", "inventory", "investments", "investmentsc", "investmentsnc",
        "ncfbus", "ncfcommon", "ncfdebt", "ncfdiv", "ncff", "ncfi", "ncfinv",
        "ncfx", "netincdis", "netincnci", "payables", "ppnenet", "prefdivis",
        "receivables", "revenue", "rnd", "sgna", "taxassets", "taxexp",
        "taxliabilities",
    }
)


#: The currency every SEP and SFP price is quoted in. SF1's un-suffixed
#: indicators are in the company's *reporting* currency instead (the ``usd``
#: variants and ``fxusd`` exist for conversion), so a foreign filer's statements
#: and its price are in different units unless someone converts them.
PRICE_CURRENCY = "USD"


#: SIC codes 6000-6999 are Finance, Insurance and Real Estate. Banks and
#: insurers do not have a meaningful operating-cash-flow-less-capex FCFF, so
#: they are rejected before anything is computed.
FINANCIAL_SIC_RANGE = (6000, 6999)

# ---------------------------------------------------------------------------
# SF1 statement partition
# ---------------------------------------------------------------------------

#: ``(indicator, caption, statement)`` for every SF1 indicator the app surfaces,
#: in display order. Captions are Sharadar's own ``title`` values from
#: ``SHARADAR/INDICATORS`` (trailing whitespace stripped), so the raw figures tab
#: reads as the publisher names things and ``schema.CONCEPTS`` can alias against
#: human-readable strings.
#:
#: The partition follows the ``[Income Statement]``/``[Balance Sheet]``/
#: ``[Cash Flow Statement]``/``[Metrics]`` prefix in each indicator's published
#: description, with one deliberate move: ``ebt`` is published under Metrics but
#: is the income statement's pretax income, which is where the model reads it.
#: Frozen here rather than fetched at runtime — the assignment is stable and a
#: live dependency on the dictionary endpoint would be a needless failure mode.
SF1_LAYOUT: tuple[tuple[str, str, str], ...] = (
    # --- Income statement -------------------------------------------------
    ("revenue", "Revenues", "PL"),
    ("cor", "Cost of Revenue", "PL"),
    ("gp", "Gross Profit", "PL"),
    ("sgna", "Selling General and Administrative Expense", "PL"),
    ("rnd", "Research and Development Expense", "PL"),
    ("opex", "Operating Expenses", "PL"),
    ("opinc", "Operating Income", "PL"),
    ("ebit", "Earning Before Interest & Taxes (EBIT)", "PL"),
    ("intexp", "Interest Expense", "PL"),
    ("ebt", "Earnings before Tax", "PL"),
    ("taxexp", "Income Tax Expense", "PL"),
    ("consolinc", "Consolidated Income", "PL"),
    ("netincnci", "Net Income to Non-Controlling Interests", "PL"),
    ("netinc", "Net Income", "PL"),
    ("prefdivis", "Preferred Dividends Income Statement Impact", "PL"),
    ("netinccmn", "Net Income Common Stock", "PL"),
    ("netincdis", "Net Loss Income from Discontinued Operations", "PL"),
    ("eps", "Earnings per Basic Share", "PL"),
    ("epsdil", "Earnings per Diluted Share", "PL"),
    ("shareswa", "Weighted Average Shares", "PL"),
    ("shareswadil", "Weighted Average Shares Diluted", "PL"),
    ("dps", "Dividends per Basic Common Share", "PL"),
    # --- Balance sheet ----------------------------------------------------
    ("cashneq", "Cash and Equivalents", "BS"),
    ("investmentsc", "Investments Current", "BS"),
    ("receivables", "Trade and Non-Trade Receivables", "BS"),
    ("inventory", "Inventory", "BS"),
    ("assetsc", "Current Assets", "BS"),
    ("investmentsnc", "Investments Non-Current", "BS"),
    ("ppnenet", "Property Plant & Equipment Net", "BS"),
    ("intangibles", "Goodwill and Intangible Assets", "BS"),
    ("taxassets", "Tax Assets", "BS"),
    ("investments", "Investments", "BS"),
    ("assetsnc", "Assets Non-Current", "BS"),
    ("assets", "Total Assets", "BS"),
    ("payables", "Trade and Non-Trade Payables", "BS"),
    ("deferredrev", "Deferred Revenue", "BS"),
    ("debtc", "Debt Current", "BS"),
    ("liabilitiesc", "Current Liabilities", "BS"),
    ("debtnc", "Debt Non-Current", "BS"),
    ("deposits", "Deposit Liabilities", "BS"),
    ("taxliabilities", "Tax Liabilities", "BS"),
    ("liabilitiesnc", "Liabilities Non-Current", "BS"),
    ("liabilities", "Total Liabilities", "BS"),
    ("debt", "Total Debt", "BS"),
    ("retearn", "Accumulated Retained Earnings (Deficit)", "BS"),
    ("accoci", "Accumulated Other Comprehensive Income", "BS"),
    ("equity", "Shareholders Equity Attributable to Parent", "BS"),
    # --- Cash flow --------------------------------------------------------
    ("depamor", "Depreciation Amortization & Accretion", "CF"),
    ("sbcomp", "Share Based Compensation", "CF"),
    ("ncfo", "Net Cash Flow from Operations", "CF"),
    ("capex", "Capital Expenditure", "CF"),
    ("ncfbus", "Net Cash Flow - Business Acquisitions and Disposals", "CF"),
    ("ncfinv", "Net Cash Flow - Investment Acquisitions and Disposals", "CF"),
    ("ncfi", "Net Cash Flow from Investing", "CF"),
    ("ncfdebt", "Issuance (Repayment) of Debt Securities", "CF"),
    ("ncfcommon", "Issuance (Purchase) of Equity Shares", "CF"),
    ("ncfdiv", "Payment of Dividends & Other Cash Distributions", "CF"),
    ("ncff", "Net Cash Flow from Financing", "CF"),
    ("ncfx", "Effect of Exchange Rate Changes on Cash", "CF"),
    ("ncf", "Net Cash Flow / Change in Cash & Cash Equivalents", "CF"),
    # --- Derived: metrics and entity-level figures ------------------------
    ("ebitda", "Earnings Before Interest Taxes & Depreciation Amortization (EBITDA)", "DERIVED"),
    ("fcf", "Free Cash Flow", "DERIVED"),
    ("workingcapital", "Working Capital", "DERIVED"),
    ("invcap", "Invested Capital", "DERIVED"),
    ("tangibles", "Tangible Asset Value", "DERIVED"),
    ("sharesbas", "Shares (Basic)", "DERIVED"),
    ("sharefactor", "Share Factor", "DERIVED"),
    ("price", "Share Price (Adjusted Close)", "DERIVED"),
    ("marketcap", "Market Capitalization", "DERIVED"),
    ("ev", "Enterprise Value", "DERIVED"),
    ("evebit", "Enterprise Value over EBIT", "DERIVED"),
    ("evebitda", "Enterprise Value over EBITDA", "DERIVED"),
    ("pe1", "Price to Earnings Ratio", "DERIVED"),
    ("pb", "Price to Book Value", "DERIVED"),
    ("ps1", "Price to Sales Ratio", "DERIVED"),
    ("grossmargin", "Gross Margin", "DERIVED"),
    ("ebitdamargin", "EBITDA Margin", "DERIVED"),
    ("netmargin", "Profit Margin", "DERIVED"),
    ("ros", "Return on Sales", "DERIVED"),
    ("roe", "Return on Average Equity", "DERIVED"),
    ("roa", "Return on Average Assets", "DERIVED"),
    ("roic", "Return on Invested Capital", "DERIVED"),
    ("currentratio", "Current Ratio", "DERIVED"),
    ("de", "Debt to Equity Ratio", "DERIVED"),
    ("bvps", "Book Value per Share", "DERIVED"),
    ("fcfps", "Free Cash Flow per Share", "DERIVED"),
    ("sps", "Sales per Share", "DERIVED"),
    ("divyield", "Dividend Yield", "DERIVED"),
    ("payoutratio", "Payout Ratio", "DERIVED"),
    ("assetturnover", "Asset Turnover", "DERIVED"),
    ("fxusd", "Foreign Currency to USD Exchange Rate", "DERIVED"),
)

#: Caption for the non-controlling interest column derived below. SF1 has no
#: balance-sheet NCI line, but its ``equity`` is explicitly "attributable to
#: parent", so the residual recovers it.
MINORITY_INTEREST_CAPTION = "Minority Interest (derived)"

SF1_CAPTIONS: dict[str, str] = {ind: cap for ind, cap, _ in SF1_LAYOUT}

#: Reverse of :data:`SF1_CAPTIONS`, for turning a resolved column name back into
#: the Sharadar indicator it came from.
CAPTION_TO_INDICATOR: dict[str, str] = {cap: ind for ind, cap, _ in SF1_LAYOUT}


def is_zero_filled(caption: str) -> bool:
    """Whether a resolved column is one Sharadar writes 0 into when absent.

    Lets the validation layer ask the question without importing the indicator
    vocabulary. See :data:`ZERO_FILLED_INDICATORS`.
    """
    indicator = CAPTION_TO_INDICATOR.get(caption)
    return indicator is not None and indicator in ZERO_FILLED_INDICATORS



@dataclass(frozen=True)
class StatementSet:
    """Statements for one company at one SF1 dimension.

    ``dimension`` and the derived ``asreported``/``ttm`` flags are recorded so
    the validation layer can prove a single model run never mixes restated with
    as-filed figures.
    """

    ticker: str
    permaticker: int
    name: str
    currency: str
    template: str
    dimension: str
    period: str
    ttm: bool
    asreported: bool
    sector: str = ""
    industry: str = ""
    sic_code: int | None = None
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)

    def frame(self, statement: str) -> pd.DataFrame:
        """Returns the frame for ``PL``/``BS``/``CF``/``DERIVED``, or an empty one."""
        return self.frames.get(statement.upper(), pd.DataFrame())

    @property
    def is_financial_template(self) -> bool:
        """True for banks, insurers, and the rest of SIC division H.

        SIC code is authoritative where present: Sharadar's ``sector`` field is
        a vendor classification that can be null, but every company in SF1 that
        files with the SEC has a SIC code. Both are checked.
        """
        low, high = FINANCIAL_SIC_RANGE
        if self.sic_code is not None and low <= int(self.sic_code) <= high:
            return True
        return self.template.strip().lower() in {
            "financial services",
            "finance insurance and real estate",
        }



# ---------------------------------------------------------------------------
# Corporate actions
# ---------------------------------------------------------------------------

#: ``action`` values seen in live ACTIONS responses (AAPL, KO, GOOGL, AMZN):
#: ``dividend``, ``split``, ``listed``, ``initiated``, ``relation``,
#: ``acquisitionof``, ``tickerchangefrom``, ``tickerchangeto``. The list is not
#: closed — Sharadar adds action types — so callers filter for the ones they
#: understand rather than exhausting it.
DIVIDEND_ACTION = "dividend"
SPLIT_ACTION = "split"


def split_factors(actions: pd.DataFrame, when: pd.Series) -> pd.Series:
    """For each date in ``when``, the product of every split effective after it.

    One where nothing split since. This is the bridge between a figure **as
    filed** and the same figure on today's basis, and it exists here rather than
    in either consumer because the two use it in **opposite directions**:

    * a *price* filed before a split is **divided** by the factor — Tesla
      insiders filed November 2021 sales at $981-$1,237 against a split-adjusted
      close of $338-$410;
    * a *share count* filed before a split is **multiplied** by it — BlackRock's
      Tesla position reads 7,720 then 38,020 across the 2020 five-for-one.

    Getting the direction wrong is as bad as not applying it, so both callers
    name which one they are doing.

    ``actions`` is a frame as :func:`fetch_actions` returns it; a split row's
    ``value`` is the ratio. Anything missing or unparseable yields a factor of
    one rather than dropping the row.
    """
    ones = pd.Series(1.0, index=when.index, dtype="float64")
    if actions is None or actions.empty:
        return ones
    if not {"action", "date", "value"}.issubset(actions.columns):
        return ones

    splits = actions[actions["action"] == SPLIT_ACTION]
    ratios = pd.to_numeric(splits["value"], errors="coerce")
    dates = pd.to_datetime(splits["date"], errors="coerce")
    keep = ratios.gt(0) & dates.notna()
    if not keep.any():
        return ones

    ratios, dates = ratios[keep], dates[keep]
    order = dates.argsort()
    ratios, dates = ratios.iloc[order], dates.iloc[order]

    # Reverse-cumulative product: entry *i* is the product of every split from
    # *i* onwards, so a date preceding *i* splits carries `trailing[i]`.
    trailing = ratios[::-1].cumprod()[::-1].tolist() + [1.0]
    target = pd.to_datetime(when, errors="coerce")
    position = dates.values.searchsorted(target.values, side="left")
    return pd.Series(
        [trailing[int(i)] for i in position], index=when.index, dtype="float64"
    )


# ---------------------------------------------------------------------------
# Insider transactions (SF2)
# ---------------------------------------------------------------------------

#: The literal SF2 writes into ``isdirector``/``isofficer``/``istenpercentowner``.
SF2_FLAG_YES = "Y"

#: ``securityadcode`` is two letters: N(on-derivative) or D(erivative), then
#: A(cquired) or D(isposed). A **bare** ``N`` or ``D`` with no second letter is a
#: holdings line rather than a trade — a Form 3, or a Form 4 line restating a
#: position. On the live sample that mapping was exact: every bare code carried a
#: null ``transactioncode`` and ``transactionshares`` of 0, and they are **28% of
#: all rows**.
#:
#: Published here because it is the vendor's vocabulary. What any of it *means*
#: to a tab — which codes count as a decision — belongs in the logic layer, the
#: same split as :data:`DIVIDEND_ACTION` against ``growth.METRICS``.
SF2_NON_DERIVATIVE = "N"
SF2_DERIVATIVE = "D"
SF2_ACQUIRED = "A"
SF2_DISPOSED = "D"


# ---------------------------------------------------------------------------
# Institutional holdings (SF3)
# ---------------------------------------------------------------------------

#: Security types SF3 reports a position in. ``SHR`` is common stock and is the
#: only one any consumer here reads today; the rest are published so a caller
#: names the vendor's letter rather than a bare string.
SF3_SHARES = "SHR"
SF3_CALLS = "CLL"
SF3_PUTS = "PUT"
