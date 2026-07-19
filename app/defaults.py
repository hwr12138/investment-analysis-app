"""Every hardcoded stand-in the model can fall back to, in one place.

Each entry is a number the app invents when reality is unavailable — a failed
quote, a line item the filings never reported, a figure that came back absurd.
Scattered through the code these read as incidental; collected here they are
what they actually are: a set of editorial choices that shape every valuation
the app produces, and that deserve to be reviewed as a set.

Each carries its own rationale, so the question "why 5.5%?" has an answer that
lives next to the number rather than in someone's memory. The bounds are the
plausible band a derived value has to clear before it's trusted over the
default; see app.provenance for how a rejection gets classified.
"""

from typing import Optional

from app.provenance import Origin, Resolved


class Default:
    """One stand-in, with everything needed to use, render and explain it."""

    def __init__(self, value: float, label: str, fmt: str = "{:.1%}",
                 bounds: Optional[tuple] = None, rationale: str = ""):
        self.value = value
        self.label = label
        self.fmt = fmt
        self.bounds = bounds
        self.rationale = rationale

    def fallback(self, origin: Origin, detail: str) -> Resolved:
        """Builds the Resolved that stands this default in for real data."""
        return Resolved.fallback(self.value, origin, detail)

    def format(self, value: float) -> str:
        return self.fmt.format(value)


# Market-wide inputs, fetched live by app.data.market_data.
_MARKET = {
    'risk_free_rate': Default(
        0.04, "Risk Free Rate", "{:.2%}",
        rationale="Roughly the 10-year Treasury yield's post-2022 range. Rates move, "
                  "so this staleness is exactly why the live quote is preferred.",
    ),
    'erp': Default(
        0.055, "Equity Risk Premium", "{:.2%}",
        rationale="A long-run average of Damodaran's implied US ERP. The live figure "
                  "has recently run well below this, so the default is not a neutral choice.",
    ),
    'beta': Default(
        1.0, "Beta", "{:.2f}",
        rationale="The market's own beta — assumes the stock moves one-for-one with "
                  "the index, which is the least opinionated guess available.",
    ),
}

# Company assumptions, normally derived from filings by app.logic.
_ASSUMPTIONS = {
    'rev_growth': Default(
        0.05, "Revenue Growth Rate",
        rationale="Approximate long-run nominal GDP growth — a company can't outgrow "
                  "the economy forever, so this is the neutral terminal assumption.",
    ),
    'ebit_margin': Default(
        0.15, "Target EBIT Margin",
        rationale="A broad median for mature US large-caps. Wrong for most individual "
                  "firms in one direction or the other.",
    ),
    'tax_rate': Default(
        0.21, "Tax Rate", bounds=(0.0, 1.0),
        rationale="The US federal statutory corporate rate. Bounded to 0-100% because "
                  "anything outside it means a loss year, not a real tax rate.",
    ),
    'dna_margin': Default(
        0.05, "D&A Margin",
        rationale="Paired with the capex default to imply a steady state where "
                  "investment roughly replaces depreciation.",
    ),
    'capex_margin': Default(
        0.05, "CapEx Margin",
        rationale="Matches the D&A default, i.e. maintenance capex with no expansion.",
    ),
    'nwc_margin': Default(
        0.02, "NWC Change Margin",
        rationale="A modest working-capital drag as revenue grows. Genuinely "
                  "inapplicable to banks and insurers, which report no working capital.",
    ),
    'cost_of_debt': Default(
        0.05, "Cost of Debt", bounds=(0.0, 0.2),
        rationale="Approximately an investment-grade corporate yield. Bounded because a "
                  "derived rate above 20% usually means the debt balance is wrong, not "
                  "that the company borrows at distressed rates.",
    ),
    'exit_multiple': Default(
        10.0, "Exit Multiple (EV/EBITDA)", "{:.1f}x",
        rationale="A conservative mid-cycle EV/EBITDA. Terminal value dominates a "
                  "5-year DCF, so this default drives more of the answer than any other.",
    ),
    'debt_weight': Default(
        0.2, "Debt Weight", "{:.1%}",
        rationale="A moderately levered capital structure, used only when market cap "
                  "and debt are both unavailable to measure the real one.",
    ),
}

DEFAULTS = {**_MARKET, **_ASSUMPTIONS}
