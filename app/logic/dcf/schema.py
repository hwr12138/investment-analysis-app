"""Concept-to-column resolution against the live SimFin response.

SimFin renames columns across versions and across statement templates, so no
column name is hardcoded at a call site. Every line item the model needs is
declared here as a :class:`Concept` with an ordered list of plausible column
names, and resolved against the ``columns`` array that actually came back.

Two failure modes are treated very differently:

* A **required** concept that matches nothing raises
  :class:`SchemaResolutionError`, which is BLOCKING — the model cannot be
  computed, so it is not rendered.
* An **optional** concept that matches nothing is recorded as "not reported"
  and displayed as such. It is never coerced to zero.

When several aliases match, the first in the list wins and the alternates are
recorded on the map so the validation panel can show them. That matters where
the alternates disagree — AAPL reports both ``Interest Expense`` and
``Interest Expense, net``, and which one feeds the FCFF interest add-back is a
decision the user should see rather than one the resolver makes quietly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


class SchemaResolutionError(Exception):
    """A required concept could not be resolved. BLOCKING.

    Names the concept, the company, and the columns that *were* found, so the
    cause is diagnosable without reading the raw payload.
    """

    def __init__(
        self,
        concept_label: str,
        ticker: str,
        statement: str,
        aliases: tuple[str, ...],
        available: list[str],
        template: str = "",
    ) -> None:
        self.concept_label = concept_label
        self.ticker = ticker
        self.statement = statement
        self.aliases = aliases
        self.available = available
        self.template = template
        super().__init__(
            f"Could not resolve '{concept_label}' on the {statement} statement for "
            f"{ticker}"
            + (f" (SimFin template: {template})" if template else "")
            + f". Tried: {', '.join(aliases)}."
        )


@dataclass(frozen=True)
class Concept:
    """One line item the model needs, and the column names it may arrive under."""

    key: str
    label: str
    statement: str
    aliases: tuple[str, ...]
    required: bool = False
    #: Concepts sharing a group satisfy the requirement collectively — capex
    #: arrives as either a gross acquisition line or a net change line, and
    #: exactly one of them being present is fine.
    required_group: str | None = None
    note: str = ""


# Alias lists are ordered by preference. The first name in each is one I
# confirmed against a live AAPL/JPM response; the rest cover the other
# statement templates and older column spellings.
CONCEPTS: tuple[Concept, ...] = (
    # --- Income statement -------------------------------------------------
    Concept("revenue", "Revenue", "PL",
            ("Revenue", "Net Revenue", "Sales & Services Revenue", "Total Revenue"),
            required=True),
    Concept("cost_of_revenue", "Cost of Revenue", "PL",
            ("Cost of revenue", "Cost of Revenue", "Cost of Goods & Services"),
            note="SimFin spells this with a lowercase 'r'."),
    Concept("gross_profit", "Gross Profit", "PL", ("Gross Profit",)),
    Concept("operating_expenses", "Operating Expenses", "PL", ("Operating Expenses",)),
    Concept("ebit", "Operating Income (EBIT)", "PL",
            ("Operating Income (Loss)", "Operating Income"), required=True),
    Concept("interest_expense_gross", "Interest Expense (gross)", "PL",
            ("Interest Expense",), required_group="interest"),
    Concept("interest_expense_net", "Interest Expense (net)", "PL",
            ("Interest Expense, net", "Total Interest Expense"), required_group="interest"),
    Concept("pretax_income", "Pretax Income", "PL",
            ("Pretax Income (Loss)", "Pretax Income (Loss), Adjusted"), required=True),
    Concept("tax_expense", "Income Tax Expense", "PL",
            ("Income Tax (Expense) Benefit, net", "Income Tax (Expense) Benefit"),
            required=True,
            note="SimFin signs this as a benefit, so tax expense is its negation."),
    Concept("net_income", "Net Income", "PL",
            ("Net Income", "Net Income Available to Common Shareholders"), required=True),

    # --- Balance sheet ----------------------------------------------------
    Concept("cash", "Cash & Cash Equivalents", "BS", ("Cash & Cash Equivalents",),
            required=True),
    Concept("short_term_investments", "Short-Term Investments", "BS",
            ("Short Term Investments",)),
    Concept("cash_aggregate", "Cash & Short-Term Investments (aggregate)", "BS",
            ("Cash, Cash Equivalents & Short Term Investments",),
            note="Cross-checked against the sum of its components."),
    Concept("total_current_assets", "Total Current Assets", "BS",
            ("Total Current Assets",), required=True),
    Concept("total_current_liabilities", "Total Current Liabilities", "BS",
            ("Total Current Liabilities",), required=True),
    Concept("short_term_debt", "Short-Term Debt", "BS",
            ("Short Term Debt", "Short Term Borrowings")),
    Concept("current_portion_ltd", "Current Portion of Long-Term Debt", "BS",
            ("Current Portion of Long Term Debt",)),
    Concept("long_term_debt", "Long-Term Debt", "BS",
            ("Long Term Debt", "Long Term Borrowings")),
    Concept("total_assets", "Total Assets", "BS", ("Total Assets",), required=True),
    Concept("total_liabilities", "Total Liabilities", "BS", ("Total Liabilities",),
            required=True),
    Concept("total_equity", "Total Equity", "BS", ("Total Equity",), required=True),
    Concept("total_liabilities_and_equity", "Total Liabilities & Equity", "BS",
            ("Total Liabilities & Equity",)),
    Concept("minority_interest", "Minority Interest", "BS", ("Minority Interest",)),
    Concept("preferred_equity", "Preferred Equity", "BS", ("Preferred Equity",)),
    Concept("long_term_investments", "Long-Term Investments", "BS",
            ("Long Term Investments", "Long Term Marketable Securities"),
            note="Non-operating asset; only enters the bridge if the user opts in."),

    # --- Cash flow --------------------------------------------------------
    Concept("cf_net_income", "Net Income (cash flow statement)", "CF",
            ("Net Income", "Net Income/Starting Line")),
    Concept("dna", "Depreciation & Amortisation", "CF",
            ("Depreciation & Amortization",), required=True),
    Concept("sbc", "Stock-Based Compensation", "CF", ("Stock-Based Compensation",),
            note="Frequently unpopulated by SimFin even for companies that report it."),
    Concept("non_cash_items", "Non-Cash Items", "CF", ("Non-Cash Items",),
            note="A superset of SBC; never an SBC figure on its own."),
    Concept("cf_change_in_wc", "Change in Working Capital (cash flow statement)", "CF",
            ("Change in Working Capital",)),
    Concept("cfo", "Net Cash from Operating Activities", "CF",
            ("Cash from Operating Activities",), required=True),
    Concept("capex_acquisition", "Acquisition of Fixed Assets & Intangibles", "CF",
            ("Acquisition of Fixed Assets & Intangibles", "Purchase of Fixed Assets",
             "Capital Expenditures"),
            required_group="capex"),
    Concept("capex_net_change", "Change in Fixed Assets & Intangibles", "CF",
            ("Change in Fixed Assets & Intangibles", "Change in Fixed Assets and Intangibles"),
            required_group="capex",
            note="Net of disposals — an approximation for capex, and labelled as one."),
    Concept("capex_disposals", "Disposition of Fixed Assets & Intangibles", "CF",
            ("Disposition of Fixed Assets & Intangibles",)),
    Concept("cfi", "Net Cash from Investing Activities", "CF",
            ("Cash from Investing Activities",), required=True),
    Concept("cff", "Net Cash from Financing Activities", "CF",
            ("Cash from Financing Activities",), required=True),
    Concept("fx_effect", "Effect of Foreign Exchange Rates", "CF",
            ("Effect of Foreign Exchange Rates",)),
    Concept("net_change_in_cash", "Net Change in Cash", "CF",
            ("Net Changes in Cash", "Net Change in Cash"), required=True),

    # --- Derived (cross-checks only, never a primary source) --------------
    Concept("ebitda", "EBITDA", "DERIVED", ("EBITDA",)),
    Concept("eps_basic", "Earnings Per Share, Basic", "DERIVED",
            ("Earnings Per Share, Basic",)),
    Concept("eps_diluted", "Earnings Per Share, Diluted", "DERIVED",
            ("Earnings Per Share, Diluted",)),
    Concept("derived_total_debt", "Total Debt (SimFin derived)", "DERIVED", ("Total Debt",)),
    Concept("derived_fcf", "Free Cash Flow (SimFin derived)", "DERIVED",
            ("Free Cash Flow",),
            note="Observed to exceed operating cash flow for companies with positive "
                 "capex. Reference only — never feeds the model."),
    Concept("derived_operating_margin", "Operating Margin (SimFin derived)", "DERIVED",
            ("Operating Margin",)),
)

CONCEPTS_BY_KEY: dict[str, Concept] = {c.key: c for c in CONCEPTS}


@dataclass(frozen=True)
class SchemaMap:
    """The resolved concept-to-column mapping for one company.

    Rendered in the validation panel and written to the export so the matching
    is auditable rather than implicit.
    """

    ticker: str
    template: str
    columns: dict[str, str] = field(default_factory=dict)
    alternates: dict[str, tuple[str, ...]] = field(default_factory=dict)
    unresolved: dict[str, str] = field(default_factory=dict)

    def column(self, key: str) -> str | None:
        return self.columns.get(key)

    def has(self, key: str) -> bool:
        return key in self.columns

    def as_frame(self) -> pd.DataFrame:
        """Tabular view for the validation panel and the export."""
        rows = []
        for concept in CONCEPTS:
            resolved = self.columns.get(concept.key)
            rows.append(
                {
                    "Concept": concept.label,
                    "Statement": concept.statement,
                    "Resolved column": resolved or "— not reported —",
                    "Status": "resolved" if resolved else "not reported",
                    "Required": "yes" if concept.required or concept.required_group else "no",
                    "Other matches": ", ".join(self.alternates.get(concept.key, ())),
                    "Note": concept.note,
                }
            )
        return pd.DataFrame(rows)

    @property
    def ambiguous(self) -> dict[str, tuple[str, ...]]:
        """Concepts where more than one candidate column was present."""
        return {k: v for k, v in self.alternates.items() if v}


def resolve_schema(
    frames: dict[str, pd.DataFrame], ticker: str, template: str = ""
) -> SchemaMap:
    """Resolves every concept against the columns actually returned.

    Raises :class:`SchemaResolutionError` for the first required concept — or
    required group — that matches nothing.
    """
    columns: dict[str, str] = {}
    alternates: dict[str, tuple[str, ...]] = {}
    unresolved: dict[str, str] = {}

    available: dict[str, list[str]] = {
        name: list(frame.columns) for name, frame in frames.items()
    }

    for concept in CONCEPTS:
        present = available.get(concept.statement, [])
        matches = [alias for alias in concept.aliases if alias in present]

        if matches:
            columns[concept.key] = matches[0]
            if len(matches) > 1:
                alternates[concept.key] = tuple(matches[1:])
            continue

        unresolved[concept.key] = (
            f"No column matching {concept.label} on the {concept.statement} statement"
        )
        if concept.required:
            raise SchemaResolutionError(
                concept_label=concept.label,
                ticker=ticker,
                statement=concept.statement,
                aliases=concept.aliases,
                available=present,
                template=template,
            )

    # Grouped requirements: at least one member must have resolved.
    groups: dict[str, list[Concept]] = {}
    for concept in CONCEPTS:
        if concept.required_group:
            groups.setdefault(concept.required_group, []).append(concept)

    for group_name, members in groups.items():
        if any(m.key in columns for m in members):
            continue
        first = members[0]
        raise SchemaResolutionError(
            concept_label=f"{group_name} (none of: "
                          f"{', '.join(m.label for m in members)})",
            ticker=ticker,
            statement=first.statement,
            aliases=tuple(a for m in members for a in m.aliases),
            available=available.get(first.statement, []),
            template=template,
        )

    return SchemaMap(
        ticker=ticker,
        template=template,
        columns=columns,
        alternates=alternates,
        unresolved=unresolved,
    )
