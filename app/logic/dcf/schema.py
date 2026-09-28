"""Concept-to-column resolution against the live Sharadar response.

Sharadar appends indicators over time and explicitly tells integrators not to
assume a column order, so no column name is hardcoded at a call site. Every
line item the model needs is declared here as a :class:`Concept` with an
ordered list of plausible column names, and resolved against the columns that
actually came back.

Two failure modes are treated very differently:

* A **required** concept that matches nothing raises
  :class:`SchemaResolutionError`, which is BLOCKING — the model cannot be
  computed, so it is not rendered.
* An **optional** concept that matches nothing is recorded as "not reported"
  and displayed as such. It is never coerced to zero.

When several aliases match, the first in the list wins and the alternates are
recorded on the map so the validation panel can show them.

Note that resolution proves a *column* exists, not that it holds a value.
Sharadar returns a fixed set of indicators for every company, so a firm that
does not operate a classified balance sheet resolves ``Current Assets``
successfully and then reports ``NaN`` in it — which is the honest answer and is
handled downstream, not here.
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
            + (f" (classified as: {template})" if template else "")
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
    #: Concepts sharing a group satisfy the requirement collectively, for a
    #: line item that can legitimately arrive under either of two forms.
    #: Currently unused — Sharadar publishes exactly one column per concept.
    required_group: str | None = None
    note: str = ""


# Aliases are Sharadar's published indicator titles, confirmed against live
# AAPL and JPM SF1 responses. Alternates cover spellings Sharadar has used
# previously; the first name in each list is the current one.
CONCEPTS: tuple[Concept, ...] = (
    # --- Income statement -------------------------------------------------
    Concept("revenue", "Revenue", "PL", ("Revenues", "Revenue"), required=True),
    Concept("cost_of_revenue", "Cost of Revenue", "PL", ("Cost of Revenue",),
            note="Sharadar reports this as a positive magnitude; the client "
                 "negates it on ingest so costs stay negative."),
    Concept("gross_profit", "Gross Profit", "PL", ("Gross Profit",)),
    Concept("operating_expenses", "Operating Expenses", "PL", ("Operating Expenses",),
            note="Negated on ingest, like cost of revenue."),
    Concept("ebit", "Operating Income (EBIT)", "PL", ("Operating Income",),
            required=True,
            note="Sharadar's `opinc` (gross profit less operating expenses), not "
                 "its `ebit`, which adds back tax and interest to net income and "
                 "so carries non-operating items. The latter is kept as a "
                 "cross-check."),
    Concept("interest_expense_gross", "Interest Expense", "PL", ("Interest Expense",),
            required=True,
            note="Negated on ingest. Sharadar reports 0 where a company does not "
                 "disclose interest separately, which is not the same as nil."),
    Concept("pretax_income", "Pretax Income", "PL", ("Earnings before Tax",),
            required=True),
    Concept("tax_expense", "Income Tax Expense", "PL", ("Income Tax Expense",),
            required=True,
            note="Negated on ingest, so a genuine tax benefit reads positive here "
                 "and tax expense is its negation."),
    Concept("net_income", "Net Income", "PL", ("Net Income",), required=True),
    Concept("eps_basic", "Earnings Per Share, Basic", "PL",
            ("Earnings per Basic Share",)),
    Concept("eps_diluted", "Earnings Per Share, Diluted", "PL",
            ("Earnings per Diluted Share",)),
    Concept("net_income_nci", "Net Income to Non-Controlling Interests", "PL",
            ("Net Income to Non-Controlling Interests",),
            note="Cross-checks the derived minority interest: a balance-sheet NCI "
                 "with no NCI income, or the reverse, points at a residual that is "
                 "really a data error."),
    Concept("preferred_dividends", "Preferred Dividends", "PL",
            ("Preferred Dividends Income Statement Impact",),
            note="SF1 has no preferred equity balance, so a non-zero dividend is the "
                 "only sign that the bridge is missing a claim senior to common."),
    Concept("derived_ebit_reported", "EBIT (Sharadar definition)", "PL",
            ("Earning Before Interest & Taxes (EBIT)",),
            note="Net income plus tax and interest. Differs from operating income "
                 "by non-operating items; shown so the gap is visible."),

    # --- Balance sheet ----------------------------------------------------
    Concept("cash", "Cash & Cash Equivalents", "BS", ("Cash and Equivalents",),
            required=True),
    Concept("short_term_investments", "Short-Term Investments", "BS",
            ("Investments Current",)),
    Concept("total_current_assets", "Total Current Assets", "BS", ("Current Assets",),
            required=True,
            note="Not reported by roughly a fifth of companies — mostly financial "
                 "firms — which do not operate a classified balance sheet."),
    Concept("total_current_liabilities", "Total Current Liabilities", "BS",
            ("Current Liabilities",), required=True),
    Concept("short_term_debt", "Short-Term Debt", "BS", ("Debt Current",),
            note="Sharadar combines short-term borrowings and the current portion "
                 "of long-term debt into one line."),
    Concept("long_term_debt", "Long-Term Debt", "BS", ("Debt Non-Current",)),
    Concept("total_assets", "Total Assets", "BS", ("Total Assets",), required=True),
    Concept("total_liabilities", "Total Liabilities", "BS", ("Total Liabilities",),
            required=True),
    Concept("total_equity", "Total Equity", "BS",
            ("Shareholders Equity Attributable to Parent",), required=True),
    Concept("minority_interest", "Minority Interest", "BS",
            ("Minority Interest (derived)",),
            note="Sharadar has no balance-sheet line for this, but its equity "
                 "figure is attributable to the parent, so the client derives it "
                 "as assets less liabilities less equity."),
    Concept("long_term_investments", "Long-Term Investments", "BS",
            ("Investments Non-Current",),
            note="Non-operating asset; only enters the bridge if the user opts in. "
                 "Mixes marketable securities with equity-method stakes and "
                 "restricted balances, all at book value."),
    Concept("derived_total_debt", "Total Debt (Sharadar aggregate)", "BS",
            ("Total Debt",),
            note="Cross-checked against the sum of the current and non-current "
                 "components."),

    # --- Cash flow --------------------------------------------------------
    Concept("dna", "Depreciation & Amortisation", "CF",
            ("Depreciation Amortization & Accretion",), required=True),
    Concept("sbc", "Stock-Based Compensation", "CF", ("Share Based Compensation",)),
    Concept("cfo", "Net Cash from Operating Activities", "CF",
            ("Net Cash Flow from Operations",), required=True),
    Concept("capex_acquisition", "Capital Expenditure", "CF",
            ("Capital Expenditure",), required=True,
            note="Already signed negative for cash spent, and already net of "
                 "disposals, so there is no gross/net choice to make."),
    Concept("acquisitions", "Business Acquisitions (net)", "CF",
            ("Net Cash Flow - Business Acquisitions and Disposals",),
            note="Signed negative for cash spent. Capex excludes it, while reported "
                 "revenue growth includes what was bought, so a serial acquirer's "
                 "history overstates organic growth relative to its capex."),
    Concept("cfi", "Net Cash from Investing Activities", "CF",
            ("Net Cash Flow from Investing",), required=True),
    Concept("cff", "Net Cash from Financing Activities", "CF",
            ("Net Cash Flow from Financing",), required=True),
    Concept("fx_effect", "Effect of Foreign Exchange Rates", "CF",
            ("Effect of Exchange Rate Changes on Cash",)),
    Concept("net_change_in_cash", "Net Change in Cash", "CF",
            ("Net Cash Flow / Change in Cash & Cash Equivalents",), required=True),

    # --- Derived (cross-checks only, never a primary source) --------------
    Concept("ebitda", "EBITDA", "DERIVED",
            ("Earnings Before Interest Taxes & Depreciation Amortization (EBITDA)",)),
    Concept("derived_fcf", "Free Cash Flow (Sharadar derived)", "DERIVED",
            ("Free Cash Flow",),
            note="Operating cash flow less capex, on Sharadar's own definition. "
                 "Reference only — never feeds the model."),
    Concept("fx_to_usd", "Reporting currency per USD (Sharadar)", "DERIVED",
            ("Foreign Currency to USD Exchange Rate",),
            note="Sharadar's `fxusd`, per period. Quoted as units of the reporting "
                 "currency per US dollar — TSM reads about 31 — the inverse of a "
                 "USD-per-unit spot quote; confirmed against `revenueusd = revenue ÷ "
                 "fxusd`. 1 for a USD reporter."),
    Concept("derived_working_capital", "Working Capital (Sharadar derived)", "DERIVED",
            ("Working Capital",),
            note="Current assets less current liabilities, before the cash and "
                 "debt exclusions the model applies."),
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
