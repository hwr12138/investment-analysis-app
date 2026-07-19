import math

import pandas as pd

from app.defaults import DEFAULTS, Default
from app.provenance import Origin, Resolved

# Where a derived assumption comes from when the filings support it.
HISTORICAL_MEAN = "simfin:historical-mean"

# Balance sheets these templates file are unclassified — no current/non-current
# split — so working capital genuinely doesn't exist for them.
UNCLASSIFIED_BS_TEMPLATES = ('BANKS', 'INSURANCES')

# --- Series Helpers ---

# Series from market_data.fetch_simfin_financials run most-recent-first, so a
# plain .pct_change()/.diff() walks backwards through time and silently inverts
# the sign of every result. These wrap the negative period offset that reads
# them forwards. The oldest year has no predecessor and comes back NaN, which
# .mean() skips.

def yoy_growth(series: pd.Series) -> pd.Series:
    """Year-over-year percentage change of a most-recent-first series."""
    return series.pct_change(-1, fill_method=None)

def yoy_change(series: pd.Series) -> pd.Series:
    """Year-over-year absolute change of a most-recent-first series."""
    return series.diff(-1)

# --- DCF Calculations ---

def _resolve(computed, default: Default, unavailable=None) -> Resolved:
    """Classifies one derived assumption against its hardcoded default.

    Preserves exactly the numeric outcomes of the guards this replaced — a
    value is still swapped for the default in precisely the same cases — and
    only records which case fired. That separation matters: a margin the
    filings never reported and a margin that came back at 300% both leave the
    default in the model, but only the second says something about the company.

    `unavailable` overrides the classification when there's nothing to compute,
    for line items a given filer legitimately doesn't report.
    """
    if computed is None or pd.isna(computed):
        origin, detail = unavailable or (
            Origin.INSUFFICIENT_DATA, "the filings don't report enough history to derive this"
        )
        return default.fallback(origin, detail)

    if math.isinf(computed):
        return default.fallback(
            Origin.IMPLAUSIBLE,
            "the historicals imply an infinite value, which points to a zero-denominator year",
        )

    if default.bounds is not None:
        low, high = default.bounds
        if computed < low or computed > high:
            return default.fallback(
                Origin.IMPLAUSIBLE,
                f"the historicals average {default.format(computed)}, outside the plausible "
                f"{default.format(low)} to {default.format(high)} band",
            )

    return Resolved.observed(computed, HISTORICAL_MEAN)


def _implied_exit_multiple(data: dict, ebit: pd.Series, dna: pd.Series, debt: pd.Series) -> Resolved:
    """Current EV/EBITDA, or the default when it can't be computed.

    Never stands a missing input up as 1: doing so yields a multiple roughly
    equal to enterprise value, which clears every downstream plausibility check
    while meaning nothing.
    """
    default = DEFAULTS['exit_multiple']
    cash = data['cash']
    if ebit.empty or dna.empty or debt.empty or cash.empty:
        return default.fallback(
            Origin.INSUFFICIENT_DATA, "EBIT, D&A, debt or cash is missing from the filings"
        )

    # _pick leaves a year NaN when SimFin files none of its candidate labels,
    # so a non-empty series still isn't a guarantee of a number.
    ebitda = ebit.iloc[0] + dna.iloc[0]
    if pd.isna(ebitda):
        return default.fallback(Origin.INSUFFICIENT_DATA, "the latest EBITDA isn't reported")
    if ebitda <= 0:
        return default.fallback(
            Origin.IMPLAUSIBLE,
            f"the latest EBITDA is {ebitda:,.0f}, so a multiple of it carries no meaning",
        )

    ev = (data['shares'] * data['current_price']) + debt.iloc[0] - cash.iloc[0]
    if pd.isna(ev):
        return default.fallback(
            Origin.INSUFFICIENT_DATA,
            "enterprise value can't be built from the reported shares, debt and cash",
        )

    multiple = ev / ebitda
    if pd.isna(multiple) or multiple <= 0:
        return default.fallback(
            Origin.IMPLAUSIBLE, f"the company currently trades at {multiple:.1f}x EV/EBITDA"
        )
    return Resolved.observed(multiple, "simfin:current-ev-ebitda")


def calculate_historical_averages(data: dict) -> dict:
    """Derives default assumptions from historical data.

    Returns a dict of Resolved, one per assumption, so a caller can tell a
    margin drawn from seven years of filings apart from the hardcoded stand-in
    sitting in the same slot.
    """
    if not data:
        return {}

    rev = data['revenue']
    ebit = data['ebit']
    tax = data['tax_provision']
    pretax = data['pretax_income']
    dna = data['dna']
    capex = data['capex']
    nwc = data['nwc']
    debt = data['total_debt']
    interest = data['interest_expense']

    # Growth & Margins
    rev_growth = _resolve(
        yoy_growth(rev).mean() if len(rev) > 1 else None, DEFAULTS['rev_growth'])

    ebit_margin = _resolve(
        (ebit / rev).mean() if not rev.empty else None, DEFAULTS['ebit_margin'])

    # A tax rate outside 0-100% is almost always a loss year dividing a benefit
    # by negative pretax income, so the rejected figure is worth reporting.
    tax_rate = _resolve(
        (tax / pretax).mean() if not pretax.empty else None, DEFAULTS['tax_rate'])

    # Capital Intensity
    dna_margin = _resolve(
        (dna / rev).mean() if not rev.empty else None, DEFAULTS['dna_margin'])

    capex_margin = _resolve(
        (capex / rev).mean() if not rev.empty else None, DEFAULTS['capex_margin'])

    # Cost of debt
    cost_of_debt = _resolve(
        (interest / debt).mean() if not debt.empty else None, DEFAULTS['cost_of_debt'])

    # NWC change. Banks and insurers file an unclassified balance sheet, so the
    # absence here is a property of the template rather than a gap in the data.
    template = (data.get('template') or '').upper()
    unclassified = (
        (Origin.NOT_APPLICABLE,
         f"{template.title()} file an unclassified balance sheet, so working capital "
         f"isn't split out — the default is the appropriate stand-in")
        if template in UNCLASSIFIED_BS_TEMPLATES else None
    )
    nwc_margin = _resolve(
        (yoy_change(nwc) / rev).mean() if not rev.empty else None, DEFAULTS['nwc_margin'],
        unavailable=unclassified)

    # Exit Multiple (EV/EBITDA)
    exit_multiple = _implied_exit_multiple(data, ebit, dna, debt)

    return {
        'rev_growth': rev_growth,
        'ebit_margin': ebit_margin,
        'tax_rate': tax_rate,
        'dna_margin': dna_margin,
        'capex_margin': capex_margin,
        'nwc_margin': nwc_margin,
        'cost_of_debt': cost_of_debt,
        'exit_multiple': exit_multiple
    }

def calculate_wacc(risk_free_rate: float, beta: float, erp: float, cost_of_debt: float, tax_rate: float, debt_weight: float, equity_weight: float) -> float:
    """Calculates the Weighted Average Cost of Capital."""
    cost_of_equity = risk_free_rate + (beta * erp)
    after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
    wacc = (equity_weight * cost_of_equity) + (debt_weight * after_tax_cost_of_debt)
    return wacc

def run_unlevered_dcf(data: dict, assumptions: dict, wacc: float, years: int = 5) -> dict:
    """Runs the 5-year Unlevered DCF."""
    latest_rev = data['revenue'].iloc[0] if not data['revenue'].empty else 0

    projected_years = []
    current_rev = latest_rev

    for i in range(1, years + 1):
        current_rev *= (1 + assumptions['rev_growth'])
        ebit = current_rev * assumptions['ebit_margin']
        taxes = ebit * assumptions['tax_rate']
        nopat = ebit - taxes

        dna = current_rev * assumptions['dna_margin']
        capex = current_rev * assumptions['capex_margin']
        change_nwc = current_rev * assumptions['nwc_margin']

        ufcf = nopat + dna - capex - change_nwc
        discount_factor = (1 + wacc) ** i
        pv_ufcf = ufcf / discount_factor

        projected_years.append({
            'Year': i,
            'Revenue': current_rev,
            'EBIT': ebit,
            'NOPAT': nopat,
            '+ D&A': dna,
            '- CapEx': capex,
            '- Change in NWC': change_nwc,
            'UFCF': ufcf,
            'PV of UFCF': pv_ufcf
        })

    df_proj = pd.DataFrame(projected_years)

    # Terminal Value
    terminal_ebitda = projected_years[-1]['EBIT'] + projected_years[-1]['+ D&A']
    terminal_value = terminal_ebitda * assumptions['exit_multiple']
    pv_terminal_value = terminal_value / ((1 + wacc) ** years)

    # Enterprise Value
    enterprise_value = df_proj['PV of UFCF'].sum() + pv_terminal_value

    # Equity Value
    latest_debt = data['total_debt'].iloc[0] if not data['total_debt'].empty else 0
    latest_cash = data['cash'].iloc[0] if not data['cash'].empty else 0
    equity_value = enterprise_value + latest_cash - latest_debt

    implied_share_price = equity_value / data['shares'] if data['shares'] > 0 else 0

    return {
        'projections': df_proj,
        'terminal_value': terminal_value,
        'pv_terminal_value': pv_terminal_value,
        'enterprise_value': enterprise_value,
        'equity_value': equity_value,
        'implied_share_price': implied_share_price
    }
