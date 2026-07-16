import pandas as pd

# --- DCF Calculations ---

def calculate_historical_averages(data: dict) -> dict:
    """Calculates default assumptions based on historical data."""
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
    rev_growth = rev.pct_change(fill_method=None).mean() if len(rev) > 1 else 0.05
    if pd.isna(rev_growth) or rev_growth == float('inf'): rev_growth = 0.05

    ebit_margin = (ebit / rev).mean() if not rev.empty else 0.15
    if pd.isna(ebit_margin): ebit_margin = 0.15

    tax_rate = (tax / pretax).mean() if not pretax.empty else 0.21
    if pd.isna(tax_rate) or tax_rate < 0 or tax_rate > 1: tax_rate = 0.21

    # Capital Intensity
    dna_margin = (dna / rev).mean() if not rev.empty else 0.05
    if pd.isna(dna_margin): dna_margin = 0.05

    capex_margin = (capex / rev).mean() if not rev.empty else 0.05
    if pd.isna(capex_margin): capex_margin = 0.05

    # Cost of debt
    cost_of_debt = (interest / debt).mean() if not debt.empty else 0.05
    if pd.isna(cost_of_debt) or cost_of_debt < 0 or cost_of_debt > 0.2: cost_of_debt = 0.05

    # NWC change
    nwc_change = nwc.diff(-1) # Assuming recent is first
    nwc_margin = (nwc_change / rev).mean() if not rev.empty else 0.02
    if pd.isna(nwc_margin): nwc_margin = 0.02

    # Exit Multiple (EV/EBITDA)
    ebitda = ebit.iloc[0] + dna.iloc[0] if not ebit.empty and not dna.empty else 1
    ev = (data['shares'] * data['current_price']) + debt.iloc[0] - data['cash'].iloc[0] if not debt.empty and not data['cash'].empty else 1
    exit_multiple = ev / ebitda if ebitda > 0 else 10.0
    if pd.isna(exit_multiple) or exit_multiple <= 0: exit_multiple = 10.0

    return {
        'rev_growth': float(rev_growth),
        'ebit_margin': float(ebit_margin),
        'tax_rate': float(tax_rate),
        'dna_margin': float(dna_margin),
        'capex_margin': float(capex_margin),
        'nwc_margin': float(nwc_margin),
        'cost_of_debt': float(cost_of_debt),
        'exit_multiple': float(exit_multiple)
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
