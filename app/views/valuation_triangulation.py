import math

import streamlit as st

from app.data import market_data
from app.defaults import DEFAULTS
from app.logic import valuation_triangulation as vt_logic
from app.provenance import Origin, Resolved

# Loudest first, so a missing API key sorts above a bank's absent working capital.
_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}

# The inputs that feed WACC. A default sitting in any of these moves the
# discount rate, and so moves every number below it.
_WACC_DRIVERS = ('risk_free_rate', 'beta', 'erp', 'cost_of_debt', 'tax_rate')


def _marked(key: str, resolved) -> str:
    """Labels an input, flagging it when the value is a stand-in for real data."""
    label = DEFAULTS[key].label
    return f"⚠️ {label}" if resolved.is_fallback else label


def _still_default(resolved, current: float) -> bool:
    """True when a defaulted input is still sitting at its default.

    Every assumption is editable, so a fallback the user has already overridden
    with their own estimate shouldn't keep caveating the results.
    """
    return resolved.is_fallback and math.isclose(current, resolved.value, rel_tol=1e-9, abs_tol=1e-12)


def _report_provenance(inputs: list) -> None:
    """Renders an audit of where every assumption came from.

    `inputs` is a list of (defaults key, Resolved, current value); the label and
    number format come from app.defaults so they can't drift from the widgets.
    """
    fallbacks = sorted(
        (row for row in inputs if row[1].is_fallback),
        key=lambda row: _SEVERITY_ORDER.get(row[1].severity, 9),
    )

    if not fallbacks:
        st.success(f"All {len(inputs)} assumptions are drawn from live data and reported financials.")
        return

    with st.expander(
        f"⚠️ {len(fallbacks)} of {len(inputs)} assumptions fall back to a default", expanded=True
    ):
        for key, resolved, current in fallbacks:
            spec = DEFAULTS[key]
            write = getattr(st, resolved.severity)
            overridden = "" if _still_default(resolved, current) else " *(you have since overridden this)*"
            write(f"**{spec.label}** — standing in {spec.format(resolved.value)}. "
                  f"{resolved.detail}.{overridden}")
            st.caption(f"Why this default: {spec.rationale}")


def render_page():
    st.title("Valuation Triangulation")

    # Add ticker input globally for all tabs
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL").upper()
    with col2:
        st.write("") # For vertical alignment
        st.write("")
        if st.button("Run Valuation", use_container_width=True):
            st.session_state['vt_run'] = True

    if 'vt_run' not in st.session_state:
        st.session_state['vt_run'] = False

    # Create the 5 tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Unlevered DCF",
        "Levered DCF",
        "Comparable Company Analysis",
        "Historical Multiple Reversion",
        "Reverse DCF"
    ])

    with tab1:
        st.header("Unlevered DCF")
        if st.session_state['vt_run']:
            with st.spinner(f"Fetching data for {ticker}..."):
                # Fetch baseline data
                financials = market_data.fetch_simfin_financials(ticker)

                if not financials:
                    st.error("Failed to fetch historical data. Check if Ticker is correct or try again.")
                else:
                    risk_free_rate = market_data.fetch_10y_treasury_rate()
                    beta = market_data.fetch_beta(ticker)
                    erp = market_data.fetch_erp()
                    defaults = vt_logic.calculate_historical_averages(financials)

                    st.subheader("Assumptions (Derived from Historicals)")
                    st.caption("A ⚠️ marks an assumption that is a hardcoded stand-in rather than "
                               "real data. Hover it for the reason, and override it if you have a better estimate.")
                    col1, col2, col3, col4 = st.columns(4)

                    def pct_input(key, step=1.0, resolved=None):
                        """A percentage assumption, flagged when it's a stand-in.

                        Owns the "(%)" suffix because it owns the /100.
                        """
                        resolved = defaults[key] if resolved is None else resolved
                        return st.number_input(
                            f"{_marked(key, resolved)} (%)", value=resolved.value * 100, step=step,
                            help=resolved.detail or f"Derived from {resolved.source}.",
                        ) / 100

                    with col1:
                        rev_growth = pct_input('rev_growth')
                        ebit_margin = pct_input('ebit_margin')
                    with col2:
                        tax_rate = pct_input('tax_rate')
                        dna_margin = pct_input('dna_margin')
                    with col3:
                        capex_margin = pct_input('capex_margin')
                        nwc_margin = pct_input('nwc_margin')
                    with col4:
                        cost_of_debt = pct_input('cost_of_debt', step=0.1)
                        exit_multiple = st.number_input(
                            _marked('exit_multiple', defaults['exit_multiple']),
                            value=defaults['exit_multiple'].value, step=1.0,
                            help=defaults['exit_multiple'].detail or f"Derived from {defaults['exit_multiple'].source}.",
                        )

                    st.subheader("WACC Inputs")
                    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
                    with w_col1:
                        rfr_input = pct_input('risk_free_rate', step=0.1, resolved=risk_free_rate)
                    with w_col2:
                        beta_input = st.number_input(
                            _marked('beta', beta), value=beta.value, step=0.1,
                            help=beta.detail or f"Calculated from {beta.source}.",
                        )
                    with w_col3:
                        erp_input = pct_input('erp', step=0.1, resolved=erp)
                    with w_col4:
                        ev_market_cap = financials['shares'] * financials['current_price']
                        latest_debt = financials['total_debt'].iloc[0] if not financials['total_debt'].empty else 0
                        ev_total = ev_market_cap + latest_debt

                        if ev_total > 0:
                            debt_weight_res = Resolved.observed(latest_debt / ev_total, "simfin:market-cap-and-debt")
                        else:
                            debt_weight_res = DEFAULTS['debt_weight'].fallback(
                                Origin.INSUFFICIENT_DATA,
                                "market cap and debt both come back at zero, so the real capital "
                                "structure can't be measured",
                            )
                        debt_weight = pct_input('debt_weight', resolved=debt_weight_res)
                        equity_weight = 1.0 - debt_weight
                        st.caption(f"Equity Weight: {equity_weight*100:.1f}%")

                    # One place recording where each input came from and whether
                    # the user has since replaced it, driving both the audit
                    # below and the caveats on the results.
                    inputs = [
                        ('rev_growth', defaults['rev_growth'], rev_growth),
                        ('ebit_margin', defaults['ebit_margin'], ebit_margin),
                        ('tax_rate', defaults['tax_rate'], tax_rate),
                        ('dna_margin', defaults['dna_margin'], dna_margin),
                        ('capex_margin', defaults['capex_margin'], capex_margin),
                        ('nwc_margin', defaults['nwc_margin'], nwc_margin),
                        ('cost_of_debt', defaults['cost_of_debt'], cost_of_debt),
                        ('exit_multiple', defaults['exit_multiple'], exit_multiple),
                        ('risk_free_rate', risk_free_rate, rfr_input),
                        ('beta', beta, beta_input),
                        ('erp', erp, erp_input),
                        ('debt_weight', debt_weight_res, debt_weight),
                    ]

                    st.subheader("Where These Assumptions Came From")
                    _report_provenance(inputs)

                    unresolved = [key for key, resolved, current in inputs
                                  if _still_default(resolved, current)]

                    wacc = vt_logic.calculate_wacc(rfr_input, beta_input, erp_input, cost_of_debt, tax_rate, debt_weight, equity_weight)
                    st.markdown(f"**Calculated WACC:** {wacc*100:.2f}%")

                    stale_wacc = [DEFAULTS[key].label for key in unresolved if key in _WACC_DRIVERS]
                    if stale_wacc:
                        st.caption(f"⚠️ Discount rate rests on defaulted {', '.join(stale_wacc)}, "
                                   f"so it is a placeholder rather than a measurement.")

                    assumptions = {
                        'rev_growth': rev_growth,
                        'ebit_margin': ebit_margin,
                        'tax_rate': tax_rate,
                        'dna_margin': dna_margin,
                        'capex_margin': capex_margin,
                        'nwc_margin': nwc_margin,
                        'exit_multiple': exit_multiple
                    }

                    dcf_results = vt_logic.run_unlevered_dcf(financials, assumptions, wacc)

                    st.subheader("Results")
                    if unresolved:
                        st.warning(
                            f"**Indicative only.** {len(unresolved)} of {len(inputs)} assumptions are still "
                            f"hardcoded defaults ({', '.join(DEFAULTS[key].label for key in unresolved)}), so the implied price below "
                            f"reflects those stand-ins as much as it reflects {ticker}."
                        )

                    res_col1, res_col2, res_col3 = st.columns(3)
                    res_col1.metric("Implied Share Price", f"${dcf_results['implied_share_price']:.2f}")
                    res_col2.metric("Current Share Price", f"${financials['current_price']:.2f}")

                    if financials['current_price'] > 0:
                        premium_discount = (dcf_results['implied_share_price'] / financials['current_price']) - 1
                        res_col3.metric("Premium / (Discount)", f"{premium_discount*100:.2f}%")

                    st.subheader("Projected Free Cash Flows")
                    st.dataframe(dcf_results['projections'].style.format({
                        'Revenue': '${:,.0f}',
                        'EBIT': '${:,.0f}',
                        'NOPAT': '${:,.0f}',
                        '+ D&A': '${:,.0f}',
                        '- CapEx': '${:,.0f}',
                        '- Change in NWC': '${:,.0f}',
                        'UFCF': '${:,.0f}',
                        'PV of UFCF': '${:,.0f}'
                    }))
        else:
            st.info("Enter a ticker in the 'Valuation Settings' section above and click 'Run Valuation' to see the DCF.")

    with tab2:
        st.header("Levered DCF")
        st.write("Levered DCF methodology goes here.")

    with tab3:
        st.header("Comparable Company Analysis")
        st.write("Comparable Company Analysis methodology goes here.")

    with tab4:
        st.header("Historical Multiple Reversion")
        st.write("Historical Multiple Reversion (Time-Series Analysis) goes here.")

    with tab5:
        st.header("The Reverse DCF")
        st.write("The Reverse DCF (Market-Implied Expectations) goes here.")
