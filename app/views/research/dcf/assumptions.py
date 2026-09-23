"""The assumptions panel: every input the model takes, each with its provenance.

:func:`collect` draws the panel top to bottom and returns the assumption set the
package ``__init__`` fingerprints and runs. Each block of the panel is its own
function, in the order it draws; they share only the list of export records.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

import pandas as pd
import streamlit as st

from app.data import market_data
from app.logic.dcf import export as dcf_export
from app.logic.dcf import financials as fin
from app.logic.dcf import model as dcf_model
from app.logic.dcf import validation as val
from app.logic.dcf import valuation as dcf
from app.logic.dcf import wacc as wacc_lib
from app.provenance import Sourced
from app.views.research.dcf import common

FINRA_FIXED_INCOME_URL = "https://www.finra.org/finra-data/fixed-income"

COST_OF_DEBT_GUIDANCE = f"""
**Where to find this:** FINRA publishes TRACE-reported corporate bond trades and yields, free, at
<{FINRA_FIXED_INCOME_URL}>. Open **Corporate & Agency Bonds**, search by **issuer name** rather
than CUSIP — CUSIPs are per-issue and a large issuer will have dozens, so name search is faster
and less error-prone — then pick an issue **maturing about ten years out** and read its **Yield to
Maturity** off the trade detail. That is the number to type in the field above. The company's most
recent 10-K debt footnote lists every outstanding issue with coupon and maturity, which tells you
what to look for before you search.

**Which bond:** a **recently traded, senior unsecured** issue. Avoid secured, subordinated,
convertible, and thinly-traded issues — none of them price the company's general cost of
borrowing.

**Which maturity:** match the tenor of your risk-free rate. If you're using the 10Y Treasury in
CAPM, use a bond with roughly 10 years remaining to maturity. Consistency between the two matters
more than hitting exactly 10 years. If the company's debt is concentrated at a different tenor,
match its weighted-average maturity and use the same-tenor Treasury.

**Use yield to maturity, not the coupon.** The coupon is what the company agreed to pay at
issuance; YTM is what the market charges today. A DCF wants today's cost.

**No public bonds?** Fall back to the synthetic rating approach — map EBIT ÷ interest expense to
a rating band, then add the corresponding spread to the risk-free rate.
"""


#: Headline terminal value method, label to engine key. The first entry is what
#: the radio selects by default, and it is ordered to match
#: ``dcf_model.DEFAULT_HEADLINE_METHOD`` so the UI and the engine agree.
_HEADLINE_METHODS = {
    "Exit multiple": "exit_multiple",
    "Gordon growth": "gordon",
}


def _untouched(widget_value: float, default_value: float, decimals: int) -> bool:
    """True when a widget still holds its pre-populated default.

    Defaults are rounded before they reach the widget, so the comparison has to
    happen at the same precision — otherwise every untouched field claims to be
    a user override.
    """
    return abs(round(widget_value, decimals) - round(default_value, decimals)) < 10**-decimals / 2


def _driver_input(
    label: str,
    key: str,
    ratios: pd.DataFrame,
    driver_key: str,
    *,
    default_basis: str = "5-year average",
) -> tuple[float | None, Sourced, tuple[str, ...]]:
    """One driver: a basis selector, an editable value, and its provenance.

    Also returns the period labels the default was averaged over, so the
    validation layer can tell which fiscal years actually reach the model.
    """
    basis = st.selectbox(
        f"{label} — basis",
        fin.BASES,
        index=fin.BASES.index(default_basis),
        key=f"dcf_input_{key}_basis",
    )
    default = fin.driver_default(ratios, driver_key, basis)

    if not default.ok:
        common.provenance(
            Sourced.unavailable(source=f"Historical {label.lower()}", reason=default.detail)
        )
        st.warning(
            f"No usable history for {label.lower()} — {default.detail} There is nothing "
            "to pre-populate from, so this starts empty and must be entered by hand."
        )
        manual = st.number_input(
            f"{label} (%)", value=0.0, step=0.5, key=f"dcf_input_{key}_manual"
        )
        return (
            manual / 100.0,
            Sourced.user(manual / 100.0, "No historical basis available."),
            (),
        )

    widget_key = f"dcf_input_{key}_value"
    default_pct = round(default.value * 100, 4)
    if widget_key not in st.session_state:
        st.session_state[widget_key] = default_pct

    value = st.number_input(f"{label} (%)", step=0.25, key=widget_key, format="%.4f")
    derived = Sourced.derived(
        value=default.value,
        source=f"{basis} of reported history",
        formula=f"{fin.DRIVER_LABELS[driver_key]} over {default.years_used} period(s): "
        f"{', '.join(str(p) for p in default.periods[-6:])}",
    )

    user_value = value / 100.0
    if _untouched(value, default_pct, 4):
        sourced = derived
    else:
        sourced = derived.with_user_value(user_value)
    common.provenance(sourced)
    if default.detail:
        st.caption(f"⚠️ {default.detail}")
    return user_value, sourced, default.periods


class _Discount(NamedTuple):
    """The discount-rate inputs, before they are weighted into a WACC."""

    risk_free: float
    erp: float
    beta: float
    beta_outcome: dcf_model.BetaOutcome
    cost_of_debt: float
    marginal_tax: float


class _Terminal(NamedTuple):
    terminal_growth: float
    exit_multiple: float
    headline_method: str
    mid_year: bool
    stub: float


class _Bridge(NamedTuple):
    include_sti: bool
    cash_floor: float
    sbc: float | None
    deduct_sbc: bool
    non_operating: float


#: Sharadar publishes a single interest expense line rather than the gross/net
#: pair, so there is no basis to choose between.
_INTEREST_BASIS = "gross"


def collect(dataset: dcf_model.CompanyDataset, valuation_date: date) -> dict | None:
    """Renders every input and returns the assembled assumption set.

    One section per block of the panel, top to bottom, in the order they draw.
    They share only ``records``, the export's Assumptions sheet, which each one
    appends to. ``None`` means a blocking screen has been drawn in place of the
    rest of the page.

    Every key of the returned dict that the valuation depends on must also be in
    ``_assumption_fingerprint`` in the package ``__init__`` — a key missing
    there leaves the previous run's numbers on screen looking current.
    """
    st.divider()
    st.subheader("Assumptions")
    st.caption(
        "Every default is derived from reported history and shows where it came from. "
        "Nothing here is invented — where a basis does not exist, the field says so."
    )

    base = dcf_model.base_period(dataset)
    records: list[dcf_export.AssumptionRecord] = []

    drivers, driver_values, periods_in_use = _forecast_drivers(dataset, base, records)

    st.markdown("#### Discount rate")
    discount = st.container(border=True)
    with discount:
        rates = _discount_inputs(dataset, records)

    counted = _share_count(dataset, records)
    if counted is None:
        return None
    shares, share_basis = counted

    wacc_result = _weighted_cost(dataset, base, rates, discount)
    if wacc_result is None:
        return None

    terminal = _terminal_and_conventions(dataset, valuation_date, records)
    bridge = _cash_sbc_and_bridge(dataset, base, records)

    # --- Effective tax rate for FCFF -------------------------------------
    etr_series = fin.effective_tax_rate(dataset.annual).dropna()
    etr_window = etr_series.tail(5)
    etr = float(etr_window.median()) if not etr_series.empty else None
    periods_in_use.update(str(p) for p in etr_window.index)
    # The base period is what everything is projected from, so it is in use by
    # definition even when no driver window happens to reach it.
    if not dataset.annual.empty:
        periods_in_use.add(str(dataset.annual.index[-1]))
    delta_nwc, delta_nwc_span = dcf_model.base_period_delta_nwc(dataset, base)

    base_fcff = dcf_model.base_period_fcff(
        dataset,
        base,
        tax_rate=etr if etr is not None else driver_values["effective_tax_rate"],
        interest_basis=_INTEREST_BASIS,
        delta_nwc=delta_nwc,
    )
    _render_fcff_panel(dataset, base, base_fcff, etr, etr_series, delta_nwc_span)

    return {
        "base": base,
        "drivers": drivers,
        "wacc": wacc_result,
        "terminal_growth": terminal.terminal_growth,
        "exit_multiple": terminal.exit_multiple,
        "stub": terminal.stub,
        "mid_year": terminal.mid_year,
        "shares": shares,
        "share_basis": share_basis,
        "include_sti": bridge.include_sti,
        "cash_floor": bridge.cash_floor,
        "non_operating": bridge.non_operating or None,
        "base_fcff": base_fcff,
        "headline_method": terminal.headline_method,
        "sbc": bridge.sbc,
        "deduct_sbc": bridge.deduct_sbc,
        "etr": etr,
        "erp": rates.erp,
        "risk_free": rates.risk_free,
        "beta_outcome": rates.beta_outcome,
        "records": records,
        "periods_in_use": frozenset(periods_in_use),
        "tolerances": val.DEFAULT_TOLERANCES,
    }


def _forecast_drivers(
    dataset: dcf_model.CompanyDataset,
    base: dcf_model.BasePeriod,
    records: list[dcf_export.AssumptionRecord],
) -> tuple[dcf.DriverPath, dict[str, float], set[str]]:
    """The six drivers, plus every fiscal year any of them averaged over."""
    st.markdown("#### Forecast drivers")
    with st.container(border=True):
        st.caption(
            f"Base period: **{base.label}** — {base.detail} "
            f"Balance sheet as of **{base.stocks_as_of}**."
        )
        years = st.slider("Forecast years", 5, 10, 5, key="dcf_input_years")

        driver_values: dict[str, float] = {}
        # Every fiscal year any driver averaged over. Reconciliation failures
        # outside this set cannot reach the valuation, and the validation layer
        # scopes their severity accordingly.
        periods_in_use: set[str] = set()
        grid = st.columns(3)
        specs = [
            ("Revenue growth", "growth", "revenue_growth"),
            ("Operating margin", "margin", "operating_margin"),
            ("Effective tax rate", "etr", "effective_tax_rate"),
            ("CapEx % of revenue", "capex", "capex_pct_revenue"),
            ("D&A % of revenue", "dna", "dna_pct_revenue"),
            ("ΔNWC % of Δrevenue", "nwc", "dnwc_pct_drevenue"),
        ]
        for index, (label, key, driver_key) in enumerate(specs):
            with grid[index % 3]:
                value, sourced, periods = _driver_input(
                    label, key, dataset.ratios, driver_key
                )
                driver_values[driver_key] = value if value is not None else 0.0
                periods_in_use.update(periods)
                records.append(
                    dcf_export.AssumptionRecord("Forecast drivers", label, value, sourced)
                )

        if dataset.capex_result.origin == "estimate":
            st.warning(
                f"**CapEx is an estimate.** {dataset.capex_result.detail} "
                f"Formula in use: `{dataset.capex_result.formula}`"
            )

    drivers = dcf.DriverPath.flat(
        years,
        revenue_growth=driver_values["revenue_growth"],
        operating_margin=driver_values["operating_margin"],
        effective_tax_rate=driver_values["effective_tax_rate"],
        capex_pct_revenue=driver_values["capex_pct_revenue"],
        dna_pct_revenue=driver_values["dna_pct_revenue"],
        dnwc_pct_drevenue=driver_values["dnwc_pct_drevenue"],
    )
    return drivers, driver_values, periods_in_use


def _discount_inputs(
    dataset: dcf_model.CompanyDataset, records: list[dcf_export.AssumptionRecord]
) -> _Discount:
    """The three columns of the discount-rate panel, and the cost-of-debt guidance.

    Drawn inside the panel's container; :func:`_weighted_cost` adds the WACC
    beneath them once the share count is known.
    """
    left, middle, right = st.columns(3)

    with left:
        st.markdown("**Risk-free rate**")
        treasury = market_data.fetch_10y_treasury_rate()
        if treasury.ok:
            default_rf = treasury.value * 100
        else:
            default_rf = 0.0
            st.warning(
                f"The 10-year Treasury rate could not be fetched — {treasury.detail}. "
                "Enter it manually; nothing has been substituted."
            )
        rf_pct = st.number_input(
            "Risk-free rate (%)",
            value=float(default_rf),
            step=0.05,
            key="dcf_input_rf",
            help="Should match the tenor of the bond behind your cost of debt.",
        )
        risk_free = rf_pct / 100.0
        rf_sourced = (
            treasury
            if treasury.ok and _untouched(rf_pct, default_rf, 4)
            else Sourced.user(risk_free, "Manually entered risk-free rate.")
        )
        common.provenance(rf_sourced)
        records.append(
            dcf_export.AssumptionRecord("Discount rate", "Risk-free rate", risk_free, rf_sourced)
        )

        st.markdown("**Equity risk premium**")
        erp_sourced_default = market_data.fetch_erp()
        if erp_sourced_default.ok:
            default_erp = erp_sourced_default.value * 100
        else:
            default_erp = 0.0
            st.warning(
                f"The equity risk premium scrape failed — {erp_sourced_default.detail}. "
                "Enter it manually; no default has been substituted."
            )
        erp_pct = st.number_input(
            "Equity risk premium (%)", value=float(default_erp), step=0.05, key="dcf_input_erp"
        )
        erp = erp_pct / 100.0
        erp_sourced = (
            erp_sourced_default
            if erp_sourced_default.ok and _untouched(erp_pct, default_erp, 4)
            else Sourced.user(erp, "Manually entered equity risk premium.")
        )
        common.provenance(erp_sourced)
        records.append(
            dcf_export.AssumptionRecord("Discount rate", "Equity risk premium", erp, erp_sourced)
        )

    with middle:
        st.markdown("**Beta**")
        frequency = st.radio(
            "Regression window",
            ["monthly", "weekly"],
            horizontal=True,
            key="dcf_input_beta_freq",
            help="5 years of monthly returns by default; 2 years of weekly as the alternative.",
        )
        blume = st.checkbox(
            "Blume adjustment (0.67 × β + 0.33)", value=False, key="dcf_input_blume"
        )
        beta_outcome = dcf_model.regress_company_beta(
            dataset,
            frequency=frequency,
            blume=blume,
            years=5 if frequency == "monthly" else 2,
        )

        if beta_outcome.result is not None:
            regressed = beta_outcome.result
            beta_default = regressed.beta
            beta_sourced_default = Sourced.derived(
                value=beta_default,
                source=beta_outcome.source,
                formula=regressed.formula,
            )
        else:
            regressed = None
            beta_default = 1.0
            beta_sourced_default = Sourced.unavailable(
                source=beta_outcome.source, reason=beta_outcome.error or "unavailable"
            )
            st.warning(
                f"Beta could not be regressed — {beta_outcome.error} Enter one manually; "
                "the field below starts at the market beta of 1.0, which is a placeholder, "
                "not an estimate for this company."
            )

        if "dcf_input_beta" not in st.session_state:
            st.session_state["dcf_input_beta"] = round(float(beta_default), 4)

        beta_value = st.number_input(
            "Beta (editable)", step=0.05, key="dcf_input_beta", format="%.4f"
        )
        if regressed is not None:
            col_reset, col_stats = st.columns([1, 2])
            with col_reset:
                if st.button("Reset to regressed", key="dcf_beta_reset", width="stretch"):
                    st.session_state["dcf_input_beta"] = round(float(regressed.beta), 4)
                    st.rerun()
            with col_stats:
                st.metric("Regressed β", f"{regressed.beta:.4f}", f"R² {regressed.r_squared:.3f}")
            st.caption(
                f"Raw β {regressed.raw_beta:.4f} · {regressed.observations} {frequency} "
                f"observations vs {regressed.index_ticker}"
            )
            if beta_outcome.price_source_note:
                st.caption(f"⚠️ {beta_outcome.price_source_note}")

        beta_sourced = (
            beta_sourced_default
            if regressed is not None and _untouched(beta_value, regressed.beta, 4)
            else Sourced.user(
                beta_value,
                f"Overrides the regressed β of {regressed.beta:.4f}"
                if regressed
                else "No regression was available.",
            )
        )
        common.provenance(beta_sourced)
        records.append(
            dcf_export.AssumptionRecord("Discount rate", "Beta", beta_value, beta_sourced)
        )

    with right:
        st.markdown("**Cost of debt**")
        cost_of_debt_pct = st.number_input(
            "Cost of debt — yield to maturity (%)",
            value=5.0,
            step=0.05,
            key="dcf_input_kd",
            help="Manual input. See the sourcing guidance below.",
        )
        cost_of_debt = cost_of_debt_pct / 100.0
        kd_sourced = Sourced.user(
            cost_of_debt, "Manual yield-to-maturity input — Sharadar carries no bond data."
        )
        common.provenance(kd_sourced)
        records.append(
            dcf_export.AssumptionRecord(
                "Discount rate", "Cost of debt (YTM)", cost_of_debt, kd_sourced
            )
        )

        interest_series = fin.interest_expense(dataset.annual)
        debt_series = fin.total_debt(dataset.annual)
        implied = None
        if len(interest_series) and len(debt_series):
            implied = wacc_lib.implied_cost_of_debt(
                float(interest_series.iloc[-1])
                if not pd.isna(interest_series.iloc[-1])
                else None,
                float(debt_series.iloc[-1]) if not pd.isna(debt_series.iloc[-1]) else None,
                float(debt_series.iloc[-2])
                if len(debt_series) > 1 and not pd.isna(debt_series.iloc[-2])
                else None,
            )
        if implied is not None:
            st.metric("Implied average cost of debt", f"{implied:.2%}")
            st.caption(
                "Interest expense ÷ average total debt. This is a **backward-looking "
                "average coupon**, not a current marginal cost — it understates when "
                "rates have risen and overstates when they have fallen. Cross-check only; "
                "it never populates the model."
            )
        else:
            st.info(
                "The implied cost of debt cross-check is unavailable: Sharadar reports no "
                "interest expense for this company."
            )

        marginal_tax_pct = st.number_input(
            "Marginal tax rate for the WACC shield (%)",
            value=21.0,
            step=0.5,
            key="dcf_input_marginal_tax",
            help=(
                "The interest tax shield is worth the *marginal* rate — what the next "
                "dollar of interest saves — not the historical effective rate that feeds "
                "FCFF. They are separate inputs on purpose."
            ),
        )
        marginal_tax = marginal_tax_pct / 100.0
        records.append(
            dcf_export.AssumptionRecord(
                "Discount rate",
                "Marginal tax rate",
                marginal_tax,
                Sourced.user(marginal_tax, "Statutory marginal rate for the tax shield."),
            )
        )

    with st.expander("Where to find a cost of debt", expanded=False):
        st.markdown(COST_OF_DEBT_GUIDANCE)
        st.link_button(
            "Open FINRA's fixed income data", FINRA_FIXED_INCOME_URL, width="stretch"
        )

    return _Discount(risk_free, erp, beta_value, beta_outcome, cost_of_debt, marginal_tax)


def _share_count(
    dataset: dcf_model.CompanyDataset, records: list[dcf_export.AssumptionRecord]
) -> tuple[float, str] | None:
    """The share count value per share divides by, and a label saying which one."""
    share_basis_choice = st.radio(
        "Share count basis for value per share",
        ["Diluted weighted-average (TTM)", "Point-in-time shares outstanding"],
        horizontal=True,
        key="dcf_input_share_basis",
        help=(
            "The spec's default is the diluted weighted-average count. The point-in-time "
            "count is more current but excludes option and RSU dilution."
        ),
    )
    if share_basis_choice.startswith("Diluted") and dataset.shares_weighted_diluted:
        shares = dataset.shares_weighted_diluted
        share_basis = f"Diluted weighted-average ({dataset.shares_weighted_label})"
    elif dataset.shares_point_in_time:
        shares = dataset.shares_point_in_time
        share_basis = f"Point-in-time outstanding (as of {dataset.shares_point_in_time_date})"
    else:
        common.render_blocking(
            "No share count available",
            f"Sharadar returned no share count for {dataset.ticker}, so equity value cannot "
            "be divided into a per-share figure.",
        )
        return None

    st.caption(
        f"Using **{shares:,.0f}** shares — {share_basis}. "
        f"Sharadar provides no treasury-stock-method calculation for outstanding options and "
        f"RSUs, so even the diluted count is not fully loaded."
    )
    records.append(
        dcf_export.AssumptionRecord(
            "Capital structure",
            "Shares used",
            shares,
            Sourced.reported(shares, share_basis, dataset.shares_point_in_time_date),
        )
    )
    return shares, share_basis


def _weighted_cost(
    dataset: dcf_model.CompanyDataset,
    base: dcf_model.BasePeriod,
    rates: _Discount,
    discount,
) -> wacc_lib.WaccResult | None:
    """Weights the discount inputs into a WACC and shows it in the discount panel."""
    total_debt_series = fin.total_debt(pd.DataFrame([base.stocks]))
    debt_value = (
        float(total_debt_series.iloc[0])
        if not total_debt_series.empty and not pd.isna(total_debt_series.iloc[0])
        else 0.0
    )

    equity_value = (dataset.last_price or 0.0) * (dataset.shares_point_in_time or 0.0)
    try:
        wacc_result = wacc_lib.compute_wacc(
            equity_value=equity_value,
            debt_value=debt_value,
            risk_free_rate=rates.risk_free,
            beta=rates.beta,
            equity_risk_premium=rates.erp,
            cost_of_debt=rates.cost_of_debt,
            marginal_tax_rate=rates.marginal_tax,
        )
    except ValueError as error:
        common.render_blocking("The WACC cannot be weighted", str(error))
        return None

    with discount:
        st.markdown("**Weighted average cost of capital**")
        wacc_columns = st.columns(5)
        wacc_columns[0].metric("WACC", f"{wacc_result.wacc:.2%}")
        wacc_columns[1].metric("Cost of equity", f"{wacc_result.cost_of_equity:.2%}")
        wacc_columns[2].metric("After-tax cost of debt", f"{wacc_result.after_tax_cost_of_debt:.2%}")
        wacc_columns[3].metric("Equity weight", f"{wacc_result.equity_weight:.1%}")
        wacc_columns[4].metric("Debt weight", f"{wacc_result.debt_weight:.1%}")
        st.caption(
            f"`{wacc_result.formula}` · E = last close × point-in-time shares "
            f"({common.money(equity_value, dataset.currency)}) · D = book value of total debt "
            f"({common.money(debt_value, dataset.currency)}), as of {base.stocks_as_of}."
        )
    return wacc_result


def _terminal_and_conventions(
    dataset: dcf_model.CompanyDataset,
    valuation_date: date,
    records: list[dcf_export.AssumptionRecord],
) -> _Terminal:
    """Both terminal value inputs, the headline choice, and the discounting conventions."""
    st.markdown("#### Terminal value and conventions")
    with st.container(border=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            terminal_growth_pct = st.number_input(
                "Perpetuity growth rate (%)", value=2.5, step=0.1, key="dcf_input_g"
            )
            terminal_growth = terminal_growth_pct / 100.0
            records.append(
                dcf_export.AssumptionRecord(
                    "Terminal value",
                    "Perpetuity growth",
                    terminal_growth,
                    Sourced.user(terminal_growth, "Gordon growth rate."),
                )
            )
        with col_b:
            ev_ebitda = dcf_model.historical_ev_ebitda(dataset)
            if ev_ebitda.ok:
                multiple_default = round(float(ev_ebitda.value), 2)
            else:
                multiple_default = 10.0
                st.warning(
                    f"No historical EV/EBITDA range available — {ev_ebitda.detail} "
                    "The field starts at a generic 10×, which is a placeholder rather than "
                    "this company's own trading range."
                )
            if "dcf_input_exit" not in st.session_state:
                st.session_state["dcf_input_exit"] = float(multiple_default)
            exit_multiple = st.number_input(
                "Exit EV/EBITDA multiple (×)", step=0.25, key="dcf_input_exit"
            )
            exit_sourced = (
                ev_ebitda
                if ev_ebitda.ok and _untouched(exit_multiple, multiple_default, 2)
                else Sourced.user(exit_multiple, "Manually set exit multiple.")
            )
            common.provenance(exit_sourced)
            records.append(
                dcf_export.AssumptionRecord(
                    "Terminal value", "Exit EV/EBITDA multiple", exit_multiple, exit_sourced
                )
            )
        with col_c:
            headline_label = st.radio(
                "Headline method",
                list(_HEADLINE_METHODS),
                key="dcf_input_headline",
                help="Drives the headline value per share and the equity bridge. Both "
                     "methods always compute and both are shown side by side.",
            )
            headline_method = _HEADLINE_METHODS[headline_label]
            mid_year = st.toggle(
                "Mid-year discounting", value=True, key="dcf_input_midyear",
                help="Mid-year: DF = 1 ÷ (1 + WACC)^(t − 0.5). Applied to the terminal value too.",
            )

        fye = fin.fiscal_year_end(
            dataset.fiscal_year_end_month,
            valuation_date,
            pd.Timestamp(dataset.annual_meta["report_date"].iloc[-1]).date()
            if not dataset.annual_meta.empty
            and not pd.isna(dataset.annual_meta["report_date"].iloc[-1])
            else None,
        )
        if fye is None:
            stub = 1.0
            st.warning(
                "No fiscal year end could be read from the reported periods, so the stub "
                "computed and the first forecast year is treated as a full year."
            )
        else:
            stub = dcf.stub_fraction(valuation_date, fye)
            st.caption(
                f"**Stub period:** {stub:.4f} of a year — {(fye - valuation_date).days} days "
                f"from {valuation_date} to the next fiscal year end of {fye} ÷ 365. "
                f"The first period's cash flow scales by this fraction and its discount "
                f"exponent is {'half of it' if mid_year else 'the fraction itself'}."
            )
        records.append(
            dcf_export.AssumptionRecord(
                "Conventions",
                "Stub fraction",
                stub,
                Sourced.derived(
                    stub,
                    "Fiscal year end derived from the latest annual report period",
                    f"days to {fye} ÷ 365" if fye else "no fiscal year end available",
                ),
            )
        )

    return _Terminal(terminal_growth, exit_multiple, headline_method, mid_year, stub)


def _cash_sbc_and_bridge(
    dataset: dcf_model.CompanyDataset,
    base: dcf_model.BasePeriod,
    records: list[dcf_export.AssumptionRecord],
) -> _Bridge:
    """Excess cash, stock-based compensation, and non-operating assets."""
    stocks = base.stocks
    st.markdown("#### Cash treatment, SBC, and bridge items")
    with st.container(border=True):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            include_sti = st.toggle(
                "Treat short-term investments as excess cash",
                value=True,
                key="dcf_input_sti",
                help="Turning this off treats short-term investments as operating assets.",
            )
            cash_value = stocks.get("cash")
            cash_floor = st.number_input(
                "Operating cash floor to hold back",
                value=0.0,
                step=1e9,
                format="%.0f",
                key="dcf_input_cash_floor",
                help=(
                    "Cash the business needs to run, excluded from the bridge. Materially "
                    "moves the answer for cash-rich companies."
                ),
            )
            short_term = stocks.get("short_term_investments")
            st.caption(
                f"Balance-sheet cash {common.money(cash_value, dataset.currency)} · short-term "
                f"investments {common.money(short_term, dataset.currency)} "
                f"(as of {base.stocks_as_of})"
            )

        with col_b:
            st.markdown("**Stock-based compensation**")
            base_sbc = base.flows.get("sbc")
            sbc_available = base_sbc is not None and not pd.isna(base_sbc)

            if not sbc_available:
                st.warning(
                    "**Sharadar reports no stock-based compensation for this company.** The "
                    "field is empty and nothing has been inferred. Enter the figure from the "
                    "cash flow statement in the latest 10-K."
                )

            sbc_input = st.number_input(
                "Stock-based compensation",
                value=float(base_sbc) if sbc_available else float(
                    st.session_state.get("dcf_input_sbc", 0.0)
                ),
                step=1e8,
                format="%.0f",
                key="dcf_input_sbc",
            )
            sbc = sbc_input if sbc_input else None

            if sbc_available:
                sbc_sourced = Sourced.reported(sbc_input, "Sharadar CF: Share Based Compensation")
            elif sbc_input:
                sbc_sourced = Sourced.user(sbc_input, "Entered manually from the filing.")
            else:
                sbc_sourced = Sourced.unavailable(
                    "Sharadar CF: Share Based Compensation",
                    "Not reported and not supplied.",
                )
            common.provenance(sbc_sourced)

            deduct_sbc = st.toggle(
                "Deduct SBC from FCFF (treat as a real cost)",
                value=False,
                key="dcf_input_deduct_sbc",
                disabled=sbc is None,
                help=(
                    "Standard FCFF adds SBC back as non-cash, which operating cash flow has "
                    "already done. Deducting it instead treats it as the economic cost it is. "
                    "Inert until an SBC figure exists."
                ),
            )
            revenue_base = base.flows.get("revenue")
            if sbc and revenue_base:
                st.caption(f"SBC ÷ revenue: **{sbc / float(revenue_base):.2%}**")

            # The number on its own is inert, and that surprises people. Standard
            # FCFF already adds SBC back as a non-cash charge inside operating
            # cash flow, so "add back" is a no-op — the figure only moves the
            # valuation once it is being deducted.
            if sbc is not None and not deduct_sbc:
                st.info(
                    "**This figure is not changing the valuation yet.** Standard FCFF adds "
                    "SBC back as a non-cash charge, which operating cash flow has already "
                    "done — so entering it alone is a no-op by construction. Turn on "
                    "**Deduct SBC from FCFF** above to treat it as a real cost.",
                    icon="ℹ️",
                )
            elif sbc is not None and deduct_sbc:
                st.caption(
                    f"Deducting {common.money(sbc, dataset.currency)} from FCFF, grown with "
                    f"revenue across the forecast."
                )
            records.append(
                dcf_export.AssumptionRecord(
                    "Cash flow", "Stock-based compensation", sbc, sbc_sourced,
                    note="Deducted from FCFF" if deduct_sbc else "Added back (standard FCFF)",
                )
            )

        with col_c:
            st.markdown("**Non-operating assets**")
            lt_investments = stocks.get("long_term_investments")
            default_non_operating = (
                float(lt_investments) if lt_investments is not None and not pd.isna(lt_investments)
                else 0.0
            )
            non_operating = st.number_input(
                "Non-operating assets to add",
                value=default_non_operating,
                step=1e8,
                format="%.0f",
                key="dcf_input_nonop",
                help="Long-term and equity-method investments that the operating forecast "
                     "does not already capture.",
            )
            if lt_investments is None or pd.isna(lt_investments):
                st.caption(
                    "Sharadar reports no non-current investments line for this company, so this "
                    "starts at zero by absence rather than by measurement."
                )

    return _Bridge(include_sti, cash_floor, sbc, deduct_sbc, non_operating)


def _render_fcff_panel(
    dataset: dcf_model.CompanyDataset,
    base: dcf_model.BasePeriod,
    base_fcff: dcf_model.BaseFcff,
    etr: float | None,
    etr_series: pd.Series,
    delta_nwc_span: str = "",
) -> None:
    st.markdown("#### Base-period free cash flow, computed two independent ways")
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.markdown("**Method A — cash flow statement build**")
            st.code("FCFF = CFO + Interest Expense × (1 − tax rate) − CapEx")
            st.metric("FCFF (method A)", common.money(base_fcff.method_a, dataset.currency))
        with right:
            st.markdown("**Method B — income statement build**")
            st.code("NOPAT = EBIT × (1 − tax rate)\nFCFF  = NOPAT + D&A − CapEx − ΔNWC")
            st.metric("FCFF (method B)", common.money(base_fcff.method_b, dataset.currency))

        gap = base_fcff.divergence
        if gap is not None:
            st.caption(f"Divergence between the two builds: **{gap:.2%}**")

        for note in base_fcff.notes:
            st.warning(note)

        inputs_frame = pd.DataFrame(
            [
                {
                    "Input": name,
                    "Value": (
                        f"{value:,.4f}" if name == "Tax rate"
                        else common.money(value, dataset.currency)
                    ) if value is not None else "not reported",
                }
                for name, value in base_fcff.inputs.items()
            ]
        )
        st.dataframe(inputs_frame, width="stretch", hide_index=True)

        if etr is not None:
            st.caption(
                f"Effective tax rate in use: **{etr:.2%}** — median of the last "
                f"{min(5, len(etr_series))} profitable year(s). Years with a pretax loss are "
                f"excluded because the ratio is meaningless there, not set to zero."
            )
        else:
            st.warning(
                "No profitable year is available to compute an effective tax rate, so the "
                "forecast tax rate driver is used for the base period as well."
            )

        if delta_nwc_span:
            st.caption(f"Change in net working capital measured over — {delta_nwc_span}.")

        st.caption(
            "Method A assumes interest paid sits in operating cash flow, which holds under "
            "US GAAP; an IFRS filer may classify it in financing. Sharadar exposes interest "
            "*expense* (accrual), not interest *paid* (cash), so that is what is used. "
            "The two builds legitimately differ where working capital moves a lot: method B "
            "subtracts the balance-sheet movement, while method A picks up whatever the cash "
            "flow statement booked, including acquisitions and FX translation."
        )
