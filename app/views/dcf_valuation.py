"""DCF valuation page — UI only.

Every number rendered here carries a provenance badge and, where it was
computed, the formula behind it. The severity banners sit above everything and
do not collapse: the governing rule for this page is that it must be impossible
to look at the output and not notice something is wrong.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app.data import market_data, simfin_client
from app.logic.dcf import export as dcf_export
from app.logic.dcf import financials as fin
from app.logic.dcf import model as dcf_model
from app.logic.dcf import validation as val
from app.logic.dcf import valuation as dcf
from app.logic.dcf import wacc as wacc_lib
from app.logic.dcf.schema import SchemaResolutionError
from app.provenance import Origin, Sourced

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

_BANNER_CSS = """
<style>
.st-key-dcf_critical_banner {
    border: 3px solid #ff2b2b;
    border-radius: 8px;
    background: rgba(255, 43, 43, 0.10);
    padding: 0.6rem 1rem 0.2rem 1rem;
    margin-bottom: 1rem;
}
.st-key-dcf_warning_banner {
    border: 2px solid #ffa421;
    border-radius: 8px;
    background: rgba(255, 164, 33, 0.10);
    padding: 0.6rem 1rem 0.2rem 1rem;
    margin-bottom: 1rem;
}
.dcf-badge {
    display: inline-block; padding: 1px 7px; border-radius: 4px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; vertical-align: middle;
}
.dcf-reported { background: #0e7c3a; color: #fff; }
.dcf-derived { background: #1f4e9c; color: #fff; }
.dcf-estimate { background: #b45309; color: #fff; }
.dcf-user { background: #5b21b6; color: #fff; }
.dcf-unavailable { background: #7a1f1f; color: #fff; }
.dcf-prov { font-size: 0.78rem; opacity: 0.75; }
</style>
"""

#: Headline terminal value method, label to engine key. The first entry is what
#: the radio selects by default, and it is ordered to match
#: ``dcf_model.DEFAULT_HEADLINE_METHOD`` so the UI and the engine agree.
_HEADLINE_METHODS = {
    "Exit multiple": "exit_multiple",
    "Gordon growth": "gordon",
}

#: Display scales for the historical figures table: divisor, header suffix, decimals.
_UNITS = {
    "As reported": (1.0, "", 0),
    "Millions": (1e6, " (m)", 1),
    "Billions": (1e9, " (bn)", 2),
}

_HISTORICAL_LABELS = {
    "revenue": "Revenue",
    "ebit": "EBIT",
    "pretax_income": "Pretax income",
    "net_income": "Net income",
    "cfo": "Operating cash flow",
    "dna": "D&A",
    "total_assets": "Total assets",
    "total_liabilities": "Total liabilities",
    "total_equity": "Total equity",
    "cash": "Cash & equivalents",
}

_BADGE_CLASS = {
    Origin.REPORTED: "dcf-reported",
    Origin.DERIVED: "dcf-derived",
    Origin.ESTIMATE: "dcf-estimate",
    Origin.USER: "dcf-user",
    Origin.UNAVAILABLE: "dcf-unavailable",
}


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------


def _badge(origin: Origin) -> str:
    return (
        f'<span class="dcf-badge {_BADGE_CLASS[origin]}">{origin.label}</span>'
    )


def _provenance(sourced: Sourced) -> None:
    st.markdown(
        f'{_badge(sourced.origin)} '
        f'<span class="dcf-prov">{sourced.describe(include_origin=False)}</span>',
        unsafe_allow_html=True,
    )


def _untouched(widget_value: float, default_value: float, decimals: int) -> bool:
    """True when a widget still holds its pre-populated default.

    Defaults are rounded before they reach the widget, so the comparison has to
    happen at the same precision — otherwise every untouched field claims to be
    a user override.
    """
    return abs(round(widget_value, decimals) - round(default_value, decimals)) < 10**-decimals / 2


def _money(value, currency: str = "", digits: int = 0) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "not reported"
    prefix = f"{currency} " if currency else ""
    magnitude = abs(value)
    if magnitude >= 1e12:
        return f"{prefix}{value / 1e12:,.2f}T"
    if magnitude >= 1e9:
        return f"{prefix}{value / 1e9:,.2f}B"
    if magnitude >= 1e6:
        return f"{prefix}{value / 1e6:,.2f}M"
    return f"{prefix}{value:,.{digits}f}"


def _pct(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{digits}%}"


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _load_company_cached(ticker: str, asreported: bool) -> dcf_model.CompanyDataset:
    """Caches the whole normalised dataset, not just the HTTP calls.

    The fetchers underneath are cached individually, but the schema resolution
    and statement normalisation are not free either. Caching here means moving a
    slider re-runs the arithmetic and nothing else — no refetch, no re-parse.
    """
    return dcf_model.load_company(ticker, asreported=asreported)


def _assumption_fingerprint(inputs: dict) -> tuple:
    """A hashable snapshot of every input the valuation depends on.

    Used to tell whether the displayed results still correspond to what the
    assumption panel currently shows. Widgets rerun the script on every change;
    this is what stops the model, the checks and the workbook from being rebuilt
    each time.
    """
    drivers = inputs["drivers"]
    return (
        drivers.revenue_growth,
        drivers.operating_margin,
        drivers.effective_tax_rate,
        drivers.capex_pct_revenue,
        drivers.dna_pct_revenue,
        drivers.dnwc_pct_drevenue,
        round(inputs["wacc"].wacc, 12),
        inputs["terminal_growth"],
        inputs["exit_multiple"],
        inputs["stub"],
        inputs["mid_year"],
        inputs["shares"],
        inputs["share_basis"],
        inputs["include_sti"],
        inputs["cash_floor"],
        inputs["non_operating"],
        inputs["headline_method"],
        inputs["sbc"],
        inputs["deduct_sbc"],
        inputs["etr"],
    )


def _session_reset(ticker: str) -> None:
    """Clears per-company state. Acknowledgements never carry across companies."""
    for key in list(st.session_state.keys()):
        if key.startswith("dcf_input_") or key in {
            "dcf_acks", "dcf_beta_override", "dcf_output"
        }:
            del st.session_state[key]
    st.session_state["dcf_acks"] = {}
    st.session_state["dcf_loaded_ticker"] = ticker


# ---------------------------------------------------------------------------
# Banners
# ---------------------------------------------------------------------------


def _render_issue(issue: val.Issue, acknowledged: dict[str, str]) -> None:
    with st.container(border=True):
        left, right = st.columns([5, 1.3])
        with left:
            st.markdown(f"**{issue.headline()}**")
            st.markdown(issue.message)
            figures = " · ".join(f"**{k}:** {v}" for k, v in issue.values.items())
            if figures:
                st.markdown(figures)
            footer = [f"`{issue.issue_type}`", f"concept: {issue.concept}"]
            if issue.tolerance:
                footer.append(f"tolerance: {issue.tolerance}")
            st.caption(" · ".join(footer))
        with right:
            if st.button("Acknowledge", key=f"ack_{issue.key}", width="stretch"):
                acknowledged[issue.key] = datetime.now().isoformat(timespec="seconds")
                st.rerun()


def _render_banners(issues: list[val.Issue], acknowledged: dict[str, str]) -> None:
    """Red block, then amber block, above everything else. Neither collapses."""
    live = [i for i in issues if i.key not in acknowledged]
    critical = [i for i in live if i.severity is val.Severity.CRITICAL]
    warnings = [i for i in live if i.severity is val.Severity.WARNING]

    if critical:
        with st.container(key="dcf_critical_banner"):
            st.markdown(f"## 🛑 {len(critical)} critical data problem(s)")
            st.markdown(
                "A number was computed, but a core input is suspect. Every figure below "
                "the fold inherits these problems."
            )
            for issue in critical:
                _render_issue(issue, acknowledged)

    if warnings:
        with st.container(key="dcf_warning_banner"):
            st.markdown(f"### ⚠️ {len(warnings)} warning(s)")
            st.markdown("Worth knowing. Not necessarily wrong.")
            for issue in warnings:
                _render_issue(issue, acknowledged)

    if acknowledged:
        acked = [i for i in issues if i.key in acknowledged]
        with st.expander(f"Acknowledged ({len(acked)})", expanded=False):
            st.caption(
                "Acknowledged checks stay listed and are stamped into the export. "
                "They reset when the ticker changes."
            )
            for issue in acked:
                col_a, col_b = st.columns([5, 1.3])
                with col_a:
                    st.markdown(
                        f"**[{issue.severity.value.upper()}] {issue.headline()}** — "
                        f"{issue.message}"
                    )
                    st.caption(f"Acknowledged at {acknowledged[issue.key]}")
                with col_b:
                    if st.button("Un-acknowledge", key=f"unack_{issue.key}", width="stretch"):
                        acknowledged.pop(issue.key, None)
                        st.rerun()

    if not live and not acknowledged:
        st.success("Every validation check passed.")


# ---------------------------------------------------------------------------
# Blocking screens
# ---------------------------------------------------------------------------


def _render_blocking(title: str, body: str, detail: str = "") -> None:
    st.error(f"**{title}**\n\n{body}")
    if detail:
        st.caption(detail)
    st.info(
        "This is a **BLOCKING** condition: the model cannot be computed, so no valuation "
        "is shown. It is not a judgement that the data looks wrong — it is that the data "
        "needed is absent."
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def render_page() -> None:
    st.markdown(_BANNER_CSS, unsafe_allow_html=True)
    st.header("DCF Valuation")
    st.write(
        "Unlevered free cash flow to the firm, discounted to an intrinsic value per share. "
        "Fundamentals from SimFin; index data and the equity risk premium from the shared "
        "market-data layer."
    )

    ticker_column, button_column, basis_column = st.columns(
        [3, 1.2, 2], vertical_alignment="bottom"
    )
    with ticker_column:
        ticker = st.text_input("Ticker", value="AAPL", key="dcf_ticker_input").strip().upper()
    with button_column:
        compute = st.button("Compute", type="primary", width="stretch")
    with basis_column:
        asreported = st.toggle(
            "Use as-reported (not restated) statements",
            value=False,
            key="dcf_asreported",
            help=(
                "Restated figures give a consistent time series, which is what ratio "
                "derivation needs, and are the default. As-reported figures match what the "
                "market saw at the time. The two are never mixed within one model run."
            ),
        )

    if not ticker:
        st.info("Enter a US-listed ticker and press **Compute**.")
        return

    state_key = f"{ticker}|{asreported}"

    if compute:
        # Loading a different company clears the previous one's acknowledgements
        # and assumption overrides. Recomputing the same one keeps them.
        if st.session_state.get("dcf_loaded_ticker") != state_key:
            _session_reset(state_key)
        st.session_state["dcf_run"] = True

    if not st.session_state.get("dcf_run"):
        st.info(
            f"Press **Compute** to load {ticker} and value it. Nothing is fetched until "
            "you do, and once loaded, changing an assumption will not re-run the model "
            "until you press Compute again."
        )
        return

    loaded_key = st.session_state.get("dcf_loaded_ticker", state_key)
    loaded_ticker, loaded_basis = loaded_key.rsplit("|", 1)
    pending_reload = loaded_key != state_key
    acknowledged: dict[str, str] = st.session_state.setdefault("dcf_acks", {})

    # --- Load ------------------------------------------------------------
    try:
        with st.spinner(f"Loading {loaded_ticker} from SimFin…"):
            dataset = _load_company_cached(loaded_ticker, loaded_basis == "True")
    except val.UnsupportedSectorError as error:
        _render_blocking("Unsupported sector", str(error))
        return
    except SchemaResolutionError as error:
        _render_blocking(
            "Could not resolve a required financial concept",
            str(error),
            f"Columns SimFin did return on the {error.statement} statement: "
            f"{', '.join(error.available[:40])}",
        )
        return
    except simfin_client.TickerNotFoundError as error:
        _render_blocking(
            "Ticker not found",
            f"{error.message} Check the symbol — this page covers US-listed equities "
            "that SimFin carries.",
        )
        return
    except simfin_client.SimFinError as error:
        _render_blocking("Could not load data from SimFin", str(error))
        return

    if dataset.annual.empty:
        _render_blocking(
            "No usable statements",
            f"SimFin returned no annual statements for {ticker}, so there is no history "
            "to build a forecast on.",
        )
        return

    valuation_date = date.today()

    # --- Company header --------------------------------------------------
    _render_header(dataset, valuation_date)

    # These slots are written to at the end of the run but occupy the position
    # they were created in, which is how the banners end up directly under the
    # header and above every number on the page. Page order follows the spec:
    # banners, validation panel, historicals, assumptions, projection,
    # valuation, export — the checks come before the answer on purpose.
    notice_slot = st.container()
    banner_slot = st.container()
    validation_slot = st.container()

    _render_historicals(dataset)

    inputs = _collect_assumptions(dataset, valuation_date)
    if inputs is None:
        return

    fingerprint = _assumption_fingerprint(inputs)
    output = st.session_state.get("dcf_output")
    compute = compute or inputs["recompute"]
    needs_run = compute or output is None or output["company"] != loaded_key

    if needs_run:
        try:
            result = dcf_model.run_model(
                dataset,
                inputs["base"],
                drivers=inputs["drivers"],
                wacc_result=inputs["wacc"],
                terminal_growth=inputs["terminal_growth"],
                exit_multiple=inputs["exit_multiple"],
                stub=inputs["stub"],
                mid_year=inputs["mid_year"],
                shares=inputs["shares"],
                share_basis=inputs["share_basis"],
                include_short_term_investments=inputs["include_sti"],
                operating_cash_floor=inputs["cash_floor"],
                non_operating_assets=inputs["non_operating"],
                base_fcff=inputs["base_fcff"],
                headline_method=inputs["headline_method"],
                sbc=inputs["sbc"],
                deduct_sbc=inputs["deduct_sbc"],
            )
        except dcf.TerminalValueError as error:
            st.error(f"**Terminal value cannot be computed.** {error}")
            st.info("Lower the terminal growth rate below the WACC, or raise the WACC.")
            return
        except dcf.ProjectionError as error:
            _render_blocking("The projection cannot be built", str(error))
            return

        context = dcf_model.build_validation_context(
            dataset,
            base=inputs["base"],
            base_fcff=inputs["base_fcff"],
            valuation_date=valuation_date,
            effective_tax_rate=inputs["etr"],
            beta=inputs["beta_outcome"],
            equity_risk_premium=inputs["erp"],
            result=result,
            terminal_growth=inputs["terminal_growth"],
            risk_free_rate=inputs["risk_free"],
            sbc=inputs["sbc"],
        )
        issues, check_results = val.run_all(context, inputs["tolerances"])

        output = {
            "company": loaded_key,
            "fingerprint": fingerprint,
            "computed_at": datetime.now(),
            "result": result,
            "issues": issues,
            "check_results": check_results,
            # The inputs are snapshotted alongside the results so the export and
            # the provenance shown always describe the numbers on screen, not
            # whatever the assumption widgets have since been changed to.
            "inputs": inputs,
        }
        st.session_state["dcf_output"] = output

    result = output["result"]
    issues = output["issues"]
    check_results = output["check_results"]
    computed_inputs = output["inputs"]

    blocking = [i for i in issues if i.severity is val.Severity.BLOCKING]
    if blocking:
        for issue in blocking:
            _render_blocking(issue.check, issue.message, "; ".join(
                f"{k}: {v}" for k, v in issue.values.items()
            ))
        return

    stale_assumptions = output["fingerprint"] != fingerprint
    with notice_slot:
        _render_staleness_notice(
            pending_reload=pending_reload,
            requested_ticker=ticker,
            loaded_ticker=loaded_ticker,
            stale_assumptions=stale_assumptions,
            computed_at=output["computed_at"],
        )

    with banner_slot:
        _render_banners(issues, acknowledged)

    with validation_slot:
        _render_validation_panel(dataset, check_results, computed_inputs["tolerances"])

    unacknowledged_critical = [
        i for i in issues if i.severity is val.Severity.CRITICAL and i.key not in acknowledged
    ]

    _render_projection(dataset, result)
    _render_valuation(
        dataset,
        result,
        computed_inputs,
        unacknowledged_critical,
        stale=stale_assumptions or pending_reload,
        computed_at=output["computed_at"],
    )
    _render_export(
        dataset, result, computed_inputs, issues, check_results, acknowledged, valuation_date
    )


def _render_staleness_notice(
    *,
    pending_reload: bool,
    requested_ticker: str,
    loaded_ticker: str,
    stale_assumptions: bool,
    computed_at: datetime,
) -> None:
    """Says plainly when what is on screen no longer matches the controls.

    Results are never silently updated and never silently left to look current:
    if the inputs have moved on, the page says so above every number.
    """
    if not (pending_reload or stale_assumptions):
        return

    lines = []
    if pending_reload:
        lines.append(
            f"The ticker box says **{requested_ticker}**, but these results are for "
            f"**{loaded_ticker}**."
        )
    if stale_assumptions:
        lines.append(
            "One or more assumptions have changed since these results were computed."
        )
    lines.append(
        f"Everything below is the run from **{computed_at.strftime('%H:%M:%S')}**. "
        "Press **Compute** to bring it up to date."
    )
    st.warning("  \n".join(lines), icon="⏳")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def _render_header(dataset: dcf_model.CompanyDataset, valuation_date: date) -> None:
    with st.container(border=True):
        st.subheader(f"{dataset.name} ({dataset.ticker})")
        columns = st.columns(5)
        columns[0].metric("Sector (SimFin)", dataset.sector_name or "not reported")
        columns[1].metric("Currency", dataset.currency or "not reported")

        fye_month = dataset.fiscal_year_end_month
        fye_label = (
            date(2000, fye_month, 1).strftime("%B") if fye_month else "not reported"
        )
        columns[2].metric("Fiscal year end", fye_label)

        price_label = (
            f"{dataset.last_price:,.2f}" if dataset.last_price is not None else "n/a"
        )
        columns[3].metric(
            "Last close",
            price_label,
            help=f"As of {dataset.price_date}" if dataset.price_date else None,
        )
        columns[4].metric("Market cap", _money(dataset.reported_market_cap, dataset.currency))

        st.caption(
            f"Price as of {dataset.price_date} · share count as of "
            f"{dataset.shares_point_in_time_date} · statements "
            f"{'as-reported' if dataset.asreported else 'restated (SimFin default)'} · "
            f"SimFin template {dataset.template} · "
            f"{len(dataset.annual)} fiscal years retrieved"
        )


# ---------------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------------


def _driver_input(
    label: str,
    key: str,
    ratios: pd.DataFrame,
    driver_key: str,
    *,
    default_basis: str = "5-year average",
) -> tuple[float | None, Sourced]:
    """One driver: a basis selector, an editable value, and its provenance."""
    basis = st.selectbox(
        f"{label} — basis",
        fin.BASES,
        index=fin.BASES.index(default_basis),
        key=f"dcf_input_{key}_basis",
    )
    default = fin.driver_default(ratios, driver_key, basis)

    if not default.ok:
        _provenance(
            Sourced.unavailable(source=f"Historical {label.lower()}", reason=default.detail)
        )
        st.warning(
            f"No usable history for {label.lower()} — {default.detail} There is nothing "
            "to pre-populate from, so this starts empty and must be entered by hand."
        )
        manual = st.number_input(
            f"{label} (%)", value=0.0, step=0.5, key=f"dcf_input_{key}_manual"
        )
        return manual / 100.0, Sourced.user(manual / 100.0, "No historical basis available.")

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
    _provenance(sourced)
    if default.detail:
        st.caption(f"⚠️ {default.detail}")
    return user_value, sourced


def _collect_assumptions(
    dataset: dcf_model.CompanyDataset, valuation_date: date
) -> dict | None:
    """Renders every input and returns the assembled assumption set."""
    st.divider()
    st.subheader("Assumptions")
    st.caption(
        "Every default is derived from reported history and shows where it came from. "
        "Nothing here is invented — where a basis does not exist, the field says so."
    )

    ratios = dataset.ratios
    base = dcf_model.base_period(dataset)
    records: list[dcf_export.AssumptionRecord] = []

    # --- Forecast drivers ------------------------------------------------
    st.markdown("#### Forecast drivers")
    with st.container(border=True):
        st.caption(
            f"Base period: **{base.label}** — {base.detail} "
            f"Balance sheet as of **{base.stocks_as_of}**."
        )
        years = st.slider("Forecast years", 5, 10, 5, key="dcf_input_years")

        driver_values: dict[str, float] = {}
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
                value, sourced = _driver_input(label, key, ratios, driver_key)
                driver_values[driver_key] = value if value is not None else 0.0
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

    # --- Discount rate ---------------------------------------------------
    st.markdown("#### Discount rate")
    discount = st.container(border=True)
    with discount:
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
            _provenance(rf_sourced)
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
            _provenance(erp_sourced)
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
            _provenance(beta_sourced)
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
                cost_of_debt, "Manual yield-to-maturity input — SimFin carries no bond data."
            )
            _provenance(kd_sourced)
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
                    "The implied cost of debt cross-check is unavailable: SimFin reports no "
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

    # --- Capital structure ----------------------------------------------
    stocks = base.stocks
    debt_frame = pd.DataFrame([stocks])
    total_debt_series = fin.total_debt(debt_frame)
    debt_value = (
        float(total_debt_series.iloc[0])
        if not total_debt_series.empty and not pd.isna(total_debt_series.iloc[0])
        else 0.0
    )

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
        _render_blocking(
            "No share count available",
            f"SimFin returned no share count for {dataset.ticker}, so equity value cannot "
            "be divided into a per-share figure.",
        )
        return None

    st.caption(
        f"Using **{shares:,.0f}** shares — {share_basis}. "
        f"SimFin provides no treasury-stock-method calculation for outstanding options and "
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

    equity_value = (dataset.last_price or 0.0) * (dataset.shares_point_in_time or 0.0)
    try:
        wacc_result = wacc_lib.compute_wacc(
            equity_value=equity_value,
            debt_value=debt_value,
            risk_free_rate=risk_free,
            beta=beta_value,
            equity_risk_premium=erp,
            cost_of_debt=cost_of_debt,
            marginal_tax_rate=marginal_tax,
        )
    except ValueError as error:
        _render_blocking("The WACC cannot be weighted", str(error))
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
            f"({_money(equity_value, dataset.currency)}) · D = book value of total debt "
            f"({_money(debt_value, dataset.currency)}), as of {base.stocks_as_of}."
        )

    # --- Terminal value & conventions ------------------------------------
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
            _provenance(exit_sourced)
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
                "SimFin reports no fiscal year end month, so the stub period cannot be "
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
                    "Fiscal year end from SimFin `endFy` plus the latest annual report date",
                    f"days to {fye} ÷ 365" if fye else "no fiscal year end available",
                ),
            )
        )

    # --- Cash, SBC, and the bridge ---------------------------------------
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
            st.caption(
                f"Balance-sheet cash {_money(cash_value, dataset.currency)} · short-term "
                f"investments {_money(stocks.get('short_term_investments'), dataset.currency)} "
                f"(as of {base.stocks_as_of})"
            )

        with col_b:
            st.markdown("**Stock-based compensation**")
            sbc_reported = stocks.get("sbc")
            base_sbc = base.flows.get("sbc")
            non_cash = base.flows.get("non_cash_items")
            sbc_available = base_sbc is not None and not pd.isna(base_sbc)

            if not sbc_available:
                st.warning(
                    "**SimFin reports no stock-based compensation for this company.** The "
                    "field is empty and nothing has been inferred. Enter the figure from the "
                    "cash flow statement in the latest 10-K, or use the upper-bound estimate."
                )
                if non_cash is not None and not pd.isna(non_cash):
                    st.caption(
                        f"For context, total **Non-Cash Items** for the base period is "
                        f"{_money(non_cash, dataset.currency)}. That is a superset of SBC — it "
                        f"also contains impairments, FX and deferred items — so it is **not** "
                        f"an SBC figure."
                    )
                    if st.button(
                        "Use Non-Cash Items as an upper-bound estimate", key="dcf_sbc_proxy"
                    ):
                        st.session_state["dcf_input_sbc"] = float(non_cash)
                        st.rerun()

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
                sbc_sourced = Sourced.reported(sbc_input, "SimFin CF: Stock-Based Compensation")
            elif sbc_input and non_cash is not None and abs(sbc_input - float(non_cash)) < 1:
                sbc_sourced = Sourced.estimate(
                    sbc_input,
                    "SimFin CF: Non-Cash Items",
                    "Upper bound — Non-Cash Items also contains impairments, FX and deferred "
                    "items, so true SBC is lower by an unknown amount.",
                )
            elif sbc_input:
                sbc_sourced = Sourced.user(sbc_input, "Entered manually from the filing.")
            else:
                sbc_sourced = Sourced.unavailable(
                    "SimFin CF: Stock-Based Compensation",
                    "Not reported and not supplied.",
                )
            _provenance(sbc_sourced)

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
                    "**Deduct SBC from FCFF** above to treat it as a real cost, then press "
                    "**Compute**.",
                    icon="ℹ️",
                )
            elif sbc is not None and deduct_sbc:
                st.caption(
                    f"Deducting {_money(sbc, dataset.currency)} from FCFF, grown with revenue "
                    f"across the forecast."
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
                    "SimFin reports no long-term investments line for this company, so this "
                    "starts at zero by absence rather than by measurement."
                )
            interest_basis = st.radio(
                "Interest expense line for FCFF method A",
                ["net", "gross"],
                horizontal=True,
                key="dcf_input_interest_basis",
                help="Where SimFin reports both, which to use for the after-tax add-back.",
            )

    # --- Effective tax rate for FCFF -------------------------------------
    etr_series = fin.effective_tax_rate(dataset.annual).dropna()
    etr = float(etr_series.tail(5).median()) if not etr_series.empty else None
    delta_nwc, delta_nwc_span = dcf_model.base_period_delta_nwc(dataset, base)

    base_fcff = dcf_model.base_period_fcff(
        dataset,
        base,
        tax_rate=etr if etr is not None else driver_values["effective_tax_rate"],
        interest_basis=interest_basis,
        delta_nwc=delta_nwc,
    )
    _render_fcff_panel(dataset, base, base_fcff, etr, etr_series, delta_nwc_span)

    # A second Compute, at the bottom of a long assumptions panel. Without it the
    # only way to apply a change is to scroll back to the top, which reads as
    # "I changed it and nothing happened".
    recompute = st.button(
        "Compute with these assumptions",
        type="primary",
        width="stretch",
        key="dcf_compute_bottom",
    )

    return {
        "recompute": recompute,
        "base": base,
        "drivers": drivers,
        "wacc": wacc_result,
        "terminal_growth": terminal_growth,
        "exit_multiple": exit_multiple,
        "stub": stub,
        "mid_year": mid_year,
        "shares": shares,
        "share_basis": share_basis,
        "include_sti": include_sti,
        "cash_floor": cash_floor,
        "non_operating": non_operating or None,
        "base_fcff": base_fcff,
        "headline_method": headline_method,
        "sbc": sbc,
        "deduct_sbc": deduct_sbc,
        "etr": etr,
        "erp": erp,
        "risk_free": risk_free,
        "beta_outcome": beta_outcome,
        "records": records,
        "tolerances": val.DEFAULT_TOLERANCES,
    }


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
            st.metric("FCFF (method A)", _money(base_fcff.method_a, dataset.currency))
        with right:
            st.markdown("**Method B — income statement build**")
            st.code("NOPAT = EBIT × (1 − tax rate)\nFCFF  = NOPAT + D&A − CapEx − ΔNWC")
            st.metric("FCFF (method B)", _money(base_fcff.method_b, dataset.currency))

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
                        else _money(value, dataset.currency)
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
            "US GAAP; an IFRS filer may classify it in financing. SimFin exposes interest "
            "*expense* (accrual), not interest *paid* (cash), so that is what is used. "
            "The two builds legitimately differ where working capital moves a lot: method B "
            "subtracts the balance-sheet movement, while method A picks up whatever the cash "
            "flow statement booked, including acquisitions and FX translation."
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def _cross_implied(terminal: dcf.TerminalValue) -> str:
    """Restates one terminal value method in the other method's units.

    Gordon growth produces a terminal value, which divided by terminal EBITDA is
    an exit multiple. An exit multiple produces a terminal value, which inverted
    through the Gordon formula is a perpetual growth rate. Each row therefore
    reports a different quantity, and the label says which.
    """
    if terminal.implied_exit_multiple is not None:
        return f"an exit multiple of {terminal.implied_exit_multiple:,.2f}×"
    if terminal.implied_growth is not None:
        return f"perpetual growth of {terminal.implied_growth:.2%}"
    return "—"


def _render_valuation(
    dataset: dcf_model.CompanyDataset,
    result: dcf_model.ModelResult,
    inputs: dict,
    unacknowledged_critical: list[val.Issue],
    *,
    stale: bool = False,
    computed_at: datetime | None = None,
) -> None:
    st.divider()
    st.subheader("Valuation")

    marker = ""
    # The staleness warning is repeated here, not just at the top of the page.
    # An assumption is changed halfway down a long page, and the number the user
    # is watching is here — a notice they have to scroll up to see is a notice
    # that reads as "nothing happened".
    if stale:
        marker = " ⏳"
        stamp = f" from {computed_at.strftime('%H:%M:%S')}" if computed_at else ""
        st.warning(
            f"**These figures are from the previous run{stamp} and do not reflect the "
            f"assumptions currently shown above.** Press **Compute** — at the top of the "
            f"page or at the end of the assumptions panel — to apply them.",
            icon="⏳",
        )

    if unacknowledged_critical:
        marker = " ⚠️"
        st.error(
            f"⚠️ **The figures below carry {len(unacknowledged_critical)} unacknowledged "
            f"critical data problem(s).** Read the red banner above before using this number."
        )

    bridge = result.headline_bridge
    upside = bridge.upside_vs(dataset.last_price) if dataset.last_price else None
    columns = st.columns(4)
    columns[0].metric(
        f"Value per share{marker}",
        f"{bridge.value_per_share:,.2f}",
        help=f"Headline method: {result.headline_terminal.method}"
        + (" · showing a previous run; press Compute to refresh" if stale else ""),
    )
    columns[1].metric("Current price", f"{dataset.last_price:,.2f}" if dataset.last_price else "n/a")
    columns[2].metric("Upside / downside", _pct(upside) if upside is not None else "n/a")
    columns[3].metric("WACC", f"{result.wacc.wacc:.2%}")

    st.markdown("#### Both terminal value methods, side by side")
    rows = []
    for method, terminal in result.terminal_values.items():
        enterprise_value = result.enterprise_values[method]
        method_bridge = result.bridges[method]
        rows.append(
            {
                "Method": terminal.method + (" (headline)" if method == result.headline_method else ""),
                "Terminal value": _money(terminal.value, dataset.currency),
                "PV of terminal value": _money(terminal.present_value, dataset.currency),
                "Enterprise value": _money(enterprise_value, dataset.currency),
                "Equity value": _money(method_bridge.equity_value, dataset.currency),
                "Value per share": f"{method_bridge.value_per_share:,.2f}",
                "TV share of EV": _pct(terminal.share_of(enterprise_value), 1),
                # Each method is cross-checked by restating it in the *other*
                # method's units, so the two columns are deliberately different
                # quantities rather than a like-for-like pair.
                "This method implies": _cross_implied(terminal),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.caption(
        "**Reading the cross-check.** Each method is restated in the other's terms, so the "
        "last column is a different quantity on each row — that is the point of it, not a "
        "misalignment. The Gordon growth row answers *“what exit multiple would my "
        "perpetuity assumption have to be worth?”*, and the exit multiple row answers *“what "
        "perpetual growth rate is my chosen multiple secretly assuming?”* If the exit "
        "multiple row implies growth you would never underwrite, the multiple is too high."
    )

    gordon_vps = result.bridges["gordon"].value_per_share
    exit_vps = result.bridges["exit_multiple"].value_per_share
    if gordon_vps:
        st.caption(
            f"The two methods differ by **{abs(exit_vps / gordon_vps - 1):.1%}** on value per "
            f"share. A gap is informative, not an error."
        )
    st.caption(
        f"Terminal value discounted at exponent {result.headline_terminal.discount_exponent:.4f}, "
        f"the same convention as the final projected year "
        f"({'mid-year' if result.projection.mid_year else 'end-of-year'})."
    )

    st.markdown("#### Equity bridge")
    st.caption(
        f"Every bridge item is taken from the balance sheet as of "
        f"**{result.base.stocks_as_of}**. SimFin provides no treasury-stock-method calculation "
        f"for outstanding options and RSUs, so the diluted count is not fully loaded."
    )
    bridge_rows = [
        {
            "Line": line.label,
            "Amount": _money(line.amount, dataset.currency),
            "Source": "reported" if line.reported else "not reported",
            "Note": line.note,
        }
        for line in bridge.lines
    ]
    bridge_rows.append(
        {
            "Line": f"÷ shares — {bridge.share_basis}",
            "Amount": f"{bridge.shares:,.0f}",
            "Source": "reported",
            "Note": "",
        }
    )
    bridge_rows.append(
        {
            "Line": "= Value per share",
            "Amount": f"{bridge.value_per_share:,.2f}",
            "Source": "derived",
            "Note": "",
        }
    )
    st.dataframe(pd.DataFrame(bridge_rows), width="stretch", hide_index=True)


def _render_projection(
    dataset: dcf_model.CompanyDataset, result: dcf_model.ModelResult
) -> None:
    st.divider()
    st.subheader("Projection")
    st.caption(
        "FCFF = EBIT × (1 − tax) + D&A − CapEx − ΔNWC − SBC deducted. "
        "PV = FCFF for the period × discount factor."
    )

    frame = dcf_export.projection_frame(result)
    display = frame.copy()
    for column in (
        "Revenue (full year)", "EBIT", "NOPAT", "D&A", "CapEx", "ΔNWC",
        "SBC deducted", "FCFF (full year)", "FCFF (this period)", "PV of FCFF",
    ):
        display[column] = frame[column].map(lambda v: _money(v, dataset.currency))
    for column in ("Revenue growth", "Operating margin", "Tax rate"):
        display[column] = frame[column].map(lambda v: f"{v:.2%}")
    for column in ("Period fraction", "Discount exponent", "Discount factor"):
        display[column] = frame[column].map(lambda v: f"{v:.4f}")

    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        f"Sum of discounted explicit cash flows: "
        f"**{_money(result.projection.pv_explicit, dataset.currency)}** · "
        f"PV of terminal value: "
        f"**{_money(result.headline_terminal.present_value, dataset.currency)}**"
    )


def _render_historicals(dataset: dcf_model.CompanyDataset) -> None:
    st.divider()
    st.subheader("Historical financials")

    ratios = dataset.ratios.copy()
    usable = {
        fin.DRIVER_LABELS[key]: int(ratios[key].notna().sum())
        for key in fin.DRIVER_KEYS
        if key in ratios.columns
    }
    st.caption(
        f"{len(dataset.annual)} fiscal years retrieved. Years used per driver: "
        + " · ".join(f"{k}: {v}" for k, v in usable.items())
        + ". Years missing a component are excluded, never zero-filled."
    )

    display = ratios.copy()
    display.columns = [fin.DRIVER_LABELS.get(c, c) for c in display.columns]
    st.dataframe(
        display.style.format("{:.2%}", na_rep="—"), width="stretch"
    )

    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Revenue and EBIT", "Driver ratios"),
    )
    index = [str(i) for i in dataset.annual.index]
    if "revenue" in dataset.annual.columns:
        figure.add_trace(
            go.Bar(x=index, y=dataset.annual["revenue"], name="Revenue"), row=1, col=1
        )
    if "ebit" in dataset.annual.columns:
        figure.add_trace(
            go.Bar(x=index, y=dataset.annual["ebit"], name="EBIT"), row=1, col=1
        )
    for key in ("operating_margin", "capex_pct_revenue", "dna_pct_revenue", "effective_tax_rate"):
        if key in ratios.columns:
            figure.add_trace(
                go.Scatter(
                    x=index, y=ratios[key], mode="lines+markers",
                    name=fin.DRIVER_LABELS[key], connectgaps=False,
                ),
                row=2, col=1,
            )
    figure.update_layout(
        height=650, template="plotly_dark", margin=dict(t=40, b=10, l=10, r=10), barmode="group"
    )
    figure.update_yaxes(tickformat=".0%", row=2, col=1)
    st.plotly_chart(figure, width="stretch")

    heading, units = st.columns([3, 2], vertical_alignment="bottom")
    with heading:
        st.markdown("**Reported figures, with the filing behind each year**")
    with units:
        unit = st.radio(
            "Units",
            list(_UNITS),
            index=1,
            horizontal=True,
            key="dcf_hist_units",
            help="Display only — the underlying figures and every calculation are "
                 "unchanged, and the export always carries the full precision.",
        )

    divisor, suffix, decimals = _UNITS[unit]

    table = dataset.annual.copy()
    table.insert(0, "Fiscal year", table.index)
    table["Source filing"] = [
        dataset.filing_link(int(year)) for year in dataset.annual.index
    ]
    keep = [
        "Fiscal year", "revenue", "ebit", "pretax_income", "net_income", "cfo", "dna",
        "total_assets", "total_liabilities", "total_equity", "cash", "Source filing",
    ]
    table = table[[c for c in keep if c in table.columns]]

    money_columns = [c for c in table.columns if c not in ("Fiscal year", "Source filing")]
    for column in money_columns:
        table[column] = table[column] / divisor
    table = table.rename(
        columns={
            column: f"{_HISTORICAL_LABELS.get(column, column)}{suffix}"
            for column in money_columns
        }
    )
    table["Fiscal year"] = table["Fiscal year"].astype(str)

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Source filing": st.column_config.LinkColumn(
                "Source filing", display_text="View filing"
            ),
            **{
                f"{_HISTORICAL_LABELS.get(c, c)}{suffix}": st.column_config.NumberColumn(
                    format=f"%.{decimals}f"
                )
                for c in money_columns
            },
        },
    )
    st.caption(
        f"Figures in {unit.lower()} of {dataset.currency or 'reporting currency'}, "
        f"as reported by SimFin. Blank cells are line items the company did not report — "
        f"they are not zeros."
    )


def _render_validation_panel(
    dataset: dcf_model.CompanyDataset,
    check_results: list[val.CheckResult],
    tolerances: val.Tolerances,
) -> None:
    st.divider()
    st.subheader("Validation panel")

    results_frame = val.results_frame(check_results)
    passed = int((results_frame["Status"] == "pass").sum())
    failed = int((results_frame["Status"] == "fail").sum())
    skipped = int((results_frame["Status"] == "not evaluated").sum())
    columns = st.columns(4)
    columns[0].metric("Checks passed", passed)
    columns[1].metric("Checks failed", failed)
    columns[2].metric("Not evaluated", skipped)
    columns[3].metric("Fiscal years retrieved", len(dataset.annual))
    if skipped:
        st.caption(
            "“Not evaluated” means the inputs a check needs were absent — it is not a pass."
        )
    st.dataframe(results_frame, width="stretch", hide_index=True)

    with st.expander("Resolved schema mappings", expanded=False):
        st.caption(
            "Concept-to-column resolution against the columns SimFin actually returned for "
            "this company. Nothing here is hardcoded at a call site."
        )
        st.dataframe(dataset.schema.as_frame(), width="stretch", hide_index=True)
        if dataset.schema.ambiguous:
            st.warning(
                "Some concepts matched more than one column. The first alias in the "
                "preference order was used; the alternates are listed above."
            )

    with st.expander("Tolerances in use", expanded=False):
        st.dataframe(tolerances.describe(), width="stretch", hide_index=True)

    with st.expander("SimFin's own integrity flags and derived figures", expanded=False):
        if "value_check" in dataset.annual_meta.columns:
            flags = dataset.annual_meta[["value_check"]].copy()
            flags.insert(0, "Fiscal year", flags.index)
            st.dataframe(flags, width="stretch", hide_index=True)
        if "derived_fcf" in dataset.annual.columns:
            st.caption(
                "SimFin's DERIVED `Free Cash Flow` is shown for reference only and drives "
                "nothing. It has been observed to exceed operating cash flow for companies "
                "with positive capex, which is not possible, so it is not wired to a check."
            )
            comparison = pd.DataFrame(
                {
                    "SimFin derived FCF": dataset.annual.get("derived_fcf"),
                    "CFO": dataset.annual.get("cfo"),
                    "CFO − CapEx": dataset.annual.get("cfo") - fin.capex(dataset.annual).values,
                }
            )
            st.dataframe(comparison, width="stretch")


def _render_export(
    dataset: dcf_model.CompanyDataset,
    result: dcf_model.ModelResult,
    inputs: dict,
    issues: list[val.Issue],
    check_results: list[val.CheckResult],
    acknowledged: dict[str, str],
    valuation_date: date,
) -> None:
    st.divider()
    st.subheader("Export")
    st.caption(
        "The workbook carries every assumption with its provenance, the projection line by "
        "line, the equity bridge, all validation results, and which checks were acknowledged. "
        "It is built from the same snapshot the page is showing."
    )

    # Acknowledgements change without a recompute, so the export is rebuilt when
    # they move — but not on every unrelated widget interaction.
    cache_key = (id(result), tuple(sorted(acknowledged.items())))
    cached = st.session_state.get("dcf_export_cache")
    if cached is None or cached["key"] != cache_key:
        try:
            workbook = dcf_export.build_workbook(
                dataset=dataset,
                result=result,
                assumptions=inputs["records"],
                issues=issues,
                check_results=check_results,
                acknowledged=acknowledged,
                tolerances=inputs["tolerances"],
                valuation_date=valuation_date,
            )
        except Exception as error:  # noqa: BLE001 - surfaced, never swallowed
            st.error(f"The workbook could not be built: {error}")
            return

        csv_text = dcf_export.build_csv(
            dataset=dataset,
            result=result,
            assumptions=inputs["records"],
            issues=issues,
            check_results=check_results,
            acknowledged=acknowledged,
            valuation_date=valuation_date,
        )
        cached = {"key": cache_key, "workbook": workbook, "csv": csv_text}
        st.session_state["dcf_export_cache"] = cached

    workbook = cached["workbook"]
    csv_text = cached["csv"]

    stamp = valuation_date.isoformat()
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download Excel workbook",
            data=workbook,
            file_name=f"{dataset.ticker}_dcf_{stamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    with right:
        st.download_button(
            "Download CSV",
            data=csv_text,
            file_name=f"{dataset.ticker}_dcf_{stamp}.csv",
            mime="text/csv",
            width="stretch",
        )
