"""The assumptions panel: every input the model takes, each with its provenance.

:func:`collect` reads the panel's widgets out of session state into an
:class:`~app.logic.dcf.inputs.Overrides`, has :func:`app.logic.dcf.inputs.resolve`
assemble the run from them — the function the web app's API calls too, so an
untouched panel is one run on both — and then draws the panel from what came
back. Nothing here decides a default, a provenance or a driver path; this
module only puts each input on screen with its badge and its notes.

Reading the widgets before drawing them works because Streamlit writes a
widget's new value into session state before the rerun it triggers. An input
nobody has touched holds the default it was last seeded with, which
:func:`_typed` reads as no override at all, so it follows its default when that
moves — a cost of debt nobody typed follows the risk-free rate it is built on.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from app.data import market_data
from app.logic.dcf import inputs as dcf_inputs
from app.logic.dcf import model as dcf_model
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

**No public bonds?** The field already starts from the synthetic rating approach — EBIT ÷ interest
expense mapped to a rating band, plus that band's spread over the risk-free rate — which is an
estimate. A traded bond's yield is better evidence where one exists.
"""

#: Headline terminal value method, label to engine key. The first entry is what
#: the radio selects by default, and it is ordered to match
#: ``dcf_model.DEFAULT_HEADLINE_METHOD`` so the UI and the engine agree.
_HEADLINE_METHODS = {label: key for key, label in dcf_inputs.HEADLINE_METHODS.items()}

_OWN_ROIC = dcf_inputs.OWN_ROIC
_RONIC_AT_WACC = dcf_inputs.RONIC_AT_WACC
_RONIC_MANUAL = dcf_inputs.RONIC_MANUAL

#: Each driver's widget-key stem: ``dcf_input_<stem>_basis`` and ``_value``.
_DRIVER_WIDGETS = {
    "revenue_growth": "growth",
    "operating_margin": "margin",
    "effective_tax_rate": "etr",
    "capex_pct_revenue": "capex",
    "dna_pct_revenue": "dna",
    "nwc_pct_revenue": "nwc",
}

#: Session keys for each target toggle and value. The margin's predate the
#: other two and keep their names so an open session is not reset.
_TARGET_KEYS = {
    "operating_margin": ("dcf_input_target_margin_on", "dcf_input_target_margin"),
    "capex_pct_revenue": ("dcf_input_target_capex_on", "dcf_input_target_capex"),
    "dna_pct_revenue": ("dcf_input_target_dna_on", "dcf_input_target_dna"),
}


# ---------------------------------------------------------------------------
# Session state ↔ overrides
# ---------------------------------------------------------------------------


def _seed(state, key: str, shown: float, *, basis: object = None) -> None:
    """Puts ``shown`` — a default, as displayed — into an input that has not been typed in.

    An input still holding the last default it was given follows its default
    when that moves. A typed value is kept until the basis it was typed against
    changes: a new basis brings its own default, so a figure typed against the
    five-year average is not read as an override of the TTM. The seed is stored
    beside the input under the ``dcf_input_`` prefix, so a change of company
    clears it with everything else.
    """
    seed_key = f"{key}_seed"
    last = state.get(seed_key)
    if key not in state or last is None or last[0] != basis or _same(state[key], last[1]):
        state[key] = shown
    state[seed_key] = (basis, shown)


def _typed(state, key: str, scale: float = 1.0, *, basis: object = None) -> float | None:
    """The value typed into ``key``, scaled to the engine's units; ``None`` when untouched."""
    last = state.get(f"{key}_seed")
    if key not in state or last is None or last[0] != basis or _same(state[key], last[1]):
        return None
    return float(state[key]) / scale


def _same(a, b) -> bool:
    return abs(float(a) - float(b)) < 1e-9


def _beta_basis(state) -> tuple[str, bool]:
    """The regression a typed beta overrides — a new window brings a new default."""
    frequency, blume, _ = dcf_inputs.beta_settings(
        dcf_inputs.Overrides(
            beta_frequency=state.get("dcf_input_beta_freq"), blume=state.get("dcf_input_blume")
        )
    )
    return frequency, blume


def _overrides(state) -> dcf_inputs.Overrides:
    """What the widgets hold that is not their default."""
    bases, values = {}, {}
    for key, stem in _DRIVER_WIDGETS.items():
        basis = state.get(f"dcf_input_{stem}_basis")
        if basis is not None:
            bases[key] = basis
        typed = _typed(state, f"dcf_input_{stem}_value", 100.0, basis=basis)
        if typed is not None:
            values[key] = typed

    target_on, target_values = {}, {}
    for key, (on_key, value_key) in _TARGET_KEYS.items():
        if on_key in state:
            target_on[key] = bool(state[on_key])
        typed = _typed(state, value_key, 100.0)
        if typed is not None:
            target_values[key] = typed

    hold = state.get("dcf_input_hold")
    return dcf_inputs.Overrides(
        years=state.get("dcf_input_years"),
        bases=bases,
        values=values,
        fade=state.get("dcf_input_fade"),
        hold=int(hold) if hold is not None else None,
        target_on=target_on,
        target_values=target_values,
        acquisitions=state.get("dcf_input_acquisitions"),
        risk_free=_typed(state, "dcf_input_rf", 100.0),
        erp=_typed(state, "dcf_input_erp", 100.0),
        beta_frequency=state.get("dcf_input_beta_freq"),
        blume=state.get("dcf_input_blume"),
        beta=_typed(state, "dcf_input_beta", basis=_beta_basis(state)),
        cost_of_debt=_typed(state, "dcf_input_kd", 100.0),
        marginal_tax=_typed(state, "dcf_input_marginal_tax", 100.0),
        share_basis=state.get("dcf_input_share_basis"),
        terminal_growth=_typed(state, "dcf_input_g", 100.0),
        ronic_basis=state.get("dcf_input_ronic_basis"),
        ronic=_typed(state, "dcf_input_ronic", 100.0),
        terminal_at_marginal=state.get("dcf_input_terminal_tax"),
        ebita_proxy=state.get("dcf_input_ebita_proxy"),
        exit_multiple=_typed(state, "dcf_input_exit"),
        headline_method=_HEADLINE_METHODS.get(state.get("dcf_input_headline")),
        mid_year=state.get("dcf_input_midyear"),
        include_sti=state.get("dcf_input_sti"),
        cash_floor=state.get("dcf_input_cash_floor"),
        sbc=_typed(state, "dcf_input_sbc"),
        non_operating=state.get("dcf_input_nonop"),
    )


def _clamp_hold(state) -> None:
    """Shortening the forecast pulls the hold in with it, before either widget draws."""
    years = state.get("dcf_input_years", dcf_model.DEFAULT_FORECAST_YEARS)
    if state.get("dcf_input_hold") is not None and state["dcf_input_hold"] > years - 1:
        state["dcf_input_hold"] = years - 1


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------


def _note(note: dcf_inputs.Note) -> None:
    {"warning": st.warning, "info": st.info}.get(note.level, st.caption)(note.text)


def _pct(value: float | None) -> float:
    return round(float(value or 0.0) * 100, 4)


def collect(dataset: dcf_model.CompanyDataset, valuation_date: date) -> dict | None:
    """Resolves the assumption set from the widgets, then draws the panel for it.

    One function per block of the panel, top to bottom. ``None`` means a
    blocking screen has been drawn in place of the rest of the page.

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

    state = st.session_state
    _clamp_hold(state)
    overrides = _overrides(state)
    frequency, blume, years = dcf_inputs.beta_settings(overrides)
    beta = dcf_model.regress_company_beta(dataset, frequency=frequency, blume=blume, years=years)
    try:
        resolved = dcf_inputs.resolve(
            dataset,
            valuation_date=valuation_date,
            risk_free=market_data.fetch_10y_treasury_rate(),
            erp=market_data.fetch_erp(),
            beta=beta,
            overrides=overrides,
        )
    except dcf_inputs.InvalidOverride as error:
        common.render_blocking("An assumption is out of range", str(error))
        return None

    _forecast_drivers(resolved)
    discount = _discount_inputs(dataset, resolved)
    _share_count(dataset, resolved)
    if resolved.blocked is not None:
        common.render_blocking(resolved.blocked.title, resolved.blocked.message)
        return None
    _weighted_cost(dataset, resolved, discount)
    _terminal_and_conventions(resolved)
    _cash_sbc_and_bridge(dataset, resolved)
    _render_fcff_panel(dataset, resolved)
    return resolved.inputs


def _forecast_drivers(a: dcf_inputs.Assumptions) -> None:
    """The six drivers, the shape of the growth path, and the year-N targets."""
    state = st.session_state
    st.markdown("#### Forecast drivers")
    with st.container(border=True):
        st.caption(
            f"Base period: **{a.base.label}** — {a.base.detail} "
            f"Balance sheet as of **{a.base.stocks_as_of}**."
        )
        st.slider(
            "Forecast years", *dcf_inputs.YEARS_RANGE, dcf_model.DEFAULT_FORECAST_YEARS,
            key="dcf_input_years",
            help="Ten years by default, so a company still growing well above the "
                 "terminal rate has room to compound before it fades. The excess return "
                 "lives in the explicit forecast; the terminal year earns only its RONIC.",
        )

        grid = st.columns(3)
        for index, driver in enumerate(a.drivers_in):
            with grid[index % 3]:
                _driver_input(driver)
                if driver.key == "capex_pct_revenue":
                    st.caption(
                        "Cash capex only. Assets obtained under finance leases — material "
                        "at some hyperscalers — are excluded, and their principal "
                        "repayments sit in financing cash flow, which FCFF does not see."
                    )
                if driver.key == "nwc_pct_revenue":
                    st.caption(
                        "Working capital is held at this share of revenue, so it moves "
                        "with each year's change in revenue. A negative share — customers "
                        "funding the business — releases cash as revenue grows."
                    )

        path_left, path_right = st.columns(2)
        with path_left:
            st.toggle(
                "Fade revenue growth toward terminal growth",
                value=True,
                key="dcf_input_fade",
                help="Year-1 growth is held for the years below, then steps linearly toward "
                     "the perpetuity rate, landing on it in the terminal year — no cliff "
                     "between year N and the terminal value. Off holds the rate flat for "
                     "every forecast year.",
            )
            if "dcf_input_hold" not in state:
                state["dcf_input_hold"] = a.hold
            st.number_input(
                "Years to hold year-1 growth",
                min_value=1,
                max_value=a.years - 1,
                step=1,
                key="dcf_input_hold",
                disabled=not a.fade,
            )
        with path_right:
            for target in a.targets:
                _driver_target(target, a.years)

        _acquisitions_toggle(a)
        if a.capex_estimate:
            st.warning(a.capex_estimate)


def _driver_input(driver: dcf_inputs.Driver) -> None:
    """One driver: a basis selector, an editable value, and its provenance."""
    stem = _DRIVER_WIDGETS[driver.key]
    st.selectbox(
        f"{driver.label} — basis",
        driver.bases,
        index=driver.bases.index(driver.default_basis),
        key=f"dcf_input_{stem}_basis",
    )
    after = [n for n in driver.input.notes if n.text.startswith("⚠️")]
    for note in driver.input.notes:
        if note not in after:
            _note(note)
    widget_key = f"dcf_input_{stem}_value"
    _seed(st.session_state, widget_key, _pct(driver.input.default), basis=driver.basis)
    st.number_input(f"{driver.label} (%)", step=0.25, key=widget_key, format="%.4f")
    common.provenance(driver.input.sourced)
    for note in after:
        _note(note)


def _driver_target(target: dcf_inputs.Target, years: int) -> None:
    """An optional year-N level the driver moves to, linearly from year 1."""
    on_key, value_key = _TARGET_KEYS[target.key]
    st.toggle(
        f"Move the {target.label} to a target",
        value=target.default_on,
        key=on_key,
        help=f"The {target.label} moves linearly from its starting value in year 1 to the "
             "target in year N; the terminal year keeps the final year's value.",
    )
    after = [n for n in target.notes if n.text.startswith("⚠️")]
    for note in target.notes:
        if note not in after:
            _note(note)
    if target.key == "capex_pct_revenue":
        st.caption(
            "D&A stays at its starting ratio, as the margin does: the margin already deducts "
            "today's D&A, so move the two together or cash flow moves by the whole change in "
            "D&A rather than by its tax shield."
        )
    if not target.on:
        return
    _seed(st.session_state, value_key, _pct(target.seed))
    st.number_input(
        f"Target {target.label} in year {years} (%)", step=0.5, key=value_key, format="%.4f"
    )
    if target.default_on:
        common.provenance(target.input.sourced)
    for note in after:
        _note(note)


def _acquisitions_toggle(a: dcf_inputs.Assumptions) -> None:
    """Folds acquisition spend into reinvestment, for a company that buys growth.

    Reported revenue growth includes what acquisitions bought; capex excludes
    what they cost. Adding the historical acquisition rate to every year's
    capex makes the forecast pay for the growth it projects.
    """
    if a.acquisitions is None:
        return
    st.toggle(
        "Include acquisitions in reinvestment",
        value=False,
        key="dcf_input_acquisitions",
        help="Adds the 5-year average of acquisition spend ÷ revenue to the capex "
             "driver in every forecast year.",
    )
    spend = a.acquisitions.spend
    st.caption(
        f"Acquisitions averaged **{spend.value:.2%}** of revenue over {spend.years_used} year(s)"
        + (" — added to capex." if a.acquisitions.on else ", not in capex.")
    )


def _discount_inputs(dataset: dcf_model.CompanyDataset, a: dcf_inputs.Assumptions):
    """The three columns of the discount-rate panel, and the cost-of-debt guidance.

    Returns the panel's container: :func:`_weighted_cost` adds the WACC beneath
    them once the share count is drawn.
    """
    state = st.session_state
    st.markdown("#### Discount rate")
    discount = st.container(border=True)
    with discount:
        left, middle, right = st.columns(3)

        with left:
            st.markdown("**Risk-free rate**")
            for note in a.risk_free.notes:
                _note(note)
            _seed(state, "dcf_input_rf", _pct(a.risk_free.default))
            st.number_input(
                "Risk-free rate (%)", step=0.05, key="dcf_input_rf",
                help="Should match the tenor of the bond behind your cost of debt.",
            )
            common.provenance(a.risk_free.sourced)

            st.markdown("**Equity risk premium**")
            for note in a.erp.notes:
                _note(note)
            _seed(state, "dcf_input_erp", _pct(a.erp.default))
            st.number_input("Equity risk premium (%)", step=0.05, key="dcf_input_erp")
            common.provenance(a.erp.sourced)

        with middle:
            _beta_inputs(a)

        with right:
            _cost_of_debt_inputs(dataset, a)

        with st.expander("Where to find a cost of debt", expanded=False):
            st.markdown(COST_OF_DEBT_GUIDANCE)
            st.link_button(
                "Open FINRA's fixed income data", FINRA_FIXED_INCOME_URL, width="stretch"
            )
    return discount


def _reset_beta() -> None:
    """Puts the regressed beta back. A callback, because the input has drawn by the click."""
    state = st.session_state
    seed = state.get("dcf_input_beta_seed")
    if seed is not None:
        state["dcf_input_beta"] = seed[1]


def _beta_inputs(a: dcf_inputs.Assumptions) -> None:
    st.markdown("**Beta**")
    st.radio(
        "Regression window",
        list(dcf_inputs.BETA_FREQUENCIES),
        horizontal=True,
        key="dcf_input_beta_freq",
        help="5 years of monthly returns by default; 2 years of weekly as the alternative.",
    )
    st.checkbox(
        "Blume adjustment (0.67 × β + 0.33)", value=dcf_model.DEFAULT_BLUME,
        key="dcf_input_blume",
        help="A single stock's regressed beta carries a standard error of 0.2–0.3 "
             "over five years; pulling it toward the market's 1.0 is the usual "
             "correction for that noise.",
    )
    regressed = a.beta_outcome.result
    for note in a.beta.notes:
        if regressed is None:
            _note(note)
    _seed(
        st.session_state, "dcf_input_beta", round(float(a.beta.default or 1.0), 4),
        basis=(a.beta_frequency, a.blume),
    )
    st.number_input("Beta (editable)", step=0.05, key="dcf_input_beta", format="%.4f")
    if regressed is not None:
        col_reset, col_stats = st.columns([1, 2])
        with col_reset:
            st.button(
                "Reset to regressed", key="dcf_beta_reset", width="stretch",
                on_click=_reset_beta,
            )
        with col_stats:
            st.metric("Regressed β", f"{regressed.beta:.4f}", f"R² {regressed.r_squared:.3f}")
        interval = regressed.confidence_interval
        st.caption(
            f"Raw β {regressed.raw_beta:.4f} · {regressed.observations} {a.beta_frequency} "
            f"observations vs {regressed.index_ticker}"
            + (
                f" · standard error {regressed.standard_error:.3f}, 95% interval "
                f"{interval[0]:.2f}–{interval[1]:.2f}"
                if interval is not None
                else ""
            )
        )
        for note in a.beta.notes:
            _note(note)
    common.provenance(a.beta.sourced)


def _cost_of_debt_inputs(dataset: dcf_model.CompanyDataset, a: dcf_inputs.Assumptions) -> None:
    state = st.session_state
    st.markdown("**Cost of debt**")
    for note in a.cost_of_debt.notes:
        _note(note)
    kd_default = a.cost_of_debt.default if a.cost_of_debt.default is not None else a.risk_free.value
    _seed(state, "dcf_input_kd", _pct(kd_default))
    st.number_input(
        "Cost of debt — yield to maturity (%)",
        step=0.05,
        key="dcf_input_kd",
        format="%.4f",
        help="Starts from a synthetic rating; a traded bond's yield is better. See the "
             "sourcing guidance below.",
    )
    common.provenance(a.cost_of_debt.sourced)

    if a.implied_cost_of_debt is not None:
        st.metric("Implied average cost of debt", f"{a.implied_cost_of_debt:.2%}")
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

    _seed(state, "dcf_input_marginal_tax", _pct(a.marginal_tax.default))
    st.number_input(
        "Marginal tax rate (%)",
        step=0.5,
        key="dcf_input_marginal_tax",
        help=(
            "The interest tax shield is worth the *marginal* rate — what the next "
            "dollar of interest saves — not the historical effective rate that feeds "
            "FCFF. They are separate inputs on purpose. The default is the 21% US "
            "federal rate plus a typical state rate."
        ),
    )
    for note in a.marginal_tax.notes:
        _note(note)
    common.provenance(a.marginal_tax.sourced)


def _share_count(dataset: dcf_model.CompanyDataset, a: dcf_inputs.Assumptions) -> None:
    """The share count value per share divides by, and a label saying which one."""
    st.radio(
        "Share count basis for value per share",
        dcf_inputs.SHARE_BASES,
        horizontal=True,
        key="dcf_input_share_basis",
        help=(
            "Point-in-time + dilution: the latest cover-page count plus weighted diluted "
            "less weighted basic shares, a proxy for in-the-money options and RSUs. "
            "Sharadar has no treasury-stock-method table, so the proxy is the best "
            "dilution figure available."
        ),
    )
    for note in a.share_notes:
        _note(note)
    if a.shares is None:
        return
    st.caption(
        f"Using **{a.shares:,.0f}** shares — {a.share_basis}. "
        f"Sharadar provides no treasury-stock-method calculation for outstanding options and "
        f"RSUs, and no if-converted treatment of convertibles, so no basis is fully loaded."
        + (
            f" Counts are per ADR: Sharadar's share factor of {dataset.share_factor:g} is "
            "applied to every one."
            if dataset.share_factor != 1.0
            else ""
        )
    )


def _weighted_cost(dataset: dcf_model.CompanyDataset, a: dcf_inputs.Assumptions, discount) -> None:
    """The WACC the discount inputs weight to, shown at the foot of the discount panel."""
    wacc_result = a.wacc
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
            f"({common.money(a.equity_value, dataset.valuation_currency)}) · D = book value of "
            f"total debt ({common.money(a.debt_value, dataset.valuation_currency)}), as of "
            f"{a.base.stocks_as_of}."
        )


def _terminal_and_conventions(a: dcf_inputs.Assumptions) -> None:
    """Both terminal value inputs, the terminal year, the headline choice, and timing."""
    state = st.session_state
    st.markdown("#### Terminal value and conventions")
    with st.container(border=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            _seed(state, "dcf_input_g", _pct(a.terminal_growth.default))
            st.number_input("Perpetuity growth rate (%)", step=0.1, key="dcf_input_g")
            common.provenance(a.terminal_growth.sourced)

            st.radio(
                "Return on new invested capital (RONIC)",
                a.ronic_bases,
                key="dcf_input_ronic_basis",
                help="The terminal year reinvests g ÷ RONIC of NOPAT to grow at g. Above the "
                     "WACC, growth beyond the forecast creates value; at it, growth neither "
                     "creates nor destroys any — the floor for a company without a durable "
                     "advantage. However high it goes, the Gordon value rises by at most "
                     "WACC ÷ (WACC − g).",
            )
            for note in a.ronic.notes:
                _note(note)
            if a.ronic_basis == _RONIC_MANUAL:
                _seed(state, "dcf_input_ronic", round(float(a.ronic.default) * 100, 2))
                st.number_input(
                    "Return on new invested capital (%)", step=0.5, key="dcf_input_ronic",
                    min_value=0.1,
                )
            common.provenance(a.ronic.sourced)
            text = (
                f"{a.incremental_return:.1%}" if a.incremental_return is not None
                else a.incremental_return_note
            )
            st.caption(
                f"**Forecast incremental return on capital: {text}** — (NOPAT in year "
                f"{a.years} − base NOPAT) ÷ Σ (capex − D&A + ΔNWC). The excess return the "
                f"explicit forecast carries before the terminal year earns its RONIC."
            )

            st.toggle(
                "Terminal tax at the marginal rate",
                value=True,
                key="dcf_input_terminal_tax",
                help="Low effective rates from credits, deferrals or a foreign mix rarely "
                     "last forever. On: the terminal year pays the marginal rate. Off: it "
                     "keeps the forecast's effective rate in perpetuity.",
            )
            st.toggle(
                "Terminal NOPAT before amortisation (EBITA proxy)",
                value=True,
                key="dcf_input_ebita_proxy",
                help="Adds back max(D&A − capex, 0) from the final forecast year, grown at "
                     "g, before tax in year N + 1 — amortisation of acquired intangibles "
                     "that capex never replaces and that runs off. Off: the terminal year "
                     "starts from EBIT, which keeps the charge but loses the add-back the "
                     "forecast years had.",
            )
            st.caption(
                "Moves the Gordon value, and the exit row's implied RONIC and "
                "zero-reinvestment growth, which read the same NOPAT; never the exit value."
            )
        with col_b:
            for note in a.exit_multiple.notes:
                _note(note)
            _seed(state, "dcf_input_exit", float(a.exit_multiple.default))
            st.number_input("Exit EV/EBITDA multiple (×)", step=0.25, key="dcf_input_exit")
            common.provenance(a.exit_multiple.sourced)
            st.caption(
                "The default is the company's *own* recent multiple, earned while it grew "
                "at its historical rate. By year N the forecast has slowed it toward "
                "terminal growth, so a mature multiple — a peer or sector median — is "
                "often the more consistent choice."
            )
        with col_c:
            st.radio(
                "Headline method",
                list(_HEADLINE_METHODS),
                key="dcf_input_headline",
                help="Drives the headline value per share and the equity bridge. Both "
                     "methods always compute and both are shown side by side.",
            )
            st.toggle(
                "Mid-year discounting", value=True, key="dcf_input_midyear",
                help="Mid-year: each year's cash flow is discounted from the middle of the "
                     "year. Applies to the Gordon value, a perpetuity of such flows, but "
                     "not to an exit multiple, which is a sale at the end of year N.",
            )
        _note(a.elapsed_note)


def _cash_sbc_and_bridge(dataset: dcf_model.CompanyDataset, a: dcf_inputs.Assumptions) -> None:
    """Excess cash, stock-based compensation, and non-operating assets."""
    stocks = a.base.stocks
    currency = dataset.valuation_currency
    st.markdown("#### Cash treatment, SBC, and bridge items")
    with st.container(border=True):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.toggle(
                "Treat short-term investments as excess cash",
                value=True,
                key="dcf_input_sti",
                help="Turning this off treats short-term investments as operating assets.",
            )
            st.number_input(
                "Operating cash floor to hold back",
                value=0.0,
                min_value=0.0,
                step=1e9,
                format="%.0f",
                key="dcf_input_cash_floor",
                help=(
                    "Cash the business needs to run, excluded from the bridge. Materially "
                    "moves the answer for cash-rich companies."
                ),
            )
            st.caption(
                f"Balance-sheet cash {common.money(stocks.get('cash'), currency)} · short-term "
                f"investments {common.money(stocks.get('short_term_investments'), currency)} "
                f"(as of {a.base.stocks_as_of})"
            )

        with col_b:
            st.markdown("**Stock-based compensation**")
            for note in a.sbc.notes:
                _note(note)
            _seed(st.session_state, "dcf_input_sbc", float(a.sbc.default or 0.0))
            st.number_input(
                "Stock-based compensation", step=1e8, format="%.0f", key="dcf_input_sbc"
            )
            common.provenance(a.sbc.sourced)
            revenue_base = a.base.flows.get("revenue")
            if a.sbc.value and revenue_base:
                st.caption(f"SBC ÷ revenue: **{a.sbc.value / float(revenue_base):.2%}**")
            if a.sbc.value is not None:
                st.info(
                    "**SBC is expensed, once.** The operating margin is taken from GAAP "
                    "operating income, which already deducts it, so this figure is shown "
                    "for reference and does not move the valuation. Most published FCF "
                    "figures add it back, so this model's FCFF sits below theirs by about "
                    "this amount.",
                    icon="ℹ️",
                )

        with col_c:
            st.markdown("**Non-operating assets**")
            lt_investments = stocks.get("long_term_investments")
            if lt_investments is not None and not pd.isna(lt_investments):
                st.caption(
                    f"Sharadar reports **{common.money(lt_investments, currency)}** of "
                    f"non-current investments (as of {a.base.stocks_as_of}), at book value. It "
                    "mixes marketable securities with equity-method stakes and restricted "
                    "balances, so it is not added unless you choose to."
                )
            else:
                st.caption(
                    "Sharadar reports no non-current investments line for this company."
                )
            st.number_input(
                "Non-operating assets to add",
                value=0.0,
                step=1e8,
                format="%.0f",
                key="dcf_input_nonop",
                help="Long-term and equity-method investments that the operating forecast "
                     "does not already capture. Opt-in: the reported figure is at book and "
                     "mixes operating and non-operating holdings.",
            )


def _render_fcff_panel(dataset: dcf_model.CompanyDataset, a: dcf_inputs.Assumptions) -> None:
    base_fcff = a.base_fcff
    currency = dataset.valuation_currency
    st.markdown("#### Base-period free cash flow, computed two independent ways")
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            st.markdown("**Method A — cash flow statement build**")
            st.code("FCFF = CFO + Interest Expense × (1 − tax rate) − CapEx")
            st.metric("FCFF (method A)", common.money(base_fcff.method_a, currency))
        with right:
            st.markdown("**Method B — income statement build**")
            st.code("NOPAT = EBIT × (1 − tax rate)\nFCFF  = NOPAT + D&A − CapEx − ΔNWC")
            st.metric("FCFF (method B)", common.money(base_fcff.method_b, currency))

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
                        f"{value:,.4f}" if name == "Tax rate" else common.money(value, currency)
                    ) if value is not None else "not reported",
                }
                for name, value in base_fcff.inputs.items()
            ]
        )
        st.dataframe(inputs_frame, width="stretch", hide_index=True)

        if a.etr is not None:
            st.caption(
                f"Effective tax rate in use: **{a.etr:.2%}** — median of the last "
                f"{a.etr_years} profitable year(s). Years with a pretax loss are "
                f"excluded because the ratio is meaningless there, not set to zero."
            )
        else:
            st.warning(
                "No profitable year is available to compute an effective tax rate, so the "
                "forecast tax rate driver is used for the base period as well."
            )

        if a.delta_nwc_span:
            st.caption(f"Change in net working capital measured over — {a.delta_nwc_span}.")

        st.caption(
            "Method A assumes interest paid sits in operating cash flow, which holds under "
            "US GAAP; an IFRS filer may classify it in financing. Sharadar exposes interest "
            "*expense* (accrual), not interest *paid* (cash), so that is what is used. "
            "The two builds legitimately differ where working capital moves a lot: method B "
            "subtracts the balance-sheet movement, while method A picks up whatever the cash "
            "flow statement booked, including acquisitions and FX translation."
        )
