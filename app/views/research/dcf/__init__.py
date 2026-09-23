"""The 💰 DCF tab — UI only.

The one tab that loads its own data. :func:`render` takes the page's ticker
rather than the snapshot, because none of what it shows comes from there: the
model needs statements normalised and schema-resolved by ``app.logic.dcf``,
which the research fetch does not do. It loads and values that ticker as soon as
the tab body renders — see :func:`render` for what that costs.

This module is the tab's shell: the load, the two caches, the session
lifecycle, and the order the page draws in. Each part of the page lives in its
own module beside it — ``assumptions`` (every input), ``checks`` (banners and
the validation panel), ``history``, ``results`` (header, projection,
valuation), ``implied`` (the reverse DCF) and ``download`` — with what several
of them draw in ``common``. Those names avoid ``valuation``, ``reverse`` and
``export`` on purpose: each is already a module in ``app.logic.dcf``.

Every number rendered here carries a provenance badge and, where it was
computed, the formula behind it. The severity banners sit above everything and
do not collapse: the governing rule for this tab is that it must be impossible
to look at the output and not notice something is wrong.
"""

from __future__ import annotations

from datetime import date

import streamlit as st

from app.data import sharadar_client
from app.logic.dcf import financials as fin
from app.logic.dcf import model as dcf_model
from app.logic.dcf import validation as val
from app.logic.dcf import valuation as dcf
from app.logic.dcf.schema import SchemaResolutionError
from app.views.research.dcf import (
    assumptions,
    checks,
    common,
    download,
    history,
    implied,
    results,
)

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


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _load_company_cached(ticker: str) -> dcf_model.CompanyDataset:
    """Caches the whole normalised dataset, not just the HTTP calls.

    The fetchers underneath are cached individually, but the schema resolution
    and statement normalisation are not free either. Caching here means moving a
    slider re-runs the arithmetic and nothing else — no refetch, no re-parse.
    It is also what keeps the ungated load in :func:`render` to once per company
    rather than once per rerun.

    Always the restated (``MR``) series. ``dcf_model.load_company`` still takes
    ``asreported``, and the export and provenance captions still read the flag
    off the dataset, so a caller that wants the as-filed series can have it —
    this page does not offer the choice, because a consistent time series is
    what the ratio derivation needs and mixing the two within one run is never
    allowed anyway.
    """
    return dcf_model.load_company(ticker)


def _assumption_fingerprint(inputs: dict) -> tuple:
    """A hashable snapshot of every input the valuation depends on.

    The key the results are memoised under: a rerun whose fingerprint matches
    reuses the stored run, and any change to it recomputes. Streamlit reruns the
    whole script on every widget interaction, including the display-only ones —
    the units selector, the statement picker — and the reverse solve underneath
    is some fifty-odd model evaluations, so this is what stops the model, the
    checks and the workbook being rebuilt when nothing that feeds them moved.

    A field missing from this tuple is therefore a correctness bug, not a
    performance one: the page would keep showing the previous run's numbers as
    if they were current.

    The reverse DCF adds nothing here. It depends only on assumptions already
    listed plus the market price, and the price reaches this tuple already: it
    is what the market value of equity is computed from, so it moves the WACC,
    which is field seven. (If the point-in-time share count is missing, equity
    value collapses to zero and that link breaks — but the price still cannot
    change mid-session, because it arrives with the cached dataset and a change
    of company forces a rerun outright.)
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


def _needs_run(output: dict | None, ticker: str, fingerprint: tuple) -> bool:
    """True when the stored run cannot serve what the page currently shows.

    With no Compute gate this is the whole of the recompute decision, and it is
    a correctness one rather than a performance one: a run is either reused
    because it matches the assumptions on screen exactly, or it is rebuilt
    before anything renders. Nothing on this tab is ever drawn from a run that
    does not match, because there is no staleness notice to fall back on.

    The company check is belt-and-braces — :func:`_session_reset` has already
    dropped the stored run if the ticker moved — but it is what makes it
    impossible for one company's numbers to appear under another's name.
    """
    return (
        output is None
        or output["company"] != ticker
        or output["fingerprint"] != fingerprint
    )


def _session_reset(ticker: str) -> None:
    """Clears per-company state. Acknowledgements never carry across companies.

    Called from :func:`render` whenever the page's ticker no longer matches what
    this tab last loaded, which is the only thing that invalidates a driver
    typed by hand or a check ticked off: both are statements about one company.

    ``dcf_reverse_price`` goes with them — a price per share typed for one
    company would otherwise be solved against the next one's model and read as
    an answer about it.
    """
    for key in list(st.session_state.keys()):
        if key.startswith("dcf_input_") or key in {
            "dcf_acks", "dcf_beta_override", "dcf_output", "dcf_reverse_price"
        }:
            del st.session_state[key]
    st.session_state["dcf_acks"] = {}
    st.session_state["dcf_loaded_ticker"] = ticker



# ---------------------------------------------------------------------------
# Tab
# ---------------------------------------------------------------------------


def render(ticker: str) -> None:
    """The whole model, for the company the research page currently has loaded.

    ``ticker`` comes from that page, which is why there is no ticker box here,
    and the model loads and values it as soon as this body renders. There is no
    Compute gate, so what is on screen always describes the company the rest of
    the page is showing and the assumptions currently set — an assumption change
    is applied on the spot.

    What that costs is one extra set of Sharadar requests per company searched,
    whether or not the reader ever opens this tab: ``st.tabs`` renders every tab
    body on every rerun and reports nothing to Python about which one is open.
    Two caches keep it to exactly that — :func:`_load_company_cached` for the
    fetch and normalisation, :func:`_needs_run` for the arithmetic — so a rerun
    triggered by any other control on the page costs nothing here.
    """
    st.markdown(_BANNER_CSS, unsafe_allow_html=True)
    st.subheader("DCF valuation")
    st.write(
        "Unlevered free cash flow to the firm, discounted to an intrinsic value per share. "
        "Fundamentals and prices from Nasdaq Data Link (Sharadar); Treasury yields and the "
        "equity risk premium from the shared market-data layer."
    )

    # Following the page to a different company clears the previous one's
    # acknowledgements and assumption overrides; a rerun on the same one keeps
    # them, which is what makes the assumption panel usable at all.
    if st.session_state.get("dcf_loaded_ticker") != ticker:
        _session_reset(ticker)
    acknowledged: dict[str, str] = st.session_state.setdefault("dcf_acks", {})

    # --- Load ------------------------------------------------------------
    try:
        with st.spinner(f"Loading {ticker} from Sharadar…"):
            dataset = _load_company_cached(ticker)
    except val.UnsupportedSectorError as error:
        common.render_blocking("Unsupported sector", str(error))
        return
    except SchemaResolutionError as error:
        common.render_blocking(
            "Could not resolve a required financial concept",
            str(error),
            f"Columns Sharadar did return on the {error.statement} statement: "
            f"{', '.join(error.available[:40])}",
        )
        return
    except sharadar_client.TickerNotFoundError as error:
        common.render_blocking(
            "Ticker not found",
            f"{error.message} Check the symbol — this page covers US-listed equities "
            "that Sharadar's fundamentals database carries.",
        )
        return
    except sharadar_client.SharadarError as error:
        common.render_blocking("Could not load data from Nasdaq Data Link", str(error))
        return

    if dataset.annual.empty:
        common.render_blocking(
            "No usable statements",
            f"Sharadar returned no annual fundamentals for {ticker}, so there is no "
            "history to build a forecast on.",
        )
        return

    valuation_date = date.today()

    # --- Company header --------------------------------------------------
    results.render_header(dataset, valuation_date)

    # These slots are written to at the end of the run but occupy the position
    # they were created in, which is how the banners end up directly under the
    # header and above every number on the page. Page order follows the spec:
    # banners, validation panel, historicals, assumptions, projection,
    # valuation, export — the checks come before the answer on purpose.
    banner_slot = st.container()
    validation_slot = st.container()

    history.render(dataset)

    inputs = assumptions.collect(dataset, valuation_date)
    if inputs is None:
        return

    fingerprint = _assumption_fingerprint(inputs)
    output = st.session_state.get("dcf_output")

    if _needs_run(output, ticker, fingerprint):
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
            common.render_blocking("The projection cannot be built", str(error))
            return

        # Solved here rather than at render time so it describes the same run as
        # everything else on the page. Reading live widget values instead would
        # let the implied rate invert a valuation that is not the one displayed.
        # No try/except: every way this can fail is a status on the result.
        reverse_result = dcf_model.solve_implied_growth(
            dataset,
            inputs["base"],
            target_price=dataset.last_price,
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
            periods_in_use=inputs["periods_in_use"],
        )
        issues, check_results = val.run_all(context, inputs["tolerances"])

        output = {
            "company": ticker,
            "fingerprint": fingerprint,
            "result": result,
            "reverse": reverse_result,
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
    # `.get` rather than `[...]`: a session dict snapshotted before this section
    # existed has no reverse result, and a stale session should not crash.
    reverse_result = output.get("reverse")

    blocking = [i for i in issues if i.severity is val.Severity.BLOCKING]
    if blocking:
        for issue in blocking:
            common.render_blocking(issue.check, issue.message, "; ".join(
                f"{k}: {v}" for k, v in issue.values.items()
            ))
        return

    with banner_slot:
        checks.render_banners(issues, acknowledged)

    with validation_slot:
        checks.render_validation_panel(dataset, check_results, computed_inputs["tolerances"])

    unacknowledged_critical = [
        i for i in issues if i.severity is val.Severity.CRITICAL and i.key not in acknowledged
    ]

    results.render_projection(dataset, result)
    results.render_valuation(dataset, result, computed_inputs, unacknowledged_critical)
    if reverse_result is not None:
        implied.render(
            dataset,
            result,
            computed_inputs,
            reverse_result,
            fin.revenue_growth_history(dataset.annual),
        )
    download.render(
        dataset, result, computed_inputs, issues, check_results, acknowledged, valuation_date
    )
