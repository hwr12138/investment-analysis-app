"""The DCF's assumption set, defaults plus overrides, as both front ends assemble it.

``app.logic.dcf.inputs.resolve`` is what the Streamlit panel and the web API both
call, so these tests pin the rules the two must agree on: an untouched input is
its default exactly, a default follows the choices it depends on, an override is
marked as the reader's, and a choice the panel does not offer is refused rather
than quietly replaced. The second half drives the real Streamlit tab, to check
its widgets reach the same function.
"""

from __future__ import annotations

from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

from app.data import market_data
from app.logic.dcf import inputs as dcf_inputs
from app.logic.dcf import model as dcf_model
from app.logic.dcf import wacc as wacc_lib
from app.provenance import Sourced

from .test_golden import build_model

_BETA = dcf_model.BetaOutcome(
    wacc_lib.BetaResult(
        beta=1.1, raw_beta=1.15, r_squared=0.4, observations=60, frequency="monthly",
        blume_adjusted=True, index_ticker="SPY",
    ),
    "stub", None,
)
_RF = Sourced.reported(0.042, "stub")
_ERP = Sourced.reported(0.045, "stub")
_TODAY = date(2026, 11, 15)


@pytest.fixture
def dataset(statements_payload, tickers_response, monkeypatch):
    return build_model(statements_payload, tickers_response, monkeypatch)["dataset"]


def _resolve(dataset, **overrides) -> dcf_inputs.Assumptions:
    return dcf_inputs.resolve(
        dataset, valuation_date=_TODAY, risk_free=_RF, erp=_ERP, beta=_BETA,
        overrides=dcf_inputs.Overrides(**overrides),
    )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_an_untouched_panel_is_the_default_value(dataset):
    """The summary chart's DCF line and the tab are one number."""
    resolved = _resolve(dataset)
    evaluation = dcf_inputs.evaluate(dataset, resolved.inputs)
    line = dcf_inputs.default_value(
        dataset, valuation_date=_TODAY, risk_free=_RF, erp=_ERP, beta=_BETA
    )
    assert evaluation.result.headline_bridge.value_per_share == pytest.approx(
        line.headline_bridge.value_per_share, rel=1e-12
    )


def test_untouched_inputs_carry_their_own_provenance_not_the_users(dataset):
    resolved = _resolve(dataset)
    origins = {
        "risk-free": resolved.risk_free.sourced.origin.value,
        "beta": resolved.beta.sourced.origin.value,
        "terminal growth": resolved.terminal_growth.sourced.origin.value,
        "marginal tax": resolved.marginal_tax.sourced.origin.value,
    }
    assert origins == {
        "risk-free": "reported", "beta": "derived",
        "terminal growth": "estimate", "marginal tax": "estimate",
    }
    assert not any(d.input.overridden for d in resolved.drivers_in)


def test_every_default_matches_the_engines_constants(dataset):
    resolved = _resolve(dataset)
    assert resolved.years == dcf_model.DEFAULT_FORECAST_YEARS
    assert resolved.hold == dcf_model.DEFAULT_GROWTH_HOLD
    assert resolved.headline_method == dcf_model.DEFAULT_HEADLINE_METHOD
    assert resolved.terminal_growth.value == dcf_model.DEFAULT_TERMINAL_GROWTH
    assert resolved.marginal_tax.value == dcf_model.DEFAULT_MARGINAL_TAX
    assert resolved.beta_frequency == dcf_model.DEFAULT_BETA_FREQUENCY
    assert resolved.blume is dcf_model.DEFAULT_BLUME
    assert resolved.inputs["terminal_tax_rate"] == dcf_model.DEFAULT_MARGINAL_TAX


def test_the_snapshot_carries_everything_the_tab_fingerprints(dataset):
    """The Streamlit tab fingerprints this dict and the API values it, so a key
    the shell reads and resolve does not write would fail on both."""
    from app.views.research import dcf as page

    inputs = _resolve(dataset).inputs
    assert page._assumption_fingerprint(inputs)
    assert dcf_inputs.run_kwargs(inputs)


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


def test_an_override_is_the_readers_and_reaches_the_run(dataset):
    resolved = _resolve(dataset, terminal_growth=0.03)
    assert resolved.terminal_growth.value == 0.03
    assert resolved.terminal_growth.sourced.origin.value == "user"
    assert resolved.inputs["terminal_growth"] == 0.03
    assert resolved.inputs["growth_fade_to"] == 0.03     # the fade lands on it


def test_an_override_equal_to_the_rounded_default_is_no_override(dataset):
    """Both front ends display a default rounded; reading it back is untouched."""
    default = _resolve(dataset).driver("operating_margin").input.default
    resolved = _resolve(dataset, values={"operating_margin": round(default * 100, 4) / 100})
    assert resolved.driver("operating_margin").input.sourced.origin.value == "derived"
    assert resolved.driver("operating_margin").input.value == default


def test_a_driver_override_marks_the_margin_as_assumed(dataset):
    resolved = _resolve(dataset, values={"operating_margin": 0.40})
    assert resolved.driver("operating_margin").input.overridden
    assert resolved.inputs["assumed_margins"]["Starting margin"] == 0.40
    assert set(resolved.drivers.operating_margin) == {0.40}


def test_a_driver_basis_brings_its_own_default(dataset):
    resolved = _resolve(dataset)
    growth = resolved.driver("revenue_growth")
    other = next(b for b in growth.bases if b != growth.basis)
    moved = _resolve(dataset, bases={"revenue_growth": other}).driver("revenue_growth")
    assert moved.basis == other
    assert moved.input.sourced.source.startswith(other)


def test_an_untouched_cost_of_debt_follows_the_risk_free_rate(dataset):
    """A synthetic rating is a spread over the risk-free rate, so it moves with it."""
    before = _resolve(dataset)
    after = _resolve(dataset, risk_free=0.052)
    assert after.cost_of_debt.value == pytest.approx(before.cost_of_debt.value + 0.010)
    assert after.cost_of_debt.sourced.origin is before.cost_of_debt.sourced.origin


def test_a_typed_cost_of_debt_retires_the_spread_table(dataset):
    resolved = _resolve(dataset, cost_of_debt=0.061)
    assert resolved.cost_of_debt.sourced.origin.value == "user"
    assert resolved.inputs["spread_table_as_of"] is None


def test_turning_the_fade_off_holds_growth_flat(dataset):
    resolved = _resolve(dataset, fade=False)
    assert resolved.inputs["growth_fade_to"] is None
    assert len(set(resolved.drivers.revenue_growth)) == 1


def test_a_shorter_forecast_shapes_every_path(dataset):
    resolved = _resolve(dataset, years=6, hold=2)
    assert resolved.drivers.years == 6
    assert resolved.inputs["growth_hold"] == 2


def test_the_exit_headline_and_conventions_reach_the_run(dataset):
    resolved = _resolve(
        dataset, headline_method="exit_multiple", mid_year=False, ebita_proxy=False,
        terminal_at_marginal=False, include_sti=False, cash_floor=1e6,
    )
    kwargs = dcf_inputs.run_kwargs(resolved.inputs)
    assert kwargs["headline_method"] == "exit_multiple"
    assert kwargs["mid_year"] is False and kwargs["ebita_proxy"] is False
    assert kwargs["terminal_tax_rate"] is None
    assert kwargs["include_short_term_investments"] is False
    assert kwargs["operating_cash_floor"] == 1e6


@pytest.mark.parametrize(
    "overrides",
    [
        {"years": 4},
        {"years": 20},
        {"hold": 10},
        {"bases": {"revenue_growth": "a basis nobody offers"}},
        {"share_basis": "Shares I made up"},
        {"headline_method": "dividend discount"},
        {"beta_frequency": "daily"},
        {"cash_floor": -1.0},
    ],
)
def test_a_choice_the_panel_does_not_offer_is_refused(dataset, overrides):
    """Never quietly replaced by the default — the API turns this into a 422."""
    with pytest.raises(dcf_inputs.InvalidOverride):
        _resolve(dataset, **overrides)


# ---------------------------------------------------------------------------
# Placeholders: the panel asks, the chart's line leaves a gap
# ---------------------------------------------------------------------------


def test_a_missing_risk_free_rate_is_asked_for_not_substituted(dataset):
    resolved = dcf_inputs.resolve(
        dataset, valuation_date=_TODAY, risk_free=Sourced.unavailable("stub", "down"),
        erp=_ERP, beta=_BETA,
    )
    assert resolved.risk_free.default is None
    assert resolved.risk_free.notes[0].level == "warning"
    with pytest.raises(dcf_model.DefaultUnavailable):
        dcf_inputs.resolve(
            dataset, valuation_date=_TODAY, risk_free=Sourced.unavailable("stub", "down"),
            erp=_ERP, beta=_BETA, strict=True,
        )


def test_a_wacc_that_cannot_be_weighted_blocks_the_run_not_the_panel(dataset, monkeypatch):
    """The panel still resolves, so the reader can undo whatever caused it."""
    def fail(**_):
        raise ValueError("Total capital is zero.")

    monkeypatch.setattr(dcf_inputs.wacc_lib, "compute_wacc", fail)
    resolved = _resolve(dataset)
    assert resolved.inputs is None
    assert resolved.blocked.title == "The WACC cannot be weighted"
    assert resolved.terminal_growth.value == dcf_model.DEFAULT_TERMINAL_GROWTH


# ---------------------------------------------------------------------------
# The valued run
# ---------------------------------------------------------------------------


def test_evaluate_solves_the_reverse_dcf_both_ways(dataset):
    evaluation = dcf_inputs.evaluate(dataset, _resolve(dataset).inputs)
    assert tuple(evaluation.reverse) == dcf_inputs.REVERSE_METHODS
    assert evaluation.sensitivity.method == "gordon"
    assert evaluation.check_results
    assert not evaluation.blocking


def test_a_what_if_price_inverts_the_model_on_screen(dataset):
    """Solving at the value the run produced gives back the run's own growth."""
    inputs = _resolve(dataset, fade=False).inputs
    evaluation = dcf_inputs.evaluate(dataset, inputs)
    value = evaluation.result.headline_bridge.value_per_share
    solved = dcf_inputs.solve_price(dataset, inputs, value, "gordon")
    assert solved.growth == pytest.approx(inputs["drivers"].revenue_growth[0], abs=1e-6)


# ---------------------------------------------------------------------------
# The Streamlit panel reaches the same function
# ---------------------------------------------------------------------------


def _dcf_tab():
    from app.views.research.dcf import render

    render("TEST")


@pytest.fixture
def tab(statements_payload, tickers_response, monkeypatch):
    build_model(statements_payload, tickers_response, monkeypatch)
    monkeypatch.setattr(dcf_model, "regress_company_beta", lambda *a, **k: _BETA)
    monkeypatch.setattr(market_data, "fetch_10y_treasury_rate", lambda: _RF)
    monkeypatch.setattr(market_data, "fetch_erp", lambda: _ERP)
    app = AppTest.from_function(_dcf_tab, default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    return app


def _inputs(app) -> dict:
    return app.session_state["dcf_output"]["inputs"]


def test_a_typed_input_reaches_the_run_as_the_readers(tab):
    tab.number_input(key="dcf_input_g").set_value(3.0).run()
    assert not tab.exception, tab.exception
    assert _inputs(tab)["terminal_growth"] == pytest.approx(0.03)
    growth = next(r for r in _inputs(tab)["records"] if r.name == "Perpetuity growth")
    assert growth.sourced.origin.value == "user"


def test_a_cost_of_debt_nobody_typed_follows_a_typed_risk_free_rate(tab):
    before = tab.number_input(key="dcf_input_kd").value
    tab.number_input(key="dcf_input_rf").set_value(5.2).run()
    assert not tab.exception, tab.exception
    # The fixture discloses no interest, so its placeholder is the risk-free rate itself.
    assert tab.number_input(key="dcf_input_kd").value == pytest.approx(before + 1.0)


def test_a_new_basis_reseeds_the_value_on_screen(tab):
    box = tab.selectbox(key="dcf_input_growth_basis")
    other = next(o for o in box.options if o != box.value)
    tab.number_input(key="dcf_input_growth_value").set_value(42.0).run()
    assert _inputs(tab)["drivers"].revenue_growth[0] == pytest.approx(0.42)
    tab.selectbox(key="dcf_input_growth_basis").set_value(other).run()
    assert not tab.exception, tab.exception
    assert tab.number_input(key="dcf_input_growth_value").value != pytest.approx(42.0)


def test_reset_to_regressed_puts_the_beta_back(tab):
    tab.number_input(key="dcf_input_beta").set_value(1.7).run()
    assert _inputs(tab)["beta"] == pytest.approx(1.7)
    tab.button(key="dcf_beta_reset").click().run()
    assert not tab.exception, tab.exception
    assert _inputs(tab)["beta"] == pytest.approx(_BETA.result.beta)


def test_shortening_the_forecast_pulls_the_hold_in(tab):
    tab.number_input(key="dcf_input_hold").set_value(9).run()
    tab.slider(key="dcf_input_years").set_value(5).run()
    assert not tab.exception, tab.exception
    assert _inputs(tab)["growth_hold"] == 4
