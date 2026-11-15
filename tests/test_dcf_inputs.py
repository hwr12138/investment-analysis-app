"""The DCF's assumption set, defaults plus overrides, as the panel and the chart assemble it.

``app.logic.dcf.inputs.resolve`` is what the web API's panel and the summary
chart's DCF line both call, so these tests pin the rules the two must agree on:
an untouched input is its default exactly, a default follows the choices it
depends on, an override is marked as the reader's, and a choice the panel does
not offer is refused rather than quietly replaced. The second half covers what
the tab draws from a run: the reverse-DCF verdicts, the cross-check between the
two terminal values, and the capex and RONIC targets.
"""

from __future__ import annotations

import inspect
from datetime import date

import pandas as pd
import pytest

from app.logic.dcf import financials as fin
from app.logic.dcf import inputs as dcf_inputs
from app.logic.dcf import model as dcf_model
from app.logic.dcf import reverse
from app.logic.dcf import valuation as dcf
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


def test_every_run_model_argument_comes_from_one_place(dataset):
    """The forward run, the reverse solve and the what-if price share
    ``run_kwargs``, so the reverse DCF cannot invert a different model. Every
    keyword ``run_model`` takes beyond the dataset, base and drivers has to be
    in it, and the resolved set has to supply every one."""
    parameters = set(inspect.signature(dcf_model.run_model).parameters)
    supplied = set(dcf_inputs.run_kwargs(_resolve(dataset).inputs))
    assert supplied == parameters - {"dataset", "base", "drivers"}


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
    """The panel displays a default rounded; reading it back is untouched."""
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
# Reverse DCF verdicts
# ---------------------------------------------------------------------------


def _reverse(status, **overrides):
    fields = dict(status=status, target=100.0, growth=None)
    fields.update(overrides)
    return reverse.ReverseDcfResult(**fields)


def test_every_reverse_status_has_a_verdict():
    """Adding a status without writing its message has to fail here."""
    for status in reverse.Status:
        result = _reverse(
            status,
            growth=0.12 if status is reverse.Status.SOLVED else None,
            local_slope=5.0 if status is reverse.Status.SOLVED else None,
            reason="measured detail",
        )
        level, message = dcf_inputs.reverse_verdict(result)
        assert level in {"", "info", "warning", "error"}
        if status is not reverse.Status.SOLVED:
            assert message, f"{status} renders nothing"


def test_an_ordinary_solved_rate_needs_no_banner():
    result = _reverse(reverse.Status.SOLVED, growth=0.12, local_slope=5.0)
    assert dcf_inputs.reverse_verdict(result) == ("", "")


def test_a_value_destroying_solve_says_the_reading_is_inverted():
    result = _reverse(reverse.Status.SOLVED, growth=0.12, local_slope=-5.0)
    level, message = dcf_inputs.reverse_verdict(result)
    assert level == "warning"
    assert "destroys value" in message
    assert "inverted" in message


def test_an_extreme_solved_rate_is_flagged_rather_than_presented_as_a_forecast():
    result = _reverse(reverse.Status.SOLVED, growth=0.90, local_slope=5.0)
    level, message = dcf_inputs.reverse_verdict(result)
    assert level == "warning"
    assert "not explained by revenue growth" in message


def test_the_precondition_message_says_it_is_not_a_failed_search():
    result = _reverse(reverse.Status.NOT_EVALUABLE, reason="Terminal growth is not below WACC.")
    level, message = dcf_inputs.reverse_verdict(result)
    assert level == "error"
    assert "not a failed search" in message


def test_an_unwritten_status_is_a_loud_failure_not_a_blank_panel():
    with pytest.raises(AssertionError):
        dcf_inputs.reverse_verdict(_reverse("invented_status"))


def test_an_implied_revenue_path_compounds_from_the_base():
    path = dcf_inputs.implied_revenue_path(1000.0, 0.10, 3)
    assert path == pytest.approx([1100.0, 1210.0, 1331.0])


def test_an_implied_revenue_path_follows_the_forecast_fade():
    path = dcf_inputs.implied_revenue_path(1000.0, 0.20, 3, fade_to=0.02)
    assert path == pytest.approx([1200.0, 1200.0 * 1.14, 1200.0 * 1.14 * 1.08])


def test_an_implied_revenue_path_holds_before_it_fades():
    path = dcf_inputs.implied_revenue_path(1000.0, 0.20, 3, fade_to=0.02, hold=2)
    assert path == pytest.approx([1200.0, 1440.0, 1440.0 * 1.11])


def test_the_reverse_dcf_reads_gordon_first_as_the_fundamentals():
    """Both methods are solved, Gordon leading."""
    assert dcf_inputs.REVERSE_METHODS == ("gordon", "exit_multiple")
    assert "fundamentals" in dcf_inputs.REVERSE_LABELS["gordon"]


# ---------------------------------------------------------------------------
# The terminal value cross-check and the headline method
# ---------------------------------------------------------------------------


def test_gordon_row_is_restated_as_an_exit_multiple():
    terminal = dcf.TerminalValue(
        method="Gordon growth", value=1000.0, present_value=800.0,
        discount_exponent=3.5, formula="", implied_exit_multiple=13.6,
    )
    assert dcf_inputs.cross_implied(terminal) == "an exit multiple of 13.60×"


def test_exit_multiple_row_is_restated_as_growth_and_a_return_on_capital():
    terminal = dcf.TerminalValue(
        method="Exit multiple", value=1000.0, present_value=800.0,
        discount_exponent=3.5, formula="", implied_growth=0.0511, implied_ronic=0.184,
    )
    assert dcf_inputs.cross_implied(terminal) == (
        "perpetual growth of at least 5.11%; RONIC of 18.4% at the terminal rate"
    )


def test_cross_check_is_blank_when_neither_is_computable():
    terminal = dcf.TerminalValue(
        method="Gordon growth", value=1000.0, present_value=800.0,
        discount_exponent=3.5, formula="",
    )
    assert dcf_inputs.cross_implied(terminal) == "—"


def test_gordon_growth_is_the_default_headline_method():
    """The owner's choice: the headline rests on the company's own cash flows
    rather than on the multiple it traded at. The exit multiple is still
    computed and shown beside it as the cross-check."""
    assert dcf_model.DEFAULT_HEADLINE_METHOD == "gordon"


def test_the_panel_offers_the_default_headline_first_and_both_methods():
    assert list(dcf_inputs.HEADLINE_METHODS)[0] == dcf_model.DEFAULT_HEADLINE_METHOD
    assert set(dcf_inputs.HEADLINE_METHODS) == {"exit_multiple", "gordon"}


# ---------------------------------------------------------------------------
# Capex normalisation — resolve_target
# ---------------------------------------------------------------------------


def _capex_target(normal_value, *, on=None, value=None):
    periods = ("2022", "2023", "2024", "2025", "2026")
    normal = fin.DriverDefault(
        "capex_pct_revenue", normal_value, fin.NORMAL_CAPEX_BASIS,
        len(periods) if normal_value is not None else 0,
        periods if normal_value is not None else (),
        "" if normal_value is not None else "No period had every component needed.",
    )
    return dcf_inputs.resolve_target(
        "capex_pct_revenue", "capex % of revenue", 0.35, 10, normal=normal, on=on, value=value
    )


def test_capex_moves_to_its_five_year_average_by_default():
    """Today's capex is where the forecast starts, not where it stays: the
    target is on without anyone touching it, and it is derived, not typed."""
    target = _capex_target(0.20)
    assert target.default_on is True and target.on is True
    assert target.input.value == pytest.approx(0.20)
    assert target.input.sourced.origin.value == "derived"


def test_a_typed_capex_target_reads_as_the_users():
    target = _capex_target(0.20, value=0.15)
    assert target.input.value == pytest.approx(0.15)
    assert target.input.sourced.origin.value == "user"


def test_a_target_typed_back_to_its_default_is_the_default_again():
    """The panel shows the default rounded; reading it back is no override."""
    target = _capex_target(0.20, value=round(0.20 * 100, 4) / 100)
    assert target.input.sourced.origin.value == "derived"


def test_turning_normalisation_off_holds_todays_capex():
    target = _capex_target(0.20, on=False)
    assert target.on is False and target.input is None


def test_no_history_leaves_capex_flat_and_says_why():
    """No silent fallback: with nothing to normalise to, nothing is substituted."""
    target = _capex_target(None)
    assert target.default_on is False and target.on is False
    assert any("No 5-year average to normalise to" in n.text for n in target.notes)


# ---------------------------------------------------------------------------
# Return on new invested capital — resolve_ronic
# ---------------------------------------------------------------------------


def _ronic(ebit, basis=None, value=None):
    """The RONIC for a company whose ROIC is ebit × 0.8 ÷ 100 a year, at a 9% WACC."""
    years = 6
    annual = pd.DataFrame(
        {
            "ebit": [ebit] * years, "pretax_income": [ebit] * years,
            "tax_expense": [-0.2 * ebit] * years, "total_equity": [100.0] * years,
        },
        index=list(range(2020, 2020 + years)),
    )
    return dcf_inputs.resolve_ronic(annual, 0.09, basis=basis, value=value)


def test_the_default_ronic_is_the_companys_own_return_above_the_wacc():
    """RONIC = WACC credits nothing to growth beyond year N; the company's own
    record is the default evidence for what its new capital earns."""
    options, basis, ronic = _ronic(ebit=30.0)            # 30 × 0.8 ÷ 100 = 24%
    assert basis == dcf_inputs.OWN_ROIC == options[0]
    assert ronic.value == pytest.approx(0.24)
    assert ronic.sourced.origin.value == "derived"


def test_an_own_return_below_the_wacc_defaults_to_the_wacc_not_to_destroying_value():
    _, _, ronic = _ronic(ebit=5.0)                       # 4%, against a 9% WACC
    assert ronic.value is None
    assert ronic.sourced.origin.value == "estimate"
    assert any("does not exceed its WACC" in n.text for n in ronic.notes)


def test_no_measurable_return_offers_only_the_wacc_and_a_manual_figure():
    options, _, ronic = _ronic(ebit=-10.0)
    assert list(options) == [dcf_inputs.RONIC_AT_WACC, dcf_inputs.RONIC_MANUAL]
    assert ronic.value is None
    assert any("No own ROIC to default to" in n.text for n in ronic.notes)


def test_ronic_can_still_be_set_to_the_wacc_or_by_hand():
    _, _, at_wacc = _ronic(ebit=30.0, basis=dcf_inputs.RONIC_AT_WACC)
    assert at_wacc.value is None
    _, _, manual = _ronic(ebit=30.0, basis=dcf_inputs.RONIC_MANUAL)
    assert manual.value == pytest.approx(0.24)       # seeded from the company's own
    _, _, typed = _ronic(ebit=30.0, basis=dcf_inputs.RONIC_MANUAL, value=0.12)
    assert typed.value == pytest.approx(0.12) and typed.sourced.origin.value == "user"


# ---------------------------------------------------------------------------
# The growth basis the checks read
# ---------------------------------------------------------------------------


def test_the_effective_growth_basis_names_the_rate_actually_used():
    """The TTM-window acquisitions check reads it: it applies only when the
    trailing rate is the one in use."""
    ttm = fin.DriverDefault("revenue_growth", 0.83, fin.TTM_GROWTH_BASIS, 1, ("P",))
    cagr = fin.DriverDefault("revenue_growth", 0.67, "5-year CAGR", 5, ("P",))
    slow = fin.DriverDefault("revenue_growth", 0.04, fin.TTM_GROWTH_BASIS, 1, ("P",))
    burst = fin.lower_of_growth(ttm, cagr)
    slowdown = fin.lower_of_growth(slow, cagr)

    assert dcf_inputs.effective_growth_basis(fin.LOWER_GROWTH_BASIS, burst, ttm) == "5-year CAGR"
    assert dcf_inputs.effective_growth_basis(
        fin.LOWER_GROWTH_BASIS, slowdown, slow
    ) == fin.TTM_GROWTH_BASIS
    assert dcf_inputs.effective_growth_basis(
        fin.TTM_GROWTH_BASIS, burst, ttm
    ) == fin.TTM_GROWTH_BASIS
