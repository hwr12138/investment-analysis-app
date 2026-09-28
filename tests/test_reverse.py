"""The reverse solver, against functions whose roots are known by hand.

The point of the injected callable is right here: a straight line and a parabola
have roots that can be written down, so these tests check the solver rather than
checking it against the thing it is supposed to be inverting.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from app.logic.dcf import financials as fin
from app.logic.dcf import model as dcf_model
from app.logic.dcf import reverse
from app.logic.dcf import valuation as dcf
from app.logic.dcf.reverse import Shape, Status
from app.logic.dcf.valuation import ProjectionError, TerminalValueError
from tests.test_golden import GOLDEN, build_model


@pytest.fixture
def built(statements_payload, tickers_response, monkeypatch):
    return build_model(statements_payload, tickers_response, monkeypatch)


def counting(function):
    """Wraps an objective so a test can see exactly what it was asked."""
    calls: list[float] = []

    def wrapped(growth: float) -> float:
        calls.append(growth)
        return function(growth)

    wrapped.calls = calls
    return wrapped


# ---------------------------------------------------------------------------
# Roots that can be written down
# ---------------------------------------------------------------------------


def test_recovers_the_root_of_a_line():
    result = reverse.solve_growth(lambda g: 2 * g + 1, 1.4)
    assert result.status is Status.SOLVED
    assert result.ok
    assert result.growth == pytest.approx(0.2)
    assert result.shape is Shape.INCREASING
    assert result.converged
    assert result.local_slope == pytest.approx(2.0)
    assert result.growth_creates_value is True


def test_recovers_the_root_of_a_decreasing_function():
    result = reverse.solve_growth(lambda g: 1 - 3 * g, 0.4)
    assert result.status is Status.SOLVED
    assert result.growth == pytest.approx(0.2)
    assert result.shape is Shape.DECREASING
    assert result.local_slope == pytest.approx(-3.0)
    assert result.growth_creates_value is False
    assert result.value_per_growth_point == pytest.approx(-0.03)


def test_reports_both_roots_of_a_hump_and_elects_no_headline():
    """Two rates reproduce the price, so there is no single implied rate."""
    result = reverse.solve_growth(lambda g: 1 - (g - 0.1) ** 2, 0.99)
    assert result.status is Status.MULTIPLE_SOLUTIONS
    assert result.growth is None
    assert len(result.roots) == 2
    assert result.roots[0] == pytest.approx(0.0)
    assert result.roots[1] == pytest.approx(0.2)
    assert result.shape is Shape.NON_MONOTONIC
    assert "not a well-posed statement" in result.reason


def test_a_root_exactly_on_a_grid_point_is_found():
    result = reverse.solve_growth(lambda g: g, 0.15)
    assert result.status is Status.SOLVED
    assert result.growth == pytest.approx(0.15)


def test_slope_is_measured_at_the_root_not_inferred():
    # f(g) = g^3 - g has a root at g = 1, where f'(g) = 3g^2 - 1 = 2.
    result = reverse.solve_growth(lambda g: g**3 - g, 0.0, bounds=(0.5, 2.0))
    assert result.growth == pytest.approx(1.0)
    assert result.local_slope == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Nothing is reported that was not verified
# ---------------------------------------------------------------------------


def test_a_target_above_everything_is_named_rather_than_guessed():
    result = reverse.solve_growth(lambda g: g, 5.0)
    assert result.status is Status.TARGET_ABOVE_RANGE
    assert result.growth is None
    assert result.value_range == pytest.approx((-1.0, 2.0))
    assert result.argmax == pytest.approx(2.0)
    assert "-100.00%" in result.reason and "200.00%" in result.reason


def test_a_target_below_everything_is_named_rather_than_guessed():
    result = reverse.solve_growth(lambda g: g, -5.0)
    assert result.status is Status.TARGET_BELOW_RANGE
    assert result.growth is None
    assert result.argmin == pytest.approx(-1.0)


def test_a_constant_function_is_degenerate():
    result = reverse.solve_growth(lambda g: 7.0, 7.0)
    assert result.status is Status.DEGENERATE
    assert result.growth is None
    assert result.shape is Shape.FLAT
    assert "does not move with revenue growth" in result.reason


def test_an_objective_that_always_raises_is_a_precondition_not_a_failed_search():
    """One probe settles it, because every refusal is invariant in growth."""

    def objective(growth: float) -> float:
        raise TerminalValueError("Terminal growth of 20.00% is not below the WACC of 8.92%.")

    result = reverse.solve_growth(counting(objective), 100.0)
    assert result.status is Status.NOT_EVALUABLE
    assert result.growth is None
    assert result.function_calls == 1
    assert "not below the WACC" in result.reason


def test_one_dead_scan_point_does_not_kill_the_solve():
    def objective(growth: float) -> float:
        if growth == 2.0:
            raise ProjectionError("no value at the ceiling")
        return 2 * growth + 1

    result = reverse.solve_growth(objective, 1.4)
    assert result.status is Status.SOLVED
    assert result.growth == pytest.approx(0.2)
    assert len(result.failures) == 1
    assert result.failures[0][0] == pytest.approx(2.0)


def test_a_hole_where_the_root_lives_is_reported_not_papered_over():
    """The target is inside the range, so a complete scan would have found it.

    Probed away from the hole on purpose: a probe that lands in it is a
    precondition failure, which is a different finding and has its own status.
    """

    def objective(growth: float) -> float:
        if 0.0 <= growth <= 0.5:
            raise ProjectionError("no value in the middle")
        return growth

    result = reverse.solve_growth(objective, 0.25, probe_at=1.0)
    assert result.status is Status.SEARCH_BROKEN
    assert result.growth is None
    assert "broke the search" in result.reason


def test_a_probe_landing_in_a_hole_is_a_precondition_failure():
    """The same objective, probed inside the hole: one call, and it stops."""

    def objective(growth: float) -> float:
        if 0.0 <= growth <= 0.5:
            raise ProjectionError("no value in the middle")
        return growth

    result = reverse.solve_growth(objective, 0.25, probe_at=0.25)
    assert result.status is Status.NOT_EVALUABLE
    assert result.function_calls == 1


def test_a_non_finite_result_is_treated_as_no_value():
    def objective(growth: float) -> float:
        return float("nan") if growth == 2.0 else 2 * growth + 1

    result = reverse.solve_growth(objective, 1.4)
    assert result.status is Status.SOLVED
    assert any("rather than a number" in message for _, message in result.failures)


def test_the_residual_is_measured_not_assumed():
    objective = lambda g: 2 * g + 1
    result = reverse.solve_growth(objective, 1.4)
    assert result.achieved == pytest.approx(objective(result.growth))
    assert abs(result.price_error) <= max(
        reverse.PRICE_ATOL, reverse.PRICE_RTOL * abs(result.target)
    )


def test_no_target_is_not_an_answer():
    for target in (None, float("nan"), float("inf")):
        result = reverse.solve_growth(lambda g: g, target)
        assert result.status is Status.NO_TARGET
        assert result.growth is None


# ---------------------------------------------------------------------------
# The search stays inside its range
# ---------------------------------------------------------------------------


def test_the_solver_never_evaluates_outside_its_bounds():
    """What guarantees the model is never asked for growth below -100%."""
    objective = counting(lambda g: 2 * g + 1)
    reverse.solve_growth(objective, 1.4, probe_at=99.0)
    assert objective.calls
    assert all(-1.0 <= g <= 2.0 for g in objective.calls)


def test_a_narrower_bound_is_respected_rather_than_widened():
    result = reverse.solve_growth(lambda g: g, 0.5, bounds=(0.0, 0.2))
    assert result.status is Status.TARGET_ABOVE_RANGE
    assert result.growth is None
    assert result.bounds == (0.0, 0.2)


def test_the_evaluation_count_is_honest_and_capped():
    objective = counting(lambda g: 2 * g + 1)
    result = reverse.solve_growth(objective, 1.4)
    assert result.function_calls == len(objective.calls)
    assert result.function_calls <= reverse.MAX_EVALUATIONS


def test_a_tight_budget_ends_the_search_instead_of_guessing():
    result = reverse.solve_growth(lambda g: 2 * g + 1, 1.4, max_evaluations=5)
    assert result.growth is None
    assert result.status in {Status.SEARCH_BROKEN, Status.NOT_EVALUABLE}


def test_the_scan_is_recorded_without_the_root_finding_iterates():
    """`samples` is a sweep, so it must not be polluted by clustered iterates."""
    result = reverse.solve_growth(lambda g: 2 * g + 1, 1.4)
    growths = [g for g, _ in result.samples]
    assert growths == sorted(growths)
    assert len(growths) == len(set(growths))
    assert len(growths) < result.function_calls
    assert growths[0] == pytest.approx(-1.0)
    assert growths[-1] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "objective",
    [
        lambda g: 2 * g + 1,
        lambda g: 7.0,
        lambda g: float("nan"),
        lambda g: 1 - (g - 0.1) ** 2,
        lambda g: 1 / g if g else 0.0,
        lambda g: math.exp(700 * (1 + g)),
    ],
    ids=["line", "constant", "nan", "hump", "pole", "overflow"],
)
def test_the_solver_never_raises(objective):
    result = reverse.solve_growth(objective, 1.4)
    assert isinstance(result, reverse.ReverseDcfResult)
    assert isinstance(result.status, Status)


def test_only_a_solved_result_carries_a_growth_rate():
    for target, expected in [(1.4, Status.SOLVED), (99.0, Status.TARGET_ABOVE_RANGE)]:
        result = reverse.solve_growth(lambda g: 2 * g + 1, target)
        assert result.status is expected
        assert (result.growth is not None) == (result.status is Status.SOLVED)


def test_a_failed_solve_explains_itself():
    for result in [
        reverse.solve_growth(lambda g: g, 5.0),
        reverse.solve_growth(lambda g: g, -5.0),
        reverse.solve_growth(lambda g: 7.0, 7.0),
        reverse.solve_growth(lambda g: g, None),
    ]:
        assert not result.ok
        assert result.reason


def test_the_solve_is_deterministic():
    first = reverse.solve_growth(lambda g: 2 * g + 1, 1.4)
    second = reverse.solve_growth(lambda g: 2 * g + 1, 1.4)
    assert first == second


# ---------------------------------------------------------------------------
# Against the real model
# ---------------------------------------------------------------------------


FIXTURE_GROWTH = 0.09545454545454546  # the fixture's own 3-year average


@pytest.fixture
def solve(built):
    """Solves against the golden fixture, with any run input overridable."""

    def run(target_price, **overrides):
        drivers = overrides.pop("drivers", built["drivers"])
        driver_overrides = {
            key: overrides.pop(key)
            for key in list(overrides)
            if key in {
                "operating_margin", "effective_tax_rate", "capex_pct_revenue",
                "dna_pct_revenue", "nwc_pct_revenue",
            }
        }
        if driver_overrides:
            drivers = dataclasses.replace(
                drivers,
                **{k: (v,) * drivers.years for k, v in driver_overrides.items()},
            )
        kwargs = dict(
            wacc_result=built["wacc_result"],
            terminal_growth=0.025,
            exit_multiple=12.0,
            elapsed=built["elapsed"],
            mid_year=True,
            shares=built["dataset"].shares_weighted_diluted,
            share_basis="Diluted weighted-average (FY2025)",
            include_short_term_investments=True,
            operating_cash_floor=0.0,
            non_operating_assets=None,
            base_fcff=built["base_fcff"],
            terminal_tax_rate=built["wacc_result"].marginal_tax_rate,
        )
        kwargs.update(overrides)
        return dcf_model.solve_implied_growth(
            built["dataset"], built["base"],
            target_price=target_price, drivers=drivers, **kwargs,
        )

    return run


def forward(built, growth: float, **overrides) -> float:
    """Value per share from the forward model at a flat ``growth``."""
    drivers = dataclasses.replace(
        built["drivers"], revenue_growth=(growth,) * built["drivers"].years
    )
    for key, value in list(overrides.items()):
        if key in {"nwc_pct_revenue", "operating_margin"}:
            drivers = dataclasses.replace(drivers, **{key: (overrides.pop(key),) * drivers.years})
    kwargs = dict(
        wacc_result=built["wacc_result"],
        terminal_growth=0.025,
        exit_multiple=12.0,
        elapsed=built["elapsed"],
        mid_year=True,
        shares=built["dataset"].shares_weighted_diluted,
        share_basis="Diluted weighted-average (FY2025)",
        include_short_term_investments=True,
        operating_cash_floor=0.0,
        non_operating_assets=None,
        base_fcff=built["base_fcff"],
        terminal_tax_rate=built["wacc_result"].marginal_tax_rate,
    )
    kwargs.update(overrides)
    return dcf_model.run_model(
        built["dataset"], built["base"], drivers=drivers, **kwargs
    ).headline_bridge.value_per_share


@pytest.mark.parametrize("growth", [-0.20, -0.05, 0.0, 0.03, FIXTURE_GROWTH, 0.25, 0.60])
@pytest.mark.parametrize("headline_method", ["gordon", "exit_multiple"])
def test_a_forward_run_round_trips_back_through_the_solver(built, solve, growth, headline_method):
    """The property the whole thing rests on: value the model at a known rate,
    hand back the price, and the same rate must come out."""
    price = forward(built, growth, headline_method=headline_method)
    result = solve(price, headline_method=headline_method)
    assert result.status is Status.SOLVED
    assert result.growth == pytest.approx(growth, abs=1e-9)


def test_the_golden_value_per_share_implies_the_golden_growth_rate(built, solve):
    """Ties the reverse model to the frozen forward snapshot."""
    for method, price in [
        ("gordon", GOLDEN["value_per_share_gordon"]),
        ("exit_multiple", GOLDEN["value_per_share_exit"]),
    ]:
        result = solve(price, headline_method=method)
        assert result.status is Status.SOLVED
        assert result.growth == pytest.approx(FIXTURE_GROWTH, abs=1e-9)


def test_only_revenue_growth_moves_and_the_other_five_drivers_survive(built, solve):
    """Guards the `DriverPath.flat` trap: flattening would silently reset five
    drivers, and the reverse model would invert a model nobody is looking at."""
    original = built["drivers"]
    price = forward(built, 0.20)
    result = solve(price)
    assert result.status is Status.SOLVED

    solved = dataclasses.replace(
        original, revenue_growth=(result.growth,) * original.years
    )
    for key in fin.DRIVER_KEYS:
        if key == "revenue_growth":
            continue
        assert getattr(solved, key) == getattr(original, key)
    assert forward(built, result.growth) == pytest.approx(price, abs=1e-9)


def test_the_fixture_price_implies_an_implausible_rate_and_says_so(built, solve):
    """The fixture trades at 50.00 — reachable, but only on heroic growth."""
    result = solve(built["dataset"].last_price, headline_method="gordon")
    assert result.status is Status.SOLVED
    assert result.growth == pytest.approx(0.837991, abs=1e-5)
    assert result.plausible is False
    assert result.growth_creates_value is True


def test_moderate_working_capital_no_longer_turns_the_curve_over(built, solve):
    """The artefact DCF_MODEL_REVIEW.md F5 describes, fixed.

    With the final forecast year's FCFF capitalised at terminal growth, 60% of
    each revenue dollar tied up in working capital made value peak and then
    fall as growth rose — reinvestment sized for forecast growth, carried into
    a 2.5% perpetuity. The normalised terminal year reinvests only what growing
    at g costs, so the same company's value now rises with growth.
    """
    result = solve(3.45, headline_method="gordon", nwc_pct_revenue=0.6)
    assert result.status is Status.SOLVED
    assert result.growth_creates_value is True


def test_growth_that_destroys_value_gives_two_rates_and_no_headline(built, solve):
    """Working capital consuming far more than the margin earns turns the curve
    over, and two rates then reproduce the same price.

    It takes a lot now: 2.45 of working capital per dollar of revenue growth,
    which releases cash as revenue shrinks and consumes it as revenue grows,
    until the larger terminal year wins.
    """
    result = solve(2.92, headline_method="gordon", nwc_pct_revenue=2.45)
    assert result.status is Status.MULTIPLE_SOLUTIONS
    assert result.growth is None
    assert len(result.roots) == 2
    assert result.shape is Shape.NON_MONOTONIC


def test_a_price_above_the_peak_is_unreachable_at_any_growth_rate(built, solve):
    result = solve(50.0, headline_method="gordon", nwc_pct_revenue=3.0)
    assert result.status is Status.TARGET_ABOVE_RANGE
    assert result.growth is None
    assert "-100.00%" in result.reason and "200.00%" in result.reason


def test_a_single_rate_on_a_falling_branch_is_solved_and_labelled(built, solve):
    """Legitimate, but the reading inverts: a higher price means *less* growth."""
    result = solve(2.0, headline_method="gordon", nwc_pct_revenue=3.0)
    assert result.status is Status.SOLVED
    assert result.local_slope < 0
    assert result.growth_creates_value is False
    assert result.value_per_growth_point < 0


def test_terminal_growth_above_the_wacc_stops_on_the_first_evaluation(built, solve):
    """A precondition, not a failed search — it fails at every growth rate."""
    result = solve(50.0, terminal_growth=0.20)
    assert result.status is Status.NOT_EVALUABLE
    assert result.growth is None
    assert result.function_calls == 1
    assert "not below the WACC" in result.reason


def test_a_price_below_the_equity_bridge_names_the_balance_sheet(built, solve):
    """With enough non-operating assets, extinguishing revenue still beats the
    price — so the disagreement is not about the forecast at all.

    The curve dips very slightly below its value at extinction before turning
    up, so the message leads with the extinction value and mentions the dip
    second. Reporting only the interior minimum would bury the useful sentence.
    """
    result = solve(50.0, non_operating_assets=1e5)
    assert result.status is Status.TARGET_BELOW_RANGE
    assert result.growth is None
    assert "equity bridge" in result.reason
    floor = dict(result.samples)[-1.0]
    assert f"{floor:,.2f}" in result.reason
    assert floor > 50.0


@pytest.mark.parametrize("price", [None, 0.0, -5.0, float("nan")])
def test_no_usable_price_is_not_an_answer(built, solve, price):
    result = solve(price)
    assert result.status is Status.NO_TARGET
    assert result.growth is None
    assert "Nothing has been substituted" in result.reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"operating_margin": -0.5},
        {"mid_year": False},
        {"elapsed": 0.0},
        {"ronic": 0.001},
        {"shares": 1.0},
        {"operating_cash_floor": 1e9},
    ],
    ids=["loss making", "end of year", "no roll-forward", "tiny ronic", "one share", "cash floor"],
)
def test_the_solve_never_raises_on_adversarial_assumptions(built, solve, overrides):
    result = solve(50.0, **overrides)
    assert isinstance(result, reverse.ReverseDcfResult)
    assert (result.growth is not None) == (result.status is Status.SOLVED)


def test_the_model_is_never_asked_for_growth_below_total_extinction(built, solve):
    result = solve(50.0)
    assert all(-1.0 <= g <= 2.0 for g, _ in result.samples)


def test_an_independent_polynomial_gives_the_same_roots(built, solve):
    """Derived from the formulae rather than from `project()`.

    With no roll-forward, no mid-year shift and no SBC, each year's cash flow is
    ``u^(i-1) * K(g)`` with ``K`` linear in ``g``, and the normalised terminal
    year is revenue ``u^N`` times constants, so enterprise value is a
    polynomial in ``u = 1 + g``. Rooting that polynomial answers the same
    question without going through the projector, which is what makes it a
    check on the solver rather than a restatement of it.
    """
    drivers = built["drivers"]
    wacc = built["wacc_result"].wacc
    base_revenue = float(built["base"].flows["revenue"])
    years = drivers.years
    margin, tax = drivers.operating_margin[0], drivers.effective_tax_rate[0]
    dna, capex = drivers.dna_pct_revenue[0], drivers.capex_pct_revenue[0]
    nwc = drivers.nwc_pct_revenue[0]

    contribution = margin * (1 - tax) + dna - capex
    terminal_growth = 0.025

    def enterprise_value(growth: float) -> float:
        u = 1.0 + growth
        constant = base_revenue * (contribution * u - nwc * growth)
        explicit = sum(u ** (i - 1) * constant / (1 + wacc) ** i for i in range(1, years + 1))
        # Year N + 1: NOPAT at the final margin and the marginal tax rate,
        # reinvesting g ÷ RONIC of it, with RONIC equal to the WACC.
        terminal_tax = built["wacc_result"].marginal_tax_rate
        next_nopat = (
            base_revenue * u**years * (1 + terminal_growth) * margin * (1 - terminal_tax)
        )
        terminal = next_nopat * (1 - terminal_growth / wacc) / (wacc - terminal_growth)
        return explicit + terminal / (1 + wacc) ** years

    target_growth = 0.11
    bridge = built["result"].bridges["gordon"]
    shares = built["dataset"].shares_weighted_diluted
    # The bridge is constant in growth, so recover it once from the forward run.
    offset = bridge.equity_value - built["result"].enterprise_values["gordon"]

    price = (enterprise_value(target_growth) + offset) / shares
    result = solve(price, headline_method="gordon", mid_year=False, elapsed=0.0)
    assert result.status is Status.SOLVED
    assert result.growth == pytest.approx(target_growth, abs=1e-7)


def test_a_faded_forecast_round_trips_with_the_same_fade(built, solve):
    """With a fade, the solve returns the year-1 rate of the same shaped path.

    Solving for a flat rate against a faded forward run would invert a
    different model from the one on screen.
    """
    start = 0.12
    drivers = dataclasses.replace(
        built["drivers"],
        revenue_growth=dcf.growth_path(start, built["drivers"].years, 0.025),
    )
    price = dcf_model.run_model(
        built["dataset"], built["base"], drivers=drivers,
        wacc_result=built["wacc_result"], terminal_growth=0.025, exit_multiple=12.0,
        elapsed=built["elapsed"], mid_year=True,
        shares=built["dataset"].shares_weighted_diluted,
        share_basis="Diluted weighted-average (FY2025)",
        include_short_term_investments=True, operating_cash_floor=0.0,
        non_operating_assets=None, base_fcff=built["base_fcff"],
        terminal_tax_rate=built["wacc_result"].marginal_tax_rate,
    ).headline_bridge.value_per_share
    result = solve(price, drivers=drivers, growth_fade_to=0.025)
    assert result.status is Status.SOLVED
    assert result.growth == pytest.approx(start, abs=1e-7)


@pytest.mark.parametrize("method", ["gordon", "exit_multiple"])
def test_a_held_then_faded_forecast_round_trips_under_both_methods(built, solve, method):
    """§2.1/§2.3: each method's solve inverts the same held-then-faded path."""
    years, hold, start = built["drivers"].years, 3, 0.18
    drivers = dataclasses.replace(
        built["drivers"],
        revenue_growth=dcf.growth_path(start, years, 0.025, hold=hold),
    )
    price = dcf_model.run_model(
        built["dataset"], built["base"], drivers=drivers,
        wacc_result=built["wacc_result"], terminal_growth=0.025, exit_multiple=12.0,
        elapsed=built["elapsed"], mid_year=True,
        shares=built["dataset"].shares_weighted_diluted,
        share_basis="Diluted weighted-average (FY2025)",
        include_short_term_investments=True, operating_cash_floor=0.0,
        non_operating_assets=None, base_fcff=built["base_fcff"],
        terminal_tax_rate=built["wacc_result"].marginal_tax_rate, headline_method=method,
    ).headline_bridge.value_per_share
    result = solve(
        price, drivers=drivers, growth_fade_to=0.025, growth_hold=hold, headline_method=method
    )
    assert result.status is Status.SOLVED
    assert result.growth == pytest.approx(start, abs=1e-7)
