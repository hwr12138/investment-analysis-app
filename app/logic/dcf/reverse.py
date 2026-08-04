"""Solving a valuation backwards: what driver value reproduces a given price.

Pure, and deliberately ignorant of the DCF. The caller supplies a function from
one driver value to one value per share; this module finds where that function
crosses a target, or says precisely why it does not.

The separation is not decoration. The objective lives in :func:`model.run_model`,
which imports :mod:`valuation`; a solver placed inside :mod:`valuation` that
called it would be an import cycle. Taking a callable instead breaks the cycle
and — more usefully — makes this module testable against a straight line and a
parabola, where the right answer is known by hand rather than by running the
thing under test.

**Value per share is not monotonic in growth.** Writing ``u = 1 + g``, each
projected year reduces to ``FCFF_i = u^(i-1) · K(g)`` with ``K`` *linear* in
``g``, so value per share is a polynomial in ``u``: a linear factor times a
positive increasing one. When incremental working capital consumes more than the
incremental margin earns, that linear factor falls and growth *destroys* value —
the curve rises to a peak and then drops, with the peak sitting well inside the
ordinary range of growth rates. A search that assumed one root would return
whichever of the two it happened to walk into first. So the range is scanned on
a grid, every crossing is found, and the shape is measured rather than assumed.

Nothing here reports a number it did not verify. Every failure is a
:class:`Status` carrying the measurements behind it; :func:`solve_growth` does
not raise.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace as _replace
from enum import Enum

from scipy.optimize import brentq

from app.logic.dcf.valuation import ProjectionError, TerminalValueError


class Status(str, Enum):
    """Why the solve ended. Only ``SOLVED`` carries a growth rate."""

    SOLVED = "solved"
    MULTIPLE_SOLUTIONS = "multiple_solutions"
    TARGET_ABOVE_RANGE = "target_above_range"
    TARGET_BELOW_RANGE = "target_below_range"
    DEGENERATE = "degenerate"
    NOT_EVALUABLE = "not_evaluable"
    NO_TARGET = "no_target"
    SEARCH_BROKEN = "search_broken"
    RESIDUAL_TOO_LARGE = "residual_too_large"


class Shape(str, Enum):
    """How value moved across the scanned range."""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    NON_MONOTONIC = "non-monotonic"
    FLAT = "flat"


#: Errors the objective is allowed to raise, meaning "no value exists here".
#: Anything else is a bug and is deliberately not caught — a solver that
#: silently absorbs a TypeError is worse than one that crashes.
EVALUATION_ERRORS = (ProjectionError, TerminalValueError, OverflowError, ZeroDivisionError)

#: Hard search range for revenue growth. Never widened.
#:
#: Below -100% revenue would alternate sign year to year, which is not a
#: quantity the model can mean. Exactly -100% is computable — revenue goes to
#: zero and stays there — and is needed, because it is the diagnosis for a price
#: below what the equity bridge alone is worth. Above +200% a year is 243x
#: revenue over five years; past that, "no growth rate explains this price" is
#: strictly more informative than a number. An expansion policy would be a
#: policy about what is economically sayable, and belongs in a visible constant
#: rather than emerging from a loop.
GROWTH_BOUNDS: tuple[float, float] = (-1.0, 2.0)

#: Band used only to *label* a solved rate as ordinary or extraordinary.
#: Never a clamp: a rate outside it is still returned, and flagged.
PLAUSIBLE_GROWTH: tuple[float, float] = (-0.30, 0.50)

#: Scan points, dense where the curvature lives and coarse in the tails. The
#: value function is a degree-N polynomial in ``1 + g`` for N forecast years, so
#: it has at most N-1 turning points (N is 5 to 10 here); 36 intervals leaves
#: room, and two roots would have to fall within 2.5 points of each other in the
#: dense region to be missed.
GROWTH_GRID: tuple[float, ...] = (
    -1.00, -0.95, -0.90, -0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.25,
    -0.20, -0.15, -0.10, -0.075, -0.05, -0.025, 0.0, 0.025, 0.05, 0.075,
    0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60,
    0.75, 0.90, 1.10, 1.35, 1.65, 2.00,
)

XTOL = 1e-10             # on growth: 1e-8 percentage points
MAXITER = 100
MAX_EVALUATIONS = 400    # defensive; a normal solve uses about 55
PRICE_ATOL = 1e-8
PRICE_RTOL = 1e-9
SLOPE_STEP = 1e-4
FLAT_TOLERANCE = 1e-12


class _BudgetExceeded(Exception):
    """The evaluation cap was hit. Private: never escapes ``solve_growth``."""


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.2f}"


@dataclass(frozen=True)
class ReverseDcfResult:
    """A solved driver value and everything needed to audit it.

    ``growth`` is set only when ``status`` is :attr:`Status.SOLVED` — that is,
    when exactly one rate was found *and* re-running the objective at it
    reproduced the target. Every other outcome leaves it ``None`` and explains
    itself in ``reason``.
    """

    status: Status
    target: float | None
    growth: float | None
    roots: tuple[float, ...] = ()
    achieved: float | None = None
    price_error: float | None = None
    converged: bool = False
    iterations: int = 0
    function_calls: int = 0
    bracket: tuple[float, float] | None = None
    bounds: tuple[float, float] = GROWTH_BOUNDS
    #: Grid scan only, ascending. Root-finding iterates are excluded — they
    #: cluster on the root and would misrepresent this as a sweep.
    samples: tuple[tuple[float, float], ...] = ()
    failures: tuple[tuple[float, str], ...] = ()
    shape: Shape = Shape.FLAT
    local_slope: float | None = None
    value_range: tuple[float, float] | None = None
    argmin: float | None = None
    argmax: float | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is Status.SOLVED

    @property
    def plausible(self) -> bool | None:
        """Whether a solved rate is inside the band a company might sustain."""
        if self.growth is None:
            return None
        low, high = PLAUSIBLE_GROWTH
        return low <= self.growth <= high

    @property
    def growth_creates_value(self) -> bool | None:
        """Whether more growth raises value *here*. ``False`` inverts the reading."""
        if self.local_slope is None:
            return None
        return self.local_slope > 0

    @property
    def value_per_growth_point(self) -> float | None:
        """Value per share gained or lost per one point of growth, at the root."""
        if self.local_slope is None:
            return None
        return self.local_slope * 0.01

    @property
    def formula(self) -> str:
        """How the number was obtained, for the provenance badge."""
        if self.status is not Status.SOLVED:
            return self.reason
        parts = [f"value per share(g) = {_num(self.target)}"]
        if self.bracket:
            parts.append(f"Brent's method on [{self.bracket[0]:.4f}, {self.bracket[1]:.4f}]")
        parts.append(f"{self.function_calls} model evaluations")
        if self.price_error is not None:
            parts.append(f"residual {self.price_error:.1e}")
        if self.local_slope is not None:
            direction = "increasing" if self.local_slope > 0 else "decreasing"
            parts.append(f"value {direction} at the root")
        return " · ".join(parts)


class _Objective:
    """Counts, records, and bounds-checks every call to the caller's function.

    Non-finite results are treated as failures rather than passed on: a NaN
    reaching a root-finder produces a confident answer to a question that had no
    answer.
    """

    def __init__(
        self,
        objective: Callable[[float], float],
        bounds: tuple[float, float],
        max_evaluations: int,
    ) -> None:
        self._objective = objective
        self._low, self._high = bounds
        self._max_evaluations = max_evaluations
        self.calls = 0
        self.failures: list[tuple[float, str]] = []

    def __call__(self, growth: float) -> float:
        value = self.evaluate(growth)
        if value is None:
            raise ProjectionError(
                f"The model could not be evaluated at a growth rate of {growth:.6%}."
            )
        return value

    def evaluate(self, growth: float) -> float | None:
        """Returns the value, or ``None`` if this point has no usable one."""
        if self.calls >= self._max_evaluations:
            raise _BudgetExceeded(
                f"The solve exceeded its budget of {self._max_evaluations} model evaluations."
            )
        # Nothing may ask the model for a growth rate outside the search range;
        # below -100% the projection has no meaning at all.
        growth = min(max(growth, self._low), self._high)
        self.calls += 1
        try:
            value = float(self._objective(growth))
        except EVALUATION_ERRORS as error:
            self.failures.append((growth, str(error)))
            return None
        if not math.isfinite(value):
            self.failures.append((growth, f"The model returned {value} rather than a number."))
            return None
        return value


def _effective_grid(
    grid: tuple[float, ...], bounds: tuple[float, float]
) -> tuple[float, ...]:
    """The scan points inside ``bounds``, with both bounds always included."""
    low, high = bounds
    points = {low, high} | {g for g in grid if low < g < high}
    return tuple(sorted(points))


def _classify_shape(samples: list[tuple[float, float]], epsilon: float) -> Shape:
    if len(samples) < 2:
        return Shape.FLAT
    diffs = [b[1] - a[1] for a, b in zip(samples, samples[1:])]
    if all(d > epsilon for d in diffs):
        return Shape.INCREASING
    if all(d < -epsilon for d in diffs):
        return Shape.DECREASING
    if all(abs(d) <= epsilon for d in diffs):
        return Shape.FLAT
    return Shape.NON_MONOTONIC


def solve_growth(
    objective: Callable[[float], float],
    target: float | None,
    *,
    bounds: tuple[float, float] = GROWTH_BOUNDS,
    grid: tuple[float, ...] = GROWTH_GRID,
    xtol: float = XTOL,
    maxiter: int = MAXITER,
    max_evaluations: int = MAX_EVALUATIONS,
    probe_at: float = 0.0,
) -> ReverseDcfResult:
    """Finds every value of ``g`` in ``bounds`` at which ``objective(g)`` equals ``target``.

    Scans ``grid``, runs Brent's method on each sign change, then re-evaluates
    the objective at the root so the residual is measured rather than assumed —
    Brent converges on the abscissa, and it is the price error the caller cares
    about.

    Never raises for any anticipated condition.
    """
    empty = ReverseDcfResult(status=Status.NO_TARGET, target=target, growth=None, bounds=bounds)

    if target is None or not math.isfinite(target):
        return _replace(empty, reason="No target value was supplied, so there is nothing to solve for.")

    counted = _Objective(objective, bounds, max_evaluations)

    # --- 1. One probe -----------------------------------------------------
    # Every way the model can refuse to produce a number is invariant in the
    # growth rate: the terminal growth rate and the discount rate are held fixed
    # across the whole solve, so a failure here is a failure at every rate. One
    # call establishes that, instead of the whole grid rediscovering it.
    try:
        probe = counted.evaluate(min(max(probe_at, bounds[0]), bounds[1]))
    except _BudgetExceeded as error:
        return _replace(empty, status=Status.SEARCH_BROKEN, reason=str(error))
    if probe is None:
        message = counted.failures[-1][1]
        return _replace(
            empty,
            status=Status.NOT_EVALUABLE,
            function_calls=counted.calls,
            failures=tuple(counted.failures),
            reason=(
                f"The model cannot be evaluated at these assumptions, so there is nothing "
                f"to invert. {message}"
            ),
        )

    # --- 2. Scan ----------------------------------------------------------
    samples: list[tuple[float, float]] = []
    try:
        for point in _effective_grid(grid, bounds):
            value = counted.evaluate(point)
            if value is not None:
                samples.append((point, value))
    except _BudgetExceeded as error:
        return _replace(empty, status=Status.SEARCH_BROKEN, function_calls=counted.calls,
                        reason=str(error))

    scanned = len(_effective_grid(grid, bounds))
    base = _replace(
        empty,
        samples=tuple(samples),
        failures=tuple(counted.failures),
        function_calls=counted.calls,
    )

    if len(samples) < 2:
        return _replace(
            base,
            status=Status.NOT_EVALUABLE,
            function_calls=counted.calls,
            reason=(
                f"The model could be evaluated at only {len(samples)} of {scanned} growth "
                f"rates across {_pct(bounds[0])} to {_pct(bounds[1])}, which is too few to "
                f"search."
            ),
        )

    values = [v for _, v in samples]
    low_value, high_value = min(values), max(values)
    argmin = min(samples, key=lambda s: s[1])[0]
    argmax = max(samples, key=lambda s: s[1])[0]
    epsilon = FLAT_TOLERANCE * max(1.0, max(abs(v) for v in values))
    shape = _classify_shape(samples, epsilon)

    base = _replace(
        base,
        value_range=(low_value, high_value),
        argmin=argmin,
        argmax=argmax,
        shape=shape,
        function_calls=counted.calls,
    )

    # --- 3. Degenerate ----------------------------------------------------
    if high_value - low_value <= epsilon:
        return _replace(
            base,
            status=Status.DEGENERATE,
            function_calls=counted.calls,
            reason=(
                f"Value per share does not move with revenue growth under these "
                f"assumptions — it is {_num(low_value)} at every rate tested. No growth "
                f"rate is implied, because every rate produces the same answer."
            ),
        )

    # --- 4. Every crossing ------------------------------------------------
    roots: list[float] = []
    iterations = 0
    bracket: tuple[float, float] | None = None
    broken: list[str] = []

    try:
        for (x0, v0), (x1, v1) in zip(samples, samples[1:]):
            d0, d1 = v0 - target, v1 - target
            if d0 == 0.0:
                roots.append(x0)
                continue
            if d0 * d1 >= 0.0:
                continue
            try:
                root, info = brentq(
                    lambda g: counted(g) - target,
                    x0, x1, xtol=xtol, maxiter=maxiter, full_output=True,
                )
            except EVALUATION_ERRORS as error:
                broken.append(
                    f"between {_pct(x0)} and {_pct(x1)} the model stopped being "
                    f"evaluable: {error}"
                )
                continue
            iterations += info.iterations
            if info.converged:
                roots.append(float(root))
                bracket = (x0, x1)
        if samples and samples[-1][1] - target == 0.0:
            roots.append(samples[-1][0])
    except _BudgetExceeded as error:
        return _replace(base, status=Status.SEARCH_BROKEN, function_calls=counted.calls,
                        reason=str(error))

    roots = _deduplicate(sorted(roots), xtol)
    base = _replace(base, roots=tuple(roots), iterations=iterations,
                    failures=tuple(counted.failures), function_calls=counted.calls)

    # --- 5. Classify ------------------------------------------------------
    if not roots:
        return _no_root(base, target, samples, low_value, high_value, argmin, argmax,
                        bounds, scanned, broken, counted)

    if len(roots) > 1:
        return _replace(
            base,
            status=Status.MULTIPLE_SOLUTIONS,
            function_calls=counted.calls,
            reason=_multiple_reason(roots, samples, target),
        )

    root = roots[0]

    # Brent converged on the abscissa. Whether the *price* comes back is a
    # separate question, and it is the one being claimed on screen.
    try:
        achieved = counted.evaluate(root)
        slope = _slope(counted, root, bounds)
    except _BudgetExceeded as error:
        return _replace(base, status=Status.SEARCH_BROKEN, function_calls=counted.calls,
                        reason=str(error))

    if achieved is None:
        return _replace(
            base,
            status=Status.SEARCH_BROKEN,
            function_calls=counted.calls,
            failures=tuple(counted.failures),
            reason=(
                f"A growth rate of {_pct(root)} was found, but the model could not be "
                f"re-evaluated there to confirm it."
            ),
        )

    price_error = achieved - target
    tolerance = max(PRICE_ATOL, PRICE_RTOL * abs(target))
    if abs(price_error) > tolerance:
        return _replace(
            base,
            status=Status.RESIDUAL_TOO_LARGE,
            achieved=achieved,
            price_error=price_error,
            bracket=bracket,
            function_calls=counted.calls,
            reason=(
                f"The search converged on a growth rate of {_pct(root)}, but re-running the "
                f"model at that rate produces {_num(achieved)} rather than {_num(target)} — "
                f"a gap of {_num(price_error)}. The rate is not reported, because it was "
                f"not verified."
            ),
        )

    return _replace(
        base,
        status=Status.SOLVED,
        growth=root,
        achieved=achieved,
        price_error=price_error,
        converged=True,
        bracket=bracket,
        local_slope=slope,
        function_calls=counted.calls,
        reason="",
    )


def _slope(counted: _Objective, root: float, bounds: tuple[float, float]) -> float | None:
    """d(value)/d(growth) at the root, by central difference.

    Measured rather than inferred from the drivers: whether growth creates or
    destroys value is the one thing that inverts how the answer reads, and it
    should not depend on rederiving the projection algebra here.
    """
    low = max(bounds[0], root - SLOPE_STEP)
    high = min(bounds[1], root + SLOPE_STEP)
    if high <= low:
        return None
    below, above = counted.evaluate(low), counted.evaluate(high)
    if below is None or above is None:
        return None
    return (above - below) / (high - low)


def _deduplicate(roots: list[float], xtol: float) -> list[float]:
    """Collapses roots closer together than the solver could tell apart."""
    unique: list[float] = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > max(xtol * 10, 1e-9):
            unique.append(root)
    return unique


def _multiple_reason(
    roots: list[float], samples: list[tuple[float, float]], target: float
) -> str:
    inner = [s for s in samples if roots[0] < s[0] < roots[-1]]
    extremum = ""
    if inner:
        peak = max(inner, key=lambda s: abs(s[1] - target))
        direction = "rises to a peak of" if peak[1] > target else "falls to a low of"
        extremum = (
            f" Value per share {direction} about {_num(peak[1])} near {_pct(peak[0])} "
            f"between them."
        )
    listed = " and ".join(_pct(r) for r in roots)
    return (
        f"{len(roots)} growth rates reproduce this price: {listed}.{extremum} "
        f"Past the turn, each extra point of revenue consumes more working capital and "
        f"capital expenditure than the extra margin it earns. \"The market is pricing in "
        f"X% growth\" is not a well-posed statement for this model — every one of these "
        f"rates is equally consistent with the price, so no single figure is shown."
    )


def _no_root(
    base: ReverseDcfResult,
    target: float,
    samples: list[tuple[float, float]],
    low_value: float,
    high_value: float,
    argmin: float,
    argmax: float,
    bounds: tuple[float, float],
    scanned: int,
    broken: list[str],
    counted: _Objective,
) -> ReverseDcfResult:
    """Names which of the three no-root situations this is."""
    span = f"{_pct(bounds[0])} to {_pct(bounds[1])}"

    if target > high_value:
        return _replace(
            base,
            status=Status.TARGET_ABOVE_RANGE,
            function_calls=counted.calls,
            reason=(
                f"No flat revenue growth rate between {span} a year produces a value of "
                f"{_num(target)}. Across that range this model values the shares between "
                f"{_num(low_value)} and {_num(high_value)}, the highest at a growth rate of "
                f"{_pct(argmax)}. The gap is not in the growth assumption — it is in the "
                f"operating margin, the reinvestment rates, the discount rate, the terminal "
                f"assumption, or the share count."
            ),
        )

    if target < low_value:
        # The lowest scanned point is total revenue extinction, where the
        # forecast contributes essentially nothing and what is left is the
        # bridge. That is the informative framing whenever the price is below
        # everything the model produces, so it leads.
        floor_value = samples[0][1]
        detail = (
            f"Even if revenue fell to zero immediately and stayed there, this model values "
            f"the shares at {_num(floor_value)}, above the market price of {_num(target)}. "
            f"At that point the forecast contributes nothing and what is left is the equity "
            f"bridge — net cash, debt, non-operating assets. No growth assumption reproduces "
            f"this price; the disagreement is with the balance sheet, not the forecast."
        )
        # A dip below the floor means growth destroys value somewhere in the
        # range. Mentioned only when it is real, and never instead of the above.
        if low_value < floor_value - abs(floor_value) * 1e-9:
            detail += (
                f" The lowest this model goes anywhere in {span} is {_num(low_value)}, at a "
                f"growth rate of {_pct(argmin)} — still above the price."
            )
        return _replace(base, status=Status.TARGET_BELOW_RANGE,
                        function_calls=counted.calls, reason=detail)

    # The target sits inside the range but no crossing was found. By the
    # intermediate value theorem this is impossible with a complete scan, so it
    # means the scan had holes — and the message should say which.
    missing = scanned - len(samples)
    notes = " ".join(broken)
    return _replace(
        base,
        status=Status.SEARCH_BROKEN,
        function_calls=counted.calls,
        reason=(
            f"The market price of {_num(target)} sits inside the range of values this model "
            f"produced ({_num(low_value)} to {_num(high_value)}), but it could not be "
            f"evaluated at {missing} of the {scanned} growth rates sampled, which broke the "
            f"search. {notes}".strip()
        ),
    )
