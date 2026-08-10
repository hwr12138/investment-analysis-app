"""The sectioned statement builder.

Nothing here is monkeypatched: ``_sf1_frames`` is pure, so the frames are built
from the same datatable envelope the client parses in production. The fixture
company balances, ties and reconciles exactly, so **any reconciliation firing
against it is firing wrongly** — several tests below assert precisely that, and
a failure means either the spec or the fixture is wrong, never the check.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from app.data import sharadar_client as sc
from app.logic import statements as st
from app.provenance import Origin
from tests.conftest import (
    SF1_COLUMNS,
    _ANNUAL_ROWS,
    _QUARTERLY_ROWS,
    _TTM_ROW,
    blank,
    sf1_payload,
)

SPECS = (
    st.INCOME_STATEMENT,
    st.BALANCE_SHEET,
    st.CASH_FLOW,
    st.DERIVED_FIGURES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def frames(rows=None) -> dict:
    """The four statement frames, exactly as ``fetch_statements`` assembles them."""
    parsed = sc.parse_datatable(sf1_payload(rows))
    built = sc._sf1_frames(parsed, source_link="https://example.test", restated=True)
    return {name: sc._order_statement_rows(frame) for name, frame in built.items()}


def bump(row: list, name: str, delta: float) -> list:
    """A copy of ``row`` with one indicator moved off its true value."""
    out = list(row)
    index = SF1_COLUMNS.index(name)
    out[index] = float(out[index]) + delta
    return out


def rows_by_key(table: st.StatementTable) -> dict[str, st.RenderedRow]:
    return {row.key: row for section in table.sections for row in section.rows}


def diagnostics_by_key(table: st.StatementTable) -> dict[str, st.ReconcileResult]:
    return {result.row_key: result for result in table.diagnostics}


@pytest.fixture
def annual() -> dict:
    return frames()


@pytest.fixture
def quarterly() -> dict:
    return frames(list(_QUARTERLY_ROWS))


@pytest.fixture
def ttm() -> dict:
    return frames([list(_TTM_ROW)])


def build(spec, periodic, latest=None, **kwargs):
    kwargs.setdefault("units", st.UNITS["As reported"])
    return st.build_statement(spec, periodic=periodic, latest=latest or {}, **kwargs)


# ---------------------------------------------------------------------------
# Spec integrity — a typo fails a named test rather than the page
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_the_spec_is_structurally_sound(spec):
    """Captions exist, keys are unique, and no dependency dangles."""
    assert st.validate_spec(spec) == []


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_every_caption_is_one_sharadar_actually_publishes(spec):
    known = set(sc.SF1_CAPTIONS.values()) | {sc.MINORITY_INTEREST_CAPTION}
    for row in spec.rows():
        if isinstance(row.source, st.FromCaption):
            assert row.source.caption in known, f"{spec.key}.{row.key}"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_a_derived_row_only_names_reported_rows_or_earlier_derived_ones(spec):
    """The evaluator resolves captions first, then derived rows in order.

    A plug is displayed *above* the subtotal it plugs off, so plain declaration
    order is not the dependency order and this is the rule that replaces it.
    """
    reported = {r.key for r in spec.rows() if isinstance(r.source, st.FromCaption)}
    seen: set[str] = set()
    for row in spec.rows():
        for named in st._dependencies(row.source):
            assert named in reported or named in seen, f"{spec.key}.{row.key} → {named}"
        seen.add(row.key)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_no_subtotal_is_checked_against_its_own_plug(spec):
    """A check that holds by construction is decoration, not verification."""
    plugged: dict[str, set[str]] = {}
    for row in spec.rows():
        if isinstance(row.source, st.Residual):
            plugged.setdefault(row.source.base, set()).add(row.key)
    for row in spec.rows():
        if row.check is None:
            continue
        named = set(st._dependencies(row.check.expected))
        assert not (named & plugged.get(row.key, set())), f"{spec.key}.{row.key}"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_a_percentage_row_is_computed_here_or_read_off_the_metrics_frame(spec):
    """Sharadar publishes percentages only on its DERIVED (metrics) frame.

    A ``PERCENT`` row reading a PL, BS or CF caption would be typing a currency
    figure as a rate, and would render, say, revenue as "39,103,500.0%".
    """
    statement_of = {c: s for _, c, s in sc.SF1_LAYOUT}
    for row in spec.rows():
        if row.unit is not st.Unit.PERCENT:
            continue
        if isinstance(row.source, st.FromCaption):
            assert statement_of[row.source.caption] == "DERIVED", (
                f"{spec.key}.{row.key} types a "
                f"{statement_of[row.source.caption]} caption as a percentage"
            )
        else:
            assert isinstance(row.source, st.Ratio), f"{spec.key}.{row.key}"


def test_the_four_statements_are_offered_in_reading_order():
    assert list(st.STATEMENTS) == [
        "Income statement", "Balance sheet", "Cash flow", "Derived figures",
    ]


# ---------------------------------------------------------------------------
# Missing stays missing
# ---------------------------------------------------------------------------


def test_a_residual_with_an_unreported_component_is_missing_not_wrong(annual):
    """The whole point of banning ``nansum``.

    Blanking share-based compensation must make Other Operating Activities
    vanish, *not* silently recompute it as if the company reported nil.
    """
    sparse = frames([blank(row, "sbcomp") for row in _ANNUAL_ROWS])
    table = build(st.CASH_FLOW, sparse)
    rows = rows_by_key(table)

    assert "ncfo_other" not in rows or all(c.missing for c in rows["ncfo_other"].cells)

    intact = rows_by_key(build(st.CASH_FLOW, annual))["ncfo_other"]
    would_have_been = intact.cells[0].value
    assert would_have_been is not None
    # What nansum would have produced instead — a plausible, wrong number.
    ncfo = rows_by_key(build(st.CASH_FLOW, sparse, show_empty=True))["ncfo"].cells[0].value
    netinc = rows_by_key(build(st.CASH_FLOW, sparse, show_empty=True))["netinc"].cells[0].value
    depamor = rows_by_key(
        build(st.CASH_FLOW, sparse, show_empty=True)
    )["depamor"].cells[0].value
    assert ncfo - netinc - depamor != would_have_been


def test_a_missing_residual_says_which_component_was_not_reported(annual):
    sparse = frames([blank(row, "sbcomp") for row in _ANNUAL_ROWS])
    table = build(st.CASH_FLOW, sparse, show_empty=True)
    cell = rows_by_key(table)["ncfo_other"].cells[0]

    assert cell.missing
    assert cell.text == st.MISSING_TEXT
    assert "Share-Based Compensation" in cell.reason
    assert "not reported" in cell.reason


def test_one_missing_line_does_not_take_down_the_residuals_that_do_not_use_it():
    """Only the sections that depend on the gap lose their plug."""
    sparse = frames([blank(row, "sbcomp") for row in _ANNUAL_ROWS])
    rows = rows_by_key(build(st.CASH_FLOW, sparse, show_empty=True))

    assert all(cell.missing for cell in rows["ncfo_other"].cells)
    assert not any(cell.missing for cell in rows["ncfi_other"].cells)
    assert not any(cell.missing for cell in rows["ncff_other"].cells)


def test_the_sparse_fixture_loses_exactly_the_rows_that_depend_on_it(
    sparse_statements_payload,
):
    """``rnd`` and ``intangibles`` blank out one residual apiece, on two pages."""
    parsed = sc.parse_datatable(sparse_statements_payload)
    built = sc._sf1_frames(parsed, source_link="x", restated=True)
    sparse = {name: sc._order_statement_rows(f) for name, f in built.items()}

    income = rows_by_key(build(st.INCOME_STATEMENT, sparse, show_empty=True))
    assert all(cell.missing for cell in income["opex_other"].cells)
    assert not any(cell.missing for cell in income["nonop"].cells)

    balance = rows_by_key(build(st.BALANCE_SHEET, sparse, show_empty=True))
    assert all(cell.missing for cell in balance["assetsnc_other"].cells)
    assert not any(cell.missing for cell in balance["assetsc_other"].cells)


def test_a_caption_sharadar_never_returned_is_unavailable_not_zero():
    """A column absent from the response differs from one returned empty."""
    keep = [i for i, name in enumerate(SF1_COLUMNS) if name != "inventory"]
    payload = {
        "datatable": {
            "data": [[row[i] for i in keep] for row in _ANNUAL_ROWS],
            "columns": [{"name": SF1_COLUMNS[i], "type": "text"} for i in keep],
        },
        "meta": {"next_cursor_id": None},
    }
    built = sc._sf1_frames(sc.parse_datatable(payload), source_link="x", restated=True)
    without = {name: sc._order_statement_rows(f) for name, f in built.items()}

    rows = rows_by_key(build(st.BALANCE_SHEET, without, show_empty=True))
    assert rows["inventory"].origin is Origin.UNAVAILABLE
    assert all(cell.missing for cell in rows["inventory"].cells)
    # And the residual that depends on it goes with it.
    assert all(cell.missing for cell in rows["assetsc_other"].cells)


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.key)
def test_every_check_ties_against_the_fixture(spec, annual):
    """The fixture reconciles exactly, so a firing check is a broken check."""
    table = build(spec, annual)
    assert table.diagnostics, f"{spec.key} declares no checks at all"
    for result in table.diagnostics:
        assert result.status == "ok", f"{result.row_key}: {result.detail}"
    assert table.flagged_rows == ()


def test_a_subtotal_that_disagrees_with_its_components_is_flagged():
    """Moving cost of revenue breaks gross profit's identity and nothing else.

    Perturbing a *subtotal* would cascade — gross profit is itself a component
    of the operating-income check — so a detail line is moved instead, which
    isolates the one identity under test.
    """
    broken = [bump(row, "cor", 25.0) for row in _ANNUAL_ROWS]
    table = build(st.INCOME_STATEMENT, frames(broken))

    assert table.flagged_rows == ("gp",)
    result = diagnostics_by_key(table)["gp"]
    assert result.status == "gap"
    assert result.worst_gap == pytest.approx(25.0)
    assert result.worst_relative == pytest.approx(0.05)
    assert result.worst_period == "FY2025"
    assert "Revenues" in result.formula and "Cost of Revenue" in result.formula


def test_a_broken_subtotal_cascades_into_the_checks_that_use_it():
    """Gross profit feeds operating income, so both identities fail together."""
    broken = [bump(row, "gp", 0.05 * float(row[SF1_COLUMNS.index("gp")]))
              for row in _ANNUAL_ROWS]
    assert build(st.INCOME_STATEMENT, frames(broken)).flagged_rows == ("gp", "opinc")


def test_a_reported_subtotal_is_shown_rather_than_the_computed_one():
    """The flag is the signal; the number stays as filed."""
    broken = [bump(row, "gp", 25.0) for row in _ANNUAL_ROWS]
    rows = rows_by_key(build(st.INCOME_STATEMENT, frames(broken)))

    assert rows["gp"].cells[0].value == 525.0     # as reported, not 500
    assert rows["gp"].flagged


def test_a_check_whose_input_is_absent_is_not_evaluated_rather_than_passing():
    without_cor = frames([blank(row, "cor") for row in _ANNUAL_ROWS])
    result = diagnostics_by_key(build(st.INCOME_STATEMENT, without_cor))["gp"]

    assert result.status == "not evaluated"
    assert result.periods_checked == 0


def test_the_absolute_floor_absorbs_a_rounding_difference_on_a_small_subtotal():
    """``ncf`` is 20 against a 0.1% tolerance — a pure relative test would fire."""
    nudged = frames([bump(row, "ncf", 0.5) for row in _ANNUAL_ROWS])
    assert diagnostics_by_key(build(st.CASH_FLOW, nudged))["ncf"].status == "ok"

    moved = frames([bump(row, "ncf", 4.0) for row in _ANNUAL_ROWS])
    assert diagnostics_by_key(build(st.CASH_FLOW, moved))["ncf"].status == "gap"


def test_a_signed_check_subtracts_rather_than_adds():
    """``netincnci`` arrives as a positive magnitude — the client does not negate it.

    Reconciling it with a plus would pass only because the fixture's figure is
    nil, so it is given a value here and the sign is what decides the outcome.
    """
    with_nci = [bump(row, "netincnci", 30.0) for row in _ANNUAL_ROWS]
    result = diagnostics_by_key(build(st.INCOME_STATEMENT, frames(with_nci)))["netinc"]

    assert result.status == "gap"
    assert "−" in result.formula

    # Consolidated income raised to match: net income is now the residual, and
    # only a subtraction reconciles it.
    consistent = [bump(bump(row, "netincnci", 30.0), "consolinc", 30.0)
                  for row in _ANNUAL_ROWS]
    assert diagnostics_by_key(
        build(st.INCOME_STATEMENT, frames(consistent))
    )["netinc"].status == "ok"


def test_diagnostics_report_the_passes_too(annual):
    """A reader must be able to see what was checked, not only what failed."""
    table = build(st.BALANCE_SHEET, annual, latest={})
    checked = {result.row_key for result in table.diagnostics}
    assert checked == {"assets", "liabilities", "debt", "investments"}


def test_free_cash_flow_per_share_states_and_verifies_its_divisor(annual):
    """The label claims levered FCF over *basic* shares; the check proves it.

    Sharadar documents no divisor for ``fcfps``, so asserting one in a note and
    leaving it unverified would be exactly the plausible-looking claim this
    table exists to prevent.
    """
    row = next(r for r in st.CASH_FLOW.rows() if r.key == "fcfps")
    assert row.label == "Free Cash Flow per Share (Levered)"
    assert row.check.expected == st.Ratio("fcf", "shareswa_adj")

    result = diagnostics_by_key(build(st.CASH_FLOW, annual))["fcfps"]
    assert result.status == "ok"
    # Both halves name themselves: levered numerator, basic divisor.
    assert result.formula == (
        "Free Cash Flow (Levered) ÷ "
        "Basic Weighted Average Shares, Share-Factor Adjusted"
    )


def test_the_share_factor_is_what_makes_an_adr_reconcile():
    """Divide by the raw basic count and every ADR is out by the ADR ratio.

    Discovered against TSM, whose share factor is 0.2 — five ordinary shares to
    the ADR — where the unadjusted divisor put free cash flow per share out by a
    factor of five. Every US company has a factor of 1.0 and ties either way,
    which is exactly why this needs its own test.
    """
    adr = []
    for row in _ANNUAL_ROWS:
        out = list(row)
        fcf = float(out[SF1_COLUMNS.index("fcf")])
        out[SF1_COLUMNS.index("sharefactor")] = 0.2
        out[SF1_COLUMNS.index("fcfps")] = round(fcf / (1000.0 * 0.2), 6)
        adr.append(out)

    table = build(st.CASH_FLOW, frames(adr))
    assert diagnostics_by_key(table)["fcfps"].status == "ok"

    rows = rows_by_key(table)
    assert rows["shareswa"].cells[0].value == 1000.0
    assert rows["shareswa_adj"].cells[0].value == pytest.approx(200.0)


def test_dividing_by_diluted_shares_would_fail_the_check(annual):
    """Proof the check discriminates rather than passing on anything."""
    # Same keys, diluted caption — so the spec stays valid and only the
    # divisor changes.
    swaps = {
        "shareswa": st.RowSpec("shareswa", "Diluted Weighted Average Shares",
                               st.FromCaption("Weighted Average Shares Diluted"),
                               unit=st.Unit.SHARES),
    }
    spec = dataclasses.replace(
        st.CASH_FLOW,
        sections=tuple(
            dataclasses.replace(
                section,
                rows=tuple(swaps.get(r.key, r) for r in section.rows),
            )
            for section in st.CASH_FLOW.sections
        ),
    )
    assert st.validate_spec(spec) == []

    # The fixture's diluted count is 1,010 against 1,000 basic — a 1% gap,
    # comfortably outside the 0.1% cash-flow tolerance.
    assert diagnostics_by_key(build(spec, annual))["fcfps"].status == "gap"


def test_a_per_share_check_is_not_swamped_by_the_money_sized_floor(annual):
    """One currency unit would let an 11% error through on a figure of ~9."""
    assert st.STATEMENT_TOLERANCES.absolute_floor == 1.0
    assert st.STATEMENT_TOLERANCES.unscaled_floor < 0.01
    assert st._floor_for(st.Unit.CURRENCY, st.STATEMENT_TOLERANCES) == 1.0
    assert st._floor_for(st.Unit.SHARES, st.STATEMENT_TOLERANCES) == 1.0
    for unit in (st.Unit.PER_SHARE, st.Unit.PERCENT, st.Unit.RATIO):
        assert st._floor_for(unit, st.STATEMENT_TOLERANCES) < 0.01


def test_the_balance_sheet_identity_is_shown_but_never_checked():
    """Minority interest is derived as that very residual, so it cannot fail."""
    row = next(r for r in st.BALANCE_SHEET.rows() if r.key == "liab_and_equity")
    assert row.check is None
    assert "construction" in row.note


# ---------------------------------------------------------------------------
# Period columns
# ---------------------------------------------------------------------------


def test_annual_columns_are_fiscal_years_newest_first(annual):
    table = build(st.INCOME_STATEMENT, annual)
    assert [c.label for c in table.periods] == ["FY2025", "FY2024", "FY2023"]
    assert table.periods[0].sublabel == "2025-09-30"


def test_quarterly_columns_carry_the_quarter(quarterly):
    table = build(st.INCOME_STATEMENT, quarterly, annual=False)
    assert [c.label for c in table.periods] == [
        "FY2026 Q2", "FY2026 Q1", "FY2025 Q4", "FY2025 Q3",
    ]


def test_the_ttm_column_leads_and_is_marked_as_such(annual, ttm):
    table = build(st.CASH_FLOW, annual, latest=ttm)
    assert table.periods[0].label == "TTM"
    assert table.periods[0].is_latest
    assert table.periods[0].sublabel.startswith("to ")
    assert [c.label for c in table.periods[1:]] == ["FY2025", "FY2024", "FY2023"]


def test_the_balance_sheet_leading_column_is_a_quarter_not_a_twelve_month_figure(
    annual, quarterly
):
    """A balance sheet is a snapshot; calling its leading column TTM would lie."""
    table = build(st.BALANCE_SHEET, annual, latest=quarterly)
    assert table.periods[0].label == "Latest (MRQ)"
    assert "FY2026 Q2" in table.periods[0].sublabel
    assert "2026-06-30" in table.periods[0].sublabel


def test_the_quarterly_balance_sheet_has_no_leading_column(quarterly):
    """It would be the most recent quarter twice over."""
    table = build(st.BALANCE_SHEET, quarterly, latest=quarterly, annual=False)
    assert not table.periods[0].is_latest
    assert [c.label for c in table.periods][0] == "FY2026 Q2"
    assert len(table.periods) == 4


def test_the_quarterly_cash_flow_keeps_its_ttm_column(quarterly, ttm):
    """TTM is a genuinely different aggregate from the latest quarter."""
    table = build(st.CASH_FLOW, quarterly, latest=ttm, annual=False)
    assert table.periods[0].label == "TTM"
    assert table.periods[1].label == "FY2026 Q2"


def test_the_period_count_trims_the_oldest(annual, ttm):
    table = build(st.CASH_FLOW, annual, latest=ttm, max_periods=2)
    assert [c.label for c in table.periods] == ["TTM", "FY2025", "FY2024"]


def test_all_periods_are_kept_when_no_limit_is_given(annual):
    table = build(st.INCOME_STATEMENT, annual, max_periods=None)
    assert len(table.periods) == 3


def test_a_row_with_no_fiscal_year_falls_back_to_its_report_date():
    """A malformed ``fiscalperiod`` must degrade, not raise, in front of the user."""
    mangled = list(_ANNUAL_ROWS)
    mangled[0] = list(mangled[0])
    mangled[0][SF1_COLUMNS.index("fiscalperiod")] = "unparseable"
    table = build(st.INCOME_STATEMENT, frames(mangled))

    assert "2025-09-30" in [c.label for c in table.periods]


def test_the_filing_behind_each_column_travels_with_the_table(annual, ttm):
    table = build(st.CASH_FLOW, annual, latest=ttm)
    assert list(table.meta["Column"]) == [c.label for c in table.periods]
    assert "Publish Date" in table.meta.columns
    assert "Restated" in table.meta.columns


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def test_units_scale_money_and_share_counts(annual):
    reported = rows_by_key(build(st.INCOME_STATEMENT, annual))
    millions = rows_by_key(
        build(st.INCOME_STATEMENT, annual, units=st.UNITS["Millions"])
    )
    assert reported["revenue"].cells[0].text == "1,200"
    assert millions["revenue"].cells[0].text == "0.0"
    assert millions["shareswa"].cells[0].text == "0.0"


@pytest.mark.parametrize("unit_name", list(st.UNITS))
def test_per_share_percent_and_ratio_rows_never_take_the_divisor(annual, unit_name):
    """The regression guard: scaling a whole frame turns EPS into 0.00.

    The DCF tab's old raw table divided every row by the units divisor, which is
    wrong for exactly these three unit kinds. The ``Unit`` enum exists so this
    module cannot make that mistake — which is why the table was safe to delete
    rather than port.
    """
    income = rows_by_key(build(st.INCOME_STATEMENT, annual, units=st.UNITS[unit_name]))
    derived = rows_by_key(build(st.DERIVED_FIGURES, annual, units=st.UNITS[unit_name]))

    assert income["epsdil"].cells[0].text == "0.25"
    assert derived["grossmargin"].cells[0].text == "41.7%"
    assert derived["pe1"].cells[0].text == "20.00"


def test_costs_render_negative(annual):
    """Proof the client's sign flip survives all the way to the cell."""
    rows = rows_by_key(build(st.INCOME_STATEMENT, annual))
    assert rows["cor"].cells[0].text.startswith("-")
    assert rows["cor"].cells[0].value == -700.0


def test_a_negated_zero_does_not_render_as_minus_zero(annual):
    """Interest expense is nil and sign-flipped; ``-0`` reads as a rounded value."""
    rows = rows_by_key(build(st.INCOME_STATEMENT, annual))
    assert rows["intexp"].cells[0].text == "0"


def test_an_unreported_row_is_hidden_unless_asked_for():
    without = frames([blank(row, "inventory") for row in _ANNUAL_ROWS])

    assert "inventory" not in rows_by_key(build(st.BALANCE_SHEET, without))
    assert "inventory" in rows_by_key(build(st.BALANCE_SHEET, without, show_empty=True))


def test_hidden_rows_are_counted_so_the_caption_can_say_how_many():
    """Three, not two: the residual that depended on inventory goes with it."""
    without = frames([blank(row, "inventory", "deposits") for row in _ANNUAL_ROWS])
    table = build(st.BALANCE_SHEET, without)

    hidden = set(rows_by_key(build(st.BALANCE_SHEET, without, show_empty=True))) - set(
        rows_by_key(table)
    )
    assert hidden == {"inventory", "deposits", "assetsc_other"}
    assert table.hidden_rows == 3
    assert table.reported_rows > 20


def test_a_section_with_nothing_left_in_it_disappears():
    """Uses Reference, whose single row makes the assertion stable.

    Pointing this at a Supplemental Items section made it fail every time a
    line item was added there, which tests the wrong thing.
    """
    assert len(dict(
        (s.title, s) for s in st.DERIVED_FIGURES.sections
    )["Reference"].rows) == 1

    without = frames([blank(row, "fxusd") for row in _ANNUAL_ROWS])
    table = build(st.DERIVED_FIGURES, without)

    assert "Reference" not in [section.title for section in table.sections]
    assert "Reference" in [
        s.title for s in build(st.DERIVED_FIGURES, without, show_empty=True).sections
    ]


def test_hiding_a_row_does_not_stop_a_residual_that_depends_on_it():
    """Hiding is a display filter applied after every row has been evaluated."""
    rows = rows_by_key(build(st.BALANCE_SHEET, frames()))
    assert rows["assetsc_other"].cells[0].value is not None


def test_a_zero_on_a_zero_filled_indicator_is_marked(annual):
    """Sharadar writes 0 both for a genuine nil and for a line it never saw."""
    rows = rows_by_key(build(st.INCOME_STATEMENT, annual))
    assert sc.is_zero_filled("Interest Expense")
    assert rows["intexp"].cells[0].zero_filled

    assert not sc.is_zero_filled("Operating Income")
    assert not rows["opinc"].cells[0].zero_filled


def test_a_zero_outside_the_zero_filled_set_is_not_marked():
    """The flag tracks the indicator, not merely the value being zero."""
    zeroed = [bump(row, "gp", -float(row[SF1_COLUMNS.index("gp")]))
              for row in _ANNUAL_ROWS]
    rows = rows_by_key(build(st.INCOME_STATEMENT, frames(zeroed)))

    assert rows["gp"].cells[0].value == 0.0
    assert not rows["gp"].cells[0].zero_filled


def test_derived_rows_carry_their_formula_and_origin(annual):
    rows = rows_by_key(build(st.INCOME_STATEMENT, annual))

    assert rows["opex_other"].origin is Origin.DERIVED
    assert rows["opex_other"].formula == (
        "Total Operating Expenses − Selling, General & Administrative − "
        "Research & Development"
    )
    assert rows["revenue"].origin is Origin.REPORTED
    assert rows["revenue"].formula == ""


def test_minority_interest_is_labelled_derived_despite_being_a_column(annual):
    """The client synthesises it; inferring "reported" from the source would lie."""
    rows = rows_by_key(build(st.BALANCE_SHEET, annual, show_empty=True))
    assert rows["nci"].origin is Origin.DERIVED


def test_subtotals_and_totals_carry_their_emphasis(annual):
    rows = rows_by_key(build(st.INCOME_STATEMENT, annual))
    assert rows["gp"].emphasis is st.Emphasis.SUBTOTAL
    assert rows["netinc"].emphasis is st.Emphasis.TOTAL
    assert rows["revenue"].emphasis is st.Emphasis.DETAIL


# ---------------------------------------------------------------------------
# Ratios and products — the supplemental rows
# ---------------------------------------------------------------------------


def test_per_share_supplementals_divide_by_the_right_share_count(annual):
    """Basic uses basic shares, diluted uses diluted — 1,000 against 1,010."""
    rows = rows_by_key(build(st.INCOME_STATEMENT, annual))

    # netinccmn 250 with no discontinued operations.
    assert rows["netinc_continuing"].cells[0].value == 250.0
    assert rows["eps_continuing"].cells[0].value == pytest.approx(250.0 / 1000.0)
    assert rows["epsdil_continuing"].cells[0].value == pytest.approx(250.0 / 1010.0)


def test_continuing_operations_strips_the_discontinued_portion():
    with_disc = [bump(row, "netincdis", 40.0) for row in _ANNUAL_ROWS]
    rows = rows_by_key(build(st.INCOME_STATEMENT, frames(with_disc)))

    assert rows["netinc_continuing"].cells[0].value == 210.0
    assert rows["eps_continuing"].cells[0].value == pytest.approx(0.21)


def test_the_effective_tax_rate_reads_as_a_positive_percentage(annual):
    """Tax expense is negative downstream of ingest; a rate is not."""
    rows = rows_by_key(build(st.INCOME_STATEMENT, annual))
    # tax 60 on pretax 310.
    assert rows["effective_tax_rate"].cells[0].value == pytest.approx(60.0 / 310.0)
    assert rows["effective_tax_rate"].cells[0].text == "19.4%"


def test_no_tax_rate_is_shown_on_a_pretax_loss():
    """A rate on a negative base is arithmetic without a meaning."""
    losses = [bump(row, "ebt", -2 * float(row[SF1_COLUMNS.index("ebt")]))
              for row in _ANNUAL_ROWS]
    rows = rows_by_key(build(st.INCOME_STATEMENT, frames(losses), show_empty=True))

    assert all(cell.missing for cell in rows["effective_tax_rate"].cells)


def test_a_ratio_with_a_zero_denominator_is_missing_not_infinite():
    no_shares = [bump(row, "shareswa", -1000.0) for row in _ANNUAL_ROWS]
    rows = rows_by_key(
        build(st.INCOME_STATEMENT, frames(no_shares), show_empty=True)
    )
    cell = rows["eps_continuing"].cells[0]

    assert cell.missing
    assert cell.value is None


def test_a_ratio_with_a_missing_input_is_missing(annual):
    without = frames([blank(row, "netincdis") for row in _ANNUAL_ROWS])
    rows = rows_by_key(build(st.INCOME_STATEMENT, without, show_empty=True))

    assert all(cell.missing for cell in rows["netinc_continuing"].cells)
    assert all(cell.missing for cell in rows["eps_continuing"].cells)
    # ...while the reported basic EPS beside it is untouched.
    assert not any(cell.missing for cell in rows["eps"].cells)


def test_shares_outstanding_applies_the_share_factor():
    """The correction Sharadar itself applies before computing market cap."""
    halved = [bump(row, "sharefactor", -0.5) for row in _ANNUAL_ROWS]
    rows = rows_by_key(build(st.BALANCE_SHEET, frames(halved)))

    assert rows["sharesbas"].cells[0].value == 1000.0
    assert rows["shares_out"].cells[0].value == 500.0
    assert rows["shares_out"].formula == (
        "Total Shares Outstanding on Filing Date × Share Factor"
    )


def test_tangible_book_value_nets_intangibles_off_equity_not_off_assets(annual):
    """Distinct from Sharadar's Tangible Asset Value, which nets off assets."""
    balance = rows_by_key(build(st.BALANCE_SHEET, annual))
    derived = rows_by_key(build(st.DERIVED_FIGURES, annual))

    equity, intangibles = 360.0, 145.0
    assert balance["equity"].cells[0].value == equity
    assert balance["intangibles"].cells[0].value == intangibles
    assert balance["tangible_bv"].cells[0].value == equity - intangibles
    # Sharadar's own figure is assets less intangibles — a different number.
    assert derived["tangibles"].cells[0].value == 1060.0 - intangibles


def test_net_debt_nets_cash_and_short_term_investments_off_total_debt(annual):
    rows = rows_by_key(build(st.BALANCE_SHEET, annual))
    assert rows["debt"].cells[0].value == 270.0
    assert rows["cash_sti"].cells[0].value == 210.0
    assert rows["net_debt"].cells[0].value == 60.0


def test_per_share_balance_sheet_rows_use_shares_outstanding(annual):
    rows = rows_by_key(build(st.BALANCE_SHEET, annual))
    assert rows["cash_per_share"].cells[0].value == pytest.approx(210.0 / 1000.0)
    assert rows["tangible_bvps"].cells[0].value == pytest.approx(215.0 / 1000.0)


def test_the_supplemental_rows_survive_every_unit_setting(annual):
    """Per-share and percentage supplementals must not take the divisor."""
    for name in st.UNITS:
        income = rows_by_key(
            build(st.INCOME_STATEMENT, annual, units=st.UNITS[name])
        )
        balance = rows_by_key(build(st.BALANCE_SHEET, annual, units=st.UNITS[name]))
        assert income["eps_continuing"].cells[0].text == "0.25"
        assert income["effective_tax_rate"].cells[0].text == "19.4%"
        assert balance["cash_per_share"].cells[0].text == "0.21"


def test_the_requested_supplemental_items_are_all_present(annual):
    """The line items asked for, as far as SF1 can support them."""
    income = rows_by_key(build(st.INCOME_STATEMENT, annual))
    for key in ("sps", "eps", "eps_continuing", "shareswa", "epsdil",
                "epsdil_continuing", "shareswadil", "netinc_continuing", "dps",
                "payoutratio", "ebitda", "effective_tax_rate"):
        assert key in income, key

    balance = rows_by_key(build(st.BALANCE_SHEET, annual))
    for key in ("sharesbas", "shares_out", "cash_per_share", "bvps",
                "tangible_bv", "tangible_bvps", "debt", "net_debt"):
        assert key in balance, key

    cash = rows_by_key(build(st.CASH_FLOW, annual))
    for key in ("fcf", "fcfps"):
        assert key in cash, key


def test_ebit_is_not_repeated_in_the_income_statement_supplementals():
    """It already has a home under Earnings from Continuing Operations."""
    section = {s.title: s for s in st.INCOME_STATEMENT.sections}
    assert "ebit" in {r.key for r in section["Earnings from Continuing Operations"].rows}
    assert "ebit" not in {r.key for r in section["Supplemental Items"].rows}
    assert "EBIT appears above" in section["Supplemental Items"].note


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


def test_nothing_fetched_yields_an_empty_table_rather_than_an_exception():
    table = st.build_statement(st.INCOME_STATEMENT, periodic={}, latest={})
    assert table.empty
    assert table.periods == ()
    assert table.sections == ()
    assert table.label == "Income statement"


def test_only_a_ttm_set_still_renders_a_single_column(ttm):
    """A company too young for an annual filing still gets a statement."""
    table = build(st.CASH_FLOW, {}, latest=ttm)
    assert [c.label for c in table.periods] == ["TTM"]
    assert rows_by_key(table)["ncfo"].cells[0].value == 340.0


def test_the_dimension_is_recorded_so_the_caption_can_name_it(annual):
    assert build(st.INCOME_STATEMENT, annual).dimension == "MRY"
    assert build(st.INCOME_STATEMENT, annual, annual=False).dimension == "MRQ"


def test_merging_the_frames_lets_the_cash_flow_open_with_net_income(annual):
    """``netinc`` is an income-statement caption read on the cash-flow view."""
    rows = rows_by_key(build(st.CASH_FLOW, annual))
    assert rows["netinc"].cells[0].value == 250.0
    assert rows["netinc"].origin is Origin.REPORTED


def test_the_merge_joins_on_the_period_not_on_row_position(annual):
    """Shuffle one frame; every figure must still land in the right column."""
    shuffled = dict(annual)
    shuffled["CF"] = annual["CF"].iloc[::-1].reset_index(drop=True)
    rows = rows_by_key(build(st.CASH_FLOW, shuffled))

    assert [c.value for c in rows["ncfo"].cells] == [340.0, 310.0, 280.0]
    assert [c.value for c in rows["netinc"].cells] == [250.0, 226.0, 202.0]


def test_a_value_that_cannot_be_read_as_a_number_is_missing_not_a_crash():
    """Sharadar has shipped text in a numeric column before."""
    text = list(_ANNUAL_ROWS)
    text[0] = list(text[0])
    text[0][SF1_COLUMNS.index("inventory")] = "n/a"
    rows = rows_by_key(build(st.BALANCE_SHEET, frames(text), show_empty=True))

    assert rows["inventory"].cells[0].missing
    assert not math.isnan(rows["inventory"].cells[1].value)
