"""Schema resolution: raises rather than silently mismatching."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data import sharadar_client as sc
from app.logic.dcf.schema import (
    CONCEPTS_BY_KEY,
    SchemaResolutionError,
    resolve_schema,
)


def _frames(payload) -> dict[str, pd.DataFrame]:
    return sc._sf1_frames(
        sc.parse_datatable(payload), source_link="https://sec.gov/x", restated=True
    )


def test_resolves_every_concept_against_real_columns(statements_payload):
    schema = resolve_schema(_frames(statements_payload), "TEST", "Technology")
    assert schema.column("revenue") == "Revenues"
    assert schema.column("cfo") == "Net Cash Flow from Operations"
    assert schema.column("net_change_in_cash") == (
        "Net Cash Flow / Change in Cash & Cash Equivalents"
    )
    assert not schema.unresolved


def test_ebit_resolves_to_operating_income_not_sharadars_ebit(statements_payload):
    """Sharadar ships both. Only `opinc` preserves the additive identity.

    Its `ebit` adds tax and interest back to net income, so it carries
    non-operating items and would quietly change every operating margin.
    """
    schema = resolve_schema(_frames(statements_payload), "TEST", "Technology")
    assert schema.column("ebit") == "Operating Income"
    assert schema.column("derived_ebit_reported") == "Earning Before Interest & Taxes (EBIT)"


def test_equity_resolves_to_the_parent_attributable_line(statements_payload):
    schema = resolve_schema(_frames(statements_payload), "TEST", "Technology")
    assert schema.column("total_equity") == "Shareholders Equity Attributable to Parent"
    assert schema.column("minority_interest") == sc.MINORITY_INTEREST_CAPTION


def test_renamed_required_field_raises(statements_payload):
    frames = _frames(statements_payload)
    frames["PL"] = frames["PL"].rename(columns={"Revenues": "Total Net Sales"})
    with pytest.raises(SchemaResolutionError) as caught:
        resolve_schema(frames, "TEST", "Technology")
    assert caught.value.concept_label == "Revenue"
    assert caught.value.ticker == "TEST"
    assert caught.value.statement == "PL"
    assert "Total Net Sales" in caught.value.available


def test_error_lists_the_columns_that_were_found(statements_payload):
    frames = _frames(statements_payload)
    frames["CF"] = frames["CF"].drop(columns=["Net Cash Flow from Operations"])
    with pytest.raises(SchemaResolutionError) as caught:
        resolve_schema(frames, "TEST", "Technology")
    assert "Depreciation Amortization & Accretion" in caught.value.available
    assert "Technology" in str(caught.value)


def test_missing_optional_concept_is_recorded_not_raised(statements_payload):
    frames = _frames(statements_payload)
    frames["CF"] = frames["CF"].drop(columns=["Share Based Compensation"])
    schema = resolve_schema(frames, "TEST", "Technology")
    assert not schema.has("sbc")
    assert "sbc" in schema.unresolved


def test_missing_capex_raises_because_it_is_required(statements_payload):
    """Sharadar publishes one capex line, so its absence is unrecoverable."""
    frames = _frames(statements_payload)
    frames["CF"] = frames["CF"].drop(columns=["Capital Expenditure"])
    with pytest.raises(SchemaResolutionError, match="Capital Expenditure"):
        resolve_schema(frames, "TEST", "Technology")


def test_multiple_matches_are_recorded_as_alternates(statements_payload):
    """Preference order picks the first; the alternates stay visible."""
    frames = _frames(statements_payload)
    frames["PL"]["Revenue"] = frames["PL"]["Revenues"]
    schema = resolve_schema(frames, "TEST", "Technology")
    assert schema.column("revenue") == "Revenues"
    assert "Revenue" in schema.ambiguous["revenue"]


def test_schema_map_frame_lists_every_concept(statements_payload):
    schema = resolve_schema(_frames(statements_payload), "TEST", "Technology")
    frame = schema.as_frame()
    assert len(frame) == len(CONCEPTS_BY_KEY)
    assert {"Concept", "Resolved column", "Status", "Required"} <= set(frame.columns)


def test_resolution_survives_reordered_columns(statements_payload):
    """A response with the same names in a different order resolves identically."""
    frames = _frames(statements_payload)
    shuffled = {name: frame[list(reversed(frame.columns))] for name, frame in frames.items()}
    baseline = resolve_schema(frames, "TEST", "Technology")
    reordered = resolve_schema(shuffled, "TEST", "Technology")
    assert baseline.columns == reordered.columns


def test_a_concept_looks_only_at_its_own_statement(statements_payload):
    """`Total Debt` is a balance-sheet line; it must not be found on the CF frame."""
    frames = _frames(statements_payload)
    frames["CF"] = frames["CF"].assign(**{"Total Debt": 0.0})
    frames["BS"] = frames["BS"].drop(columns=["Total Debt"])
    schema = resolve_schema(frames, "TEST", "Technology")
    assert not schema.has("derived_total_debt")
