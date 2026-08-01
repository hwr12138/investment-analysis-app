"""Schema resolution: raises rather than silently mismatching."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data import simfin_client as sc
from app.logic.dcf.schema import (
    CONCEPTS_BY_KEY,
    SchemaResolutionError,
    resolve_schema,
)


def _frames(payload) -> dict[str, pd.DataFrame]:
    return {
        statement["statement"]: sc.parse_columnar(statement["columns"], statement["data"])
        for statement in payload[0]["statements"]
    }


def test_resolves_every_concept_against_real_columns(statements_payload):
    schema = resolve_schema(_frames(statements_payload), "TEST", "GENERAL")
    assert schema.column("revenue") == "Revenue"
    assert schema.column("ebit") == "Operating Income (Loss)"
    assert schema.column("cfo") == "Cash from Operating Activities"
    assert schema.column("net_change_in_cash") == "Net Changes in Cash"


def test_lowercase_spelling_is_resolved_not_guessed(statements_payload):
    """SimFin spells it 'Cost of revenue'. An exact-name table has to carry that."""
    schema = resolve_schema(_frames(statements_payload), "TEST", "GENERAL")
    assert schema.column("cost_of_revenue") == "Cost of revenue"


def test_renamed_required_field_raises(statements_payload):
    frames = _frames(statements_payload)
    frames["PL"] = frames["PL"].rename(columns={"Revenue": "Total Net Sales"})
    with pytest.raises(SchemaResolutionError) as caught:
        resolve_schema(frames, "TEST", "GENERAL")
    assert caught.value.concept_label == "Revenue"
    assert caught.value.ticker == "TEST"
    assert caught.value.statement == "PL"
    assert "Total Net Sales" in caught.value.available


def test_error_lists_the_columns_that_were_found(statements_payload):
    frames = _frames(statements_payload)
    frames["CF"] = frames["CF"].drop(columns=["Cash from Operating Activities"])
    with pytest.raises(SchemaResolutionError) as caught:
        resolve_schema(frames, "TEST", "GENERAL")
    assert "Depreciation & Amortization" in caught.value.available
    assert "GENERAL" in str(caught.value)


def test_missing_optional_concept_is_recorded_not_raised(statements_payload):
    frames = _frames(statements_payload)
    frames["BS"] = frames["BS"].drop(columns=["Minority Interest"])
    schema = resolve_schema(frames, "TEST", "GENERAL")
    assert not schema.has("minority_interest")
    assert "minority_interest" in schema.unresolved


def test_required_group_is_satisfied_by_either_member(statements_payload):
    """CapEx may arrive as a gross acquisition line or a net change line."""
    frames = _frames(statements_payload)
    frames["CF"] = frames["CF"].drop(columns=["Acquisition of Fixed Assets & Intangibles"])
    schema = resolve_schema(frames, "TEST", "GENERAL")
    assert schema.has("capex_net_change")
    assert not schema.has("capex_acquisition")


def test_required_group_raises_when_no_member_resolves(statements_payload):
    frames = _frames(statements_payload)
    frames["CF"] = frames["CF"].drop(
        columns=[
            "Acquisition of Fixed Assets & Intangibles",
            "Change in Fixed Assets & Intangibles",
        ]
    )
    with pytest.raises(SchemaResolutionError, match="capex"):
        resolve_schema(frames, "TEST", "GENERAL")


def test_multiple_matches_are_recorded_as_alternates(statements_payload):
    """Preference order picks the first; the alternates stay visible."""
    frames = _frames(statements_payload)
    frames["PL"]["Net Revenue"] = frames["PL"]["Revenue"]
    schema = resolve_schema(frames, "TEST", "GENERAL")
    assert schema.column("revenue") == "Revenue"
    assert "Net Revenue" in schema.ambiguous["revenue"]


def test_schema_map_frame_lists_every_concept(statements_payload):
    schema = resolve_schema(_frames(statements_payload), "TEST", "GENERAL")
    frame = schema.as_frame()
    assert len(frame) == len(CONCEPTS_BY_KEY)
    assert {"Concept", "Resolved column", "Status", "Required"} <= set(frame.columns)


def test_resolution_survives_reordered_columns(statements_payload):
    """A response with the same names in a different order resolves identically."""
    frames = _frames(statements_payload)
    shuffled = {name: frame[list(reversed(frame.columns))] for name, frame in frames.items()}
    baseline = resolve_schema(frames, "TEST", "GENERAL")
    reordered = resolve_schema(shuffled, "TEST", "GENERAL")
    assert baseline.columns == reordered.columns
