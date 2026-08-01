"""Columnar parsing, response shapes, and the API key never leaking."""

from __future__ import annotations

import pandas as pd
import pytest

from app.data import simfin_client as sc


def test_parses_plain_string_columns():
    frame = sc.parse_columnar(["Fiscal Year", "Revenue"], [[2024, 100], [2025, 120]])
    assert list(frame.columns) == ["Fiscal Year", "Revenue"]
    assert frame["Revenue"].tolist() == [100, 120]


def test_parses_detailed_column_objects(detailed_columns):
    """``details=true`` turns each column into an object keyed on ``caption``."""
    names = sc.column_names(detailed_columns)
    assert names[0] == "Fiscal Period"
    assert "Operating Income (Loss)" in names


def test_column_order_may_differ_between_calls():
    """Nothing may depend on positional stability across calls."""
    first = sc.parse_columnar(
        ["Fiscal Year", "Revenue", "Net Income"], [[2025, 1200, 240]]
    )
    second = sc.parse_columnar(
        ["Net Income", "Fiscal Year", "Revenue"], [[240, 2025, 1200]]
    )
    for column in ("Fiscal Year", "Revenue", "Net Income"):
        assert first[column].iloc[0] == second[column].iloc[0]


def test_metadata_columns_keep_their_type():
    frame = sc.parse_columnar(
        ["Fiscal Period", "Source", "Revenue"], [["FY", "https://sec.gov/x", 100]]
    )
    assert frame["Fiscal Period"].iloc[0] == "FY"
    assert frame["Source"].iloc[0] == "https://sec.gov/x"
    assert frame["Revenue"].iloc[0] == 100


def test_non_numeric_line_items_become_nan_not_zero():
    """A junk value must not silently read as a zero balance."""
    frame = sc.parse_columnar(["Revenue"], [["n/a"], [100]])
    assert pd.isna(frame["Revenue"].iloc[0])
    assert frame["Revenue"].iloc[1] == 100


def test_rows_are_sorted_chronologically_not_by_period():
    """SimFin groups rows by fiscal period then year, newest quarter not last."""
    frame = sc.parse_columnar(
        ["Fiscal Period", "Fiscal Year", "Report Date", "Revenue"],
        [
            ["Q1", 2025, "2024-12-31", 10],
            ["Q1", 2026, "2025-12-31", 12],
            ["Q2", 2025, "2025-03-31", 11],
            ["Q2", 2026, "2026-03-31", 13],
            ["Q4", 2025, "2025-09-30", 15],
        ],
    )
    ordered = sc._order_statement_rows(frame)
    assert ordered["Report Date"].iloc[-1] == pd.Timestamp("2026-03-31")
    assert ordered["Revenue"].iloc[-1] == 13


def test_restated_duplicate_keeps_the_later_filing():
    frame = sc.parse_columnar(
        ["Fiscal Period", "Fiscal Year", "Report Date", "Publish Date", "Revenue"],
        [
            ["FY", 2025, "2025-09-30", "2025-10-31", 100],
            ["FY", 2025, "2025-09-30", "2026-10-31", 105],
        ],
    )
    ordered = sc._order_statement_rows(frame)
    assert len(ordered) == 1
    assert ordered["Revenue"].iloc[0] == 105


def test_empty_columns_yield_an_empty_frame():
    assert sc.parse_columnar([], []).empty


def test_statement_set_flags_bank_and_insurance_templates():
    for template, expected in (
        ("BANKS", True), ("INSURANCES", True), ("GENERAL", False), ("general", False)
    ):
        statements = sc.StatementSet(
            ticker="X", simfin_id=1, name="X", currency="USD", template=template,
            isin="", period="FY", ttm=False, asreported=False, frames={},
        )
        assert statements.is_financial_template is expected


class _FakeResponse:
    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def test_error_never_carries_the_api_key(monkeypatch):
    """No error surface may echo the key, the headers, or the full URL."""
    secret = "SUPERSECRETKEY123"
    monkeypatch.setattr(sc, "_api_key", lambda: secret)
    monkeypatch.setattr(sc.requests, "get", lambda *a, **k: _FakeResponse(401))

    with pytest.raises(sc.SimFinError) as caught:
        sc._request("companies/list", {"ticker": "AAPL"})

    rendered = f"{caught.value} {caught.value.args!r} {caught.value.__dict__!r}"
    assert secret not in rendered
    assert "Authorization" not in rendered
    assert caught.value.status == 401


@pytest.mark.parametrize("status", [400, 404, 401, 403])
def test_each_status_gets_a_distinct_message(monkeypatch, status):
    monkeypatch.setattr(sc, "_api_key", lambda: "k")
    monkeypatch.setattr(sc.requests, "get", lambda *a, **k: _FakeResponse(status))
    with pytest.raises(sc.SimFinError) as caught:
        sc._request("companies/list", {})
    assert str(status) in caught.value.message


def test_rate_limit_is_retried_then_raised(monkeypatch):
    calls = {"n": 0}

    def _get(*args, **kwargs):
        calls["n"] += 1
        return _FakeResponse(429)

    monkeypatch.setattr(sc, "_api_key", lambda: "k")
    monkeypatch.setattr(sc.requests, "get", _get)
    monkeypatch.setattr(sc.time, "sleep", lambda *_: None)

    with pytest.raises(sc.SimFinError) as caught:
        sc._request("companies/list", {})
    assert calls["n"] == sc._MAX_ATTEMPTS
    assert caught.value.status == 429


def test_rate_limit_recovers_on_retry(monkeypatch):
    responses = [_FakeResponse(429), _FakeResponse(200, {"ok": True})]
    monkeypatch.setattr(sc, "_api_key", lambda: "k")
    monkeypatch.setattr(sc.requests, "get", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(sc.time, "sleep", lambda *_: None)
    assert sc._request("companies/list", {}) == {"ok": True}


def test_missing_key_raises_before_any_request(monkeypatch):
    monkeypatch.setattr(sc.st, "secrets", {})

    def _fail(*args, **kwargs):
        raise AssertionError("a request was made without an API key")

    monkeypatch.setattr(sc.requests, "get", _fail)
    with pytest.raises(sc.SimFinError, match="SIMFIN_API_KEY"):
        sc._request("companies/list", {})


def test_fetch_statements_parses_the_nested_shape(monkeypatch, statements_payload):
    monkeypatch.setattr(sc, "_request", lambda *a, **k: statements_payload)
    statements = sc._fetch_statements("TEST")
    assert statements.template == "GENERAL"
    assert statements.currency == "USD"
    assert set(statements.frames) == {"PL", "BS", "CF", "DERIVED"}
    assert len(statements.frame("PL")) == 3
    assert statements.frame("PL")["Fiscal Year"].iloc[-1] == 2025


def test_general_endpoint_returns_a_bare_dict_not_a_list(monkeypatch):
    """Unlike every other endpoint, this one is not wrapped in an array."""
    payload = {"columns": ["endFy", "name", "ticker"], "data": [[9, "APPLE INC", "AAPL"]]}
    monkeypatch.setattr(sc, "_request", lambda *a, **k: payload)
    record = sc._fetch_general("AAPL")
    assert record["endFy"] == 9
    assert record["ticker"] == "AAPL"


def test_company_list_is_a_plain_array_of_objects(monkeypatch):
    payload = [
        {"id": 1, "ticker": "aapl", "name": "APPLE INC", "sectorCode": 101001,
         "sectorName": "Computer Hardware", "industryName": "Technology"},
        {"id": 2, "ticker": "NSU", "name": "NEVSUN", "sectorCode": None,
         "sectorName": None, "industryName": None},
    ]
    monkeypatch.setattr(sc, "_request", lambda *a, **k: payload)
    frame = sc._fetch_company_list()
    assert frame["ticker"].tolist() == ["AAPL", "NSU"]
    assert pd.isna(frame["sectorCode"].iloc[1])


def test_weighted_shares_are_parsed_with_both_bases(monkeypatch):
    payload = [
        {"pid": 1, "endDate": "2024-09-28", "fyear": 2024, "period": "FY",
         "basic": 15343783000, "diluted": 15408095000},
        {"pid": 1, "endDate": "2025-09-27", "fyear": 2025, "period": "FY",
         "basic": 14948500000, "diluted": 15004697000},
    ]
    monkeypatch.setattr(sc, "_request", lambda *a, **k: payload)
    frame = sc._fetch_weighted_shares("AAPL")
    assert frame["diluted"].iloc[-1] == 15004697000
    assert frame["endDate"].iloc[-1] == pd.Timestamp("2025-09-27")
