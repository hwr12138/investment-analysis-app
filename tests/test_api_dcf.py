"""The 💰 DCF routes, over the golden fixture company.

The routes run the real engine: ``load_company`` against the recorded SF1
payload, :func:`app.logic.dcf.inputs.resolve` and ``evaluate``. These tests check
what the API adds — the overrides it accepts and refuses, the strict JSON, a
refused company as a response rather than an error, and the export — and that an
untouched request is the summary chart's own number, ``default_value``.
"""

from __future__ import annotations

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.api import dcf as api
from app.api import main
from app.data import market_data
from app.logic.dcf import inputs as dcf_inputs
from app.logic.dcf import model as dcf_model
from app.logic.dcf import validation as val
from app.logic.dcf import wacc as wacc_lib
from app.provenance import Sourced

from .test_golden import build_model

TODAY = date(2026, 11, 15)
_BETA = dcf_model.BetaOutcome(
    wacc_lib.BetaResult(
        beta=1.1, raw_beta=1.15, r_squared=0.4, observations=60, frequency="monthly",
        blume_adjusted=True, index_ticker="SPY",
    ),
    "stub", None,
)
_RF = Sourced.reported(0.042, "stub")
_ERP = Sourced.reported(0.045, "stub")


def _strict(response) -> dict:
    assert response.status_code == 200, response.text

    def refuse(constant):
        raise ValueError(f"non-finite number {constant} in the response")

    return json.loads(response.text, parse_constant=refuse)


@pytest.fixture
def built(statements_payload, tickers_response, monkeypatch):
    return build_model(statements_payload, tickers_response, monkeypatch)


@pytest.fixture
def client(built, monkeypatch):
    monkeypatch.setenv("EQUITI_AUTH", "off")
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.setattr(api, "_today", lambda: TODAY)
    monkeypatch.setattr(dcf_model, "regress_company_beta", lambda *a, **k: _BETA)
    monkeypatch.setattr(market_data, "fetch_10y_treasury_rate", lambda: _RF)
    monkeypatch.setattr(market_data, "fetch_erp", lambda: _ERP)
    return TestClient(main.app, raise_server_exceptions=False)


def _dcf(client, overrides=None) -> dict:
    return _strict(client.post("/api/research/test/dcf", json=overrides))


def test_an_untouched_request_is_the_tabs_default_value(client, built):
    body = _dcf(client)
    line = dcf_inputs.default_value(
        built["dataset"], valuation_date=TODAY, risk_free=_RF, erp=_ERP, beta=_BETA
    )
    assert body["blocked"] is None
    assert body["valuation"]["value_per_share"] == pytest.approx(
        line.headline_bridge.value_per_share, rel=1e-12
    )
    assert body["valuation"]["headline_method"] == "gordon"


def test_an_empty_body_and_no_body_are_both_the_defaults(client):
    assert _dcf(client)["valuation"] == _dcf(client, {})["valuation"]


def test_the_panel_names_every_input_with_its_default_and_provenance(client):
    panel = _dcf(client)["panel"]
    assert [d["key"] for d in panel["drivers"]] == list(dcf_inputs.DRIVER_KEYS)
    assert panel["years"] == dcf_model.DEFAULT_FORECAST_YEARS
    assert panel["risk_free"]["value"] == pytest.approx(0.042)
    assert panel["risk_free"]["provenance"]["origin"] == "reported"
    assert panel["beta"]["provenance"]["origin"] == "derived"
    assert panel["regression"]["beta"] == pytest.approx(1.1)
    assert not any(d["input"]["overridden"] for d in panel["drivers"])
    assert panel["wacc"]["wacc"] > 0
    assert [t["key"] for t in panel["targets"]] == [k for k, _ in dcf_inputs.TARGETS]


def test_an_override_moves_the_value_and_is_marked_the_readers(client):
    before = _dcf(client)
    after = _dcf(client, {"terminal_growth": 0.03})
    assert after["panel"]["terminal_growth"]["value"] == pytest.approx(0.03)
    assert after["panel"]["terminal_growth"]["provenance"]["origin"] == "user"
    assert after["panel"]["terminal_growth"]["overridden"] is True
    assert after["valuation"]["value_per_share"] != before["valuation"]["value_per_share"]


def test_a_driver_basis_and_value_reach_the_projection(client):
    body = _dcf(client, {"values": {"revenue_growth": 0.2}, "fade": False})
    growth = [row["revenue_growth"] for row in body["valuation"]["projection"]]
    assert growth == pytest.approx([0.2] * len(growth))


def test_the_valuation_carries_both_methods_the_grid_and_the_reverse_dcf(client):
    valuation = _dcf(client)["valuation"]
    assert [m["method"] for m in valuation["methods"]] == ["gordon", "exit_multiple"]
    assert sum(m["headline"] for m in valuation["methods"]) == 1
    grid = valuation["sensitivity"]
    assert len(grid["values"]) == len(grid["waccs"]) == 5
    assert all(len(row) == len(grid["columns"]) for row in grid["values"])
    assert [s["method"] for s in valuation["implied"]["solves"]] == list(dcf_inputs.REVERSE_METHODS)
    assert valuation["checks"]
    kinds = {p["kind"] for p in valuation["implied"]["chart"]["paths"]}
    assert "forecast" in kinds


@pytest.mark.parametrize(
    "overrides",
    [
        {"years": 4},
        {"hold": 12},                                   # within 1–14, beyond years − 1
        {"bases": {"revenue_growth": "Tea leaves"}},
        {"bases": {"not_a_driver": "5-year average"}},
        {"headline_method": "dividend discount"},
        {"share_basis": "Shares I made up"},
        {"cash_floor": -5},
        {"beta_frequency": "daily"},
    ],
)
def test_an_override_the_panel_does_not_offer_is_a_422(client, overrides):
    response = client.post(
        "/api/research/test/dcf",
        json=overrides,
    )
    assert response.status_code == 422, response.text


def test_a_refused_company_is_a_response_that_says_why(client, monkeypatch):
    def bank(ticker, **_):
        raise val.UnsupportedSectorError("JPM", "Financial Services", "Financial Services")

    monkeypatch.setattr(dcf_model, "load_company", bank)
    body = _strict(client.post("/api/research/jpm/dcf"))
    assert body["blocked"]["title"] == "Unsupported sector"
    assert body["panel"] is None and body["valuation"] is None


def test_a_model_that_cannot_be_built_keeps_the_panel(client):
    """So the reader can undo the input that caused it."""
    body = _dcf(client, {"terminal_growth": 0.5, "headline_method": "gordon"})
    assert body["valuation"] is None
    assert body["blocked"]["title"]
    assert body["panel"]["terminal_growth"]["value"] == pytest.approx(0.5)


def test_a_what_if_price_is_solved_against_the_same_run(client):
    value = _dcf(client, {"fade": False})["valuation"]["value_per_share"]
    body = _strict(client.post(
        "/api/research/test/dcf/what-if", json={"overrides": {"fade": False}, "price": value},
    ))
    forecast = _dcf(client, {"fade": False})["valuation"]["implied"]["forecast_growth"]
    gordon = next(s for s in body["solves"] if s["method"] == "gordon")
    assert gordon["growth"] == pytest.approx(forecast, abs=1e-6)


@pytest.mark.parametrize("kind,media", [("xlsx", "spreadsheetml"), ("csv", "text/csv")])
def test_the_export_is_the_run_on_screen(client, kind, media):
    response = client.post(
        "/api/research/test/dcf/export",
        json={"overrides": {"terminal_growth": 0.03}, "acknowledged": {}, "format": kind},
    )
    assert response.status_code == 200, response.text
    assert media in response.headers["content-type"]
    assert f'TEST_dcf_{TODAY.isoformat()}.{kind}' in response.headers["content-disposition"]
    assert len(response.content) > 100


def test_the_history_and_reference_come_with_the_panel(client):
    body = _dcf(client)
    assert body["history"]["ratios"]
    assert len(body["history"]["ratio_keys"]) == len(body["history"]["ratio_labels"])
    assert body["reference"]["schema_map"]["rows"]
    assert body["company"]["years_retrieved"] == len(body["history"]["ratios"])


def test_the_export_counts_only_this_runs_acknowledgements(client):
    issues = _dcf(client, {"terminal_growth": 0.03})["valuation"]["issues"]
    stamped = {issue["key"]: "2026-11-15 12:00:00" for issue in issues[:1]}
    response = client.post(
        "/api/research/test/dcf/export",
        json={
            "overrides": {"terminal_growth": 0.03},
            "acknowledged": {**stamped, "an_issue_this_run_did_not_raise": "2026-11-15 12:00:00"},
            "format": "csv",
        },
    )
    assert response.status_code == 200, response.text
    assert f"Acknowledged checks,{len(stamped)}" in response.text
