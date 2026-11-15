"""The sign-in gate in front of every data route — and that it fails closed.

Token verification itself is google-auth's; these tests replace
``auth.verify`` and check what the gate does with its answer.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import auth, main

PROJECT = "equiti-test"


def _claims(email="owner@example.com", verified=True) -> dict:
    return {"sub": "uid-1", "email": email, "email_verified": verified}


@pytest.fixture
def client(monkeypatch):
    for name in ("EQUITI_AUTH", "K_SERVICE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FIREBASE_PROJECT_ID", PROJECT)
    monkeypatch.setenv("EQUITI_ALLOWED_EMAILS", "Owner@Example.com, friend@example.com")
    return TestClient(main.app, raise_server_exceptions=False)


def _signed_in(monkeypatch, claims=None, error=None):
    seen = {}

    def verify(token, project_id):
        seen.update(token=token, project_id=project_id)
        if error:
            raise error
        return claims or _claims()

    monkeypatch.setattr(auth, "verify", verify)
    return seen


def _me(client, token="t0ken"):
    return client.get("/api/me", headers={"Authorization": f"Bearer {token}"})


def test_auth_is_on_unless_switched_off(client):
    assert client.get("/api/me").status_code == 401


def test_no_token_is_a_401_that_asks_for_one(client):
    response = client.get("/api/momentum/AAPL")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_an_allowed_verified_email_passes(client, monkeypatch):
    seen = _signed_in(monkeypatch)
    response = _me(client)
    assert response.status_code == 200
    assert response.json() == {"auth_enabled": True, "uid": "uid-1", "email": "owner@example.com"}
    assert seen == {"token": "t0ken", "project_id": PROJECT}


def test_a_valid_token_for_someone_else_is_a_403(client, monkeypatch):
    _signed_in(monkeypatch, _claims(email="stranger@example.com"))
    assert _me(client).status_code == 403


def test_an_unverified_email_is_not_trusted_even_on_the_list(client, monkeypatch):
    _signed_in(monkeypatch, _claims(verified=False))
    assert _me(client).status_code == 403


def test_an_empty_allowlist_admits_nobody(client, monkeypatch):
    monkeypatch.setenv("EQUITI_ALLOWED_EMAILS", "")
    _signed_in(monkeypatch)
    assert _me(client).status_code == 403


def test_a_bad_or_expired_token_is_a_401(client, monkeypatch):
    _signed_in(monkeypatch, error=ValueError("Token expired"))
    response = _me(client)
    assert response.status_code == 401
    assert "Token expired" not in response.text


def test_google_unreachable_is_a_503_not_a_rejection(client, monkeypatch):
    from google.auth import exceptions

    _signed_in(monkeypatch, error=exceptions.TransportError("timeout"))
    assert _me(client).status_code == 503


def test_a_missing_project_fails_closed(client, monkeypatch):
    monkeypatch.delenv("FIREBASE_PROJECT_ID")
    _signed_in(monkeypatch)
    assert _me(client).status_code == 503


def test_switching_auth_off_works_locally(client, monkeypatch):
    monkeypatch.setenv("EQUITI_AUTH", "off")
    assert client.get("/api/me").json()["auth_enabled"] is False


def test_switching_auth_off_is_ignored_on_cloud_run(client, monkeypatch):
    monkeypatch.setenv("EQUITI_AUTH", "off")
    monkeypatch.setenv("K_SERVICE", "equiti-api")
    assert client.get("/api/me").status_code == 401


def test_health_and_the_schema_need_no_sign_in(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/openapi.json").status_code == 200


def test_every_data_route_carries_the_gate():
    """A router mounted without ``require_user`` would serve licensed data openly."""
    open_paths = {"/api/health", "/api/openapi.json", "/api/docs", "/api/docs/oauth2-redirect"}
    for route in main.app.routes:
        if getattr(route, "path", None) in open_paths or not hasattr(route, "dependant"):
            continue
        calls = {dep.call for dep in route.dependant.dependencies}
        assert auth.require_user in calls, route.path
