"""The options store reader on its local backend, and the freshness arithmetic."""

from __future__ import annotations

import json
import os
from datetime import date

import pandas as pd
import pytest

from app.data import options_store, settings
from app import options_layout as layout
from app.logic.options import freshness


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(options_store, "_location", lambda: f"file://{tmp_path}")
    return tmp_path


def _write_frame(root, path, frame):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target)
    return target


def test_reads_a_frame_and_the_manifest(store):
    _write_frame(store, layout.METRICS, pd.DataFrame({"a": [1, 2]}))
    (store / layout.MANIFEST).write_text(json.dumps({"last_session": "2026-10-23"}))
    assert options_store.fetch_frame(layout.METRICS)["a"].tolist() == [1, 2]
    assert options_store.fetch_manifest()["last_session"] == "2026-10-23"


def test_a_new_generation_is_read_afresh(store):
    target = _write_frame(store, layout.METRICS, pd.DataFrame({"a": [1]}))
    assert options_store.fetch_frame(layout.METRICS)["a"].tolist() == [1]
    _write_frame(store, layout.METRICS, pd.DataFrame({"a": [2]}))
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    options_store._cached_generation.clear()  # the five-minute TTL, elapsed
    assert options_store.fetch_frame(layout.METRICS)["a"].tolist() == [2]


def test_a_missing_object_raises_and_is_not_cached(store):
    with pytest.raises(options_store.OptionsStoreError, match="not in the options store"):
        options_store.fetch_frame(layout.METRICS)
    _write_frame(store, layout.METRICS, pd.DataFrame({"a": [3]}))
    assert options_store.fetch_frame(layout.METRICS)["a"].tolist() == [3]


def test_an_unconfigured_store_says_how_to_configure_it(monkeypatch):
    # conftest points the secrets file at nothing; the environment is ours to clear.
    monkeypatch.delenv("OPTIONS_STORE_URI", raising=False)
    monkeypatch.delenv("OPTIONS_BUCKET", raising=False)
    with pytest.raises(options_store.OptionsStoreError, match="OPTIONS_STORE_URI"):
        options_store.fetch_manifest()


def test_the_secrets_file_names_the_store(monkeypatch, tmp_path):
    """Local development: the store's location in .streamlit/secrets.toml."""
    monkeypatch.delenv("OPTIONS_STORE_URI", raising=False)
    monkeypatch.delenv("OPTIONS_BUCKET", raising=False)
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('OPTIONS_BUCKET = "from-file"\n')
    monkeypatch.setattr(settings, "SECRETS_FILE", secrets)
    assert options_store._location() == "gs://from-file"
    monkeypatch.setenv("OPTIONS_STORE_URI", "file:///from-env")
    assert options_store._location() == "file:///from-env"  # the environment wins


def test_a_service_account_table_is_read_from_the_secrets_file(monkeypatch, tmp_path):
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('[gcp_service_account]\ntype = "service_account"\nproject_id = "p"\n')
    monkeypatch.setattr(settings, "SECRETS_FILE", secrets)
    assert options_store._service_account_info() == {"type": "service_account", "project_id": "p"}


def test_without_a_key_the_bucket_uses_the_ambient_account(monkeypatch):
    """Cloud Run has no secrets.toml (conftest points the file at nothing)."""
    from google.cloud import storage

    made = {}

    class Client:
        def __init__(self, **kwargs):
            made.update(kwargs)

        def bucket(self, name):
            return ("bucket", name)

    monkeypatch.setattr(storage, "Client", Client)
    options_store._gcs_bucket.clear()
    try:
        assert options_store._gcs_bucket("options-store") == ("bucket", "options-store")
        assert made == {}  # no credentials passed: Application Default Credentials
    finally:
        options_store._gcs_bucket.clear()


@pytest.mark.parametrize(
    "last, today, behind",
    [
        (date(2026, 10, 23), date(2026, 10, 24), 0),  # Saturday: Friday is the latest
        (date(2026, 10, 23), date(2026, 10, 26), 0),  # Monday before the run: still Friday
        (date(2026, 10, 23), date(2026, 10, 27), 1),  # Tuesday: Monday is missing
        (date(2026, 10, 21), date(2026, 10, 27), 3),
    ],
)
def test_sessions_behind(last, today, behind):
    assert freshness.sessions_behind(last, today) == behind
