"""The options store reader on its local backend, and the freshness arithmetic."""

from __future__ import annotations

import json
import os
from datetime import date

import pandas as pd
import pytest

from app.data import options_store
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
    monkeypatch.delenv("OPTIONS_STORE_URI", raising=False)
    monkeypatch.setattr(options_store.st, "secrets", {})
    with pytest.raises(options_store.OptionsStoreError, match="OPTIONS_STORE_URI"):
        options_store.fetch_manifest()


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
