"""Reads what the nightly options collector wrote.

The 🎯 Options page fetches nothing from a market-data vendor. ThetaData's free
tier serves one request at a time at one to two seconds a session, so a page
that fetched the chains it draws would take minutes; instead a Cloud Run job
(``collector/``) writes them to a private GCS bucket overnight and this module
reads the bucket. :mod:`app.options_layout` names every object.

Where the store is comes from the ``OPTIONS_STORE_URI`` environment variable, else
Streamlit secrets:

* ``OPTIONS_STORE_URI = "gs://bucket"`` or ``"file:///abs/path"`` — the local
  form is for development and tests, and needs no credentials;
* or ``OPTIONS_BUCKET = "bucket"`` with a ``[gcp_service_account]`` table holding
  a read-only service account's key.

Without a key the GCS client authenticates as whatever the environment
provides — on Cloud Run, the API service's own account, which needs only
``roles/storage.objectViewer`` on the bucket.

**Two cache layers, keyed so a new night is picked up and an old one never
re-downloaded.** Each object's *generation* (GCS's version stamp, or a local
file's mtime) is read with a five-minute TTL, and the object itself is cached
under ``(path, generation)``. **Failures raise :class:`OptionsStoreError` and are
never cached** — Streamlit does not cache an exception — for the reason
``market_data.FetchFailed`` gives: a transient outage cached for a day.

Error messages name the object and the store, never the credentials.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from app import options_layout as layout


class OptionsStoreError(Exception):
    """The options store could not be read. Carries no credentials."""


def _location() -> str:
    # The environment wins, so a local store can be pointed at for one run
    # without editing secrets.toml.
    uri = os.environ.get("OPTIONS_STORE_URI")
    bucket = None
    if not uri:
        try:
            uri = st.secrets.get("OPTIONS_STORE_URI")
            bucket = st.secrets.get("OPTIONS_BUCKET")
        except Exception:  # secrets.toml absent entirely
            uri = bucket = None
    if uri:
        return str(uri)
    if bucket:
        return f"gs://{bucket}"
    raise OptionsStoreError(
        "The options store is not configured. Set OPTIONS_STORE_URI (gs://… or file:///…) "
        "or OPTIONS_BUCKET with a [gcp_service_account] in .streamlit/secrets.toml."
    )


@st.cache_resource(show_spinner=False)
def _gcs_bucket(name: str):
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError:
        raise OptionsStoreError("google-cloud-storage is not installed") from None
    info = _service_account_info()
    try:
        if info:
            credentials = service_account.Credentials.from_service_account_info(info)
            client = storage.Client(credentials=credentials, project=credentials.project_id)
        else:
            # Application Default Credentials: on Cloud Run, the service's own account.
            client = storage.Client()
    except Exception:
        raise OptionsStoreError(f"Could not authenticate to the options bucket gs://{name}") from None
    return client.bucket(name)


def _service_account_info() -> dict | None:
    """The key from ``[gcp_service_account]``, or ``None`` to use the ambient account.

    Cloud Run has no ``secrets.toml`` at all, and reading ``st.secrets`` without
    one raises rather than returning nothing.
    """
    try:
        info = st.secrets.get("gcp_service_account")
    except Exception:
        return None
    return dict(info) if info else None


def _split(location: str) -> tuple[str, str]:
    scheme, _, rest = location.partition("://")
    if scheme not in ("gs", "file") or not rest:
        raise OptionsStoreError(f"Unsupported options store location scheme {scheme!r}")
    return scheme, rest


def _generation(path: str) -> str:
    scheme, rest = _split(_location())
    if scheme == "file":
        target = Path(rest) / path
        if not target.exists():
            raise OptionsStoreError(f"{path} is not in the options store yet")
        return str(target.stat().st_mtime_ns)
    try:
        blob = _gcs_bucket(rest).get_blob(path)
    except OptionsStoreError:
        raise
    except Exception:
        raise OptionsStoreError(f"Could not reach the options store for {path}") from None
    if blob is None:
        raise OptionsStoreError(f"{path} is not in the options store yet")
    return str(blob.generation)


def _read(path: str, generation: str) -> bytes:
    scheme, rest = _split(_location())
    if scheme == "file":
        try:
            return (Path(rest) / path).read_bytes()
        except OSError:
            raise OptionsStoreError(f"Could not read {path} from the options store") from None
    try:
        return _gcs_bucket(rest).blob(path, generation=int(generation)).download_as_bytes()
    except Exception:
        raise OptionsStoreError(f"Could not read {path} from the options store") from None


@st.cache_data(show_spinner=False, ttl=300)
def _cached_generation(path: str) -> str:
    return _generation(path)


@st.cache_data(show_spinner=False, ttl=3600 * 36, max_entries=32)
def _cached_frame(path: str, generation: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_read(path, generation)))


@st.cache_data(show_spinner=False, ttl=3600 * 36, max_entries=8)
def _cached_json(path: str, generation: str) -> dict:
    return json.loads(_read(path, generation))


def fetch_manifest() -> dict:
    """The last complete run's manifest; raises :class:`OptionsStoreError`."""
    return _cached_json(layout.MANIFEST, _cached_generation(layout.MANIFEST))


def fetch_frame(path: str) -> pd.DataFrame:
    """A Parquet object from the store, by its :mod:`~app.options_layout` name."""
    return _cached_frame(path, _cached_generation(path))
