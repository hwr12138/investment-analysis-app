"""The writer's side of the options store: a local directory or a GCS bucket.

The page reads through ``app.data.options_store``; the collector cannot import
that (it is the data layer and imports Streamlit), so the handful of operations
it needs are here. On Cloud Run the GCS client authenticates as the job's
service account; no key file is involved.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd


class Store:
    def read_bytes(self, path: str) -> bytes | None:
        raise NotImplementedError

    def write_bytes(self, path: str, data: bytes) -> None:
        raise NotImplementedError

    def list(self, prefix: str) -> list[str]:
        """Every object path under ``prefix``, sorted."""
        raise NotImplementedError

    def read_frame(self, path: str) -> pd.DataFrame | None:
        data = self.read_bytes(path)
        return None if data is None else pd.read_parquet(io.BytesIO(data))

    def write_frame(self, path: str, frame: pd.DataFrame) -> None:
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        self.write_bytes(path, buffer.getvalue())

    def read_json(self, path: str) -> dict | None:
        data = self.read_bytes(path)
        return None if data is None else json.loads(data)

    def write_json(self, path: str, payload: dict) -> None:
        self.write_bytes(path, json.dumps(payload, indent=2, sort_keys=True, default=str).encode())


class LocalStore(Store):
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def read_bytes(self, path: str) -> bytes | None:
        target = self.root / path
        return target.read_bytes() if target.exists() else None

    def write_bytes(self, path: str, data: bytes) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)  # a reader never sees half a file

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file() and p.suffix != ".tmp")


class GcsStore(Store):
    def __init__(self, bucket: str):
        from google.cloud import storage

        self._bucket = storage.Client().bucket(bucket)

    def read_bytes(self, path: str) -> bytes | None:
        blob = self._bucket.get_blob(path)
        return None if blob is None else blob.download_as_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        self._bucket.blob(path).upload_from_string(data)

    def list(self, prefix: str) -> list[str]:
        return sorted(blob.name for blob in self._bucket.list_blobs(prefix=prefix))


def open_store(uri: str) -> Store:
    scheme, _, rest = uri.partition("://")
    if scheme == "file" and rest:
        return LocalStore(rest)
    if scheme == "gs" and rest:
        return GcsStore(rest.strip("/"))
    raise ValueError(f"Unsupported store URI {uri!r}: use gs://bucket or file:///abs/path")
