"""The writer's side of the options store: a local directory or a GCS bucket.

The page reads through ``app.data.options_store``; the collector cannot import
that (it is the data layer and imports Streamlit), so the handful of operations
it needs are here. On Cloud Run the GCS client authenticates as the job's
service account; no key file is involved.

The conditional operations (:meth:`Store.create`, :meth:`Store.replace`,
:meth:`Store.delete` with a token) are what :mod:`collector.lease` is built on.
A token names one version of an object: GCS's generation, which the server
checks atomically; on disk the inode and modification time, checked and then
acted on — enough for one machine's local runs, not a lock between machines.
"""

from __future__ import annotations

import io
import json
import os
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

    def create(self, path: str, data: bytes) -> object | None:
        """Write ``path`` only if it does not exist: its token, or ``None`` if it did."""
        raise NotImplementedError

    def read_with_token(self, path: str) -> tuple[bytes, object] | None:
        raise NotImplementedError

    def replace(self, path: str, data: bytes, token: object) -> object | None:
        """Overwrite ``path`` only if it is still the version ``token`` names: the new token, or ``None``."""
        raise NotImplementedError

    def delete(self, path: str, token: object | None = None) -> bool:
        """Remove ``path`` (only the version ``token`` names, if one is given). ``False`` if it was not there."""
        raise NotImplementedError

    def reopen(self) -> Store:
        """An independent handle on the same store, for another thread."""
        raise NotImplementedError


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

    @staticmethod
    def _token(target: Path) -> str | None:
        try:
            stat = target.stat()
        except FileNotFoundError:
            return None
        return f"{stat.st_ino}:{stat.st_mtime_ns}"

    def create(self, path: str, data: bytes) -> str | None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return None
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return self._token(target)

    def read_with_token(self, path: str) -> tuple[bytes, str] | None:
        target = self.root / path
        token = self._token(target)
        if token is None:
            return None
        try:
            return target.read_bytes(), token
        except FileNotFoundError:
            return None

    def replace(self, path: str, data: bytes, token: object) -> str | None:
        target = self.root / path
        if self._token(target) != token:
            return None
        self.write_bytes(path, data)  # a new inode, so the old token no longer matches
        return self._token(target)

    def delete(self, path: str, token: object | None = None) -> bool:
        target = self.root / path
        if token is not None and self._token(target) != token:
            return False
        try:
            target.unlink()
        except FileNotFoundError:
            return False
        return True

    def reopen(self) -> LocalStore:
        return LocalStore(self.root)


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

    def create(self, path: str, data: bytes) -> int | None:
        from google.api_core.exceptions import PreconditionFailed

        blob = self._bucket.blob(path)
        try:
            blob.upload_from_string(data, if_generation_match=0)  # 0: only if absent
        except PreconditionFailed:
            return None
        return blob.generation

    def read_with_token(self, path: str) -> tuple[bytes, int] | None:
        from google.api_core.exceptions import NotFound, PreconditionFailed

        blob = self._bucket.get_blob(path)
        if blob is None:
            return None
        try:
            return blob.download_as_bytes(if_generation_match=blob.generation), blob.generation
        except (NotFound, PreconditionFailed):  # replaced or removed in between
            return None

    def replace(self, path: str, data: bytes, token: object) -> int | None:
        from google.api_core.exceptions import NotFound, PreconditionFailed

        blob = self._bucket.blob(path)
        try:
            blob.upload_from_string(data, if_generation_match=token)
        except (NotFound, PreconditionFailed):
            return None
        return blob.generation

    def delete(self, path: str, token: object | None = None) -> bool:
        from google.api_core.exceptions import NotFound, PreconditionFailed

        try:
            self._bucket.blob(path).delete(if_generation_match=token)
        except (NotFound, PreconditionFailed):
            return False
        return True

    def reopen(self) -> GcsStore:
        return GcsStore(self._bucket.name)  # a client of its own: one is not shared across threads


def open_store(uri: str) -> Store:
    scheme, _, rest = uri.partition("://")
    if scheme == "file" and rest:
        return LocalStore(rest)
    if scheme == "gs" and rest:
        return GcsStore(rest.strip("/"))
    raise ValueError(f"Unsupported store URI {uri!r}: use gs://bucket or file:///abs/path")
