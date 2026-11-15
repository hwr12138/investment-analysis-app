"""Keys and settings: the environment first, then ``.streamlit/secrets.toml``.

On Cloud Run everything comes from the environment — Secret Manager hands
``SHARADAR_API_KEY`` over as a variable, and the deploy sets
``OPTIONS_STORE_URI`` — and there is no secrets file at all (``.gcloudignore``
keeps ``.streamlit/`` out of the build). Locally the same names are read from
``.streamlit/secrets.toml`` at the repository root, the file the Streamlit front
end used; the directory keeps its name so an existing checkout keeps working.

The file is parsed with :mod:`tomllib` and re-read when it changes. Nothing here
puts a value in an exception or a log line: a malformed file is reported by its
path alone, because a TOML parser's message can quote the line it stopped on.
"""

from __future__ import annotations

import functools
import os
import tomllib
from pathlib import Path
from typing import Any

#: The local secrets file. Anchored on the package rather than the working
#: directory, so a server or a test run started anywhere reads the same file.
SECRETS_FILE = Path(__file__).resolve().parents[2] / ".streamlit" / "secrets.toml"


class SettingsError(Exception):
    """The secrets file exists but cannot be read. Carries its path, never its contents."""


def get(name: str) -> str | None:
    """A top-level setting: the environment's value, else the file's, else ``None``.

    An empty environment variable counts as unset, so it does not hide the file.
    """
    value = os.environ.get(name)
    if value:
        return value
    value = _file().get(name)
    if value is None or isinstance(value, dict):
        return None
    return str(value) or None


def table(name: str) -> dict[str, Any] | None:
    """A ``[table]`` from the secrets file — the environment cannot hold one."""
    value = _file().get(name)
    return dict(value) if isinstance(value, dict) and value else None


def _file() -> dict[str, Any]:
    path = SECRETS_FILE
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:  # no file: Cloud Run, CI, a fresh checkout
        return {}
    return _parse(path, stamp)


@functools.lru_cache(maxsize=4)
def _parse(path: Path, stamp: int) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        raise SettingsError(f"{path} could not be read as TOML") from None
