"""Who may call the API.

The data behind it is licensed — Sharadar via Nasdaq Data Link, ThetaData for
options — and a JSON API is far easier to scrape than a Streamlit page. Firebase
Hosting can only reach a Cloud Run service that allows unauthenticated
invocations, so the gate cannot be IAM: every data route requires a Firebase ID
token, and the token's verified email must be on an allowlist.

Configured from the environment, read on every request so tests can set it:

* ``EQUITI_AUTH`` — ``firebase`` (the default) or ``off``. ``off`` is honoured
  only off Cloud Run (no ``K_SERVICE``), so a deploy cannot be opened by a
  stray variable.
* ``FIREBASE_PROJECT_ID`` — the project whose tokens are accepted.
* ``EQUITI_ALLOWED_EMAILS`` — comma-separated. Google sign-in admits any Google
  account, so an empty list admits **nobody**: the gate fails closed.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth import exceptions, transport
from google.auth.transport.requests import Request as _GoogleRequest
from google.oauth2 import id_token

#: Google's signing certificates rotate every few days and are served with a
#: max-age of hours; an hour here keeps verification off the network.
CERTS_TTL_SECONDS = 3600

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class User:
    uid: str
    email: str | None


@dataclass(frozen=True)
class AuthSettings:
    enabled: bool
    project_id: str | None
    allowed_emails: frozenset[str]


def settings() -> AuthSettings:
    on_cloud_run = bool(os.environ.get("K_SERVICE"))
    requested_off = os.environ.get("EQUITI_AUTH", "firebase").strip().lower() == "off"
    allowed = os.environ.get("EQUITI_ALLOWED_EMAILS", "")
    return AuthSettings(
        enabled=on_cloud_run or not requested_off,
        project_id=os.environ.get("FIREBASE_PROJECT_ID") or None,
        allowed_emails=frozenset(
            email.strip().lower() for email in allowed.split(",") if email.strip()
        ),
    )


class _CachingRequest(transport.Request):
    """google-auth's transport, remembering successful GETs for a while.

    The verifier fetches the signing certificates on every call; without this
    each API request would pay a round trip to Google before doing any work.
    """

    def __init__(self, ttl: float = CERTS_TTL_SECONDS) -> None:
        self._inner = _GoogleRequest()
        self._ttl = ttl
        self._cache: dict[str, tuple[float, transport.Response]] = {}
        self._lock = threading.Lock()

    def __call__(self, url, method="GET", body=None, headers=None, timeout=None, **kwargs):
        if method == "GET":
            with self._lock:
                hit = self._cache.get(url)
            if hit and hit[0] > time.monotonic():
                return hit[1]
        response = self._inner(
            url, method=method, body=body, headers=headers, timeout=timeout, **kwargs
        )
        if method == "GET" and response.status == 200:
            with self._lock:
                self._cache[url] = (time.monotonic() + self._ttl, response)
        return response


_request = _CachingRequest()


def verify(token: str, project_id: str) -> dict:
    """The token's claims, or ``ValueError`` if it is not a live token for ``project_id``.

    Checks the signature, expiry, audience, issuer and subject — everything
    ``verify_firebase_token`` checks. The one indirection tests replace.
    """
    return id_token.verify_firebase_token(
        token, _request, audience=project_id, clock_skew_in_seconds=10
    )


def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> User | None:
    """The FastAPI dependency every data route carries."""
    config = settings()
    if not config.enabled:
        return None
    if not config.project_id:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "The API is not configured: FIREBASE_PROJECT_ID is not set.",
        )
    if credentials is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Sign in to use this API.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        claims = verify(credentials.credentials, config.project_id)
    except exceptions.TransportError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Could not reach Google to check your sign-in. Try again shortly.",
        ) from None
    except ValueError:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Your sign-in is invalid or has expired. Sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    email = (claims.get("email") or "").lower()
    if not (claims.get("email_verified") and email in config.allowed_emails):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "This account is not allowed to use Equiti.",
        )
    return User(uid=claims["sub"], email=email)
