"""The FastAPI application: ``uvicorn app.api.main:app``.

Everything is served under ``/api`` because Firebase Hosting forwards exactly
that prefix to this service (``firebase.json``) and serves the frontend for
every other path — so the same URLs work locally behind Vite's proxy.

Data routes carry :func:`app.api.auth.require_user`; only ``/api/health`` and
the schema are open. A Sharadar failure becomes a JSON error with the client's
own message, which names the table, ticker and status but never the key; an
options store that cannot be read is a 503 naming the object, never the
credentials.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api import momentum, options, portfolio, rates, research
from app.api.auth import User, require_user
from app.api.contract import Contract
from app.data import options_store, sharadar_client

app = FastAPI(
    title="Equiti API",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

_signed_in = [Depends(require_user)]
for _router in (momentum.router, options.router, portfolio.router, rates.router, research.router):
    app.include_router(_router, prefix="/api", dependencies=_signed_in)


class Health(Contract):
    status: str


class Me(Contract):
    auth_enabled: bool
    uid: str | None
    email: str | None


@app.get("/api/health", tags=["meta"])
def health() -> Health:
    return Health(status="ok")


@app.get("/api/me", tags=["meta"])
def me(user: User | None = Depends(require_user)) -> Me:
    """Who the token belongs to — the frontend's check that sign-in admits it."""
    if user is None:
        return Me(auth_enabled=False, uid=None, email=None)
    return Me(auth_enabled=True, uid=user.uid, email=user.email)


@app.exception_handler(sharadar_client.TickerNotFoundError)
def _ticker_not_found(_: Request, error: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(error)}, status_code=status.HTTP_404_NOT_FOUND)


@app.exception_handler(sharadar_client.SharadarError)
def _sharadar_failed(_: Request, error: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(error)}, status_code=status.HTTP_502_BAD_GATEWAY)


@app.exception_handler(options_store.OptionsStoreError)
def _options_store_failed(_: Request, error: Exception) -> JSONResponse:
    return JSONResponse({"detail": str(error)}, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
