"""Running independent fetches at the same time.

A research search used to make its requests one after another — 23 of them,
27.6s on MSFT, of which 27.3s was waiting on the network. Almost none depend
on each other, so they can wait together. This lives in the data layer because
the one Streamlit detail it has to handle is a data-layer concern: the fetchers
are ``st.cache_data`` functions, and a worker thread without the script's run
context logs a warning on every call into one.

Nothing here decides what may fail. :func:`gather` re-raises the first failure
in submission order once every call has finished, so a caller that tolerates a
fetch failing wraps that fetch before handing it over, exactly as it would have
wrapped the sequential call.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

#: Nasdaq Data Link allows far more than this per second; the cap is about not
#: opening a connection per fetch when a caller hands over a long list.
MAX_WORKERS = 6


def gather(
    calls: Mapping[str, Callable[[], Any]], *, max_workers: int = MAX_WORKERS
) -> dict[str, Any]:
    """Runs every zero-argument call concurrently and returns results by name.

    Waits for all of them even when one fails, so no request is left running
    behind a raised exception, then re-raises the first failure in the order the
    calls were given — the same exception a sequential run would have hit first.
    """
    if not calls:
        return {}
    if len(calls) == 1:
        ((name, call),) = calls.items()
        return {name: call()}

    context = get_script_run_ctx(suppress_warning=True)

    def attach() -> None:
        if context is not None:
            add_script_run_ctx(threading.current_thread(), context)

    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(calls)), initializer=attach
    ) as pool:
        futures = {name: pool.submit(call) for name, call in calls.items()}
        outcomes = {name: _settle(future) for name, future in futures.items()}

    for value, error in outcomes.values():
        if error is not None:
            raise error
    return {name: value for name, (value, _) in outcomes.items()}


def _settle(future) -> tuple[Any, BaseException | None]:
    try:
        return future.result(), None
    except Exception as error:  # noqa: BLE001 - re-raised by the caller in order
        return None, error
