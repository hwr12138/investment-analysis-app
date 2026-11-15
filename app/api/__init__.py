"""The JSON API the web frontend (``web/``) reads — the successor to ``app/views``.

It plays the part a Streamlit view plays today: it fetches through ``app/data``,
hands the frames to ``app/logic``, and shapes what comes back for display. What
it never does is arithmetic of its own; that stays in the logic layer, which is
why the Streamlit page and the API can share it while both exist.

One router module per page, each mounted under ``/api`` by :mod:`app.api.main`.
Every response model derives from :class:`app.api.contract.Contract`, which
refuses NaN and infinity at construction — see there for why.

The data layer still caches through ``st.cache_data``. That works outside a
Streamlit server (the cache is in-process either way) but logs a warning per
cached function at import and per call, so Streamlit's logging is turned down
here, before any submodule can import the data layer. This is the package's one
Streamlit import, and it goes once the caches move off Streamlit.
"""

from streamlit import logger as _streamlit_logger

_streamlit_logger.set_log_level("ERROR")
