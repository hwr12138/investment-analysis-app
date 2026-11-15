"""The JSON API the web frontend (``web/``) reads — the app's one front end.

It is the top layer: it fetches through ``app/data``, hands the frames to
``app/logic``, and shapes what comes back for display. What it never does is
arithmetic of its own; that stays in the logic layer, where it can be tested
without a request.

One router module per page, each mounted under ``/api`` by :mod:`app.api.main`.
Every response model derives from :class:`app.api.contract.Contract`, which
refuses NaN and infinity at construction — see there for why.
"""
