"""The Sharadar client's implementation. Import the public API from
``app.data.sharadar_client``; the logic layer imports only ``vocabulary``.

Deliberately empty: anything imported here would load for every submodule,
including ``vocabulary``, whose purity is the point of the split.
"""
