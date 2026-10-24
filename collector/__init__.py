"""The nightly options collector — a Cloud Run job, not part of the Streamlit app.

It fetches end-of-day option chains, closes and the Treasury curve from
ThetaData, computes one metrics row per name per session with
``app.logic.options``, and writes everything to the options store the page reads
(``app.options_layout`` names the objects). It imports nothing else of the app:
``tests/test_architecture.py`` checks it never reaches ``app.data``,
``app.views`` or Streamlit, so its image installs neither.

Run ``python -m collector.main --help``.
"""
