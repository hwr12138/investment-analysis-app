"""The 🎯 Options page's arithmetic — pure, shared by the page and the collector.

Two consumers import this package and nothing else of the app's: the Streamlit
views, which compute the snapshot screens from the latest stored chain, and the
nightly collector (``collector/``), which computes the per-session metrics the
history panels read. That is why it may import no Streamlit, no network client,
no ``thetadata`` and no Google SDK — ``tests/test_architecture.py`` enforces it,
and the collector's image does not install Streamlit at all.

The pipeline for one symbol on one session:

    chain  →  surface.build_surface  →  screens.*   (what to buy today)
                                    →  history.session_metrics  (one row a day)
    metrics rows  →  history.own_history_stats, score.buy_vol_score

Three conventions run through every module:

* **Time is calendar days ÷ 365**, the convention the quoted IVs are solved in.
  Annualising a *cost* uses trading days instead (``screens.trading_days``),
  because that is what reproduces the screenshot this page was built from.
* **Mids, never last trades.** EOD ``close`` is the last trade of the day and on
  most strikes is hours stale; the 17:15 ET NBBO is what a quote means.
* **Missing stays missing.** A strike outside the quoted range, an expiry with
  too few quotes, a percentile over too short a history — each is ``None`` or
  NaN and the view says so. Nothing is extrapolated or filled.
"""
