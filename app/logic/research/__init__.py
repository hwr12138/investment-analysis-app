"""Logic for the Stock Research page — one module per tab, plus what they share.

Split across modules rather than the repo's usual single
``app/logic/<feature>.py`` because the page carries ten tabs: leaving them in
one file would reproduce the problem ``app/logic/dcf/`` exists to avoid, at
roughly the same scale.

The layout is deliberately flat and the dependencies run one way:

    formatting  ←  summary, valuation  ←  dataset

``formatting`` is shared by every tab. Each tab module owns its own arithmetic
and depends on nothing but ``formatting`` and the data layer — ``summary`` and
``valuation`` do not know about each other. ``dataset`` sits on top because it
assembles the snapshot the tabs read, which means calling into ``summary`` for
the key statistics and ``valuation`` for the peer universe. Those are the only
upward edges, and both exist for the same reason: the snapshot is built once per
search, so work the tabs would otherwise repeat on every rerun happens there.

``valuation`` also carries a name collision worth knowing about: it is unrelated
to ``app/logic/dcf/valuation.py``. This one compares a company's multiples to its
peers; that one discounts cash flows. Their test files disambiguate with a
``test_research_`` prefix.

Nothing here imports Streamlit. Two modules deliberately live *outside* this
package: ``app/logic/statements.py``, which the financials tab consumes but does
not own, so a second consumer needs no move, and ``app/logic/dcf/``, which stays
independently reusable so batch screening remains an extension rather than a
rewrite.
"""
