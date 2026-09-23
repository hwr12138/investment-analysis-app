"""Logic for the Stock Research page — one module per tab, plus what they share.

Split across modules rather than the repo's usual single
``app/logic/<feature>.py`` because the page carries ten tabs: leaving them in
one file would reproduce the problem ``app/logic/dcf/`` exists to avoid, at
roughly the same scale.

The layout is deliberately flat and the dependencies run one way:

    comparison  ←  summary, valuation, growth  ←  dataset

Number formatting is shared by every page, not just this one, so it lives at
``app/formatting.py``. ``comparison`` is shared by the two tabs that
answer a metric against its peers and against its own past — ``valuation`` and
``growth``, the ⚖️ tab's sub-tabs — and holds the record they both produce, the
median-and-percentile summary, and the peer universe join. Beyond those, each tab
module owns its own arithmetic and knows nothing of its siblings.

``dataset`` sits on top because it assembles the snapshot the tabs read, calling
into ``summary`` for the key statistics and ``comparison`` for the peer universe.
Those are the only upward edges, and both exist for the same reason: the snapshot
is built once per search, so work the tabs would otherwise repeat on every rerun
happens there.

What ``comparison`` deliberately does *not* hold is either sub-tab's notion of a
usable number, because they are opposites. A valuation multiple must be positive
to mean anything; a growth rate is meaningful at any sign but needs a positive
*base*, or a narrowing loss reads as a collapse. Nor does it hold their
tolerances: one counts trading sessions and the other filed quarters.

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
