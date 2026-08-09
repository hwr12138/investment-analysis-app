"""Logic for the Stock Research page — one module per tab, plus what they share.

Split across modules rather than the repo's usual single
``app/logic/<feature>.py`` because the page carries ten tabs: leaving them in
one file would reproduce the problem ``app/logic/dcf/`` exists to avoid, at
roughly the same scale.

The layout is deliberately flat and the dependencies run one way:

    formatting  ←  summary  ←  dataset

``formatting`` is shared by every tab. Each tab module owns its own arithmetic
and depends on nothing but ``formatting`` and the data layer. ``dataset`` sits
on top because it assembles the snapshot the tabs read, which today means
calling into ``summary`` to build the key statistics — the one upward edge, and
the seam described in that module's docstring.

Nothing here imports Streamlit. Two modules deliberately live *outside* this
package: ``app/logic/statements.py``, because the DCF tab's reported-figures
table is its intended second consumer and it is not owned by this page, and
``app/logic/dcf/``, which stays independently reusable so batch screening
remains an extension rather than a rewrite.
"""
