"""The ⚖️ Valuation tab: two sub-tabs over one peer universe.

*Valuation* asks what the market is paying for the company today; *Growth* asks
how fast the thing being paid for is moving. Both answer per metric in the same
three columns — current, the company's own five-year median, the industry median
— so a reader crossing between them is reading one shape twice.

```
__init__.py    this shell: the stylesheet, the sub-tab strip, the dispatch
table.py       the grid and the four cells the two sub-tabs render identically
multiples.py   the Valuation body
growth.py      the Growth body
```

A package rather than one module because the two bodies have no reason to know
about each other: they share a table and a universe, not arithmetic.

**Fetches nothing.** ``st.tabs`` renders every body on every rerun and reports
nothing to Python about which is open — and this tab now nests a second strip
inside the first, so *both* sub-tab bodies are built whenever anything on the
page changes. That is affordable only because both are pure computation over the
snapshot: the peer universe arrives once per search, keyed on a date rather than
a ticker, and switching sub-tabs touches no network.

The stylesheet is injected **here, once**, rather than by each body. One tab, one
stylesheet, however many sub-tabs it holds; injecting it twice would leave the
second copy fighting the first for no benefit.
"""

from __future__ import annotations

import streamlit as st

from app.logic.research.dataset import ResearchDataset
from app.views.research.valuation import growth, multiples

#: Sub-tabs in display order. Valuation opens because it is the question the tab
#: is named for; Growth reads as the follow-up to it.
_SUB_TABS = ("Valuation", "Growth")

_VALUATION_CSS = """
<style>
/* `table-layout: fixed` holds the colgroup widths; without it the browser
   reflows the grid per company as the numbers change length. Doubled selector
   to outrank Streamlit's own hashed table rules. */
.val-table.val-table {
    width: 100%; border-collapse: collapse; table-layout: fixed;
    margin: 0.25rem 0 0.5rem 0; font-variant-numeric: tabular-nums;
}
.val-table th {
    font-size: 0.72rem; font-weight: 600; opacity: 0.6; text-align: right;
    padding: 0.35rem 0.7rem; border-bottom: 1px solid rgba(250, 250, 250, 0.22);
    text-transform: uppercase; letter-spacing: 0.04em;
}
.val-table th.val-metric, .val-table td.val-metric { text-align: left; }
.val-table td {
    padding: 0.5rem 0.7rem; text-align: right; vertical-align: middle;
    border-top: 1px solid rgba(250, 250, 250, 0.1); font-size: 0.9rem;
}
.val-table tr:last-child td { border-bottom: 1px solid rgba(250, 250, 250, 0.14); }
.val-metric { font-size: 0.85rem; font-weight: 600; }
.val-current { font-weight: 700; }
/* Missing values are stated, never blank: an empty cell reads as zero. */
.val-missing { font-weight: 400; font-style: italic; opacity: 0.42; font-size: 0.82rem; }
.val-sub { font-size: 0.7rem; opacity: 0.5; display: block; }
/* Named for the valuation reading — dear and cheap — and reused by Growth for
   slower and faster. One pair of colours, because the reader is being told the
   same thing either way: green is the direction they would rather see. */
.val-rich { color: #f0857d; }
.val-cheap { color: #5fbf82; }
.val-flat { opacity: 0.6; }

/* Six columns of numbers against a long first column; on a narrow viewport the
   table scrolls inside its own box rather than the page scrolling. */
.val-scroll { overflow-x: auto; }
.val-scroll .val-table { min-width: 620px; }
</style>
"""


def render(dataset: ResearchDataset) -> None:
    st.markdown(_VALUATION_CSS, unsafe_allow_html=True)

    tabs = dict(zip(_SUB_TABS, st.tabs(list(_SUB_TABS))))
    with tabs["Valuation"]:
        multiples.render(dataset)
    with tabs["Growth"]:
        growth.render(dataset)
