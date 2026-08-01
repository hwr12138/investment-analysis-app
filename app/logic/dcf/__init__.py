"""DCF valuation engine.

Split across several modules rather than the repo's usual single
``app/logic/<feature>.py`` because the accuracy mandate needs the schema
resolution, the statement normalisation, the validation checks and the
arithmetic to be independently testable.

``wacc`` and ``valuation`` are pure: no Streamlit, no network, no globals.
Numbers in, numbers out. That is what keeps them testable and what makes batch
screening an extension rather than a rewrite.
"""
