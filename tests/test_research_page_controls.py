"""Stock research page controls: search gating and session-state lifecycle.

The page's two behavioural promises are that nothing is fetched before Search
is pressed, and that changing the price window never refetches. Both are
properties of the session-state handling rather than of any arithmetic, so they
are tested here rather than in ``test_stock_research``.

No Streamlit runtime exists in the suite, so ``st.session_state`` is replaced
with a dict subclass, as in ``test_acknowledgements``.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.logic import stock_research as sr
from app.views import stock_research as page


class _SessionState(dict):
    """Stands in for ``st.session_state``, which needs a Streamlit runtime."""

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


@pytest.fixture
def session(monkeypatch) -> _SessionState:
    state = _SessionState()
    monkeypatch.setattr(page.st, "session_state", state)
    return state


def _dataset(ticker: str = "AAPL") -> sr.ResearchDataset:
    index = pd.bdate_range("2025-01-01", "2026-08-05", name="Date")
    ohlcv = pd.DataFrame(
        {
            "Open": 100.0, "High": 102.0, "Low": 98.0,
            "Close": 101.0, "Adj Close": 101.0, "Volume": 1e6,
        },
        index=index,
    )
    return sr.ResearchDataset(
        ticker=ticker,
        profile={"name": f"{ticker} INC", "currency": "USD"},
        ohlcv=ohlcv,
        daily=pd.DataFrame(),
        actions=pd.DataFrame(),
        ttm_pl=pd.DataFrame(),
        quarterly_ar=pd.DataFrame(),
        stats=[],
        currency="USD",
        loaded_at=date(2026, 8, 5),
    )


# ---------------------------------------------------------------------------
# Search gating
# ---------------------------------------------------------------------------


def test_nothing_is_loaded_before_search_is_pressed(session):
    """The gate is a sticky flag, so an unpressed page holds no dataset."""
    assert not session.get("research_run")
    assert session.get("research_output") is None


def test_reset_clears_the_previous_company(session):
    session["research_output"] = _dataset("AAPL")
    session["research_loaded_ticker"] = "AAPL"

    page._session_reset("MSFT")

    assert "research_output" not in session
    assert session["research_loaded_ticker"] == "MSFT"


def test_reset_leaves_the_selected_window_alone(session):
    """The window is a viewing preference, not a property of the company."""
    session["research_period"] = "5Y"
    session["research_output"] = _dataset("AAPL")

    page._session_reset("MSFT")

    assert session["research_period"] == "5Y"


def test_reset_leaves_other_pages_state_alone(session):
    """Every page shares one session_state; only this page's keys may be cleared."""
    session["default_rf_rate"] = 4.2
    session["dcf_acks"] = {"tv_share": "2026-07-31T09:00:00"}
    session["ps_rf_input"] = 3.9

    page._session_reset("MSFT")

    assert session["default_rf_rate"] == 4.2
    assert session["dcf_acks"] == {"tv_share": "2026-07-31T09:00:00"}
    assert session["ps_rf_input"] == 3.9


# ---------------------------------------------------------------------------
# The snapshot is reused, not refetched
# ---------------------------------------------------------------------------


def test_changing_the_window_does_not_reload_the_company():
    """The whole point of fetching full history once: period buttons are free."""
    assert not page._needs_load(_dataset("AAPL"), "AAPL")


def test_a_snapshot_for_a_different_ticker_is_never_reused():
    """Stale state from the previous company must not render under a new one."""
    assert page._needs_load(_dataset("AAPL"), "MSFT")


def test_an_empty_session_always_loads():
    assert page._needs_load(None, "AAPL")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def test_every_placeholder_names_what_it_will_show_and_where_it_comes_from():
    for label, (what, source) in page._PLACEHOLDERS.items():
        assert what.strip()
        assert "SHARADAR" in source or "sharadar" in source, label


def test_summary_is_not_a_placeholder():
    """Summary is built; the other eight are not, and the counts must agree."""
    assert not any("Summary" in label for label in page._PLACEHOLDERS)
    assert len(page._PLACEHOLDERS) == 8


# ---------------------------------------------------------------------------
# The statistics table
# ---------------------------------------------------------------------------


def test_the_table_is_four_columns_of_four():
    assert len(sr.STAT_COLUMNS) == 4
    assert all(len(column) == 4 for column in sr.STAT_COLUMNS)


def test_a_cell_shows_its_value_on_the_right_of_its_label():
    tile = sr.StatTile(
        label="P/E", display="35.20", origin=sr.Origin.REPORTED, source="Sharadar DAILY"
    )
    cell = page._stat_cell("P/E", tile)
    assert cell.index("P/E") < cell.index("35.20")


def test_a_missing_cell_says_not_reported_rather_than_rendering_blank():
    cell = page._stat_cell("P/E", sr.StatTile.unavailable("P/E", "Sharadar", "gone"))
    assert "not reported" in cell
    # The reason belongs in the sources panel, not crammed into the cell.
    assert "gone" not in cell


def test_a_label_with_no_matching_tile_still_renders_a_cell():
    """A layout/tile mismatch must degrade, not raise, in front of the user."""
    assert "not reported" in page._stat_cell("Nonexistent", None)


def test_vendor_values_are_escaped_not_trusted_as_markup():
    tile = sr.StatTile(
        label="X", display="<img src=x onerror=alert(1)>",
        origin=sr.Origin.REPORTED, source="Sharadar",
    )
    cell = page._stat_cell("X", tile)
    assert "<img" not in cell
    assert "&lt;img" in cell
