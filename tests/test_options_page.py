"""The 🎯 Options page, run headlessly against a local store of synthetic data."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app import options_layout as layout
from app.logic.options.history import METRIC_COLUMNS
from collector.store import LocalStore
from tests import options_fixtures as fx

SYMBOLS = ("GLD", "SPY", "MU")


def _metrics(sessions=200):
    days = [d.date() for d in pd.bdate_range(end=fx.SESSION, periods=sessions)]
    rng = np.random.default_rng(5)
    return pd.DataFrame([
        {c: np.nan for c in METRIC_COLUMNS} | {
            "symbol": s, "session": d, "spot": fx.SPOT, "atm_iv_30": 0.2 + 0.02 * rng.standard_normal(),
            "rv_21": 0.18, "skew_30": 0.01 * rng.standard_normal(), "reward_risk": 1 + 0.1 * rng.standard_normal(),
        }
        for s in SYMBOLS for d in days
    ])


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = LocalStore(tmp_path)
    s.write_frame(layout.LATEST_CHAINS, pd.concat([fx.build_chain(symbol=sym) for sym in SYMBOLS], ignore_index=True))
    s.write_frame(layout.METRICS, _metrics())
    s.write_frame(layout.RATES, pd.DataFrame({"session": [fx.SESSION] * 3, "tenor": [1 / 12, 1.0, 5.0],
                                              "yield_pct": fx.YIELD_PCT}))
    s.write_frame(layout.EARNINGS, pd.DataFrame({"symbol": ["MU"], "next_earnings": [fx.SESSION + timedelta(days=10)]}))
    s.write_json(layout.MANIFEST, {
        "last_session": fx.SESSION.isoformat(), "written_at": "2026-10-24T06:00:00+00:00",
        "symbols": {"GLD": {"error": None}, "SPY": {"error": None}, "MU": {"error": None},
                    "SLV": {"error": "option EOD SLV: INVALID_ARGUMENT"}},
    })
    monkeypatch.setenv("OPTIONS_STORE_URI", f"file://{tmp_path}")
    return s


def _page(today: str):
    from datetime import date

    from app.views import options as page

    page._today_et = lambda: date.fromisoformat(today)
    page.render_page()


def _every_tab(today: str):
    from datetime import date

    from app.views import options as page
    from app.views.options.common import OptionsContext

    page._today_et = lambda: date.fromisoformat(today)
    manifest, snaps, missing, metrics, earnings = page._load()
    context = OptionsContext(date.fromisoformat(manifest["last_session"]), snaps, metrics, sorted(snaps))
    for name in page._TAB_ORDER:
        page._render_tab(name, context)


def _run(fn, *args):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(fn, args=args, default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    return app


def test_the_page_says_which_close_it_shows_and_hides_names_near_earnings(store):
    app = _run(_page, "2026-10-24")
    captions = " ".join(c.value for c in app.caption)
    assert "23 Oct 2026" in captions or "2026-10-23" in captions
    assert "Hidden for earnings within 21 days: MU" in captions
    assert not app.warning  # one weekday behind is not stale


def test_a_store_two_sessions_behind_raises_a_banner(store):
    app = _run(_page, "2026-10-28")
    assert any("sessions behind" in w.value for w in app.warning)


def test_a_failed_name_is_listed_not_dropped(store):
    app = _run(_page, "2026-10-24")
    assert any("SLV" in m.value and "INVALID_ARGUMENT" in m.value for m in app.markdown)


def test_every_tab_renders(store):
    app = _run(_every_tab, "2026-10-24")
    assert len(app.get("plotly_chart")) == 2  # Discover's scatter and History's line
    assert len(app.dataframe) == 3  # Discover's calls, Screen's spreads, Evaluate's grid


def test_every_tab_label_has_a_body():
    from app.views import options as page

    with pytest.raises(ValueError):
        page._render_tab("no such tab", None)
    assert len(page._TAB_ORDER) == len(set(page._TAB_ORDER)) == 4


def test_an_unconfigured_store_blocks_the_page(monkeypatch):
    monkeypatch.delenv("OPTIONS_STORE_URI", raising=False)
    app = _run(_page, "2026-10-24")
    assert any("options store" in e.value for e in app.error)
