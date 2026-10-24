"""🎯 Options — buying upside through calls, across ~90 names, from a stored night.

The shell: it reads the store the nightly collector wrote (``app.data.options_store``),
solves every name's surface once per stored night, applies the theme and
earnings filters, and draws the tab strip. Each tab body is its own module.

Two behaviours are load-bearing:

* **It is never silently stale.** The caption always names the session the
  data closed on, and a banner appears once the store is more than one session
  behind the market — a collector that stopped running must be impossible to
  miss, because every number below would still look current.
* **Only the open tab runs**, as on Stock Research: ``st.tabs(...,
  on_change="rerun")`` and each body built only when its ``.open`` is true.

Nothing here fetches from a market-data vendor; see ``collector/`` for why.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from app import options_layout as layout
from app.data import options_store
from app.formatting import format_date
from app.logic.options import freshness, score, snapshot, universe
from app.views.components import load_tab, render_blocking
from app.views.options import discover, evaluate, history, screen
from app.views.options.common import OptionsContext

_TAB_ORDER = ("🔭 Discover", "📋 Screen", "🧮 Evaluate", "📈 History")
_TAB_KEY = "options_tab"

#: Earnings filter: label → calendar days ahead within which a single stock is hidden.
_EARNINGS_WINDOWS = {"Show all": None, "Hide within 1 week": 7, "Hide within 2 weeks": 14, "Hide within 3 weeks": 21}


def _today_et() -> date:
    return datetime.now(ZoneInfo("America/New_York")).date()


@st.cache_resource(show_spinner=False, max_entries=2)
def _solved(last_session: str, written_at: str):
    """Every name's surface for the stored night. Keyed on the manifest, so a new
    night is solved once and every rerun after reuses the same objects."""
    session = date.fromisoformat(last_session)
    chains = options_store.fetch_frame(layout.LATEST_CHAINS)
    metrics = options_store.fetch_frame(layout.METRICS)
    rates = options_store.fetch_frame(layout.RATES)
    snaps, missing = snapshot.build(chains, metrics, rates, session)
    return snaps, missing, metrics


def _load():
    manifest = options_store.fetch_manifest()
    if not manifest.get("last_session"):
        raise options_store.OptionsStoreError("The collector has not completed a daily run yet")
    snaps, missing, metrics = _solved(manifest["last_session"], manifest.get("written_at", ""))
    try:
        earnings = options_store.fetch_frame(layout.EARNINGS)
    except options_store.OptionsStoreError:
        earnings = None
    return manifest, snaps, missing, metrics, earnings


def render_page() -> None:
    st.header("Options")
    st.write(
        "Buying upside with calls: what it costs, how that compares with each name's own year, "
        "and which call spreads pay most if a move arrives. End-of-day ThetaData chains, collected nightly."
    )
    loaded = load_tab("the options store", _load)
    if loaded is None:
        return
    manifest, snaps, missing, metrics, earnings = loaded
    session = date.fromisoformat(manifest["last_session"])
    if not snaps:
        render_blocking("No name could be priced", f"The stored chain for {session} produced no surfaces.")
        return

    _render_freshness(manifest, session)
    visible = _render_filters(snaps, earnings, session)
    _render_missing(manifest, missing)
    if not visible:
        st.info("The filters leave no names on screen.")
        return

    context = OptionsContext(session=session, snapshots=snaps, metrics=metrics, visible=visible)
    strip = st.tabs(list(_TAB_ORDER), key=_TAB_KEY, on_change="rerun")
    for name, tab in zip(_TAB_ORDER, strip):
        if tab.open:
            with tab:
                _render_tab(name, context)
    _render_definitions()


def _render_tab(name: str, context: OptionsContext) -> None:
    if name == "🔭 Discover":
        discover.render(context)
    elif name == "📋 Screen":
        screen.render(context)
    elif name == "🧮 Evaluate":
        evaluate.render(context)
    elif name == "📈 History":
        history.render(context)
    else:  # pragma: no cover - guarded by a test
        raise ValueError(f"No body for tab {name!r}")


def _render_freshness(manifest: dict, session: date) -> None:
    behind = freshness.sessions_behind(session, _today_et())
    written = manifest.get("written_at", "")[:16].replace("T", " ")
    st.caption(f"Data as of the **{format_date(session)}** close · collected {written} UTC · "
               f"{len(universe.UNIVERSE)} names in the universe")
    if behind > 1:
        st.warning(
            f"**The options store is {behind} sessions behind.** The newest stored close is "
            f"{format_date(session)}; every figure on this page is from that day. The nightly collector "
            "has not completed since — check the Cloud Run job's logs.",
            icon="⚠️",
        )


def _render_filters(snaps, earnings: pd.DataFrame | None, session: date) -> list[str]:
    themes = sorted({s.theme for s in snaps.values()})
    row = st.columns([3, 1.4], vertical_alignment="bottom")
    with row[0]:
        chosen = st.multiselect("Themes", themes, default=themes, key="options_themes")
    with row[1]:
        window = st.selectbox("Earnings", list(_EARNINGS_WINDOWS), index=3, key="options_earnings")
    hidden = set()
    days = _EARNINGS_WINDOWS[window]
    if days is not None and earnings is not None and not earnings.empty:
        for _, item in earnings.iterrows():
            upcoming = item.get("next_earnings")
            if upcoming is not None and not pd.isna(upcoming) and 0 <= (pd.Timestamp(upcoming).date() - session).days <= days:
                hidden.add(item["symbol"])
    if hidden:
        st.caption(f"Hidden for earnings within {days} days: {', '.join(sorted(hidden))}")
    return [s for s in universe.UNIVERSE if s in snaps and snaps[s].theme in chosen and s not in hidden]


def _render_missing(manifest: dict, missing: dict[str, str]) -> None:
    problems = {s: v.get("error") for s, v in manifest.get("symbols", {}).items() if v.get("error")}
    problems.update(missing)
    not_collected = [s for s in universe.UNIVERSE if s not in manifest.get("symbols", {})]
    if not problems and not not_collected:
        return
    with st.expander(f"{len(problems) + len(not_collected)} names not on this page"):
        for symbol, reason in sorted(problems.items()):
            st.write(f"**{symbol}** — {reason}")
        if not_collected:
            st.write(f"Not collected yet: {', '.join(not_collected)}")


def _render_definitions() -> None:
    with st.expander("Universe and definitions"):
        st.markdown(
            "**Universe.** A curated list, transcribed from the page this one was modelled on and grouped "
            "into themes as reading aids (edit `app/logic/options/universe.py`). "
            + " · ".join(f"*{t}*: {', '.join(s for s, th in universe.UNIVERSE.items() if th == t)}"
                         for t in universe.themes())
        )
        st.markdown(
            "**Prices** are mids of ThetaData's 17:15 ET end-of-day NBBO; a zero bid is no market. "
            "**IVs** are Black-76 on each expiry's put–call-parity forward, so dividends need no estimate. "
            "**30-day** figures interpolate total variance between the expiries either side of 30 days; "
            "**25Δ** is read at constant delta on the call wing."
        )
        st.markdown(
            f"**Buy-vol score** = {score.SCALE:g} × the mean of −z(30d ATM IV), −z(30d IV − 21d realized) and "
            "−z(25Δ call skew), each against the name's own trailing year. Higher means calls are cheaper "
            "than usual for that name. Our definition — the page this was modelled on did not publish one."
        )
        st.markdown(
            "**+k SD target** = spot × e^(k·σ·√t) at the expiry's ATM IV. **Reward:risk** = the +1 SD spread's "
            "payout ÷ the payout a flat smile at ATM IV gives; 1.0× is fairly priced. On Screen it is the two quotes "
            "you would trade; on History and in percentiles it is read off the smile at a constant 30 days, so one "
            "stale quote or an expiry roll does not move it."
        )
