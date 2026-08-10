"""Per-check acknowledgement lifecycle.

Acknowledged items stay listed, reset when the ticker changes, never apply to
BLOCKING issues, and are stamped into the export.
"""

from __future__ import annotations

import pytest

from app.logic.dcf import validation as val
from app.views.research import dcf as page


class _SessionState(dict):
    """Stands in for ``st.session_state``, which needs a Streamlit runtime."""

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def setdefault(self, key, default=None):
        return super().setdefault(key, default)


@pytest.fixture
def session(monkeypatch) -> _SessionState:
    state = _SessionState()
    monkeypatch.setattr(page.st, "session_state", state)
    return state


def _issues() -> list[val.Issue]:
    return [
        val.ReconciliationIssue(
            key="balance_sheet:2025", severity=val.Severity.CRITICAL,
            check="Balance sheet balances", concept="A = L + E", company="TEST",
            message="Does not balance.", period="FY2025",
            values={"Total assets": "1", "Liabilities + equity": "2"}, tolerance="0.1%",
        ),
        val.ImplausibleValueIssue(
            key="tv_share", severity=val.Severity.WARNING,
            check="Terminal value share of enterprise value", concept="Terminal value",
            company="TEST", message="Mostly assumption.",
            values={"Terminal value share": "88.0%"}, tolerance="≤ 75%",
        ),
    ]


def test_acknowledgements_reset_when_the_ticker_changes(session):
    session["dcf_acks"] = {"tv_share": "2026-07-31T09:00:00"}
    session["dcf_loaded_ticker"] = "AAPL"
    session["dcf_input_growth_value"] = 9.34

    page._session_reset("MSFT")

    assert session["dcf_acks"] == {}
    assert session["dcf_loaded_ticker"] == "MSFT"
    assert "dcf_input_growth_value" not in session


def test_reset_also_clears_the_computed_run(session):
    """The previous company's numbers must not outlive the reset that drops its
    inputs — the tab has no staleness notice to fall back on."""
    session["dcf_output"] = {"company": "AAPL"}

    page._session_reset("MSFT")

    assert "dcf_output" not in session


def test_reset_also_clears_assumption_overrides(session):
    """A driver typed for one company must not silently carry to the next."""
    session["dcf_acks"] = {}
    session["dcf_input_beta"] = 1.8
    session["dcf_beta_override"] = True

    page._session_reset("MSFT")

    assert "dcf_input_beta" not in session
    assert "dcf_beta_override" not in session


def test_reset_clears_the_reverse_what_if_price(session):
    """Solved against the next company's model, it would read as an answer about it."""
    session["dcf_reverse_price"] = 214.50

    page._session_reset("MSFT")

    assert "dcf_reverse_price" not in session


def test_reset_leaves_unrelated_pages_state_alone(session):
    """Other pages share session_state; only this page's keys may be cleared."""
    session["default_rf_rate"] = 4.2
    session["ps_rf_input"] = 3.9

    page._session_reset("MSFT")

    assert session["default_rf_rate"] == 4.2
    assert session["ps_rf_input"] == 3.9


def test_acknowledged_items_stay_in_the_issue_list():
    issues = _issues()
    acknowledged = {"tv_share": "2026-07-31T09:00:00"}
    live = [i for i in issues if i.key not in acknowledged]
    acked = [i for i in issues if i.key in acknowledged]
    assert len(live) == 1
    assert len(acked) == 1
    assert len(live) + len(acked) == len(issues)


def test_blocking_issues_are_never_acknowledgeable():
    """A BLOCKING issue stops the page before any acknowledge control renders."""
    blocking = val.InsufficientHistoryIssue(
        key="history_none", severity=val.Severity.BLOCKING,
        check="Sufficient usable history", concept="Historical revenue",
        company="TEST", message="Nothing to forecast from.",
        values={"Usable fiscal years": "0"}, tolerance="10 years",
    )
    issues = _issues() + [blocking]
    rendered_with_controls = [
        i for i in issues if i.severity in (val.Severity.CRITICAL, val.Severity.WARNING)
    ]
    assert blocking not in rendered_with_controls
    assert len(rendered_with_controls) == 2


def test_acknowledgement_keys_match_the_issue_keys():
    """The handle stored in session state is exactly the issue's key."""
    for issue in _issues():
        assert issue.key
        assert " " not in issue.key


def test_acknowledgement_reaches_the_export():
    frame = val.issues_frame(_issues(), {"tv_share": "2026-07-31T09:00:00"})
    row = frame[frame["Key"] == "tv_share"].iloc[0]
    assert row["Acknowledged"] == "yes"
    assert row["Acknowledged at"] == "2026-07-31T09:00:00"
