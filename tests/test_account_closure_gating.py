"""Unit tests for the closed-account gating helper in server/api.py.

The helper decides whether a selected slice of accounts is entirely "closed"
(every account has a closure date on or before today). When that's true, the
API gates rate-of-return metrics (TWR / IRR / yields) to avoid the
residual-dividend inflation bug — see plan in
.claude/plans/there-are-some-cards-optimized-creek.md.
"""

import os
import sys
from datetime import date, timedelta

import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(TEST_DIR, "..", "src"))
sys.path.insert(0, SRC_DIR)

try:
    from server.api import compute_account_closure_state
except ImportError:
    pytest.skip("server.api not importable in this environment", allow_module_level=True)


TODAY = date(2025, 6, 1)
PAST = (TODAY - timedelta(days=30)).isoformat()
FUTURE = (TODAY + timedelta(days=30)).isoformat()


def test_no_selection_is_never_closed():
    closed, all_closed = compute_account_closure_state(None, {"X": PAST}, TODAY)
    assert closed == []
    assert all_closed is False

    closed, all_closed = compute_account_closure_state([], {"X": PAST}, TODAY)
    assert closed == []
    assert all_closed is False


def test_single_closed_account_triggers_gating():
    closed, all_closed = compute_account_closure_state(
        ["Sharebuilder"], {"Sharebuilder": PAST}, TODAY
    )
    assert closed == ["Sharebuilder"]
    assert all_closed is True


def test_mixed_selection_does_not_trigger():
    closed, all_closed = compute_account_closure_state(
        ["Sharebuilder", "ETrade"], {"Sharebuilder": PAST}, TODAY
    )
    assert closed == ["Sharebuilder"]
    assert all_closed is False


def test_future_closure_date_does_not_trigger():
    closed, all_closed = compute_account_closure_state(
        ["Sharebuilder"], {"Sharebuilder": FUTURE}, TODAY
    )
    assert closed == []
    assert all_closed is False


def test_closure_on_today_triggers():
    closed, all_closed = compute_account_closure_state(
        ["Sharebuilder"], {"Sharebuilder": TODAY.isoformat()}, TODAY
    )
    assert closed == ["Sharebuilder"]
    assert all_closed is True


def test_malformed_date_treated_as_open():
    closed, all_closed = compute_account_closure_state(
        ["Sharebuilder"], {"Sharebuilder": "not-a-date"}, TODAY
    )
    assert closed == []
    assert all_closed is False


def test_account_with_no_entry_is_open():
    closed, all_closed = compute_account_closure_state(
        ["Sharebuilder", "ETrade"],
        {"Sharebuilder": PAST},  # ETrade has no entry → open
        TODAY,
    )
    assert closed == ["Sharebuilder"]
    assert all_closed is False


def test_all_selected_closed_with_multiple_accounts():
    closed, all_closed = compute_account_closure_state(
        ["Sharebuilder", "OldFidelity"],
        {"Sharebuilder": PAST, "OldFidelity": PAST},
        TODAY,
    )
    assert closed == ["Sharebuilder", "OldFidelity"]
    assert all_closed is True


def test_empty_closure_map():
    closed, all_closed = compute_account_closure_state(
        ["Sharebuilder"], {}, TODAY
    )
    assert closed == []
    assert all_closed is False
