"""Tests for the Thai SEC NAV provider and the fund_nav store.

The API needs a subscription key, so every network call here is stubbed. What is
actually under test is the logic that would otherwise silently do the wrong
thing: cursor pagination, exact-abbreviation resolution (a partial search for
SCBRM1 also returns SCBRM10), and using NAV-per-unit rather than total net
assets as the price.
"""

import os
import sys
from datetime import date

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from market_db import MarketDatabase  # noqa: E402
from sec_thailand_provider import (  # noqa: E402
    SECThailandError,
    SECThailandNotConfiguredError,
    SECThailandProvider,
)


class FakeAPI:
    """Stands in for _get, recording calls and replaying queued pages."""

    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def __call__(self, path, params):
        self.calls.append((path, dict(params)))
        if not self.pages:
            return {"items": [], "next_cursor": ""}
        return self.pages.pop(0)


@pytest.fixture
def provider():
    return SECThailandProvider(api_key="test-key")


# --- configuration ---------------------------------------------------------


@pytest.fixture
def unconfigured(monkeypatch):
    """
    A provider with no key, whatever the developer's .env holds.

    The constructor falls back to the module-level SEC_TH_API_KEY, so passing
    api_key=None is not enough once a real key exists — these tests silently
    stopped testing anything the moment the key was added.
    """
    import sec_thailand_provider

    monkeypatch.setattr(sec_thailand_provider, "SEC_TH_API_KEY", None)
    return sec_thailand_provider.SECThailandProvider(api_key=None)


def test_missing_key_raises_before_any_network_call(unconfigured):
    assert not unconfigured.is_configured
    with pytest.raises(SECThailandNotConfiguredError):
        unconfigured.lookup_fund("SCBRM1")


def test_verify_reports_missing_key_without_raising(unconfigured):
    result = unconfigured.verify()
    assert result["configured"] is False
    assert "SEC_TH_API_KEY" in result["error"]


def test_auth_failures_are_flagged_as_config_errors():
    assert SECThailandError("nope", status=401).is_config_error
    assert SECThailandError("nope", status=403).is_config_error
    assert not SECThailandError("boom", status=500).is_config_error


# --- pagination ------------------------------------------------------------


def test_pagination_follows_the_cursor_to_the_end(provider, monkeypatch):
    api = FakeAPI(
        [
            {
                "items": [{"nav_date": "2024-01-02", "last_val": 10.0}],
                "next_cursor": "c1",
            },
            {
                "items": [{"nav_date": "2024-01-03", "last_val": 11.0}],
                "next_cursor": "",
            },
        ]
    )
    monkeypatch.setattr(provider, "_get", api)

    rows = provider.fetch_nav("P_1", date(2024, 1, 1), date(2024, 1, 31))
    assert [r["date"] for r in rows] == ["2024-01-02", "2024-01-03"]
    assert len(api.calls) == 2
    assert api.calls[1][1]["next_cursor"] == "c1"


def test_repeated_cursor_does_not_loop_forever(provider, monkeypatch):
    stuck = {
        "items": [{"nav_date": "2024-01-02", "last_val": 10.0}],
        "next_cursor": "same",
    }
    api = FakeAPI([stuck, stuck, stuck])
    monkeypatch.setattr(provider, "_get", api)

    provider.fetch_nav("P_1")
    assert len(api.calls) <= 2


def test_page_size_is_capped_at_the_api_maximum(provider, monkeypatch):
    api = FakeAPI([{"items": [], "next_cursor": ""}])
    monkeypatch.setattr(provider, "_get", api)

    list(provider._paginate("/x", {"page_size": 5000}))
    assert api.calls[0][1]["page_size"] == 100


# --- NAV parsing -----------------------------------------------------------


def test_nav_uses_last_val_not_net_asset(provider, monkeypatch):
    """
    `net_asset` is the fund's total size — hundreds of millions of baht. Using
    it as a price would value a holding absurdly.
    """
    monkeypatch.setattr(
        provider,
        "_get",
        FakeAPI(
            [
                {
                    "items": [
                        {
                            "nav_date": "2024-01-02",
                            "last_val": 15.0833,
                            "net_asset": 248999361.26,
                        }
                    ],
                    "next_cursor": "",
                }
            ]
        ),
    )
    rows = provider.fetch_nav("P_1")
    assert rows[0]["nav"] == pytest.approx(15.0833)


def test_rows_without_a_usable_nav_are_dropped(provider, monkeypatch):
    monkeypatch.setattr(
        provider,
        "_get",
        FakeAPI(
            [
                {
                    "items": [
                        {"nav_date": "2024-01-02", "last_val": None},
                        {"nav_date": "", "last_val": 10.0},
                        {"nav_date": "2024-01-04", "last_val": 0},
                        {"nav_date": "2024-01-05", "last_val": "not a number"},
                        {"nav_date": "2024-01-06", "last_val": 12.5},
                    ],
                    "next_cursor": "",
                }
            ]
        ),
    )
    rows = provider.fetch_nav("P_1")
    assert [r["date"] for r in rows] == ["2024-01-06"]


def test_rows_come_back_in_date_order(provider, monkeypatch):
    monkeypatch.setattr(
        provider,
        "_get",
        FakeAPI(
            [
                {
                    "items": [
                        {"nav_date": "2024-03-01", "last_val": 3.0},
                        {"nav_date": "2024-01-01", "last_val": 1.0},
                        {"nav_date": "2024-02-01", "last_val": 2.0},
                    ],
                    "next_cursor": "",
                }
            ]
        ),
    )
    assert [r["nav"] for r in provider.fetch_nav("P_1")] == [1.0, 2.0, 3.0]


def test_date_range_is_passed_through(provider, monkeypatch):
    api = FakeAPI([{"items": [], "next_cursor": ""}])
    monkeypatch.setattr(provider, "_get", api)

    provider.fetch_nav("P_1", date(2020, 1, 1), date(2020, 12, 31))
    params = api.calls[0][1]
    assert params["start_nav_date"] == "2020-01-01"
    assert params["end_nav_date"] == "2020-12-31"
    assert params["proj_id"] == "P_1"


# --- fund resolution -------------------------------------------------------


def _profiles(*entries):
    return [{"items": list(entries), "next_cursor": ""}]


def test_exact_abbreviation_wins_over_partial_matches(provider, monkeypatch):
    """A partial search for SCBRM1 also returns SCBRM10 — picking that would
    backfill an entirely different fund's history."""
    monkeypatch.setattr(
        provider,
        "_get",
        FakeAPI(
            _profiles(
                {"proj_id": "P_10", "proj_abbr_name": "SCBRM10"},
                {"proj_id": "P_1", "proj_abbr_name": "SCBRM1"},
                {"proj_id": "P_100", "proj_abbr_name": "SCBRM1000"},
            )
        ),
    )
    assert provider.resolve_proj_id("SCBRM1") == "P_1"


def test_resolution_is_case_insensitive(provider, monkeypatch):
    monkeypatch.setattr(
        provider,
        "_get",
        FakeAPI(_profiles({"proj_id": "P_1", "proj_abbr_name": "es-gqg"})),
    )
    assert provider.resolve_proj_id("ES-GQG") == "P_1"


def test_one_abbreviation_across_two_projects_refuses_to_choose(provider, monkeypatch):
    """
    Two projects sharing an abbreviation is unresolvable, and picking the
    livelier-looking one would backfill a whole history under the wrong code.
    Refusing is the safe answer; the caller adds an explicit alias.
    """
    monkeypatch.setattr(
        provider,
        "_get",
        FakeAPI(
            _profiles(
                {
                    "proj_id": "OLD",
                    "proj_abbr_name": "SCBRM1",
                    "cancel_date": "2019-01-01",
                },
                {"proj_id": "LIVE", "proj_abbr_name": "SCBRM1", "cancel_date": ""},
            )
        ),
    )
    match = provider.resolve_fund("SCBRM1")
    assert not match.resolved
    assert match.matched_on == "ambiguous-project"


# --- share classes ---------------------------------------------------------


def test_a_code_naming_a_share_class_resolves_to_project_plus_class(
    provider, monkeypatch
):
    """
    SCBCHA-SSF is a *class* of project SCBCHAFUND, and `project_info` does not
    search class names — so a direct lookup finds nothing and the resolver has
    to retry on the stem before the dash.
    """
    api = FakeAPI(
        [
            {"items": [], "next_cursor": ""},  # direct 'SCBCHA-SSF' search: no rows
            {
                "items": [
                    {
                        "proj_id": "M0005",
                        "proj_abbr_name": "SCBCHAFUND",
                        "fund_class_name": "SCBCHA",
                    },
                    {
                        "proj_id": "M0005",
                        "proj_abbr_name": "SCBCHAFUND",
                        "fund_class_name": "SCBCHA-SSF",
                    },
                    {
                        "proj_id": "M0005",
                        "proj_abbr_name": "SCBCHAFUND",
                        "fund_class_name": "SCBCHAR",
                    },
                ],
                "next_cursor": "",
            },
        ]
    )
    monkeypatch.setattr(provider, "_get", api)

    match = provider.resolve_fund("SCBCHA-SSF")
    assert match.resolved
    assert match.proj_id == "M0005"
    assert match.fund_class_name == "SCBCHA-SSF"
    assert match.matched_on == "class"
    # It had to search the stem to find it.
    assert api.calls[1][1]["project_info"] == "SCBCHA"


def test_single_main_class_needs_no_class_filter(provider, monkeypatch):
    monkeypatch.setattr(
        provider,
        "_get",
        FakeAPI(
            _profiles(
                {
                    "proj_id": "M0079",
                    "proj_abbr_name": "SCBRM1",
                    "fund_class_name": "main",
                }
            )
        ),
    )
    match = provider.resolve_fund("SCBRM1")
    assert match.resolved
    assert match.fund_class_name is None
    assert match.matched_on == "abbr"


def test_abbreviation_hitting_several_classes_is_ambiguous(provider, monkeypatch):
    """
    Matching the project but not a class leaves it undecided which NAV series
    is wanted — blending eight classes into one would be silent nonsense.
    """
    monkeypatch.setattr(
        provider,
        "_get",
        FakeAPI(
            _profiles(
                {"proj_id": "M1", "proj_abbr_name": "FUND", "fund_class_name": "A"},
                {"proj_id": "M1", "proj_abbr_name": "FUND", "fund_class_name": "B"},
            )
        ),
    )
    match = provider.resolve_fund("FUND")
    assert not match.resolved
    assert match.matched_on == "ambiguous-class"
    assert match.candidates == ["A", "B"]


def test_no_content_is_an_empty_result_not_a_failure(provider, monkeypatch):
    """
    The API answers an unmatched filter with 204 No Content. Treating that as an
    error turned 'no such abbreviation' into a hard lookup failure.
    """

    class Response:
        status_code = 204
        content = b""
        text = ""

    monkeypatch.setattr(provider, "_throttle", lambda: None)
    monkeypatch.setattr(
        provider,
        "_get_session",
        lambda: type("S", (), {"get": lambda *a, **k: Response()})(),
    )

    assert provider._get("/x", {}) == {"items": [], "next_cursor": ""}


def test_no_exact_match_resolves_to_nothing(provider, monkeypatch):
    monkeypatch.setattr(
        provider,
        "_get",
        FakeAPI(_profiles({"proj_id": "P_9", "proj_abbr_name": "SOMETHING-ELSE"})),
    )
    assert provider.resolve_proj_id("SCBRM1") is None


# --- storage ---------------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    database = MarketDatabase(str(tmp_path / "market_test.db"))
    with database._get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fund_nav (
                fund_code TEXT NOT NULL, date TEXT NOT NULL, nav REAL NOT NULL,
                currency TEXT, source TEXT NOT NULL,
                PRIMARY KEY (fund_code, date))
            """
        )
        conn.commit()
    return database


def test_navs_round_trip_through_the_store(db):
    written = db.upsert_fund_nav(
        "SCBRM1", [("2024-01-02", 15.0833), ("2024-01-03", 15.12)]
    )
    assert written == 2

    frame = db.get_fund_nav("SCBRM1", date(2024, 1, 1), date(2024, 1, 31))
    assert list(frame.columns) == ["price"]
    assert frame["price"].iloc[0] == pytest.approx(15.0833)


def test_reingesting_a_day_updates_it_rather_than_duplicating(db):
    db.upsert_fund_nav("SCBRM1", [("2024-01-02", 15.0)])
    db.upsert_fund_nav("SCBRM1", [("2024-01-02", 15.5)])

    frame = db.get_fund_nav("SCBRM1", date(2024, 1, 1), date(2024, 1, 31))
    assert len(frame) == 1
    assert frame["price"].iloc[0] == pytest.approx(15.5)


def test_coverage_reports_range_and_count(db):
    db.upsert_fund_nav("SCBRM1", [("2024-01-02", 15.0), ("2024-06-03", 16.0)])
    coverage = db.get_fund_nav_coverage()
    assert coverage["SCBRM1"] == ("2024-01-02", "2024-06-03", 2)


def test_empty_input_writes_nothing(db):
    assert db.upsert_fund_nav("SCBRM1", []) == 0
    assert db.get_fund_nav_coverage() == {}
