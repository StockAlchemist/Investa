"""Per-user cache eviction and settings-driven invalidation.

Two regressions are guarded here:

1. The raw-calculation cache was never invalidated by anything. Its key is
   (currency, accounts_key, db_path, db_mtime) and carries no fingerprint of
   manual overrides / symbol map / interest-rate settings, so a settings write —
   which leaves db_mtime untouched — kept serving pre-change numbers for the
   full TTL even though the summary cache above it had been cleared.

2. The per-user evictors rebuilt the default DB path from the username instead
   of resolving it the way the loader does. Whenever `transactions_file` pointed
   somewhere else, the cache keys held the override path, the evictor looked for
   the default one, and eviction silently matched nothing.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config
import server.dependencies as deps
import server.portfolio_service as ps


# --- DB path resolution --------------------------------------------------


@pytest.fixture
def user_root(tmp_path, monkeypatch):
    """Point the app data dir at a temp tree with one user, 'alice'."""
    monkeypatch.setattr(config, "get_app_data_dir", lambda: str(tmp_path))
    cfg_dir = tmp_path / config.USERS_DIR / "alice" / config.CONFIG_DIR
    cfg_dir.mkdir(parents=True)
    return tmp_path


def _write_gui_config(root, username, payload):
    import json

    path = root / config.USERS_DIR / username / config.CONFIG_DIR / config.GUI_CONFIG_FILENAME
    path.write_text(json.dumps(payload))


def test_resolves_default_path_without_override(user_root):
    expected = os.path.join(
        str(user_root), config.USERS_DIR, "alice", config.PORTFOLIO_DB_FILENAME
    )
    assert deps.resolve_user_db_path("alice") == expected


def test_resolves_transactions_file_override(user_root, tmp_path):
    elsewhere = tmp_path / "elsewhere.db"
    elsewhere.write_bytes(b"")
    _write_gui_config(user_root, "alice", {"transactions_file": str(elsewhere)})

    assert deps.resolve_user_db_path("alice") == str(elsewhere)


def test_ignores_override_pointing_at_a_missing_file(user_root):
    _write_gui_config(user_root, "alice", {"transactions_file": "/nope/missing.db"})

    assert deps.resolve_user_db_path("alice").endswith(config.PORTFOLIO_DB_FILENAME)


# --- Eviction reaches the override path ----------------------------------


def test_evictors_match_keys_written_under_an_override(user_root, tmp_path):
    """The regression: keys carry the override path, so an evictor that
    reconstructs the default path would leave them all behind."""
    elsewhere = tmp_path / "elsewhere.db"
    elsewhere.write_bytes(b"")
    _write_gui_config(user_root, "alice", {"transactions_file": str(elsewhere)})
    db = str(elsewhere)

    ps._PORTFOLIO_SUMMARY_CACHE.clear()
    ps._PORTFOLIO_HISTORY_CACHE.clear()
    ps._RAW_CALC_CACHE.clear()

    other = os.path.join(str(user_root), config.USERS_DIR, "bob", config.PORTFOLIO_DB_FILENAME)
    ps._PORTFOLIO_SUMMARY_CACHE[("USD", "ALL", db, 1.0, 7)] = "alice"
    ps._PORTFOLIO_SUMMARY_CACHE[("USD", "ALL", other, 1.0, 7)] = "bob"
    ps._RAW_CALC_CACHE.put(("USD", "ALL", db, 1.0), "alice")
    ps._RAW_CALC_CACHE.put(("USD", "ALL", other, 1.0), "bob")
    ps._PORTFOLIO_HISTORY_CACHE.put((db, "USD", "ALL"), "alice")
    ps._PORTFOLIO_HISTORY_CACHE.put((other, "USD", "ALL"), "bob")

    ps._evict_user_summary_cache("alice")
    ps._evict_user_raw_cache("alice")
    ps._evict_user_history_cache("alice")

    # alice gone, bob untouched — eviction stays per-user.
    assert list(ps._PORTFOLIO_SUMMARY_CACHE) == [("USD", "ALL", other, 1.0, 7)]
    assert ps._RAW_CALC_CACHE.peek(("USD", "ALL", db, 1.0)) is None
    assert ps._RAW_CALC_CACHE.peek(("USD", "ALL", other, 1.0)) == "bob"
    assert ps._PORTFOLIO_HISTORY_CACHE.peek((db, "USD", "ALL")) is None
    assert ps._PORTFOLIO_HISTORY_CACHE.peek((other, "USD", "ALL")) == "bob"


def test_summary_evictor_tolerates_short_keys(user_root):
    ps._PORTFOLIO_SUMMARY_CACHE.clear()
    ps._PORTFOLIO_SUMMARY_CACHE[("USD", "ALL")] = "malformed"

    ps._evict_user_summary_cache("alice")  # must not IndexError

    assert list(ps._PORTFOLIO_SUMMARY_CACHE) == [("USD", "ALL")]
    ps._PORTFOLIO_SUMMARY_CACHE.clear()


# --- Settings writes must drop the raw layer -----------------------------


def test_clear_portfolio_caches_drops_raw_cache(monkeypatch):
    """A settings write calls clear_portfolio_caches(). If the raw cache
    survives it, the recompute reads the stale result straight back out."""
    monkeypatch.setattr(ps, "clear_market_history_cache", lambda: None)

    ps._RAW_CALC_CACHE.put(("USD", "ALL", "/db", 1.0), "pre-change")
    ps._PORTFOLIO_SUMMARY_CACHE[("USD", "ALL", "/db", 1.0, 7)] = "pre-change"

    ps.clear_portfolio_caches()

    assert ps._RAW_CALC_CACHE.peek(("USD", "ALL", "/db", 1.0)) is None
    assert len(ps._PORTFOLIO_SUMMARY_CACHE) == 0


def test_reload_all_users_drops_raw_cache(monkeypatch):
    monkeypatch.setattr(ps, "clear_market_history_cache", lambda: None)
    monkeypatch.setattr(ps, "reload_data", lambda username=None: None)

    ps._RAW_CALC_CACHE.put(("USD", "ALL", "/db", 1.0), "stale")

    ps.reload_data_and_clear_cache(None)  # no user => clear everything

    assert ps._RAW_CALC_CACHE.peek(("USD", "ALL", "/db", 1.0)) is None
