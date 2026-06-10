# tests/test_user_data_cache.py

import json
import os
import sys

import pandas as pd

# --- Add src directory to sys.path for module import ---
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# --- End Path Addition ---

import config
import server.dependencies as deps
from server.dependencies import UserDataCache

USERNAME = "cachetest"


def _setup_user(tmp_path, monkeypatch, load_counter):
    """Point the app data dir at tmp_path, create a user dir with a portfolio
    DB file, and stub out the (heavy) transaction loader with a counter."""
    monkeypatch.setattr(deps.config, "get_app_data_dir", lambda: str(tmp_path))

    user_dir = tmp_path / config.USERS_DIR / USERNAME
    user_dir.mkdir(parents=True)
    db_path = user_dir / config.PORTFOLIO_DB_FILENAME
    db_path.write_text("stub")
    os.utime(db_path, (1000, 1000))

    def fake_loader(source_path, account_currency_map, default_currency, is_db_source):
        load_counter.append(source_path)
        df = pd.DataFrame({"Symbol": ["AAPL"], "Quantity": [1.0]})
        return df, None, set(), {}, None, None, None

    monkeypatch.setattr(deps, "load_and_clean_transactions", fake_loader)
    return user_dir, db_path


def test_second_call_is_a_cache_hit(tmp_path, monkeypatch):
    loads = []
    _setup_user(tmp_path, monkeypatch, loads)
    cache = UserDataCache()

    df1, *_ = cache.get_or_load(USERNAME)
    df2, *_ = cache.get_or_load(USERNAME)
    assert len(loads) == 1
    assert df1.equals(df2)


def test_db_mtime_change_triggers_reload(tmp_path, monkeypatch):
    loads = []
    _, db_path = _setup_user(tmp_path, monkeypatch, loads)
    cache = UserDataCache()

    cache.get_or_load(USERNAME)
    os.utime(db_path, (2000, 2000))
    cache.get_or_load(USERNAME)
    assert len(loads) == 2


def test_wal_side_file_mtime_triggers_reload(tmp_path, monkeypatch):
    loads = []
    _, db_path = _setup_user(tmp_path, monkeypatch, loads)
    cache = UserDataCache()

    cache.get_or_load(USERNAME)
    wal = str(db_path) + "-wal"
    with open(wal, "w") as f:
        f.write("x")
    os.utime(wal, (3000, 3000))
    cache.get_or_load(USERNAME)
    assert len(loads) == 2


def test_invalidate_forces_reload(tmp_path, monkeypatch):
    loads = []
    _setup_user(tmp_path, monkeypatch, loads)
    cache = UserDataCache()

    cache.get_or_load(USERNAME)
    cache.invalidate(USERNAME)
    cache.get_or_load(USERNAME)
    assert len(loads) == 2


def test_overrides_change_reloads_overrides_without_db_reload(tmp_path, monkeypatch):
    loads = []
    user_dir, _ = _setup_user(tmp_path, monkeypatch, loads)
    config_dir = user_dir / config.CONFIG_DIR
    config_dir.mkdir()
    overrides_path = config_dir / config.MANUAL_OVERRIDES_FILENAME
    overrides_path.write_text(json.dumps({"manual_price_overrides": {"AAPL": 1.0}}))
    os.utime(overrides_path, (1000, 1000))

    cache = UserDataCache()
    _, manual, *_ = cache.get_or_load(USERNAME)
    assert manual == {"AAPL": 1.0}
    assert len(loads) == 1

    overrides_path.write_text(json.dumps({"manual_price_overrides": {"AAPL": 2.0}}))
    os.utime(overrides_path, (2000, 2000))
    _, manual, *_ = cache.get_or_load(USERNAME)
    assert manual == {"AAPL": 2.0}
    # Overrides reload must NOT re-parse the transactions DB
    assert len(loads) == 1


def test_clear_settings_rereads_overrides(tmp_path, monkeypatch):
    loads = []
    user_dir, _ = _setup_user(tmp_path, monkeypatch, loads)
    config_dir = user_dir / config.CONFIG_DIR
    config_dir.mkdir()
    overrides_path = config_dir / config.MANUAL_OVERRIDES_FILENAME
    overrides_path.write_text(json.dumps({"manual_price_overrides": {"AAPL": 1.0}}))
    os.utime(overrides_path, (1000, 1000))

    cache = UserDataCache()
    cache.get_or_load(USERNAME)

    # Same mtime, new content — normally undetected, but clear_settings forces a re-read
    overrides_path.write_text(json.dumps({"manual_price_overrides": {"AAPL": 9.0}}))
    os.utime(overrides_path, (1000, 1000))
    cache.clear_settings(USERNAME)
    _, manual, *_ = cache.get_or_load(USERNAME)
    assert manual == {"AAPL": 9.0}


def test_load_failure_returns_empty_and_does_not_poison_cache(tmp_path, monkeypatch):
    loads = []
    _setup_user(tmp_path, monkeypatch, loads)

    def boom(**kwargs):
        raise RuntimeError("corrupt db")

    monkeypatch.setattr(deps, "load_and_clean_transactions", boom)
    cache = UserDataCache()
    df, *_rest = cache.get_or_load(USERNAME)
    assert df.empty

    # Restore a working loader: next call should succeed (no poisoned entry)
    def ok_loader(source_path, account_currency_map, default_currency, is_db_source):
        loads.append(source_path)
        return pd.DataFrame({"Symbol": ["MSFT"]}), None, set(), {}, None, None, None

    monkeypatch.setattr(deps, "load_and_clean_transactions", ok_loader)
    df, *_rest = cache.get_or_load(USERNAME)
    assert not df.empty


def test_users_are_isolated(tmp_path, monkeypatch):
    loads = []
    _setup_user(tmp_path, monkeypatch, loads)
    other_dir = tmp_path / config.USERS_DIR / "otheruser"
    other_dir.mkdir(parents=True)
    (other_dir / config.PORTFOLIO_DB_FILENAME).write_text("stub")

    cache = UserDataCache()
    cache.get_or_load(USERNAME)
    cache.get_or_load("otheruser")
    assert len(loads) == 2  # one load each

    cache.invalidate("otheruser")
    cache.get_or_load(USERNAME)  # still cached
    assert len(loads) == 2
