# tests/test_config.py

import pytest
import sys
import os

# --- Add src directory to sys.path for module import ---
# This ensures that the test runner can find the 'config' module.
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# --- End Path Addition ---

# --- Import constants from the NEW config module ---
try:
    from config import (
        CASH_SYMBOL_CSV,
        DEFAULT_CURRENT_CACHE_FILE_PATH,
        HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
        DAILY_RESULTS_CACHE_PATH_PREFIX,
        YFINANCE_CACHE_DURATION_HOURS,
        YFINANCE_INDEX_TICKER_MAP,
        DEFAULT_INDEX_QUERY_SYMBOLS,
        SYMBOL_MAP_TO_YFINANCE,
        YFINANCE_EXCLUDED_SYMBOLS,
        SHORTABLE_SYMBOLS,
        DEFAULT_CURRENCY,
        LOGGING_LEVEL,
        # --- Constants moved from main_gui.py ---
        DEBOUNCE_INTERVAL_MS,
        MANUAL_OVERRIDES_FILENAME,
        # APP_NAME is already tested, APP_NAME_FOR_QT was merged
        DEFAULT_API_KEY,
        CHART_MAX_SLICES,
        PIE_CHART_FIG_SIZE,
        PERF_CHART_FIG_SIZE,
        CHART_DPI,
        INDICES_FOR_HEADER,
        CSV_DATE_FORMAT,
        COMMON_CURRENCIES,
        DEFAULT_GRAPH_DAYS_AGO,
        DEFAULT_GRAPH_INTERVAL,
        DEFAULT_GRAPH_BENCHMARKS,
        BENCHMARK_MAPPING,
        BENCHMARK_OPTIONS_DISPLAY,
        COLOR_BG_DARK,
        COLOR_TEXT_DARK,
        COLOR_GAIN,
        COLOR_LOSS,  # Sample of colors
        DEFAULT_CSV,  # Ensure DEFAULT_CSV is tested
        # --- End moved constants ---
    )

except ImportError as e:
    pytest.fail(f"Failed to import from config module: {e}")


def test_import_and_types():
    """Tests if key constants can be imported and have expected types."""
    assert isinstance(CASH_SYMBOL_CSV, str)
    assert isinstance(DEFAULT_CURRENT_CACHE_FILE_PATH, str)
    assert isinstance(HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX, str)
    assert isinstance(DAILY_RESULTS_CACHE_PATH_PREFIX, str)
    assert isinstance(YFINANCE_CACHE_DURATION_HOURS, int)
    assert isinstance(YFINANCE_INDEX_TICKER_MAP, dict)
    assert isinstance(DEFAULT_INDEX_QUERY_SYMBOLS, list)
    assert isinstance(SYMBOL_MAP_TO_YFINANCE, dict)  # Remains dict, starts empty
    assert isinstance(YFINANCE_EXCLUDED_SYMBOLS, set)
    assert isinstance(SHORTABLE_SYMBOLS, set)
    assert isinstance(DEFAULT_CURRENCY, str)
    assert isinstance(
        LOGGING_LEVEL, int
    )  # logging levels are ints (e.g., logging.INFO)
    assert isinstance(DEFAULT_CSV, str)  # Test for DEFAULT_CSV

    # --- Test types for constants moved from main_gui.py ---
    assert isinstance(DEBOUNCE_INTERVAL_MS, int)
    assert isinstance(MANUAL_OVERRIDES_FILENAME, str)
    assert isinstance(DEFAULT_API_KEY, str)
    assert isinstance(CHART_MAX_SLICES, int)
    assert isinstance(PIE_CHART_FIG_SIZE, tuple)
    assert isinstance(PERF_CHART_FIG_SIZE, tuple)
    assert isinstance(CHART_DPI, int)
    assert isinstance(INDICES_FOR_HEADER, list)
    assert isinstance(CSV_DATE_FORMAT, str)
    assert isinstance(COMMON_CURRENCIES, list)
    assert isinstance(DEFAULT_GRAPH_DAYS_AGO, int)
    assert isinstance(DEFAULT_GRAPH_INTERVAL, str)
    assert isinstance(DEFAULT_GRAPH_BENCHMARKS, list)
    assert isinstance(BENCHMARK_MAPPING, dict)
    assert isinstance(BENCHMARK_OPTIONS_DISPLAY, list)
    assert isinstance(COLOR_BG_DARK, str)
    assert isinstance(COLOR_TEXT_DARK, str)
    assert isinstance(COLOR_GAIN, str)
    assert isinstance(COLOR_LOSS, str)

    # Basic content checks
    assert CASH_SYMBOL_CSV == "$CASH"
    assert DEFAULT_CURRENCY == "USD"
    assert DEFAULT_CSV == "my_transactions.csv"
    assert ".DJI" in YFINANCE_INDEX_TICKER_MAP
    # assert "AAPL" in SYMBOL_MAP_TO_YFINANCE # Removed: SYMBOL_MAP_TO_YFINANCE starts empty
    # assert "BBW" in YFINANCE_EXCLUDED_SYMBOLS  # Removed: YFINANCE_EXCLUDED_SYMBOLS starts empty
    # assert "AAPL" in SHORTABLE_SYMBOLS  # Example shortable symbol

    # Basic content checks for moved constants
    assert DEBOUNCE_INTERVAL_MS == 400
    assert MANUAL_OVERRIDES_FILENAME == "manual_overrides.json"
    assert "USD" in COMMON_CURRENCIES
    assert "S&P 500" in BENCHMARK_MAPPING
    assert BENCHMARK_MAPPING["Total US Market (VTI)"] == "VTI"
    assert BENCHMARK_MAPPING["All-World (VT)"] == "VT"
    assert BENCHMARK_MAPPING["US Total Bond (BND)"] == "BND"
    assert BENCHMARK_MAPPING["Gold (GLD)"] == "GLD"
    assert BENCHMARK_MAPPING["Bitcoin (BTC-USD)"] == "BTC-USD"
    assert "Total US Market (VTI)" in BENCHMARK_OPTIONS_DISPLAY
    assert "SPY (S&P 500 ETF)" not in BENCHMARK_OPTIONS_DISPLAY
    assert len(BENCHMARK_OPTIONS_DISPLAY) == len(
        set(BENCHMARK_OPTIONS_DISPLAY)
    )  # No duplicates
    assert COLOR_GAIN == "#198754"


def test_per_user_config_defaults_to_own_db(tmp_path):
    """A per-user ConfigManager must default `transactions_file` to the DB in
    its own directory — not the shared/global DB. Regression test for new web
    users whose configs all pointed at data/db/portfolio.db and so read the
    same empty database instead of their own transactions.
    """
    from config_manager import ConfigManager
    from db_utils import DB_FILENAME, get_database_path

    # User dir that already contains its own portfolio.db (as created at
    # registration) should resolve to that file.
    user_dir = tmp_path / "users" / "alice"
    user_dir.mkdir(parents=True)
    own_db = user_dir / DB_FILENAME
    own_db.write_bytes(b"")  # create the file so the existence check passes

    cm = ConfigManager(str(user_dir))
    assert cm._get_default_gui_config()["transactions_file"] == str(own_db)

    # A directory without its own portfolio.db falls back to the centralized
    # lookup (preserves legacy single-user GUI behaviour).
    other_dir = tmp_path / "no_db_here"
    other_dir.mkdir()
    cm2 = ConfigManager(str(other_dir))
    assert cm2._get_default_gui_config()["transactions_file"] == get_database_path(
        DB_FILENAME
    )


API_KEY_ENV_NAMES = (
    "GEMINI_API_KEY",
    "FMP_API_KEY",
    "SEC_TH_API_KEY",
    "BOT_API_KEY",
    "TIINGO_API_KEY",
)


@pytest.fixture
def api_key_settings(tmp_path, monkeypatch):
    """A settings route wired to a throwaway .env, with all global state undone.

    The route writes process-wide state - os.environ, attributes on `config`
    and the .env file - so without this every key it sets would leak into the
    rest of the session.
    """
    import config
    import server.routes.settings as settings_mod
    from config_manager import ConfigManager
    from server.auth import User

    for name in API_KEY_ENV_NAMES:
        # monkeypatch records the current value and restores it at teardown.
        monkeypatch.setenv(name, os.environ.get(name, ""))
        monkeypatch.setattr(config, name, getattr(config, name, None), raising=False)

    # Don't leave a precalculation thread running past the test.
    monkeypatch.setattr(
        settings_mod, "trigger_background_precalculation", lambda user: None
    )

    user_dir = tmp_path / "users" / "test_user"
    user_dir.mkdir(parents=True)
    (user_dir / "portfolio.db").write_bytes(b"")

    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=initial_gemini\n")
    monkeypatch.setattr(settings_mod, "project_root", str(tmp_path))

    user = User(
        id=1,
        username="test_user",
        alias="Tester",
        is_active=True,
        created_at="2026-01-01T00:00:00",
    )

    class Harness:
        config_manager = ConfigManager(str(user_dir))
        env = env_file

        @staticmethod
        def update(**fields):
            return settings_mod.update_settings(
                settings=settings_mod.SettingsUpdate(**fields),
                config_manager=Harness.config_manager,
                current_user=user,
            )

    return Harness


def test_api_keys_settings_update(api_key_settings):
    """Saved keys reach os.environ, the config module, and .env."""
    import config

    res = api_key_settings.update(
        gemini_api_key="updated_gemini_key",
        fmp_api_key="updated_fmp_key",
        sec_th_api_key="updated_sec_key",
        bot_api_key="updated_bot_key",
        tiingo_api_key="updated_tiingo_key",
    )
    assert res.get("status") == "success"

    expected = {
        "GEMINI_API_KEY": "updated_gemini_key",
        "FMP_API_KEY": "updated_fmp_key",
        "SEC_TH_API_KEY": "updated_sec_key",
        "BOT_API_KEY": "updated_bot_key",
        "TIINGO_API_KEY": "updated_tiingo_key",
    }
    env_content = api_key_settings.env.read_text()
    for name, value in expected.items():
        assert os.environ.get(name) == value
        assert getattr(config, name) == value
        # Written quoted - see the injection test below.
        assert f"{name}='{value}'" in env_content


def test_api_key_cannot_inject_env_lines(api_key_settings):
    """A newline in a key must not smuggle a second assignment into .env.

    config.py loads .env with override=True, so an injected AUTH_SECRET_KEY
    line would hand the submitter the JWT signing key.
    """
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        api_key_settings.update(
            gemini_api_key="legit_key\nAUTH_SECRET_KEY=forged_signing_key"
        )
    assert excinfo.value.status_code == 400

    env_content = api_key_settings.env.read_text()
    assert "AUTH_SECRET_KEY" not in env_content
    # The rejected write must not have half-applied.
    assert os.environ.get("GEMINI_API_KEY") != "legit_key"


def test_api_keys_are_masked_and_round_trip_safely(api_key_settings):
    """GET hands out previews, and echoing one back leaves the key alone."""
    from server.routes.settings import _mask_secret, _resolve_api_key_updates

    api_key_settings.update(gemini_api_key="sk-secret-value-1234")

    preview = _mask_secret("sk-secret-value-1234")
    assert preview == "\u2022" * 8 + "1234"
    assert "secret" not in preview

    # A client that saves the settings it was handed must not overwrite keys.
    assert _resolve_api_key_updates({"GEMINI_API_KEY": preview}) == {}
    # An omitted field is likewise untouched; a retyped one is applied.
    assert _resolve_api_key_updates({"GEMINI_API_KEY": None}) == {}
    assert _resolve_api_key_updates({"GEMINI_API_KEY": "sk-new"}) == {
        "GEMINI_API_KEY": "sk-new"
    }

    api_key_settings.update(gemini_api_key=preview)
    assert os.environ.get("GEMINI_API_KEY") == "sk-secret-value-1234"


def test_api_key_update_leaves_other_keys_alone(api_key_settings):
    """Saving one key must not clear the four the payload omits."""
    api_key_settings.update(
        gemini_api_key="gemini_v1",
        fmp_api_key="fmp_v1",
        sec_th_api_key="sec_v1",
        bot_api_key="bot_v1",
        tiingo_api_key="tiingo_v1",
    )

    api_key_settings.update(fmp_api_key="fmp_v2")

    assert os.environ.get("FMP_API_KEY") == "fmp_v2"
    assert os.environ.get("GEMINI_API_KEY") == "gemini_v1"
    assert os.environ.get("SEC_TH_API_KEY") == "sec_v1"
    assert os.environ.get("BOT_API_KEY") == "bot_v1"
    assert os.environ.get("TIINGO_API_KEY") == "tiingo_v1"
