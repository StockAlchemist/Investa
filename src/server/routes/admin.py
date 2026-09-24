"""Admin routes: webhook-triggered refresh, cache clearing."""

# ruff: noqa: E402
import hmac
import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import config
from server.auth import User
from server.dependencies import get_current_user
from server.portfolio_service import (
    _PORTFOLIO_SUMMARY_CACHE,
    reload_data_and_clear_cache,
)

router = APIRouter()


class WebhookRefreshRequest(BaseModel):
    secret: str


@router.post("/webhook/refresh")
def webhook_refresh(request: WebhookRefreshRequest):
    """
    Webhook to trigger a market data refresh (cache invalidation).
    Requires a shared secret.
    """
    # No default: a fallback secret committed to the repo is a secret everyone
    # has. Until INVESTA_WEBHOOK_SECRET is set, the webhook is off.
    expected_secret = os.environ.get("INVESTA_WEBHOOK_SECRET", "")
    if not expected_secret:
        raise HTTPException(
            status_code=503,
            detail="Webhook disabled: set INVESTA_WEBHOOK_SECRET on the server",
        )

    if not hmac.compare_digest(
        request.secret.strip().encode(), expected_secret.encode()
    ):
        logging.warning("Webhook refresh rejected: secret mismatch.")
        raise HTTPException(status_code=403, detail="Invalid secret")

    try:
        # 1. Invalidate Market Cache
        # The cache file is typically DEFAULT_CURRENT_CACHE_FILE_PATH
        # We can either delete it or rely on MarketDataProvider to manage it.
        # Safest is to delete the file.

        app_data_dir = config.get_app_data_dir()
        cache_path = os.path.join(app_data_dir, config.DEFAULT_CURRENT_CACHE_FILE_PATH)

        if os.path.exists(cache_path):
            os.remove(cache_path)
            logging.info(f"Webhook: Deleted market data cache at {cache_path}")
        else:
            logging.info("Webhook: Cache file not found (already clean).")

        # 2. Reload internal transaction cache
        reload_data_and_clear_cache()

        return {
            "status": "success",
            "message": "Market data cache invalidated and data reloaded.",
        }

    except Exception as e:
        logging.error(f"Error in webhook refresh: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Webhook refresh failed")


@router.post("/clear_cache")
def clear_cache(current_user: User = Depends(get_current_user)):
    """Clears all application caches (files and in-memory).

    Requires a login: the caches are shared, and clearing them forces every
    user's next request to re-download decades of price history. Unguarded, any
    device on the network could trigger that, and so could any web page — a
    bodiless POST needs no CORS preflight.
    """
    try:
        logging.info("Starting Cache Clearing Process...")
        deleted_count = 0

        # 1. Clear In-Memory Caches
        _PORTFOLIO_SUMMARY_CACHE.clear()
        # Also try to clear MarketDataProvider's internal cache if possible (it re-reads file anyway)

        # 2. Identify Cache Directories
        app_data_dir = config.get_app_data_dir()
        app_cache_dir = config.get_app_cache_dir()

        targets = [app_data_dir]
        if app_cache_dir and app_cache_dir != app_data_dir:
            targets.append(app_cache_dir)

        # Removed broad system cache scanning for safety and performance.
        # We only want to clear OUR app's cache.

        logging.info(f"Target Directories for cleanup: {targets}")

        # Files/Dirs that MUST be deleted
        # Explicit filenames to target in ANY target directory
        EXPLICIT_FILES_TO_DELETE = {
            "portfolio_cache_yf.json",
            "yf_metadata_cache.json",
            "invalid_symbols_cache.json",
            "portfolio_cache.json",  # Legacy name?
        }

        # Directory names to recursively delete
        CACHE_DIR_NAMES = {
            "historical_data_cache",
            # 'fundamentals_cache', # Preserved by user request
            "all_holdings_cache_new",
            "daily_results_cache",
            "test_fx_cache",  # Added this
        }

        # Extensions that imply cache (be careful not to delete config/overrides)
        CACHE_EXTENSIONS = (".json", ".feather", ".npy", ".key")

        # Safe-List (Never Delete) — top-level files only. A directory named in
        # CACHE_DIR_NAMES is removed with rmtree, which does not consult this.
        #
        # auth_secret.key is here because '.key' is a CACHE_EXTENSION: a top-level
        # key file matches the extension branch below and nothing else would stop
        # it, and deleting the JWT signing key logs out every user. It survives
        # today only because it sits in data/config/, which this never descends
        # into — that is luck, not intent. (`investa_transactions.db` used to be
        # listed here; it was the pre-multi-user single-file store, is long gone,
        # and every real database is covered by KEEP_EXTENSIONS anyway.)
        KEEP_FILES = {"gui_config.json", "manual_overrides.json", "auth_secret.key"}
        KEEP_EXTENSIONS = (".db", ".sqlite", ".sqlite3", ".bak")

        import shutil

        for base_dir in targets:
            if not os.path.exists(base_dir):
                continue

            logging.info(f"Scanning {base_dir}...")

            try:
                items = os.listdir(base_dir)
            except Exception as e:
                logging.warning(f"Could not list {base_dir}: {e}")
                continue

            for item in items:
                item_path = os.path.join(base_dir, item)

                # PROTECTED CHECKS
                if item in KEEP_FILES:
                    continue
                if any(item.lower().endswith(ext) for ext in KEEP_EXTENSIONS):
                    continue

                # A. Handle Subdirectories
                if os.path.isdir(item_path):
                    if item in CACHE_DIR_NAMES or item.startswith("yf_portfolio_hist"):
                        try:
                            # Count files inside for reporting
                            for _, _, files in os.walk(item_path):
                                deleted_count += len(files)
                            shutil.rmtree(item_path)
                            deleted_count += 1
                            logging.info(f"Deleted Directory: {item}")
                        except Exception as e:
                            logging.warning(
                                f"Failed to delete cache dir {item_path}: {e}"
                            )
                    continue

                # B. Handle Files
                if os.path.isfile(item_path):
                    should_delete = False

                    # 1. Explicit Match
                    if item in EXPLICIT_FILES_TO_DELETE:
                        should_delete = True

                    # 2. Prefix Match (High Confidence)
                    elif any(
                        item.startswith(p)
                        for p in [
                            config.HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
                            config.DAILY_RESULTS_CACHE_PATH_PREFIX,
                            "yf_portfolio_",  # Catch-all for yf caches
                        ]
                    ):
                        should_delete = True

                    # 3. Extension Match (Low Confidence - only in specific dirs)
                    # Only delete by extension if we are SURE it's a cache file
                    elif any(item.lower().endswith(ext) for ext in CACHE_EXTENSIONS):
                        # Extra safety: Don't delete random JSONs in app_data_dir unless they look like cache
                        if base_dir == app_data_dir and item.endswith(".json"):
                            if "cache" in item.lower():
                                should_delete = True
                            else:
                                should_delete = (
                                    False  # Skip unknown JSONs in config dir
                                )
                        else:
                            should_delete = (
                                True  # In Caches/ folder, delete all JSONs/Feathers
                            )

                    if should_delete:
                        try:
                            os.remove(item_path)
                            deleted_count += 1
                            logging.info(f"Deleted File: {item}")
                        except Exception as e:
                            logging.warning(
                                f"Failed to delete cache file {item_path}: {e}"
                            )

        # 3. Reload Data
        logging.info("Reloading data after cache clear...")
        reload_data_and_clear_cache(None)

        return {
            "status": "success",
            "message": f"Cache cleared. {deleted_count} items removed.",
        }
    except Exception as e:
        logging.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear cache")
