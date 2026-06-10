"""Admin routes: webhook-triggered refresh, cache clearing."""

# ruff: noqa: E402
import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import config
from server.portfolio_service import _PORTFOLIO_SUMMARY_CACHE, reload_data_and_clear_cache

router = APIRouter()


class WebhookRefreshRequest(BaseModel):
    secret: str


@router.post("/webhook/refresh")
def webhook_refresh(
    request: WebhookRefreshRequest
):
    """
    Webhook to trigger a market data refresh (cache invalidation).
    Requires a shared secret.
    """
    # Simple hardcoded check for now, or load from env/config
    # For personal local app, a default simple secret is acceptable if not exposed to internet
    # Ideally should be in config.py or env var.
    EXPECTED_SECRET = os.environ.get("INVESTA_WEBHOOK_SECRET", "investa_refresh_secret_123")
    
    if request.secret.strip() != EXPECTED_SECRET:
        logging.warning(f"Webhook Secret Mismatch. Input: '{request.secret}' != Expected: (hidden)")
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
        
        return {"status": "success", "message": "Market data cache invalidated and data reloaded."}
        
    except Exception as e:
        logging.error(f"Error in webhook refresh: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear_cache")
def clear_cache():
    """Clears all application caches (files and in-memory)."""
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
            "portfolio_cache.json", # Legacy name?
        }
        
        # Directory names to recursively delete
        CACHE_DIR_NAMES = {
            'historical_data_cache', 
            # 'fundamentals_cache', # Preserved by user request
            'all_holdings_cache_new',
            'daily_results_cache',
            'test_fx_cache' # Added this
        }
        
        # Extensions that imply cache (be careful not to delete config/overrides)
        CACHE_EXTENSIONS = ('.json', '.feather', '.npy', '.key') 
        
        # Safe-List (Never Delete)
        KEEP_FILES = {'gui_config.json', 'manual_overrides.json', 'investa_transactions.db'}
        KEEP_EXTENSIONS = ('.db', '.sqlite', '.sqlite3', '.bak')

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
                            logging.warning(f"Failed to delete cache dir {item_path}: {e}")
                    continue
                
                # B. Handle Files
                if os.path.isfile(item_path):
                    should_delete = False
                    
                    # 1. Explicit Match
                    if item in EXPLICIT_FILES_TO_DELETE:
                        should_delete = True
                        
                    # 2. Prefix Match (High Confidence)
                    elif any(item.startswith(p) for p in [
                        config.HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX, 
                        config.DAILY_RESULTS_CACHE_PATH_PREFIX,
                        "yf_portfolio_" # Catch-all for yf caches
                    ]):
                        should_delete = True
                        
                    # 3. Extension Match (Low Confidence - only in specific dirs)
                    # Only delete by extension if we are SURE it's a cache file
                    elif any(item.lower().endswith(ext) for ext in CACHE_EXTENSIONS):
                         # Extra safety: Don't delete random JSONs in app_data_dir unless they look like cache
                         if base_dir == app_data_dir and item.endswith(".json"):
                             if "cache" in item.lower():
                                 should_delete = True
                             else:
                                 should_delete = False # Skip unknown JSONs in config dir
                         else:
                             should_delete = True # In Caches/ folder, delete all JSONs/Feathers

                    if should_delete:
                        try:
                            os.remove(item_path)
                            deleted_count += 1
                            logging.info(f"Deleted File: {item}")
                        except Exception as e:
                            logging.warning(f"Failed to delete cache file {item_path}: {e}")
        
        # 3. Reload Data
        logging.info("Reloading data after cache clear...")
        reload_data_and_clear_cache(None)
        
        return {"status": "success", "message": f"Cache cleared. {deleted_count} items removed."}
    except Exception as e:
        logging.error(f"Error clearing cache: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
