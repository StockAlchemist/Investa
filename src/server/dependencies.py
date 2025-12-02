import os
import sys
import pandas as pd
import logging
import json
from typing import Optional, Tuple, Set, Dict

# Ensure src is in path (redundant if imported from main, but good for standalone testing)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from db_utils import get_database_path, load_all_transactions_from_db, DB_FILENAME
from data_loader import load_and_clean_transactions
import config

# Global cache for transactions to avoid reloading on every request
# In a real production app, this might be handled differently, but for a personal app this is fine.
_TRANSACTIONS_CACHE: Optional[pd.DataFrame] = None
_IGNORED_INDICES: Set[int] = set()
_IGNORED_REASONS: Dict[int, str] = {}

def get_transaction_data() -> Tuple[pd.DataFrame, Set[int], Dict[int, str]]:
    """
    Loads transaction data from the database.
    Uses a simple in-memory cache.
    """
    global _TRANSACTIONS_CACHE, _IGNORED_INDICES, _IGNORED_REASONS
    
    # Simple cache invalidation could be added here (e.g. check file mtime), 
    # but for now we'll load once per server restart or if explicitly cleared.
    if _TRANSACTIONS_CACHE is not None:
        return _TRANSACTIONS_CACHE, _IGNORED_INDICES, _IGNORED_REASONS

    # Use my_transactions.db in the project root if it exists, otherwise fallback to db_utils default
    db_path = os.path.join(project_root, "my_transactions.db")
    if not os.path.exists(db_path):
        logging.warning(f"my_transactions.db not found in {project_root}, falling back to db_utils default.")
        db_path = get_database_path(DB_FILENAME)
    
    logging.info(f"Loading transactions from: {db_path}")
    
    # Try to load gui_config.json for account_currency_map
    account_currency_map = {"SET": "THB"} # Default
    default_currency = config.DEFAULT_CURRENCY
    
    config_path = os.path.join(project_root, "gui_config.json")
    if os.path.exists(config_path):
        try:
            import json
            with open(config_path, "r") as f:
                gui_config = json.load(f)
                account_currency_map = gui_config.get("account_currency_map", account_currency_map)
                default_currency = gui_config.get("default_currency", default_currency)
        except Exception as e:
            logging.warning(f"Could not load gui_config.json: {e}")

    # We use the existing data_loader logic which expects a file path (CSV or DB)
    # load_and_clean_transactions handles DB loading if the path ends in .db
    try:
        # Check if it's a DB path
        is_db = db_path.lower().endswith((".db", ".sqlite", ".sqlite3"))
        
        df, _, ignored_indices, ignored_reasons, _, _, _ = load_and_clean_transactions(
            source_path=db_path,
            account_currency_map=account_currency_map,
            default_currency=default_currency,
            is_db_source=is_db
        )
        _TRANSACTIONS_CACHE = df
        _IGNORED_INDICES = ignored_indices
        _IGNORED_REASONS = ignored_reasons
        logging.info(f"Loaded {len(df)} transactions.")
        return df, ignored_indices, ignored_reasons
    except Exception as e:
        logging.error(f"Error loading transactions: {e}", exc_info=True)
        return pd.DataFrame(), set(), {}

def reload_data():
    """Forces a reload of the transaction data."""
    global _TRANSACTIONS_CACHE
    _TRANSACTIONS_CACHE = None
    get_transaction_data()
