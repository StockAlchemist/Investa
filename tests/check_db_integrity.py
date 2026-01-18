
import sys
import os
import pandas as pd
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from config_manager import ConfigManager
import config
from data_loader import load_and_clean_transactions
from db_utils import initialize_database

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def check_db_integrity():
    print("=== Checking Database Integrity ===")
    
    # 1. Initialize Config and Loader
    config_mgr = ConfigManager(config.get_app_data_dir())
    db_path = config_mgr.gui_config.get("transactions_file")
    
    if not db_path or not str(db_path).endswith(".db"):
        print(f"Error: Configured transactions file is not a DB: {db_path}")
        return

    print(f"Database Path: {db_path}")
    if not os.path.exists(db_path):
        print("Error: Database file not found.")
        return

    # No DataLoader class
    
    # 2. Load Data (cleaned)
    print("\n--- Loading Data via load_and_clean_transactions ---")
    
    df, _, _, _, _, _, _ = load_and_clean_transactions(
        source_path=db_path,
        account_currency_map=config_mgr.gui_config.get("account_currency_map", {}),
        default_currency=config_mgr.gui_config.get("default_currency", "USD"),
        is_db_source=True
    )
    
    if df is None or df.empty:
        print("Error: DataFrame is empty or failed to load.")
        return

    print(f"Total Rows Loaded: {len(df)}")
    
    # 3. Integrity Checks
    issues_found = 0
    
    # A. Duplicates
    # Check for exact duplicates excluding original_index
    cols_for_dup_check = [c for c in df.columns if c != "original_index"]
    duplicates = df[df.duplicated(subset=cols_for_dup_check, keep=False)]
    if not duplicates.empty:
        print(f"\n[WARN] Found {len(duplicates)} potential duplicate rows:")
        print(duplicates[['Date', 'Type', 'Symbol', 'Quantity', 'Total Amount']].head(10))
        issues_found += 1
    else:
        print("\n[OK] No exact duplicates found.")

    # B. Missing Critical Fields
    # Date, Type, Symbol should not be null/empty
    missing_critical = df[df['Date'].isna() | (df['Symbol'] == "") | (df['Type'] == "")]
    if not missing_critical.empty:
        print(f"\n[FAIL] Found {len(missing_critical)} rows with missing Date, Symbol, or Type:")
        print(missing_critical[['original_index', 'Date', 'Type', 'Symbol']].head())
        issues_found += 1
    else:
        print("[OK] Critical fields (Date, Symbol, Type) are populated.")

    # C. Invalid Numeric Values
    # Quantity: Critical for Buy/Sell/DivReinvest. Optional for Div (Cash).
    # Price: Critical for Buy/Sell. Optional for Div.
    # Total Amount: Critical for all.
    
    # Check Quantity
    qty_critical_types = ['buy', 'sell', 'dividend_reinvest']
    bad_qty = df[df['Quantity'].isna() & df['Type'].str.lower().isin(qty_critical_types)]
    if not bad_qty.empty:
         print(f"\n[FAIL] Found {len(bad_qty)} critical rows with NaN Quantity (Buy/Sell/Reinvest):")
         print(bad_qty[['Date', 'Type', 'Symbol', 'Quantity']].head())
         issues_found += 1
    else:
         print("[OK] Quantity is valid for Buy/Sell/Reinvest.")

    # Check Price
    price_critical_types = ['buy', 'sell']
    bad_price = df[df['Price/Share'].isna() & df['Type'].str.lower().isin(price_critical_types)]
    if not bad_price.empty:
         print(f"\n[FAIL] Found {len(bad_price)} critical rows with NaN Price (Buy/Sell):")
         print(bad_price[['Date', 'Type', 'Symbol', 'Price/Share']].head())
         issues_found += 1
    else:
         print("[OK] Price is valid for Buy/Sell.")
         
    # Check Total Amount (General)
    bad_amt = df[df['Total Amount'].isna()]
    if not bad_amt.empty:
         print(f"\n[FAIL] Found {len(bad_amt)} rows with NaN Total Amount:")
         print(bad_amt[['Date', 'Type', 'Symbol', 'Total Amount']].head())
         issues_found += 1
    else:
         print("[OK] Total Amount is populated.")

    # D. Future Dates
    # Transactions should typically be in the past or today (unless user entered future ones)
    now = pd.Timestamp.now()
    future_tx = df[df['Date'] > now]
    if not future_tx.empty:
        print(f"\n[INFO] Found {len(future_tx)} transactions with future dates (might be intentional):")
        print(future_tx[['Date', 'Type', 'Symbol']].head())
    else:
        print("[OK] No future transactions found.")

    # E. Account consistency
    # Check if any accounts have transactions but are not in the 'selected_accounts' (though that's config, not DB integrity)
    # Check for empty Account names
    empty_accounts = df[df['Account'] == ""]
    if not empty_accounts.empty:
        print(f"\n[WARN] Found {len(empty_accounts)} transactions with empty Account field.")
        issues_found += 1
    else:
        print("[OK] All transactions have an Account assigned.")
        
    # F. Split Anomalies
    # Check for 'Split' type with missing Ratio
    splits = df[df['Type'].str.lower() == 'split']
    if not splits.empty:
        # Check 'Split Ratio' column if exists
        if 'Split Ratio' in df.columns:
            bad_splits = splits[splits['Split Ratio'].isna() | (splits['Split Ratio'] == 0)]
            if not bad_splits.empty:
                 print(f"\n[WARN] Found {len(bad_splits)} Split transactions with missing or zero Ratio:")
                 print(bad_splits[['Date', 'Symbol', 'Split Ratio']].head())
                 issues_found += 1
            else:
                 print(f"[OK] All {len(splits)} Split transactions have valid Ratios.")
        else:
             print("\n[WARN] 'Split Ratio' column missing from dataframe!")
             issues_found += 1

    print("\n" + "="*30)
    print(f"Integrity Check Complete. Issues Found: {issues_found}")
    print("="*30)

if __name__ == "__main__":
    check_db_integrity()
