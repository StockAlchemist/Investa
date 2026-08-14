"""Transaction routes: CRUD, document parsing, batch import, IBKR sync."""

import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional

import fastapi
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

import config
from db_utils import (
    add_transaction_to_db,
    delete_transaction_from_db,
    get_db_connection,
    update_transaction_in_db,
)
from finutils import is_cash_symbol
from ibkr_connector import IBKRConnector
from server.auth import User
from server.dependencies import (
    get_config_manager,
    get_current_user,
    get_transaction_data,
    get_user_db_connection,
)
from server.pdf_parser import extract_transactions_from_file
from server.route_utils import clean_nans

# Project root (…/Investa) for temp upload storage
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

router = APIRouter()


def reload_data_and_clear_cache(current_user: Optional[User] = None):
    """Proxy to api.reload_data_and_clear_cache (lazy import — api.py includes this router)."""
    from server.portfolio_service import reload_data_and_clear_cache as _impl

    return _impl(current_user)


@router.get("/transactions")
def get_transactions(
    accounts: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data),
):
    """
    Returns the list of transactions, optionally filtered by account.

    Args:
        accounts (List[str], optional): List of account names.
        data (tuple): Dependency injection.

    Returns:
        List[Dict]: A list of transaction records.
    """
    df, _, _, _, _, _, _, _ = data

    if df.empty:
        return []

    try:
        # Filter by accounts if provided
        if accounts:
            df = df[df["Account"].isin(accounts)].copy()

        # Ensure we include the ID in the response if it's in the index or a column
        if "original_index" in df.columns:
            # Make sure original_index is available as 'id' for the frontend
            df["id"] = df["original_index"]
        elif df.index.name == "original_index" or "original_index" in df.index.names:
            df["id"] = df.index.get_level_values("original_index")

        # Sort by Date descending, then ID descending
        if "Date" in df.columns and "id" in df.columns:
            df = df.sort_values(by=["Date", "id"], ascending=[False, False])
        elif "Date" in df.columns:
            df = df.sort_values(by="Date", ascending=False)

        # Handle NaNs and convert to list of dicts
        df = df.where(pd.notnull(df), None)

        records = df.to_dict(orient="records")
        return clean_nans(records)

    except Exception as e:
        logging.error(f"Error getting transactions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch transactions")


class TransactionInput(BaseModel):
    Date: str
    Type: str
    Symbol: str
    Quantity: float
    Price_Share: float = Field(0.0, alias="Price/Share")
    Total_Amount: Optional[float] = Field(None, alias="Total Amount")
    Commission: float = 0.0
    Account: str
    Split_Ratio: Optional[float] = Field(None, alias="Split Ratio")
    Note: Optional[str] = None
    Local_Currency: str = Field(..., alias="Local Currency")
    To_Account: Optional[str] = Field(None, alias="To Account")
    Tags: Optional[str] = None
    Auto_Add_Cash: bool = Field(False, alias="Auto-add Cash")

    model_config = ConfigDict(populate_by_name=True)


class TransactionBatchInput(BaseModel):
    transactions: List[TransactionInput]
    auto_add_cash: bool = False


def _handle_auto_cash_generation(
    conn: sqlite3.Connection,
    tx_data: Dict[str, Any],
    account_cash_mode_map: Optional[Dict[str, str]] = None,
):
    """
    Automatically creates associated $CASH transactions for a stock Buy or Sell.

    Posts the trade principal (gross, commission excluded) as one leg and the
    commission, when there is one, as a separate fee leg. Accounts in *Auto*
    cash mode are skipped: the valuation engine already derives cash deltas
    from the trade rows themselves there, so explicit legs double-count.
    """
    tx_type = tx_data.get("Type", "").strip().lower()
    if tx_type not in ["buy", "sell"]:
        return

    symbol = tx_data.get("Symbol", "")
    if is_cash_symbol(symbol):
        return

    account = tx_data.get("Account", "")
    # Gated here rather than at the call sites so every route is covered. The
    # clients hide the option for Auto accounts, but a stale client or a direct
    # API call must not be able to corrupt an Auto account's cash balance.
    acc_mode = str((account_cash_mode_map or {}).get(account, "Manual")).strip().lower()
    if acc_mode == "auto":
        logging.info(
            f"Skipping auto-cash generation for '{account}': account is in Auto cash mode."
        )
        return

    date_str = tx_data.get("Date")
    local_currency = tx_data.get("Local Currency", "USD")
    user_id = tx_data.get("user_id")

    qty = abs(float(tx_data.get("Quantity") or 0))
    price = float(tx_data.get("Price/Share") or 0)
    commission = float(tx_data.get("Commission") or 0)

    # Gross principal, commission excluded — the commission is posted as its own
    # leg below. "Total Amount" cannot be used here: the web and SwiftUI forms
    # and ibkr_connector all fold commission into it, while imported and legacy
    # rows store it net of commission, so reading it charges the commission
    # twice for the former. qty * price is unambiguous under either convention.
    # Fall back to "Total Amount" only when the price is missing (batch/PDF
    # imports default Price/Share to 0), backing the commission out by direction.
    principal = qty * price
    if principal <= 1e-9:
        total_amt = tx_data.get("Total Amount")
        if total_amt is not None and pd.notna(total_amt):
            settled = abs(float(total_amt))
            principal = (
                settled - commission if tx_type == "buy" else settled + commission
            )

    if principal <= 1e-9:
        logging.warning(
            f"Auto-cash: no usable principal for {tx_type} {symbol} in '{account}'; skipping cash legs."
        )
        return

    def post_leg(leg_type: str, amount: float, note: str):
        """Post one $CASH leg. 'Total Amount' is stored signed, matching the
        convention the clients write: Buy/Withdrawal negative, Sell positive."""
        add_transaction_to_db(
            conn,
            {
                "Date": date_str,
                "Type": leg_type,
                "Symbol": "$CASH",
                "Quantity": abs(amount),
                "Price/Share": 1.0,
                "Total Amount": -abs(amount)
                if leg_type in ("Buy", "Withdrawal")
                else abs(amount),
                "Account": account,
                "Local Currency": local_currency,
                "Note": note,
                "user_id": user_id,
            },
        )

    # Funding a buy sells $CASH; proceeds from a sell buy $CASH.
    verb = "Buy" if tx_type == "buy" else "Sell"
    post_leg(
        "Sell" if tx_type == "buy" else "Buy",
        principal,
        f"Auto-cash for {verb} {symbol}",
    )
    if commission > 0:
        post_leg("Withdrawal", commission, f"Auto-cash Fee for {verb} {symbol}")


@router.post("/transactions")
def create_transaction(
    transaction: TransactionInput,
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
):
    """
    Creates a new transaction.

    Args:
        transaction (TransactionInput): The transaction data payload.
        data (tuple): Dependency injection.

    Returns:
        Dict: Status message and the new transaction ID.
    """
    try:
        _, _, _, _, _, account_cash_mode_map, db_path, _ = data
        conn = get_db_connection(db_path)
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        # Convert Pydantic model to dict with correct keys for DB
        tx_data = transaction.dict(by_alias=True)
        tx_data["user_id"] = current_user.id

        success, new_id = add_transaction_to_db(conn, tx_data)

        if success and tx_data.get("Auto-add Cash"):
            _handle_auto_cash_generation(conn, tx_data, account_cash_mode_map)

        conn.close()

        if success:
            reload_data_and_clear_cache(
                current_user
            )  # Refresh transaction and summary caches
            return {"status": "success", "id": new_id, "message": "Transaction added"}
        else:
            raise HTTPException(
                status_code=500, detail="Failed to add transaction to database"
            )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error adding transaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to add transaction")


@router.post("/transactions/parse_document")
def parse_document(
    file: UploadFile = File(...),
    cash_mode: Optional[str] = fastapi.Form(None),
    account_override: Optional[str] = fastapi.Form(None),
    current_user: User = Depends(get_current_user),
):
    """
    Parses a brokerage statement (PDF or Image) and returns extracted transactions.
    Supports IBKR trade confirmations (deterministic) and general statements (AI fallback).
    """
    try:
        # Save the uploaded file temporarily to pass to the parser
        # Use a safe unique name
        temp_dir = os.path.join(project_root, "data", "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(
            temp_dir, f"{current_user.id}_{int(time.time())}_{file.filename}"
        )

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse the document
        transactions = []
        try:
            transactions = extract_transactions_from_file(
                temp_file_path,
                user_id=current_user.id,
                cash_mode=cash_mode,
                account_override=account_override,
            )
        finally:
            # Always clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        if not transactions:
            return {
                "status": "success",
                "message": "No transactions found in document.",
                "transactions": [],
                "count": 0,
            }

        return {
            "status": "success",
            "message": f"Successfully extracted {len(transactions)} transactions.",
            "transactions": transactions,
            "count": len(transactions),
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error parsing document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to parse document")


@router.post("/transactions/batch")
def add_transactions_batch(
    payload: TransactionBatchInput,
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
):
    """
    Adds a batch of transactions to the database. Used for the Review &
    Confirm step after document parsing.

    Auto-add-cash behaviour is decided **per account** rather than globally:
    the explicit $CASH legs only need to be generated for accounts that are
    in *Manual* cash mode. Accounts already in *Auto* cash mode have the
    engine generate cash deltas on the fly during valuation, so adding
    explicit legs there double-counts every imported trade. ``payload.auto_add_cash``
    therefore acts as a user opt-in that still gets gated by the account's
    cash mode — the gate itself lives in ``_handle_auto_cash_generation`` so
    that the single-transaction route is covered by the same rule.
    """
    try:
        _, _, _, _, _, account_cash_mode_map, db_path, _ = data
        conn = get_db_connection(db_path)
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        imported_count = 0
        errors = []
        cash_mode_map = account_cash_mode_map or {}

        for tx_input in payload.transactions:
            # Convert back to dict for the lower-level add_transaction_to_db
            # This handles JSON key aliases correctly
            tx_data = tx_input.model_dump(by_alias=True)

            # Set the user_id for isolation
            tx_data["user_id"] = current_user.id

            success, error = add_transaction_to_db(conn, tx_data)
            if success:
                imported_count += 1
                if payload.auto_add_cash:
                    # Accounts absent from the user's config default to Manual
                    # inside the helper — opt-in stays safer for new accounts.
                    _handle_auto_cash_generation(conn, tx_data, cash_mode_map)
            else:
                errors.append({"symbol": tx_input.Symbol, "error": error})

        conn.close()

        if imported_count > 0:
            reload_data_and_clear_cache(current_user)

        return {
            "status": "success",
            "message": f"Successfully imported {imported_count} transactions.",
            "count": imported_count,
            "errors": errors,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in batch import: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to batch import transactions"
        )


@router.put("/transactions/{transaction_id}")
def update_transaction(
    transaction_id: int,
    transaction: TransactionInput,
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
):
    """
    Updates an existing transaction.

    Args:
        transaction_id (int): The ID of the transaction to update.
        transaction (TransactionInput): The updated transaction data.
        data (tuple): Dependency injection.

    Returns:
        Dict: Status message.
    """
    try:
        _, _, _, _, _, _, db_path, _ = data
        conn = get_db_connection(db_path)
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        tx_data = transaction.dict(by_alias=True)
        tx_data["user_id"] = current_user.id

        success = update_transaction_in_db(conn, transaction_id, tx_data)
        conn.close()

        if success:
            reload_data_and_clear_cache(current_user)
            return {"status": "success", "message": "Transaction updated"}
        else:
            raise HTTPException(
                status_code=404, detail="Transaction not found or update failed"
            )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating transaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update transaction")


@router.delete("/transactions/{transaction_id}")
def delete_transaction(
    transaction_id: int,
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
):
    """
    Deletes a transaction.

    Args:
        transaction_id (int): ID of the transaction to delete.
        data (tuple): Dependency injection.

    Returns:
        Dict: Status message.
    """
    try:
        _, _, _, _, _, _, db_path, _ = data
        conn = get_db_connection(db_path)
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        success = delete_transaction_from_db(conn, transaction_id)
        conn.close()

        if success:
            reload_data_and_clear_cache(current_user)
            return {"status": "success", "message": "Transaction deleted"}
        else:
            raise HTTPException(
                status_code=404, detail="Transaction not found or delete failed"
            )

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting transaction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete transaction")


class HoldingTagUpdate(BaseModel):
    account: str
    symbol: str
    tags: str

    model_config = ConfigDict(populate_by_name=True)


@router.post("/holdings/update_tags")
def update_holding_tags(
    update_data: HoldingTagUpdate,
    data: tuple = Depends(get_transaction_data),
    current_user: User = Depends(get_current_user),
):
    """
    Updates tags for all transactions associated with a specific holding (Symbol + Account).
    """
    try:
        _, _, _, _, _, _, db_path, _ = data
        conn = get_db_connection(db_path)
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        cursor = conn.cursor()
        # Clean tags
        tags_value = update_data.tags.strip()

        # Update all transactions for this symbol and account
        # Note: We probably want to update ALL types (Buy, Sell, Div, etc) so they stay grouped?
        # Or just open positions?
        # ShareSight groups by holding. So updating all history is consistent.
        sql = "UPDATE transactions SET Tags = ? WHERE Symbol = ? AND Account = ?"
        cursor.execute(sql, (tags_value, update_data.symbol, update_data.account))
        conn.commit()
        rows_affected = cursor.rowcount
        conn.close()

        reload_data_and_clear_cache(current_user)
        return {
            "status": "success",
            "message": f"Updated tags for {rows_affected} transactions",
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating holding tags: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update holding tags")


@router.post("/sync/ibkr")
async def sync_ibkr(
    current_user: User = Depends(get_current_user),
    data: tuple = Depends(get_transaction_data),
    config_manager=Depends(get_config_manager),
):
    """
    Syncs transactions from IBKR via Flex Web Service.
    """
    try:
        # Prioritize values from config_manager (persisted in manual_overrides.json)
        # Fall back to config.py (environment variables)
        token = config_manager.manual_overrides.get("ibkr_token") or config.IBKR_TOKEN
        query_id = (
            config_manager.manual_overrides.get("ibkr_query_id") or config.IBKR_QUERY_ID
        )

        if not token or not query_id:
            # We check here to provide a clear error to the user via the API
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "code": "CONFIG_MISSING",
                    "message": "IBKR API not configured. Please set IBKR Token and Query ID in your settings.",
                },
            )

        _, _, _, _, _, _, db_path, _ = data
        conn = get_db_connection(db_path)
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        connector = IBKRConnector(token=token, query_id=query_id)
        # Flex sync involves network I/O, run in threadpool to avoid blocking event loop
        try:
            new_transactions = await run_in_threadpool(connector.sync)
        except Exception as sync_err:
            logging.error(f"Error in connector.sync: {sync_err}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "code": "SYNC_FAILED",
                    "message": "IBKR sync failed",
                },
            )

        if not new_transactions:
            return {
                "status": "success",
                "message": "Sync successful, but no new transactions were found in the report.",
                "added_count": 0,
            }

        staged_count = 0
        duplicate_count = 0

        # Helper logic for staging (since we had issues adding it to db_utils)
        def _stage_tx(conn, tx_data, u_id):
            ext_id = tx_data.get("ExternalID")
            cursor = conn.cursor()
            if ext_id:
                # Check main table
                cursor.execute(
                    "SELECT id FROM transactions WHERE ExternalID = ?", (ext_id,)
                )
                if cursor.fetchone():
                    return False, "duplicate_main"
                # Check pending table
                cursor.execute(
                    "SELECT id FROM pending_transactions WHERE ExternalID = ?",
                    (ext_id,),
                )
                if cursor.fetchone():
                    return False, "duplicate_pending"

            db_cols = [
                "Date",
                "Type",
                "Symbol",
                "Quantity",
                "Price/Share",
                "Total Amount",
                "Commission",
                "Account",
                "Split Ratio",
                "Note",
                "Local Currency",
                "To Account",
                "Tags",
                "ExternalID",
                "user_id",
            ]
            placeholders = ", ".join(
                [f":{c.replace('/', '_').replace(' ', '_')}" for c in db_cols]
            )
            cols_str = ", ".join([f'"{c}"' for c in db_cols])

            sql = f"INSERT INTO pending_transactions ({cols_str}) VALUES ({placeholders});"
            sql_data = {}
            for col in db_cols:
                val = tx_data.get(col) if col != "user_id" else u_id

                # Fallback for Total Amount if missing (or provided as 'Amount')
                if col == "Total Amount" and val is None:
                    val = tx_data.get("Amount")  # Try alternate key
                    if val is None:  # Calculate
                        q = tx_data.get("Quantity", 0)
                        p = tx_data.get("Price/Share", 0)
                        c = tx_data.get("Commission", 0)
                        t = tx_data.get("Type", "").upper()
                        if q and p:
                            val = (q * p) + (c if t == "BUY" else -c)

                if col == "Type" and isinstance(val, str):
                    val = val.strip().title()
                if col == "Date" and isinstance(val, (datetime, date)):
                    val = val.strftime("%Y-%m-%d")
                sql_data[col.replace("/", "_").replace(" ", "_")] = (
                    None if pd.isna(val) else val
                )

            cursor.execute(sql, sql_data)
            return True, cursor.lastrowid

        for tx_data in new_transactions:
            success, result = _stage_tx(conn, tx_data, current_user.id)
            if success:
                staged_count += 1
            elif result in ["duplicate_main", "duplicate_pending"]:
                duplicate_count += 1

        conn.commit()
        conn.close()

        return {
            "status": "success",
            "message": f"Sync successful. {staged_count} transactions staged for review ({duplicate_count} skipped as duplicates).",
            "staged_count": staged_count,
            "duplicate_count": duplicate_count,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error during IBKR sync: {e}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"status": "error", "message": "IBKR sync failed"}
        )


@router.get("/sync/ibkr/pending")
def get_pending_ibkr(
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection),
):
    """Fetch pending transactions for review."""
    try:
        query = "SELECT * FROM pending_transactions ORDER BY Date DESC"
        df = pd.read_sql_query(query, conn)
        return df.to_dict(orient="records")
    except Exception as e:
        logging.error(f"Error fetching pending IBKR transactions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to fetch pending transactions"
        )


@router.post("/sync/ibkr/approve")
def approve_ibkr(
    ids: List[int],
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection),
):
    """Approve and move transactions from staging to main table."""
    try:
        cursor = conn.cursor()
        approved_count = 0

        # Columns in pending (exclude ID)
        cols = [
            "Date",
            "Type",
            "Symbol",
            "Quantity",
            "Price/Share",
            "Total Amount",
            "Commission",
            "Account",
            "Split Ratio",
            "Note",
            "Local Currency",
            "To Account",
            "Tags",
            "ExternalID",
            "user_id",
        ]
        cols_str = ", ".join([f'"{c}"' for c in cols])

        for p_id in ids:
            # Fetch from pending
            cursor.execute(
                f"SELECT {cols_str} FROM pending_transactions WHERE id = ?", (p_id,)
            )
            row = cursor.fetchone()
            if row:
                # Insert into main table
                placeholders = ", ".join(["?"] * len(cols))
                cursor.execute(
                    f"INSERT INTO transactions ({cols_str}) VALUES ({placeholders})",
                    row,
                )
                # Delete from pending
                cursor.execute("DELETE FROM pending_transactions WHERE id = ?", (p_id,))
                approved_count += 1

        conn.commit()
        if approved_count > 0:
            reload_data_and_clear_cache(current_user)

        return {
            "status": "success",
            "message": f"Successfully approved {approved_count} transactions.",
            "count": approved_count,
        }
    except Exception as e:
        conn.rollback()
        logging.error(f"Error approving pending IBKR transactions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to approve pending transactions"
        )


@router.post("/sync/ibkr/reject")
def reject_ibkr(
    ids: List[int],
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection),
):
    """Discard pending transactions."""
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"DELETE FROM pending_transactions WHERE id IN ({','.join(['?'] * len(ids))})",
            (*ids,),
        )
        deleted_count = cursor.rowcount
        conn.commit()
        if deleted_count > 0:
            reload_data_and_clear_cache(current_user)
        return {
            "status": "success",
            "message": f"Discarded {deleted_count} transactions.",
            "count": deleted_count,
        }
    except Exception as e:
        conn.rollback()
        logging.error(f"Error rejecting pending IBKR transactions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to reject pending transactions"
        )
