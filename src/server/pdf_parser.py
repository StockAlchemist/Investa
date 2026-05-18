import re
import pdfplumber
import logging
import os
import mimetypes
from typing import List, Dict, Any, Optional

# When pdfplumber merges several logical sections into one wide table (common
# in multi-account IBKR Activity Statements), the first cell of a row can act
# as an inline section header. Map those values to the section name the rest
# of the loop expects. "SKIP" is a sentinel for rows we should not import as
# transactions (e.g. accrual tracking tables).
_INLINE_SECTION_HEADERS = {
    "trades": "Trades",
    "transfers": "Transfers",
    "dividends": "Dividends",
    "withholding tax": "Tax",
    "deposits & withdrawals": "Cash",
    "other fees": "Fees",
    "interest": "Interest",
    "change in dividend accruals": "SKIP",
    "change in interest accruals": "SKIP",
    "financial instrument information": "SKIP",
    "codes": "SKIP",
    "notes/legal notes": "SKIP",
    "disclosure": "SKIP",
}


def parse_ibkr_pdf(file_path: str, user_id: int, cash_mode: str, account_override: str) -> List[Dict[str, Any]]:
    transactions = []
    account_name = account_override or "IBKR Account"
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if not tables: continue

                for table in tables:
                    if not table or not table[0]: continue
                    header_text = " ".join([str(x) for x in table[0] if x is not None])

                    section = None
                    if "Trades" in header_text: section = "Trades"
                    elif "Transfers" in header_text: section = "Transfers"
                    elif "Dividends" in header_text: section = "Dividends"
                    elif "Withholding Tax" in header_text: section = "Tax"
                    elif "Deposits & Withdrawals" in header_text: section = "Cash"
                    elif "Other Fees" in header_text: section = "Fees"
                    elif "Interest" in header_text: section = "Interest"

                    if not section: continue

                    # Track section state across rows — a single pdfplumber
                    # table can splice several IBKR logical sections together.
                    current_section = section

                    for row in table:
                        if not row or len(row) < 3: continue
                        row = [str(x).replace("\n", " ").strip() if x is not None else "" for x in row]

                        first_val = row[0]

                        # Mid-table section switch: when first cell exactly
                        # matches a known section heading, swap and skip.
                        fv_key = first_val.lower()
                        if fv_key in _INLINE_SECTION_HEADERS:
                            current_section = _INLINE_SECTION_HEADERS[fv_key]
                            continue
                        if current_section == "SKIP":
                            continue
                        # Rebind for the rest of the loop body below.
                        section = current_section

                        if any(x in first_val for x in [section, "Symbol", "Date", "Total", "Stocks", "USD", "Description"]): continue
                        if not first_val or first_val == "None": continue

                        try:
                            if section == "Trades":
                                if row[0].startswith("U"):
                                    account = row[0]
                                    sym, date_str, qty_str, price_str, proceeds_str, comm_str = row[1], row[2].split(",")[0].strip(), row[6], row[7], row[9], row[10]
                                else:
                                    account = account_name
                                    sym, date_str, _, qty_str, price_str, _, proceeds_str, comm_str = row[0], row[1].split(",")[0].strip(), row[2], row[3], row[4], row[5], row[6], row[7]
                                
                                q_val = float(qty_str.replace(",", ""))
                                transactions.append({
                                    "Date": date_str, "Type": "Buy" if q_val > 0 else "Sell", "Symbol": sym,
                                    "Quantity": abs(q_val), "Price/Share": float(price_str.replace(",", "")),
                                    "Total Amount": abs(float(proceeds_str.replace(",", ""))),
                                    "Commission": abs(float(comm_str.replace(",", ""))),
                                    "Account": account, "Note": f"IBKR Trade", "Local Currency": "USD", "user_id": user_id
                                })

                            elif section == "Transfers":
                                if row[0].startswith("U"):
                                    account = row[0]
                                    sym, date_str, t_type, qty_str, mval_str = row[1], row[2], row[3], row[7], row[9]
                                else:
                                    account = account_name
                                    sym, date_str, t_type, qty_str, mval_str = row[0], row[1], row[2], row[6], row[8]
                                q_val = float(qty_str.replace(",", ""))
                                transactions.append({
                                    "Date": date_str, "Type": "Transfer", "Symbol": sym,
                                    "Quantity": abs(q_val), "Price/Share": float(mval_str.replace(",", "")) / abs(q_val) if q_val != 0 else 0,
                                    "Total Amount": abs(float(mval_str.replace(",", ""))),
                                    "Commission": 0.0, "Account": account, "Note": f"IBKR Transfer ({t_type})", "Local Currency": "USD", "user_id": user_id
                                })
                                
                            elif section in ["Dividends", "Tax", "Cash", "Fees", "Interest"]:
                                # IBKR Activity Statements vary column counts:
                                # 3-4 columns for single-account dividends,
                                # but Withholding Tax tables in multi-account
                                # PDFs balloon to 15 columns with the amount at
                                # index 6 (not 2). Locate date / description /
                                # amount by content rather than fixed index so
                                # both layouts work.
                                if row[0].startswith("U"):
                                    account = row[0]
                                    scan_start = 1
                                else:
                                    account = account_name
                                    scan_start = 0

                                # Date: first cell matching YYYY-MM-DD.
                                date_str = None
                                for c in row[scan_start:]:
                                    if c and re.match(r"^\d{4}-\d{2}-\d{2}$", c):
                                        date_str = c
                                        break
                                if not date_str: continue

                                # Amount: rightmost cell that parses as a
                                # signed decimal. Scanning right-to-left skips
                                # the trailing empty "Code" column naturally.
                                amt_str = None
                                for c in reversed(row):
                                    if not c: continue
                                    s = c.replace(",", "").strip()
                                    if not s: continue
                                    try:
                                        float(s)
                                        amt_str = c
                                        break
                                    except ValueError:
                                        continue
                                if amt_str is None: continue

                                # Description: longest concat of non-numeric,
                                # non-date text cells. Single-char codes like
                                # "R" or "Po" at the end are filtered by
                                # length to avoid polluting symbol detection.
                                desc_pieces = []
                                for c in row:
                                    if not c or c == date_str or c == amt_str:
                                        continue
                                    s = c.strip()
                                    if not s: continue
                                    try:
                                        float(s.replace(",", ""))
                                        continue
                                    except ValueError:
                                        pass
                                    if len(s) > 2:
                                        desc_pieces.append(s)
                                desc = " ".join(desc_pieces).strip()
                                if not desc: continue

                                if "Total" in desc or "Starting" in desc: continue

                                amt = float(amt_str.replace(",", ""))
                                sym = "$CASH"
                                # Only treat the parenthesised payload as a
                                # ticker identifier (CUSIP/ISIN) when the
                                # prefix is plausibly a ticker — short and
                                # alphanumeric. Otherwise this misparses
                                # IBKR Interest descriptions like "USD IBKR
                                # Managed Securities (SYEP) for Apr-2026"
                                # into a fake symbol.
                                if "(" in desc:
                                    prefix = desc.split("(", 1)[0].strip()
                                    tokens = prefix.split()
                                    if (
                                        prefix
                                        and len(prefix) <= 12
                                        and len(tokens) <= 2
                                        and prefix.replace(" ", "").replace(".", "").replace("-", "").isalnum()
                                    ):
                                        sym = prefix

                                l_desc = desc.lower()
                                t_type = "Other"
                                if section == "Dividends": t_type = "Dividend"
                                elif section == "Tax": t_type = "Tax"
                                elif section == "Interest": t_type = "Interest"
                                elif section == "Fees": t_type = "Fees"
                                elif "electronic fund" in l_desc or "deposit" in l_desc: t_type = "Deposit"
                                elif "withdrawal" in l_desc: t_type = "Withdrawal"
                                elif "acats transfer" in l_desc or "transfer" in l_desc: t_type = "Transfer"
                                elif "commission adj" in l_desc: t_type = "Deposit" if amt > 0 else "Fees"

                                # Always drop internal sweeps — they aren't
                                # real external cash movements and they double-
                                # count once the engine reconciles trades.
                                if t_type in ["Deposit", "Withdrawal"] and ("sweep" in l_desc or "internal" in l_desc):
                                    continue

                                abs_amt = abs(amt)
                                signed_amt = abs_amt if amt >= 0 else -abs_amt

                                # Canonical row format expected by the engine:
                                #   - $CASH-symbol rows (Deposit / Withdrawal /
                                #     Interest / standalone Fees-on-cash): the
                                #     engine reads `qty` for cash math. Put the
                                #     dollar amount in BOTH Quantity and Total
                                #     Amount so it works whichever side the
                                #     engine ends up reading.
                                #   - Stock-symbol rows (Dividend / Tax / Fees
                                #     tied to a ticker): leave Qty=0; the
                                #     engine's auto-cash path uses Total Amount.
                                if sym == "$CASH":
                                    tx_qty = abs_amt
                                    tx_price = 1.0
                                else:
                                    tx_qty = 0.0
                                    tx_price = 0.0

                                transactions.append({
                                    "Date": date_str, "Type": t_type, "Symbol": sym,
                                    "Quantity": tx_qty,
                                    "Price/Share": tx_price,
                                    "Total Amount": signed_amt,
                                    "Commission": 0.0,
                                    "Account": account, "Note": desc[:100],
                                    "Local Currency": "USD", "user_id": user_id
                                })
                        except (ValueError, IndexError): continue
    except Exception as e:
        logging.error(f"Error parsing IBKR Comprehensive PDF: {e}")
        
    return transactions

def parse_tdameritrade_pdf(file_path: str, user_id: int, cash_mode: str, account_override: str) -> List[Dict[str, Any]]:
    # Fallback to AI for now, or implement a basic text scanner
    return []

def parse_etrade_pdf(file_path: str, user_id: int, cash_mode: str, account_override: str) -> List[Dict[str, Any]]:
    # Fallback to AI for now
    return []

def extract_transactions_from_file(file_path: str, user_id: int, cash_mode: Optional[str] = None, account_override: Optional[str] = None) -> List[Dict[str, Any]]:
    ext = os.path.splitext(file_path)[1].lower()
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if not mime_type:
        if ext == '.pdf': mime_type = 'application/pdf'
        elif ext in ['.png', '.jpg', '.jpeg']: mime_type = f'image/{ext[1:]}'
        else: mime_type = 'application/octet-stream'

    if mime_type == 'application/pdf':
        try:
            # Check if it's IBKR
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for i in range(min(3, len(pdf.pages))):
                    text += pdf.pages[i].extract_text() or ""
            
            if "INTERACTIVE BROKERS" in text.upper() or "IBKR" in text.upper():
                res = parse_ibkr_pdf(file_path, user_id, cash_mode, account_override)
                if res: return res
            elif "TD AMERITRADE" in text.upper():
                res = parse_tdameritrade_pdf(file_path, user_id, cash_mode, account_override)
                if res: return res
            elif "E*TRADE" in text.upper() or "ETRADE" in text.upper():
                res = parse_etrade_pdf(file_path, user_id, cash_mode, account_override)
                if res: return res
                
        except Exception as e:
            logging.warning(f"Deterministic parse failed: {e}")

    # Fallback to AI Vision Parser
    try:
        from server.vision_parser import parse_document_with_ai
        logging.info(f"Parser: Falling back to AI Vision for {file_path}")
        txs = parse_document_with_ai(file_path, mime_type)
        # Inject user_id and account_override if possible
        for t in txs:
            t["user_id"] = user_id
            if account_override and not t.get("Account"):
                t["Account"] = account_override
        return txs
    except ImportError:
        logging.error("AI Vision Parser not available.")
        return []
