import re
import pdfplumber
import logging
import os
import mimetypes
from typing import List, Dict, Any, Optional

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
                    
                    for row in table:
                        if not row or len(row) < 3: continue
                        row = [str(x).replace("\n", " ").strip() if x is not None else "" for x in row]
                        
                        first_val = row[0]
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
                                if row[0].startswith("U"):
                                    account = row[0]
                                    date_str, desc, amt_str = row[1], row[2], row[3]
                                else:
                                    account = account_name
                                    date_str, desc, amt_str = row[0], row[1], row[2]
                                
                                if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str): continue
                                if "Total" in desc or "Starting" in desc: continue
                                
                                amt = float(amt_str.replace(",", ""))
                                sym = "$CASH"
                                if "(" in desc: sym = desc.split("(")[0].strip()
                                
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
                                
                                if cash_mode == "Auto" and t_type in ["Deposit", "Withdrawal"]:
                                    if "sweep" in l_desc or "internal" in l_desc:
                                        continue # skip internal sweeps for auto cash
                                
                                transactions.append({
                                    "Date": date_str, "Type": t_type, "Symbol": sym, "Quantity": 0.0,
                                    "Price/Share": 0.0, "Total Amount": abs(amt), "Commission": 0.0,
                                    "Account": account, "Note": desc[:100], "Local Currency": "USD", "user_id": user_id
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
