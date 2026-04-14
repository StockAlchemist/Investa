import re
import pdfplumber
import logging
from datetime import datetime
from typing import List, Dict, Any
import mimetypes
import os

from server.vision_parser import parse_document_with_ai

def parse_ibkr_pdf(file_path: str) -> List[Dict[str, Any]]:
    extracted_transactions = []
    
    # Regex to match the trade confirmation row.
    # U13340051 AAPL 2026-03-09, 13:07:27 2026-03-10 - SELL -78 257.337948718 20,072.36 -1.02 0.00 LMT C;P
    # Account   Symbol Date/Time          SettleDate Exchange Type Qty  Price         Amount    Comm  Fee Code
    # The crucial part is "Exchange" = "-" to avoid double counting the sub-exchanges like NASDAQ and NYSE
    
    # We'll use a regex that matches rows where the Exchange is '-'
    # Group extraction pattern:
    row_pattern = re.compile(
        r"^(?P<account>U\d+)\s+"                 # Account
        r"(?P<symbol>[A-Z0-9]+(?: [A-Z0-9]+)?)\s+" # Symbol
        r"(?P<datetime>\d{4}-\d{2}-\d{2},\s\d{2}:\d{2}:\d{2})\s+" # Date/Time
        r"(?P<settle_date>\d{4}-\d{2}-\d{2})\s+" # Settle Date
        r"(?P<exchange>-)\s+"                    # Exchange (must be hyphen)
        r"(?P<type>BUY|SELL)\s+"                 # Type
        r"(?P<qty>-?[\d,]+(?:\.\d+)?)\s+"        # Quantity
        r"(?P<price>[\d,]+(?:\.\d+)?)\s+"        # Price
        r"(?P<amount>-?[\d,]+(?:\.\d+)?)\s+"     # Amount / Proceeds
        r"(?P<commission>-?[\d,]+(?:\.\d+)?)\s+" # Commission
        r"(?P<fee>[\d,]+(?:\.\d+)?)\s+"          # Fee
    )
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                for line in text.split('\n'):
                    match = row_pattern.search(line)
                    if match:
                        data = match.groupdict()
                        
                        # Parse Full Timestamp
                        # IBKR Format: "2026-03-09, 13:07:27" -> "2026-03-09 13:07:27"
                        raw_ts = data['datetime'].replace(', ', ' ').strip()
                        
                        # Parse Type
                        tx_type = data['type']
                        
                        # Parse Quantities and Amounts
                        qty = float(data['qty'].replace(',', ''))
                        price = float(data['price'].replace(',', ''))
                        amount = float(data['amount'].replace(',', ''))
                        comm = float(data['commission'].replace(',', ''))
                        fee = float(data['fee'].replace(',', ''))
                        
                        # IBKR represents sell quantity as negative, backend expects positive quantity for SELL and BUY
                        # But Proceeds / Amounts might be positive or negative.
                        # Wait, in the pdf: SELL -78, Proceeds 20,072.36 (Positive).
                        # BUY 16, Proceeds -10,176.56 (Negative).
                        # The database generally expects Total Amount to be the magnitude or positive if it's an outflow?
                        # Actually TransactionInput for "Total_Amount" can handle signs or just positive. Let's make Qty positive.
                        abs_qty = abs(qty)
                        
                        # Commission: from pdf -1.02. Backend usually takes positive comm.
                        abs_comm = abs(comm) + abs(fee)
                        
                        tx = {
                            "Date": raw_ts, # Store full timestamp for chronological order
                            "Type": tx_type.capitalize(), # "Buy" or "Sell"
                            "Symbol": data['symbol'].replace(" ", "."),
                            "Quantity": abs_qty,
                            "Price/Share": price,
                            "Total Amount": amount, 
                            "Commission": abs_comm,
                            "Account": data['account'],
                            "Local Currency": "USD", # Default to USD for now based on header "Trades ... Stocks ... USD"
                            "Note": "Imported from IBKR PDF"
                        }
                        
                        extracted_transactions.append(tx)
        
        # Sort transactions chronologically by Date
        extracted_transactions.sort(key=lambda x: x['Date'])
        
    except Exception as e:
        logging.error(f"Error parsing IBKR PDF: {e}", exc_info=True)
        raise
        
    return extracted_transactions


def extract_transactions_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Main entry point for transaction extraction.
    Attempts deterministic IBKR parsing first, then falls back to AI Vision.
    """
    ext = os.path.splitext(file_path)[1].lower()
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Pre-checks
    if not mime_type:
        if ext == '.pdf':
            mime_type = 'application/pdf'
        elif ext in ['.png', '.jpg', '.jpeg']:
            mime_type = f'image/{ext[1:]}'
        else:
            mime_type = 'application/octet-stream'

    # 1. If it's a PDF, try IBKR Regex first
    if mime_type == 'application/pdf':
        try:
            logging.info(f"Parser: Attempting deterministic IBKR parse for {file_path}...")
            results = parse_ibkr_pdf(file_path)
            if results:
                logging.info(f"Parser: Successfully parsed {len(results)} transactions using IBKR regex.")
                return results
        except Exception as e:
            logging.warning(f"Parser: IBKR regex parse failed or not applicable: {e}")

    # 2. Fallback to AI Vision Parser
    logging.info(f"Parser: Falling back to AI Vision for {file_path} (Mime: {mime_type})...")
    return parse_document_with_ai(file_path, mime_type)

