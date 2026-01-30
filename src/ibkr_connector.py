# -*- coding: utf-8 -*-
import requests
import xml.etree.ElementTree as ET
import time
import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from config import IBKR_TOKEN, IBKR_QUERY_ID
from datetime import datetime

# --- Constants ---
FLEX_SEND_REQUEST_URL = "https://www.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest"
# Note: The actual URL to get the statement comes from the SendRequest response, 
# but it usually points to the FlexStatementService.GetStatement servlet.

class IBKRConnector:
    """
    Handles communication with the Interactive Brokers Flex Web Service.
    Supports requesting Activity Flex Queries and parsing the resulting XML.
    """
    def __init__(self, token: Optional[str] = None, query_id: Optional[str] = None):
        self.token = token or IBKR_TOKEN
        self.query_id = query_id or IBKR_QUERY_ID
        self.logger = logging.getLogger(__name__)

    def _make_request(self, url: str, params: Dict[str, str]) -> Optional[str]:
        """Generic wrapper for GET requests to IBKR."""
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            self.logger.error(f"IBKR API Request failed: {e}")
            return None

    def request_report(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Initiates a Flex report request.
        Returns (reference_code, download_url)
        """
        if not self.token or not self.query_id:
            self.logger.error("IBKR Token or Query ID not configured.")
            return None, None

        params = {
            "t": self.token,
            "q": self.query_id,
            "v": "3"
        }
        
        xml_resp = self._make_request(FLEX_SEND_REQUEST_URL, params)
        if not xml_resp:
            return None, None

        try:
            root = ET.fromstring(xml_resp)
            status = root.find("Status").text if root.find("Status") is not None else "Fail"
            
            if status == "Success":
                reference_code = root.find("ReferenceCode").text
                url = root.find("Url").text
                self.logger.info(f"IBKR Report request successful. Ref: {reference_code}")
                return reference_code, url
            else:
                err_msg = root.find("ErrorMessage").text if root.find("ErrorMessage") is not None else "Unknown Error"
                self.logger.error(f"IBKR Report request failed: {err_msg}")
                raise Exception(f"IBKR API Error: {err_msg}")
        except Exception as e:
            if "IBKR API Error" in str(e): raise e
            self.logger.error(f"Failed to parse IBKR SendRequest response: {e}")
            raise Exception(f"Failed to initiate IBKR sync: {str(e)}")

    def download_report(self, reference_code: str, url: str) -> Optional[str]:
        """Downloads the actual report XML using the reference code."""
        params = {
            "t": self.token,
            "q": reference_code,
            "v": "3"
        }
        
        # Sometimes IBKR needs a few seconds to prepare the report
        # Increased retries and wait time as first-time reports can be slow
        max_retries = 6
        for i in range(max_retries):
            xml_content = self._make_request(url, params)
            if xml_content:
                # Check if it's an actual report (FlexQueryResponse or FlexStatementResponse)
                if "<FlexQueryResponse" in xml_content or "<FlexStatementResponse" in xml_content:
                     return xml_content
                
                # Log the unexpected content for debugging
                self.logger.warning(f"Unexpected IBKR response (Attempt {i+1}): {xml_content[:200]}...")
                
                # If we got a status=Warn or code=1018, it's still being prepared
                try:
                    if "<Status>Warn</Status>" in xml_content or "<ErrorCode>1018</ErrorCode>" in xml_content:
                         self.logger.warning("IBKR Report still preparing, waiting 10s...")
                         time.sleep(10)
                         continue
                except:
                    pass
            
            self.logger.error(f"Failed to download IBKR report (Attempt {i+1})")
            time.sleep(5)
            
        raise Exception("Failed to download IBKR report after 6 attempts. IBKR might be experiencing delays, or your query is still being generated. Please wait 1-2 minutes and try again.")

    def parse_activity_flex_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        Parses the Activity Flex XML into standardized transaction dictionaries.
        This focuses on:
        - Trades (Trades section)
        - Dividends/Interest/Fees (CashTransactions section)
        """
        transactions = []
        try:
            root = ET.fromstring(xml_content)
            # IBKR Flex XML is deeply nested. 
            # Structure: FlexStatementResponse -> FlexStatements -> FlexStatement -> [Sections]
            
            statements = root.findall(".//FlexStatement")
            if not statements:
                self.logger.warning("No FlexStatement found in XML.")
                return []

            for statement in statements:
                # 1. Parse Trades
                trades = statement.findall(".//Trade")
                for trade in trades:
                    tx = self._map_trade_to_internal(trade)
                    if tx:
                        transactions.append(tx)

                # 2. Parse Cash Transactions (Dividends, Interest, Fees)
                cash_txs = statement.findall(".//CashTransaction")
                for ctx in cash_txs:
                    tx = self._map_cash_transaction_to_internal(ctx)
                    if tx:
                        transactions.append(tx)

        except Exception as e:
            self.logger.error(f"Error parsing IBKR Activity Flex XML: {e}")
            
        return transactions

    def _map_trade_to_internal(self, trade_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """Maps a <Trade> element to our internal transaction format."""
        try:
            # IBKR fields: symbol, dateTime, quantity, tradePrice, ibCommission, currency, buySell
            symbol = trade_elem.get("symbol")
            dt_str = trade_elem.get("dateTime") # formats: YYYYMMDD;HHMMSS
            qty = float(trade_elem.get("quantity", 0))
            price = float(trade_elem.get("tradePrice", 0))
            comm = abs(float(trade_elem.get("ibCommission", 0)))
            currency = trade_elem.get("currency")
            side = trade_elem.get("buySell") # 'BUY' or 'SELL'
            asset_category = trade_elem.get("assetCategory") # STK, OPT, etc.
            trade_id = trade_elem.get("tradeID")
            
            if not symbol or asset_category != "STK": # For now only stocks/ETFs
                return None

            # Standardize type
            tx_type = "BUY" if side == "BUY" else "SELL"
            
            # Parse date (IBKR format 20240130;201500)
            try:
                dt = datetime.strptime(dt_str.split(";")[0], "%Y%m%d")
            except:
                dt = datetime.now()

            return {
                "Date": dt.strftime("%Y-%m-%d"),
                "Type": tx_type,
                "Symbol": symbol,
                "Quantity": abs(qty),
                "Price/Share": price,
                "Commission": comm,
                "Total Amount": abs(qty * price) + (comm if tx_type == "BUY" else -comm),
                "Local Currency": currency,
                "Account": "IBKR",
                "ExternalID": f"IBKR_TRADE_{trade_id}" if trade_id else None,
                "Source": "IBKR_API"
            }
        except Exception as e:
            self.logger.warning(f"Failed to map IBKR trade: {e}")
            return None

    def _map_cash_transaction_to_internal(self, ctx_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """Maps a <CashTransaction> element (Dividends, etc.) to internal format."""
        try:
            # IBKR types: Dividends, Withholding Tax, Payment In Lieu of Dividend, Broker Interest Paid, etc.
            ib_type = ctx_elem.get("type", "")
            amount = float(ctx_elem.get("amount", 0))
            symbol = ctx_elem.get("symbol", "$CASH")
            dt_str = ctx_elem.get("dateTime")
            currency = ctx_elem.get("currency")
            description = ctx_elem.get("description", "")
            transaction_id = ctx_elem.get("transactionID")

            # Filter relevant types
            internal_type = None
            if "Dividend" in ib_type:
                internal_type = "DIVIDEND"
            elif "Interest" in ib_type:
                internal_type = "INTEREST"
            elif "Withholding Tax" in ib_type:
                internal_type = "TAX"
            elif "Fee" in ib_type or "Commission" in ib_type:
                internal_type = "FEE"
            
            if not internal_type:
                return None

            try:
                dt = datetime.strptime(dt_str.split(";")[0], "%Y%m%d")
            except:
                dt = datetime.now()

            return {
                "Date": dt.strftime("%Y-%m-%d"),
                "Type": internal_type,
                "Symbol": symbol if symbol and symbol != "None" else "$CASH",
                "Quantity": 1.0,
                "Price/Share": amount,
                "Commission": 0.0,
                "Total Amount": amount,
                "Local Currency": currency,
                "Account": "IBKR",
                "Description": description,
                "ExternalID": f"IBKR_CASH_{transaction_id}" if transaction_id else None,
                "Source": "IBKR_API"
            }
        except Exception as e:
            self.logger.warning(f"Failed to map IBKR cash transaction: {e}")
            return None

    def sync(self) -> List[Dict[str, Any]]:
        """Execute the full sync flow."""
        ref_code, url = self.request_report()
        # Give it a moment for IBKR to finalize the report
        time.sleep(3)
        
        xml_content = self.download_report(ref_code, url)
        return self.parse_activity_flex_xml(xml_content)
