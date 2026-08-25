# -*- coding: utf-8 -*-
import re
import requests
import xml.etree.ElementTree as ET
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from config import IBKR_TOKEN, IBKR_QUERY_ID
from datetime import datetime

# --- Constants ---
FLEX_SEND_REQUEST_URL = "https://www.interactivebrokers.com/Universal/servlet/FlexStatementService.SendRequest"
# Note: The actual URL to get the statement comes from the SendRequest response,
# but it usually points to the FlexStatementService.GetStatement servlet.

# Flex Web Service error codes that mean "ask again in a moment" — the query is
# valid, IBKR just has not finished (or is rate-limiting) this statement. Every
# other code is a configuration problem the user has to fix themselves.
TRANSIENT_FLEX_CODES = {
    "1004",  # Statement is incomplete at this time.
    "1005",  # Settlement data is not ready at this time.
    "1006",  # FIFO P/L data is not ready at this time.
    "1007",  # MTM P/L data is not ready at this time.
    "1008",  # MTM and FIFO P/L data is not ready at this time.
    "1009",  # Server is under heavy load; statement could not be generated.
    "1018",  # Too many requests have been made from this token.
    "1019",  # Statement generation in progress.
    "1021",  # Statement could not be retrieved at this time.
}

# Codes that mean the token/query itself is wrong — retrying never helps.
CONFIG_FLEX_CODES = {
    "1010",  # Legacy Flex Queries are no longer supported.
    "1011",  # Service account is inactive.
    "1012",  # Token has expired.
    "1013",  # IP restriction.
    "1014",  # Query is invalid.
    "1015",  # Token is invalid.
    "1016",  # Account in invalid.
    "1020",  # Invalid request or unable to validate request.
}


class IBKRError(Exception):
    """A Flex Web Service call failed. Carries IBKR's own code and message."""

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.code = code

    @property
    def is_config_error(self) -> bool:
        return self.code in CONFIG_FLEX_CODES


class IBKRBusyError(IBKRError):
    """IBKR has not finished generating the statement — retry later, not now."""


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

    @staticmethod
    def _flex_error(root: ET.Element) -> Tuple[Optional[str], str]:
        """Pull (ErrorCode, ErrorMessage) out of a Flex response."""
        code_elem = root.find("ErrorCode")
        msg_elem = root.find("ErrorMessage")
        code = (
            code_elem.text.strip() if code_elem is not None and code_elem.text else None
        )
        message = (
            msg_elem.text.strip()
            if msg_elem is not None and msg_elem.text
            else "Unknown Error"
        )
        return code, message

    @classmethod
    def _envelope_error(cls, xml_content: str) -> Optional[Tuple[Optional[str], str]]:
        """Return (code, message) when the payload is a Flex error envelope
        rather than a statement, else None."""
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return None
        code, message = cls._flex_error(root)
        if code is None and root.findtext("Status") not in ("Fail", "Warn"):
            return None
        return code, message

    @staticmethod
    def _is_transient(code: Optional[str], message: str) -> bool:
        """Transient by code, or — when IBKR omits the code — by its wording."""
        if code:
            return code in TRANSIENT_FLEX_CODES
        text = message.lower()
        return "try again shortly" in text or "in progress" in text

    def request_report(self) -> Tuple[str, str]:
        """
        Initiates a Flex report request.
        Returns (reference_code, download_url).

        Raises `IBKRBusyError` when IBKR is still generating the statement and
        `IBKRError` for anything else, so callers never have to guess what a
        `(None, None)` return meant.
        """
        if not self.token or not self.query_id:
            raise IBKRError(
                "IBKR Token or Query ID not configured. Set them in Settings.",
                code="1015",
            )

        params = {"t": self.token, "q": self.query_id, "v": "3"}

        # 5s then 15s: long enough for IBKR to finish a queued statement,
        # short enough that the request does not outlive the client's patience.
        backoff = [5, 15]
        last_error: Optional[IBKRError] = None
        for attempt in range(len(backoff) + 1):
            xml_resp = self._make_request(FLEX_SEND_REQUEST_URL, params)
            if not xml_resp:
                raise IBKRError(
                    "Could not reach the IBKR Flex Web Service. Check your network and try again."
                )

            try:
                root = ET.fromstring(xml_resp)
            except ET.ParseError as e:
                self.logger.error(f"Failed to parse IBKR SendRequest response: {e}")
                raise IBKRError(f"Failed to initiate IBKR sync: {e}") from e

            status_elem = root.find("Status")
            status = status_elem.text if status_elem is not None else "Fail"

            if status == "Success":
                ref_elem = root.find("ReferenceCode")
                url_elem = root.find("Url")
                if ref_elem is None or url_elem is None:
                    raise IBKRError(
                        "IBKR accepted the request but returned no reference code."
                    )
                reference_code, url = ref_elem.text, url_elem.text
                self.logger.info(
                    f"IBKR Report request successful. Ref: {reference_code}"
                )
                return reference_code, url

            code, err_msg = self._flex_error(root)
            if not self._is_transient(code, err_msg):
                self.logger.error(
                    f"IBKR Report request failed ({code or 'no code'}): {err_msg}"
                )
                raise IBKRError(err_msg, code=code)

            last_error = IBKRBusyError(err_msg, code=code)
            if attempt < len(backoff):
                wait_time = backoff[attempt]
                self.logger.warning(
                    f"IBKR Report generation busy ({code or 'no code'}): {err_msg}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        self.logger.warning(
            f"IBKR still busy after {len(backoff) + 1} attempts: {last_error}"
        )
        raise last_error

    def download_report(self, reference_code: str, url: str) -> str:
        """Downloads the actual report XML using the reference code."""
        params = {"t": self.token, "q": reference_code, "v": "3"}

        # Sometimes IBKR needs a few seconds to prepare the report
        # Increased retries and wait time as first-time reports can be slow
        max_retries = 6
        last_error: Optional[IBKRError] = None
        for i in range(max_retries):
            xml_content = self._make_request(url, params)
            if xml_content:
                # An error envelope is also a <FlexStatementResponse>, so look
                # for the failure first — matching on the root tag alone would
                # hand IBKR's "please try again shortly" to the XML parser as
                # if it were a statement.
                error = self._envelope_error(xml_content)
                if error is not None:
                    code, err_msg = error
                    self.logger.warning(
                        f"Unexpected IBKR response (Attempt {i + 1}): {xml_content[:200]}..."
                    )
                    if not self._is_transient(code, err_msg):
                        self.logger.error(
                            f"IBKR report download failed ({code or 'no code'}): {err_msg}"
                        )
                        raise IBKRError(err_msg, code=code)
                    last_error = IBKRBusyError(err_msg, code=code)
                    self.logger.warning(
                        f"IBKR Report still preparing ({code or 'no code'}), waiting 10s..."
                    )
                    time.sleep(10)
                    continue

                # Otherwise it is the report itself.
                if (
                    "<FlexQueryResponse" in xml_content
                    or "<FlexStatementResponse" in xml_content
                ):
                    return xml_content

                # Log the unexpected content for debugging
                self.logger.warning(
                    f"Unexpected IBKR response (Attempt {i + 1}): {xml_content[:200]}..."
                )

            self.logger.error(f"Failed to download IBKR report (Attempt {i + 1})")
            time.sleep(5)

        raise IBKRBusyError(
            "IBKR is still generating your statement (gave up after 6 attempts). "
            "Please wait a minute or two and sync again.",
            code=last_error.code if last_error else None,
        )

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

                # 3. Parse Corporate Actions (spin-offs). The child's allocated
                #    cost basis lives in OpenPositions (costBasisMoney), exactly
                #    like the PDF's Open Positions table — harvest it first.
                basis_map: Dict[str, float] = {}
                for op in statement.findall(".//OpenPosition"):
                    sym = op.get("symbol")
                    cb = op.get("costBasisMoney")
                    if sym and cb:
                        try:
                            basis_map[sym] = abs(float(cb))
                        except (ValueError, TypeError):
                            continue
                for ca in statement.findall(".//CorporateAction"):
                    transactions.extend(
                        self._map_corporate_action_to_internal(ca, basis_map)
                    )

        except Exception as e:
            self.logger.error(f"Error parsing IBKR Activity Flex XML: {e}")

        return transactions

    def _map_trade_to_internal(
        self, trade_elem: ET.Element
    ) -> Optional[Dict[str, Any]]:
        """Maps a <Trade> element to our internal transaction format."""
        try:
            # IBKR fields: symbol, dateTime, quantity, tradePrice, ibCommission, currency, buySell
            symbol = trade_elem.get("symbol")
            dt_str = trade_elem.get("dateTime")  # formats: YYYYMMDD;HHMMSS
            qty = float(trade_elem.get("quantity", 0))
            price = float(trade_elem.get("tradePrice", 0))
            comm = abs(float(trade_elem.get("ibCommission", 0)))
            currency = trade_elem.get("currency")
            side = trade_elem.get("buySell")  # 'BUY' or 'SELL'
            asset_category = trade_elem.get("assetCategory")  # STK, OPT, etc.
            trade_id = trade_elem.get("tradeID")

            if not symbol or asset_category != "STK":  # For now only stocks/ETFs
                return None

            # Standardize type
            tx_type = "BUY" if side == "BUY" else "SELL"

            # Parse date (IBKR format 20240130;201500)
            try:
                dt = datetime.strptime(dt_str.split(";")[0], "%Y%m%d")
            except Exception:
                dt = datetime.now()

            return {
                "Date": dt.strftime("%Y-%m-%d"),
                "Type": tx_type,
                "Symbol": symbol,
                "Quantity": abs(qty),
                "Price/Share": price,
                "Commission": comm,
                "Total Amount": abs(qty * price)
                + (comm if tx_type == "BUY" else -comm),
                "Local Currency": currency,
                "Account": "IBKR",
                "ExternalID": f"IBKR_TRADE_{trade_id}" if trade_id else None,
                "Source": "IBKR_API",
            }
        except Exception as e:
            self.logger.warning(f"Failed to map IBKR trade: {e}")
            return None

    def _map_cash_transaction_to_internal(
        self, ctx_elem: ET.Element
    ) -> Optional[Dict[str, Any]]:
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
            except Exception:
                dt = datetime.now()

            tx_qty = 1.0
            tx_price = amount

            if internal_type == "DIVIDEND":
                match = re.search(
                    r"(\d+(?:\.\d+)?)\s*per Share", description, re.IGNORECASE
                )
                if match:
                    try:
                        div_per_share = float(match.group(1))
                        if div_per_share > 0:
                            tx_price = div_per_share
                            tx_qty = round(abs(amount) / div_per_share, 4)
                    except ValueError:
                        pass

            return {
                "Date": dt.strftime("%Y-%m-%d"),
                "Type": internal_type,
                "Symbol": symbol if symbol and symbol != "None" else "$CASH",
                "Quantity": tx_qty,
                "Price/Share": tx_price,
                "Commission": 0.0,
                "Total Amount": amount,
                "Local Currency": currency,
                "Account": "IBKR",
                "Description": description,
                "ExternalID": f"IBKR_CASH_{transaction_id}" if transaction_id else None,
                "Source": "IBKR_API",
            }
        except Exception as e:
            self.logger.warning(f"Failed to map IBKR cash transaction: {e}")
            return None

    def _map_corporate_action_to_internal(
        self, ca_elem: ET.Element, basis_map: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Maps a <CorporateAction> element to the spin-off legs the engine
        applies. Only spin-offs are handled today; other actions return [].
        Shares description parsing / row construction with the PDF importer
        (corporate_actions.parse_spinoff_description / build_spinoff_legs)."""
        try:
            from corporate_actions import build_spinoff_legs, parse_spinoff_description

            description = ca_elem.get("description", "")
            parsed = parse_spinoff_description(description)
            if not parsed:
                return []
            parent, child, ratio = parsed

            # IBKR emits one CorporateAction per affected symbol. Act only on the
            # element that delivers the child shares so a parent-side twin can't
            # double-count the event.
            symbol = ca_elem.get("symbol")
            if symbol and symbol != child:
                return []

            qty = abs(float(ca_elem.get("quantity", 0) or 0))
            if qty <= 1e-9:
                return []

            dt_str = ca_elem.get("dateTime") or ca_elem.get("reportDate") or ""
            try:
                dt = datetime.strptime(dt_str.split(";")[0][:8], "%Y%m%d")
                date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                date_str = datetime.now().strftime("%Y-%m-%d")

            currency = ca_elem.get("currency") or "USD"
            legs = build_spinoff_legs(
                parent,
                child,
                qty,
                date_str,
                "IBKR",
                user_id=0,
                allocated_basis=basis_map.get(child, 0.0),
                ratio=ratio,
                currency=currency,
            )
            # Match the connector's row conventions (no user_id; carry Source /
            # ExternalID so re-syncs dedupe).
            action_id = ca_elem.get("actionID") or ca_elem.get("transactionID")
            for i, leg in enumerate(legs):
                leg.pop("user_id", None)
                leg["Account"] = "IBKR"
                leg["Source"] = "IBKR_API"
                leg["ExternalID"] = f"IBKR_CA_{action_id}_{i}" if action_id else None
            return legs
        except Exception as e:
            self.logger.warning(f"Failed to map IBKR corporate action: {e}")
            return []

    def sync(self) -> List[Dict[str, Any]]:
        """Execute the full sync flow."""
        ref_code, url = self.request_report()
        # Give it a moment for IBKR to finalize the report
        time.sleep(3)

        xml_content = self.download_report(ref_code, url)
        return self.parse_activity_flex_xml(xml_content)
