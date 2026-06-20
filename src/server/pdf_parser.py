import re
import pdfplumber
import logging
import os
import mimetypes
from datetime import datetime
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

# IBKR account identifiers look like "U13340051". Match that exact shape
# rather than a bare "starts with U" — otherwise tickers such as UNH, UPS or
# UBER are mistaken for an account column and the row is misparsed.
_ACCOUNT_ID_RE = re.compile(r"^U\d{6,}$")


def _to_float(value: str) -> float:
    """Parse an IBKR numeric cell, tolerating thousands separators and the
    embedded space pdfplumber inserts when a long decimal wraps across two
    lines (e.g. a trade price rendered as '271.02159793 8')."""
    return float(value.replace(",", "").replace(" ", ""))


# IBKR ships the "Trades" section in two PDF layouts with different column
# orders, so positional indexing can't serve both:
#   Activity Statement:  Symbol | Date/Time | | Quantity | T. Price | C. Price | Proceeds | Comm/Fee | Basis | ...
#   Trade Confirmation:  Acct ID | Symbol | Trade Date/Time | Settle Date | Exchange | Type | Quantity | Price | Proceeds | Comm | Fee | ...
# (Proceeds is column 6 in the first, column 8 in the second.) Map the header
# row to field names so each layout reads the right column.
def _ibkr_trades_colmap(header: List[str]) -> Dict[str, int]:
    cols: Dict[str, int] = {}
    for i, raw in enumerate(header):
        c = (raw or "").strip().lower()
        if not c:
            continue
        if c == "acct id":
            cols.setdefault("acct", i)
        elif c == "symbol":
            cols.setdefault("sym", i)
        elif "date" in c:
            cols.setdefault("date", i)  # 'Date/Time' / 'Trade Date/Time' (first wins, not 'Settle Date')
        elif c == "exchange":
            cols.setdefault("exchange", i)
        elif c == "quantity":
            cols.setdefault("qty", i)
        elif c in ("price", "t. price"):
            cols.setdefault("price", i)  # trade price (ignore 'C. Price')
        elif c == "proceeds":
            cols.setdefault("proceeds", i)
        elif c.startswith("comm"):
            cols.setdefault("comm", i)  # 'Comm' or 'Comm/Fee'
        elif c == "fee":
            cols.setdefault("fee", i)
    return cols


def _ibkr_trade_from_row(
    row: List[str], colmap: Dict[str, int], account_name: str, user_id: int
) -> Optional[Dict[str, Any]]:
    """Build a trade transaction from a Trades data row using the header-derived
    column map. Returns None for label/subtotal rows and for per-venue fills
    (see the Exchange handling below)."""

    def cell(key: str) -> str:
        i = colmap.get(key)
        return row[i].strip() if i is not None and i < len(row) and row[i] is not None else ""

    sym = cell("sym")
    qty_raw = cell("qty")
    if not sym or not qty_raw:
        return None  # 'Stocks' / 'USD' / 'Total ...' / blank rows

    # Trade Confirmations list an order-summary row (Exchange "-") followed by
    # the per-venue fills. Import the summary (it carries the order totals) and
    # skip the fills so the order isn't double counted. Activity Statements have
    # no Exchange column, so every data row is imported.
    exch = cell("exchange")
    if exch and exch != "-":
        return None

    q_val = _to_float(qty_raw)

    # IBKR may split commission and exchange fees into separate columns; combine.
    commission = abs(_to_float(cell("comm"))) if cell("comm") else 0.0
    fee = cell("fee")
    if fee:
        commission += abs(_to_float(fee))

    account = account_name
    acct_i = colmap.get("acct")
    if acct_i is not None and acct_i < len(row) and _ACCOUNT_ID_RE.match((row[acct_i] or "").strip()):
        account = row[acct_i].strip()

    is_buy = q_val > 0
    gross = abs(_to_float(cell("proceeds")))
    # Match web app formula: Buy total includes commission; Sell total is net of commission.
    total_amount = gross + commission if is_buy else max(0.0, gross - commission)
    return {
        "Date": cell("date").split(",")[0].strip(),
        "Type": "Buy" if is_buy else "Sell",
        "Symbol": sym,
        "Quantity": abs(q_val),
        "Price/Share": _to_float(cell("price")),
        "Total Amount": total_amount,
        "Commission": commission,
        "Account": account,
        "Note": "IBKR Trade",
        "Local Currency": "USD",
        "user_id": user_id,
    }


def parse_ibkr_pdf(file_path: str, user_id: int, cash_mode: str, account_override: str) -> List[Dict[str, Any]]:
    transactions = []
    account_name = account_override or "IBKR Account"
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if not tables:
                    continue

                for table in tables:
                    if not table or not table[0]:
                        continue
                    header_text = " ".join([str(x) for x in table[0] if x is not None])

                    section = None
                    if "Trades" in header_text:
                        section = "Trades"
                    elif "Transfers" in header_text:
                        section = "Transfers"
                    elif "Dividends" in header_text:
                        section = "Dividends"
                    elif "Withholding Tax" in header_text:
                        section = "Tax"
                    elif "Deposits & Withdrawals" in header_text:
                        section = "Cash"
                    elif "Other Fees" in header_text:
                        section = "Fees"
                    elif "Interest" in header_text:
                        section = "Interest"

                    if not section:
                        continue

                    # Track section state across rows — a single pdfplumber
                    # table can splice several IBKR logical sections together.
                    current_section = section
                    trades_colmap: Optional[Dict[str, int]] = None  # header-derived map for Trades

                    for row in table:
                        if not row or len(row) < 3:
                            continue
                        row = [str(x).replace("\n", " ").strip() if x is not None else "" for x in row]

                        # Mid-table section switch: a known section heading can
                        # appear in ANY column, not just the first. When
                        # pdfplumber merges the right-hand page tables (e.g.
                        # Dividends + Deposits & Withdrawals) it prepends several
                        # empty cells, so the heading lands in a middle column.
                        switched = False
                        for cell in row:
                            key = cell.lower()
                            if key in _INLINE_SECTION_HEADERS:
                                current_section = _INLINE_SECTION_HEADERS[key]
                                switched = True
                                break
                        if switched:
                            continue
                        if current_section == "SKIP":
                            continue
                        # Rebind for the rest of the loop body below.
                        section = current_section

                        # Capture the Trades column header so data rows can be
                        # mapped by name (IBKR's two layouts order Proceeds/Comm/
                        # Fee differently). Must run before the label-skip below,
                        # which would otherwise drop the 'Symbol ...' header row.
                        if section == "Trades":
                            low_cells = [c.lower() for c in row]
                            if "quantity" in low_cells and "proceeds" in low_cells:
                                trades_colmap = _ibkr_trades_colmap(row)
                                continue

                        first_val = row[0]
                        # Header / label / subtotal rows. The fixed-column
                        # Trades / Transfers branches index by position, so a
                        # stray label row must be filtered here. The generic
                        # branch below instead locates data by content and
                        # filters non-data rows itself (no date / no amount),
                        # so skipping it here would wrongly drop right-aligned
                        # tables whose first cell is empty.
                        if section in ("Trades", "Transfers"):
                            if any(x in first_val for x in [section, "Symbol", "Date", "Total", "Stocks", "USD", "Description"]):
                                continue
                            if not first_val or first_val == "None":
                                continue

                        try:
                            if section == "Trades":
                                if trades_colmap is not None:
                                    txn = _ibkr_trade_from_row(row, trades_colmap, account_name, user_id)
                                    if txn:
                                        transactions.append(txn)
                                    continue

                                # Fallback: no column header captured for this
                                # table — use the legacy multi-account positional
                                # layout.
                                if _ACCOUNT_ID_RE.match(row[0]):
                                    account = row[0]
                                    sym, date_str, qty_str, price_str, proceeds_str, comm_str = row[1], row[2].split(",")[0].strip(), row[6], row[7], row[9], row[10]
                                else:
                                    account = account_name
                                    sym, date_str, _, qty_str, price_str, _, proceeds_str, comm_str = row[0], row[1].split(",")[0].strip(), row[2], row[3], row[4], row[5], row[6], row[7]

                                q_val = _to_float(qty_str)
                                transactions.append({
                                    "Date": date_str, "Type": "Buy" if q_val > 0 else "Sell", "Symbol": sym,
                                    "Quantity": abs(q_val), "Price/Share": _to_float(price_str),
                                    "Total Amount": abs(_to_float(proceeds_str)),
                                    "Commission": abs(_to_float(comm_str)),
                                    "Account": account, "Note": "IBKR Trade", "Local Currency": "USD", "user_id": user_id
                                })

                            elif section == "Transfers":
                                if _ACCOUNT_ID_RE.match(row[0]):
                                    account = row[0]
                                    sym, date_str, t_type, qty_str, mval_str = row[1], row[2], row[3], row[7], row[9]
                                else:
                                    account = account_name
                                    sym, date_str, t_type, qty_str, mval_str = row[0], row[1], row[2], row[6], row[8]
                                q_val = _to_float(qty_str)
                                transactions.append({
                                    "Date": date_str, "Type": "Transfer", "Symbol": sym,
                                    "Quantity": abs(q_val), "Price/Share": _to_float(mval_str) / abs(q_val) if q_val != 0 else 0,
                                    "Total Amount": abs(_to_float(mval_str)),
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
                                if _ACCOUNT_ID_RE.match(row[0]):
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
                                if not date_str:
                                    continue

                                # Amount: rightmost cell that parses as a
                                # signed decimal. Scanning right-to-left skips
                                # the trailing empty "Code" column naturally.
                                amt_str = None
                                for c in reversed(row):
                                    if not c:
                                        continue
                                    s = c.replace(",", "").strip()
                                    if not s:
                                        continue
                                    try:
                                        float(s)
                                        amt_str = c
                                        break
                                    except ValueError:
                                        continue
                                if amt_str is None:
                                    continue

                                # Description: longest concat of non-numeric,
                                # non-date text cells. Single-char codes like
                                # "R" or "Po" at the end are filtered by
                                # length to avoid polluting symbol detection.
                                desc_pieces = []
                                for c in row:
                                    if not c or c == date_str or c == amt_str:
                                        continue
                                    s = c.strip()
                                    if not s:
                                        continue
                                    try:
                                        float(s.replace(",", ""))
                                        continue
                                    except ValueError:
                                        pass
                                    if len(s) > 2:
                                        desc_pieces.append(s)
                                desc = " ".join(desc_pieces).strip()
                                if not desc:
                                    continue

                                if "Total" in desc or "Starting" in desc:
                                    continue

                                amt = _to_float(amt_str)
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
                                if section == "Dividends":
                                    t_type = "Dividend"
                                elif section == "Tax":
                                    t_type = "Tax"
                                elif section == "Interest":
                                    t_type = "Interest"
                                elif section == "Fees":
                                    t_type = "Fees"
                                elif "electronic fund" in l_desc or "deposit" in l_desc:
                                    t_type = "Deposit"
                                elif "withdrawal" in l_desc or "disbursement" in l_desc:
                                    t_type = "Withdrawal"
                                elif "acats transfer" in l_desc or "transfer" in l_desc:
                                    t_type = "Transfer"
                                elif "commission adj" in l_desc:
                                    t_type = "Deposit" if amt > 0 else "Fees"
                                elif section == "Cash":
                                    t_type = "Deposit" if amt >= 0 else "Withdrawal"

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
                        except (ValueError, IndexError) as e:
                            logging.debug(f"IBKR parser skipped {section} row {row!r}: {e}")
                            continue
    except Exception as e:
        logging.error(f"Error parsing IBKR Comprehensive PDF: {e}")
        
    return transactions

# --- Webull monthly statement parsing ----------------------------------------
#
# Webull (Thailand) "Monthly Account Statement" PDFs are *summary* statements:
# most pages are end-of-month snapshots (Net Account Value, Cash Report Summary,
# Portfolio Summary / holdings) rather than a trade blotter. The transaction-
# level data lives in a handful of dated detail tables — INTEREST, DIVIDENDS,
# DEPOSITS & WITHDRAWALS, CORPORATE ACTION, and (in active months) TRADES.
#
# Webull statements don't repeat the section title inside the table pdfplumber
# extracts, and several cash-detail tables share the identical
# "Date | Description | Currency | Amount" header — so we classify each table by
# its header signature and then classify each *row* by its description text
# (mirroring the IBKR generic branch) instead of relying on the section title.

# Currency codes Webull uses as standalone column values. Kept explicit so a
# 3-letter *ticker* (GLD, MMM, ...) is never mistaken for a currency cell.
_WEBULL_CURRENCIES = {
    "USD", "THB", "HKD", "CNH", "CNY", "SGD", "EUR", "GBP",
    "JPY", "AUD", "CAD", "NZD", "CHF",
}

# Webull's TRADES table has no currency column — the trade currency is implied
# by the listing exchange. Maps the exchanges Webull (Thailand) routes to.
_WEBULL_EXCHANGE_CURRENCY = {
    "NASDAQ": "USD", "NYSE": "USD", "NYSEARCA": "USD", "ARCA": "USD",
    "AMEX": "USD", "BATS": "USD", "OTC": "USD",
    "SEHK": "HKD", "HKEX": "HKD", "HKG": "HKD",
    "SET": "THB", "SGX": "SGD",
}

# Webull renders dates as DD/MM/YYYY (e.g. 01/04/2026 = 1 April 2026). Note the
# *description* text can embed US-format source dates ("03/01/2026 to 03/31/2026")
# — those are left untouched; only the dedicated Date column is normalised.
_WEBULL_DATE_RE = re.compile(r"^\s*(\d{2})/(\d{2})/(\d{4})")


def _webull_date(value: str) -> Optional[str]:
    """Normalise a Webull DD/MM/YYYY date cell to ISO YYYY-MM-DD, or None."""
    if not value:
        return None
    m = _WEBULL_DATE_RE.match(value)
    if not m:
        return None
    day, month, year = m.groups()
    return f"{year}-{month}-{day}"


def _webull_float(value: str) -> float:
    """Parse a Webull numeric cell, tolerating thousands separators and the
    parenthesised / trailing-minus conventions some negative amounts use."""
    s = value.replace(",", "").replace(" ", "").strip()
    if not s:
        raise ValueError("empty")
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1]
    elif s.endswith("-"):
        negative, s = True, s[:-1]
    elif s.startswith("-"):
        negative, s = True, s[1:]
    result = float(s)
    return -result if negative else result


def _webull_num(value: str) -> Optional[float]:
    """Like _webull_float but returns None instead of raising on a bad cell."""
    try:
        return _webull_float(value)
    except (ValueError, AttributeError):
        return None


def _webull_col_index(header: List[str], *keywords: str) -> Optional[int]:
    """Index of the first header cell containing all keywords (case-insensitive).

    Webull's detail tables have stable, labelled columns, so mapping by header
    name is far more reliable than guessing positions — a trade row's
    Comm/Fee/Tax and Net Amount columns can't then be transposed."""
    for i, cell in enumerate(header):
        text = (cell or "").lower()
        if all(k in text for k in keywords):
            return i
    return None


def _webull_is_holdings(rows: List[List[str]]) -> bool:
    """True for the Portfolio Summary holdings table (Symbol & Name + valuation
    columns). Used to harvest a name→ticker map, not to import positions."""
    blob = " ".join(str(c).lower() for c in rows[0] if c) if rows else ""
    return "symbol" in blob and ("average price" in blob or "cost basis" in blob)


def _webull_match_symbol(desc: str, name_to_symbol: Dict[str, str]) -> Optional[str]:
    """Resolve a dividend description ("MASTERCARD INCORPORATED - Cash Div ...")
    to a ticker using the holdings name→symbol map, longest name first."""
    low = desc.lower()
    for name in sorted(name_to_symbol, key=len, reverse=True):
        if name and low.startswith(name):
            return name_to_symbol[name]
    return None


def _webull_classify_table(rows: List[List[str]]) -> str:
    """Map a table to 'cashflow', 'trades', or 'skip' from its header cells.

    Header-signature classification is robust to pdfplumber splicing the section
    title onto the table or shifting columns, and cleanly separates the dated
    detail tables we import from the snapshot/accrual tables we never do."""
    blob = " ".join(
        str(c).replace("\n", " ").lower()
        for header_row in rows[:2]
        for c in header_row
        if c
    )
    if not blob:
        return "skip"
    # Accrued dividends are declared-but-unpaid: importing them double-counts
    # once the cash actually lands in a later statement's Dividends table.
    if "ex-date" in blob or "pay date" in blob:
        return "skip"
    # Portfolio Summary (holdings snapshot) — positions, not transactions.
    if any(k in blob for k in ("average price", "cost basis", "unrealized", "market value")):
        return "skip"
    # Net Account Value / Cash Report Summary — period aggregates by currency.
    if "total(thb)" in blob or "total cash" in blob:
        return "skip"
    # Currency Exchange (Initial/Converted Currency, Converted Rate) — an
    # internal THB↔USD conversion, not external cash flow. Importing it would
    # double-count against the deposit and the USD trades it funds.
    if "converted" in blob:
        return "skip"
    # Dividend detail table (Gross Amount / Withholding Tax / Net Amount). Split
    # into a gross Dividend + a Dividend Tax leg so a reinvested dividend nets to
    # zero against its reinvestment buy (gross − WHT − reinvest = 0).
    if "withholding" in blob and ("gross" in blob or "net amount" in blob):
        return "dividends"
    # Trades blotter: dated, per-symbol rows (holdings lack a Date column, so the
    # Date + Symbol pairing excludes the snapshot table).
    if "date" in blob and ("symbol" in blob or "name" in blob) and (
        "price" in blob or "side" in blob or "quantity" in blob
    ):
        return "trades"
    # Generic dated cash-flow detail (Interest / Dividends / Deposits &
    # Withdrawals / Corporate Action / Fees).
    if "date" in blob and ("amount" in blob or "description" in blob):
        return "cashflow"
    return "skip"


def _webull_cashflow_row(
    row: List[str],
    account: str,
    user_id: int,
    default_currency: str,
    name_to_symbol: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Convert one Webull cash-detail row to a transaction dict, classifying the
    type from the description text. Returns None for header/total/blank rows."""
    date_str = next((_webull_date(c) for c in row if _webull_date(c)), None)
    if not date_str:
        return None

    currency = default_currency
    for c in row:
        if c and c.strip().upper() in _WEBULL_CURRENCIES:
            currency = c.strip().upper()
            break

    # Amount: rightmost cell that parses as a number (skips the trailing empty
    # columns and the currency/date cells naturally).
    amount = None
    for c in reversed(row):
        if not c or _webull_date(c) or c.strip().upper() in _WEBULL_CURRENCIES:
            continue
        try:
            amount = _webull_float(c)
            break
        except ValueError:
            continue
    if amount is None:
        return None

    # Description: the remaining non-numeric, non-date, non-currency text.
    desc_pieces = []
    for c in row:
        if not c or _webull_date(c) or c.strip().upper() in _WEBULL_CURRENCIES:
            continue
        try:
            _webull_float(c)
            continue
        except ValueError:
            pass
        desc_pieces.append(c.strip())
    desc = " ".join(p for p in desc_pieces if p).strip()
    low = desc.lower()
    if not desc or any(k in low for k in ("total", "opening", "closing", "starting", "subtotal")):
        return None

    # Classify by description keywords; sign the amount by direction so the
    # engine's cash math is consistent regardless of the statement's own sign.
    if "withholding" in low or " wht" in f" {low}":
        t_type, inflow = "Tax", False
    elif "interest" in low:
        t_type, inflow = "Interest", True
    elif "dividend" in low or "cash div" in low or "div on" in low:
        # Webull phrases dividends as "… - Cash Div on N shares - Rec … Pay …".
        t_type, inflow = "Dividend", True
    elif "deposit" in low:
        t_type, inflow = "Deposit", True
    elif "withdraw" in low or "disbursement" in low:
        t_type, inflow = "Withdrawal", False
    elif any(k in low for k in ("fee", "commission", "vat", "stamp", "tax")):
        t_type, inflow = "Fees", False
    elif "currency exchange" in low or "fx " in low:
        # Internal currency conversions are not external cash flow.
        return None
    else:
        t_type, inflow = ("Deposit", True) if amount >= 0 else ("Withdrawal", False)

    # Tie dividends to the paying ticker when the holdings name→symbol map
    # resolves it; otherwise book the cash leg against $CASH.
    symbol = "$CASH"
    if t_type == "Dividend" and name_to_symbol:
        matched = _webull_match_symbol(desc, name_to_symbol)
        if matched:
            symbol = matched

    magnitude = abs(amount)
    signed = magnitude if inflow else -magnitude
    # $CASH legs carry the amount in Quantity (the engine reads it for cash
    # math); ticker-tied rows leave Quantity 0 and use Total Amount only.
    if symbol == "$CASH":
        tx_qty, tx_price = magnitude, 1.0
    else:
        tx_qty, tx_price = 0.0, 0.0
    return {
        "Date": date_str,
        "Type": t_type,
        "Symbol": symbol,
        "Quantity": tx_qty,
        "Price/Share": tx_price,
        "Total Amount": signed,
        "Commission": 0.0,
        "Account": account,
        "Note": desc[:100],
        "Local Currency": currency,
        "user_id": user_id,
    }


def _webull_trade_row(
    row: List[str], header: List[str], account: str, user_id: int, default_currency: str
) -> Optional[Dict[str, Any]]:
    """Convert one Webull TRADES row using the header column map. The Webull
    layout is:
        Symbol & Name | Trade Date | Time | Settlement Date | Buy/Sell |
        Quantity | Traded Price | Gross Amount | Net Amount | Comm/Fee/Tax |
        VAT | Exchange | Remarks | Status
    There is no currency column — currency is inferred from the exchange."""

    def cell(idx: Optional[int]) -> str:
        if idx is None or idx >= len(row) or not row[idx]:
            return ""
        return row[idx].strip()

    i_date = _webull_col_index(header, "trade", "date")
    if i_date is None:
        i_date = _webull_col_index(header, "date")
    date_str = _webull_date(cell(i_date))
    if not date_str:
        return None

    sym_cell = cell(_webull_col_index(header, "symbol"))
    symbol = sym_cell.split()[0] if sym_cell else ""
    if not re.fullmatch(r"[A-Z0-9.]{1,6}", symbol):
        return None

    qty = _webull_num(cell(_webull_col_index(header, "quantity")))
    price = _webull_num(cell(_webull_col_index(header, "price")))
    if qty is None or price is None:
        return None

    gross = _webull_num(cell(_webull_col_index(header, "gross")))
    commission = abs(_webull_num(cell(_webull_col_index(header, "comm"))) or 0.0) + abs(
        _webull_num(cell(_webull_col_index(header, "vat"))) or 0.0
    )

    side_label = cell(_webull_col_index(header, "buy")).lower()
    if side_label in ("buy", "bought", "b"):
        side = "Buy"
    elif side_label in ("sell", "sold", "s"):
        side = "Sell"
    else:
        side = "Buy" if qty >= 0 else "Sell"

    cur_idx = _webull_col_index(header, "currency")
    currency = cell(cur_idx).upper()
    if currency not in _WEBULL_CURRENCIES:
        exchange = cell(_webull_col_index(header, "exchange")).upper()
        currency = _WEBULL_EXCHANGE_CURRENCY.get(exchange, default_currency)

    gross_abs = abs(gross) if gross else abs(qty * price)
    # Match web app formula: Buy total includes commission; Sell total is net of commission.
    total = gross_abs + commission if side == "Buy" else max(0.0, gross_abs - commission)

    return {
        "Date": date_str,
        "Type": side,
        "Symbol": symbol,
        "Quantity": abs(qty),
        "Price/Share": abs(price),
        "Total Amount": total,
        "Commission": commission,
        "Account": account,
        "Note": "Webull Trade",
        "Local Currency": currency,
        "user_id": user_id,
    }


def _webull_dividend_rows(
    row: List[str],
    header: List[str],
    account: str,
    user_id: int,
    default_currency: str,
    name_to_symbol: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Convert one Webull DIVIDENDS row
        Posting Date | Description | Currency | Gross Amount | Withholding Tax |
        Net Amount | Status
    into the gross Dividend (+ separate Dividend Tax) legs the ledger expects.
    Splitting gross/WHT lets a reinvested dividend net to zero against its
    reinvestment buy. Returns [] for header/blank rows."""

    def cell(idx: Optional[int]) -> str:
        if idx is None or idx >= len(row) or not row[idx]:
            return ""
        return row[idx].strip()

    i_date = _webull_col_index(header, "posting", "date")
    if i_date is None:
        i_date = _webull_col_index(header, "date")
    date_str = _webull_date(cell(i_date))
    if not date_str:
        return []

    gross = _webull_num(cell(_webull_col_index(header, "gross")))
    if gross is None:
        gross = _webull_num(cell(_webull_col_index(header, "net")))
    if gross is None:
        return []
    wht = _webull_num(cell(_webull_col_index(header, "withholding")))

    cur_idx = _webull_col_index(header, "currency")
    currency = cell(cur_idx).upper()
    if currency not in _WEBULL_CURRENCIES:
        currency = default_currency

    desc = cell(_webull_col_index(header, "description"))
    symbol = "$CASH"
    if name_to_symbol:
        matched = _webull_match_symbol(desc, name_to_symbol)
        if matched:
            symbol = matched

    rows: List[Dict[str, Any]] = [
        {
            "Date": date_str,
            "Type": "Dividend",
            "Symbol": symbol,
            "Quantity": 1.0,
            "Price/Share": abs(gross),
            "Total Amount": abs(gross),
            "Commission": 0.0,
            "Account": account,
            "Note": "Gross Dividend",
            "Local Currency": currency,
            "user_id": user_id,
        }
    ]
    if wht is not None and abs(wht) > 1e-9:
        rows.append(
            {
                "Date": date_str,
                "Type": "Tax",
                "Symbol": symbol,
                "Quantity": 0.0,
                "Price/Share": 0.0,
                "Total Amount": abs(wht),
                "Commission": 0.0,
                "Account": account,
                "Note": "Dividend Tax",
                "Local Currency": currency,
                "user_id": user_id,
            }
        )
    return rows


def _webull_days_apart(d1: str, d2: str) -> int:
    """Absolute day gap between two ISO dates, or a large sentinel on error."""
    try:
        a = datetime.strptime(d1[:10], "%Y-%m-%d")
        b = datetime.strptime(d2[:10], "%Y-%m-%d")
        return abs((a - b).days)
    except (ValueError, TypeError):
        return 9999


def _webull_tag_reinvestments(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mark each buy that reinvests a dividend with the "Dividend Reinvestment"
    note. Webull settles the dividend to cash and immediately buys shares for the
    net amount the next day; matching on symbol + date proximity + net amount
    distinguishes a DRIP buy from an ordinary purchase. Both legs are kept so
    cash nets to zero (gross dividend − WHT − reinvestment buy)."""
    dividends = [t for t in transactions if t["Type"] == "Dividend" and t["Symbol"] != "$CASH"]
    taxes = [t for t in transactions if t["Type"] == "Tax"]
    buys = [t for t in transactions if t["Type"] == "Buy"]
    used: set = set()
    for div in dividends:
        symbol, div_date = div["Symbol"], div["Date"]
        gross = abs(div["Total Amount"])
        wht = sum(
            abs(t["Total Amount"])
            for t in taxes
            if t["Symbol"] == symbol and t["Date"] == div_date
        )
        net = gross - wht
        for buy in buys:
            if id(buy) in used or buy["Symbol"] != symbol:
                continue
            if _webull_days_apart(div_date, buy["Date"]) > 7:
                continue
            if abs(abs(buy["Total Amount"]) - net) <= max(0.05, 0.03 * net):
                buy["Note"] = "Dividend Reinvestment"
                used.add(id(buy))
                break
    return transactions


def parse_webull_pdf(file_path: str, user_id: int, cash_mode: str, account_override: str) -> List[Dict[str, Any]]:
    transactions: List[Dict[str, Any]] = []
    account_name = account_override or "Webull"
    default_currency = "THB"
    try:
        with pdfplumber.open(file_path) as pdf:
            # Pull defaults (account number, base currency) from the header text.
            header_text = ""
            for page in pdf.pages[:1]:
                header_text += page.extract_text() or ""
            if not account_override:
                m = re.search(r"Account\s*No\.?\s*:?\s*([A-Z]{2,}\d{3,})", header_text)
                if m:
                    account_name = m.group(1)
            m = re.search(r"Base\s*Currency\s*:?\s*([A-Z]{3})", header_text)
            if m and m.group(1) in _WEBULL_CURRENCIES:
                default_currency = m.group(1)

            # First pass: collect detail tables and harvest a holdings
            # name→ticker map so dividend rows can be tied to their security.
            detail_tables: List[tuple] = []
            name_to_symbol: Dict[str, str] = {}
            for page in pdf.pages:
                for table in page.extract_tables():
                    if not table:
                        continue
                    rows = [
                        [str(c).replace("\n", " ").strip() if c is not None else "" for c in r]
                        for r in table
                        if r
                    ]
                    if not rows:
                        continue

                    if _webull_is_holdings(rows):
                        for r in rows[1:]:
                            parts = r[0].split() if r and r[0] else []
                            if len(parts) >= 2 and re.fullmatch(r"[A-Z0-9.]{1,6}", parts[0]):
                                name_to_symbol[" ".join(parts[1:]).lower()] = parts[0]

                    kind = _webull_classify_table(rows)
                    if kind in ("cashflow", "trades", "dividends"):
                        # rows[0] is the column header; data rows follow.
                        detail_tables.append((kind, rows[0], rows[1:]))

            # Second pass: parse rows now that the name→symbol map is complete.
            for kind, header, data_rows in detail_tables:
                for row in data_rows:
                    if len(row) < 2:
                        continue
                    try:
                        if kind == "cashflow":
                            tx = _webull_cashflow_row(
                                row, account_name, user_id, default_currency, name_to_symbol
                            )
                            if tx:
                                transactions.append(tx)
                        elif kind == "dividends":
                            transactions.extend(
                                _webull_dividend_rows(
                                    row, header, account_name, user_id, default_currency, name_to_symbol
                                )
                            )
                        else:  # trades
                            tx = _webull_trade_row(
                                row, header, account_name, user_id, default_currency
                            )
                            if tx:
                                transactions.append(tx)
                    except (ValueError, IndexError) as e:
                        logging.debug(f"Webull parser skipped {kind} row {row!r}: {e}")
                        continue

            # Link DRIP buys to their dividend so the reinvestment is labelled
            # and the gross/WHT/buy legs net to zero in cash.
            transactions = _webull_tag_reinvestments(transactions)
    except Exception as e:
        logging.error(f"Error parsing Webull statement PDF: {e}")

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
        if ext == '.pdf':
            mime_type = 'application/pdf'
        elif ext in ['.png', '.jpg', '.jpeg']:
            mime_type = f'image/{ext[1:]}'
        else:
            mime_type = 'application/octet-stream'

    if mime_type == 'application/pdf':
        try:
            # Check if it's IBKR
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for i in range(min(3, len(pdf.pages))):
                    text += pdf.pages[i].extract_text() or ""
            
            if "INTERACTIVE BROKERS" in text.upper() or "IBKR" in text.upper():
                res = parse_ibkr_pdf(file_path, user_id, cash_mode, account_override)
                if res:
                    return res
            elif "WEBULL" in text.upper():
                res = parse_webull_pdf(file_path, user_id, cash_mode, account_override)
                if res:
                    return res
            elif "TD AMERITRADE" in text.upper():
                res = parse_tdameritrade_pdf(file_path, user_id, cash_mode, account_override)
                if res:
                    return res
            elif "E*TRADE" in text.upper() or "ETRADE" in text.upper():
                res = parse_etrade_pdf(file_path, user_id, cash_mode, account_override)
                if res:
                    return res
                
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
