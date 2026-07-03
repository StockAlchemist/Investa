"""End-to-end tests for spin-off corporate actions.

Covers the three layers the SPGI→MBGL spin-off touches:
  1. corporate_actions.apply_spin_off  — the reference arithmetic.
  2. The JIT holdings dispatcher        — parent basis cut + child receipt,
                                          cash-neutral.
  3. server.pdf_parser                  — the IBKR Corporate Actions importer.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from corporate_actions import apply_spin_off, SUPPORTED_TYPES, RESERVED_CORPORATE_ACTION_TYPES  # noqa: E402
from portfolio_valuation_kernels import _calculate_holdings_numba  # noqa: E402
from server.pdf_parser import _ibkr_corporate_action_txns, _ibkr_open_positions_basis  # noqa: E402


class _FakePage:
    def __init__(self, tables):
        self._tables = tables

    def extract_tables(self):
        return self._tables


class _FakePDF:
    def __init__(self, pages):
        self.pages = pages


# ---------- pure arithmetic ----------

def test_apply_spin_off_moves_basis_to_child():
    p_qty, p_cost, c_qty, c_cost = apply_spin_off(
        parent_qty=105.0, parent_cost=50466.21, child_qty=105.0, allocated_basis=2451.80
    )
    assert p_qty == 105.0                       # parent share count unchanged
    assert round(p_cost, 2) == 48014.41         # parent basis reduced
    assert c_qty == 105.0
    assert round(c_cost, 2) == 2451.80          # child gets the moved basis
    assert round(p_cost + c_cost, 2) == 50466.21  # total basis preserved


def test_apply_spin_off_clamps_to_available_basis():
    # An allocation larger than the parent's remaining basis can't push it below 0.
    _, p_cost, _, c_cost = apply_spin_off(100.0, 1000.0, 10.0, 5000.0)
    assert p_cost == 0.0
    assert c_cost == 1000.0


def test_spin_off_is_supported_not_reserved():
    assert "spin off" in SUPPORTED_TYPES
    assert "spin off" not in RESERVED_CORPORATE_ACTION_TYPES


# ---------- JIT dispatcher ----------

def _run_holdings(txns, symbol_to_id, type_to_id):
    """Drive _calculate_holdings_numba over a tiny ledger and return
    (holdings_qty, holdings_cost) at the last date. Each txn is a dict."""
    n = len(txns)
    base = np.int64(1_700_000_000_000_000_000)
    tx_dates = np.array([base + i for i in range(n)], dtype=np.int64)
    tx_symbols = np.array([symbol_to_id[t["sym"]] for t in txns], dtype=np.int64)
    tx_accounts = np.zeros(n, dtype=np.int64)
    tx_to_accounts = np.full(n, -1, dtype=np.int64)
    tx_types = np.array([type_to_id[t["type"]] for t in txns], dtype=np.int64)
    tx_qty = np.array([t.get("qty", 0.0) for t in txns], dtype=np.float64)
    tx_prices = np.array([t.get("price", 0.0) for t in txns], dtype=np.float64)
    tx_totals = np.array([t.get("total", 0.0) for t in txns], dtype=np.float64)
    tx_comm = np.zeros(n, dtype=np.float64)
    tx_splits = np.zeros(n, dtype=np.float64)
    tx_curr = np.zeros(n, dtype=np.int64)

    qty, cost, *_ = _calculate_holdings_numba(
        base + n,  # target date after all txns
        tx_dates, tx_symbols, tx_accounts, tx_to_accounts, tx_types,
        tx_qty, tx_prices, tx_totals, tx_comm, tx_splits, tx_curr,
        len(symbol_to_id), 1, 1,
        type_to_id["split"], type_to_id["stock split"], type_to_id["buy"],
        type_to_id["deposit"], type_to_id["sell"], type_to_id["withdrawal"],
        type_to_id["short sell"], type_to_id["buy to cover"],
        type_to_id["transfer"], type_to_id["spin off"],
        type_to_id["fees"], type_to_id["dividend"], type_to_id["interest"],
        type_to_id["tax"], symbol_to_id["$CASH"], 1e-6,
        np.array([], dtype=np.int64), np.zeros(1, dtype=np.int64),
    )
    return qty, cost


_TYPES = {
    "buy": 0, "sell": 1, "dividend": 2, "fees": 3, "transfer": 4,
    "split": 5, "stock split": 6, "deposit": 7, "withdrawal": 8,
    "short sell": 9, "buy to cover": 10, "interest": 11, "tax": 12,
    "spin off": 13,
}


def test_spin_off_reallocates_basis_in_engine():
    symbols = {"SPGI": 0, "MBGL": 1, "$CASH": 2}
    txns = [
        {"sym": "SPGI", "type": "buy", "qty": 105.0, "price": 50466.21 / 105.0, "total": 50466.21},
        {"sym": "MBGL", "type": "spin off", "qty": 105.0, "price": 2451.80 / 105.0, "total": 2451.80},
        {"sym": "SPGI", "type": "spin off", "qty": 0.0, "price": 0.0, "total": 2451.80},
    ]
    qty, cost = _run_holdings(txns, symbols, _TYPES)

    # Parent: share count unchanged, basis reduced by the allocated amount.
    assert round(qty[symbols["SPGI"], 0], 6) == 105.0
    assert round(cost[symbols["SPGI"], 0], 2) == 48014.41
    # Child: created with the allocated basis.
    assert round(qty[symbols["MBGL"], 0], 6) == 105.0
    assert round(cost[symbols["MBGL"], 0], 2) == 2451.80
    # Total basis preserved (cash-neutral reallocation).
    assert round(cost[symbols["SPGI"], 0] + cost[symbols["MBGL"], 0], 2) == 50466.21


def test_spin_off_is_cash_neutral_in_engine():
    symbols = {"SPGI": 0, "MBGL": 1, "$CASH": 2}
    txns = [
        {"sym": "SPGI", "type": "buy", "qty": 105.0, "price": 480.63, "total": 50466.21},
        {"sym": "MBGL", "type": "spin off", "qty": 105.0, "price": 23.3505, "total": 2451.80},
        {"sym": "SPGI", "type": "spin off", "qty": 0.0, "price": 0.0, "total": 2451.80},
    ]
    qty, cost = _run_holdings(txns, symbols, _TYPES)
    # The spin-off legs must not create or destroy $CASH.
    assert round(qty[symbols["$CASH"], 0], 6) == 0.0


# ---------- IBKR importer ----------

_CA_ROW = [
    "2026-07-01", "2026-06-30, 20:25:00",
    "SPGI(US78409V1044) Spinoff 1 for 1 (MBGL, MOBILITY GLOBAL INC, US60744M1062)",
    "105", "0.00", "0.00", "0.00", "",
]


def test_importer_emits_two_spin_off_legs():
    basis = {"MBGL": 2451.80, "SPGI": 48014.41}
    txns = _ibkr_corporate_action_txns(_CA_ROW, "U13340051", user_id=1, basis_map=basis)
    assert len(txns) == 2

    child = next(t for t in txns if t["Symbol"] == "MBGL")
    assert child["Type"] == "Spin-off"
    assert child["Quantity"] == 105.0
    assert round(child["Total Amount"], 2) == 2451.80
    assert round(child["Price/Share"], 4) == 23.3505

    parent = next(t for t in txns if t["Symbol"] == "SPGI")
    assert parent["Quantity"] == 0.0
    assert round(parent["Total Amount"], 2) == 2451.80


def test_importer_child_only_when_basis_unknown():
    # No Open Positions basis harvested — still surface the received shares so
    # the position isn't silently dropped; skip the no-op parent reduction.
    txns = _ibkr_corporate_action_txns(_CA_ROW, "U13340051", user_id=1, basis_map={})
    assert len(txns) == 1
    assert txns[0]["Symbol"] == "MBGL"
    assert txns[0]["Total Amount"] == 0.0


def test_importer_ignores_non_spinoff_rows():
    header = ["Report Date", "Date/Time", "Description", "Quantity", "Proceeds", "Value", "Realized P/L", "Code"]
    assert _ibkr_corporate_action_txns(header, "A", 1, {}) == []
    total = ["", "", "Total", "", "0.00", "0.00", "0.00", ""]
    assert _ibkr_corporate_action_txns(total, "A", 1, {}) == []


# ---------- holdings summary (Cost Basis / Total Buy Cost) ----------

def _spinoff_holdings():
    import pandas as pd
    from datetime import date
    from portfolio_analyzer import _process_transactions_to_holdings

    rows = [
        # Original SPGI position: 105 sh, total basis 50,466.21.
        {"Type": "Buy", "Symbol": "SPGI", "Quantity": 105.0, "Price/Share": 50466.21 / 105.0,
         "Total Amount": 50466.21, "Date": "2025-01-02"},
        # Spin-off child receipt: 105 MBGL at allocated basis 2,451.80.
        {"Type": "Spin-off", "Symbol": "MBGL", "Quantity": 105.0, "Price/Share": 2451.80 / 105.0,
         "Total Amount": 2451.80, "Date": "2026-06-30"},
        # Spin-off parent basis reduction: SPGI, qty 0, 2,451.80.
        {"Type": "Spin-off", "Symbol": "SPGI", "Quantity": 0.0, "Price/Share": 0.0,
         "Total Amount": 2451.80, "Date": "2026-06-30"},
    ]
    for i, r in enumerate(rows):
        r.update({"Commission": 0.0, "Account": "IBKR", "Local Currency": "USD",
                  "To Account": "", "Split Ratio": 0.0, "original_index": i})
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    holdings, *_ = _process_transactions_to_holdings(
        df, default_currency="USD", shortable_symbols=set(),
        historical_fx_lookup={}, display_currency_for_hist_fx="USD",
        report_date=date(2026, 7, 1),
    )
    return holdings


def test_spin_off_reallocates_cost_basis_in_holdings_summary():
    holdings = _spinoff_holdings()
    spgi = holdings[("SPGI", "IBKR")]
    mbgl = holdings[("MBGL", "IBKR")]

    # Parent: share count intact, Cost Basis and Total Buy Cost both cut by the
    # allocated amount (was 50,466.21).
    assert round(spgi["qty"], 6) == 105.0
    assert round(spgi["total_cost_local"], 2) == 48014.41
    assert round(spgi["total_buy_cost_local"], 2) == 48014.41
    # Child: created with the allocated basis.
    assert round(mbgl["qty"], 6) == 105.0
    assert round(mbgl["total_cost_local"], 2) == 2451.80
    # Total basis conserved across the two symbols.
    assert round(spgi["total_cost_local"] + mbgl["total_cost_local"], 2) == 50466.21


def test_spin_off_reallocates_cost_basis_in_fifo_lots():
    import pandas as pd
    from portfolio_analyzer import calculate_fifo_lots_and_gains

    rows = [
        {"Type": "Buy", "Symbol": "SPGI", "Quantity": 105.0, "Price/Share": 50466.21 / 105.0,
         "Total Amount": 50466.21, "Date": "2025-01-02"},
        {"Type": "Spin-off", "Symbol": "MBGL", "Quantity": 105.0, "Price/Share": 2451.80 / 105.0,
         "Total Amount": 2451.80, "Date": "2026-06-30"},
        {"Type": "Spin-off", "Symbol": "SPGI", "Quantity": 0.0, "Price/Share": 0.0,
         "Total Amount": 2451.80, "Date": "2026-06-30"},
    ]
    for i, r in enumerate(rows):
        r.update({"Commission": 0.0, "Account": "IBKR", "Local Currency": "USD",
                  "To Account": "", "Split Ratio": 0.0, "original_index": i})
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])

    _, open_lots = calculate_fifo_lots_and_gains(
        df, display_currency="USD", historical_fx_yf={},
        default_currency="USD", shortable_symbols=set(),
    )

    def basis(lots):
        return sum(lot["qty"] * lot["cost_per_share_local_net"] for lot in lots)

    # Parent lots reduced; child lot opened at the allocated basis.
    assert round(basis(open_lots[("SPGI", "IBKR")]), 2) == 48014.41
    assert round(basis(open_lots[("MBGL", "IBKR")]), 2) == 2451.80


# ---------- pure-Python valuation fallback ----------

def test_spin_off_child_valued_in_python_fallback():
    import pandas as pd
    from datetime import date
    from portfolio_valuation_kernels import _calculate_portfolio_value_at_date_unadjusted_python

    rows = [
        {"Type": "Buy", "Symbol": "SPGI", "Quantity": 105.0, "Price/Share": 480.63,
         "Total Amount": 50466.21, "Date": "2025-01-02"},
        {"Type": "Spin-off", "Symbol": "MBGL", "Quantity": 105.0, "Price/Share": 23.35,
         "Total Amount": 2451.80, "Date": "2026-06-30"},
        {"Type": "Spin-off", "Symbol": "SPGI", "Quantity": 0.0, "Price/Share": 0.0,
         "Total Amount": 2451.80, "Date": "2026-06-30"},
    ]
    for i, r in enumerate(rows):
        r.update({"Commission": 0.0, "Account": "IBKR", "Local Currency": "USD",
                  "To Account": "", "Split Ratio": 0.0, "original_index": i})
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])

    idx = pd.DatetimeIndex([pd.Timestamp("2026-06-30")])
    prices = {
        "SPGI": pd.DataFrame({"price": [414.97]}, index=idx),  # 105 -> 43,571.85
        "MBGL": pd.DataFrame({"price": [21.19]}, index=idx),   # 105 ->  2,224.95
    }

    value, had_nan = _calculate_portfolio_value_at_date_unadjusted_python(
        target_date=date(2026, 7, 1),
        transactions_df=df,
        historical_prices_yf_unadjusted=prices,
        historical_fx_yf={},
        target_currency="USD",
        internal_to_yf_map={"SPGI": "SPGI", "MBGL": "MBGL"},
        account_currency_map={"IBKR": "USD"},
        default_currency="USD",
        manual_overrides_dict=None,
        processed_warnings=set(),
    )

    assert not had_nan
    # Without the spin-off branch MBGL's 105 shares are dropped and the total is
    # only SPGI's 43,571.85; the child must contribute its 2,224.95.
    assert round(value, 2) == 45796.80


# ---------- Open Positions basis harvest (split-table regression) ----------

_OP_HEADER = ["Symbol", "Quantity", "Mult", "Cost Price", "Cost Basis", "Close Price", "Value", "Unrealized P/L", "Code"]


def test_open_positions_basis_spans_split_tables():
    # Mirrors the real IBKR statement: the Open Positions table is split across
    # pages — the first chunk has the column header + AAPL, the continuation
    # repeats only the "Open Positions" title (no header) and holds MBGL. A
    # same-shaped "Net Stock Position Summary" sits between them and must NOT
    # clobber the persisted Cost-Basis column (its col 4 is "Shares Lent" = 0).
    page1 = _FakePage([[
        ["Open Positions"] + [None] * 8,
        _OP_HEADER,
        ["Stocks"] + [None] * 8,
        ["AAPL", "190", "1", "202.30", "38,437.06", "294.38", "55,932.20", "17,495.14", ""],
    ]])
    page2 = _FakePage([
        [  # continuation of Open Positions — title only, NO column header
            ["Open Positions"] + [None] * 8,
            ["ASML", "72", "1", "705.81", "50,819.00", "1,843.04", "132,698.88", "81,879.88", ""],
            ["MBGL", "105", "1", "23.350517724", "2,451.80", "21.19", "2,224.95", "-226.85", ""],
        ],
        [  # neighbouring table with a ticker in col 0 and "0" in col 4
            ["Net Stock Position Summary"] + [None] * 5,
            ["Symbol", "Description", "Shares at IB", "Shares Borrowed", "Shares Lent", "Net Shares"],
            ["MBGL", "MOBILITY GLOBAL INC", "105", "0", "0", "105"],
        ],
    ])
    basis = _ibkr_open_positions_basis(_FakePDF([page1, page2]))
    assert round(basis["AAPL"], 2) == 38437.06
    assert round(basis["ASML"], 2) == 50819.00
    # The continuation row is captured despite the missing header...
    assert round(basis["MBGL"], 2) == 2451.80
    # ...and the Net Stock Position Summary did not overwrite it with 0.
    assert basis["MBGL"] != 0.0


# ---------- IBKR Flex-XML connector ----------

_FLEX_XML = """
<FlexQueryResponse>
  <FlexStatements count="1">
    <FlexStatement accountId="U13340051">
      <OpenPositions>
        <OpenPosition symbol="MBGL" costBasisMoney="2451.80" />
        <OpenPosition symbol="SPGI" costBasisMoney="48014.41" />
      </OpenPositions>
      <CorporateActions>
        <CorporateAction symbol="MBGL" type="SO" quantity="105" dateTime="20260630;202500"
          actionID="99887766" currency="USD"
          description="SPGI(US78409V1044) Spinoff 1 for 1 (MBGL, MOBILITY GLOBAL INC, US60744M1062)" />
        <CorporateAction symbol="SPGI" type="SO" quantity="0" dateTime="20260630;202500"
          actionID="99887766"
          description="SPGI(US78409V1044) Spinoff 1 for 1 (MBGL, MOBILITY GLOBAL INC, US60744M1062)" />
      </CorporateActions>
    </FlexStatement>
  </FlexStatements>
</FlexQueryResponse>
"""


def test_flex_connector_parses_spinoff():
    from ibkr_connector import IBKRConnector

    conn = IBKRConnector(token="x", query_id="y")
    txns = conn.parse_activity_flex_xml(_FLEX_XML)
    spinoffs = [t for t in txns if t["Type"] == "Spin-off"]
    # Only the child-symbol CorporateAction is acted on (the parent twin is
    # skipped), and it expands to the two engine legs.
    assert len(spinoffs) == 2
    child = next(t for t in spinoffs if t["Symbol"] == "MBGL")
    assert child["Quantity"] == 105.0
    assert round(child["Total Amount"], 2) == 2451.80
    assert child["Date"] == "2026-06-30"
    assert child["ExternalID"] == "IBKR_CA_99887766_0"
    parent = next(t for t in spinoffs if t["Symbol"] == "SPGI")
    assert parent["Quantity"] == 0.0
    assert round(parent["Total Amount"], 2) == 2451.80
