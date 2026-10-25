"""Phase 12: data_loader, csv_utils and io_handlers in Rust
(rust/investa_core/src/dl.rs) against the originals frozen in
tests/reference/.

Random CSV ledgers — header aliases and verbose headers, unknown and missing
columns, dates in every accepted shape (ordinals, two-digit days, invalid),
numeric cells holding text, cash spelled four ways, spin-off spellings and
reserved corporate-action types, blank, "<NA>" and four-letter currencies,
"nan" tags — go through `load_and_clean_transactions` on both sides; so do
database sources (populated, empty, unreachable). All seven results, the log
records and the warnings must agree. The header standardiser and the export
writers are compared on the files they write.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import csv_utils as new_csv  # noqa: E402
import data_loader as new  # noqa: E402
import io_handlers as new_io  # noqa: E402
from reference import csv_utils as ref_csv  # noqa: E402
from reference import data_loader as ref  # noqa: E402
from reference import io_handlers as ref_io  # noqa: E402
from test_rust_market_db_parity import call  # noqa: E402

HEADERS = {
    "Date": ["Date", "Date (MMM DD, YYYY)", "Transaction Date", " Date "],
    "Type": ["Type", "Transaction Type"],
    "Symbol": ["Symbol", "Ticker", "Stock / ETF Symbol"],
    "Quantity": ["Quantity", "Shares", "Quantity of Units"],
    "Price": ["Price/Share", "Price", "Amount per unit"],
    "Total": ["Total Amount", "Cost Basis"],
    "Commission": ["Commission", "Fees"],
    "Account": ["Account", "Broker", "Investment Account"],
    "Note": ["Note"],
    "Currency": ["Local Currency", "Currency"],
    "Tags": ["Tags", "Labels"],
    "To": ["To Account", "Transfer To"],
    "Junk": ["Description", "Unused"],
}
DATES = [
    "2024-01-05",
    "01/06/2024",
    "13/01/2024",
    "Jan 07, 2024",
    "7th Jan, 2024",
    "08-Feb-2024",
    "January 09, 2024",
    "20240110",
    "2024-02-30",
    "soon",
    "",
    " ",
]
TYPES = [
    "Buy",
    "sell",
    " Dividend ",
    "deposit",
    "Withdrawal",
    "Spin-off",
    "spinoff",
    "merger",
    "Transfer",
    "fees",
]


def ledger(rng, path):
    cols = {}
    for key, options in HEADERS.items():
        if key in ("Date",) or rng.random() < 0.85:
            cols[str(rng.choice(options))] = key
    n = int(rng.integers(0, 25))
    rows = []
    for _ in range(n):
        row = []
        for header, key in cols.items():
            if key == "Date":
                row.append(str(rng.choice(DATES)))
            elif key == "Type":
                row.append(str(rng.choice(TYPES)))
            elif key == "Symbol":
                row.append(str(rng.choice(["AAPL", "cash", "USD", "", "nan", "MSFT "])))
            elif key in ("Quantity", "Price", "Total", "Commission"):
                row.append(
                    str(rng.choice(["10", "1,000", "abc", "", "12.5", "-3", "None"]))
                )
            elif key == "Currency":
                row.append(str(rng.choice(["usd", "", "<NA>", "THBX", "EUR", "None"])))
            elif key == "Tags":
                row.append(str(rng.choice(["", "nan", "core", "NONE"])))
            elif key == "Account":
                row.append(str(rng.choice(["IBKR", "E*Trade", " SCB "])))
            else:
                row.append(str(rng.choice(["", "x", "note"])))
        rows.append(row)
    df = pd.DataFrame(rows, columns=list(cols))
    df.to_csv(path, index=False)
    return path


def same(a, b, tmp_path):
    return repr(a).replace(str(tmp_path / "new"), "<d>").replace(
        str(tmp_path / "ref"), "<d>"
    ) == repr(b).replace(str(tmp_path / "new"), "<d>").replace(
        str(tmp_path / "ref"), "<d>"
    )


@pytest.mark.parametrize("seed", range(8))
def test_csv_loads_match(tmp_path, caplog, seed):
    rng = np.random.default_rng(8800 + seed)
    for trial in range(15):
        path = ledger(rng, tmp_path / f"t{trial}.csv")
        cmap = {"IBKR": "USD", "E*Trade": "usd"} if rng.random() < 0.7 else {}
        a = call(
            lambda: ref.load_and_clean_transactions(str(path), cmap, "THB", False),
            caplog,
        )
        b = call(
            lambda: new.load_and_clean_transactions(str(path), cmap, "THB", False),
            caplog,
        )
        assert a == b, (seed, trial, a, b)
    for odd in ("missing.csv", "empty.csv", "headers_only.csv"):
        p = tmp_path / odd
        if odd == "empty.csv":
            p.write_text("")
        elif odd == "headers_only.csv":
            p.write_text("Date,Type\n")
        assert call(
            lambda: ref.load_and_clean_transactions(str(p), {}, "USD"), caplog
        ) == call(lambda: new.load_and_clean_transactions(str(p), {}, "USD"), caplog), (
            odd
        )


def test_db_sources_match(tmp_path, caplog):
    import db_utils

    for name, rows in (("full", 6), ("empty", 0)):
        outs = []
        for side, mod in (("ref", ref), ("new", new)):
            (tmp_path / side).mkdir(exist_ok=True)
            db = tmp_path / side / f"{name}.db"
            conn = db_utils.initialize_database(str(db))
            for i in range(rows):
                db_utils.add_transaction_to_db(
                    conn,
                    {
                        "Date": f"2024-01-0{i + 1}",
                        "Type": ["buy", "spin-off", "deposit"][i % 3],
                        "Symbol": ["AAPL", "cash", ""][i % 3],
                        "Quantity": 2.0,
                        "Price/Share": 10.0,
                        "Total Amount": 20.0,
                        "Account": "IBKR",
                        "Local Currency": ["", "usd", "THB"][i % 3],
                    },
                )
            outs.append(
                call(
                    lambda: mod.load_and_clean_transactions(
                        str(db), {"IBKR": "USD"}, "THB", True
                    ),
                    caplog,
                )
            )
        assert same(outs[0], outs[1], tmp_path), name
    bad = tmp_path / "file_not_dir"
    bad.write_text("x")
    path = str(bad / "x.db")
    assert same(
        call(lambda: ref.load_and_clean_transactions(path, {}, "USD", True), caplog),
        call(lambda: new.load_and_clean_transactions(path, {}, "USD", True), caplog),
        tmp_path,
    )


def test_header_standardiser_and_writers_match(tmp_path, caplog):
    rng = np.random.default_rng(8900)
    for trial in range(20):
        src = ledger(rng, tmp_path / f"in{trial}.csv")
        outs = []
        for side, mod in (("ref", ref_csv), ("new", new_csv)):
            out = tmp_path / f"out_{side}_{trial}.csv"
            got = call(
                lambda: mod.convert_csv_headers_to_cleaned_format(str(src), str(out)),
                caplog,
            )
            outs.append(
                (
                    repr(got).replace(f"out_{side}_", "out_X_"),
                    out.read_bytes() if out.exists() else None,
                )
            )
        assert outs[0] == outs[1], trial
    for odd in ("missing.csv", "empty_in.csv"):
        p = tmp_path / odd
        if odd == "empty_in.csv":
            p.write_text("")
        outs = []
        for side, mod in (("ref", ref_csv), ("new", new_csv)):
            out = tmp_path / f"{side}_{odd}"
            got = call(
                lambda: mod.convert_csv_headers_to_cleaned_format(str(p), str(out)),
                caplog,
            )
            outs.append(
                (
                    repr(got).replace(f"{side}_{odd}", "X"),
                    out.read_bytes() if out.exists() else None,
                )
            )
        assert outs[0] == outs[1], odd
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y,z"], "c": [1.5, np.nan]})
    for side, mod in (("ref", ref_io), ("new", new_io)):
        mod.write_dataframe_to_csv(df, str(tmp_path / f"{side}.csv"))
        mod.write_dataframe_to_excel(df, str(tmp_path / f"{side}.xlsx"))
    assert (tmp_path / "ref.csv").read_bytes() == (tmp_path / "new.csv").read_bytes()
    assert pd.read_excel(tmp_path / "ref.xlsx").equals(
        pd.read_excel(tmp_path / "new.xlsx")
    )


def test_csv_migration_matches(tmp_path, caplog):
    import db_utils
    from reference import db_utils as ref_db

    rng = np.random.default_rng(9000)
    for trial in range(10):
        src = ledger(rng, tmp_path / f"m{trial}.csv")
        if trial == 9:
            src = tmp_path / "absent.csv"
        outs = []
        for side, mod in (("ref", ref_db), ("new", db_utils)):
            (tmp_path / side).mkdir(exist_ok=True)
            conn = db_utils.initialize_database(str(tmp_path / side / f"m{trial}.db"))
            got = call(
                lambda: mod.migrate_csv_to_db(str(src), conn, {"IBKR": "USD"}, "THB"),
                caplog,
            )
            rows = conn.execute("SELECT * FROM transactions ORDER BY id").fetchall()
            outs.append((got, rows))
        assert same(outs[0], outs[1], tmp_path), trial
