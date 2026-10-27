"""Phase 16d: the symbol mapper, safe_sum and the formatters of finutils in
Rust (rust/investa_core/src/fu_fmt.rs) against the original frozen in
tests/reference/finutils_p16.py.

Symbols cover every rule: cash, exclusions, explicit and system maps (keys
and values in any case, with spaces), Thai `:BKK` names (valid, dotted,
empty), dotted class tickers and exchange suffixes, characters the pattern
rejects (lowercase after a non-ASCII upper-casing, spaces, slashes), long
names, non-ASCII text and non-strings. Sums take numeric, text and missing
columns. Formatters take None, NaN, infinities, NumPy scalars, text, huge
integers and decimals that cannot build a spec. Results and logs agree.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import finutils as new  # noqa: E402
from reference import finutils_p16 as ref  # noqa: E402
from test_rust_market_db_parity import call  # noqa: E402


def _both(caplog, name, *args):
    outs = [call(lambda mod=mod: getattr(mod, name)(*args), caplog) for mod in (ref, new)]
    assert outs[0] == outs[1], (name, args)


SYMBOLS = ["AAPL", " aapl ", "$CASH", "Cash (THB)", "BRK.B", "brk.a", "BF.B", "ADVANC:BKK", "PTT.R:BKK", ":BKK",
           "KBANK:bkk", "XYZ.A", "ABC.BK", "AB.L", "SHOP.TO", "TOOLONG.AB", "AB.CDE", "A.B.C", "^GSPC", "EUR=X",
           "MY FUND", "K-USXNDQ-A(A)", "ŞIRKET", "ß", "ABCDEFGHIJKLMNOP", "ABCDEFGHIJKLMNO", "", None, 5,
           "SCBRMS&P500", "IBKR/1", "X\n", "1234.T", "VTI", "ﬀ"]  # fmt: skip


@pytest.mark.parametrize("user_map", [{}, {"vti ": " vti.l ", "XYZ.A": "xyz-a", "adVANC:BKK": "ADVANC.BK"}])
@pytest.mark.parametrize("excluded", [set(), {"AAPL", "VTI", "MY FUND"}])
def test_map_to_yf_symbol(caplog, user_map, excluded):
    for s in SYMBOLS:
        _both(caplog, "map_to_yf_symbol", s, user_map, excluded)


def test_safe_sum(caplog):
    df = pd.DataFrame({"a": [1.0, np.nan, 2.5], "b": ["1", "x", None], "c": [True, False, True],
                       "d": [np.inf, -np.inf, 1.0], "e": [[1], [2], [3]]})  # fmt: skip
    for col in ("a", "b", "c", "d", "e", "missing"):
        _both(caplog, "safe_sum", df, col)
    _both(caplog, "safe_sum", df.iloc[:0], "a")


VALUES = [None, float("nan"), 0.0, 1e-12, -1e-12, 1234.5678, -98765.4321, 1e15, 2.5e9, 3.4e6, 999.0, float("inf"),
          float("-inf"), np.float64(42.125), np.int64(7), "12.5", "x", 10**30, True, pd.NA, pd.NaT]  # fmt: skip


@pytest.mark.parametrize("decimals", [2, 0, 4, 2.5, "2", -1])
def test_formatters(caplog, decimals):
    for v in VALUES:
        for plus in (False, True):
            for is_abs in (False, True):
                _both(caplog, "format_currency_value", v, "฿", decimals, plus, is_abs)
            _both(caplog, "format_percentage_value", v, decimals, plus)
        _both(caplog, "format_large_number_display", v, "$", decimals)
        _both(caplog, "format_float_with_commas", v, decimals)
    for v in VALUES:
        _both(caplog, "format_integer_with_commas", v)
