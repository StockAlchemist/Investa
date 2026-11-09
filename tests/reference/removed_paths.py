"""Names the frozen oracles import from code that has since been removed.

The per-date history path (the single-date valuation in its "python" and
"numba" forms, the single-date holdings kernel and the per-date worker) was
deleted from `src/` on 9 Nov 2026: the history always replays holdings
chronologically. `portfolio_history.py` and `portfolio_logic.py` here still
import those names, so they resolve to these stubs. The parity tests run only
the chronological path, so nothing reaches them; one that does fails loudly
instead of comparing the oracle with code that no longer exists.
"""

HISTORICAL_COMPARE_METHODS = False  # the removed config flag's value


def _removed(name):
    def stub(*args, **kwargs):
        raise NotImplementedError(f"{name} was removed with the per-date history path (9 Nov 2026)")

    stub.__name__ = name
    return stub


_calculate_holdings_numba = _removed("_calculate_holdings_numba")
_calculate_portfolio_value_at_date_unadjusted = _removed("_calculate_portfolio_value_at_date_unadjusted")
_calculate_portfolio_value_at_date_unadjusted_numba = _removed("_calculate_portfolio_value_at_date_unadjusted_numba")
_calculate_portfolio_value_at_date_unadjusted_python = _removed("_calculate_portfolio_value_at_date_unadjusted_python")
_calculate_daily_metrics_worker = _removed("_calculate_daily_metrics_worker")
