# ruff: noqa: E402  # sys.path is mutated below; project imports must follow that block
"""Top-level API router: aggregates the domain routers under one APIRouter.

Route implementations live in server/routes/*; the calculation engine and its
caches live in server/portfolio_service.py. A few engine names are re-exported
here for backward compatibility (tests and older callers import them from
server.api).
"""

import os
import sys

from fastapi import APIRouter

# Ensure src is in path for imports
current_file_path = os.path.abspath(__file__)
src_path = os.path.dirname(os.path.dirname(current_file_path))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from server.route_utils import get_mdp, clean_nans, _lru_get, _lru_put  # noqa: F401 (re-exported)
from server.portfolio_service import (  # noqa: F401 (re-exported)
    _PORTFOLIO_SUMMARY_CACHE,
    _calculate_historical_performance_internal,
    _calculate_portfolio_summary_internal,
    _compute_raw_summary,
    _get_historical_performance_cached,
    clear_portfolio_caches,
    compute_account_closure_state,
    reload_data_and_clear_cache,
    trigger_background_precalculation,
)

from server.routes.admin import router as _admin_router
from server.routes.analytics import router as _analytics_router
from server.routes.auth import router as _auth_router
from server.routes.market import router as _market_router
from server.routes.portfolio import router as _portfolio_router
from server.routes.screener import router as _screener_router
from server.routes.settings import router as _settings_router
from server.routes.transactions import router as _transactions_router
from server.routes.watchlist import router as _watchlist_router

router = APIRouter()
router.include_router(_auth_router)
router.include_router(_market_router)
router.include_router(_transactions_router)
router.include_router(_portfolio_router)
router.include_router(_analytics_router)
router.include_router(_settings_router)
router.include_router(_admin_router)
router.include_router(_screener_router)
router.include_router(_watchlist_router)
