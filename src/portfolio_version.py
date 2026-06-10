"""Cache-version anchor for the portfolio engine.

CURRENT_HIST_VERSION embeds a hash of every engine source file, so on-disk
historical caches invalidate automatically whenever calculation code changes.
"""

import hashlib
import os

# Every file whose logic affects calculated/cached results.
_ENGINE_FILES = (
    "portfolio_logic.py",
    "portfolio_history.py",
    "portfolio_cashflows.py",
    "portfolio_valuation_kernels.py",
    "finutils.py",
)


def _get_self_hash():
    """Hash all engine source files so caches invalidate when any of them changes."""
    try:
        h = hashlib.md5()
        base = os.path.dirname(os.path.abspath(__file__))
        for name in _ENGINE_FILES:
            path = os.path.join(base, name)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    h.update(f.read())
        return h.hexdigest()[:8]
    except Exception:
        return "UNKNOWN"


CURRENT_HIST_VERSION = f"v2.0_AUTO_{_get_self_hash()}"
