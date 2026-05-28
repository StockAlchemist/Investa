"""Backfill per-dimension AI analysis text into screener_cache.

The screener_cache table historically stored only the four scorecard
*scores* (REAL) and the top-level `ai_summary` (TEXT). The four narrative
paragraphs (`analysis.moat`, `analysis.financial_strength`,
`analysis.predictability`, `analysis.growth_perspective`) were dropped on
write, so when the disk JSON cache was missing the modal rendered "N/A"
for every section.

After the schema added the four `ai_*_analysis` TEXT columns, this script
walks `data/cache/ai_analysis_cache/*_analysis.json` and copies the
narrative text into the DB so existing reviews don't need to be regenerated.

Run from the repo root:

    python scripts/backfill_ai_review_analysis_text.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402
from db_utils import open_screener_db_conn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill-ai-analysis")


def _coerce_text(v) -> Optional[str]:
    if isinstance(v, str) and v.strip():
        return v
    return None


def iter_cache_files(cache_dir: str):
    if not os.path.isdir(cache_dir):
        log.warning("AI analysis cache dir not found: %s", cache_dir)
        return
    for name in os.listdir(cache_dir):
        if not name.endswith("_analysis.json"):
            continue
        symbol = name[: -len("_analysis.json")].upper()
        yield symbol, os.path.join(cache_dir, name)


def load_analysis(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
        return None
    analysis_obj = payload.get("analysis") if isinstance(payload, dict) else None
    if not isinstance(analysis_obj, dict):
        return None
    inner = analysis_obj.get("analysis")
    return inner if isinstance(inner, dict) else None


def backfill(dry_run: bool) -> None:
    cache_dir = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "ai_analysis_cache")
    conn = open_screener_db_conn()
    if conn is None:
        log.error("Could not open global screener DB")
        sys.exit(1)

    sql = """
    UPDATE screener_cache SET
        ai_moat_analysis = COALESCE(ai_moat_analysis, ?),
        ai_financial_strength_analysis = COALESCE(ai_financial_strength_analysis, ?),
        ai_predictability_analysis = COALESCE(ai_predictability_analysis, ?),
        ai_growth_perspective_analysis = COALESCE(ai_growth_perspective_analysis, ?)
    WHERE symbol = ?
      AND (
        ai_moat_analysis IS NULL OR
        ai_financial_strength_analysis IS NULL OR
        ai_predictability_analysis IS NULL OR
        ai_growth_perspective_analysis IS NULL
      )
    """

    total_files = 0
    eligible = 0
    updated_rows = 0
    skipped_no_text = 0

    try:
        cursor = conn.cursor()
        for symbol, path in iter_cache_files(cache_dir):
            total_files += 1
            analysis = load_analysis(path)
            if not analysis:
                continue
            moat = _coerce_text(analysis.get("moat"))
            fin = _coerce_text(analysis.get("financial_strength"))
            pred = _coerce_text(analysis.get("predictability"))
            growth = _coerce_text(analysis.get("growth_perspective"))
            if not any([moat, fin, pred, growth]):
                skipped_no_text += 1
                continue
            eligible += 1
            if dry_run:
                log.info("[dry-run] would update %s", symbol)
                continue
            cursor.execute(sql, (moat, fin, pred, growth, symbol))
            updated_rows += cursor.rowcount

        if not dry_run:
            conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    log.info(
        "Done. cache files scanned=%d, with text=%d, no_text=%d, rows updated=%d%s",
        total_files,
        eligible,
        skipped_no_text,
        updated_rows,
        " (dry-run)" if dry_run else "",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    args = parser.parse_args()
    backfill(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
