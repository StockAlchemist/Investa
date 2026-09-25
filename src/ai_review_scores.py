# -*- coding: utf-8 -*-
"""
The AI review as a ranking input.

Every reviewed company carries four 1-10 judgements from the stock review
(`server/ai_analyzer.py`): moat, financial strength, predictability and growth.
This module turns them into one score on the same 0-100 percentile scale the
quality and value components already use, so `buffett_rank.blend_scores` can
mix all three with plain weights.

**Read at request time, not frozen into a run.** The reviews live in the shared
screener cache and are refreshed by their own workers on their own schedule; a
ranking run is a nightly EDGAR batch. Joining them when the ranking is served
means a new review counts the moment it lands, and the stored snapshot stays
exactly what the filings produced — its `rank` is the pre-AI rank, and
`base_rank` in the served rows says so.

**A missing review is absent, not bad (P7).** A company nobody has reviewed yet
keeps its quality/value score untouched rather than being blended with zero.
The same rule applies per dimension: the review parser writes `0.0` when the
model omitted a score, and 0 is off the 1-10 scale, so it is read as missing
rather than as the worst possible business.

**A percentile, not the raw mean.** The raw scores bunch: the average is ~5.8
and nearly everything sits between 4 and 8. Blending that directly against
percentiles would make the AI term a near-constant that shifts every company by
the same amount. Ranking it first gives it the same spread as the other two
components, so its weight means what it says.

**Not backtestable.** The reviews are written today, with web search, about
businesses whose last decade is already known. There is no point-in-time AI
review for 2013, so no historical figure in `strategies.py` includes this term.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# The four scorecard dimensions, as `screener_cache` stores them.
AI_DIMENSIONS = (
    "ai_moat",
    "ai_financial_strength",
    "ai_predictability",
    "ai_growth",
)

# Columns this module adds to a ranking frame. Listed once so the route and the
# strategies agree on what a served row carries.
AI_COLUMNS = (*AI_DIMENSIONS, "ai_rating", "ai_score", "ai_reviewed_at")


def load_ai_reviews(symbols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    The latest scored AI review per symbol, indexed by symbol.

    A symbol can appear under several screener universes (S&P 500, a watchlist,
    a manual review); the most recently updated scored row wins. Returns an
    empty frame rather than raising when the screener cache is unavailable —
    the ranking must still be served without it.
    """
    from db_utils import get_db_connection, get_global_screener_db_path

    columns = ", ".join(AI_DIMENSIONS)
    query = (
        f"SELECT symbol, {columns}, updated_at FROM screener_cache "
        "WHERE " + " OR ".join(f"{c} > 0" for c in AI_DIMENSIONS)
    )
    try:
        conn = get_db_connection(get_global_screener_db_path(), use_cache=False)
        if conn is None:
            return pd.DataFrame(columns=[*AI_DIMENSIONS, "ai_reviewed_at"])
        try:
            frame = pd.read_sql_query(query, conn)
        finally:
            conn.close()
    except Exception as exc:
        logging.warning(f"AI review scores unavailable: {exc}")
        return pd.DataFrame(columns=[*AI_DIMENSIONS, "ai_reviewed_at"])

    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if symbols is not None:
        wanted = {str(s).upper() for s in symbols}
        frame = frame[frame["symbol"].isin(wanted)]

    frame = (
        frame.sort_values("updated_at", ascending=False, na_position="last")
        .drop_duplicates("symbol")
        .rename(columns={"updated_at": "ai_reviewed_at"})
        .set_index("symbol")
    )
    for column in AI_DIMENSIONS:
        values = pd.to_numeric(frame[column], errors="coerce")
        # 0 is the parser's "missing", not a verdict: the scale is 1-10.
        frame[column] = values.where(values > 0)
    return frame


def attach_ai_scores(
    frame: pd.DataFrame, reviews: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Add the AI columns to a ranking frame keyed by a `symbol` column.

    `ai_rating` is the mean of the dimensions present, on the review's own 1-10
    scale, for display. `ai_score` is that rating as a 0-100 percentile across
    the reviewed companies *in this frame* — the component the blend uses.
    Unreviewed companies get NaN in every AI column.
    """
    result = frame.copy()
    if result.empty or "symbol" not in result.columns:
        for column in AI_COLUMNS:
            result[column] = np.nan
        return result

    symbols = result["symbol"].astype(str).str.upper()
    if reviews is None:
        reviews = load_ai_reviews(symbols)

    aligned = reviews.reindex(symbols.values)
    for column in (*AI_DIMENSIONS, "ai_reviewed_at"):
        if column in aligned.columns:
            result[column] = aligned[column].to_numpy()
        else:
            result[column] = np.nan
    for column in AI_DIMENSIONS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    rating = result[list(AI_DIMENSIONS)].mean(axis=1, skipna=True)
    result["ai_rating"] = rating
    result["ai_score"] = rating.rank(pct=True, na_option="keep") * 100.0
    return result
