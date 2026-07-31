"""Buffett/value ranking routes.

Read-only views over the snapshots written by `buffett_pipeline`. Ranking runs
are batch jobs measured in minutes, so nothing here triggers one — these
endpoints serve the most recent completed snapshot.

The unranked bucket is exposed as a first-class endpoint rather than being
hidden. Roughly four fifths of the all-listed universe fails a quality gate, and
a client that cannot see why a company is absent has no way to tell a
deliberate exclusion from a data failure.
"""

# ruff: noqa: E402
import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from buffett_store import get_store
from server.auth import User
from server.dependencies import get_current_user
from server.route_utils import clean_nans

router = APIRouter()

# Valuation models a client may filter on. Validated against this set so a
# typo returns an empty page rather than a confusing 500 from SQLite.
_VALID_MODELS = {"generic", "bank", "insurer", "reit"}


# Security-registration boilerplate. 1,279 of the 1,305 distinct names in a run
# carry it ("Analog Devices, Inc. - Common Stock"), and it is the same on nearly
# every row, so it identifies nothing while pushing the part that does identify
# the company out of a narrow column.
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")
_SHARE_BOILERPLATE = re.compile(
    r"^(?P<base>.*?)[\s,\-–]*"
    r"(?P<klass>(?:Class\s+[A-Z]\b\s*)?)"
    r"(?:Non-Voting\s+)?(?:Common|Ordinary|Capital)\s+(?:Stock|Shares|Share)\b.*$",
    re.IGNORECASE,
)


def _display_name(name: Optional[str]) -> Optional[str]:
    """
    The company name with its share-class boilerplate removed.

    Applied here rather than in each client so the three of them cannot drift,
    and rather than at ingest so the stored name stays exactly what the filing
    said. A class designation survives — it is the only thing separating GOOG
    from GOOGL — as does the original whenever stripping would leave nothing.
    """
    if not name:
        return name

    cleaned = name.strip()
    while True:
        stripped = _TRAILING_PAREN.sub("", cleaned).strip()
        if stripped == cleaned:
            break
        cleaned = stripped

    match = _SHARE_BOILERPLATE.match(cleaned)
    if match:
        base = match.group("base").strip().strip(",-– ").strip()
        klass = match.group("klass").strip()
        if base:
            cleaned = f"{base} {klass}".strip()

    return cleaned or name.strip()


def _with_display_names(rows: list) -> list:
    """Rewrite each row's `name` in place-safe fashion for the response."""
    return [
        {**row, "name": _display_name(row.get("name"))}
        if isinstance(row, dict)
        else row
        for row in rows
    ]


class RankRunSummary(BaseModel):
    run_id: int
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    universe_size: Optional[int] = None
    ranked_count: Optional[int] = None
    excluded_count: Optional[int] = None


@router.get("/buffett-rank/latest")
async def get_latest_run(current_user: User = Depends(get_current_user)):
    """Metadata for the most recent completed ranking run."""
    try:
        run = await run_in_threadpool(get_store().get_run)
        if not run:
            raise HTTPException(status_code=404, detail="No completed ranking run yet")
        return clean_nans(run)
    except HTTPException:
        raise
    except Exception as exc:
        logging.error(f"Buffett rank: failed to load run metadata: {exc}")
        raise HTTPException(status_code=500, detail="Could not load ranking run")


@router.get("/buffett-rank")
async def get_rankings(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    model: Optional[str] = Query(None, description="generic, bank, insurer or reit"),
    search: Optional[str] = Query(None, description="Match symbol or company name"),
    run_id: Optional[int] = Query(None, description="Defaults to the latest run"),
    current_user: User = Depends(get_current_user),
):
    """
    The ranked list, best first.

    Each row carries its pillar breakdown and confidence so a client can explain
    a position without a second request. `search` is applied across the whole
    run rather than the returned page, so a client never has to load the full
    list to find one company.
    """
    if model and model not in _VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Expected one of {sorted(_VALID_MODELS)}",
        )
    try:
        store = get_store()
        rows = await run_in_threadpool(
            store.get_ranked, run_id, limit, offset, model, search
        )
        total = await run_in_threadpool(store.count_ranked, run_id, model, search)
        # Wrapped rather than a bare array: the client needs the match count to
        # distinguish "last page" from "no results", and to size the pager.
        return clean_nans({"total": total, "rows": _with_display_names(rows)})
    except Exception as exc:
        logging.error(f"Buffett rank: failed to load rankings: {exc}")
        raise HTTPException(status_code=500, detail="Could not load rankings")


@router.get("/buffett-rank/exclusions")
async def get_exclusions(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None, description="Match symbol or company name"),
    run_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
):
    """Companies excluded from the ranking, each with the reasons it failed."""
    try:
        store = get_store()
        rows = await run_in_threadpool(
            store.get_exclusions, run_id, limit, offset, search
        )
        total = await run_in_threadpool(store.count_exclusions, run_id, search)
        return clean_nans({"total": total, "rows": _with_display_names(rows)})
    except Exception as exc:
        logging.error(f"Buffett rank: failed to load exclusions: {exc}")
        raise HTTPException(status_code=500, detail="Could not load exclusions")


@router.get("/track-record/{symbol}")
async def get_track_record(
    symbol: str,
    current_user: User = Depends(get_current_user),
):
    """
    The measured quality record for one company: the same metrics the ranking
    scores on, labelled for a reader.

    404 for anything that does not file with the SEC — foreign listings and SET
    holdings among them. That is a normal state for a portfolio that isn't
    all-US, so clients hide the panel rather than showing an error.
    """
    import market_data
    import track_record

    cik = await run_in_threadpool(market_data.cik_for_symbol, symbol.upper())
    if not cik:
        raise HTTPException(
            status_code=404, detail=f"{symbol.upper()} does not file with the SEC"
        )

    try:
        record = await run_in_threadpool(track_record.build, symbol, cik)
    except Exception as exc:
        logging.error(f"Track record: failed to build for {symbol}: {exc}")
        raise HTTPException(status_code=500, detail="Could not build track record")

    if not record.get("period_count"):
        raise HTTPException(
            status_code=404, detail=f"No SEC fundamentals on file for {symbol.upper()}"
        )

    record["name"] = _display_name(record.get("name"))
    return clean_nans(record)


@router.get("/buffett-rank/history/{symbol}")
async def get_symbol_history(
    symbol: str,
    limit: int = Query(24, ge=1, le=200),
    current_user: User = Depends(get_current_user),
):
    """One company's rank across runs — the reason snapshots are kept."""
    try:
        rows = await run_in_threadpool(get_store().get_symbol_history, symbol, limit)
        return clean_nans(rows)
    except Exception as exc:
        logging.error(f"Buffett rank: failed to load history for {symbol}: {exc}")
        raise HTTPException(status_code=500, detail="Could not load rank history")
