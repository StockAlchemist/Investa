#!/usr/bin/env python3
"""
Golden snapshot / diff for the market-archive migration (plan Phase 1.7).

The raw-storage conversion is designed to be numerically invisible: reading a
converted symbol with adjust='split' must reproduce, to the bit, the value that
was stored before conversion. This script is the gate that proves it.

It captures three layers, weakest coupling last:

  1. archive  — every (symbol, date, close) in daily_ohlcv, hashed per symbol.
                Exact and cheap. This is the layer the conversion touches, so
                it is the one that must not move.
  2. portfolio — the daily portfolio value series per user, from the same
                function GET /api/history calls. Catches integration mistakes
                the archive hash cannot see (a caller reading a different
                column, an adjustment applied twice).
  3. ranking  — the top 20 of the newest finished Buffett run.

Usage:
    cd src && python ../scripts/golden_snapshot.py capture --label before
    #   ... make changes ...
    cd src && python ../scripts/golden_snapshot.py capture --labelapply after
    cd src && python ../scripts/golden_snapshot.py diff before after

Snapshots land in data/backups/golden/<label>/. They are plain JSON so a diff
can be eyeballed or committed alongside a migration PR.

--layers lets you capture a subset; the portfolio layer is much slower than the
other two (a cold full-history recompute per user), so `--layers archive` is the
right choice for a quick inner-loop check.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

# Run as if from src/ (mirrors how uvicorn launches, and how profile_history.py
# sets itself up) so the `import market_data` style used across src/ resolves.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC)

from db_utils import connect_readonly  # noqa: E402

SNAPSHOT_ROOT = os.path.join(REPO_ROOT, "data", "backups", "golden")
MARKET_DB = os.path.join(REPO_ROOT, "data", "db", "market_data.db")
RANKS_DB = os.path.join(REPO_ROOT, "data", "db", "buffett_ranks.db")

DEFAULT_USERS = ["kitmatan", "dheematan"]
ALL_LAYERS = ("archive", "portfolio", "ranking")


# --- layer 1: the archive itself -------------------------------------------


def _default_as_of() -> str:
    """
    Strictly before today's market date.

    Today's bar is live: while a market is open the refresh worker rewrites it
    every 15 minutes, so including it makes the hash unstable for reasons that
    have nothing to do with the migration. Measured: two captures two seconds
    apart moved 8 symbols with identical row counts and date ranges. The
    migration only concerns settled history, so cutting today costs nothing.
    """
    try:
        from utils_time import get_est_today

        today = get_est_today()
    except Exception:
        from datetime import date as _date

        today = _date.today()
    return today.isoformat()


def capture_archive(as_of: str | None = None) -> dict:
    """
    Per-symbol hash of the close series, plus enough shape to localize a
    mismatch to a symbol without storing 647k rows of JSON.

    Hashes `close` only. open/high/low/volume ride along through the same
    conversion arithmetic, but close is what every downstream consumer reads,
    so a close-level match is the claim that matters. adj_close is deliberately
    excluded: the plan deprecates it (defect D3) and it is expected to change.

    Rows on or after `as_of` are excluded — see _default_as_of.
    """
    if not os.path.exists(MARKET_DB):
        return {"error": "market_data.db not found"}

    as_of = as_of or _default_as_of()

    con = connect_readonly(MARKET_DB)
    try:
        rows = con.execute(
            "SELECT symbol, date, close FROM daily_ohlcv "
            "WHERE interval = '1d' AND date < ? ORDER BY symbol, date",
            (as_of,),
        ).fetchall()
    finally:
        con.close()

    per_symbol: dict[str, dict] = {}
    digest: hashlib._Hash | None = None
    current: str | None = None

    def flush(sym, h, n, first, last):
        per_symbol[sym] = {
            "rows": n,
            "first": first,
            "last": last,
            "sha256": h.hexdigest(),
        }

    n = 0
    first = last = None
    for symbol, day, close in rows:
        if symbol != current:
            if current is not None:
                flush(current, digest, n, first, last)
            current, digest, n, first = symbol, hashlib.sha256(), 0, day
        # repr() of a float round-trips exactly in CPython, so this hash moves
        # if and only if a stored value actually changed.
        digest.update(f"{day}|{close!r}\n".encode())
        n += 1
        last = day
    if current is not None:
        flush(current, digest, n, first, last)

    overall = hashlib.sha256()
    for sym in sorted(per_symbol):
        overall.update(f"{sym}:{per_symbol[sym]['sha256']}\n".encode())

    return {
        "as_of_exclusive": as_of,
        "symbols": len(per_symbol),
        "total_rows": sum(v["rows"] for v in per_symbol.values()),
        "sha256": overall.hexdigest(),
        "per_symbol": per_symbol,
    }


# --- layer 2: portfolio value series ---------------------------------------


def capture_portfolio(users: list[str], currency: str = "THB", as_of: str | None = None) -> dict:
    """
    Daily portfolio value series per user, via the same internal function the
    /api/history endpoint calls. Benchmarks are skipped so the snapshot does not
    depend on a live network fetch of index data.
    """
    import asyncio

    import pandas as pd  # noqa: F401  (imported for its side effect on dtypes)

    os.chdir(SRC)
    from server.auth import User
    from server.dependencies import get_transaction_data
    from server.portfolio_service import _calculate_historical_performance_internal

    out: dict[str, dict] = {}
    for username in users:
        try:
            user = User(id=1, username=username, alias=None, is_active=True, created_at="")
            data = get_transaction_data(user)
            if data[0].empty:
                out[username] = {"error": "no transactions"}
                continue

            df = asyncio.run(
                _calculate_historical_performance_internal(
                    currency=currency,
                    period="all",  # NB: "all", not "max" — an unknown token
                    # falls through to the 1y default, which would silently
                    # snapshot one year of a twenty-four year ledger.
                    accounts=None,
                    benchmarks=[],
                    data=data,
                    return_df=True,
                    interval="1d",
                    force=True,  # bypass caches: we want a real recompute
                )
            )
            out[username] = _summarize_value_series(df, as_of)
        except Exception as exc:  # noqa: BLE001 - a failed user must not abort the run
            out[username] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _summarize_value_series(df, as_of: str | None = None) -> dict:
    """
    Hash the daily value series and keep month-end points for a readable diff.

    The frame comes back with a RangeIndex and an explicit `date` column, so the
    date is taken from that column rather than the index. Rows on/after `as_of`
    are dropped for the same reason the archive layer drops them: today's bar is
    still moving while a market is open.
    """
    if df is None or getattr(df, "empty", True):
        return {"error": "empty result"}

    import pandas as pd

    if "date" not in df.columns or "value" not in df.columns:
        return {"error": f"unexpected columns: {list(df.columns)[:20]}"}

    frame = df[["date", "value"]].dropna()
    dates = pd.to_datetime(frame["date"], utc=True)
    frame = frame.assign(_day=dates.dt.strftime("%Y-%m-%d"))

    as_of = as_of or _default_as_of()
    frame = frame[frame["_day"] < as_of]
    if frame.empty:
        return {"error": f"no rows before {as_of}"}

    digest = hashlib.sha256()
    for day, val in zip(frame["_day"], frame["value"]):
        digest.update(f"{day}|{round(float(val), 6)!r}\n".encode())

    # Month-end points: last observation of each calendar month.
    monthly = frame.groupby(frame["_day"].str.slice(0, 7))["value"].last()

    return {
        "as_of_exclusive": as_of,
        "rows": int(len(frame)),
        "first": frame["_day"].iloc[0],
        "last": frame["_day"].iloc[-1],
        "final_value": round(float(frame["value"].iloc[-1]), 4),
        "sha256": digest.hexdigest(),
        "monthly": {k: round(float(v), 4) for k, v in monthly.items()},
    }


# --- layer 3: ranking -------------------------------------------------------


def capture_ranking(top_n: int = 20) -> dict:
    """
    Top N of the newest *finished* run — the same selection the app makes, so
    this snapshot tracks what a user would actually see.

    Captures `price` alongside the scores because Phase 4.3 repoints the
    ranking's price source at the local archive; a shift there shows up here
    first, and per-symbol prices localize it immediately.
    """
    if not os.path.exists(RANKS_DB):
        return {"error": "buffett_ranks.db not found"}

    con = connect_readonly(RANKS_DB)
    try:
        run = con.execute(
            "SELECT MAX(run_id) FROM rank_runs WHERE finished_at IS NOT NULL"
        ).fetchone()[0]
        if run is None:
            return {"error": "no finished runs"}

        rows = con.execute(
            "SELECT rank, symbol, composite_score, quality_score, value_score, price "
            "FROM rank_scores WHERE run_id = ? ORDER BY rank ASC LIMIT ?",
            (run, top_n),
        ).fetchall()
    finally:
        con.close()

    def r6(v):
        return round(float(v), 6) if v is not None else None

    return {
        "run_id": run,
        "top": [
            {
                "rank": rank,
                "symbol": sym,
                "composite": r6(comp),
                "quality": r6(qual),
                "value": r6(val),
                "price": r6(price),
            }
            for rank, sym, comp, qual, val, price in rows
        ],
    }


# --- capture / diff ---------------------------------------------------------


def capture(label: str, layers: tuple[str, ...], users: list[str], as_of: str | None = None) -> str:
    snap = {
        "label": label,
        "captured_at": datetime.now().astimezone().isoformat(),
        "layers": list(layers),
    }
    if "archive" in layers:
        print("  archive...", flush=True)
        snap["archive"] = capture_archive(as_of)
    if "ranking" in layers:
        print("  ranking...", flush=True)
        snap["ranking"] = capture_ranking()
    if "portfolio" in layers:
        print("  portfolio (slow, full recompute per user)...", flush=True)
        snap["portfolio"] = capture_portfolio(users, as_of=as_of)

    out_dir = os.path.join(SNAPSHOT_ROOT, label)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "snapshot.json")
    with open(path, "w") as fh:
        json.dump(snap, fh, indent=1, sort_keys=True)
    return path


def _load(label: str) -> dict:
    path = os.path.join(SNAPSHOT_ROOT, label, "snapshot.json")
    if not os.path.exists(path):
        sys.exit(f"No snapshot labelled '{label}' at {path}")
    with open(path) as fh:
        return json.load(fh)


def diff(before_label: str, after_label: str) -> int:
    before, after = _load(before_label), _load(after_label)
    findings: list[str] = []

    # archive
    b, a = before.get("archive"), after.get("archive")
    if b and a and "error" not in b and "error" not in a:
        if b.get("as_of_exclusive") != a.get("as_of_exclusive"):
            # Different cutoffs compare different row sets; the hashes cannot
            # agree even if nothing changed. Recapture rather than trust it.
            findings.append(
                f"archive    INVALID  cutoffs differ: {b.get('as_of_exclusive')} vs "
                f"{a.get('as_of_exclusive')} — recapture both with the same --as-of"
            )
        elif b["sha256"] == a["sha256"]:
            print(f"archive    OK   {b['symbols']} symbols, {b['total_rows']} rows, hash unchanged")
        else:
            bs, as_ = b["per_symbol"], a["per_symbol"]
            changed = [s for s in bs.keys() & as_.keys() if bs[s]["sha256"] != as_[s]["sha256"]]
            added, removed = sorted(as_.keys() - bs.keys()), sorted(bs.keys() - as_.keys())
            findings.append(
                f"archive    MOVED  {len(changed)} symbols changed, "
                f"{len(added)} added, {len(removed)} removed"
            )
            for s in sorted(changed)[:20]:
                findings.append(
                    f"             {s}: {bs[s]['rows']}→{as_[s]['rows']} rows, "
                    f"{bs[s]['first']}..{bs[s]['last']} → {as_[s]['first']}..{as_[s]['last']}"
                )
            if len(changed) > 20:
                findings.append(f"             ... and {len(changed) - 20} more")
            if added:
                findings.append(f"             added: {', '.join(added[:15])}")
            if removed:
                findings.append(f"             removed: {', '.join(removed[:15])}")

    # portfolio
    b, a = before.get("portfolio"), after.get("portfolio")
    if b and a:
        for user in sorted(b.keys() | a.keys()):
            bu, au = b.get(user, {}), a.get(user, {})
            if "error" in bu or "error" in au:
                findings.append(f"portfolio  SKIP {user}: {bu.get('error') or au.get('error')}")
            elif bu.get("sha256") == au.get("sha256"):
                print(f"portfolio  OK   {user}: {bu['rows']} days, final {bu['final_value']}, hash unchanged")
            else:
                findings.append(
                    f"portfolio  MOVED  {user}: final {bu.get('final_value')} → {au.get('final_value')}"
                )
                bm, am = bu.get("monthly", {}), au.get("monthly", {})
                moved = [k for k in sorted(bm.keys() & am.keys()) if bm[k] != am[k]]
                for k in moved[:10]:
                    findings.append(f"             {k}: {bm[k]} → {am[k]}")
                if len(moved) > 10:
                    findings.append(f"             ... and {len(moved) - 10} more months")

    # ranking
    b, a = before.get("ranking"), after.get("ranking")
    if b and a and "error" not in b and "error" not in a:
        bt = [x["symbol"] for x in b["top"]]
        at = [x["symbol"] for x in a["top"]]
        bmap = {x["symbol"]: x for x in b["top"]}
        amap = {x["symbol"]: x for x in a["top"]}
        moved = [
            s
            for s in bmap.keys() & amap.keys()
            if (bmap[s]["composite"], bmap[s]["price"]) != (amap[s]["composite"], amap[s]["price"])
        ]
        if bt == at and not moved:
            print(f"ranking    OK   top {len(bt)} unchanged (run {b['run_id']} → {a['run_id']})")
        else:
            findings.append(f"ranking    MOVED  (run {b['run_id']} → {a['run_id']})")
            if bt != at:
                findings.append(f"             order before: {', '.join(bt)}")
                findings.append(f"             order after:  {', '.join(at)}")
            for s in sorted(moved)[:10]:
                findings.append(
                    f"             {s}: composite {bmap[s]['composite']} → {amap[s]['composite']}, "
                    f"price {bmap[s]['price']} → {amap[s]['price']}"
                )
            if len(moved) > 10:
                findings.append(f"             ... and {len(moved) - 10} more")

    if findings:
        print("\n".join(findings))
        print(
            "\nFAIL — the conversion is meant to be numerically invisible.\n"
            "Every line above is a bug until explained."
        )
        return 1

    print("\nPASS — no movement across captured layers.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", help="record a snapshot")
    cap.add_argument("--label", required=True, help="snapshot name, e.g. 'before'")
    cap.add_argument(
        "--layers",
        default="archive,ranking",
        help=f"comma-separated subset of {','.join(ALL_LAYERS)} (default: archive,ranking)",
    )
    cap.add_argument("--users", default=",".join(DEFAULT_USERS))
    cap.add_argument(
        "--as-of",
        dest="as_of",
        default=None,
        help="exclude archive rows on/after this yyyy-MM-dd (default: today's market date, "
        "because today's bar is still moving)",
    )

    d = sub.add_parser("diff", help="compare two snapshots")
    d.add_argument("before")
    d.add_argument("after")

    args = parser.parse_args()

    if args.cmd == "capture":
        layers = tuple(x.strip() for x in args.layers.split(",") if x.strip())
        bad = set(layers) - set(ALL_LAYERS)
        if bad:
            sys.exit(f"Unknown layer(s): {', '.join(sorted(bad))}. Valid: {', '.join(ALL_LAYERS)}")
        users = [u.strip() for u in args.users.split(",") if u.strip()]
        print(f"Capturing '{args.label}' [{', '.join(layers)}]")
        path = capture(args.label, layers, users, args.as_of)
        print(f"Wrote {path}")
        return 0

    return diff(args.before, args.after)


if __name__ == "__main__":
    sys.exit(main())
