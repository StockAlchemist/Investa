#!/usr/bin/env python3
"""Nightly check: does the price series actually reflect the splits on record?

A split that has been applied is invisible — the provider divides every earlier
price by the ratio, so the line stays continuous. A split that has *not* been
applied leaves the ratio sitting in the data as a jump. That asymmetry is the
whole detector: a price move matching the recorded ratio near an ex-date is
evidence the series was never adjusted.

Why this runs nightly rather than being fixed on sight. Repairing needs an
independent reference — see `repair_bars_against_reference.py` for why price
data alone cannot say which side of a disagreement is right — and that reference
has to be collected by hand. What can be automated is *noticing*, quickly, so a
bad series is a known item rather than something discovered in a backtest months
later.

Two shapes are looked for, because Yahoo produces both:

  unapplied  a jump matching the ratio near the ex-date: the series carries the
             pre-split basis and nobody rebased it. WLFC, 3:1, July 2026.
  mixed      individual bars off by the ratio from both neighbours: the provider
             returned adjusted data on some dates and not others. AEHL and CURX
             were wrong on exactly the same days, which is how you can tell it
             is the provider and not the company.

Findings are remembered between runs, so a nightly run reports what is *new* and
exits non-zero only then — quiet unless something changed, which is the only way
a scheduled check stays worth reading.

    python scripts/check_split_consistency.py                # last 45 days
    python scripts/check_split_consistency.py --all          # whole archive
    python scripts/check_split_consistency.py --days 90 --quiet

Nightly, after the delta:

    30 2 * * *  cd /path/to/Investa && python3 scripts/backfill_market_history.py \\
                  --tier A --days 5 --apply && \\
                python3 scripts/check_split_consistency.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, NamedTuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402
from db_utils import connect_readonly  # noqa: E402

# A ratio nearer to 1 than this cannot be told from an ordinary session: a 1.046
# spin-off ratio matches any day the stock moved 4.6%. Flagging those produced
# 515 false positives out of 586 on the first attempt, five of them on held
# symbols. Same line the verifier draws.
MIN_DEVIATION = 1.4

# How close an observed move must sit to the ratio to be attributed to it.
RATIO_TOLERANCE = 0.06

DEFAULT_WINDOW_DAYS = 45


class Finding(NamedTuple):
    symbol: str
    ex_date: str
    ratio: float
    shape: str          # 'unapplied' | 'mixed'
    detail: str

    @property
    def key(self) -> str:
        return f"{self.symbol}|{self.ex_date}|{self.shape}"


def _ordinal(day: str) -> int:
    """Days since epoch for a yyyy-MM-dd string, for cheap date arithmetic."""
    from datetime import date as _d

    y, m, d = (int(p) for p in day[:10].split("-"))
    return _d(y, m, d).toordinal()


def db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def state_path() -> str:
    return os.path.join(config.get_app_data_dir(), "reference", "split_check_state.json")


def load_state() -> Dict[str, str]:
    try:
        with open(state_path()) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(state_path()), exist_ok=True)
    with open(state_path(), "w") as fh:
        json.dump(state, fh, indent=1, sort_keys=True)


def check(conn, since: str | None) -> List[Finding]:
    query = (
        "SELECT symbol, date, value FROM corporate_action "
        "WHERE kind = 'split' AND value > 0"
    )
    params: list = []
    if since:
        query += " AND date >= ?"
        params.append(since)

    findings: List[Finding] = []
    for symbol, ex_date, ratio in conn.execute(query, params):
        if not ratio or max(ratio, 1.0 / ratio) < MIN_DEVIATION:
            continue

        bars = conn.execute(
            """
            SELECT date, close FROM daily_ohlcv
            WHERE symbol = ? AND interval = '1d' AND close > 0
              AND date BETWEEN date(?, '-30 day') AND date(?, '+15 day')
            ORDER BY date
            """,
            (symbol, ex_date, ex_date),
        ).fetchall()
        if len(bars) < 3:
            continue

        # A step matching the ratio. Where it sits decides what it means: on the
        # ex-date it is a split nobody applied; well away from it, the ratio is
        # showing up mid-series, which is the interleaved-basis shape wearing a
        # step's clothing. Calling both "unapplied" sent the reader looking for
        # a boundary that is not there.
        for (d1, c1), (d2, c2) in zip(bars, bars[1:]):
            if not (c1 and c2):
                continue
            observed = c1 / c2
            if abs(observed - ratio) / ratio < RATIO_TOLERANCE:
                on_ex_date = abs(
                    (_ordinal(d2) - _ordinal(ex_date))
                ) <= 3
                findings.append(
                    Finding(
                        symbol, ex_date, float(ratio),
                        "unapplied" if on_ex_date else "ratio-step-off-ex-date",
                        f"{d1} {c1:.4f} -> {d2} {c2:.4f} (x{observed:.3f})",
                    )
                )
                break

        # mixed: a bar off by the ratio from BOTH neighbours. Reported, never
        # acted on — when most bars are wrong this flags the right ones.
        strays = []
        for i in range(1, len(bars) - 1):
            (_, prev), (day, cur), (_, nxt) = bars[i - 1], bars[i], bars[i + 1]
            if not (prev and cur and nxt):
                continue
            for factor in (ratio, 1.0 / ratio):
                if (
                    abs(cur / prev - factor) / factor < 0.08
                    and abs(cur / nxt - factor) / factor < 0.08
                ):
                    strays.append(day)
                    break
        if strays:
            findings.append(
                Finding(
                    symbol, ex_date, float(ratio), "mixed",
                    f"{len(strays)} bar(s) on a different basis: {', '.join(strays[:4])}",
                )
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=None)
    parser.add_argument("--days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--all", action="store_true", help="check every recorded split")
    parser.add_argument("--quiet", action="store_true", help="print only new findings")
    parser.add_argument(
        "--forget", action="store_true", help="clear the seen-state and report everything as new"
    )
    args = parser.parse_args()

    path = args.db or db_path()
    if not os.path.exists(path):
        print(f"No market database at {path}.")
        return 0

    since = None
    if not args.all:
        conn = connect_readonly(path)
        try:
            row = conn.execute(
                "SELECT date(MAX(date), ?) FROM daily_ohlcv WHERE interval = '1d'",
                (f"-{args.days} day",),
            ).fetchone()
        finally:
            conn.close()
        since = row[0] if row else None

    conn = connect_readonly(path)
    try:
        findings = check(conn, since)
    finally:
        conn.close()

    state = {} if args.forget else load_state()
    new = [f for f in findings if f.key not in state]

    scope = "every recorded split" if args.all else f"splits since {since}"
    if not args.quiet:
        print(f"Checked {scope}: {len(findings)} finding(s), {len(new)} new.")

    if new:
        print(f"\n{len(new)} NEW split-consistency finding(s):\n")
        for f in sorted(new, key=lambda x: x.ex_date, reverse=True):
            print(f"  {f.symbol:8} {f.ex_date}  ratio {f.ratio:<9.4f} [{f.shape}]")
            print(f"           {f.detail}")
        print(
            "\nRepair needs an independent reference — collect IBKR bars for these\n"
            "symbols, load with ingest_ibkr_actions.py, then run\n"
            "repair_bars_against_reference.py. Price data alone cannot decide."
        )
    elif not args.quiet:
        print("Nothing new.")

    for f in findings:
        state.setdefault(f.key, f.ex_date)
    save_state(state)

    return 1 if new else 0


if __name__ == "__main__":
    raise SystemExit(main())
