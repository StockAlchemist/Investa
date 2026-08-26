#!/usr/bin/env python3
"""Fill the FX archive from the ECB's official reference rates (plan Phase 5.1).

Every rate in `daily_fx` comes from Yahoo, so the currency conversion behind
every portfolio figure fails whenever the price feed does — and it already has:
`EUR=X`, `GBP=X` and `CNY=X` were frozen at 15 Jun 2026 until the Tier A
backfill noticed in late August. A frozen rate does not look like an outage. The
numbers keep rendering, two months stale.

The ECB publishes its reference rates as static files, free and without a key,
back to 1999. This loads them as a *second* source rather than a replacement:

  * A day the archive already holds is left alone. The ECB fixes at 14:15 CET
    and Yahoo takes a close, which over ~5,500 overlapping days disagree by a
    median 0.21%. Overwriting would move every historical portfolio figure by
    that much and make neither series truer. `--overwrite` exists, is not the
    default, and moves the golden gate if you use it.
  * A day the archive lacks is written, tagged `source='ecb'`. There the
    alternative is not a slightly different rate — it is the last known rate
    carried forward, which is how a two-month freeze reads as data.
  * A stored rate that is not a number at all is repaired either way. Those are
    corrupt, not a second opinion (`USD=X` held its synthetic 1 as an 8-byte
    BLOB, because a numpy scalar stores through sqlite3 as a buffer).

**Two feeds, because neither covers the book alone.** THB and CNY only joined
the ECB list on 2005-04-01, so the euro reference rates cannot price the
ledger's first three Thai years — and the baht is the base currency, so those
years are exactly the ones that matter. `--provider bot` reads the Bank of
Thailand's own daily average series, which starts 2002-01-02 and covers the
ledger entirely. It needs `BOT_API_KEY` (see `src/bot_fx_provider.py`).

The BOT half only requests the windows that hold a day the archive is missing.
It serves 31 days per call, so the full span is ~290 calls and the gateway's
quota refuses that long before it finishes — with no `Retry-After` and no
`X-RateLimit-*` header to pace against. Asking only for the gaps turns the first
fill into 28 calls, and a nightly run into one or none.

    python scripts/backfill_fx_rates.py                    # dry run: what would change
    python scripts/backfill_fx_rates.py --apply
    python scripts/backfill_fx_rates.py --apply --recent   # nightly: last 90 days
    python scripts/backfill_fx_rates.py --pair SGD=X --apply
    python scripts/backfill_fx_rates.py --provider bot --apply          # THB, from 2002
    python scripts/backfill_fx_rates.py --provider bot --recent --apply
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from typing import Dict, List, Sequence, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import bot_fx_provider as bot  # noqa: E402
import config  # noqa: E402
import ecb_fx_provider as ecb  # noqa: E402
from db_utils import connect_readonly  # noqa: E402
from fx_pairs import pair_series, split_pair  # noqa: E402
from market_db import MarketDatabase  # noqa: E402

# Below this the two providers are simply quoting different moments of the same
# day; above it, one of them is wrong about something and the pair is worth a
# look before its gaps are filled from the other.
DIVERGENCE_ALERT_PCT = 2.0

# The BOT publishes 19 currencies, but the ECB already covers the others from
# 1999 — three years before the BOT series even starts. What only the BOT has is
# the baht before Apr 2005, so that is what it is asked for by default.
BOT_DEFAULT_PAIRS = ["THB=X", "USDTHB=X", "THBUSD=X"]


def db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def stored_pairs(path: str) -> List[str]:
    conn = connect_readonly(path)
    try:
        return [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT pair FROM daily_fx WHERE interval = '1d' ORDER BY pair"
            )
        ]
    finally:
        conn.close()


def stored_rates(path: str, pair: str) -> Dict[str, object]:
    """Every stored day for a pair, rates left exactly as sqlite returns them.

    Deliberately untyped: a BLOB or a NULL here is a finding, not something to
    coerce away quietly.
    """
    conn = connect_readonly(path)
    try:
        return {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT date, rate FROM daily_fx WHERE pair = ? AND interval = '1d'",
                (pair,),
            )
        }
    finally:
        conn.close()


def as_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if number == number and number > 0 else None
    return None


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


class PairPlan:
    """What this run would do to one pair, before anything is written."""

    def __init__(self, pair: str, source: str = "feed"):
        self.pair = pair
        self.source = source
        self.fills: List[Tuple[str, float]] = []  # days the archive lacks
        self.repairs: List[Tuple[str, float]] = []  # days stored as non-numbers
        self.overlap = 0
        self.diffs: List[float] = []
        # (diff%, day, stored, feed) for the few worst days, so a flagged pair
        # says which days provoked it instead of only that something did.
        self.worst: List[Tuple[float, str, float, float]] = []
        self.days = 0
        self.first_day = ""

    @property
    def touches(self) -> int:
        return len(self.fills) + len(self.repairs)

    def summary(self) -> str:
        median = percentile(self.diffs, 50)
        p95 = percentile(self.diffs, 95)
        worst = max(self.diffs) if self.diffs else 0.0
        flag = "  <-- check" if worst >= DIVERGENCE_ALERT_PCT * 5 else ""
        detail = ""
        if flag:
            for diff, day, stored, feed in sorted(self.worst, reverse=True)[:3]:
                detail += (
                    f"\n      {day}  stored {stored:10.4f}  {self.source} {feed:10.4f}"
                    f"  {diff:6.2f}%"
                )
        return (
            f"  {self.pair:10} {self.source:3} {self.days:5d} from {self.first_day or '-':10}  "
            f"overlap {self.overlap:5d}  diff med {median:5.2f}% p95 {p95:5.2f}% "
            f"max {worst:6.2f}%  fill {len(self.fills):4d}  repair {len(self.repairs):3d}{flag}"
            + detail
        )


def missing_windows(path: str, pairs: List[str], begin: date, end: date) -> List[tuple]:
    """The request windows that actually contain a day the archive is missing.

    The BOT serves 31 days a request, so the whole series is ~290 of them — and
    the gateway's quota refuses that long before it finishes, with no
    `X-RateLimit-*` header to pace against and no `Retry-After` to wait on. It
    is also unnecessary: a fill-only run writes nothing on a day already stored,
    so requesting one is pure cost.

    Weekends are excluded from what counts as missing. The BOT publishes on
    business days only, so a Saturday can never be filled and must not be
    allowed to pull a whole window into the run.
    """
    have = {pair: stored_rates(path, pair) for pair in pairs}
    needed: List[date] = []
    day = begin
    while day <= end:
        if day.weekday() < 5:
            iso = day.isoformat()
            if any(iso not in have[pair] for pair in pairs):
                needed.append(day)
        day += timedelta(days=1)
    if not needed:
        return []

    windows: List[tuple] = []
    for start, stop in bot.month_windows(begin, end):
        if any(start <= d <= stop for d in needed):
            windows.append((start, stop))
    return windows


def unit_source(unit: str) -> str:
    """Which feed a table quoted in `unit` came from — for the report only."""
    return bot.SOURCE if unit == bot.UNIT else ecb.SOURCE


def build_plan(
    path: str,
    pair: str,
    rates,
    start: date | None,
    overwrite: bool,
    unit: str = ecb.UNIT,
) -> PairPlan:
    plan = PairPlan(pair, unit_source(unit))
    series = pair_series(rates, pair, unit)
    if start:
        floor = start.isoformat()
        series = [(day, rate) for day, rate in series if day >= floor]
    plan.days = len(series)
    plan.first_day = series[0][0] if series else ""
    if not series:
        return plan

    # USD=X is a currency against itself: a flat 1.0 on every day the ECB
    # published. Backfilling twenty-seven years of that would add 7,000 rows
    # carrying no information — and nothing reads it, because both valuation
    # paths short-circuit the base currency to a synthetic 1.0 series. The rows
    # already stored are another matter: they hold their 1 as an 8-byte BLOB and
    # read back as bytes, so they are still repaired — and repaired on every
    # stored day, not only the days the ECB happened to publish. A currency
    # against itself is 1.0 by definition; a provider is not needed to say so,
    # which matters because two of the three bad rows fell on a Saturday and on
    # a day whose fix was not out yet.
    legs = split_pair(pair)
    identity = bool(legs) and legs[0] == legs[1]

    existing = stored_rates(path, pair)
    if identity:
        plan.repairs = [
            (day, 1.0)
            for day, value in sorted(existing.items())
            if as_number(value) != 1.0
        ]
        return plan

    for day, rate in series:
        if day not in existing:
            plan.fills.append((day, rate))
            continue
        current = as_number(existing[day])
        if current is None:
            plan.repairs.append((day, rate))
            continue
        plan.overlap += 1
        diff = abs(rate - current) / current * 100.0
        plan.diffs.append(diff)
        plan.worst.append((diff, day, current, rate))
        plan.worst = sorted(plan.worst, reverse=True)[:3]
        if overwrite:
            plan.repairs.append((day, rate))
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=None, help="path to market_data.db")
    parser.add_argument(
        "--provider",
        choices=("ecb", "bot"),
        default="ecb",
        help="ecb: everything from 1999, THB and CNY from Apr 2005. "
        "bot: the official Thai rate, THB from Jan 2002 — the only source for "
        "the years the ECB does not carry",
    )
    parser.add_argument(
        "--pair",
        action="append",
        default=None,
        help="pair to fill (repeatable); default is every pair already stored "
        "(ECB) or the THB pairs (BOT)",
    )
    parser.add_argument(
        "--recent",
        action="store_true",
        help="ECB: read the 90-day file instead of the full history. "
        "BOT: fetch the last 31 days rather than the whole series",
    )
    parser.add_argument(
        "--start", default=None, help="ignore published days before this date"
    )
    parser.add_argument("--end", default=None, help="BOT: last day to request")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace stored rates on days the archive already has "
        "(rebases the series onto this provider — moves portfolio history)",
    )
    parser.add_argument(
        "--apply", action="store_true", help="write; default is a dry run"
    )
    args = parser.parse_args()

    path = args.db or db_path()
    if not os.path.exists(path):
        print(f"No market database at {path}.")
        return 1

    start = date.fromisoformat(args.start) if args.start else None
    is_bot = args.provider == "bot"
    unit = bot.UNIT if is_bot else ecb.UNIT
    source = bot.SOURCE if is_bot else ecb.SOURCE

    pairs = args.pair or (BOT_DEFAULT_PAIRS if is_bot else stored_pairs(path))
    if not pairs:
        print("No FX pairs to fill.")
        return 0

    try:
        if is_bot:
            end = date.fromisoformat(args.end) if args.end else date.today()
            begin = start or (
                end - timedelta(days=bot.MAX_PERIOD_DAYS - 1)
                if args.recent
                else bot.SERIES_START
            )
            provider = bot.BOTFXProvider()
            begin = max(begin, bot.SERIES_START)
            windows = (
                bot.month_windows(begin, end)
                if args.overwrite
                else missing_windows(path, pairs, begin, end)
            )
            whole = len(bot.month_windows(begin, end))
            print(
                f"BOT daily average rates ({begin} to {end}) -> {path}\n"
                f"{len(windows)} of {whole} window(s) hold a day the archive lacks"
            )
            if not windows:
                print("Nothing missing in that range.")
                return 0
            # The loop below fetches a window at a time, so the provider's own
            # guard would only ever see one call. Check the whole run here.
            provider.check_budget(len(windows))

            def note(done, total, w_start, w_end, so_far):
                print(f"  {done}/{total}  {w_start} → {w_end}  {so_far} days")

            # Every currency costs the same number of requests as one, so ask
            # for all of them and let the pair list decide what gets stored.
            rates = {}
            for index, (w_start, w_stop) in enumerate(windows):
                rates.update(
                    provider.fetch_daily_avg(w_start, w_stop, currency=None, throttle=0)
                )
                note(index + 1, len(windows), w_start, w_stop, len(rates))
                if index + 1 < len(windows):
                    time.sleep(bot.THROTTLE_SECONDS)
        else:
            provider = ecb.ECBFXProvider()
            window = "the last 90 days" if args.recent else "1999 to date"
            print(f"ECB reference rates ({window}) -> {path}")
            rates = (
                provider.fetch_recent() if args.recent else provider.fetch_history()
            )
    except (ecb.ECBFXError, bot.BOTFXError) as exc:
        print(f"FAILED: {exc}")
        return 1

    if not rates:
        print("Provider returned no rates for that range.")
        return 1
    print(f"{len(rates)} published days, latest {max(rates)}\n")

    plans = [
        build_plan(path, pair, rates, start, args.overwrite, unit) for pair in pairs
    ]
    for plan in sorted(plans, key=lambda p: p.pair):
        print(plan.summary())

    unpriceable = [p.pair for p in plans if not p.days]
    if unpriceable:
        print(f"\nNot derivable from this feed: {', '.join(unpriceable)}")

    fills = sum(len(p.fills) for p in plans)
    repairs = sum(len(p.repairs) for p in plans)
    print(f"\n{fills} day(s) to fill, {repairs} to overwrite/repair.")

    if not args.apply:
        print("Dry run — nothing written. Re-run with --apply.")
        return 0
    if fills + repairs == 0:
        print("Nothing to write.")
        return 0

    db = MarketDatabase(path)
    written = 0
    for plan in plans:
        if plan.fills:
            written += db.upsert_fx_rows(
                plan.pair, plan.fills, source=source, fill_only=True
            )
        if plan.repairs:
            # Not fill_only: these days are being replaced on purpose, either
            # because what is stored is not a number or because --overwrite was
            # asked for explicitly.
            written += db.upsert_fx_rows(
                plan.pair, plan.repairs, source=source, fill_only=False
            )
    print(f"Wrote {written} row(s) tagged source='{source}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
