"""Bounded, cached US daily-price fallback; Yahoo remains the primary feed.

Fetch through today even for an old window: splits AFTER the requested window
are needed to put raw Tiingo prices on Yahoo's split-only adjusted basis.
Never use adjClose here, since that also adjusts for dividends.
"""

import logging
import math
import re
import threading
import time
from collections import OrderedDict
from datetime import date

import pandas as pd

import config
from tiingo_provider import TiingoError, TiingoProvider
from utils_time import get_est_today

logger = logging.getLogger(__name__)


def eligible_symbol(symbol: str) -> bool:
    # Yahoo's US share classes already use Tiingo's hyphen convention. Exclude
    # foreign suffixes, indices, FX, futures, and Yahoo crypto pairs.
    return bool(re.fullmatch(r"[A-Z][A-Z0-9]{0,9}(?:-[A-Z])?", symbol))


def price_frame(rows: list[dict]) -> pd.DataFrame:
    """Raw OHLC -> split-only OHLC; retain the exchange's calendar date."""
    records = []
    factor = 1.0
    for row in sorted(rows, key=lambda r: r["date"], reverse=True):
        ratio = float(row["splitFactor"]) if row.get("splitFactor") is not None else 1.0
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("Invalid Tiingo split factor")
        close = float(row.get("close") or 0)
        if math.isfinite(close) and close > 0:
            record = {"date": str(row["date"])[:10], "Close": close / factor}
            for field in ("open", "high", "low"):
                value = row.get(field)
                record[field.title()] = (
                    float(value) / factor if value is not None else float("nan")
                )
            volume = row.get("volume")
            record["Volume"] = float(volume) * factor if volume is not None else 0
            record["price"] = record["Close"]
            record["open_price"] = record["Open"]
            records.append(record)
        # An action still applies when its own day's close is missing.
        factor *= ratio
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Invalid Tiingo cumulative split factor")
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records).set_index("date").sort_index()
    frame.index = pd.to_datetime(frame.index)
    if frame.index.has_duplicates:
        raise ValueError("Duplicate Tiingo session dates")
    frame.attrs["market_data_source"] = "tiingo"
    return frame


class TiingoDailyFallback:
    """Serialize fallback calls, cache successes/misses, and back off on outages.

    One instance per backend process. A runtime key change clears both cached
    responses and cooldowns without exposing the key in cache keys or logs.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._key = None
        self._cache = OrderedDict()
        self._retry_at = 0.0

    def fetch(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        if not config.TIINGO_API_KEY:
            return {}
        result = {}
        with self._lock:
            if self._key != config.TIINGO_API_KEY:
                self._key = config.TIINGO_API_KEY
                self._cache.clear()
                self._retry_at = 0.0
            provider = TiingoProvider(api_key=self._key, timeout=8)
            for symbol in dict.fromkeys(symbols):
                if not eligible_symbol(symbol):
                    continue
                now = time.monotonic()
                cached = self._cache.get(symbol)
                if cached and cached[0] > now:
                    frame = cached[1]
                    self._cache.move_to_end(symbol)
                else:
                    if now < self._retry_at:
                        continue
                    try:
                        frame = price_frame(
                            provider.fetch_prices(
                                symbol, start=date(1900, 1, 1), end=get_est_today()
                            )
                        )
                    except TiingoError as exc:
                        # Do not log response bodies: credentials may be echoed
                        # by a provider or proxy. Status alone diagnoses access.
                        logger.info(
                            "Tiingo daily fallback unavailable for %s (HTTP %s)",
                            symbol,
                            exc.status,
                        )
                        frame = pd.DataFrame()
                        if exc.status != 404:
                            self._retry_at = time.monotonic() + 900
                    except (ValueError, TypeError, KeyError, OverflowError):
                        logger.warning(
                            "Tiingo daily fallback rejected malformed data for %s",
                            symbol,
                        )
                        frame = pd.DataFrame()
                    self._cache[symbol] = (time.monotonic() + 900, frame)
                    self._cache.move_to_end(symbol)
                    while len(self._cache) > 128:
                        self._cache.popitem(last=False)
                if not frame.empty:
                    window = frame.loc[
                        (frame.index.date >= start) & (frame.index.date <= end)
                    ].copy()
                    if not window.empty:
                        result[symbol] = window
        return result


daily_fallback = TiingoDailyFallback()
