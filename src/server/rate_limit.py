"""In-memory sliding-window rate limiting for the auth endpoints.

Single-process by design — Investa runs one uvicorn worker per deployment
(web on :8000, desktop on :8001). If the backend ever moves to multiple
workers, this needs a shared store (e.g. SQLite or Redis).
"""

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional

from fastapi import HTTPException, Request


class SlidingWindowLimiter:
    """Tracks event timestamps per key and rejects keys that exceed the limit."""

    def __init__(self, max_events: int, window_seconds: float):
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, events: Deque[float], now: float) -> None:
        cutoff = now - self.window_seconds
        while events and events[0] <= cutoff:
            events.popleft()

    def retry_after(self, key: str) -> Optional[float]:
        """Seconds until the key is allowed again, or None if not limited."""
        now = time.monotonic()
        with self._lock:
            events = self._events.get(key)
            if not events:
                return None
            self._prune(events, now)
            if len(events) < self.max_events:
                return None
            return events[0] + self.window_seconds - now

    def record(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            self._prune(events, now)
            events.append(now)

    def reset(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)


# Failed logins / password changes: 5 wrong passwords per identity per 15 min.
# Keyed per username so a brute force can't dodge the limit by rotating IPs.
failed_auth_limiter = SlidingWindowLimiter(max_events=5, window_seconds=15 * 60)

# Login attempts per IP (right or wrong): generous, catches username spraying.
login_ip_limiter = SlidingWindowLimiter(max_events=30, window_seconds=15 * 60)

# Account creation: 5 per IP per hour.
register_ip_limiter = SlidingWindowLimiter(max_events=5, window_seconds=60 * 60)


def get_client_ip(request: Request) -> str:
    """Client IP, honoring X-Forwarded-For from the Tailscale Serve / reverse proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_limit(limiter: SlidingWindowLimiter, key: str, what: str) -> None:
    """Raise 429 with Retry-After if the key is currently rate limited."""
    retry_after = limiter.retry_after(key)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Too many {what}. Try again in {max(1, int(retry_after))} seconds.",
            headers={"Retry-After": str(max(1, int(retry_after)))},
        )
