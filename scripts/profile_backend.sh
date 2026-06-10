#!/bin/bash
#
# profile_backend.sh — sample-profile the Investa FastAPI backend with py-spy.
#
# py-spy reads another process's memory, which on macOS requires root, so this
# script calls sudo for you (you'll be prompted for your password once).
#
# Usage:
#   scripts/profile_backend.sh record [seconds]   # flamegraph SVG (default 30s)
#   scripts/profile_backend.sh top                # live top-style view
#   scripts/profile_backend.sh dump               # one-shot stack dump of all threads
#
# Typical session:
#   1. Start the backend:   cd src && uvicorn server.main:app --port 8000
#   2. In another terminal: scripts/profile_backend.sh record 30
#   3. While it records, exercise the app (load the dashboard, run the screener,
#      open a stock detail — whatever feels slow).
#   4. Open the resulting profiles/*.svg in a browser. Wide bars = where time goes.
#
# Reading the flamegraph: width = time spent. Look for wide frames in your own
# code (portfolio_logic, market_data, db_utils). Frames in socket/recv/select =
# network/IO wait (yfinance, requests); frames in sqlite3 = disk/DB. If those
# dominate, a faster language won't help — fix caching/queries/async instead.

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
PROFILE_DIR="profiles"
mkdir -p "$PROFILE_DIR"

PORT="${INVESTA_BACKEND_PORT:-8000}"

# Find the uvicorn/backend PID listening on $PORT.
PID="$(lsof -ti:"$PORT" -sTCP:LISTEN 2>/dev/null | head -n1 || true)"
if [ -z "${PID:-}" ]; then
  # Fall back to matching the process command line.
  PID="$(pgrep -f 'server.main:app|uvicorn' | head -n1 || true)"
fi

if [ -z "${PID:-}" ]; then
  echo "ERROR: No backend process found on port $PORT (and no uvicorn process)." >&2
  echo "Start it first:  cd src && uvicorn server.main:app --port $PORT" >&2
  exit 1
fi

PY_SPY="$(command -v py-spy)"
MODE="${1:-record}"
echo "Profiling backend PID $PID on port $PORT (mode: $MODE)"

case "$MODE" in
  record)
    SECS="${2:-30}"
    TS="$(date +%Y%m%d_%H%M%S)"
    OUT="$PROFILE_DIR/flame_${TS}.svg"
    echo "Recording for ${SECS}s -> $OUT"
    echo ">> Exercise the app now (load slow pages / endpoints)..."
    # --idle includes threads blocked on IO so you can SEE the IO wait;
    # --subprocesses catches worker threads/processes the backend spawns.
    sudo "$PY_SPY" record \
      --pid "$PID" \
      --duration "$SECS" \
      --rate 100 \
      --idle \
      --subprocesses \
      --output "$OUT"
    echo "Done. Open in a browser:  open $OUT"
    ;;
  top)
    sudo "$PY_SPY" top --pid "$PID" --subprocesses
    ;;
  dump)
    sudo "$PY_SPY" dump --pid "$PID"
    ;;
  *)
    echo "Unknown mode '$MODE'. Use: record [seconds] | top | dump" >&2
    exit 1
    ;;
esac
