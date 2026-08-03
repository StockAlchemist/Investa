#!/bin/bash

# Starts both the Backend Server and the Web App concurrently.
# Automatically cleans up existing processes on ports 8000 and 3000.
#
# By default the frontend is served as an optimized production build
# (rebuilt automatically when sources change). Pass --dev for the
# hot-reloading dev server when actively working on the frontend.

# Ensure we are running from the script's directory
cd "$(dirname "$0")"

DEV_MODE=0
if [ "$1" = "--dev" ]; then
    DEV_MODE=1
fi

# Function to kill all child processes on exit
cleanup() {
    # Disarm first: with EXIT trapped as well as the signals, a SIGTERM would
    # otherwise run this twice — once for the signal, once for the exit it calls.
    trap - SIGINT SIGTERM EXIT
    echo "Shutting down Investa..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup.
#
# SIGTERM and EXIT are in here too, and they are the ones that matter for the
# ranking worker: it holds no port, so the port sweep below cannot reap it, and
# on SIGINT alone it survived every exit that was not Ctrl+C. Four orphaned
# workers had accumulated that way, the oldest three days old, each still
# ranking once a day into the same store.
trap cleanup SIGINT SIGTERM EXIT

# --- KILL EXISTING PROCESSES ---
echo "Checking for existing Investa processes..."

# Kill process on port 8000 (Backend)
if lsof -ti:8000 >/dev/null; then
    echo "Killing existing backend on port 8000..."
    kill -9 $(lsof -ti:8000)
fi

# Kill process on port 3000 (Frontend)
if lsof -ti:3000 >/dev/null; then
    echo "Killing existing frontend on port 3000..."
    kill -9 $(lsof -ti:3000)
fi

# Kill process on port 3001 (Frontend alternative) just in case
if lsof -ti:3001 >/dev/null; then
    echo "Killing existing frontend on port 3001..."
    kill -9 $(lsof -ti:3001)
fi

# The ranking worker listens on nothing, so it cannot be found by port. Left
# alone it accumulated one orphan per start — each a daily loop writing a full
# universe snapshot into the same store, which is what took that store to 51 MB
# in a week. Matched on its command line instead.
if pgrep -f "buffett_rank_worker.py" >/dev/null; then
    echo "Killing existing ranking worker(s)..."
    pkill -f "buffett_rank_worker.py"
fi
# -----------------------------

# --- BACKEND STARTUP LOGIC ---
start_backend() {
    # Function to get Tailscale IP
    get_tailscale_ip() {
        if command -v tailscale &> /dev/null; then
            tailscale ip -4
        else
            echo "Tailscale not found."
        fi
    }

    TS_IP=$(get_tailscale_ip)
    LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | grep -v "100\." | awk '{print $2}' | head -n 1)

    echo "=================================================="
    if [ -n "$TS_IP" ] && [ "$TS_IP" != "Tailscale not found." ]; then
        echo "Tailscale IP: $TS_IP"
        echo "Web App: http://$TS_IP:3000"
        echo "API:     http://$TS_IP:8000/api/"
        
        # Check for Tailscale Serve (HTTPS)
        if command -v tailscale &> /dev/null; then
            TS_SERVE_STATUS=$(tailscale serve status 2>/dev/null)
            if [[ "$TS_SERVE_STATUS" == *"https://"* ]]; then
                 HTTPS_URL=$(echo "$TS_SERVE_STATUS" | grep -o 'https://[^ ]*' | head -n 1)
                 echo "HTTPS:   $HTTPS_URL"
            fi
        fi
    fi
    
    if [ -n "$LOCAL_IP" ]; then
        echo "Local IP:     $LOCAL_IP"
        echo "Web App: http://$LOCAL_IP:3000"
        echo "API:     http://$LOCAL_IP:8000/api/"
    fi

    if [ -z "$TS_IP" ] && [ -z "$LOCAL_IP" ]; then
        echo "IP addresses not found. Server will still run on 0.0.0.0"
    fi
    echo "=================================================="

    # Ensure we are in the script's directory or set PYTHONPATH correctly
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src

    echo "Starting Investa Server..."
    python3 src/server/main.py
}
# -----------------------------

# Start Backend in background
start_backend &
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 2

# Start the Buffett ranking worker (daily loop) in the background.
#
# Without it the Strategies and Rankings tabs silently freeze at whichever run
# was last kicked off by hand: every endpoint keeps serving the last good
# snapshot and looks perfectly healthy. A run costs about a minute, and the
# worker backs off on its own after a failure, so a daily loop is close to
# free. Set INVESTA_SKIP_RANKING=1 to leave it out (metered connection, CI).
if [ "$INVESTA_SKIP_RANKING" = "1" ]; then
    echo "Skipping ranking worker (INVESTA_SKIP_RANKING=1)"
    RANKING_PID="(skipped)"
else
    mkdir -p data/logs
    echo "Starting Buffett ranking worker (daily loop)..."
    python3 src/buffett_rank_worker.py --loop >> data/logs/buffett_rank_worker.log 2>&1 &
    RANKING_PID=$!
fi

# Start Frontend in background
cd web_app
if [ "$DEV_MODE" = "1" ]; then
    echo "Starting Web App (dev mode)..."
    # Clean up Next.js lock file if it exists
    rm -rf .next/dev/lock
    npm run dev -- -H 0.0.0.0 &
else
    # Production mode. Rebuild when sources changed since the last web build.
    BUILD_MARKER=".next/.investa-web-build"
    if [ ! -f "$BUILD_MARKER" ] || [ -n "$(find app components context lib src public package.json next.config.ts tailwind.config.ts postcss.config.mjs tsconfig.json -newer "$BUILD_MARKER" -print -quit 2>/dev/null)" ]; then
        echo "Building Web App (production)..."
        if npm run build; then
            touch "$BUILD_MARKER"
        elif [ -f "$BUILD_MARKER" ]; then
            echo "WARNING: Frontend build failed — serving the previous production build."
        else
            echo "ERROR: Frontend build failed and no previous build exists."
            echo "Fix the build, or run './start_investa.sh --dev' to use the dev server."
            kill $BACKEND_PID 2>/dev/null
            exit 1
        fi
    fi
    echo "Starting Web App (production)..."
    npm run start -- -H 0.0.0.0 &
fi
FRONTEND_PID=$!

echo "Investa is running."
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Ranking worker PID: $RANKING_PID  (log: data/logs/buffett_rank_worker.log)"
echo "Press Ctrl+C to stop both."

# Wait for processes to finish
wait
