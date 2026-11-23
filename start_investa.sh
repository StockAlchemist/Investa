#!/bin/bash

# Starts both the Backend Server and the Web App concurrently.
# Automatically cleans up existing Investa processes on ports 8800 and 3000.
#
# The backend listens on 8800 (INVESTA_PORT overrides). Not 8000: that is
# Tuva's daemon's port on this machine, with fallbacks across 8000–8050.
#
# By default the frontend is served as an optimized production build
# (rebuilt automatically when sources change). Pass --dev for the
# hot-reloading dev server when actively working on the frontend.

# Ensure we are running from the script's directory
cd "$(dirname "$0")"

DEV_MODE=0

# Read by the backend (src/server/main.py) and, through API_URL, by the web
# app's /api rewrite, which Next bakes into the production build.
export INVESTA_PORT="${INVESTA_PORT:-8800}"
export API_URL="http://127.0.0.1:$INVESTA_PORT"
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

# PIDs listening on TCP port $1.
#
# Not lsof: it walks every mounted filesystem before answering, so a single
# dead network mount stalls it in the kernel — an sshfs share whose host had
# dropped off the network cost ~100 s per call (-b and -nP do not help), and
# the three checks below held startup for five minutes before it printed a
# line. netstat reads the socket table and never touches a filesystem.
# The process column is "name:pid" and the name may contain spaces, hence the
# scan for the first field ending in ":<digits>" rather than a fixed column.
pids_on_port() {
    netstat -anv -p tcp 2>/dev/null | awk -v port="$1" '
        $6 == "LISTEN" && $4 ~ ("\\." port "$") {
            for (i = 7; i <= NF; i++)
                if ($i ~ /:[0-9]+$/) { sub(/.*:/, "", $i); print $i; break }
        }' | sort -u
}

# Is $1 one of ours? Matched on the command line. Only ours are killed: when
# the backend still used 8000 this sweep killed Tuva's daemon by port, and its
# supervisor respawned it onto 127.0.0.1:8000, where it shadowed the backend
# for the web app.
is_investa_pid() {
    ps -o command= -p "$1" 2>/dev/null |
        grep -Eq 'src/server/main\.py|server\.main:app|next-server|start-standalone|next (dev|start)'
}

# A backend started before the move to 8800 may still be holding 8000. Reap
# it if it is ours; anything else on 8000 is none of our business.
for pid in $(pids_on_port 8000); do
    if is_investa_pid "$pid"; then
        echo "Killing old backend on port 8000 (PID $pid)..."
        kill -9 "$pid"
    fi
done

# Backend, frontend (3000) and its fallback port (3001).
for port_role in "$INVESTA_PORT:backend" 3000:frontend 3001:frontend; do
    port=${port_role%%:*}
    for pid in $(pids_on_port "$port"); do
        if is_investa_pid "$pid"; then
            echo "Killing existing ${port_role#*:} on port $port (PID $pid)..."
            kill -9 "$pid"
        else
            echo "ERROR: port $port is held by another program, not Investa:"
            echo "  PID $pid: $(ps -o command= -p "$pid" 2>/dev/null | cut -c1-160)"
            echo "Quit it (or move it to another port) and run this script again."
            exit 1
        fi
    done
done

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
        echo "API:     http://$TS_IP:$INVESTA_PORT/api/"
        
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
        echo "API:     http://$LOCAL_IP:$INVESTA_PORT/api/"
    fi

    if [ -z "$TS_IP" ] && [ -z "$LOCAL_IP" ]; then
        echo "IP addresses not found. Server will still run on 0.0.0.0"
    fi
    echo "=================================================="

    # Ensure we are in the script's directory or set PYTHONPATH correctly
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src

    # The calculation core is Rust (rust/investa_core); rebuild it when its
    # sources have changed since the installed build.
    if ! python3 scripts/ensure_rust_core.py; then
        echo "ERROR: could not build the Rust calculation core; backend not started."
        return 1
    fi

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
    # The marker records the API_URL the build was made with: the /api rewrite
    # is fixed at build time, so a new backend port needs a new build.
    BUILD_MARKER=".next/.investa-web-build"
    if [ ! -f "$BUILD_MARKER" ] || [ "$(cat "$BUILD_MARKER")" != "$API_URL" ] || [ -n "$(find app components context lib src public package.json next.config.ts tailwind.config.ts postcss.config.mjs tsconfig.json -newer "$BUILD_MARKER" -print -quit 2>/dev/null)" ]; then
        echo "Building Web App (production)..."
        if npm run build; then
            echo "$API_URL" > "$BUILD_MARKER"
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
