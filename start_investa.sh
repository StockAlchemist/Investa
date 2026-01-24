#!/bin/bash

# Starts both the Backend Server and the Web App concurrently.
# Automatically cleans up existing processes on ports 8000 and 3000.

# Ensure we are running from the script's directory
cd "$(dirname "$0")"

# Function to kill all child processes on exit
cleanup() {
    echo "Shutting down Investa..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

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

# Start Frontend in background
echo "Starting Web App..."
cd web_app
# Clean up Next.js lock file if it exists
rm -rf .next/dev/lock
npm run dev -- -H 0.0.0.0 &
FRONTEND_PID=$!

echo "Investa is running."
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Press Ctrl+C to stop both."

# Wait for processes to finish
wait
