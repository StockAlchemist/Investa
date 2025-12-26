#!/bin/bash

# start_tailscale_server.sh
# This script starts the Investa backend server and prints the Tailscale IP address.

# Function to get Tailscale IP
get_tailscale_ip() {
    if command -v tailscale &> /dev/null; then
        tailscale ip -4
    else
        echo "Tailscale not found."
    fi
}

TS_IP=$(get_tailscale_ip)

echo "=================================================="
if [ -n "$TS_IP" ] && [ "$TS_IP" != "Tailscale not found." ]; then
    echo "Tailscale IP: $TS_IP"
    echo "You can access the API at: http://$TS_IP:8000/api/"
else
    echo "Tailscale IP not found. Server will still run on 0.0.0.0"
fi
echo "=================================================="

# Ensure we are in the script's directory or set PYTHONPATH correctly
# This assumes the script is in the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "Starting Investa Server..."
python3 src/server/main.py
