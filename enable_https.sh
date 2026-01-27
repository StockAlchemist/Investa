#!/bin/bash

# enable_https.sh
# Configures Tailscale Serve to proxy traffic to Investa

echo "Configuring Tailscale Serve for Investa..."

# Ensure we are logged in or have tailscale
if ! command -v tailscale &> /dev/null; then
    echo "Error: Tailscale CLI not found."
    exit 1
fi

# Reset any previous serve config to be clean
tailscale serve reset

# Map / to Frontend (3000)
# Map /api to Backend (8000)
# Using --bg to run in background (persist)

echo "Mapping root (/) to localhost:3000..."
tailscale serve --bg http://localhost:3000

echo "Mapping /api to localhost:8000..."
tailscale serve --bg --set-path /api http://localhost:8000

echo "=================================================="
echo "Tailscale Serve Configured!"
tailscale serve status
echo "=================================================="
echo "You can now run ./start_investa.sh"
