#!/bin/bash

# Ensure we are running from the script's directory
cd "$(dirname "$0")"

# Clean up existing processes to avoid port conflicts
kill $(lsof -ti:8001) 2>/dev/null 

# Run the Electron Desktop App
echo "Starting Investa Desktop App (Electron)..."
cd desktop-electron
npm start
