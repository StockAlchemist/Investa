#!/bin/bash

# Ensure we are running from the script's directory
cd "$(dirname "$0")"

# Clean up existing processes if necessary (optional for specific desktop-only launch, 
# but good practice if it relies on resources)
# kill $(lsof -ti:8000) 2>/dev/null 

echo "Starting Investa Desktop App..."

# Set PYTHONPATH to include src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run the GUI application
python3 src/main_gui.py
