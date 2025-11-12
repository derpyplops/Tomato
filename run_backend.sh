#!/bin/bash
# Start the Flask backend server

echo "Starting Flask backend on http://0.0.0.0:5000"
echo "Server will be accessible from remote machines"
echo ""

cd "$(dirname "$0")"
uv run python backend/app.py
