#!/bin/bash

# Startup script for Tomato steganography detection game
# Launches backend, frontend, and ngrok services

set -e

echo "======================================"
echo "Starting Tomato Stego Game Services"
echo "======================================"

# Configure ngrok if auth token is provided
if [ -n "$NGROK_AUTHTOKEN" ]; then
    echo "Configuring ngrok with provided auth token..."
    ngrok config add-authtoken "$NGROK_AUTHTOKEN"
else
    echo "Warning: NGROK_AUTHTOKEN not set. Ngrok may not work without authentication."
fi

# Start backend server
echo "Starting Flask backend on port 5000..."
cd /app
python3 backend/app.py > /var/log/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait for backend to initialize
sleep 5

# Start frontend server
echo "Starting Vite frontend on port 5173..."
cd /app/stego-game
pnpm run dev > /var/log/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

# Wait for frontend to initialize
sleep 3

# Start ngrok tunnels
echo "Starting ngrok tunnels..."
cd /app
if [ -f "ngrok.yml" ]; then
    ngrok start --all --config=ngrok.yml --log=stdout > /var/log/ngrok.log 2>&1 &
    NGROK_PID=$!
    echo "Ngrok started with PID: $NGROK_PID"

    # Wait for ngrok to initialize and extract URLs
    sleep 5
    echo ""
    echo "======================================"
    echo "Ngrok Public URLs:"
    echo "======================================"
    curl -s http://localhost:4040/api/tunnels | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for tunnel in data.get('tunnels', []):
        print(f\"{tunnel['name']}: {tunnel['public_url']}\")
except:
    print('Could not retrieve ngrok URLs. Check /var/log/ngrok.log')
" || echo "Ngrok API not ready yet. Check /var/log/ngrok.log for URLs."
    echo "======================================"
else
    echo "Warning: ngrok.yml not found. Skipping ngrok setup."
fi

echo ""
echo "All services started successfully!"
echo "Backend: http://localhost:5000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Logs available at:"
echo "  Backend:  /var/log/backend.log"
echo "  Frontend: /var/log/frontend.log"
echo "  Ngrok:    /var/log/ngrok.log"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    kill $NGROK_PID 2>/dev/null || true
    echo "All services stopped."
    exit 0
}

# Trap SIGINT and SIGTERM
trap cleanup SIGINT SIGTERM

# Keep the script running and tail logs
tail -f /var/log/backend.log /var/log/frontend.log /var/log/ngrok.log 2>/dev/null || sleep infinity
