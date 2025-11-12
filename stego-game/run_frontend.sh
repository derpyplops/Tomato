#!/bin/bash
# Start the React frontend development server

echo "Starting React frontend"
echo "Frontend will be accessible at http://localhost:5173"
echo ""

cd "$(dirname "$0")"
pnpm run dev
