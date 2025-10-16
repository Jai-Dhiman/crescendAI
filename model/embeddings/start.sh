#!/bin/bash
set -e

echo "🎹 CrescendAI Local Embedding Service - Quick Start"

# Check if we're in the right directory
if [ ! -f "server.py" ]; then
    echo "❌ Error: Please run this script from the model/embeddings/ directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Error: Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

echo "🚀 Starting embedding server..."

# Start the server in the background
source .venv/bin/activate
nohup python server.py > server.log 2>&1 &
SERVER_PID=$!

# Give the server a moment to start
sleep 3

echo "📡 Server started (PID: $SERVER_PID)"
echo "📋 Logs: tail -f server.log"

# Test the server
echo "🧪 Running tests..."
python test_client.py

echo ""
echo "ℹ️  Server is running in the background"
echo "   To stop: kill $SERVER_PID"
echo "   To view logs: tail -f server.log"
echo ""
echo "🎯 Your embedding service is ready!"
echo "   API URL: http://localhost:8001"
echo "   Swagger UI: http://localhost:8001/docs"