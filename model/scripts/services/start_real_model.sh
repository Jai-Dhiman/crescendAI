#!/bin/bash

# Start the real model service for CrescendAI

echo "🎹 Starting CrescendAI Real Model Service..."

# Check if model file exists
MODEL_FILE="results/final_finetuned_model.pkl"
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Error: Model file not found at $MODEL_FILE"
    echo "Please ensure the trained model is available"
    exit 1
fi

echo "✅ Model file found: $MODEL_FILE"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔄 Activating virtual environment..."
    source .venv/bin/activate
fi

# Install required dependencies if needed
echo "📦 Installing dependencies..."
uv sync

# Start the service
echo "🚀 Starting real model service on port 8002..."
echo "📚 API docs will be available at http://localhost:8002/docs"

python3 real_model_service.py --host 0.0.0.0 --port 8002 --reload