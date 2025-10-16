#!/bin/bash
set -e

echo "🎹 CrescendAI Local Embedding Setup"
echo "Setting up Google Gemma embedding model for local inference..."

# Check if we're in the right directory
if [ ! -f "server.py" ]; then
    echo "❌ Error: Please run this script from the model/embeddings/ directory"
    exit 1
fi

# Create virtual environment using uv
echo "📦 Creating Python environment with uv..."
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create and activate virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
uv pip install -r requirements.txt

# Check if MLX is available (Apple Silicon)
echo "🔍 Checking for Apple MLX support..."
if python -c "import mlx.core" 2>/dev/null; then
    echo "✅ MLX detected - will use Apple Silicon acceleration"
    BACKEND="mlx"
elif python -c "import torch; print('MPS available:', torch.backends.mps.is_available())" 2>/dev/null | grep "True"; then
    echo "✅ PyTorch MPS detected - will use Apple Metal acceleration"
    BACKEND="transformers"
elif python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep "True"; then
    echo "✅ PyTorch CUDA detected - will use GPU acceleration"
    BACKEND="transformers"
else
    echo "⚠️  CPU-only mode detected - consider installing MLX for faster inference on Apple Silicon"
    BACKEND="transformers"
fi

# Download Gemma model
echo "🚀 Testing model loading..."
python -c "
import torch
from transformers import AutoTokenizer, AutoModel

print('Downloading Gemma model...')
model_name = 'google/gemma-2-2b-it'
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    print(f'✅ Model loaded successfully (Parameters: ~2B)')
    print(f'✅ Model dimensions: {model.config.hidden_size}')
except Exception as e:
    print(f'❌ Error loading model: {e}')
    print('Note: You may need to authenticate with Hugging Face for Gemma models')
    print('Run: huggingface-cli login')
    exit(1)
"

# Create environment file
echo "📝 Creating environment configuration..."
cat > .env << EOF
# Local Embedding Service Configuration
LOCAL_EMBEDDING_URL=http://localhost:8001
LOCAL_EMBEDDING_TIMEOUT_MS=5000
LOCAL_EMBEDDING_MODEL=google/gemma-2-2b-it
USE_LOCAL_EMBEDDINGS=true

# Fallback providers (optional)
# CF_ACCOUNT_ID=your_cloudflare_account_id
# CF_API_TOKEN=your_cloudflare_token
# OPENAI_API_KEY=your_openai_key
EOF

# Create systemd service file (optional)
if command -v systemctl &> /dev/null; then
    echo "📝 Creating systemd service file..."
    cat > crescendai-embeddings.service << EOF
[Unit]
Description=CrescendAI Local Embedding Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/.venv/bin
ExecStart=$(pwd)/.venv/bin/python server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    echo "ℹ️  Systemd service created: crescendai-embeddings.service"
    echo "   To install: sudo cp crescendai-embeddings.service /etc/systemd/system/"
    echo "   To enable: sudo systemctl enable --now crescendai-embeddings"
fi

# Create test script
echo "📝 Creating test script..."
cat > test_embedding.py << 'EOF'
#!/usr/bin/env python3
"""Test script for local embedding service."""

import requests
import json
import time

def test_embedding_service():
    base_url = "http://localhost:8001"
    
    # Test health check
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Service healthy: {health}")
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot reach service: {e}")
        return False
    
    # Test single embedding
    print("\n🔍 Testing single embedding...")
    test_text = "Piano practice requires consistent technique and musical expression."
    
    try:
        response = requests.post(
            f"{base_url}/embed",
            json={"text": test_text},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Single embedding successful:")
            print(f"   Dimensions: {result['dimensions']}")
            print(f"   Processing time: {result['processing_time_ms']:.1f}ms")
            print(f"   Model: {result['model']}")
        else:
            print(f"❌ Single embedding failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Single embedding request failed: {e}")
        return False
    
    # Test batch embedding
    print("\n📦 Testing batch embedding...")
    test_texts = [
        "Chopin etudes focus on technical challenges",
        "Bach inventions develop polyphonic thinking", 
        "Scales and arpeggios build finger strength"
    ]
    
    try:
        response = requests.post(
            f"{base_url}/embed/batch",
            json={"texts": test_texts},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch embedding successful:")
            print(f"   Batch size: {result['batch_size']}")
            print(f"   Dimensions: {result['dimensions']}")
            print(f"   Processing time: {result['processing_time_ms']:.1f}ms")
            print(f"   Avg time per text: {result['processing_time_ms']/len(test_texts):.1f}ms")
        else:
            print(f"❌ Batch embedding failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch embedding request failed: {e}")
        return False
    
    # Test stats
    print("\n📊 Getting service stats...")
    try:
        response = requests.get(f"{base_url}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Service stats:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        else:
            print(f"⚠️  Stats not available: HTTP {response.status_code}")
    except:
        print("⚠️  Could not retrieve stats")
    
    print(f"\n🎉 All tests passed! Local embedding service is working correctly.")
    return True

if __name__ == "__main__":
    print("CrescendAI Local Embedding Service - Test Suite")
    print("=" * 50)
    test_embedding_service()
EOF

chmod +x test_embedding.py

# Final instructions
echo ""
echo "🎉 Setup complete! Next steps:"
echo ""
echo "1. Start the embedding service:"
echo "   python server.py"
echo ""
echo "2. In another terminal, test the service:"
echo "   python test_embedding.py"
echo ""
echo "3. Update your Cloudflare Worker environment variables:"
echo "   LOCAL_EMBEDDING_URL=http://localhost:8001"
echo "   USE_LOCAL_EMBEDDINGS=true"
echo "   LOCAL_EMBEDDING_MODEL=google/gemma-2-2b-it"
echo ""
echo "4. For production, consider:"
echo "   - Deploy to fly.io or Google Cloud Run with GPU"
echo "   - Use Cloudflare Tunnel for secure Worker→Local connection"
echo "   - Monitor costs vs. cloud embedding APIs"
echo ""
echo "Configuration saved to .env file"
echo "Backend detected: $BACKEND"