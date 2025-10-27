#!/bin/bash
# Test script to verify Workers AI integration is working
# Usage: ./test_workers_ai.sh

set -e

echo "=== Workers AI Integration Test ==="
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found"
    echo "   Create .env from .env.example and add your Cloudflare credentials"
    exit 1
fi

# Load environment variables
source .env

# Check required variables
if [ -z "$CLOUDFLARE_ACCOUNT_ID" ]; then
    echo "❌ Error: CLOUDFLARE_ACCOUNT_ID not set in .env"
    exit 1
fi

if [ -z "$CLOUDFLARE_WORKERS_AI_API_TOKEN" ]; then
    echo "❌ Error: CLOUDFLARE_WORKERS_AI_API_TOKEN not set in .env"
    exit 1
fi

echo "✅ Environment variables configured"
echo "   Account ID: ${CLOUDFLARE_ACCOUNT_ID:0:8}..."
echo "   API Token: ${CLOUDFLARE_WORKERS_AI_API_TOKEN:0:8}..."
echo ""

# Test embedding API directly
echo "Testing Workers AI Embedding API..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
  "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/ai/run/@cf/baai/bge-base-en-v1.5" \
  -H "Authorization: Bearer $CLOUDFLARE_WORKERS_AI_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": ["Hello, world!", "This is a test embedding."]
  }')

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" -eq 200 ]; then
    echo "✅ Workers AI API is accessible"
    echo "   HTTP Status: $HTTP_CODE"

    # Parse response to check format
    SUCCESS=$(echo "$BODY" | jq -r '.success')
    if [ "$SUCCESS" = "true" ]; then
        echo "✅ Embedding generation successful"

        # Check embedding dimensions
        DIM=$(echo "$BODY" | jq -r '.result.data[0] | length')
        echo "   Embedding dimension: $DIM"

        if [ "$DIM" -eq 768 ]; then
            echo "✅ Correct embedding dimension (768)"
        else
            echo "⚠️  Warning: Unexpected embedding dimension (expected 768, got $DIM)"
        fi

        # Show first few values
        echo "   First 5 values: $(echo "$BODY" | jq -r '.result.data[0][0:5]')"
    else
        echo "❌ Embedding generation failed"
        echo "$BODY" | jq '.'
        exit 1
    fi
else
    echo "❌ Workers AI API request failed"
    echo "   HTTP Status: $HTTP_CODE"
    echo "   Response:"
    echo "$BODY" | jq '.'
    exit 1
fi

echo ""
echo "=== Test Summary ==="
echo "✅ Workers AI integration is properly configured!"
echo ""
echo "Next steps:"
echo "1. Start the API server: cargo run"
echo "2. Look for log: 'Initializing Cloudflare Workers AI client'"
echo "3. Create a knowledge base document and trigger processing"
echo "4. Verify real embeddings are stored in database"
echo ""
echo "See docs/WORKERS_AI_SETUP.md for detailed instructions"
