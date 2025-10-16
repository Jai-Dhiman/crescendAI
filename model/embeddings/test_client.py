#!/usr/bin/env python3
"""Test client for local embedding service."""

import requests
import json
import time

def test_embedding_service():
    base_url = "http://localhost:8001"
    
    print("🎹 CrescendAI Local Embedding Service - Test Client")
    print("=" * 50)
    
    # Test health check
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ Service healthy")
            print(f"   Backend: {health['backend']}")
            print(f"   Model loaded: {health['model_loaded']}")
            print(f"   Cache size: {health['cache_size']}")
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot reach service: {e}")
        print("   Make sure the server is running: python server.py")
        return False
    
    # Test single embedding
    print("\n🔍 Testing single embedding...")
    test_text = "Piano practice requires consistent technique and musical expression."
    
    try:
        response = requests.post(
            f"{base_url}/embed",
            json={"text": test_text},  # Use default model
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
    print("\nNext steps:")
    print("1. Update your Cloudflare Worker environment variables:")
    print("   LOCAL_EMBEDDING_URL=http://localhost:8001")
    print("   USE_LOCAL_EMBEDDINGS=true")
    print("2. Start saving money on embedding API costs! 💰")
    return True

if __name__ == "__main__":
    test_embedding_service()