#!/usr/bin/env python3
"""
Test script for the Real Model Service API
"""

import requests
import base64
import numpy as np
import json
import time

def test_model_api(host="localhost", port=8002):
    """Test the model API endpoints"""
    
    base_url = f"http://{host}:{port}"
    
    print(f"🧪 Testing CrescendAI Real Model Service at {base_url}")
    print("=" * 60)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Health check passed: {health_data}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            root_data = response.json()
            print(f"   ✅ Root endpoint passed: {root_data['service']}")
        else:
            print(f"   ❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Root endpoint error: {e}")
    
    # Test 3: Models endpoint
    print("\n3. Testing models endpoint...")
    try:
        response = requests.get(f"{base_url}/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            print(f"   ✅ Models endpoint passed: {len(models_data['models'])} model(s)")
            for model in models_data['models']:
                print(f"      - {model['name']} ({model['status']})")
        else:
            print(f"   ❌ Models endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Models endpoint error: {e}")
    
    # Test 4: Analysis endpoint
    print("\n4. Testing analysis endpoint...")
    try:
        # Create test spectrogram data (128x128 float32)
        test_spectrogram = np.random.rand(128, 128).astype(np.float32)
        
        # Convert to bytes and encode as base64
        spectrogram_bytes = test_spectrogram.tobytes()
        encoded_spectrogram = base64.b64encode(spectrogram_bytes).decode('utf-8')
        
        # Prepare request
        request_data = {
            "file_id": "test-file-123",
            "spectrogram_data": encoded_spectrogram,
            "metadata": {
                "test": True,
                "timestamp": time.time()
            }
        }
        
        print(f"   Sending {len(encoded_spectrogram)} chars of base64 spectrogram data...")
        
        response = requests.post(
            f"{base_url}/analyze", 
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            analysis_data = response.json()
            print(f"   ✅ Analysis completed in {analysis_data['processing_time']}s")
            print(f"   📊 Analysis dimensions:")
            
            # Show key scores
            analysis = analysis_data['analysis']
            key_dimensions = ['rhythm', 'technique', 'expression', 'overall_performance']
            for dim in key_dimensions:
                if dim in analysis:
                    print(f"      - {dim}: {analysis[dim]}")
            
            print(f"   💡 Insights ({len(analysis_data['insights'])}):")
            for i, insight in enumerate(analysis_data['insights'][:3], 1):
                print(f"      {i}. {insight}")
                
        else:
            print(f"   ❌ Analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Analysis error: {e}")
    
    print("\n" + "=" * 60)
    print("🎹 Test completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CrescendAI Model API")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8002, help="API port")
    
    args = parser.parse_args()
    
    test_model_api(args.host, args.port)