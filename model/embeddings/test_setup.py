#!/usr/bin/env python3
"""Test script to verify embedding setup."""

print("Testing embedding setup...")

try:
    from sentence_transformers import SentenceTransformer
    import torch
    print("✅ Imports successful")
    
    # Load model
    print("Loading model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print(f"✅ Model loaded (dimensions: {model.get_sentence_embedding_dimension()})")
    
    # Test embedding
    text = "Piano practice improves technique and musicality."
    print("Computing test embedding...")
    embedding = model.encode([text])
    print(f"✅ Test embedding computed (shape: {embedding.shape})")
    
    # Check device capabilities
    if torch.backends.mps.is_available():
        print("✅ MPS (Apple Metal) available for acceleration")
    else:
        print("ℹ️  Using CPU inference")
    
    print("\n🎉 Setup test completed successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()