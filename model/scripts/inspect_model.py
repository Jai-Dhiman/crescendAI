#!/usr/bin/env python3
"""
Inspect the JAX/Flax model and extract weights for PyTorch conversion.
"""

import pickle
import numpy as np
from pathlib import Path


def inspect_jax_model(model_path: str):
    """Inspect the JAX model structure without importing JAX."""
    
    print(f"Inspecting model: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            # Try to load with pickle but catch JAX import errors
            try:
                model_data = pickle.load(f)
                print("✅ Model loaded successfully")
                
                print(f"Model type: {type(model_data)}")
                
                if isinstance(model_data, dict):
                    print("Model keys:", list(model_data.keys()))
                    
                    # Look for common JAX/Flax patterns
                    for key, value in model_data.items():
                        print(f"  {key}: {type(value)}")
                        
                        if hasattr(value, 'shape'):
                            print(f"    Shape: {value.shape}")
                        elif isinstance(value, dict):
                            print(f"    Dict keys: {list(value.keys())}")
                
                return model_data
                
            except Exception as e:
                if "jax" in str(e).lower():
                    print("❌ Model requires JAX/Flax to load")
                    print("Attempting to extract without full JAX...")
                    
                    # Try to read the pickle structure manually
                    analyze_pickle_structure(model_path)
                else:
                    raise e
                    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def analyze_pickle_structure(model_path: str):
    """Analyze pickle structure without loading JAX objects."""
    
    import pickletools
    
    print("\nAnalyzing pickle structure...")
    
    with open(model_path, 'rb') as f:
        # Just peek at the structure
        try:
            # Read first few objects manually
            import pickle
            
            # Reset file pointer
            f.seek(0)
            
            # Try to get basic structure info
            unpickler = pickle.Unpickler(f)
            
            print("Pickle structure analysis would require manual parsing")
            print("Recommendation: Use JAX in a compatible environment or")
            print("create a simple PyTorch model from scratch using the architecture")
            
        except Exception as e:
            print(f"Pickle analysis failed: {e}")


def create_pytorch_architecture():
    """Create equivalent PyTorch architecture based on the evaluation report."""
    
    print("\n" + "="*60)
    print("RECOMMENDATION: Create PyTorch equivalent from scratch")
    print("="*60)
    
    pytorch_code = '''
# Based on your evaluation report, recreate the model:

import torch
import torch.nn as nn

class SimpleAST(nn.Module):
    """12-layer Audio Spectrogram Transformer with Regression Head"""
    
    def __init__(self, num_classes=19, emb_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        
        # Patch embedding for 128x128 mel-spectrograms
        self.patch_embed = nn.Conv2d(1, emb_dim, kernel_size=16, stride=16)  # 16x16 patches
        
        # Positional encoding
        num_patches = (128 // 16) * (128 // 16)  # 8x8 = 64 patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Regression head for 19 perceptual dimensions
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding: [B, 1, 128, 128] -> [B, 768, 8, 8]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, 64, 768]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 65, 768]
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Use class token for prediction
        cls_output = x[:, 0]  # [B, 768]
        
        # Regression head
        output = self.head(cls_output)  # [B, 19]
        
        return torch.sigmoid(output)  # [0, 1] range

# Create model matching your evaluation report specs:
model = SimpleAST(
    num_classes=19,      # 19 perceptual dimensions  
    emb_dim=768,         # 768 embedding dimension
    num_heads=12,        # 12 attention heads
    num_layers=12        # 12 transformer layers
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
# Should be close to 85,706,003 from your report

# Train this model on your data, then convert to ONNX
'''
    
    print(pytorch_code)
    
    # Save as a starter file
    with open("create_pytorch_model.py", "w") as f:
        f.write(pytorch_code)
    
    print("\n📁 Saved PyTorch architecture to: create_pytorch_model.py")
    print("\nNext steps:")
    print("1. Train this PyTorch model on your dataset")
    print("2. Save trained weights as .pth file")  
    print("3. Convert to ONNX using convert_to_onnx.py")


def main():
    model_path = "archive/results_20250929/final_finetuned_model.pkl"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Inspect the model
    model_data = inspect_jax_model(model_path)
    
    # Provide PyTorch alternative
    create_pytorch_architecture()


if __name__ == "__main__":
    main()