#!/usr/bin/env python3
"""
Convert minimal PyTorch evaluator to ONNX for immediate WASM deployment.
"""

import torch
import torch.onnx
import numpy as np
import sys
from pathlib import Path


class MinimalEvaluator(torch.nn.Module):
    """Minimal CNN-based evaluator for immediate deployment."""
    
    def __init__(self, num_dims=19):
        super().__init__()
        
        # Simple CNN backbone
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        
        # Regression head
        self.head = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, num_dims),
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return torch.sigmoid(output)


def load_minimal_model(model_path: str) -> MinimalEvaluator:
    """Load the minimal trained model."""
    
    print(f"Loading minimal model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    model = MinimalEvaluator(num_dims=19)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def export_to_onnx(model: MinimalEvaluator, output_path: str):
    """Export minimal model to ONNX."""
    
    print("Converting to ONNX...")
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 128, 128)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['mel_spectrogram'],
        output_names=['performance_scores'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch_size'},
            'performance_scores': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"✅ ONNX export complete: {output_path}")
    
    # Verify the conversion
    verify_onnx_export(output_path, dummy_input, model)


def verify_onnx_export(onnx_path: str, dummy_input: torch.Tensor, original_model: MinimalEvaluator):
    """Verify ONNX model works correctly."""
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  onnxruntime not available, skipping verification")
        return
    
    print("Verifying ONNX conversion...")
    
    # Run PyTorch model
    with torch.no_grad():
        torch_output = original_model(dummy_input).numpy()
    
    # Run ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(
        None,
        {'mel_spectrogram': dummy_input.numpy()}
    )[0]
    
    # Compare outputs
    diff = np.abs(torch_output - onnx_output).max()
    print(f"Max difference: {diff:.6f}")
    
    if diff < 1e-5:
        print("✅ ONNX verification passed!")
    else:
        print("⚠️  Large difference detected")
    
    # Show model info
    import os
    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"📏 ONNX model size: {model_size_mb:.2f} MB")


def main():
    # Check if models exist
    models_dir = Path("models")
    
    # Try trained model first, then untrained
    trained_path = models_dir / "minimal_evaluator_trained.pth"
    untrained_path = models_dir / "minimal_evaluator.pth"
    
    if trained_path.exists():
        model_path = trained_path
        output_name = "crescend_minimal_trained.onnx"
        print("🎯 Using trained minimal model")
    elif untrained_path.exists():
        model_path = untrained_path
        output_name = "crescend_minimal_untrained.onnx"
        print("⚠️  Using untrained minimal model")
    else:
        print("❌ No minimal model found. Run create_minimal_model.py first")
        sys.exit(1)
    
    try:
        # Load model
        model = load_minimal_model(str(model_path))
        
        # Export to ONNX
        onnx_path = models_dir / output_name
        export_to_onnx(model, str(onnx_path))
        
        print(f"\n🎉 Conversion complete!")
        print(f"📁 ONNX model: {onnx_path}")
        print(f"📏 Size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
        
        print(f"\n🔄 Next steps:")
        print(f"1. Quantize: python scripts/quantize_model.py {onnx_path}")
        print(f"2. Copy to server: cp {onnx_path} ../server/models/")
        print(f"3. Update Rust server integration")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()