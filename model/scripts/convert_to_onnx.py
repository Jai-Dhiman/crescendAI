#!/usr/bin/env python3
"""
Convert PyTorch model to ONNX format for edge deployment.
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json
import logging
import sys
sys.path.append('scripts')
from convert_jax_to_pytorch import PyTorchAST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_pytorch_to_onnx(pytorch_model_path: str, output_dir: str):
    """Convert PyTorch model to ONNX format."""
    
    # Load PyTorch model
    logger.info(f"Loading PyTorch model from {pytorch_model_path}")
    model = torch.load(pytorch_model_path, map_location='cpu')
    model.eval()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Define input shape (batch_size, channels, height, width)
    input_shape = (1, 1, 128, 128)
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    onnx_path = output_path / "crescend_evaluator.onnx"
    
    logger.info("Converting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['mel_spectrogram'],
        output_names=['percepiano_scores'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch_size'},
            'percepiano_scores': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    logger.info("Verifying ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    
    # Test ONNX Runtime
    logger.info("Testing ONNX Runtime inference...")
    ort_session = ort.InferenceSession(str(onnx_path))
    
    # Test inference
    test_input = np.random.randn(*input_shape).astype(np.float32)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    logger.info(f"ONNX output shape: {ort_outputs[0].shape}")
    logger.info(f"ONNX output range: [{ort_outputs[0].min():.3f}, {ort_outputs[0].max():.3f}]")
    
    # Get model size
    model_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX model size: {model_size_mb:.2f} MB")
    
    # Save model metadata
    metadata = {
        "model_type": "percepiano_ast_onnx",
        "input_shape": list(input_shape),
        "output_shape": [1, 19],
        "dimension_names": [
            "timing_stable_unstable",
            "articulation_short_long", 
            "articulation_soft_hard",
            "pedal_sparse_saturated",
            "pedal_clean_blurred",
            "timbre_even_colorful",
            "timbre_shallow_rich",
            "timbre_bright_dark", 
            "timbre_soft_loud",
            "dynamic_sophisticated_raw",
            "dynamic_range_little_large",
            "music_making_fast_slow",
            "music_making_flat_spacious",
            "music_making_disproportioned_balanced",
            "music_making_pure_dramatic",
            "emotion_mood_optimistic_dark",
            "emotion_mood_low_high_energy",
            "emotion_mood_honest_imaginative",
            "interpretation_unsatisfactory_convincing"
        ],
        "model_size_mb": model_size_mb,
        "opset_version": 14
    }
    
    with open(output_path / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ ONNX conversion complete! Model saved to {onnx_path}")
    return onnx_path, metadata


def quantize_onnx_model(onnx_model_path: str, output_dir: str):
    """Quantize ONNX model to reduce size for edge deployment."""
    
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        output_path = Path(output_dir)
        quantized_path = output_path / "crescend_evaluator_quantized.onnx"
        
        logger.info("Quantizing ONNX model...")
        quantize_dynamic(
            str(onnx_model_path),
            str(quantized_path),
            weight_type=QuantType.QUInt8
        )
        
        # Check size reduction
        original_size = Path(onnx_model_path).stat().st_size / (1024 * 1024)
        quantized_size = quantized_path.stat().st_size / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        
        logger.info(f"Original size: {original_size:.2f} MB")
        logger.info(f"Quantized size: {quantized_size:.2f} MB")
        logger.info(f"Size reduction: {reduction:.1f}%")
        
        # Test quantized model
        logger.info("Testing quantized model...")
        ort_session = ort.InferenceSession(str(quantized_path))
        test_input = np.random.randn(1, 1, 128, 128).astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: test_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        logger.info(f"Quantized output shape: {ort_outputs[0].shape}")
        logger.info(f"Quantized output range: [{ort_outputs[0].min():.3f}, {ort_outputs[0].max():.3f}]")
        
        return quantized_path, quantized_size
        
    except ImportError:
        logger.warning("ONNX quantization not available - using original model")
        return onnx_model_path, Path(onnx_model_path).stat().st_size / (1024 * 1024)


if __name__ == "__main__":
    pytorch_model_path = "models/pytorch_converted/pytorch_ast_complete.pth"
    output_dir = "models/onnx"
    
    # Convert to ONNX
    onnx_path, metadata = convert_pytorch_to_onnx(pytorch_model_path, output_dir)
    
    # Quantize for production
    quantized_path, size_mb = quantize_onnx_model(str(onnx_path), output_dir)
    
    print(f"\n✅ Model conversion pipeline complete!")
    print(f"📦 ONNX model: {onnx_path}")
    print(f"⚡ Quantized model: {quantized_path} ({size_mb:.2f} MB)")
    print(f"🎯 19 PercePiano dimensions ready for edge deployment")
    print(f"\nNext: Integrate into Rust WASM backend")
