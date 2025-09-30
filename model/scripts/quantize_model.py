#!/usr/bin/env python3
"""
Quantize ONNX model for optimal WASM deployment.

Reduces model size from ~327MB to 20-40MB while maintaining accuracy.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np


def quantize_onnx_model(input_path: str, output_path: str, target_size_mb: int = 25):
    """Quantize ONNX model to reduce size for edge deployment."""
    
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        import onnxruntime as ort
    except ImportError:
        print("Error: Missing dependencies. Install with:")
        print("  uv add onnx onnxruntime")
        sys.exit(1)
    
    print(f"Quantizing model: {input_path}")
    
    # Check original model size
    original_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Original model size: {original_size_mb:.2f} MB")
    
    if original_size_mb <= target_size_mb:
        print(f"Model already smaller than target ({target_size_mb}MB), copying as-is...")
        import shutil
        shutil.copy2(input_path, output_path)
        return
    
    # Dynamic quantization (most compatible with WASM)
    print("Applying dynamic quantization...")
    
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8  # 8-bit unsigned integers
    )
    
    # Check quantized model size
    quantized_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    reduction_pct = ((original_size_mb - quantized_size_mb) / original_size_mb) * 100
    
    print(f"Quantized model size: {quantized_size_mb:.2f} MB")
    print(f"Size reduction: {reduction_pct:.1f}%")
    
    # Verify model still works
    verify_quantized_model(output_path)
    
    if quantized_size_mb <= target_size_mb:
        print(f"✅ Target size achieved ({quantized_size_mb:.1f}MB ≤ {target_size_mb}MB)")
    else:
        print(f"⚠️  Still above target ({quantized_size_mb:.1f}MB > {target_size_mb}MB)")
        print("   Consider aggressive quantization or model pruning")


def verify_quantized_model(model_path: str):
    """Verify the quantized model can be loaded and run."""
    
    try:
        import onnxruntime as ort
        
        print("Verifying quantized model...")
        
        # Create inference session
        session = ort.InferenceSession(model_path)
        
        # Get input shapes
        input_shapes = {inp.name: inp.shape for inp in session.get_inputs()}
        print(f"Model inputs: {input_shapes}")
        
        # Create dummy data
        dummy_inputs = {}
        for inp in session.get_inputs():
            shape = [1 if dim is None or isinstance(dim, str) else dim for dim in inp.shape]
            if inp.name == 'mel_spectrogram':
                dummy_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
            elif inp.name == 'dataset_id':
                dummy_inputs[inp.name] = np.zeros(shape, dtype=np.int64)
            else:
                dummy_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, dummy_inputs)
        
        print(f"✅ Quantized model verification passed!")
        print(f"   Output shapes: {[out.shape for out in outputs]}")
        
    except Exception as e:
        print(f"❌ Quantized model verification failed: {e}")
        sys.exit(1)


def create_wasm_optimized_model(input_path: str, output_path: str):
    """Create WASM-optimized version with additional optimizations."""
    
    try:
        import onnx
        from onnx import optimizer
        
        print("Creating WASM-optimized model...")
        
        # Load model
        model = onnx.load(input_path)
        
        # Apply WASM-specific optimizations
        passes = [
            'eliminate_deadend',
            'eliminate_identity',
            'eliminate_nop_dropout',
            'eliminate_nop_monotone_argmax',
            'eliminate_nop_pad',
            'eliminate_nop_transpose',
            'eliminate_unused_initializer',
            'extract_constant_to_initializer',
            'fuse_add_bias_into_conv',
            'fuse_bn_into_conv',
            'fuse_consecutive_concats',
            'fuse_consecutive_log_softmax',
            'fuse_consecutive_reduce_unsqueeze',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
            'fuse_matmul_add_bias_into_gemm',
            'fuse_pad_into_conv',
            'fuse_transpose_into_gemm',
            'lift_lexical_references',
        ]
        
        # Apply optimizations
        optimized_model = optimizer.optimize(model, passes)
        
        # Save optimized model
        onnx.save(optimized_model, output_path)
        
        # Compare sizes
        original_size = os.path.getsize(input_path) / (1024 * 1024)
        optimized_size = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"WASM optimization complete:")
        print(f"  Before: {original_size:.2f} MB")
        print(f"  After: {optimized_size:.2f} MB")
        
        if optimized_size < original_size:
            print(f"  Additional reduction: {((original_size - optimized_size) / original_size * 100):.1f}%")
        
    except ImportError:
        print("Warning: ONNX optimizer not available, skipping WASM optimization")
        import shutil
        shutil.copy2(input_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model for WASM deployment")
    parser.add_argument("input_model", help="Input ONNX model path")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=25,
        help="Target model size in MB (default: 25)"
    )
    parser.add_argument(
        "--wasm-optimize",
        action="store_true",
        help="Apply additional WASM-specific optimizations"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input_model)
    if not input_path.exists():
        print(f"Error: Input model not found: {input_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate output paths
    base_name = input_path.stem
    quantized_path = output_dir / f"{base_name}_quantized.onnx"
    
    try:
        # Quantize model
        quantize_onnx_model(str(input_path), str(quantized_path), args.target_size)
        
        # Apply WASM optimizations if requested
        if args.wasm_optimize:
            wasm_path = output_dir / f"{base_name}_wasm.onnx"
            create_wasm_optimized_model(str(quantized_path), str(wasm_path))
            final_path = wasm_path
        else:
            final_path = quantized_path
        
        final_size_mb = os.path.getsize(final_path) / (1024 * 1024)
        
        print(f"\n🎉 Quantization complete!")
        print(f"📁 Output: {final_path}")
        print(f"📏 Final size: {final_size_mb:.2f} MB")
        print(f"\n🚀 Ready for Rust WASM integration!")
        
        # Show next steps
        print(f"\nNext steps:")
        print(f"1. Copy to server: cp {final_path} ../server/models/")
        print(f"2. Update server/Cargo.toml with ort dependency")
        print(f"3. Integrate model into processing.rs")
        
    except Exception as e:
        print(f"Error during quantization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()