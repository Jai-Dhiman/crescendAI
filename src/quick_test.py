#!/usr/bin/env python3
"""
Quick Pipeline Test - Fast validation of AST+SSAST architecture
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_leaves

# Import models
from models.ast_transformer import AudioSpectrogramTransformer, create_train_state
from models.ssast_pretraining import SSASTPreTrainingModel, create_ssast_train_state


def quick_test():
    """Quick validation test"""
    print("🧪 Quick Pipeline Test")
    print("=" * 30)

    # Small synthetic data
    rng = jax.random.PRNGKey(42)
    # Split keys for all random operations
    rng_keys = jax.random.split(rng, 6)  # Need 6 keys total
    batch_size = 2  # Smaller batch for speed
    time_frames, freq_bins = 64, 64  # Smaller input for speed

    # Generate synthetic mel-spectrogram
    spectrograms = (
        jax.random.normal(rng_keys[0], (batch_size, time_frames, freq_bins)) * 20 - 40
    )

    print(f"✓ Synthetic data: {spectrograms.shape}")

    # Test 1: AST Forward Pass
    print("\n1. Testing AST...")
    ast_model = AudioSpectrogramTransformer(
        patch_size=8,  # Smaller patches for speed
        embed_dim=192,  # Smaller model
        num_layers=3,  # Fewer layers
        num_heads=4,
    )

    ast_state = create_train_state(ast_model, rng_keys[1], spectrograms.shape, 1e-4)
    predictions, attention = ast_model.apply(
        ast_state.params, spectrograms, training=False
    )

    print(f"   ✓ Predictions: {len(predictions)} dimensions")
    print(f"   ✓ Example prediction shape: {next(iter(predictions.values())).shape}")
    print(f"   ✓ Attention: {len(attention)} layers")

    # Count parameters
    ast_params = sum(x.size for x in tree_leaves(ast_state.params))
    print(f"   ✓ AST parameters: {ast_params:,}")

    # Test 2: SSAST Forward Pass
    print("\n2. Testing SSAST...")
    ssast_model = SSASTPreTrainingModel(
        patch_size=8, embed_dim=192, num_layers=3, num_heads=4
    )

    ssast_state = create_ssast_train_state(
        ssast_model, rng_keys[2], spectrograms.shape, 1e-4
    )

    outputs = ssast_model.apply(
        ssast_state.params, spectrograms, rng_keys[3], training=False
    )

    print(f"   ✓ Discriminative: {outputs['predictions']['discriminative'].shape}")
    print(f"   ✓ Generative: {outputs['predictions']['generative'].shape}")
    print(f"   ✓ Masked patches: {jnp.sum(outputs['mask'])}")

    ssast_params = sum(x.size for x in tree_leaves(ssast_state.params))
    print(f"   ✓ SSAST parameters: {ssast_params:,}")

    # Test 3: Loss computation
    print("\n3. Testing Loss...")

    # Simple MSE loss for AST
    # Split additional keys for synthetic labels if needed
    label_keys = jax.random.split(rng_keys[4], len(predictions))
    synthetic_labels = {
        dim: jax.random.uniform(label_keys[i], (batch_size,))
        for i, dim in enumerate(predictions.keys())
    }

    ast_loss = sum(
        jnp.mean((predictions[dim] - synthetic_labels[dim]) ** 2)
        for dim in predictions.keys()
    ) / len(predictions)
    print(f"   ✓ AST loss: {ast_loss:.4f}")

    # SSAST loss
    from models.ssast_pretraining import compute_ssast_loss

    targets = {"original_patches": outputs["original_patches"]}
    ssast_losses = compute_ssast_loss(outputs, targets, outputs["mask"])
    print(f"   ✓ SSAST loss: {ssast_losses['total_loss']:.4f}")

    print(f"\n🎉 SUCCESS!")
    print("=" * 30)
    print("✅ AST architecture working")
    print("✅ SSAST pre-training working")
    print("✅ Loss computation working")
    print("✅ Pipeline ready for full training!")

    return True


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
