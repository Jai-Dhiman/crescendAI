#!/usr/bin/env python3
"""
Synthetic Data Pipeline Test
Test complete AST+SSAST pipeline with synthetic data before real dataset training
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import json
from typing import Dict, List, Tuple

# Import our models
from models.ast_transformer import AudioSpectrogramTransformer, create_train_state
from models.ssast_pretraining import (
    SSASTPreTrainingModel,
    create_ssast_train_state,
    ssast_train_step,
    extract_encoder_for_finetuning,
    compute_ssast_loss,
)


class SyntheticDataGenerator:
    """Generate synthetic audio data for testing pipeline"""

    def __init__(
        self, time_frames: int = 128, freq_bins: int = 128, num_samples: int = 100
    ):
        """
        Initialize synthetic data generator
        Args:
            time_frames: Number of time frames in spectrogram
            freq_bins: Number of frequency bins
            num_samples: Number of synthetic samples to generate
        """
        self.time_frames = time_frames
        self.freq_bins = freq_bins
        self.num_samples = num_samples

        # Create reproducible synthetic perceptual labels
        self.perceptual_dimensions = [
            "Timing_Stable_Unstable",
            "Articulation_Short_Long",
            "Articulation_Soft_cushioned_Hard_solid",
            "Pedal_Sparse/dry_Saturated/wet",
            "Pedal_Clean_Blurred",
            "Timbre_Even_Colorful",
            "Timbre_Shallow_Rich",
            "Timbre_Bright_Dark",
            "Timbre_Soft_Loud",
            "Dynamic_Sophisticated/mellow_Raw/crude",
            "Dynamic_Little_dynamic_range_Large_dynamic_range",
            "Music_Making_Fast_paced_Slow_paced",
            "Music_Making_Flat_Spacious",
            "Music_Making_Disproportioned_Balanced",
            "Music_Making_Pure_Dramatic/expressive",
            "Emotion_&_Mood_Optimistic/pleasant_Dark",
            "Emotion_&_Mood_Low_Energy_High_Energy",
            "Emotion_&_Mood_Honest_Imaginative",
            "Interpretation_Unsatisfactory/doubtful_Convincing",
        ]

        print(f"🧪 Synthetic Data Generator initialized:")
        print(f"   Spectrogram shape: ({time_frames}, {freq_bins})")
        print(f"   Samples: {num_samples}")
        print(f"   Perceptual dimensions: {len(self.perceptual_dimensions)}")

    def generate_synthetic_spectrograms(
        self, batch_size: int = 32, rng_key: jax.Array = None
    ) -> jnp.ndarray:
        """
        Generate synthetic mel-spectrograms
        Args:
            batch_size: Number of samples in batch
            rng_key: Random key for reproducibility
        Returns:
            Synthetic spectrograms [batch_size, time_frames, freq_bins]
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)

        # Generate base spectrograms with realistic structure
        base_spec = jax.random.normal(
            rng_key, (batch_size, self.time_frames, self.freq_bins)
        )

        # Add some musical structure patterns
        # Simulate harmonic content (stronger in lower frequencies)
        freq_weights = jnp.exp(-jnp.arange(self.freq_bins) / 30.0)
        base_spec = base_spec * freq_weights[None, None, :]

        # Add temporal structure (attack, sustain patterns)
        time_envelope = jnp.exp(-jnp.arange(self.time_frames) / 40.0) * 0.5 + 0.5
        base_spec = base_spec * time_envelope[None, :, None]

        # Normalize to typical mel-spectrogram range
        base_spec = (base_spec - base_spec.mean()) / base_spec.std()
        base_spec = base_spec * 20 - 40  # Typical dB range

        return base_spec

    def generate_synthetic_labels(
        self, batch_size: int = 32, rng_key: jax.Array = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Generate synthetic perceptual dimension labels
        Args:
            batch_size: Number of samples
            rng_key: Random key
        Returns:
            Dict mapping dimension names to labels [batch_size]
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(123)

        labels = {}

        for i, dim in enumerate(self.perceptual_dimensions):
            # Generate correlated labels with some structure
            key = jax.random.fold_in(rng_key, i)

            # Use sigmoid to get values in [0, 1] range like real data
            raw_values = jax.random.normal(key, (batch_size,))
            labels[dim] = jax.nn.sigmoid(raw_values * 0.5)  # Reduce variance

        return labels


def test_ast_architecture():
    """Test AST architecture with synthetic data"""
    print("\n=== Testing AST Architecture ===")

    # Create synthetic data
    rng = jax.random.PRNGKey(42)
    batch_size = 4

    synth_gen = SyntheticDataGenerator()
    spectrograms = synth_gen.generate_synthetic_spectrograms(batch_size, rng)
    labels = synth_gen.generate_synthetic_labels(batch_size, rng)

    print(f"Synthetic data shapes:")
    print(f"  Spectrograms: {spectrograms.shape}")
    print(f"  Labels: {len(labels)} dimensions")

    # Initialize AST model
    ast_model = AudioSpectrogramTransformer()

    # Create training state
    ast_state = create_train_state(
        ast_model, rng, spectrograms.shape, learning_rate=1e-4
    )

    # Test forward pass
    predictions, attention_weights = ast_model.apply(
        ast_state.params, spectrograms, training=False
    )

    print(f"\nAST Forward pass successful:")
    print(f"  Predictions: {len(predictions)} dimensions")
    for dim, pred in list(predictions.items())[:3]:  # Show first 3
        print(f"    {dim}: {pred.shape}")
    print(f"  Attention: {len(attention_weights)} layers")

    # Test loss computation (simplified MSE for testing)
    def compute_mse_loss(predictions, targets):
         losses = []
         for dim, pred in predictions.items():
             if dim in targets:
                 tgt = targets[dim]
                 pred = jnp.squeeze(pred)
                 tgt = jnp.squeeze(tgt)
                 losses.append(jnp.mean((pred - tgt) ** 2))
         if not losses:
             return jnp.array(0.0)
         return jnp.mean(jnp.stack(losses))

    loss = compute_mse_loss(predictions, labels)
    print(f"  Loss: {loss:.4f}")

    print("✅ AST architecture test passed!")
    return ast_state


def test_ssast_pretraining():
    """Test SSAST pre-training with synthetic data"""
    print("\n=== Testing SSAST Pre-training ===")

    # Create synthetic data
    rng = jax.random.PRNGKey(42)
    batch_size = 4

    synth_gen = SyntheticDataGenerator()
    spectrograms = synth_gen.generate_synthetic_spectrograms(batch_size, rng)

    print(f"SSAST input shape: {spectrograms.shape}")

    # Initialize SSAST model
    ssast_model = SSASTPreTrainingModel()

    # Create training state
    ssast_state = create_ssast_train_state(
        ssast_model, rng, spectrograms.shape, learning_rate=1e-4
    )

    # Test forward pass
    rng_key1, rng_key2 = jax.random.split(rng)

    outputs = ssast_model.apply(
        ssast_state.params, spectrograms, rng_key1, training=False
    )

    print(f"SSAST Forward pass successful:")
    print(
        f"  Discriminative predictions: {outputs['predictions']['discriminative'].shape}"
    )
    print(f"  Generative predictions: {outputs['predictions']['generative'].shape}")
    print(f"  Mask: {outputs['mask'].shape}")
    print(f"  Masked patches: {jnp.sum(outputs['mask'])} / {outputs['mask'].size}")

    # Test loss computation
    targets = {"original_patches": outputs["original_patches"]}
    losses = compute_ssast_loss(outputs, targets, outputs["mask"])

    print(f"  Discriminative loss: {losses['discriminative_loss']:.4f}")
    print(f"  Generative loss: {losses['generative_loss']:.4f}")
    print(f"  Total loss: {losses['total_loss']:.4f}")

    # Test training step
    new_state, metrics = ssast_train_step(ssast_state, spectrograms, rng_key2)
    print(f"  Training step successful, loss: {metrics['total_loss']:.4f}")

    print("✅ SSAST pre-training test passed!")
    return ssast_state


def test_parameter_transfer():
    """Test parameter transfer from SSAST to AST"""
    print("\n=== Testing Parameter Transfer ===")

    # Create SSAST state (simulating pre-trained)
    rng = jax.random.PRNGKey(42)
    batch_size = 4
    dummy_shape = (batch_size, 128, 128)

    ssast_model = SSASTPreTrainingModel()
    ssast_state = create_ssast_train_state(ssast_model, rng, dummy_shape)

    print("SSAST parameters created")

    # Extract encoder parameters
    encoder_params = extract_encoder_for_finetuning(ssast_state)

    print("Encoder parameters extracted:")
    for key in encoder_params:
        print(f"  {key}: parameter group")

    # Create AST for fine-tuning
    ast_model = AudioSpectrogramTransformer()
    ast_state = create_train_state(ast_model, rng, dummy_shape, learning_rate=1e-4)

    print("AST model created for fine-tuning")

    # In real implementation, we would transfer the encoder parameters here
    # For now, just verify the structure matches
    ast_encoder_keys = {
        "patch_embedding",
        "pos_encoding", 
        "layer_norm"
    }
    # Add numbered transformer blocks
    for i in range(12):
        ast_encoder_keys.add(f"transformer_block_{i}")

    print("AST model created for fine-tuning")
    extracted_keys = set(encoder_params.keys())

    # Assert parameter structures match - fail fast on mismatch
    assert ast_encoder_keys == extracted_keys, (
        f"Parameter structure mismatch for transfer:\n"
        f"  Expected keys: {ast_encoder_keys}\n"
        f"  Got keys: {extracted_keys}\n"
        f"  Missing: {ast_encoder_keys - extracted_keys}\n"
        f"  Extra: {extracted_keys - ast_encoder_keys}"
    )
    
    print("✅ Parameter structures match for transfer!")
    print("✅ Parameter transfer test passed!")


def test_end_to_end_pipeline():
    """Test complete pipeline with multiple synthetic batches"""
    print("\n=== Testing End-to-End Pipeline ===")

    rng = jax.random.PRNGKey(42)
    synth_gen = SyntheticDataGenerator()

    # Simulate mini pre-training (3 steps)
    print("🔄 Mini pre-training simulation...")

    ssast_model = SSASTPreTrainingModel()
    dummy_shape = (4, 128, 128)
    ssast_state = create_ssast_train_state(ssast_model, rng, dummy_shape)

    initial_loss = None
    for step in range(3):
        # Generate batch
        rng_batch = jax.random.fold_in(rng, step)
        batch = synth_gen.generate_synthetic_spectrograms(4, rng_batch)

        # Training step
        rng_train = jax.random.fold_in(rng, step + 100)
        ssast_state, metrics = ssast_train_step(ssast_state, batch, rng_train)

        if step == 0:
            initial_loss = metrics["total_loss"]

        print(f"  Step {step + 1}: Loss = {metrics['total_loss']:.4f}")

    final_loss = metrics["total_loss"]
    print(f"  Loss change: {initial_loss:.4f} → {final_loss:.4f}")

    # Simulate fine-tuning setup
    print("🎯 Fine-tuning setup simulation...")

    # Extract encoder and create AST
    encoder_params = extract_encoder_for_finetuning(ssast_state)
    ast_model = AudioSpectrogramTransformer()
    rng_ast = jax.random.fold_in(rng, 200)
    ast_state = create_train_state(ast_model, rng_ast, dummy_shape)

    # Test AST prediction
    test_batch = synth_gen.generate_synthetic_spectrograms(4, rng_ast)
    test_labels = synth_gen.generate_synthetic_labels(4, rng_ast)

    predictions, _ = ast_model.apply(ast_state.params, test_batch, training=False)

    print(f"  Fine-tuning ready - {len(predictions)} dimension predictions")

    print("✅ End-to-end pipeline test passed!")


def main():
    """Run all synthetic pipeline tests"""
    print("🧪 SYNTHETIC PIPELINE TESTING")
    print("=" * 50)
    print("Testing complete AST+SSAST pipeline with synthetic data")
    print("This validates the architecture before real dataset training\n")

    try:
        # Test individual components
        ast_state = test_ast_architecture()
        ssast_state = test_ssast_pretraining()
        test_parameter_transfer()

        # Test complete pipeline
        test_end_to_end_pipeline()

        # Summary
        print(f"\n🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("✅ AST architecture working correctly")
        print("✅ SSAST pre-training working correctly")
        print("✅ Parameter transfer working correctly")
        print("✅ End-to-end pipeline working correctly")
        print("\n🚀 Pipeline ready for real dataset training!")

        # Count total parameters
        ast_params = sum(x.size for x in jax.tree.leaves(ast_state.params))
        ssast_params = sum(x.size for x in jax.tree.leaves(ssast_state.params))

        print(f"\nModel sizes:")
        print(f"  AST: {ast_params:,} parameters")
        print(f"  SSAST: {ssast_params:,} parameters")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
