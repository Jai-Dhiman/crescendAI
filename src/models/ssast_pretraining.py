#!/usr/bin/env python3
"""
SSAST: Self-Supervised Audio Spectrogram Transformer
Masked Spectrogram Patch Modeling (MSPM) for pre-training
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from functools import partial

from .ast_transformer import (
    AudioSpectrogramTransformer, 
    PatchEmbedding, 
    PositionalEncoding2D,
    TransformerBlock
)


class MaskedSpectrogramPatchModeling(nn.Module):
    """
    SSAST pre-training head for Masked Spectrogram Patch Modeling (MSPM)
    Combines discriminative and generative objectives
    """
    embed_dim: int = 768
    patch_size: int = 16
    num_classes: int = 2  # masked vs unmasked (discriminative)
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        """
        MSPM prediction heads
        Args:
            x: Patch embeddings [batch, num_patches, embed_dim]
        Returns:
            Dict with discriminative and generative predictions
        """
        
        # Discriminative head: predict if patch is masked
        discriminative_logits = nn.Dense(
            self.num_classes, 
            name='discriminative_head'
        )(x)  # [batch, num_patches, num_classes]
        
        # Generative head: reconstruct original patch values
        generative_output = nn.Dense(
            self.patch_size * self.patch_size,
            name='generative_head'
        )(x)  # [batch, num_patches, patch_size^2]
        
        return {
            'discriminative': discriminative_logits,
            'generative': generative_output
        }


class SSASTPreTrainingModel(nn.Module):
    """
    Self-supervised AST model for pre-training with MSPM
    """
    # AST backbone parameters
    patch_size: int = 16
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout_rate: float = 0.1
    
    # SSAST specific parameters
    mask_prob: float = 0.15  # Probability of masking patches
    
    def setup(self):
        """Initialize SSAST components"""
        
        # AST backbone (encoder only)
        self.patch_embedding = PatchEmbedding(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim
        )
        
        self.pos_encoding = PositionalEncoding2D(embed_dim=self.embed_dim)
        
        # Special [MASK] token embedding
        self.mask_token = self.param(
            'mask_token',
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.embed_dim)
        )
        
        # Transformer encoder layers
        self.transformer_blocks = [
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name=f'transformer_block_{i}'
            ) for i in range(self.num_layers)
        ]
        
        self.layer_norm = nn.LayerNorm()
        
        # MSPM pre-training head
        self.msmp_head = MaskedSpectrogramPatchModeling(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size
        )
    
    def create_mask(self, 
                   rng: jax.Array, 
                   batch_size: int, 
                   num_patches: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Create random mask for patches
        Args:
            rng: Random key
            batch_size: Batch size
            num_patches: Number of patches per sample
        Returns:
            mask: Boolean mask [batch, num_patches] (True = masked)
            mask_indices: Indices of masked patches
        """
        # Generate random mask
        mask_probs = jax.random.uniform(rng, (batch_size, num_patches))
        mask = mask_probs < self.mask_prob
        
        return mask, mask
    
    @nn.compact
    def __call__(self, 
                x, 
                rng: jax.Array, 
                training: bool = True) -> Dict:
        """
        SSAST forward pass with masking
        Args:
            x: Input mel-spectrograms [batch, time, freq]
            rng: Random key for masking
            training: Training mode
        Returns:
            Dict with predictions and mask information
        """
        batch_size = x.shape[0]
        
        # Store original input for reconstruction targets
        original_input = x
        
        # Extract patch embeddings  
        patch_embeddings = self.patch_embedding(x)  # [batch, num_patches, embed_dim]
        num_patches = patch_embeddings.shape[1]
        
        # Create original patch values for reconstruction (before embedding)
        time_frames, freq_bins = x.shape[1], x.shape[2]
        time_pad = (self.patch_size - time_frames % self.patch_size) % self.patch_size
        freq_pad = (self.patch_size - freq_bins % self.patch_size) % self.patch_size
        
        if time_pad > 0 or freq_pad > 0:
            x_padded = jnp.pad(x, ((0, 0), (0, time_pad), (0, freq_pad)), mode='constant')
        else:
            x_padded = x
            
        # Extract original patches for reconstruction target
        time_patches = x_padded.shape[1] // self.patch_size
        freq_patches = x_padded.shape[2] // self.patch_size
        
        original_patches = x_padded.reshape(
            batch_size, time_patches, self.patch_size, freq_patches, self.patch_size
        ).transpose(0, 1, 3, 2, 4).reshape(
            batch_size, time_patches * freq_patches, self.patch_size * self.patch_size
        )
        
        # Create mask
        mask, mask_indices = self.create_mask(rng, batch_size, num_patches)
        
        # Apply positional encoding
        patch_embeddings = self.pos_encoding(patch_embeddings)
        
        # Replace masked patches with [MASK] token
        mask_tokens = jnp.broadcast_to(
            self.mask_token, 
            (batch_size, num_patches, self.embed_dim)
        )
        
        # Use mask to select between original patches and mask tokens
        masked_embeddings = jnp.where(
            mask[..., None],  # Broadcast mask to embed_dim
            mask_tokens,
            patch_embeddings
        )
        
        # Add dropout
        masked_embeddings = nn.Dropout(self.dropout_rate)(
            masked_embeddings, 
            deterministic=not training
        )
        
        # Pass through transformer encoder
        x = masked_embeddings
        attention_weights = []
        
        for block in self.transformer_blocks:
            x, attn = block(x, training=training)
            attention_weights.append(attn)
        
        x = self.layer_norm(x)
        
        # Apply MSMP head to get predictions
        msmp_predictions = self.msmp_head(x, training=training)
        
        return {
            'predictions': msmp_predictions,
            'mask': mask,
            'attention_weights': attention_weights,
            'original_patches': original_patches,  # Raw patch values for reconstruction
            'masked_embeddings': masked_embeddings
        }


def compute_ssast_loss(predictions: Dict, 
                      targets: Dict, 
                      mask: jnp.ndarray) -> Dict:
    """
    Compute SSAST pre-training loss (discriminative + generative)
    Args:
        predictions: Model predictions
        targets: Target values  
        mask: Masking pattern [batch, num_patches]
    Returns:
        Dict with individual and total losses
    """
    
    # Discriminative loss: predict which patches are masked
    discriminative_targets = mask.astype(jnp.int32)  # 0 = unmasked, 1 = masked
    discriminative_loss = optax.softmax_cross_entropy_with_integer_labels(
        predictions['predictions']['discriminative'],
        discriminative_targets
    ).mean()
    
    # Generative loss: reconstruct original patch values (only for masked patches)
    generative_preds = predictions['predictions']['generative']
    generative_targets = targets['original_patches']  # Original patch values
    
    # MSE loss only on masked patches
    reconstruction_error = jnp.square(generative_preds - generative_targets)
    masked_reconstruction_error = reconstruction_error * mask[..., None]  # Apply mask
    
    # Average over masked patches only
    num_masked = jnp.sum(mask) + 1e-8  # Avoid division by zero
    generative_loss = jnp.sum(masked_reconstruction_error) / num_masked
    
    # Combined loss
    total_loss = discriminative_loss + generative_loss
    
    return {
        'total_loss': total_loss,
        'discriminative_loss': discriminative_loss, 
        'generative_loss': generative_loss,
        'num_masked_patches': jnp.sum(mask)
    }


def create_ssast_train_state(model: SSASTPreTrainingModel,
                            rng_key: jax.Array,
                            input_shape: Tuple[int, ...],
                            learning_rate: float = 1e-4) -> train_state.TrainState:
    """Create training state for SSAST pre-training"""
    
    # Initialize parameters
    dummy_input = jnp.ones(input_shape)
    dummy_rng = jax.random.PRNGKey(0)
    
    params = model.init(rng_key, dummy_input, dummy_rng, training=False)
    
    # Create optimizer with cosine decay schedule
    warmup_steps = 5000
    total_steps = 100000
    
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps
    )
    
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps - warmup_steps
    )
    
    schedule = optax.join_schedules(
        schedules=[warmup_schedule, cosine_schedule],
        boundaries=[warmup_steps]
    )
    
    # AdamW optimizer with weight decay
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(learning_rate=schedule, weight_decay=0.05)
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


@jax.jit
def ssast_train_step(state: train_state.TrainState,
                    batch: jnp.ndarray,
                    rng: jax.Array) -> Tuple[train_state.TrainState, Dict]:
    """Single SSAST training step"""
    
    def loss_fn(params, dropout_rng):
        # Forward pass
        outputs = state.apply_fn(
            params, 
            batch, 
            rng, 
            training=True,
            rngs={'dropout': dropout_rng}
        )
        
        # Prepare targets for loss computation
        targets = {
            'original_patches': outputs['original_patches']
        }
        
        # Compute loss
        losses = compute_ssast_loss(
            outputs, 
            targets, 
            outputs['mask']
        )
        
        return losses['total_loss'], (losses, outputs)
    
    # Split RNG for dropout
    dropout_rng = jax.random.fold_in(rng, state.step)
    
    # Compute gradients
    (loss, (losses, outputs)), grads = jax.value_and_grad(
        loss_fn, 
        has_aux=True
    )(state.params, dropout_rng)
    
    # Update parameters
    state = state.apply_gradients(grads=grads)
    
    # Return metrics
    metrics = {
        **losses,
        'learning_rate': 1e-4  # Simplified for now
    }
    
    return state, metrics


def extract_encoder_for_finetuning(ssast_state: train_state.TrainState) -> Dict:
    """
    Extract encoder parameters from pre-trained SSAST for fine-tuning
    Args:
        ssast_state: Pre-trained SSAST training state
    Returns:
        Encoder parameters for transfer to downstream tasks
    """
    # Extract encoder parameters (exclude MSMP head)
    params = ssast_state.params['params']
    encoder_params = {
        'patch_embedding': params['patch_embedding'],
        'pos_encoding': params['pos_encoding'],
        'layer_norm': params['layer_norm']
    }
    
    # Extract numbered transformer blocks
    for i in range(12):  # 12 layers in default config
        block_name = f'transformer_block_{i}'
        if block_name in params:
            encoder_params[block_name] = params[block_name]
    
    return encoder_params


if __name__ == "__main__":
    # Test SSAST implementation
    print("=== Self-Supervised AST (SSAST) Implementation ===\n")
    
    # Initialize SSAST model
    ssast_model = SSASTPreTrainingModel()
    
    # Test with dummy input
    rng = jax.random.PRNGKey(42)
    batch_size, time_frames, freq_bins = 4, 128, 128
    dummy_batch = jax.random.normal(rng, (batch_size, time_frames, freq_bins))
    
    print(f"Input batch shape: {dummy_batch.shape}")
    
    # Create training state
    train_state = create_ssast_train_state(
        ssast_model, 
        rng, 
        dummy_batch.shape,
        learning_rate=1e-4
    )
    
    # Test forward pass
    rng_key1, rng_key2 = jax.random.split(rng)
    outputs = ssast_model.apply(
        train_state.params, 
        dummy_batch, 
        rng_key1, 
        training=False
    )
    
    print(f"\nSSAST Outputs:")
    print(f"  Discriminative predictions: {outputs['predictions']['discriminative'].shape}")
    print(f"  Generative predictions: {outputs['predictions']['generative'].shape}")
    print(f"  Mask shape: {outputs['mask'].shape}")
    print(f"  Num masked patches: {jnp.sum(outputs['mask'])}")
    
    # Test loss computation
    targets = {'original_patches': outputs['original_patches']}
    losses = compute_ssast_loss(outputs, targets, outputs['mask'])
    
    print(f"\nLoss computation:")
    print(f"  Total loss: {losses['total_loss']:.4f}")
    print(f"  Discriminative loss: {losses['discriminative_loss']:.4f}")
    print(f"  Generative loss: {losses['generative_loss']:.4f}")
    
    # Test training step
    print(f"\nTesting training step...")
    new_state, metrics = ssast_train_step(train_state, dummy_batch, rng_key2)
    
    print(f"  Training step metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float, jnp.ndarray)):
            if hasattr(value, 'shape') and value.shape == ():
                print(f"    {key}: {float(value):.4f}")
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree.leaves(train_state.params))
    print(f"\nTotal SSAST parameters: {param_count:,}")
    
    print(f"\n✅ SSAST implementation complete!")
    print(f"Ready for self-supervised pre-training on MAESTRO dataset")