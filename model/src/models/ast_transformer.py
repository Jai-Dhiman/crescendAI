#!/usr/bin/env python3
"""
Audio Spectrogram Transformer (AST) Implementation
Following Gong et al. 2021 "AST: Audio Spectrogram Transformer"
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from functools import partial


class PatchEmbedding(nn.Module):
    """16x16 patch embedding for mel-spectrograms following AST specification"""
    patch_size: int = 16
    embed_dim: int = 768
    
    @nn.compact
    def __call__(self, x):
        """
        Convert mel-spectrogram to patch embeddings
        Args:
            x: Input mel-spectrogram [batch, time, freq]
        Returns:
            Patch embeddings [batch, num_patches, embed_dim]
        """
        batch_size, time_frames, freq_bins = x.shape
        
        # Ensure input can be divided into patches
        time_pad = (self.patch_size - time_frames % self.patch_size) % self.patch_size
        freq_pad = (self.patch_size - freq_bins % self.patch_size) % self.patch_size
        
        if time_pad > 0 or freq_pad > 0:
            x = jnp.pad(x, ((0, 0), (0, time_pad), (0, freq_pad)), mode='constant')
        
        # Extract patches: [batch, time, freq] -> [batch, num_patches, patch_size*patch_size]
        time_patches = x.shape[1] // self.patch_size
        freq_patches = x.shape[2] // self.patch_size
        
        # Reshape to patches
        x = x.reshape(batch_size, time_patches, self.patch_size, freq_patches, self.patch_size)
        x = x.transpose(0, 1, 3, 2, 4)  # [batch, time_patches, freq_patches, patch_size, patch_size]
        x = x.reshape(batch_size, time_patches * freq_patches, self.patch_size * self.patch_size)
        
        # Linear projection to embed_dim
        x = nn.Dense(self.embed_dim, name='patch_projection')(x)
        
        return x


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spectrograms (time and frequency dimensions)"""
    embed_dim: int = 768
    max_time_len: int = 1000
    max_freq_len: int = 128
    
    @nn.compact 
    def __call__(self, x):
        """
        Add 2D positional encoding to patch embeddings
        Args:
            x: Patch embeddings [batch, num_patches, embed_dim]
        Returns:
            Position-encoded embeddings [batch, num_patches, embed_dim]
        """
        batch_size, num_patches, embed_dim = x.shape
        
        # Create learned positional embeddings
        pos_embedding = self.param('pos_embedding', 
                                 nn.initializers.normal(stddev=0.02),
                                 (1, num_patches, embed_dim))
        
        return x + pos_embedding


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention following transformer specification"""
    num_heads: int = 12
    head_dim: int = 64  # 768 // 12
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        """
        Multi-head self-attention
        Args:
            x: Input embeddings [batch, seq_len, embed_dim]
        Returns:
            Attended embeddings [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear projections for Q, K, V
        q = nn.Dense(self.num_heads * self.head_dim, name='query')(x)
        k = nn.Dense(self.num_heads * self.head_dim, name='key')(x)  
        v = nn.Dense(self.num_heads * self.head_dim, name='value')(x)
        
        # Reshape to multi-head format
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attention_scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.head_dim)
        attention_weights = nn.softmax(attention_scores, axis=-1)
        
        # Apply dropout to attention weights
        if training:
            attention_weights = nn.Dropout(self.dropout_rate)(attention_weights, deterministic=False)
        
        # Apply attention to values
        attended = jnp.matmul(attention_weights, v)
        
        # Reshape back to original format
        attended = attended.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Final linear projection
        output = nn.Dense(embed_dim, name='output_projection')(attended)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """Transformer encoder block following AST specification"""
    embed_dim: int = 768
    num_heads: int = 12
    mlp_dim: int = 3072  # 4 * embed_dim
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        """
        Transformer encoder block with pre-LayerNorm
        Args:
            x: Input embeddings [batch, seq_len, embed_dim]
        Returns:
            Output embeddings [batch, seq_len, embed_dim]
        """
        # Multi-head self-attention with residual connection
        norm1_x = nn.LayerNorm(name='layernorm1')(x)
        attn_output, attn_weights = MultiHeadAttention(
            num_heads=self.num_heads, 
            dropout_rate=self.dropout_rate,
            name='attention'
        )(norm1_x, training=training)
        
        x = x + nn.Dropout(self.dropout_rate)(attn_output, deterministic=not training)
        
        # MLP with residual connection  
        norm2_x = nn.LayerNorm(name='layernorm2')(x)
        mlp_output = nn.Dense(self.mlp_dim, name='mlp_dense1')(norm2_x)
        mlp_output = nn.gelu(mlp_output)
        mlp_output = nn.Dropout(self.dropout_rate)(mlp_output, deterministic=not training)
        mlp_output = nn.Dense(self.embed_dim, name='mlp_dense2')(mlp_output)
        
        x = x + nn.Dropout(self.dropout_rate)(mlp_output, deterministic=not training)
        
        return x, attn_weights


class GroupedMultiTaskHead(nn.Module):
    """Grouped multi-task regression heads for perceptual dimensions"""
    group_configs: Dict[str, List[str]]
    embed_dim: int = 768
    hidden_dim: int = 256
    dropout_rate: float = 0.3
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        """
        Apply grouped multi-task heads
        Args:
            x: Global representation [batch, embed_dim] 
        Returns:
            Dict of predictions for each dimension
        """
        predictions = {}
        
        for group_name, dimensions in self.group_configs.items():
            # Shared group processing
            shared_features = nn.Dense(self.hidden_dim, name=f'{group_name}_shared_1')(x)
            shared_features = nn.relu(shared_features)
            shared_features = nn.Dropout(self.dropout_rate, deterministic=not training)(shared_features)
            shared_features = nn.Dense(self.hidden_dim // 2, name=f'{group_name}_shared_2')(shared_features)
            shared_features = nn.relu(shared_features)
            
            # Individual dimension predictions
            for dim in dimensions:
                dim_features = nn.Dropout(self.dropout_rate, deterministic=not training)(shared_features)
                dim_output = nn.Dense(1, name=f'{group_name}_{dim}_output')(dim_features)
                predictions[dim] = dim_output.squeeze(-1)  # Remove last dimension
                
        return predictions


class AudioSpectrogramTransformer(nn.Module):
    """
    Audio Spectrogram Transformer (AST) for piano performance analysis
    Following Gong et al. 2021 specifications with grouped multi-task learning
    """
    # Architecture hyperparameters
    patch_size: int = 16
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout_rate: float = 0.1
    
    # Task-specific configurations
    perceptual_groups: Dict[str, List[str]] = None
    
    def setup(self):
        """Initialize AST components"""
        # Default perceptual dimension groupings based on correlation analysis
        if self.perceptual_groups is None:
            perceptual_groups = {
                'timing': ['Timing_Stable_Unstable'],
                'dynamics_articulation': [
                    'Articulation_Short_Long', 
                    'Articulation_Soft_cushioned_Hard_solid',
                    'Dynamic_Sophisticated/mellow_Raw/crude',
                    'Dynamic_Little_dynamic_range_Large_dynamic_range'
                ],
                'expression_emotion': [
                    'Music_Making_Fast_paced_Slow_paced',
                    'Music_Making_Flat_Spacious', 
                    'Music_Making_Disproportioned_Balanced',
                    'Music_Making_Pure_Dramatic/expressive',
                    'Emotion_&_Mood_Optimistic/pleasant_Dark',
                    'Emotion_&_Mood_Low_Energy_High_Energy',
                    'Emotion_&_Mood_Honest_Imaginative',
                    'Interpretation_Unsatisfactory/doubtful_Convincing'
                ],
                'timbre_pedal': [
                    'Pedal_Sparse/dry_Saturated/wet',
                    'Pedal_Clean_Blurred',
                    'Timbre_Even_Colorful',
                    'Timbre_Shallow_Rich', 
                    'Timbre_Bright_Dark',
                    'Timbre_Soft_Loud'
                ]
            }
        else:
            perceptual_groups = self.perceptual_groups
        
        # Core transformer components
        self.patch_embedding = PatchEmbedding(
            patch_size=self.patch_size,
            embed_dim=self.embed_dim
        )
        
        self.pos_encoding = PositionalEncoding2D(embed_dim=self.embed_dim)
        
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
        
        # Multi-task regression heads
        self.task_heads = GroupedMultiTaskHead(
            group_configs=perceptual_groups,
            embed_dim=self.embed_dim,
            dropout_rate=self.dropout_rate
        )
    
    @nn.compact  
    def __call__(self, x, training: bool = True):
        """
        Forward pass through AST
        Args:
            x: Mel-spectrogram [batch, time, freq]
        Returns:
            predictions: Dict of perceptual dimension predictions
            attention_weights: List of attention weights from each layer
        """
        # Patch embedding
        x = self.patch_embedding(x)
        x = self.pos_encoding(x)
        
        # Add dropout after embeddings
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        
        # Transformer encoder layers
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn = block(x, training=training)
            attention_weights.append(attn)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Global average pooling to get sequence representation
        x = jnp.mean(x, axis=1)  # [batch, embed_dim]
        
        # Multi-task prediction heads
        predictions = self.task_heads(x, training=training)
        
        return predictions, attention_weights


def create_ast_model() -> AudioSpectrogramTransformer:
    """Create AST model with default configuration"""
    return AudioSpectrogramTransformer()


def create_train_state(model: nn.Module, rng_key: jax.Array, input_shape: Tuple[int, ...], learning_rate: float = 1e-4) -> train_state.TrainState:
    """
    Create training state for AST model
    Args:
        model: AST model
        rng_key: Random key for initialization
        input_shape: Input shape (batch, time, freq)
        learning_rate: Learning rate for optimizer
    Returns:
        TrainState for training
    """
    # Initialize parameters
    dummy_input = jnp.ones(input_shape)
    params = model.init(rng_key, dummy_input, training=False)
    
    # Create optimizer (AdamW with cosine decay)
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=0.05)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


if __name__ == "__main__":
    # Test AST model
    print("=== Audio Spectrogram Transformer Implementation ===\n")
    
    # Initialize model
    ast_model = create_ast_model()
    
    # Test with dummy mel-spectrogram
    rng = jax.random.PRNGKey(42)
    batch_size, time_frames, freq_bins = 4, 128, 128  # Example dimensions
    dummy_input = jax.random.normal(rng, (batch_size, time_frames, freq_bins))
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Initialize and test forward pass
    train_state = create_train_state(ast_model, rng, dummy_input.shape, learning_rate=1e-4)
    
    # Forward pass
    predictions, attention_weights = ast_model.apply(train_state.params, dummy_input, training=False)
    
    print(f"\nPredictions for each perceptual dimension:")
    for dim, pred in predictions.items():
        print(f"  {dim}: {pred.shape}")
    
    print(f"\nAttention weights: {len(attention_weights)} layers")
    print(f"Attention shape per layer: {attention_weights[0].shape}")
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree.leaves(train_state.params))
    print(f"\nTotal parameters: {param_count:,}")
    
    print("\n✅ AST model implementation complete!")