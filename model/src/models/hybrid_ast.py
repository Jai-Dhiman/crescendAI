#!/usr/bin/env python3
"""
Hybrid Audio Spectrogram Transformer (Hybrid AST)
Combines AST with traditional audio features to leverage domain knowledge
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from typing import Dict, List, Tuple, Optional
import numpy as np
from functools import partial

from .ast_transformer import AudioSpectrogramTransformer, PatchEmbedding, PositionalEncoding2D, TransformerBlock


class TraditionalFeatureExtractor(nn.Module):
    """Neural network for processing traditional audio features"""
    feature_dim: int = 64  # Total traditional features
    hidden_dim: int = 128
    output_dim: int = 256
    dropout_rate: float = 0.3
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        """
        Process traditional audio features
        Args:
            x: Traditional features [batch, feature_dim]
        Returns:
            Processed features [batch, output_dim]
        """
        # First dense layer
        x = nn.Dense(self.hidden_dim, name='traditional_dense1')(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        
        # Second dense layer
        x = nn.Dense(self.hidden_dim, name='traditional_dense2')(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        
        # Output layer
        x = nn.Dense(self.output_dim, name='traditional_output')(x)
        
        return x


class AttentionFusion(nn.Module):
    """Attention-based fusion of AST and traditional features"""
    embed_dim: int = 768
    traditional_dim: int = 256
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, ast_features, traditional_features, training: bool = True):
        """
        Fuse AST and traditional features using attention
        Args:
            ast_features: AST output [batch, embed_dim]
            traditional_features: Traditional features [batch, traditional_dim]
        Returns:
            Fused features [batch, embed_dim]
        """
        batch_size = ast_features.shape[0]
        
        # Project traditional features to same dimension as AST
        traditional_proj = nn.Dense(self.embed_dim, name='traditional_projection')(traditional_features)
        
        # Stack features for attention
        features = jnp.stack([ast_features, traditional_proj], axis=1)  # [batch, 2, embed_dim]
        
        # Multi-head attention for fusion
        query = nn.Dense(self.embed_dim, name='fusion_query')(ast_features)
        key = nn.Dense(self.embed_dim, name='fusion_key')(features.reshape(batch_size * 2, self.embed_dim))
        value = nn.Dense(self.embed_dim, name='fusion_value')(features.reshape(batch_size * 2, self.embed_dim))
        
        # Reshape for attention computation
        key = key.reshape(batch_size, 2, self.embed_dim)
        value = value.reshape(batch_size, 2, self.embed_dim)
        query = query[:, None, :]  # [batch, 1, embed_dim]
        
        # Compute attention scores
        attention_scores = jnp.matmul(query, key.transpose(0, 2, 1)) / jnp.sqrt(self.embed_dim)
        attention_weights = nn.softmax(attention_scores, axis=-1)  # [batch, 1, 2]
        
        # Apply attention
        fused = jnp.matmul(attention_weights, value).squeeze(1)  # [batch, embed_dim]
        
        # Add residual connection
        fused = fused + ast_features
        
        # Final normalization
        fused = nn.LayerNorm(name='fusion_norm')(fused)
        
        return fused, attention_weights.squeeze(1)


class GatedFusion(nn.Module):
    """Gated fusion of AST and traditional features"""
    embed_dim: int = 768
    traditional_dim: int = 256
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, ast_features, traditional_features, training: bool = True):
        """
        Fuse AST and traditional features using gating mechanism
        Args:
            ast_features: AST output [batch, embed_dim]
            traditional_features: Traditional features [batch, traditional_dim]
        Returns:
            Fused features [batch, embed_dim]
        """
        # Project traditional features to same dimension
        traditional_proj = nn.Dense(self.embed_dim, name='traditional_projection')(traditional_features)
        
        # Compute gate
        concat_features = jnp.concatenate([ast_features, traditional_proj], axis=-1)
        gate = nn.Dense(self.embed_dim, name='gate_dense')(concat_features)
        gate = nn.sigmoid(gate)
        
        # Apply gating
        fused = gate * ast_features + (1 - gate) * traditional_proj
        
        # Add residual and normalize
        fused = fused + ast_features
        fused = nn.LayerNorm(name='gated_norm')(fused)
        
        return fused, gate


class HybridAudioSpectrogramTransformer(nn.Module):
    """
    Hybrid AST that combines spectrogram transformer with traditional audio features
    """
    # AST architecture parameters
    patch_size: int = 16
    embed_dim: int = 512  # Reduced from 768 for smaller model
    num_layers: int = 6   # Reduced from 12 for smaller model
    num_heads: int = 8    # Reduced from 12 for smaller model
    dropout_rate: float = 0.2  # Increased dropout for regularization
    
    @property
    def mlp_dim(self):
        """MLP dimension computed as 4x embed_dim"""
        return 4 * self.embed_dim
    
    # Traditional feature parameters
    traditional_feature_dim: int = 64
    traditional_hidden_dim: int = 128
    traditional_output_dim: int = 256
    
    # Fusion strategy: 'concat', 'attention', 'gated'
    fusion_strategy: str = 'attention'
    
    # Task-specific configurations
    perceptual_groups: Dict[str, List[str]] = None
    
    def setup(self):
        """Initialize hybrid AST components"""
        # Default perceptual dimension groupings
        if self.perceptual_groups is None:
            self.perceptual_groups = {
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
        
        # AST components (smaller architecture)
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
        
        self.ast_layer_norm = nn.LayerNorm()
        
        # Traditional feature processing
        self.traditional_processor = TraditionalFeatureExtractor(
            feature_dim=self.traditional_feature_dim,
            hidden_dim=self.traditional_hidden_dim,
            output_dim=self.traditional_output_dim,
            dropout_rate=self.dropout_rate
        )
        
        # Fusion module
        if self.fusion_strategy == 'attention':
            self.fusion = AttentionFusion(
                embed_dim=self.embed_dim,
                traditional_dim=self.traditional_output_dim,
                dropout_rate=self.dropout_rate
            )
        elif self.fusion_strategy == 'gated':
            self.fusion = GatedFusion(
                embed_dim=self.embed_dim,
                traditional_dim=self.traditional_output_dim,
                dropout_rate=self.dropout_rate
            )
        else:  # 'concat'
            self.fusion_dense = nn.Dense(self.embed_dim, name='fusion_dense')
        
        # Multi-task heads (reuse from original AST)
        from .ast_transformer import GroupedMultiTaskHead
        self.task_heads = GroupedMultiTaskHead(
            group_configs=self.perceptual_groups,
            embed_dim=self.embed_dim,
            dropout_rate=self.dropout_rate
        )
    
    @nn.compact
    def __call__(self, spectrogram, traditional_features, training: bool = True):
        """
        Forward pass through Hybrid AST
        Args:
            spectrogram: Mel-spectrogram [batch, time, freq]
            traditional_features: Traditional audio features [batch, feature_dim]
        Returns:
            predictions: Dict of perceptual dimension predictions
            attention_weights: List of attention weights from transformer layers
            fusion_weights: Fusion attention weights (if using attention fusion)
        """
        # AST path
        x = self.patch_embedding(spectrogram)
        x = self.pos_encoding(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        
        # Transformer encoder layers
        attention_weights = []
        for block in self.transformer_blocks:
            x, attn = block(x, training=training)
            attention_weights.append(attn)
        
        # Final layer norm and global pooling
        x = self.ast_layer_norm(x)
        ast_features = jnp.mean(x, axis=1)  # [batch, embed_dim]
        
        # Traditional feature path
        traditional_processed = self.traditional_processor(traditional_features, training=training)
        
        # Fusion
        fusion_weights = None
        if self.fusion_strategy in ['attention', 'gated']:
            fused_features, fusion_weights = self.fusion(ast_features, traditional_processed, training=training)
        else:  # concat
            concat_features = jnp.concatenate([ast_features, traditional_processed], axis=-1)
            fused_features = self.fusion_dense(concat_features)
            fused_features = nn.relu(fused_features)
        
        # Multi-task prediction heads
        predictions = self.task_heads(fused_features, training=training)
        
        return predictions, attention_weights, fusion_weights


def create_hybrid_ast_model(
    embed_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    fusion_strategy: str = 'attention'
) -> HybridAudioSpectrogramTransformer:
    """Create Hybrid AST model with specified configuration"""
    return HybridAudioSpectrogramTransformer(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        fusion_strategy=fusion_strategy
    )


def create_hybrid_train_state(
    model: nn.Module, 
    rng_key: jax.Array, 
    spec_shape: Tuple[int, ...], 
    trad_shape: Tuple[int, ...],
    learning_rate: float = 1e-4
) -> train_state.TrainState:
    """
    Create training state for Hybrid AST model
    Args:
        model: Hybrid AST model
        rng_key: Random key for initialization
        spec_shape: Spectrogram input shape (batch, time, freq)
        trad_shape: Traditional features shape (batch, feature_dim)
        learning_rate: Learning rate for optimizer
    Returns:
        TrainState for training
    """
    # Initialize parameters
    dummy_spec = jnp.ones(spec_shape)
    dummy_trad = jnp.ones(trad_shape)
    params = model.init(rng_key, dummy_spec, dummy_trad, training=False)
    
    # Create optimizer with slightly higher weight decay for smaller model
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=0.1)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


if __name__ == "__main__":
    # Test Hybrid AST model
    print("=== Hybrid Audio Spectrogram Transformer Implementation ===\n")
    
    # Test different fusion strategies
    fusion_strategies = ['concat', 'attention', 'gated']
    
    for strategy in fusion_strategies:
        print(f"\n--- Testing {strategy} fusion ---")
        
        # Initialize model
        hybrid_model = create_hybrid_ast_model(
            embed_dim=512,
            num_layers=6, 
            num_heads=8,
            fusion_strategy=strategy
        )
        
        # Test with dummy inputs
        rng = jax.random.PRNGKey(42)
        batch_size = 4
        
        # Spectrogram input
        time_frames, freq_bins = 128, 128
        dummy_spec = jax.random.normal(rng, (batch_size, time_frames, freq_bins))
        
        # Traditional features input
        feature_dim = 64
        dummy_trad = jax.random.normal(rng, (batch_size, feature_dim))
        
        print(f"Spectrogram shape: {dummy_spec.shape}")
        print(f"Traditional features shape: {dummy_trad.shape}")
        
        # Initialize and test forward pass
        train_state = create_hybrid_train_state(
            hybrid_model, rng, dummy_spec.shape, dummy_trad.shape, learning_rate=1e-4
        )
        
        # Forward pass
        predictions, attention_weights, fusion_weights = hybrid_model.apply(
            train_state.params, dummy_spec, dummy_trad, training=False
        )
        
        print(f"\nPredictions for each perceptual dimension:")
        for dim, pred in predictions.items():
            print(f"  {dim}: {pred.shape}")
        
        print(f"\nAttention weights: {len(attention_weights)} layers")
        if fusion_weights is not None:
            print(f"Fusion weights shape: {fusion_weights.shape}")
        
        # Count parameters
        param_count = sum(x.size for x in jax.tree.leaves(train_state.params))
        print(f"\nTotal parameters: {param_count:,}")
    
    print("\n✅ Hybrid AST model implementation complete!")