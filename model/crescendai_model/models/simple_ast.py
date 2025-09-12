#!/usr/bin/env python3
"""
Simple Audio Spectrogram Transformer for inference
Compatible with existing pickle files
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class SimpleAST(nn.Module):
    """Simplified AST model for inference"""
    patch_size: int = 16
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout_rate: float = 0.1
    num_classes: int = 19
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        Forward pass through simplified AST
        Args:
            x: Mel-spectrogram [batch, time, freq] 
        Returns:
            predictions: [batch, 19] perceptual dimension scores
        """
        batch_size, time_frames, freq_bins = x.shape
        
        # Patch embedding
        time_pad = (self.patch_size - time_frames % self.patch_size) % self.patch_size
        freq_pad = (self.patch_size - freq_bins % self.patch_size) % self.patch_size
        
        if time_pad > 0 or freq_pad > 0:
            x = jnp.pad(x, ((0, 0), (0, time_pad), (0, freq_pad)), mode='constant')
        
        # Extract patches
        time_patches = x.shape[1] // self.patch_size
        freq_patches = x.shape[2] // self.patch_size
        
        x = x.reshape(batch_size, time_patches, self.patch_size, freq_patches, self.patch_size)
        x = x.transpose(0, 1, 3, 2, 4)
        x = x.reshape(batch_size, time_patches * freq_patches, self.patch_size * self.patch_size)
        
        # Patch projection
        x = nn.Dense(self.embed_dim)(x)
        
        # Add positional encoding
        num_patches = time_patches * freq_patches
        pos_embedding = self.param('pos_embedding',
                                 nn.initializers.normal(stddev=0.02),
                                 (1, num_patches, self.embed_dim))
        x = x + pos_embedding
        
        # Dropout
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        
        # Transformer layers
        for i in range(self.num_layers):
            # Layer norm + attention
            norm_x = nn.LayerNorm()(x)
            
            # Multi-head attention
            qkv = nn.Dense(3 * self.embed_dim)(norm_x)
            qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.embed_dim // self.num_heads)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            
            q = q.transpose(0, 2, 1, 3)  # [batch, heads, patches, head_dim]
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            
            # Attention
            attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.embed_dim // self.num_heads)
            attn = nn.softmax(attn, axis=-1)
            attn = nn.Dropout(self.dropout_rate)(attn, deterministic=not training)
            
            out = jnp.matmul(attn, v)
            out = out.transpose(0, 2, 1, 3).reshape(batch_size, num_patches, self.embed_dim)
            out = nn.Dense(self.embed_dim)(out)
            
            # Residual
            x = x + nn.Dropout(self.dropout_rate)(out, deterministic=not training)
            
            # Layer norm + MLP
            norm_x = nn.LayerNorm()(x)
            mlp = nn.Dense(self.mlp_dim)(norm_x)
            mlp = nn.gelu(mlp)
            mlp = nn.Dropout(self.dropout_rate)(mlp, deterministic=not training)
            mlp = nn.Dense(self.embed_dim)(mlp)
            
            # Residual
            x = x + nn.Dropout(self.dropout_rate)(mlp, deterministic=not training)
        
        # Final layer norm
        x = nn.LayerNorm()(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=1)  # [batch, embed_dim]
        
        # Classification head
        x = nn.Dense(self.num_classes)(x)  # [batch, 19]
        
        return x