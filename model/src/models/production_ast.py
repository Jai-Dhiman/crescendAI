#!/usr/bin/env python3
"""
Production AST model matching the exact parameter structure from your pickle file
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Any


class ProductionAST(nn.Module):
    """AST model matching your trained parameter structure"""
    patch_size: int = 16
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout_rate: float = 0.1
    num_classes: int = 19
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        """Forward pass matching your trained model structure"""
        batch_size, time_frames, freq_bins = x.shape
        
        # Patch embedding - matches 'patch_embedding' in your params
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
        
        # Patch embedding Dense layer
        x = nn.Dense(self.embed_dim, name='patch_embedding')(x)
        
        # Positional embedding - matches 'pos_embedding' shape (1, 64, 768)
        num_patches = time_patches * freq_patches
        pos_embedding = self.param('pos_embedding',
                                 nn.initializers.normal(stddev=0.02),
                                 (1, num_patches, self.embed_dim))
        x = x + pos_embedding
        
        # Dropout
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        
        # Transformer layers - matching your layer naming convention
        for i in range(self.num_layers):
            # Layer norm 1 - matches 'norm1_layer{i}'
            norm1_x = nn.LayerNorm(name=f'norm1_layer{i}')(x)
            
            # Multi-head attention - matches 'attention_layer{i}' structure
            q = nn.Dense(self.embed_dim, name=f'attention_layer{i}_query')(norm1_x)
            k = nn.Dense(self.embed_dim, name=f'attention_layer{i}_key')(norm1_x)  
            v = nn.Dense(self.embed_dim, name=f'attention_layer{i}_value')(norm1_x)
            
            # Reshape for multi-head attention
            q = q.reshape(batch_size, num_patches, self.num_heads, self.embed_dim // self.num_heads)
            k = k.reshape(batch_size, num_patches, self.num_heads, self.embed_dim // self.num_heads)
            v = v.reshape(batch_size, num_patches, self.num_heads, self.embed_dim // self.num_heads)
            
            q = q.transpose(0, 2, 1, 3)  # [batch, heads, patches, head_dim]
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)
            
            # Attention computation
            attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / jnp.sqrt(self.embed_dim // self.num_heads)
            attn = nn.softmax(attn, axis=-1)
            attn = nn.Dropout(self.dropout_rate)(attn, deterministic=not training)
            
            out = jnp.matmul(attn, v)
            out = out.transpose(0, 2, 1, 3).reshape(batch_size, num_patches, self.embed_dim)
            
            # Output projection - matches 'attention_layer{i}_out'
            attn_out = nn.Dense(self.embed_dim, name=f'attention_layer{i}_out')(out)
            
            # Residual connection
            x = x + nn.Dropout(self.dropout_rate)(attn_out, deterministic=not training)
            
            # Layer norm 2 - matches 'norm2_layer{i}'
            norm2_x = nn.LayerNorm(name=f'norm2_layer{i}')(x)
            
            # MLP layers - matches 'mlp_dense1_layer{i}' and 'mlp_dense2_layer{i}'
            mlp1 = nn.Dense(self.mlp_dim, name=f'mlp_dense1_layer{i}')(norm2_x)
            mlp1 = nn.gelu(mlp1)
            mlp1 = nn.Dropout(self.dropout_rate)(mlp1, deterministic=not training)
            mlp2 = nn.Dense(self.embed_dim, name=f'mlp_dense2_layer{i}')(mlp1)
            
            # Residual connection
            x = x + nn.Dropout(self.dropout_rate)(mlp2, deterministic=not training)
        
        # Final layer norm - matches 'final_norm'
        x = nn.LayerNorm(name='final_norm')(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=1)  # [batch, embed_dim]
        
        # Regression head - matches 'regression_hidden' and 'regression_output'
        x = nn.Dense(256, name='regression_hidden')(x)  # Hidden layer
        x = nn.relu(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(self.num_classes, name='regression_output')(x)  # Final output
        
        return x


def remap_attention_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remap flat parameter names to nested structure expected by ProductionAST
    """
    remapped = {}
    
    for param_name, param_value in params.items():
        if param_name.startswith('attention_layer'):
            # Extract layer number and component
            parts = param_name.split('_')
            layer_num = parts[1][5:]  # Remove 'layer' prefix
            
            if len(parts) == 2:  # e.g., 'attention_layer0'
                # This contains subkeys like 'query', 'key', 'value', 'out'
                for subkey, subvalue in param_value.items():
                    new_name = f'attention_layer{layer_num}_{subkey}'
                    remapped[new_name] = subvalue
            else:
                remapped[param_name] = param_value
        else:
            remapped[param_name] = param_value
    
    return remapped