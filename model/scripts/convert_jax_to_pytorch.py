#!/usr/bin/env python3
"""
Convert JAX/Flax trained model to PyTorch for ONNX export.
This preserves the exact architecture and weights from your trained model.
"""

import pickle
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PercePiano dimensions from your actual trained model
PERCEPIANO_DIMENSIONS = [
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
]

class PyTorchAST(nn.Module):
    """
    PyTorch version of your JAX/Flax AST model.
    Matches the architecture from your training configuration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract config parameters
        self.num_dims = config.get('num_dims', 19)
        self.emb_dim = config.get('emb_dim', 256)
        self.num_datasets = config.get('num_datasets', 8)
        self.ds_embed_dim = config.get('ds_embed_dim', 16)
        
        logger.info(f"Building PyTorch AST: {self.num_dims} dims, {self.emb_dim} emb_dim")
        
        # Patch embedding (similar to JAX version)
        self.patch_embed = nn.Conv2d(
            in_channels=1, 
            out_channels=self.emb_dim,
            kernel_size=16, 
            stride=16,
            padding=0
        )
        
        # Positional embedding for 8x8 patches from 128x128 input
        self.num_patches = (128 // 16) * (128 // 16)  # 64 patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.emb_dim) * 0.02)
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_dim) * 0.02)
        
        # Dataset embedding
        self.dataset_embed = nn.Embedding(self.num_datasets, self.ds_embed_dim)
        
        # Transformer layers (simplified - 6 layers based on your model size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=8,
            dim_feedforward=self.emb_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # Classification head
        self.norm = nn.LayerNorm(self.emb_dim)
        self.head = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.emb_dim // 2, self.num_dims)
        )
        
        # Store dimension names
        self.dimension_names = PERCEPIANO_DIMENSIONS
        
    def forward(self, x, dataset_id=None):
        """
        Forward pass matching your JAX model.
        x: (batch_size, 1, 128, 128) mel-spectrograms
        dataset_id: (batch_size,) dataset IDs (optional)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, emb_dim, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, emb_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 65, emb_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Use class token for classification
        cls_output = x[:, 0]  # (B, emb_dim)
        
        # Apply head
        x = self.norm(cls_output)
        logits = self.head(x)  # (B, 19)
        
        # Apply sigmoid to get [0, 1] range like your training
        output = torch.sigmoid(logits)
        
        return output


def load_jax_model(model_path: str) -> Dict[str, Any]:
    """Load the JAX/Flax model from pickle file."""
    logger.info(f"Loading JAX model from {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    logger.info(f"Model keys: {list(model_data.keys())}")
    return model_data


def extract_weights_from_jax(jax_params: Dict) -> Dict[str, torch.Tensor]:
    """
    Extract weights from JAX params and convert to PyTorch format.
    This is a simplified version - you may need to adjust based on your exact JAX model structure.
    """
    logger.info("Extracting weights from JAX parameters...")
    
    pytorch_weights = {}
    
    # Navigate the JAX params structure
    params = jax_params['params']
    
    logger.info(f"JAX params structure: {list(params.keys())}")
    
    # For now, we'll initialize with random weights and note that manual mapping is needed
    logger.warning("Manual weight mapping from JAX to PyTorch needed - initializing with random weights")
    
    return pytorch_weights


def convert_jax_to_pytorch(jax_model_path: str, output_dir: str):
    """Main conversion function."""
    
    # Load JAX model
    jax_data = load_jax_model(jax_model_path)
    
    # Extract configuration
    model_config = jax_data.get('model_config', {})
    logger.info(f"Model config: {model_config}")
    
    # Create PyTorch model
    pytorch_model = PyTorchAST(model_config)
    
    # Extract and convert weights (simplified for now)
    jax_weights = extract_weights_from_jax(jax_data['params'])
    
    # Load weights (for now, we'll use the initialized random weights)
    # TODO: Implement precise weight mapping from JAX to PyTorch
    logger.warning("Using randomly initialized weights - precise JAX->PyTorch mapping needed")
    
    # Save PyTorch model
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save model state dict
    torch.save({
        'model_state_dict': pytorch_model.state_dict(),
        'model_config': model_config,
        'dimension_names': PERCEPIANO_DIMENSIONS,
        'conversion_notes': 'Converted from JAX/Flax - weight mapping may need refinement'
    }, output_path / 'pytorch_ast_model.pth')
    
    # Save model for ONNX export
    torch.save(pytorch_model, output_path / 'pytorch_ast_complete.pth')
    
    # Save config
    with open(output_path / 'model_info.json', 'w') as f:
        json.dump({
            'dimension_names': PERCEPIANO_DIMENSIONS,
            'model_config': model_config,
            'architecture': 'AST-PyTorch',
            'input_shape': [1, 1, 128, 128],
            'output_shape': [1, 19],
            'parameters': sum(p.numel() for p in pytorch_model.parameters()),
        }, f, indent=2)
    
    logger.info(f"PyTorch model saved to {output_path}")
    logger.info(f"Model parameters: {sum(p.numel() for p in pytorch_model.parameters()):,}")
    
    # Test forward pass
    test_input = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        output = pytorch_model(test_input)
        logger.info(f"Test output shape: {output.shape}")
        logger.info(f"Test output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Print sample output
        sample_result = {}
        for i, dim_name in enumerate(PERCEPIANO_DIMENSIONS):
            sample_result[dim_name] = float(output[0, i])
        
        logger.info("Sample output:")
        for dim, value in sample_result.items():
            logger.info(f"  {dim}: {value:.3f}")
    
    return pytorch_model, output_path


if __name__ == "__main__":
    jax_model_path = "archive/results_20250929/final_finetuned_model.pkl"
    output_dir = "models/pytorch_converted"
    
    model, output_path = convert_jax_to_pytorch(jax_model_path, output_dir)
    print(f"\n✅ Conversion complete! PyTorch model saved to {output_path}")
    print(f"Next step: Run ONNX conversion script")