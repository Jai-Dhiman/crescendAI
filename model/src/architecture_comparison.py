#!/usr/bin/env python3
"""
Architecture Comparison for Ultra-Small Models
Test different small configurations to find optimal parameter count for 832 samples
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import argparse

from models.hybrid_ast import HybridAudioSpectrogramTransformer, create_hybrid_train_state


def count_parameters(params) -> int:
    """Count total parameters in model"""
    return sum(x.size for x in jax.tree.leaves(params))


def test_architecture_configurations() -> Dict[str, Dict]:
    """Test different small architecture configurations"""
    
    print("=== Ultra-Small Architecture Comparison ===\n")
    print("Testing architectures optimized for small datasets (832 samples)")
    print("Target: Find models with 5-20M parameters (avoiding 100:1 parameter:sample ratio)\n")
    
    # Architecture configurations to test
    configurations = {
        'ultra_tiny': {
            'embed_dim': 256,
            'num_layers': 3,
            'num_heads': 4,
            'target_params': '~5M',
            'description': 'Minimal overfitting risk'
        },
        'tiny': {
            'embed_dim': 320,
            'num_layers': 4,
            'num_heads': 5,
            'target_params': '~8M',
            'description': 'Sweet spot candidate'
        },
        'small': {
            'embed_dim': 384,
            'num_layers': 4,
            'num_heads': 6,
            'target_params': '~12M',
            'description': 'Moderate capacity'
        },
        'medium': {
            'embed_dim': 512,
            'num_layers': 4,
            'num_heads': 8,
            'target_params': '~18M',
            'description': 'Capacity ceiling test'
        },
        'current_baseline': {
            'embed_dim': 768,
            'num_layers': 12,
            'num_heads': 12,
            'target_params': '~86M',
            'description': 'Original (too large)'
        }
    }
    
    results = {}
    
    # Test each configuration
    for config_name, config in configurations.items():
        print(f"--- Testing {config_name.upper()} Configuration ---")
        print(f"Architecture: {config['embed_dim']}D, {config['num_layers']}L, {config['num_heads']}H")
        print(f"Expected: {config['target_params']} parameters")
        print(f"Purpose: {config['description']}")
        
        try:
            # Create model
            model = HybridAudioSpectrogramTransformer(
                embed_dim=config['embed_dim'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                fusion_strategy='attention',
                traditional_feature_dim=145,  # From enhanced features
                dropout_rate=0.3
            )
            
            # Initialize with dummy inputs
            rng = jax.random.PRNGKey(42)
            dummy_spec = jnp.ones((1, 128, 128))  # (batch, time, freq)
            dummy_trad = jnp.ones((1, 145))       # (batch, features)
            
            # Create training state
            state = create_hybrid_train_state(
                model, rng, dummy_spec.shape, dummy_trad.shape, learning_rate=1e-4
            )
            
            # Count parameters
            param_count = count_parameters(state.params)
            
            # Test forward pass
            predictions, attention_weights, fusion_weights = model.apply(
                state.params, dummy_spec, dummy_trad, training=False
            )
            
            # Calculate parameter-to-sample ratio
            samples = 832  # PercePiano dataset size
            param_sample_ratio = param_count / samples
            
            # Determine overfitting risk
            if param_sample_ratio > 50000:
                overfitting_risk = "VERY HIGH"
            elif param_sample_ratio > 10000:
                overfitting_risk = "HIGH"
            elif param_sample_ratio > 1000:
                overfitting_risk = "MODERATE"
            else:
                overfitting_risk = "LOW"
            
            # Store results
            results[config_name] = {
                'embed_dim': config['embed_dim'],
                'num_layers': config['num_layers'],
                'num_heads': config['num_heads'],
                'parameters': param_count,
                'param_sample_ratio': param_sample_ratio,
                'overfitting_risk': overfitting_risk,
                'description': config['description'],
                'forward_pass_success': True,
                'num_predictions': len(predictions),
                'attention_layers': len(attention_weights)
            }
            
            print(f"✅ Success!")
            print(f"   Actual parameters: {param_count:,}")
            print(f"   Parameter:sample ratio: {param_sample_ratio:.0f}:1")
            print(f"   Overfitting risk: {overfitting_risk}")
            print(f"   Predictions: {len(predictions)} dimensions")
            print(f"   Memory footprint: ~{param_count * 4 / 1024 / 1024:.1f} MB")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results[config_name] = {
                'error': str(e),
                'forward_pass_success': False
            }
        
        print()
    
    # Analysis and recommendations
    print("=== ANALYSIS & RECOMMENDATIONS ===\n")
    
    successful_configs = {k: v for k, v in results.items() if v.get('forward_pass_success', False)}
    
    if successful_configs:
        print("📊 Parameter Analysis:")
        for name, result in successful_configs.items():
            params = result['parameters']
            risk = result['overfitting_risk']
            print(f"   {name:15s}: {params:8,} params, {risk:10s} overfitting risk")
        
        # Find best candidates
        low_risk_configs = {k: v for k, v in successful_configs.items() 
                           if v['overfitting_risk'] in ['LOW', 'MODERATE']}
        
        if low_risk_configs:
            print(f"\n🎯 RECOMMENDED CONFIGURATIONS ({len(low_risk_configs)} candidates):")
            
            # Sort by parameter count
            sorted_configs = sorted(low_risk_configs.items(), key=lambda x: x[1]['parameters'])
            
            for name, result in sorted_configs:
                params = result['parameters']
                ratio = result['param_sample_ratio']
                print(f"   ✅ {name:12s}: {params:7,} params ({ratio:4.0f}:1 ratio) - {result['description']}")
            
            # Specific recommendations
            print(f"\n💡 SPECIFIC RECOMMENDATIONS:")
            print(f"   🥇 START WITH: {sorted_configs[0][0]} ({sorted_configs[0][1]['parameters']:,} params)")
            print(f"      Reason: Lowest overfitting risk while maintaining capacity")
            
            if len(sorted_configs) > 1:
                print(f"   🥈 BACKUP: {sorted_configs[1][0]} ({sorted_configs[1][1]['parameters']:,} params)")
                print(f"      Reason: Slightly higher capacity if first is too constrained")
            
            print(f"\n📈 EXPECTED IMPROVEMENTS:")
            print(f"   • Reduced overfitting: +0.05-0.08 correlation")
            print(f"   • Better generalization to validation set")
            print(f"   • Faster training and convergence")
            print(f"   • More stable gradients")
            
        else:
            print("⚠️  All configurations have high overfitting risk!")
            print("💡 Consider even smaller architectures or more aggressive regularization")
    
    # Training recommendations
    print(f"\n🏋️  TRAINING RECOMMENDATIONS:")
    print(f"   • Use aggressive dropout (0.3-0.5)")
    print(f"   • Apply strong weight decay (0.1-0.2)")
    print(f"   • Use smaller learning rates (1e-5 to 5e-5)")
    print(f"   • Enable early stopping with patience")
    print(f"   • Apply data augmentation (3x dataset expansion)")
    print(f"   • Use correlation-based validation (not loss-based)")
    
    return results


def compare_with_random_forest():
    """Provide context about Random Forest baseline"""
    print("\n🌲 RANDOM FOREST BASELINE CONTEXT:")
    print("   Current performance: 0.5869 correlation")
    print("   Advantages:")
    print("     • Hand-crafted musical features (50+ features)")
    print("     • Ensemble method (100+ trees)")
    print("     • Excellent small-data performance")
    print("     • No overfitting with 832 samples")
    print("   ")
    print("   🎯 Hybrid AST Strategy to Beat RF:")
    print("     • Ultra-small architecture (5-15M params)")
    print("     • 145 traditional features (more than RF)")
    print("     • Smart fusion (attention/gating)")
    print("     • Conservative data augmentation")
    print("     • Correlation-optimized loss function")
    print("   ")
    print("   Expected gain: +0.08-0.12 correlation → Target: 0.66-0.70")


def save_results(results: Dict, output_path: Path = Path("../results/architecture_comparison.json")):
    """Save architecture comparison results"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ultra-small architectures")
    parser.add_argument("--output", type=str, default="../results/architecture_comparison.json", 
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Run architecture comparison
    results = test_architecture_configurations()
    
    # Show Random Forest context
    compare_with_random_forest()
    
    # Save results
    save_results(results, Path(args.output))
    
    print(f"\n✅ Architecture comparison complete!")
    print(f"   Next step: Train recommended architecture with hybrid features + augmentation")
    print(f"   Expected outcome: Beat Random Forest baseline (0.5869 → 0.66+)")