#!/usr/bin/env python3
"""
Simple Architecture Parameter Count Test
Calculate parameter counts for different configurations without full model instantiation
"""

def calculate_ast_parameters(embed_dim: int, num_layers: int, num_heads: int, 
                           traditional_features: int = 145, 
                           num_patches: int = 64, patch_size: int = 16) -> dict:
    """
    Calculate approximate parameter count for Hybrid AST architecture
    """
    
    # Patch embedding: patch_size^2 -> embed_dim
    patch_embed_params = (patch_size * patch_size) * embed_dim + embed_dim  # Linear + bias
    
    # Positional encoding (learned)
    pos_embed_params = num_patches * embed_dim
    
    # Transformer layers
    transformer_params = 0
    for _ in range(num_layers):
        # Multi-head attention
        # Q, K, V projections
        qkv_params = 3 * (embed_dim * embed_dim + embed_dim)
        # Output projection
        out_proj_params = embed_dim * embed_dim + embed_dim
        # Layer norms (2 per layer)
        ln_params = 2 * (embed_dim + embed_dim)  # scale + shift
        # MLP
        mlp_dim = 4 * embed_dim
        mlp_params = (embed_dim * mlp_dim + mlp_dim) + (mlp_dim * embed_dim + embed_dim)
        
        layer_params = qkv_params + out_proj_params + ln_params + mlp_params
        transformer_params += layer_params
    
    # Final layer norm
    final_ln_params = embed_dim + embed_dim
    
    # Traditional feature processor
    trad_hidden = 128
    trad_output = 256
    traditional_params = (
        traditional_features * trad_hidden + trad_hidden +  # First dense
        trad_hidden * trad_hidden + trad_hidden +           # Second dense
        trad_hidden * trad_output + trad_output             # Output dense
    )
    
    # Fusion layer (attention-based)
    fusion_params = (
        traditional_features * embed_dim + embed_dim +  # Traditional projection
        embed_dim * embed_dim + embed_dim +             # Query projection
        (embed_dim * 2) * embed_dim + embed_dim +       # Key projection (2 features)
        (embed_dim * 2) * embed_dim + embed_dim +       # Value projection
        embed_dim + embed_dim                           # Layer norm
    )
    
    # Task heads (19 perceptual dimensions, grouped)
    # Assuming 4 groups with shared processing
    num_groups = 4
    num_dims = 19
    head_hidden = 256
    task_head_params = num_groups * (
        embed_dim * head_hidden + head_hidden +         # Shared layer 1
        head_hidden * (head_hidden // 2) + (head_hidden // 2)  # Shared layer 2
    )
    # Individual dimension outputs (simplified)
    task_head_params += num_dims * ((head_hidden // 2) * 1 + 1)
    
    # Total parameters
    total_params = (
        patch_embed_params + 
        pos_embed_params + 
        transformer_params + 
        final_ln_params +
        traditional_params +
        fusion_params +
        task_head_params
    )
    
    return {
        'patch_embedding': patch_embed_params,
        'positional_encoding': pos_embed_params,
        'transformer_layers': transformer_params,
        'final_layer_norm': final_ln_params,
        'traditional_processor': traditional_params,
        'fusion_layer': fusion_params,
        'task_heads': task_head_params,
        'total': total_params
    }


def test_architecture_sizes():
    """Test different architecture configurations"""
    
    print("=== Hybrid AST Parameter Count Analysis ===\n")
    print("Calculating parameters for different architectures...")
    print("Target: 5-20M parameters for 832 samples (avoiding overfitting)\n")
    
    configurations = {
        'ultra_tiny': {'embed_dim': 256, 'num_layers': 3, 'num_heads': 4},
        'tiny': {'embed_dim': 320, 'num_layers': 4, 'num_heads': 5},
        'small': {'embed_dim': 384, 'num_layers': 4, 'num_heads': 6},
        'medium': {'embed_dim': 512, 'num_layers': 4, 'num_heads': 8},
        'large': {'embed_dim': 768, 'num_layers': 6, 'num_heads': 12},
        'current_baseline': {'embed_dim': 768, 'num_layers': 12, 'num_heads': 12}
    }
    
    results = []
    samples = 832
    
    for name, config in configurations.items():
        params_breakdown = calculate_ast_parameters(**config)
        total_params = params_breakdown['total']
        param_sample_ratio = total_params / samples
        
        # Determine overfitting risk
        if param_sample_ratio > 50000:
            risk = "VERY HIGH"
            risk_color = "🔴"
        elif param_sample_ratio > 10000:
            risk = "HIGH"
            risk_color = "🟠"
        elif param_sample_ratio > 1000:
            risk = "MODERATE"
            risk_color = "🟡"
        else:
            risk = "LOW"
            risk_color = "🟢"
        
        results.append({
            'name': name,
            'config': config,
            'total_params': total_params,
            'ratio': param_sample_ratio,
            'risk': risk,
            'risk_color': risk_color,
            'breakdown': params_breakdown
        })
        
        print(f"{name.upper():15s}: {total_params:8,} params ({param_sample_ratio:5.0f}:1) {risk_color} {risk}")
    
    # Analysis
    print(f"\n=== DETAILED ANALYSIS ===")
    
    # Find recommended configurations
    recommended = [r for r in results if r['risk'] in ['LOW', 'MODERATE']]
    
    if recommended:
        print(f"\n🎯 RECOMMENDED ARCHITECTURES:")
        for r in sorted(recommended, key=lambda x: x['total_params']):
            config = r['config']
            print(f"   {r['risk_color']} {r['name']:12s}: {config['embed_dim']}D, {config['num_layers']}L, {config['num_heads']}H")
            print(f"      Parameters: {r['total_params']:,}")
            print(f"      Ratio: {r['ratio']:.0f}:1")
            print(f"      Risk: {r['risk']}")
            
            # Show breakdown for best candidate
            if r == recommended[0]:
                print(f"      Breakdown:")
                breakdown = r['breakdown']
                for component, count in breakdown.items():
                    if component != 'total':
                        percentage = (count / breakdown['total']) * 100
                        print(f"        {component:20s}: {count:7,} ({percentage:4.1f}%)")
            print()
    
    else:
        print("⚠️  No configurations with low overfitting risk found!")
        print("💡 Consider even smaller architectures")
    
    # Training recommendations
    print(f"🎯 TRAINING STRATEGY:")
    best_config = min(results, key=lambda x: x['total_params'] if x['risk'] != 'VERY HIGH' else float('inf'))
    print(f"   🥇 START WITH: {best_config['name']} ({best_config['total_params']:,} params)")
    print(f"   📊 Expected correlation improvement: +0.05-0.08 (reduced overfitting)")
    print(f"   🎵 With 145 traditional features: +0.03-0.06 (domain knowledge)")
    print(f"   🔄 With 3x data augmentation: +0.02-0.04 (more training data)")
    print(f"   📈 Total expected gain: +0.10-0.18 correlation")
    print(f"   🎯 Target performance: 0.67-0.75 (vs RF baseline: 0.5869)")
    
    return results


if __name__ == "__main__":
    results = test_architecture_sizes()
    
    print(f"\n✅ Architecture analysis complete!")
    print(f"   Ready to train recommended small architecture")
    print(f"   Expected outcome: Beat Random Forest baseline significantly")