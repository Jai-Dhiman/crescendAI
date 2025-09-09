# Technical Implementation Plan: Architecture-First Optimization

*Based on senior ML engineering evaluation of model performance roadmap*

## Critical Problem Analysis

### Current State

- **Model**: 86M parameter AST achieving 0.5285 correlation
- **Baseline**: Random Forest achieving 0.5869 correlation  
- **Dataset**: 832 samples across 19 perceptual dimensions
- **Parameter-to-Sample Ratio**: 103,000:1 (catastrophic overfitting territory)

### Why Current Approach Fails

1. **Massive Overfitting**: 86M parameters on 832 samples violates all ML best practices
2. **Feature Gap**: RF uses hand-crafted musical features, AST learns from raw spectrograms
3. **Data Inefficiency**: Transformers need 10k+ samples, we have 832
4. **Single Model vs Ensemble**: RF is inherently ensemble (100+ trees), AST is single model

## Architecture-First Solution Strategy

### Phase 1: Ultra-Small Architecture Design (Week 1, Priority 1)

#### Target Architectures

```python
# Ultra-Tiny Configuration (5M parameters)
config_ultra_tiny = {
    'num_layers': 3,
    'embed_dim': 256, 
    'num_heads': 4,
    'patch_size': 16,
    'mlp_ratio': 2,  # Reduced from 4
    'dropout': 0.4,   # Aggressive regularization
}

# Tiny Configuration (10M parameters)  
config_tiny = {
    'num_layers': 4,
    'embed_dim': 320,
    'num_heads': 6, 
    'patch_size': 16,
    'mlp_ratio': 3,
    'dropout': 0.3,
}

# Small Configuration (15M parameters)
config_small = {
    'num_layers': 6,
    'embed_dim': 384,
    'num_heads': 8,
    'patch_size': 16, 
    'mlp_ratio': 3,
    'dropout': 0.2,
}
```

#### Expected Results

- **5M model**: Baseline for improvement direction  
- **10M model**: Sweet spot candidate, 80:1 parameter ratio
- **15M model**: Capacity ceiling before overfitting dominates

### Phase 2: Hybrid Feature Integration (Week 1, Priority 2)

#### Traditional Feature Strategy

- **MFCC Features**: 13 coefficients + derivatives = 39 features
- **Spectral Features**: centroid, rolloff, bandwidth, ZCR = 8 features  
- **Harmonic Features**: chroma, harmonic/percussive = 12 features
- **Temporal Features**: tempo, beat consistency = 6 features
- **Total**: ~65 traditional features (same as Random Forest uses)

### Phase 3: Correlation-Aware Loss Functions (Week 2, Priority 1)

#### Direct Correlation Optimization

```python
def pearson_correlation_loss(predictions, targets):
    """Directly optimize Pearson correlation"""
    pred_centered = predictions - jnp.mean(predictions, axis=0)
    target_centered = targets - jnp.mean(targets, axis=0)
    
    correlation = jnp.sum(pred_centered * target_centered, axis=0) / (
        jnp.sqrt(jnp.sum(pred_centered ** 2, axis=0)) * 
        jnp.sqrt(jnp.sum(target_centered ** 2, axis=0)) + 1e-8
    )
    return -jnp.mean(correlation)  # Minimize negative correlation
```

### Phase 4: Ensemble Strategy (Week 3, Priority 1)

#### Multiple Small Models > Single Large Model

- Train 5-10 models (5M-15M parameters each)
- Different seeds, architectures, augmentation strategies
- Ensemble prediction averaging
- Expected: +0.02-0.04 correlation improvement over single model

## Implementation Timeline

### Week 1: Architecture + Hybrid Features

1. **Days 1-2**: Implement 5M, 10M, 15M AST configurations
2. **Days 3-4**: Traditional feature extraction pipeline
3. **Days 5-7**: Hybrid model training and evaluation

### Week 2: Loss Functions + Training Optimization  

1. **Days 1-3**: Correlation-aware loss functions
2. **Days 4-5**: Small-dataset training strategies
3. **Days 6-7**: Hyperparameter optimization

### Week 3: Ensembles + Analysis

1. **Days 1-4**: Train ensemble of small models
2. **Days 5-7**: Error analysis and interpretability

## Realistic Performance Targets

- **Week 1**: >0.5869 correlation (beat Random Forest)
- **Week 2**: >0.60 correlation (significant improvement)
- **Week 3**: >0.62 correlation (strong result for 832 samples)

## Key Success Factors

1. **Drastically reduce model size**: 5-15M parameters maximum
2. **Immediate hybrid features**: Don't wait, RF advantage is domain knowledge
3. **Ensemble approach**: Multiple small models beat single large model
4. **Correlation-optimized training**: MSE is wrong objective

This approach directly addresses the overfitting problem while leveraging musical domain knowledge through traditional features.
