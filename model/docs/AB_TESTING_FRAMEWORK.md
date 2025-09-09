# 🧪 Piano Perception Transformer - A/B Testing Framework

## Overview

Systematic experimentation framework to improve model performance from 0.5285 to 0.70+ correlation through rigorous A/B testing and evaluation.

## 🎯 A/B Test Categories (Updated Priority Order)

**CRITICAL**: Follow architecture-first approach - test ultra-small models before heavy augmentation

### 1. **Data & Augmentation Tests**

#### Test A1: Augmentation Strength

```python
# Variants to test
augmentation_configs = {
    'light': {
        'spec_augment_prob': 0.3,
        'mixup_prob': 0.2,
        'time_stretch_range': (0.95, 1.05),
        'pitch_shift_range': (-1, 1)
    },
    'medium': {  # Current proposal
        'spec_augment_prob': 0.5,
        'mixup_prob': 0.3,
        'time_stretch_range': (0.85, 1.15),
        'pitch_shift_range': (-2, 2)
    },
    'heavy': {
        'spec_augment_prob': 0.7,
        'mixup_prob': 0.5,
        'time_stretch_range': (0.8, 1.2),
        'pitch_shift_range': (-3, 3)
    }
}
```

#### Test A2: Augmentation Factor (Dataset Size)

```python
augmentation_factors = [2, 3, 5, 8, 10]  # 2x to 10x data
# Test: Does more augmented data always help, or is there a sweet spot?
```

#### Test A3: Data Split Strategy

```python
split_strategies = {
    'random': 'Random 70/15/15 split',
    'performer_split': 'No performer overlap between train/test',
    'composer_split': 'No composer overlap between train/test',
    'temporal_split': 'Earlier recordings in train, later in test',
    'stratified_difficulty': 'Balanced difficulty across splits'
}
```

### 2. **Architecture Tests**

#### Test B1: Ultra-Small Model Size Ablation (HIGHEST PRIORITY)

```python
model_configs = {
    'ultra_tiny': {'layers': 3, 'dim': 256, 'heads': 4},  # 5M params - NEW
    'tiny': {'layers': 4, 'dim': 320, 'heads': 6},        # 10M params - REVISED
    'small': {'layers': 6, 'dim': 384, 'heads': 8},       # 15M params - REVISED
    'medium': {'layers': 6, 'dim': 512, 'heads': 8},      # 20M params - CEILING
    # REMOVED: All 30M+ configs due to overfitting on 832 samples
}
```

#### Test B2: Patch Size Optimization

```python
patch_sizes = [8, 12, 16, 20, 24, 32]
# Hypothesis: Smaller patches capture finer temporal details
# Larger patches capture broader musical phrases
```

#### Test B3: Attention Mechanisms

```python
attention_types = {
    'vanilla': 'Standard multi-head attention',
    'local': 'Local attention with window size',
    'linear': 'Linear attention (Performer/Linformer)',
    'sparse': 'Sparse attention patterns',
    'multi_scale': 'Different patch sizes in parallel'
}
```

### 3. **Feature Engineering Tests**

#### Test C1: Traditional Feature Combinations

```python
feature_sets = {
    'spectral_only': ['spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth'],
    'mfcc_only': ['mfcc_0' to 'mfcc_12'],
    'temporal_only': ['tempo', 'onset_rate', 'beat_consistency'],
    'harmonic_only': ['chroma_features', 'harmonic_energy', 'key_strength'],
    'all_traditional': 'All engineered features',
    'top_k_selected': 'Feature selection based on Random Forest importance'
}
```

#### Test C2: Spectrogram Preprocessing

```python
spectrogram_configs = {
    'mel_128': {'n_mels': 128, 'fmin': 20, 'fmax': 8000},
    'mel_256': {'n_mels': 256, 'fmin': 20, 'fmax': 8000}, 
    'log_mel': {'scale': 'log', 'n_mels': 128},
    'cqt': {'constant_q_transform': True, 'bins_per_octave': 24},
    'multi_resolution': 'Combine mel + CQT features'
}
```

### 4. **Training Strategy Tests**

#### Test D1: Loss Function Comparison

```python
loss_functions = {
    'mse': 'Mean Squared Error (current)',
    'correlation': 'Direct correlation optimization',
    'ranking': 'Pairwise ranking loss',
    'combined': 'MSE + Correlation + Ranking weighted',
    'huber': 'Huber loss (robust to outliers)',
    'focal_mse': 'Focus on hard examples'
}
```

#### Test D2: Learning Rate Schedules

```python
lr_schedules = {
    'constant': 1e-4,
    'cosine': 'Cosine annealing',
    'warmup_cosine': 'Linear warmup + cosine',
    'step_decay': 'Step decay at epochs [50, 100, 150]',
    'reduce_on_plateau': 'Adaptive based on validation',
    'one_cycle': 'One cycle policy'
}
```

#### Test D3: Regularization Techniques

```python
regularization_configs = {
    'baseline': {'dropout': 0.1, 'weight_decay': 1e-4},
    'heavy_dropout': {'dropout': 0.3, 'weight_decay': 1e-4},
    'label_smoothing': {'label_smooth': 0.1, 'dropout': 0.1},
    'mixup_cutmix': {'mixup': 0.3, 'cutmix': 0.2},
    'spectral_norm': {'spectral_normalization': True},
    'stochastic_depth': {'stochastic_depth': 0.2}
}
```

### 5. **Advanced ML Techniques Tests**

#### Test E1: Ensemble Methods

```python
ensemble_strategies = {
    'single_model': 'Baseline single model',
    'bagging': '5 models with different random seeds',
    'boosting': 'Sequential training with error focus',
    'stacking': 'Meta-learner combining multiple models',
    'snapshot_ensemble': 'Multiple checkpoints from one training run',
    'test_time_augmentation': 'Average predictions over augmented inputs'
}
```

#### Test E2: Meta-Learning Approaches

```python
meta_learning_tests = {
    'maml': 'Model-Agnostic Meta-Learning',
    'few_shot': 'Few-shot adaptation to new performers/styles',
    'domain_adaptation': 'Adapt from MAESTRO to PercePiano',
    'continual_learning': 'Learn new dimensions without forgetting'
}
```

## 📊 Comprehensive Evaluation Framework

### 1. **Primary Metrics**

```python
primary_metrics = {
    'pearson_correlation': 'Overall and per-dimension Pearson correlation',
    'spearman_correlation': 'Rank correlation (robust to outliers)',
    'mse': 'Mean Squared Error',
    'mae': 'Mean Absolute Error', 
    'r2_score': 'R-squared coefficient of determination'
}
```

### 2. **Advanced Evaluation Metrics**

```python
advanced_metrics = {
    'dimension_weighted_correlation': 'Weight by dimension importance',
    'top_k_accuracy': 'Accuracy in ranking top performances',
    'distribution_matching': 'How well predicted distribution matches true',
    'confidence_calibration': 'Are model confidence scores meaningful?',
    'worst_case_performance': 'Performance on hardest 10% of samples'
}
```

### 3. **Diagnostic Evaluations**

```python
diagnostic_tests = {
    'learning_curves': 'Training/validation curves over time',
    'loss_landscape': 'Visualize loss surface around optima',
    'gradient_analysis': 'Gradient flow and magnitude analysis',
    'attention_analysis': 'What musical patterns does model attend to?',
    'feature_importance': 'Which input features matter most?',
    'error_analysis': 'Systematic analysis of failure cases'
}
```

### 4. **Robustness Tests**

```python
robustness_evaluations = {
    'noise_robustness': 'Performance under audio noise',
    'tempo_robustness': 'Different tempo variations',
    'cross_composer': 'Generalization to unseen composers',
    'cross_performer': 'Generalization to unseen performers',
    'cross_recording_quality': 'Different audio qualities',
    'adversarial_examples': 'Worst-case input perturbations'
}
```

### 5. **Statistical Significance Testing**

```python
significance_tests = {
    'paired_t_test': 'Compare two models on same test set',
    'bootstrap_ci': 'Bootstrap confidence intervals',
    'permutation_test': 'Non-parametric significance testing',
    'multiple_comparisons': 'Bonferroni correction for multiple tests',
    'effect_size': 'Cohen\'s d for practical significance'
}
```

## 🏗️ Experimental Design Best Practices

### 1. **Controlled Experiments**

```python
experiment_design = {
    'fixed_seed': 'Same random seed for fair comparison',
    'same_hardware': 'Consistent computational environment',
    'multiple_runs': 'Average over 5 independent runs',
    'same_data_splits': 'Identical train/val/test across experiments',
    'hyperparameter_budget': 'Equal compute budget for hyperparameter search'
}
```

### 2. **Tracking & Logging**

```python
tracking_requirements = {
    'wandb_logging': 'All metrics, hyperparameters, and artifacts',
    'code_versioning': 'Git commit hash for each experiment',
    'data_versioning': 'Track data preprocessing versions',
    'model_artifacts': 'Save all model checkpoints',
    'reproducibility': 'Docker container + requirements.txt'
}
```

### 3. **Validation Strategy**

```python
validation_strategies = {
    'holdout_test': '20% completely held out until final evaluation',
    'k_fold_cv': '5-fold cross-validation for robust estimates',
    'time_series_split': 'Respect temporal ordering in data',
    'nested_cv': 'Inner CV for hyperparameters, outer for performance',
    'leave_one_performer_out': 'Ultimate generalization test'
}
```

## 🎖️ ML Engineering Best Practices to Learn

### 1. **Experiment Management**

- **Hypothesis-driven experiments**: Always have a clear hypothesis
- **One variable at a time**: Change only one thing per experiment  
- **Power analysis**: Determine required sample size for significance
- **Early stopping criteria**: When to stop an unpromising experiment

### 2. **Model Development Workflow**

- **Baseline establishment**: Always establish strong baselines first
- **Ablation studies**: Understand which components contribute what
- **Error analysis**: Deep dive into failure cases systematically
- **Feature engineering**: Automated feature selection and engineering

### 3. **Production ML Practices**

- **Model monitoring**: Track performance degradation over time
- **A/B testing infrastructure**: Safely deploy model improvements
- **Feature stores**: Consistent feature computation across train/serve
- **Model explainability**: Understand and communicate model decisions

### 4. **Research Best Practices**

- **Literature integration**: Compare against published baselines
- **Reproducibility**: All experiments should be reproducible
- **Statistical rigor**: Proper significance testing and confidence intervals
- **Negative results**: Document what doesn't work and why

## 📈 Implementation Priority

### Phase 1 (Week 1): Architecture-First Foundation

1. **PRIORITY 1**: Ultra-small model size ablation (B1) - 5M, 10M, 15M params
2. **PRIORITY 2**: Hybrid feature integration (traditional + AST)
3. Set up evaluation framework and tracking
4. Conservative augmentation tests (A1) - support small models

### Phase 2 (Week 2): Training Optimization

1. **PRIORITY 1**: Correlation-aware loss functions (D1)
2. **PRIORITY 2**: Small-dataset regularization techniques (D3)
3. Feature combination ablation (C1)
4. Learning rate schedule optimization (D2)

### Phase 3 (Week 3): Advanced Techniques

1. Ensemble methods (E1)
2. Advanced regularization (D3)
3. Robustness evaluations
4. Statistical significance testing

### Phase 4 (Week 4): Production Readiness

1. Final model selection based on comprehensive evaluation
2. Cross-validation and error analysis
3. Model interpretability and documentation
4. Preparation for potential publication/deployment

This framework will teach you industry-standard ML practices while systematically improving your model performance!
