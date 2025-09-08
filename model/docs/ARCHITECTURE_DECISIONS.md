# Architecture Decisions for Piano Perception Transformer

*Decision Log - Updated: 2025-01-03*

## **Core Architecture Decisions**

### **1. Transformer Architecture: Audio Spectrogram Transformer (AST)**

**Decision**: Implement pure Audio Spectrogram Transformer with SSAST pre-training

**Rationale**:

- AST achieves SOTA performance (0.485 mAP AudioSet, 95.6% ESC-50)
- Pure attention-based, no convolutions required
- Treats spectrograms as 16×16 patches (proven approach)
- SSAST provides 60.9% performance improvement via self-supervised pre-trainings

**Implementation Details**:

- **Input**: Mel-spectrograms (128 frequency bins × time frames)
- **Patch size**: 16×16 patches → 768-dimensional embeddings
- **Architecture**: 12-layer transformer encoder with multi-head self-attention
- **Pre-training**: SSAST on MAESTRO dataset (unlabeled piano audio)
- **Fine-tuning**: PercePiano dataset (19 perceptual dimensions)

### **2. Framework: JAX/Flax**

**Decision**: Continue with JAX/Flax implementation

**Rationale**:

- High-performance XLA compilation for transformer training
- Functional programming enables clean, composable architectures
- Research flexibility for attention mechanism experimentation
- Already established infrastructure in existing codebase

**Benefits**:

- Superior training speed vs PyTorch for large transformers
- Native TPU support via XLA compilation
- Easy experimentation with novel attention patterns
- Production-ready scaling capabilities
- Seamless Google Colab integration

### **3. Multi-task Learning Design: Shared Encoder + Grouped Heads**

**Decision**: Shared AST encoder with task-grouped regression heads

**Architecture**:

```
AST Encoder (Shared)
    ├── Timing Group Head
    ├── Dynamics Group Head (Dynamics, Articulation)  
    └── Expression Group Head (Musical Expression, Emotion, Interpretation)
```

**Rationale**:

- Leverages perceptual dimension correlations from dataset analysis
- Reduces parameter count while maintaining task-specific learning
- Enables cross-task attention and information sharing
- Follows 2024 SOTA multi-task transformer patterns

**Loss Design**:

- Weighted MSE across task groups
- Adaptive loss balancing during training
- Group-specific normalization strategies

### **4. Data Strategy: MAESTRO + PercePiano**

**Decision**: Use MAESTRO v3 for pre-training, PercePiano for fine-tuning

**MAESTRO v3** (Self-supervised pre-training):

- 200 hours virtuosic piano performances
- 3ms audio-MIDI alignment precision
- Unlabeled audio for SSAST masked patch modeling

**PercePiano** (Supervised fine-tuning):

- 1202 performances across 19 perceptual dimensions
- Expert-annotated perceptual ratings
- Target task-specific training data

**Augmentation Strategy**:

- Time-stretch (0.8x - 1.2x) from MAESTRO
- Pitch-shift (±2 semitones)
- Gaussian noise injection (low SNR)
- Mixup between performances

## **Implementation Timeline**

### **Weeks 1-2: AST Foundation**

- Implement AST architecture in JAX/Flax
- MAESTRO data pipeline for SSAST pre-training
- 16×16 patch embedding with positional encoding

### **Weeks 3-4: Multi-task Architecture**

- Grouped regression heads design
- Cross-task attention mechanisms
- Adaptive loss weighting implementation

### **Weeks 5-6: Training & Evaluation**

- SSAST pre-training on MAESTRO
- PercePiano fine-tuning pipeline
- Comprehensive evaluation framework

## **Technical Specifications**

### **Model Architecture**

- **Encoder**: 12 transformer layers, 768 hidden dimensions
- **Attention**: 12 heads, 64-dimensional per head
- **Patch Embedding**: Linear projection of 16×16 spectrogram patches
- **Positional Encoding**: 2D learnable for frequency-time structure
- **Dropout**: 0.1 throughout network

### **Training Configuration**

- **Platform**: Google Colab (GPU/TPU access due to local hardware constraints)
- **Compute**: TPU v3-8 for pre-training, GPU for fine-tuning and development
- **Optimizer**: AdamW (β₁=0.9, β₂=0.999, weight_decay=0.01)
- **Learning Rate**: Cosine scheduling (peak 1e-4, warmup 10%)
- **Batch Size**: 128 (TPU), 32 (GPU) with gradient accumulation
- **Pre-training**: 100 epochs on MAESTRO (~10-12 hours on TPU)
- **Fine-tuning**: 50 epochs on PercePiano (~2-3 hours on GPU)

### **Evaluation Metrics**

- **Primary**: Pearson correlation per perceptual dimension
- **Target**: >0.7 correlation on Timing, Dynamics, Musical Expression
- **Statistical**: Significance testing, confidence intervals
- **Comparative**: Baseline CNN, feedforward model comparisons

## **Research Extensions**

### **Phase 2 Directions** (Post-implementation)

1. **Interpretable Attention**: Visualize musical focus patterns
2. **Cross-Cultural Adaptation**: Transfer to non-Western music
3. **Few-Shot Learning**: Rapid adaptation to new instruments
4. **Real-Time Applications**: Efficient inference architectures

### **Publication Strategy**

- **Target Venues**: ISMIR, ICASSP, ICML workshops
- **Contributions**: Novel AST application to music perception
- **Insights**: What transformer attention reveals about musical structure
- **Impact**: Open-source codebase advancing music AI research

## **Training Infrastructure**

### **Google Colab Platform Strategy**

**Hardware Selection**:

- **SSAST Pre-training**: TPU v3-8 (optimal for large-scale self-supervised learning)
- **Fine-tuning**: GPU (T4/V100) for faster iteration and debugging
- **Development/Testing**: CPU for code development and small experiments

**TPU vs GPU Decision Matrix**:

```
Task                    | Hardware | Reasoning
------------------------|----------|------------------------------------------
SSAST Pre-training      | TPU      | Large batch sizes, long training time
PercePiano Fine-tuning  | GPU      | Smaller dataset, frequent checkpointing  
Architecture Testing    | GPU/CPU  | Fast iteration cycles
Attention Visualization | GPU      | Interactive analysis and plotting
```

**Colab-Specific Optimizations**:

- Automatic session management and checkpoint saving
- Google Drive integration for dataset storage
- Efficient memory management for 12GB GPU/32GB TPU limits
- Progressive loading for large datasets (MAESTRO)

## **Success Criteria**

### **Technical Performance**

- >0.7 correlation on primary perceptual dimensions
- Significant improvement over CNN baselines (>20%)
- Training convergence within 12 hours on TPU (pre-training), 3 hours on GPU (fine-tuning)
- Reproducible results with fixed random seeds across TPU/GPU

### **Research Impact**

- Publication-ready implementation and evaluation
- Novel insights into music perception via attention visualization
- Community-useful open-source contribution
- Graduate school portfolio demonstration of cutting-edge ML skills

---

*This architecture prioritizes research impact and technical excellence while leveraging Google Colab's TPU/GPU infrastructure for scalable training.*
