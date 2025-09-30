# CrescendAI Model - Two-Model Architecture (Evaluator + Tutor)

PyTorch Lightning implementation of the **Evaluator** component in CrescendAI's dual-model piano performance analysis system.

## 🏗️ **Two-Model Architecture Overview**

### **Evaluator** (This Repository)
- **Purpose**: Audio segments → multidimensional performance scores
- **Input**: 10-20s piano audio segments
- **Output**: 16 performance dimensions [0-1] + uncertainties + time-local hotspots
- **Technology**: PyTorch Lightning + Audio Spectrogram Transformer (AST)
- **Location**: `model/` (this directory)

### **Tutor** (RAG System)
- **Purpose**: Evaluator outputs → actionable, cited feedback
- **Input**: Performance scores + piece metadata + user context
- **Output**: Personalized practice suggestions with citations
- **Technology**: RAG (dense + sparse search) + instruction model
- **Location**: `web/` and `server/` directories

## 🎯 **Current Focus**: Evaluator Data Labeling & Training

## Data Flow (Complete System)

1. **Audio Ingestion** → 10-20s segments (8-16 bars where aligned)
2. **Human Labeling** → Targeted dimensions per dataset via Streamlit interface
3. **Evaluator Training** → PyTorch Lightning with masked multi-task loss
4. **Evaluator Inference** → Performance scores + uncertainties + hotspots
5. **Tutor RAG** → Retrieve relevant teaching content + generate feedback
6. **User Interface** → Actionable, cited practice recommendations

## 📊 **Performance Dimensions (16 Core)**

### Execution Dimensions
- `timing_stability`, `tempo_control`, `rhythmic_accuracy`
- `articulation_length`, `articulation_hardness` 
- `pedal_density`, `pedal_clarity`
- `dynamic_range`, `dynamic_control`, `balance_melody_vs_accomp`

### Musical Shaping
- `phrasing_continuity`, `expressiveness_intensity`, `energy_level`

### Timbre Dimensions  
- `timbre_brightness`, `timbre_richness`, `timbre_color_variety`

## 🏗️ **Project Structure**

```
model/                       # Evaluator implementation
├── src/                     # PyTorch Lightning codebase
│   ├── models/              # AST architecture + Lightning module
│   ├── data/                # Data loading & preprocessing
│   ├── training/            # Training utilities & callbacks
│   └── evaluation/          # Metrics & validation
├── labeling/                # Human annotation tools
│   └── quick_labeler.py     # Streamlit labeling interface
├── data/                    # Datasets and annotations
│   ├── manifests/           # JSONL segment manifests
│   ├── anchors/             # Low/mid/high exemplars per dimension
│   └── splits/              # Train/val/test splits
├── configs/                 # Training configurations
├── symbolic/                # VirtuosoNet + pseudo-label integration
└── train.py                 # Main training script
```

## 🚀 **Quick Start (Evaluator)**

```bash
# Setup environment
uv venv && source .venv/bin/activate
uv sync --extra labeling

# Label data
streamlit run labeling/quick_labeler.py

# Train evaluator model
python train.py --data_dir ./data --experiment_name evaluator_v1

# Monitor training
tensorboard --logdir ./logs
```

## 🔄 **Integration with Complete System**

### **Dataset Targeting Strategy**
- **MAESTRO/ASAP/MAPS**: Execution + phrasing dimensions
- **CCMusic/MusicNet/YouTube**: Timbre + interpretation anchors
- **VirtuosoNet Integration**: Bar/beat-aligned segments + symbolic pseudo-labels
- **MidiBERT-Piano**: Optional distillation for structural understanding

### **Evaluator → Tutor Data Contract**
```json
{
  "segment_scores": {"timing_stability": 0.62, "expressiveness": 0.41, ...},
  "uncertainties": {"timing_stability": 0.15, "expressiveness": 0.23, ...},
  "hotspots": [{"t0": 12.3, "t1": 15.7, "dims": ["timing_stability"], "severity": 0.8}],
  "piece_metadata": {"composer": "Chopin", "opus": "10 No. 1", "difficulty": 8},
  "provenance": {"model_version": "v1.2", "confidence": 0.87}
}
```

### **Tutor System (Separate Implementation)**
- **Knowledge Base**: Curated masterclasses, teacher docs, technique guides
- **Retrieval**: Hybrid search (dense + sparse) filtered by dimension/difficulty
- **Generation**: Small instruction model producing structured feedback
- **Citations**: Linked references to retrieved teaching materials

## 📈 **Training Pipeline**

### **Active Learning Workflow**
1. **Initial Labeling**: 1-2k mixed segments across datasets
2. **Baseline Training**: 30 epochs → evaluator v0
3. **Uncertainty Sampling**: MC-dropout to identify informative segments
4. **Model-Assisted Labeling**: Pre-fill UI with predictions for faster annotation
5. **Iterative Improvement**: Retrain → recalibrate → repeat

### **Multi-Modal Training** (Advanced)
- **Human Labels**: Primary supervision with masked multi-task loss
- **Pseudo-Labels**: VirtuosoNet-derived symbolic proxies (α=0.3 weight)
- **Distillation**: Optional MidiBERT-Piano structural embeddings (β=0.1 weight)

## 📊 **Evaluation Metrics**

### **Evaluator Performance**
- **Per-dimension MAE** + **Pearson correlation** by dataset
- **Calibration quality**: Before/after temperature scaling
- **Uncertainty quality**: Correlation with prediction errors

### **System-Level Goals**
- **Evaluator**: >0.7 correlation on key dimensions, <5s inference time
- **Tutor**: Actionable feedback with verified citations
- **Integration**: Real-time analysis → personalized practice recommendations

## 🎯 **Next Steps**

### **Immediate (Evaluator Focus)**
1. Complete human labeling workflow with Streamlit interface
2. Train baseline evaluator with masked multi-task learning
3. Implement MC-dropout uncertainty estimation
4. Set up active learning pipeline

### **Future Integration**
1. Deploy evaluator as inference service for web/iOS apps
2. Build RAG knowledge base with teaching materials
3. Develop tutor prompt engineering and citation system
4. Integrate complete pipeline: audio → scores → feedback

## 🔗 **Related Components**

- **Web App** (`../web/`): SvelteKit interface for audio upload + feedback display
- **iOS App** (`../ios/`): Native Swift app with real-time audio analysis  
- **Server** (`../server/`): Rust backend for evaluator inference + tutor RAG
- **Docs** (`../docs/`): System architecture and API specifications

Ready to train the evaluator and build the complete CrescendAI teaching system! 🎹🚀