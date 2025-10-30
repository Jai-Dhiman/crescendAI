# Piano Performance Evaluation - Deployment Roadmaps

**Analysis Date:** 2025-10-29
**Context:** Two scenarios for achieving production-ready piano performance evaluation

---

# Scenario 1: Unlimited Resources

## "Build the Research-Grade System"

**Goal:** Match or exceed research targets (r=0.68-0.75 overall, approaching professional evaluator reliability)

---

## Phase 1: Infrastructure & Team (Month 1)

### Team Assembly

- **1 ML Engineer (Full-time)** - Architecture implementation, training infrastructure
- **1 Music Technologist (Part-time)** - Data curation, pseudo-label validation
- **1 Annotation Manager (Part-time)** - Recruit and train expert annotators
- **8-12 Expert Annotators (Contract)** - Music graduates or professional pianists

**Cost:** $40-50K/month for team

### Infrastructure

- **4× A100 (80GB) GPUs** - Google Cloud or AWS
  - On-demand: ~$12/hour per GPU = $48/hour for 4
  - 1-year reserved: ~40% discount = ~$210K/year
  - Alternative: Use spot instances (~$8/hour per GPU)
- **Storage:** 5TB for datasets + checkpoints (~$250/month)
- **Compute budget:** $15-20K for full training pipeline

**Monthly infrastructure cost:** $15-20K (spot instances) or $35-40K (on-demand)

---

## Phase 2: Data Collection & Annotation (Months 2-6)

### 2.1 Dataset Expansion

**Existing datasets (free):**

- ✅ MAESTRO v3.0.0 (200 hours, 1,276 performances)
- ✅ ASAP (92 hours, 1,067 performances)
- ✅ MAPS (65 hours, 9 piano configurations)

**Additional datasets to license/collect:**

- **Chopin Competition recordings** (License: $5-10K)
- **Conservatory student recordings** (Partner with Juilliard/Curtis/Berklee)
- **Amateur pianist recordings** (YouTube Creative Commons, careful licensing)

**Target:** 500+ hours across all skill levels (beginner → virtuoso)

**Cost:** $10-15K for licensing

### 2.2 Expert Annotation Campaign

**Annotation Platform:**

- Custom web interface (build or use Label Studio)
- Audio playback + waveform visualization
- Optional score display
- 12 slider scales (0-100) per dimension
- Uncertainty checkbox per dimension
- Quality control dashboard

**Development cost:** $15-20K (outsource to web dev)

**Annotation Protocol:**

**Dimensions to annotate (12 total):**

*Technical (6):*

1. Note accuracy (pitch correctness)
2. Rhythmic precision (timing accuracy)
3. Dynamics control (volume variation)
4. Articulation quality (note separation)
5. Pedaling technique (sustain pedal usage)
6. Tone quality (timbre, color)

*Interpretive (6):*
7. Phrasing (musical sentence structure)
8. Expressiveness (emotional communication)
9. Musical interpretation (artistic choices)
10. Stylistic appropriateness (period/composer style)
11. Overall musicality (holistic musical quality)
12. Overall technical quality (holistic technical execution)

**Target annotations:**

- **2,500 segments** (20-30 seconds each)
- **5-8 raters per segment** (for reliability)
- **Total ratings:** 12,500-20,000 individual ratings
- **Time per segment:** 3-5 minutes (listen + rate)
- **Total annotation hours:** ~800-1,200 hours

**Annotator recruitment:**

- Graduate students in piano performance ($30/hour)
- Professional pianists ($50/hour)
- Mix: 60% grad students, 40% professionals
- **Average rate:** $36/hour

**Cost:** $30-45K for annotation

**Quality control:**

- 2-hour calibration training per annotator
- 10% anchor segments (rated by all)
- Weekly check-ins during annotation
- Remove outliers with ICC < 0.5
- Target inter-rater reliability: ICC(1,k) > 0.80

### 2.3 Dataset Stratification

Ensure diversity across:

- **Difficulty:** 20% beginner, 30% intermediate, 30% advanced, 20% virtuoso
- **Composers:** 40% Romantic (Chopin, Liszt, Rachmaninoff), 30% Classical (Mozart, Beethoven), 20% Baroque (Bach), 10% Contemporary
- **Recording quality:** 50% studio, 30% concert hall, 20% practice room
- **Performance quality:** Distribute evenly across score ranges (0-100)

**Total Phase 2 cost:** $55-80K
**Total Phase 2 time:** 5 months (parallel annotation + dataset collection)

---

## Phase 3: Architecture Implementation (Months 2-3, parallel with Phase 2)

### 3.1 Upgrade from MVP to Research Architecture

**Current MVP:** 112M parameters
**Research target:** 380-450M parameters

**Component upgrades:**

| Component | Current | Upgrade To | Params |
|-----------|---------|------------|--------|
| **Audio Encoder** | MERT-95M | MERT-330M | 330M |
| **MIDI Encoder** | MIDIBert (6 layers) | MIDIBert-Piano (12 layers) | 12M |
| **Fusion** | Cross-attention (1 layer) | Cross-attention (2 layers) | 15M |
| **Aggregation** | BiLSTM (2 layers) | Music Transformer (6 layers) | 25M |
| **MTL Head** | Shared (6 dims) | Shared (12 dims) | 1M |
| **Ensemble Specialists** | None | 6 models × 3 variants | 90M |
| **Total** | 112M | **473M** | - |

**Key architectural additions:**

1. **Music Transformer for hierarchical aggregation:**

   ```python
   # Replace BiLSTM with Music Transformer
   from x_transformers import RelativePositionBias

   class MusicTransformerAggregator(nn.Module):
       def __init__(self, dim=512, depth=6, heads=8):
           super().__init__()
           self.rel_pos_bias = RelativePositionBias(
               bidirectional=True,
               num_buckets=64,
               max_distance=512,
               heads=heads
           )
           self.transformer = TransformerEncoder(
               dim=dim,
               depth=depth,
               heads=heads,
               attn_bias=self.rel_pos_bias
           )
   ```

2. **Ensemble specialists for interpretive dimensions:**

   ```python
   # Train 3-5 variants per subjective dimension
   specialists = {
       'phrasing': [Model(seed=i) for i in range(3)],
       'expressiveness': [Model(seed=i) for i in range(3)],
       'interpretation': [Model(seed=i) for i in range(3)],
       'stylistic_appropriateness': [Model(seed=i) for i in range(3)],
       'musicality': [Model(seed=i) for i in range(3)],
       'overall_quality': [Model(seed=i) for i in range(3)],
   }

   # Ensemble prediction with temperature scaling
   def ensemble_predict(specialists, x, temperature=1.5):
       predictions = [model(x) for model in specialists]
       mean_pred = torch.stack(predictions).mean(0)
       return mean_pred / temperature  # Post-ensemble scaling
   ```

3. **Hierarchical temporal modeling:**
   - Note-level (3-5 sec) → BiLSTM
   - Phrase-level (10-30 sec) → Multi-head attention
   - Piece-level (full performance) → Music Transformer

4. **Uncertainty quantification:**
   - Aleatoric (inherent randomness): Heteroscedastic regression
   - Epistemic (model uncertainty): Ensemble variance
   - Combined: `σ_total = sqrt(σ_aleatoric² + σ_epistemic²)`

**Implementation time:** 6-8 weeks (full-time ML engineer)

**Cost:** $20-25K in engineering time

---

## Phase 4: Training Pipeline (Months 3-5)

### 4.1 Stage 1: Load MERT-330M Pre-trained Weights

**Action:** Load from HuggingFace (m-a-p/MERT-v1-330M)

**Time:** Instant (download ~1.3GB)

**Cost:** Free

### 4.2 Stage 2: Pseudo-Label Pre-training

**Dataset:**

- MAESTRO: 1,276 performances → 10,000-15,000 segments (20-30 sec)
- ASAP: 1,067 performances → 8,000-12,000 segments
- **Total:** 18,000-27,000 pseudo-labeled segments

**Pseudo-label improvements (vs. current MVP):**

1. **Reference-based scoring:**
   - Compare each performance to top-10 MAESTRO pieces (consensus "ideal")
   - Use DTW alignment to match corresponding passages
   - Score = similarity to reference ensemble

2. **Multi-feature heuristics:**
   - Note accuracy: Transcription F1 + onset detection
   - Rhythm: Grid alignment + tempo stability
   - Dynamics: Velocity range + variation + smoothness
   - Articulation: Legato ratio + note duration consistency
   - Pedaling: Sustain timing + harmony support score
   - Tone: Spectral balance + brightness + inharmonicity

3. **Confidence weighting:**
   - Assign confidence score to each pseudo-label
   - Use confident samples first (curriculum learning)

**Training config:**

- **GPU:** 4× A100 (80GB)
- **Batch size:** 64 (16 per GPU)
- **Gradient accumulation:** 2 → effective batch size = 128
- **Epochs:** 15-20
- **Time:** 3-5 days
- **Cost:** $3,456-5,760 (4 GPUs × 72-120 hours × $12/hour)

### 4.3 Stage 3: Expert Fine-tuning (Multi-task heads)

**Dataset:** 2,500 expert-labeled segments (80/10/10 train/val/test split)

- Train: 2,000 segments
- Val: 250 segments
- Test: 250 segments

**Training config:**

- **GPU:** 4× A100
- **Batch size:** 32 (8 per GPU)
- **Gradient accumulation:** 4 → effective batch size = 128
- **Epochs:** 50 (with early stopping)
- **Backbone LR:** 2e-5
- **Heads LR:** 1e-4
- **Time:** 4-7 days
- **Cost:** $4,608-8,064

**Active learning:**
After initial 500 labels, use active learning to select next 2,000:

- Select high-uncertainty samples
- Prioritize underrepresented segments (difficulty, composer)
- Reduces total labels needed by 40-50%

### 4.4 Stage 4: Ensemble Specialist Training

**For each of 6 interpretive dimensions:**

- Train 3 variants with different seeds
- Use 80%-overlapping training splits
- Fine-tune from Stage 3 checkpoint

**Training config:**

- **Dimensions:** 6 (phrasing, expressiveness, interpretation, stylistic, musicality, overall)
- **Variants per dimension:** 3
- **Total models:** 18
- **Time per model:** 2-3 days
- **Parallelization:** Train 4 at once (1 GPU each)
- **Total time:** 9-14 days
- **Cost:** $10,368-16,128

### 4.5 Stage 5: Calibration

**Temperature scaling:**

- Hold out 15% of expert labels for calibration
- Optimize temperature parameter per dimension
- Use Expected Calibration Error (ECE) as metric

**Time:** 1 day
**Cost:** $288

**Total Phase 4 cost:** $18,720-30,240 (compute only)
**Total Phase 4 time:** 3-4 weeks training + 2 weeks experimentation

---

## Phase 5: Evaluation & Validation (Month 6)

### 5.1 Internal Validation

**Test on held-out expert labels (250 segments):**

Target metrics:

- **Technical dimensions:** Pearson r > 0.70, MAE < 8 points
- **Interpretive dimensions:** Pearson r > 0.60, MAE < 10 points
- **Overall weighted correlation:** r > 0.68
- **Pairwise ranking accuracy:** > 80%
- **Calibration (ECE):** < 0.10

### 5.2 Cross-Dataset Validation

**Test on PercePiano dataset (external validation):**

- 1,202 segments with expert ratings
- Different annotator pool (53 experts)
- Different annotation protocol
- Expected performance drop: 10-15%

**Target:** r > 0.55 on technical dimensions

### 5.3 Ablation Studies

Test contribution of each component:

1. Audio-only vs. multi-modal (expected: +15-20% from MIDI)
2. With/without pre-training (expected: +20-30%)
3. With/without augmentation (expected: +10-15%)
4. With/without ensemble (expected: +15-20% on interpretive)
5. MERT-95M vs. MERT-330M (expected: +5-10%)

**Time:** 2 weeks (retraining variants)
**Cost:** $5,000-8,000

### 5.4 Human Comparison Study

**Compare model to human experts:**

- 10 expert pianists rate same 100 test segments
- Compare model ratings to:
  - Individual experts (ICC target: > 0.50)
  - Consensus (average of 10 experts, ICC target: > 0.75)

**Cost:** $3,000-5,000 (expert time)

**Total Phase 5 cost:** $8,000-13,000
**Total Phase 5 time:** 1 month

---

## Phase 6: Production Deployment (Month 7-8)

### 6.1 Model Optimization

**Distillation (optional):**

- Train smaller student model (100M params) from 473M teacher
- Target: 90-95% of teacher performance
- 3-5× inference speedup

**Quantization:**

- INT8 quantization using PyTorch or TensorRT
- 40-50% memory reduction
- Minimal accuracy loss (<2%)

**Time:** 2 weeks
**Cost:** $2,000-3,000 (compute)

## Expected Performance: Unlimited Resources

| Metric | Target | Likelihood |
|--------|--------|------------|
| **Overall correlation (r)** | 0.68-0.75 | 80% likely |
| **Technical dimensions (r)** | 0.70-0.85 | 90% likely |
| **Interpretive dimensions (r)** | 0.60-0.68 | 70% likely |
| **MAE (0-100 scale)** | < 8 points | 80% likely |
| **Pairwise ranking accuracy** | > 80% | 85% likely |
| **Match single expert** | ICC > 0.50 | 95% likely |
| **Approach expert consensus** | ICC > 0.75 | 60% likely |

**Outcome:** Professional-grade system suitable for conservatory education, competition pre-screening, and pedagogical applications.

---

---

# Scenario 2: Bootstrapped Solo

## "Build an MVP That Actually Works"

**Goal:** Achieve useful performance (r=0.50-0.60 on technical dimensions) with minimal cost, validate end-to-end training, then scale strategically

**Constraints:**

- Solo developer (you)
- Limited budget (~$5-10K for compute + crowdsourcing)
- Unlimited time and effort
- Access to cloud compute (GCP, AWS, or RunPod)

---

## Philosophy: "Labels Are the Bottleneck, Not Compute"

**Key insight:** You need 200-300 expert labels minimum. Since hiring experts is expensive ($30-50/hour), your strategy should be:

1. **Self-annotate** first 50-100 segments (free, establishes baseline)
2. **Recruit musician friends** for 50-100 more (beer/coffee/favor economy)
3. **Use crowdsourcing** for remaining 100-200 (cheap but needs quality control)
4. **Active learning** to maximize label efficiency

**This gets you to 200-300 labels for under $2,000** vs. $30-45K in the unlimited scenario.

---

## Phase 1: Validation - Complete Current Training (Week 1)

**Goal:** Verify end-to-end training works on Colab

### 1.1 Current Training

- ✅ Let current MERT-95M training complete (20 epochs, ~5 hours total)
- ✅ Monitor for early stopping
- ✅ Checkpoint best model (currently epoch 6, val_loss=4.37)

**Cost:** $0 (Colab free tier) or $10 (Colab Pro)

### 1.2 Post-Training Analysis

Run diagnostics on completed model:

```bash
# After training completes
python scripts/diagnose_training.py \
  --config /tmp/colab_config.yaml \
  --checkpoint /path/to/best.ckpt
```

**Check:**

- Per-dimension Pearson correlations (target: r > 0.3 on any dimension)
- Which dimensions learned well vs. poorly
- Prediction variance (are predictions diverse or near-constant?)

**Decision point:**

- If r > 0.3 on 3+ dimensions → **Proceed to Phase 2**
- If r < 0.2 on all dimensions → **Pseudo-labels are too weak, skip to self-annotation**

**Time:** 1 day
**Cost:** $0-10

---

## Phase 2: Self-Annotation - Bootstrap Initial Dataset (Weeks 2-4)

**Goal:** Create 50-100 high-quality expert labels (yourself)

### 2.1 Recording Collection

**Sources (all free):**

1. **Your own performances** (record 10-20 pieces at varying quality levels)
2. **YouTube Creative Commons** piano performances
3. **Internet Archive** public domain recordings
4. **University music department** recordings (ask for permission)
5. **IMSLP** (International Music Score Library Project) - some have recordings

**Target diversity:**

- Difficulty: 30% beginner, 40% intermediate, 30% advanced
- Quality: 20% excellent, 40% good, 30% okay, 10% poor
- Composers: Varied (Bach, Beethoven, Chopin, etc.)

**Segmentation:**

- 20-30 second segments (matches research)
- Extract ~5-10 segments per full recording
- **Total:** 50-100 segments

**Time:** 1 week (recording + collecting)
**Cost:** $0

### 2.2 Annotation Interface

**Option A: Google Sheets (quick & dirty):**

```
Columns:
- audio_path
- start_time, end_time
- note_accuracy (0-100)
- rhythmic_precision (0-100)
- dynamics_control (0-100)
- articulation (0-100)
- pedaling (0-100)
- tone_quality (0-100)
- [4 interpretive dimensions if desired]
- notes (free text)
```

**Option B: Label Studio (better UX):**

```bash
# Install locally
pip install label-studio
label-studio start

# Import your audio files
# Create labeling interface with sliders
```

**Time:** 3 hours setup
**Cost:** $0

### 2.3 Self-Annotation Process

**Protocol:**

1. Listen to segment 2-3 times
2. Rate each dimension on 0-100 scale
3. Take notes on specific issues
4. Rate 5-10 segments per session (avoid fatigue)
5. Re-rate 10% after 1 week (check consistency)

**Dimensions to focus on (start with 6, add interpretive later):**

- Note accuracy (easiest - count mistakes)
- Rhythmic precision (timing deviations)
- Dynamics control (volume variation)
- Articulation (note separation)
- Pedaling (sustain pedal usage)
- Tone quality (timbre, harshness)

**Time investment:**

- 5 minutes per segment
- 50-100 segments = 4-8 hours total
- Spread over 2 weeks (to avoid burnout)

**Quality check:**

- Re-rate 10 random segments after 1 week
- Check consistency: MAE < 10 points
- If inconsistent, refine rubric and re-rate

**Time:** 2 weeks (part-time)
**Cost:** $0 (your time)

---

## Phase 3: Recruit Friends - Expand Dataset (Weeks 5-6)

**Goal:** Get 50-100 more labels from musician friends

### 3.1 Recruiter Strategy

**Target:**

- Fellow music students
- Local piano teachers
- Online music communities (Reddit r/piano, PianoWorld forums)
- University music department (offer to share results for their research)

**Pitch:**

- "Help validate my ML model for piano assessment"
- "30-60 minutes of your time"
- "Your feedback will improve music education"
- Offer: Coffee/beer/wine, or small gift card ($10-20)

**Target:** 3-5 musician friends

### 3.2 Annotation Session

**In-person (best):**

- Sit together, play segments
- Walk through rubric (10 min calibration)
- They annotate 10-20 segments (~30-60 min)
- You take notes on their thought process

**Remote:**

- Send them Label Studio link or Google Sheet
- 15-min Zoom calibration call
- They annotate asynchronously
- Follow-up call to discuss challenging cases

### 3.3 Inter-Rater Reliability

Calculate ICC (Intraclass Correlation Coefficient):

```python
# After collecting ratings from multiple annotators
from pingouin import intraclass_corr

# Shape: (n_segments, n_raters)
ratings = np.array([...])  # Your ratings + friends' ratings

icc = intraclass_corr(
    data=ratings_df,
    targets='segment_id',
    raters='annotator_id',
    ratings='score'
)

# Target: ICC(1,k) > 0.70
print(f"ICC: {icc['ICC'].iloc[2]}")  # ICC(1,k)
```

**If ICC < 0.70:**

- Revise rubric with concrete examples
- Re-calibrate with group discussion
- Remove outlier annotators

**Expected outcome:**

- 50-100 segments with 2-4 raters each
- Average ratings per segment
- Total unique segments: 100-150 (with some overlap for reliability)

**Time:** 2 weeks (scheduling + annotation)
**Cost:** $50-150 (gift cards/thank you gifts)

---

## Phase 4: Crowdsourcing - Scale to 200-300 Labels (Weeks 7-10)

**Goal:** Use cheap crowdsourcing with quality control to reach 200-300 total labels

### 4.1 Crowdsourcing Platform

**Option A: Amazon Mechanical Turk (MTurk)**

- Pros: Large worker pool, cheap ($0.05-0.10 per rating)
- Cons: Lower quality, needs strong quality control

**Option B: Prolific**

- Pros: Higher quality workers, pre-screening available
- Cons: 2-3× more expensive than MTurk (~$0.20-0.30 per rating)

**Recommendation:** Start with **Prolific**, fall back to MTurk with quality control

### 4.2 Task Design

**Qualification screening:**

- "Do you have formal music training?" (Yes/No)
- "How many years of piano lessons?" (0, 1-3, 4-7, 8+)
- "Rate this sample performance" (golden standard, must score within ±15 points)

**Filter:** Only workers with 4+ years piano lessons, passing golden standard

**Main task:**

- Rate 5 audio segments (20-30 sec each)
- 6 dimensions per segment (0-100 sliders)
- 3-5 minutes per segment
- Pay: $3-5 per HIT (5 segments)
- Effective rate: $36-60/hour (competitive)

### 4.3 Quality Control

**Honeypots (10% of segments):**

- Include segments you've rated with high confidence
- Compare worker rating to your rating
- Reject if MAE > 20 points on 3+ dimensions

**Redundancy:**

- 3-5 workers per segment
- Take median rating (robust to outliers)
- Flag segments with high variance (std > 20) for review

**Iterative filtering:**

- Track per-worker accuracy against honeypots
- Block workers with <70% pass rate
- Bonus reliable workers ($1-2 extra)

### 4.4 Budget Calculation

**Scenario: 200 segments, 3 raters each**

- Total ratings: 600
- Cost per rating: $0.50 (after platform fees)
- **Total: $300**

**Scenario: 300 segments, 3 raters each**

- Total ratings: 900
- **Total: $450**

**Quality failures (expect 20% rejection):**

- Add 20% buffer: $360-540

**Total crowdsourcing cost:** $400-600

**Time:** 3-4 weeks (iterative batches)
**Cost:** $400-600

---

## Phase 5: Fine-tune on Expert Labels (Weeks 11-12)

**Goal:** Train MERT-95M on your 200-300 expert labels (Stage 3 fine-tuning)

### 5.1 Prepare Dataset

**Convert annotations to JSONL format:**

```python
# scripts/prepare_expert_annotations.py
import json

annotations = []
for segment in your_annotations:
    annotations.append({
        "audio_path": segment['audio_path'],
        "midi_path": None,  # Optional
        "start_time": segment['start_time'],
        "end_time": segment['end_time'],
        "labels": {
            "note_accuracy": segment['note_accuracy'],
            "rhythmic_precision": segment['rhythmic_precision'],
            # ... other dimensions
        }
    })

# Split 80/10/10
train = annotations[:int(0.8*len(annotations))]
val = annotations[int(0.8*len(annotations)):int(0.9*len(annotations))]
test = annotations[int(0.9*len(annotations)):]

# Save
with open('expert_labels_train.jsonl', 'w') as f:
    for item in train:
        f.write(json.dumps(item) + '\n')
# ... same for val, test
```

**Dataset size:**

- Train: 160-240 segments
- Val: 20-30 segments
- Test: 20-30 segments

### 5.2 Training Configuration

**Start from Stage 2 checkpoint (best pseudo-label model):**

```yaml
# configs/expert_finetune_bootstrapped.yaml

data:
  train_path: data/annotations/expert_labels_train.jsonl
  val_path: data/annotations/expert_labels_val.jsonl
  test_path: data/annotations/expert_labels_test.jsonl
  dimensions:
    - note_accuracy
    - rhythmic_precision
    - dynamics_control
    - articulation
    - pedaling
    - tone_quality

  batch_size: 4  # Smaller dataset
  augmentation:
    enabled: true  # Critical for small dataset
    # ... same augmentation config

model:
  # Same as Stage 2 (MERT-95M, 112M params)

training:
  max_epochs: 100  # More epochs for small dataset
  backbone_lr: 5e-6  # Lower LR (preserve pseudo-label pre-training)
  heads_lr: 5e-5    # Lower than Stage 2

  # Key: Strong regularization for small dataset
  weight_decay: 0.02  # Higher than Stage 2
  gradient_clip_val: 0.5  # Tighter clipping
  accumulate_grad_batches: 8  # Effective batch size = 32

callbacks:
  early_stopping:
    patience: 10  # More patience (loss will be noisy)
    min_delta: 0.01
  checkpoint:
    save_top_k: 5

logging:
  log_every_n_steps: 10  # Log more frequently
```

### 5.3 Training Options

**Option A: Continue on Colab**

- Pros: Free (or $10/month Pro), already set up
- Cons: Session limits (12 hours free, 24 hours Pro)
- **Strategy:** Train in 8-hour sessions, checkpoint frequently

**Option B: Use cheap cloud GPU**

**RunPod (cheapest):**

- RTX 4090 (24GB): $0.39/hour
- A40 (48GB): $0.59/hour
- A100 (80GB): $1.39/hour (cheapest A100 available)

**Cost estimate:**

- Training time: ~20-40 hours (100 epochs, small dataset)
- GPU: RTX 4090 ($0.39/hour) is sufficient for MERT-95M
- **Total: $8-16**

**Google Cloud Spot Instances:**

- V100 (16GB): ~$0.50/hour (spot pricing, preemptible)
- **Total: $10-20**

**Recommendation:** Use RunPod RTX 4090 ($0.39/hour) - best price/performance

### 5.4 Setup RunPod (if chosen)

```bash
# 1. Sign up at runpod.io
# 2. Add credit ($10-20)
# 3. Deploy pod:
#    - Template: PyTorch 2.0
#    - GPU: RTX 4090 (24GB)
#    - Volume: 50GB persistent storage
#    - Cost: $0.39/hour

# 4. SSH into pod
ssh root@<pod-ip>

# 5. Clone repo and install
git clone https://github.com/YOUR_USERNAME/crescendai.git
cd crescendai/model
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install -e .

# 6. Copy data from Google Drive (use rclone)
rclone copy gdrive:piano_eval_data/ data/ --progress

# 7. Train
python train.py --config configs/expert_finetune_bootstrapped.yaml \
  --checkpoint checkpoints/pseudo_pretrain/best.ckpt
```

**Time:** 1-2 days (including setup + training)
**Cost:** $8-20

---

## Phase 6: Evaluation & Iteration (Week 13-14)

### 6.1 Evaluate on Test Set

**Metrics to check:**

```python
# After training completes
python scripts/evaluate_model.py \
  --checkpoint checkpoints/expert_finetune/best.ckpt \
  --test-data data/annotations/expert_labels_test.jsonl

# Expected output:
# ===============================
# TEST SET RESULTS (20-30 segments)
# ===============================
#
# Overall:
#   MAE: 12.5 ± 3.2
#   Pearson r: 0.48 ± 0.12
#   Spearman ρ: 0.52 ± 0.10
#
# Per-dimension:
#   note_accuracy:       r=0.62, MAE=10.2
#   rhythmic_precision:  r=0.58, MAE=11.5
#   dynamics_control:    r=0.45, MAE=13.8
#   articulation:        r=0.41, MAE=14.2
#   pedaling:            r=0.38, MAE=15.1
#   tone_quality:        r=0.35, MAE=16.3
```

**Success criteria (200-300 labels, MERT-95M):**

- ✅ **r > 0.50** on 2-3 dimensions (note accuracy, rhythm)
- ✅ **MAE < 15** points overall
- ✅ **r > 0.40** on 4+ dimensions

**If you hit these targets:** Your model is **useful** (better than random, approaching single-expert reliability)

### 6.2 Qualitative Analysis

**Listen to predictions:**

```python
# scripts/analyze_predictions.py

# 1. Sort test segments by error
segments_by_error = sorted(test_results, key=lambda x: x['mae'], reverse=True)

# 2. Listen to worst 10 predictions
for segment in segments_by_error[:10]:
    print(f"Segment: {segment['audio_path']}")
    print(f"  True: {segment['true_labels']}")
    print(f"  Pred: {segment['pred_labels']}")
    print(f"  MAE: {segment['mae']}")
    # Play audio manually

# 3. Identify patterns:
#    - Struggles with beginner vs. advanced?
#    - Confused by specific composers?
#    - Recording quality issues?
```

**Common failure modes:**

- Over-predicts quality (model is optimistic)
- Under-predicts quality (model is harsh)
- Conflates dimensions (dynamics ↔ tone quality)
- Biased by recording quality
- Struggles with specific styles (e.g., Baroque ornamentations)

### 6.3 Iteration Strategy

**If r < 0.40 on all dimensions:**

1. **Add 50-100 more labels** (focus on failure modes)
2. **Simplify to 3-4 easiest dimensions** (note accuracy, rhythm, dynamics)
3. **Use audio-only** (drop MIDI if causing issues)
4. **Try lower learning rate** (1e-6 backbone, 2e-5 heads)

**If r = 0.40-0.50:**

1. **Add 50-100 targeted labels** (active learning - high uncertainty samples)
2. **Improve crowdsourcing quality** (stricter worker filtering)
3. **Experiment with architectures** (try different aggregators)

**If r > 0.50:**

1. **You're on track!** Scale to 500+ labels with crowdsourcing
2. **Add interpretive dimensions** (4 more: phrasing, expression, interpretation, overall)
3. **Consider upgrading to MERT-330M** (see Phase 7)

**Time:** 1 week
**Cost:** $0 (analysis) + $200-400 (if adding labels)

---

## Phase 7: Scale to MERT-330M (Optional, Weeks 15-18)

**Goal:** If Phase 6 shows promise (r > 0.50), upgrade to research-scale model

### 7.1 Decision Point

**Upgrade to MERT-330M if:**

- ✅ r > 0.50 on 3+ dimensions with MERT-95M
- ✅ You have 300+ expert labels
- ✅ Willing to spend $100-200 on training

**Stay with MERT-95M if:**

- r < 0.40 (fix data quality first)
- Budget constrained (<$50 remaining)
- Just want to validate end-to-end pipeline

### 7.2 Architecture Upgrade

**Changes from MERT-95M to MERT-330M:**

```python
# configs/expert_finetune_mert330m.yaml

model:
  mert_model_name: m-a-p/MERT-v1-330M  # Changed from MERT-v1-95M
  audio_dim: 1024  # MERT-330M output dim (was 768)

  # Increase other components proportionally
  fusion_dim: 1536  # 1.5× larger
  aggregator_dim: 768  # 1.5× larger

  # Rest stays same
  midi_dim: 256
  num_dimensions: 6

training:
  batch_size: 4  # Might need to reduce to 2 for memory
  accumulate_grad_batches: 16  # Increase to maintain effective batch size
  gradient_checkpointing: true  # Critical for memory
```

**New model size:** ~380M parameters (3.4× larger)

### 7.3 Training on Cloud GPU

**RunPod options:**

| GPU | VRAM | Price/hour | Fits MERT-330M? | Recommended |
|-----|------|-----------|----------------|-------------|
| RTX 4090 | 24GB | $0.39 | ❌ Too small (OOM) | No |
| A40 | 48GB | $0.59 | ✅ With batch_size=2 | **Yes** |
| A100 (40GB) | 40GB | $0.79 | ⚠️ Tight fit | Maybe |
| A100 (80GB) | 80GB | $1.39 | ✅ Comfortable | Yes (if budget allows) |

**Recommendation:** A40 (48GB) @ $0.59/hour

**Memory calculation:**

- Model: ~380M params × 4 bytes (FP32) = 1.52GB
- With FP16 + gradients + optimizer: ~8-10GB
- Per sample: ~1.5GB (audio embeddings)
- Batch size 2: ~3GB
- **Total: ~11-13GB** (fits in 48GB A40 comfortably)

**Training time estimate:**

- Dataset: 200-300 segments
- Batch size: 2, accumulation: 16 → effective batch = 32
- Epochs: 100
- Steps per epoch: 200÷32 ≈ 6-10 steps
- Time per step: ~20-30 seconds (larger model)
- **Total: 40-80 hours**

**Cost:**

- A40 @ $0.59/hour × 40-80 hours = **$24-48**

### 7.4 Transfer Learning Strategy

**Option A: Fine-tune from scratch**

- Start from pre-trained MERT-330M
- Train all layers (slower, more expensive)

**Option B: Transfer from MERT-95M (recommended)**

- Copy shared components (MIDI encoder, fusion, aggregator, heads)
- Only fine-tune MERT-330M audio encoder
- Faster convergence, cheaper

```python
# scripts/transfer_to_mert330m.py

# Load MERT-95M checkpoint
ckpt_95m = torch.load('checkpoints/expert_finetune_95m/best.ckpt')

# Initialize MERT-330M model
model_330m = PerformanceEvaluationModel(
    mert_model_name='m-a-p/MERT-v1-330M',
    audio_dim=1024,
    # ... other config
)

# Transfer weights
model_330m.midi_encoder.load_state_dict(ckpt_95m['midi_encoder'])
model_330m.fusion.load_state_dict(ckpt_95m['fusion'], strict=False)
model_330m.aggregator.load_state_dict(ckpt_95m['aggregator'], strict=False)
model_330m.mtl_head.load_state_dict(ckpt_95m['mtl_head'])

# Only audio_encoder is randomly initialized
# Fine-tune with lower LR for transferred components
```

**Training config:**

```yaml
training:
  max_epochs: 50  # Fewer epochs (transfer learning)
  backbone_lr: 2e-5  # Higher (MERT-330M is new)
  heads_lr: 1e-5    # Lower (transferred, already trained)
```

**Time:** 20-40 hours (faster than training from scratch)
**Cost:** $12-24

### 7.5 Expected Performance Gains

**MERT-95M → MERT-330M expected improvements:**

- Technical dimensions: +5-10% (r=0.55-0.65 → r=0.60-0.75)
- Interpretive dimensions: +10-15% (r=0.40-0.50 → r=0.50-0.65)
- Overall: +8-12% (r=0.48 → r=0.56-0.60)

**Is it worth it?**

- If r_95m > 0.50: **Yes**, likely to reach r > 0.60 (useful performance)
- If r_95m < 0.40: **No**, fix data quality first

**Time:** 2-3 weeks (training + evaluation)
**Cost:** $24-48

---

## Phase 8: MVP Deployment (Weeks 19-20)

**Goal:** Package model into usable tool (for yourself or early testers)

### 8.1 Simple API (FastAPI)

```python
# api/main.py
from fastapi import FastAPI, UploadFile
import torch
import librosa

app = FastAPI()

# Load model at startup
model = PerformanceEvaluationModel.load_from_checkpoint('best.ckpt')
model.eval()
model = model.cuda()

@app.post("/evaluate")
async def evaluate_performance(audio: UploadFile):
    # Save uploaded file
    audio_path = f"/tmp/{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    # Load and preprocess
    y, sr = librosa.load(audio_path, sr=24000)

    # Segment into 20-second chunks
    segments = segment_audio(y, sr, duration=20.0, overlap=5.0)

    # Predict
    predictions = []
    for segment in segments:
        pred = model(torch.tensor(segment).unsqueeze(0).cuda())
        predictions.append(pred['scores'].cpu().numpy())

    # Average predictions
    avg_pred = np.mean(predictions, axis=0)

    return {
        "dimensions": model.dimension_names,
        "scores": avg_pred.tolist(),
        "uncertainties": pred['uncertainties'].cpu().numpy().tolist()
    }
```

**Deployment options:**

**Option A: Run locally**

```bash
# Install dependencies
pip install fastapi uvicorn

# Run server
uvicorn api.main:app --reload

# Test
curl -X POST "http://localhost:8000/evaluate" \
  -F "audio=@test_performance.wav"
```

**Option B: Deploy to cloud (Hugging Face Spaces, free tier)**

```yaml
# spaces/app.py
import gradio as gr

def evaluate(audio_file):
    # ... same evaluation logic
    return results

demo = gr.Interface(
    fn=evaluate,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.JSON(),
    title="Piano Performance Evaluator"
)

demo.launch()
```

**Cost:** $0 (Hugging Face Spaces free tier)

### 8.2 Simple Web UI (Gradio)

```python
# app.py
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

def evaluate_and_visualize(audio_file):
    # Evaluate
    scores = evaluate_performance(audio_file)

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(scores['dimensions'], scores['scores'])
    ax.set_xlim(0, 100)
    ax.set_xlabel('Score (0-100)')
    ax.set_title('Performance Evaluation')

    return fig, scores

demo = gr.Interface(
    fn=evaluate_and_visualize,
    inputs=gr.Audio(type="filepath"),
    outputs=[gr.Plot(), gr.JSON()],
    title="Piano Performance Evaluator MVP",
    description="Upload a piano recording to get AI feedback"
)

demo.launch(share=True)  # Creates public link
```

**Features:**

- Drag-and-drop audio upload
- Visual bar chart of scores
- JSON output with details
- Public shareable link (for testing with friends)

**Time:** 1-2 days
**Cost:** $0

---

## Phase 9: Active Learning Loop (Ongoing)

**Goal:** Continuously improve with minimal additional labeling

### 9.1 Identify High-Uncertainty Samples

```python
# scripts/active_learning_selection.py

# 1. Run model on large unlabeled dataset (YouTube, IMSLP)
unlabeled_predictions = []
for audio_path in unlabeled_dataset:
    pred = model(audio_path)
    uncertainty = pred['uncertainties'].mean()  # Average uncertainty across dims
    unlabeled_predictions.append({
        'audio_path': audio_path,
        'uncertainty': uncertainty
    })

# 2. Sort by uncertainty (highest = most informative)
sorted_samples = sorted(unlabeled_predictions, key=lambda x: x['uncertainty'], reverse=True)

# 3. Select top 50 for annotation
next_batch = sorted_samples[:50]

print("Next 50 samples to annotate (highest uncertainty):")
for sample in next_batch:
    print(f"  {sample['audio_path']}: uncertainty={sample['uncertainty']:.3f}")
```

### 9.2 Iterative Improvement

**Monthly cycle:**

1. Annotate 20-50 high-uncertainty samples (yourself or crowdsource)
2. Retrain model with expanded dataset
3. Evaluate on test set
4. Track performance over time

**Cost per cycle:**

- Annotation: $0 (self) or $50-100 (crowdsource)
- Training: $5-10 (RunPod)
- **Total: $5-110/month**

**Expected gains:**

- +2-5% correlation per 50 labels added (diminishing returns)
- After 500 total labels: r ≈ 0.60-0.70 (approaching research targets)

---

## Total Costs: Bootstrapped Solo Scenario

| Phase | Cost | Time |
|-------|------|------|
| **Phase 1: Validation (Colab)** | $0-10 | 1 week |
| **Phase 2: Self-Annotation** | $0 | 2-3 weeks |
| **Phase 3: Recruit Friends** | $50-150 | 2 weeks |
| **Phase 4: Crowdsourcing** | $400-600 | 3-4 weeks |
| **Phase 5: Fine-tune (MERT-95M)** | $8-20 | 1 week |
| **Phase 6: Evaluation & Iteration** | $0-400 | 1-2 weeks |
| **Phase 7: Scale to MERT-330M (optional)** | $24-48 | 2-3 weeks |
| **Phase 8: MVP Deployment** | $0 | 1 week |
| **TOTAL (First 4-5 months)** | **$482-1,228** | **14-19 weeks** |

**Ongoing costs:**

- Active learning: $5-110/month
- Cloud inference (if deployed): $10-50/month

**Total investment to useful MVP: <$1,500**

---

## Expected Performance: Bootstrapped Solo

**With 200-300 expert labels + MERT-95M:**

| Metric | Expected Range | Interpretation |
|--------|----------------|----------------|
| **Overall correlation (r)** | 0.45-0.55 | Useful, matches single expert |
| **Technical dimensions (r)** | 0.50-0.65 | Good on objective dimensions |
| **Interpretive dimensions (r)** | 0.35-0.50 | Moderate, needs more data |
| **MAE (0-100 scale)** | 12-18 points | Within inter-rater variability |
| **Pairwise ranking** | 65-75% | Better than random, useful for feedback |

**With 500+ labels + MERT-330M:**

| Metric | Expected Range | Interpretation |
|--------|----------------|----------------|
| **Overall correlation (r)** | 0.55-0.68 | Approaching research targets |
| **Technical dimensions (r)** | 0.60-0.75 | Strong performance |
| **Interpretive dimensions (r)** | 0.45-0.60 | Useful performance |
| **MAE (0-100 scale)** | 10-15 points | Good accuracy |
| **Pairwise ranking** | 70-82% | Reliable comparisons |

**Outcome:** Useful tool for personal practice, teaching 1-on-1 students, or early product validation. Not professional-grade, but practical and affordable.

---

## Key Differences: Unlimited vs. Bootstrapped

| Aspect | Unlimited Resources | Bootstrapped Solo |
|--------|-------------------|-------------------|
| **Expert labels** | 2,500 (hired annotators) | 200-300 (self + crowdsource) |
| **Model size** | 473M (MERT-330M + specialists) | 112M → 380M (MERT-95M → 330M) |
| **Dimensions** | 12 (tech + interpretive) | 6-10 (focus on technical) |
| **Training time** | 4 weeks (parallel GPUs) | 2-8 weeks (iterative) |
| **Total cost** | $174-246K | $482-1,228 |
| **Time to MVP** | 8 months | 4-5 months |
| **Expected r** | 0.68-0.75 | 0.45-0.68 |
| **Use case** | Conservatory education, competitions | Personal practice, early validation |

**Bottom line:**

- **Unlimited:** Professional-grade system, suitable for commercial deployment
- **Bootstrapped:** Useful MVP, validates concept, enables fundraising for scale-up

---

## Recommended Path: Hybrid Approach

**Phase 1-2: Bootstrap (Months 1-5)**

- Self-annotate + crowdsource to 200-300 labels ($500-1,000)
- Train MERT-95M to validate (cost: $10-50)
- If r > 0.50 on technical dimensions → **Proof of concept validated**

**Phase 3: Fundraising (Month 6)**

- Use MVP to pitch investors, grants, or educational institutions
- Show: "We achieved r=0.52 with $1K, imagine with $50K..."
- Target: $50-100K for professional development

**Phase 4: Scale (Months 7-12)**

- Hire expert annotators (1,000+ labels)
- Upgrade to MERT-330M + ensemble specialists
- Target: r > 0.65, suitable for product launch

**Total timeline:** 12 months
**Total cost:** $1K (bootstrap) + $50-100K (scale-up) = **$51-101K**
**Outcome:** Production-ready system with validated proof-of-concept

This is the **most realistic path** for a solo founder with limited initial capital.
