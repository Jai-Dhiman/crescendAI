# Multimodal Piano Performance Evaluation: Paper Roadmap

## Paper Vision

**Title (Working):** *Multimodal Fusion of Audio and Symbolic Features Achieves State-of-the-Art Piano Performance Evaluation*

**Core Thesis:** Combining audio foundation model representations (MERT) with symbolic MIDI features via late fusion achieves R2=0.51 on the PercePiano benchmark, a 28% improvement over the published state-of-the-art, with audio alone outperforming complex symbolic architectures.

**Target Venue:** ISMIR 2025 (primary), ICASSP 2025 (backup)

---

## Current State

### What We Have

| Component | Status | Result |
|-----------|--------|--------|
| Audio Model (MERT + MLP) | Trained | R2 = 0.434 |
| Symbolic Model (PercePiano HAN) | Trained | R2 = 0.350 |
| Late Fusion (Simple Average) | Tested | R2 = 0.510 |
| Late Fusion (Optimal Weights) | Tested | R2 = 0.512 |
| Per-dimension Analysis | Complete | 17/19 audio wins |
| Sanity Checks | Passed | Dispersion 0.69, energy correlation low |

### Gap vs Published Results

| Model | Our Result | Published | Gap |
|-------|------------|-----------|-----|
| Symbolic (HAN) | 0.350 | 0.397 | -0.047 |
| Audio (MERT) | 0.434 | N/A (novel) | - |
| Fusion | 0.510 | N/A (novel) | - |

---

## Paper Structure

```
1. Introduction
   - Piano performance evaluation is subjective but important
   - Current approaches use symbolic (MIDI) features
   - Audio foundation models are unexplored for this task
   - We show: audio beats symbolic, fusion beats both

2. Related Work
   - Piano performance assessment (PercePiano, VirtuosoNet)
   - Audio foundation models (MERT, Jukebox, MusicGen)
   - Multimodal music analysis

3. Methods
   3.1 Problem Formulation
   3.2 Audio Baseline (MERT + MLP)
   3.3 Symbolic Baseline (HAN architecture)
   3.4 Fusion Strategies (late fusion variants)

4. Experiments
   4.1 Dataset: PercePiano (1,202 segments, 19 dimensions)
   4.2 Evaluation Protocol (4-fold piece-split CV)
   4.3 Baselines and Ablations
   4.4 Main Results

5. Analysis
   5.1 Per-dimension breakdown
   5.2 Modality complementarity
   5.3 Error analysis
   5.4 Qualitative examples

6. Discussion
   - Why audio outperforms symbolic
   - Limitations
   - Future work (early fusion, cross-attention)

7. Conclusion
```

---

## Phase 1: Match Symbolic Baseline - COMPLETE

**Goal:** Achieve R2 >= 0.39 on symbolic model to match published SOTA

**Result:** R2 = 0.395 (Fold 2) - SOTA MATCHED

**Checkpoint:** `gdrive:crescendai_data/checkpoints/percepiano_sota/fold2_best.pt`

---

## Phase 2: Strengthen Audio Baseline

**Goal:** Validate audio results and establish strong audio-only baseline

### Required Experiments

1. **Additional Audio Baselines**

   | Baseline | Purpose | Priority |
   |----------|---------|----------|
   | Linear probe on MERT | Is MLP necessary? | HIGH |
   | CNN on Mel spectrogram | Non-foundation baseline | HIGH |
   | VGGish embeddings | Older audio embeddings | MEDIUM |
   | PANNs embeddings | Alternative foundation model | MEDIUM |
   | Raw audio statistics | Trivial audio baseline | HIGH |

2. **MERT Ablations**

   | Ablation | Question | Priority |
   |----------|----------|----------|
   | Layer selection (1-24) | Which layers encode performance? | HIGH |
   | Pooling (mean/max/attention/LSTM) | Best temporal aggregation? | HIGH |
   | Fine-tuning vs frozen | Is adaptation needed? | MEDIUM |
   | MERT-95M vs MERT-330M | Model size impact? | LOW |

3. **Sanity Checks (Already Done, Document)**
   - [x] Dispersion analysis (ratio = 0.69)
   - [x] Energy correlation (low except expected dims)
   - [x] Per-piece error analysis

### Expected Outcomes

- Linear probe should underperform MLP (justifies architecture)
- CNN on Mel should underperform MERT (justifies foundation model)
- MERT layers 12-24 should outperform early layers
- Mean pooling competitive with attention (simplicity wins)

---

## Phase 3: Fusion Experiments

**Goal:** Systematically compare fusion strategies and establish SOTA

### Fusion Strategies to Test

1. **Late Fusion (Prediction Level)**

   | Strategy | Formula | Status |
   |----------|---------|--------|
   | Simple average | (audio + symbolic) / 2 | Done: R2=0.510 |
   | Optimal per-dim weights | w_d *audio + (1-w_d)* symbolic | Done: R2=0.512 |
   | Learned MLP | MLP([audio, symbolic]) | TODO |
   | Stacking (meta-learner) | Ridge/RF on predictions | TODO |

2. **Feature Fusion (Embedding Level)**

   | Strategy | Description | Priority |
   |----------|-------------|----------|
   | Concatenation | [MERT_pool; HAN_out] -> MLP | MEDIUM |
   | Cross-attention | MERT attends to HAN | LOW |
   | FiLM conditioning | HAN modulates MERT | LOW |

3. **Analysis Required**

   | Analysis | Insight | Priority |
   |----------|---------|----------|
   | Error correlation | Why averaging > oracle? | HIGH |
   | Per-dimension fusion benefit | Which dims benefit from fusion? | HIGH |
   | Fusion weight stability | Cross-val fusion weights | HIGH |
   | Sample-level analysis | When does fusion help/hurt? | MEDIUM |

### Key Finding to Explain

**Observation:** Simple average (R2=0.510) beats oracle selection (R2=0.466)

**Hypothesis:** Models have complementary errors - when one is wrong, the other is often closer to correct. Averaging smooths individual errors.

**Required Analysis:**

- Compute prediction error correlation between models
- Identify samples where averaging helps most
- Visualize error distribution overlap

---

## Phase 4: Ablations and Analysis

**Goal:** Provide evidence for all claims and strengthen scientific rigor

### Required Ablations

1. **For Audio Model**

   ```
   Full model (MERT-330M, layers 12-24, mean pool, MLP)
     - Layer selection: early (1-6) vs mid (7-12) vs late (12-24)
     - Pooling: mean vs max vs attention vs LSTM
     - Head: linear vs 1-layer MLP vs 2-layer MLP
     - Model size: MERT-95M vs MERT-330M
   ```

2. **For Fusion**

   ```
   Optimal weighted fusion
     - Weight stability across CV folds
     - Weight sensitivity analysis
     - Comparison: tuned weights vs equal weights
   ```

### Required Analysis

1. **Per-Category Performance**

   | Category | Dimensions | Expected Winner |
   |----------|------------|-----------------|
   | Timing | timing | Audio |
   | Articulation | length, touch | Mixed |
   | Pedal | amount, clarity | Audio |
   | Timbre | variety, depth, brightness, loudness | Audio |
   | Dynamics | dynamic_range | Audio |
   | Tempo/Space | tempo, space, balance, drama | Symbolic (tempo) |
   | Emotion | valence, energy, imagination | Mixed |
   | Interpretation | sophistication, interpretation | Mixed |

2. **Statistical Significance**
   - Bootstrap 95% CI for all R2 values
   - Paired t-test: audio vs symbolic
   - Paired t-test: fusion vs best single

3. **Qualitative Examples**
   - 3-5 samples where audio wins decisively
   - 3-5 samples where symbolic wins
   - 3-5 samples where fusion helps most
   - Include audio clips in supplementary

---

## Phase 5: Paper Writing

### Key Claims (Each Needs Evidence)

| Claim | Required Evidence |
|-------|-------------------|
| Audio beats symbolic | R2 comparison + significance test |
| MERT captures perceptual features | Per-dimension analysis + ablations |
| Fusion achieves SOTA | R2=0.51 > 0.40 + significance |
| Modalities are complementary | Error correlation + per-dim analysis |
| Simple fusion is effective | Comparison with complex fusion |

### Figures to Create

1. **Main Results Bar Chart**
   - All baselines + our models + fusion
   - With error bars (bootstrap CI)

2. **Per-Dimension Comparison**
   - Grouped bar chart: audio vs symbolic vs fusion
   - Organized by category

3. **Fusion Analysis**
   - Scatter: audio R2 vs symbolic R2 per dimension
   - Color by fusion improvement

4. **Error Correlation**
   - Heatmap or scatter showing model error correlation

5. **Qualitative Examples**
   - Waveform + predictions + ground truth
   - For selected illustrative samples

### Tables to Create

1. **Main Results Table**

   ```
   Model                    R2      95% CI      Params
   -------------------------------------------------
   Mean predictor          0.000   -           -
   Mel-CNN                 0.XXX   [X, X]      XM
   VGGish                  0.XXX   [X, X]      XM
   MERT Linear             0.XXX   [X, X]      XM
   MERT + MLP (Ours)       0.434   [0.41, 0.46] XM
   -------------------------------------------------
   Symbolic (PercePiano)   0.397*  -           XM
   Symbolic (Our repro)    0.350   [X, X]      XM
   -------------------------------------------------
   Late Fusion (Avg)       0.510   [X, X]      -
   Late Fusion (Weighted)  0.512   [X, X]      -
   ```

2. **Per-Dimension Table**
   - All 19 dimensions
   - Audio R2, Symbolic R2, Fusion R2, Best modality

3. **Ablation Table**
   - MERT layers, pooling, head architecture

---

## Timeline (Suggested)

### For ISMIR 2025 (Deadline ~March 2025)

| Week | Tasks |
|------|-------|
| 1 | Investigate symbolic gap, run audio baselines |
| 2 | Complete audio ablations, start fusion experiments |
| 3 | Finish fusion experiments, statistical tests |
| 4 | Analysis and visualizations |
| 5 | First draft of paper |
| 6 | Revision and polish |
| 7 | Internal review, final edits |
| 8 | Submit |

### Minimum Viable Paper (4 weeks)

| Week | Tasks |
|------|-------|
| 1 | Add 2 audio baselines (Mel-CNN, linear probe) |
| 2 | Bootstrap CIs, basic ablations |
| 3 | Write paper |
| 4 | Revise and submit |

---

## Experiment Tracking

### Experiments to Run

```
[ ] Phase 1: Symbolic
    [ ] Hyperparameter audit
    [ ] Data processing verification
    [ ] Longer training attempt

[ ] Phase 2: Audio Baselines
    [ ] Linear probe on MERT
    [ ] CNN on Mel spectrogram
    [ ] Raw audio statistics baseline
    [ ] MERT layer ablation
    [ ] Pooling ablation

[ ] Phase 3: Fusion
    [ ] Learned MLP fusion
    [ ] Stacking (Ridge regression)
    [ ] Error correlation analysis
    [ ] Cross-validated fusion weights

[ ] Phase 4: Analysis
    [ ] Bootstrap confidence intervals
    [ ] Significance tests
    [ ] Per-category breakdown
    [ ] Qualitative examples selection

[ ] Phase 5: Paper
    [ ] Introduction draft
    [ ] Methods section
    [ ] Experiments section
    [ ] Analysis section
    [ ] Figures and tables
    [ ] Abstract
    [ ] Related work
    [ ] Conclusion
```

---

## Risk Mitigation

### Risk 1: Symbolic Gap Persists

**Mitigation:** Frame as "our reproduction" vs "published". Still valid if audio beats our symbolic AND fusion improves.

### Risk 2: Audio Baseline Beats Fusion

**Mitigation:** Unlikely given current results. If happens, pivot to "audio alone is sufficient" narrative.

### Risk 3: Results Don't Generalize

**Mitigation:** Use rigorous cross-validation. Acknowledge single-dataset limitation in paper.

### Risk 4: Reviewer Questions Foundation Model Fairness

**Mitigation:** Include smaller MERT (95M) and non-foundation baselines to show it's not just scale.

---

## Success Criteria

### Minimum for Submission

- [ ] Audio R2 > Symbolic R2 (statistically significant)
- [ ] Fusion R2 > Audio R2 (any improvement)
- [ ] 2+ audio baselines for comparison
- [ ] Bootstrap confidence intervals
- [ ] Clean, reproducible code

### Strong Paper

- [ ] All of minimum, plus:
- [ ] Match or explain symbolic gap
- [ ] Comprehensive ablations
- [ ] Error analysis explaining fusion benefit
- [ ] Qualitative examples
- [ ] Public code and model release

### Excellent Paper

- [ ] All of strong, plus:
- [ ] Novel insights about audio vs symbolic
- [ ] Generalizable fusion methodology
- [ ] Clear practical recommendations
- [ ] Supplementary audio examples

---

## Next Immediate Steps

1. **Today:** Review this roadmap, decide on timeline
2. **This Week:**
   - Run linear probe on MERT (quick experiment)
   - Start Mel-CNN baseline training
   - Compute bootstrap CIs for current results
3. **Next Week:**
   - Complete audio baselines
   - Investigate symbolic gap
   - Begin fusion analysis
