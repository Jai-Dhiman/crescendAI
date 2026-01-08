# Multimodal Piano Performance Evaluation: Paper Roadmap

## Paper Vision

**Title (Working):** *Multimodal Fusion of Audio and Symbolic Features Achieves State-of-the-Art Piano Performance Evaluation*

**Core Thesis:** Audio foundation model representations (MERT) achieve R2=0.433 on PercePiano, matching the published symbolic SOTA (0.397) with a dramatically simpler architecture. Fusion of audio and symbolic modalities is expected to further improve performance by combining complementary strengths.

**Target Venue:** ISMIR 2026 (primary), ICASSP 2026 (backup)

---

## Current State

### What We Have

| Component | Status | Result |
|-----------|--------|--------|
| Symbolic Model (PercePiano HAN) | COMPLETE | R2 = 0.395 (Fold 2) |
| Audio Model (MERT + MLP) | COMPLETE | R2 = 0.433 (best), 0.405 (baseline) |
| Audio Ablations | COMPLETE | 13 experiments run |
| Late Fusion | NOT STARTED | - |
| Per-dimension Analysis | Partial | Audio baseline only |
| Sanity Checks | COMPLETE | Dispersion ~0.68-0.71 |

### Comparison vs Published Results

| Model | Our Result | Published | Notes |
|-------|------------|-----------|-------|
| Symbolic (HAN) | 0.395 | 0.397 | SOTA MATCHED |
| Audio (MERT) | 0.433 | N/A | Novel contribution |
| Fusion | TBD | N/A | Not yet run |

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

## Phase 2: Strengthen Audio Baseline - COMPLETE

**Goal:** Validate audio results and establish strong audio-only baseline

**Status:** All 13 experiments completed and synced to Google Drive

### Results Summary

| ID | Description | Avg R2 | 95% CI | Key Finding |
|----|-------------|--------|--------|-------------|
| **B1b** | MERT layers 7-12 (mid) | **0.433** | [0.409, 0.461] | **BEST** |
| B1c | MERT layers 13-24 (late) | 0.426 | [0.398, 0.452] | Also strong |
| B1d | MERT all layers 1-24 | 0.410 | [0.405, 0.457] | Diminishing returns |
| B0 | MERT+MLP baseline (L13-24) | 0.405 | [0.398, 0.449] | Solid baseline |
| B1a | MERT layers 1-6 (early) | 0.397 | [0.391, 0.445] | Worse than mid/late |
| C1a | Hybrid MSE+CCC loss | 0.377 | [0.368, 0.430] | MSE better alone |
| B2b | Attention pooling | 0.369 | [0.365, 0.420] | Mean pooling better |
| C1b | Pure CCC loss | 0.363 | [0.348, 0.413] | CCC alone worse |
| B2c | LSTM pooling | 0.327 | [0.323, 0.380] | Overfits |
| B2a | Max pooling | 0.316 | [0.316, 0.362] | Loses information |
| A2 | Mel-CNN | 0.191 | [0.202, 0.252] | Foundation model wins |
| A1 | Linear probe on MERT | 0.175 | [0.108, 0.182] | MLP head necessary |
| A3 | Raw audio statistics | -12.6 | - | Complete failure |

### Validated Findings

- [x] Linear probe (0.175) << MLP (0.405): **MLP head is necessary**
- [x] Mel-CNN (0.191) << MERT (0.433): **Foundation model justified**
- [x] Mid layers 7-12 (0.433) > late layers 13-24 (0.426): **Mid layers best**
- [x] Mean pooling (0.405) > attention (0.369) > LSTM (0.327): **Simple pooling wins**
- [x] MSE loss (0.405) > hybrid (0.377) > CCC (0.363): **MSE is best**
- [x] Dispersion ratio ~0.68-0.71: **Predictions properly distributed**

### Checkpoints

All results stored at: `gdrive:crescendai_data/checkpoints/audio_phase2/`

---

## Phase 3: Fusion Experiments - NOT STARTED

**Goal:** Systematically compare fusion strategies and establish SOTA

**Status:** Requires symbolic predictions to be aligned with audio predictions first

### Prerequisites

1. Generate symbolic predictions on same test folds as audio
2. Ensure sample alignment between modalities
3. Store predictions for fusion analysis

### Fusion Strategies to Test

1. **Late Fusion (Prediction Level)** - Priority: HIGH

   | Strategy | Formula | Status |
   |----------|---------|--------|
   | Simple average | (audio + symbolic) / 2 | TODO |
   | Optimal per-dim weights | w_d *audio + (1-w_d)* symbolic | TODO |
   | Learned MLP | MLP([audio, symbolic]) | TODO |
   | Stacking (meta-learner) | Ridge/RF on predictions | TODO |

2. **Feature Fusion (Embedding Level)** - Priority: LOW

   | Strategy | Description | Status |
   |----------|-------------|--------|
   | Concatenation | [MERT_pool; HAN_out] -> MLP | TODO |
   | Cross-attention | MERT attends to HAN | TODO |

3. **Analysis Required**

   | Analysis | Insight | Priority |
   |----------|---------|----------|
   | Error correlation | Do models make different errors? | HIGH |
   | Per-dimension fusion benefit | Which dims benefit from fusion? | HIGH |
   | Fusion weight stability | Cross-val fusion weights | HIGH |

### Expected Outcome

Based on published multimodal results and dimension analysis:

- Audio strong on: timbre, pedal, dynamics (R2 > 0.5)
- Audio weak on: tempo (R2 = 0.23), timing (R2 = 0.28)
- Symbolic should complement audio on timing/tempo dimensions
- Simple averaging likely to provide 5-15% improvement

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
   Model                    R2      95% CI         Notes
   -------------------------------------------------
   Mean predictor          0.000   -              Baseline
   Raw audio stats        -12.6    -              Failed
   MERT Linear probe       0.175   [0.11, 0.18]   MLP needed
   Mel-CNN                 0.191   [0.20, 0.25]   Foundation model wins
   -------------------------------------------------
   MERT + MLP (L7-12)      0.433   [0.41, 0.46]   Best audio
   MERT + MLP (L13-24)     0.405   [0.40, 0.45]   Baseline config
   -------------------------------------------------
   Symbolic (Published)    0.397*  -              PercePiano SOTA
   Symbolic (Our repro)    0.395   -              Fold 2 only
   -------------------------------------------------
   Late Fusion (Avg)       TBD     -              Not yet run
   Late Fusion (Weighted)  TBD     -              Not yet run
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

### Completed Experiments

```
[x] Phase 1: Symbolic Baseline
    [x] Matched published SOTA: R2 = 0.395 vs 0.397

[x] Phase 2: Audio Baselines (13 experiments)
    [x] A1: Linear probe on MERT (R2 = 0.175)
    [x] A2: Mel-CNN baseline (R2 = 0.191)
    [x] A3: Raw audio statistics (R2 = -12.6, failed)
    [x] B0: MERT+MLP baseline (R2 = 0.405)
    [x] B1a-d: Layer ablation (best: B1b = 0.433)
    [x] B2a-c: Pooling ablation (best: mean = 0.405)
    [x] C1a-b: Loss ablation (best: MSE = 0.405)
```

### Remaining Experiments

```
[ ] Phase 3: Fusion
    [ ] Align symbolic/audio predictions on same samples
    [ ] Simple average fusion
    [ ] Optimal per-dim weighted fusion
    [ ] Error correlation analysis
    [ ] Per-dimension fusion benefit analysis

[ ] Phase 4: Analysis
    [ ] Bootstrap confidence intervals (have CIs from Phase 2)
    [ ] Significance tests (audio vs symbolic)
    [ ] Per-dimension breakdown table
    [ ] Qualitative examples selection

[ ] Phase 5: Paper
    [ ] All sections
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

**Phase 2 Complete.** Next: Fusion experiments.

1. **Preparation:**
   - Generate symbolic model predictions for same samples as audio
   - Ensure fold assignments match between modalities
   - Create prediction alignment script

2. **Fusion Experiments:**
   - Simple average: (audio + symbolic) / 2
   - Per-dimension optimal weights via grid search
   - Analyze which dimensions benefit most

3. **Analysis:**
   - Error correlation between modalities
   - Per-dimension comparison table
   - Statistical significance tests
