# Encoder Status

Historical encoder results and the current inference-loading contract.

> **Status (2026-08-04):** The serving loader uses the frozen stock `OpenMuQ/MuQ-large-msd-iter` checkpoint plus separately trained attention, encoder, and regression/Gaussian heads. It does not load MuQ LoRA adapters; the pretrained-MuQ training path raises `NotImplementedError`. The clean-fold A1-Max result (**79.85% pairwise, R2=0.336**) and Aria experiments below remain historical evidence. The MuQ/Aria parallel-stream training program is closed; #139 owns the benchmark-first successor.

**Notebooks:** `model/notebooks/model_improvement/01_audio_training.ipynb`, `02_symbolic_training.ipynb`
**Code:** `model/src/model_improvement/`
**Taxonomy:** `docs/model/02-teacher-grounded-taxonomy.md`

---

## IMPORTANT: Fold Leak Invalidation

**All pairwise accuracy and R2 numbers reported below from prior experiments are INVALID.** The original fold splits leaked pieces across train/validation boundaries. Clean piece-stratified folds have been regenerated and verified in `model/data/labels/percepiano/folds.json`. All models must be retrained and re-evaluated on clean folds before any number can be cited.

Numbers are retained below for relative comparison only (all experiments used the same leaked folds, so relative ordering may hold). Absolute values should not be reported externally.

---

## Audio Encoder

### Historical training recipe: A1-Max (MuQ + LoRA)

Frozen MuQ backbone (pretrained on 160K hours of music) with LoRA rank-32 adapters on layers 7-12 (<1% of MuQ params). Multi-task training on PercePiano (1,202 segments, 6 teacher-grounded dimensions, 4-fold piece-stratified CV).

```
MuQ embeddings [93, 1024]
  -> LoRA adapters (rank-32, layers 7-12)
  -> Attention pooling -> [1024]
  -> Shared encoder (2-layer MLP + LayerNorm) -> z_audio [512]
  -> Per-dimension ranking heads (6 dims)
  -> Regression head (6 dims, sigmoid output)
```

**Loss (original):** `L = L_ranking + 1.5 * L_ListMLE + 0.3 * L_contrastive + 0.3 * L_CCC + 0.1 * L_invariance`

**Loss (optimized, 2026-03-19):** `L = L_ranking + 1.5 * L_ListMLE + 0.6 * L_contrastive + 0.8 * L_CCC + 0.1 * L_invariance`

Autoresearch (8 iterations) found that the original weights drastically underweighted regression -- producing a model that ranked well but gave meaningless absolute scores (R2 < 0). Doubling contrastive and nearly tripling regression weight recovers R2 to +0.24 without sacrificing pairwise accuracy. Key findings: contrastive provides crucial regularization even when near-converged; mixup is essential for PercePiano's small sample size (~900 training samples per fold).

Key improvements over A1 baseline: ListMLE ranking loss (Plackett-Luce likelihood), CCC regression loss, embedding mixup, hard negative mining with curriculum, wider LoRA adaptation (layers 7-12 vs 9-12), label smoothing (0.1).

### Positive-Mining Discipline (SemiSupCon)

From the Mahler wiki's Music Representation Learning page: in audio SSL, the **positive-mining strategy inside the contrastive loss matters more than the overall labeled-to-unlabeled ratio**. Two adjacent 2.7-second segments from the same track are a weak positive pair -- they share style (same piece, same performer, same recording conditions) but can differ in expressive quality. Naive SSL conflates style-similarity with quality-similarity; the encoder learns a style-aware metric dressed up as a quality metric.

SemiSupCon's fix, which we adopt, is threefold:

1. **Label-derived positives.** Pairs known to share a quality-relevant label (PercePiano segments rated near-identical on the target dimension, or T2 competition placements within the same round) are forced into the positive set.
2. **Hard negatives from labels.** Labeled samples with *different* quality levels serve as hard negatives even when their style features overlap (same piece, same performer, different takes).
3. **Quality-orthogonal augmentation.** Augmentations that preserve style but perturb quality (or vice versa) are deliberately included to prevent the encoder from collapsing the two axes.

This discipline shaped the retired 20%/80% PercePiano-anchor/ordinal program. Its infrastructure shipped, but the gated sweeps did not run before the program closed. Any future reuse must verify that positive pairs carry quality-label agreement, not only piece identity.

### Current inference-loading contract

- **Backbone:** frozen stock `OpenMuQ/MuQ-large-msd-iter`
- **Heads:** four separately trained pooling/prediction heads loaded from fold checkpoints
- **Input:** 15-second audio chunks at 24kHz mono
- **Output:** 6 dimension scores (0-1 range)
- **Calibration:** MAESTRO calibration stats in `model/data/maestro_cache/calibration_stats.json`

### Results: Clean Piece-Stratified Folds (2026-03-19)

#### Ablation Sweep (original loss weights)

| Config | Pairwise (4-fold mean) | R2 (4-fold mean) |
|--------|----------------------|-----------------|
| **full_a1max_repro** | **77.50%** | **0.119** |
| bce_listmle_ccc | 75.43% | -1.087 |
| bce_ranking_only | 73.02% | -0.258 |
| bce_plus_listmle | 69.73% | -11.138 |
| frozen_probe | 53.42% | 0.461 |

The ~3.3pp drop from leaked 80.8% to clean 77.5% confirms the fold leak was real and inflated results. Ranking losses (ListMLE) are essential -- removing them drops pairwise by 4-8pp. Frozen probe (regression-only, no LoRA) gets the best absolute R2 (0.46) but terrible ranking (53.4%).

#### Loss ablation (pre-clean-fold, relative ordering)

From pre-fix (fold-leaked) runs archived in `docs/model/archive/05-ismir-paper-status.md`. Absolute numbers are inflated; relative ordering is expected to hold on clean folds since the pattern reflects loss dynamics, not data leakage.

- BCE-only: 73.0% pairwise
- BCE + ListMLE (no regression anchor): 69.7% -- ListMLE alone degrades pairwise, likely due to degenerate ranking solutions without an absolute reference
- BCE + ListMLE + CCC: 75.4% -- CCC regression loss anchors ListMLE and recovers above BCE-only

**Takeaway:** ranking losses require a regression anchor (CCC) to avoid destabilization. Do not add ListMLE without CCC.

#### Optimized Loss Weights (4-fold validated)

Loss weight autoresearch (8 iterations, single-fold search, then 4-fold validation): found that contrastive=0.6, regression=0.8 (vs original 0.3, 0.3) dramatically improves both metrics.

| Config | Pairwise (4-fold mean) | R2 (4-fold mean) |
|--------|----------------------|-----------------|
| Original weights | 77.50% | 0.119 |
| **Optimized weights** | **79.85%** | **0.336** |

Per-fold optimized results: 76.7%, 78.9%, 81.2%, 82.5% pairwise. R2 per fold: 0.21, 0.24, 0.42, 0.48.

The +2.35pp pairwise gain comes from two fixes: (1) optimized loss weights (contrastive 2x, regression 2.7x), and (2) evaluating the best checkpoint instead of the last epoch. R2 nearly tripled, meaning the model now produces meaningful absolute scores. Full autoresearch log: `model/data/results/loss_weight_autoresearch.tsv`, `model/data/results/loss_weight_changelog.md`.

### Results (INVALID -- fold leak, retained for relative comparison only)

#### A1-Max 4-Fold Ensemble (historical leaked-fold result)

| Metric | Value (INVALID) | vs A1 Baseline |
|--------|-----------------|---------------|
| **Pairwise accuracy** | **80.77%** | +6.84pp |
| **R2** | **0.5021** | +0.0989 |
| **Robustness (score drop)** | **0.08%** | Same |
| **Robustness (Pearson r)** | **1.0000** | Same |

#### A1-Max Top 5 Configs (from 18-config sweep)

| Config | LoRA Rank | Layers | Label Smooth | Pairwise (INVALID) | R2 (INVALID) |
|--------|-----------|--------|-------------|---------------------|---------------|
| **r32_L7-12_ls0.1** | **32** | **7-12** | **0.1** | **0.7872** | **0.1553** |
| r8_L7-12_ls0.1 | 8 | 7-12 | 0.1 | 0.7866 | 0.1514 |
| r32_L7-12_ls0.05 | 32 | 7-12 | 0.05 | 0.7861 | 0.0974 |
| r8_L9-12_ls0.0 | 8 | 9-12 | 0.0 | 0.7859 | 0.1616 |
| r16_L9-12_ls0.1 | 16 | 9-12 | 0.1 | 0.7852 | 0.1393 |

#### All Audio Experiments (averaged across 4 folds, INVALID)

| Model | Strategy | Pairwise Acc (INVALID) | R2 (INVALID) |
|-------|----------|------------------------|--------------|
| **A1-Max (ensemble)** | **LoRA rank-32 + ListMLE/CCC/mixup** | **80.8%** | **0.50** |
| A1 | MuQ + LoRA rank-16 | 73.9% | 0.40 |
| A2 | Staged domain adaptation | 71.4% | 0.42 |
| A3 | Full unfreeze, gradual | 69.9% | 0.28 |

**A2 MAESTRO ablation:** Adding 24K MAESTRO segments to Stage 1 contrastive pretraining showed no improvement. MuQ was pretrained on 160K hours -- more piano audio doesn't help.

### Audio Interpretation

**Why LoRA wins:** More aggressive adaptation (A2, A3) doesn't improve and A3 actively hurts (fold 0 R2=0.059 = catastrophic forgetting). MuQ's pretrained representations are already well-suited. With ~750 training samples per fold, there isn't enough data to reshape the backbone.

**ListMLE is the biggest A1-Max contributor.** Ranking-dominant loss (lambda=1.5) explicitly optimizes Plackett-Luce ranking likelihood, aligning loss directly with pairwise accuracy.

**R2 trade-off in A1-Max:** Individual fold R2 (~0.15) drops below A1 (~0.40) because ranking-dominant loss de-emphasizes regression. But 4-fold ensemble R2 (0.50) recovers -- regression heads learn complementary patterns across folds.

**Fold variance:** A1 pairwise ranges 70.3-77.7% (~7pp spread). Driven by which of 61 multi-performance pieces land in validation. Data quantity constraint, not model problem.

### Per-Dimension MuQ Predictability

| Dimension | MuQ Probing R2 | Teacher Frequency |
|-----------|---------------|-------------------|
| articulation | 0.607 | 11.4% |
| dynamics | 0.587 | 14.1% |
| phrasing | 0.569 | 13.1% |
| interpretation | 0.524 | 36.7% |
| pedaling | 0.513 | 6.8% |
| timing | 0.332 | 18.0% |

Timing is hardest for audio (R2=0.332) -- strongest candidate for symbolic support. Articulation is strongest (0.607) -- note attack/release directly audible.

### Aria vs MuQ: Frozen Linear Probe Comparison (2026-03-19)

Linear probe on frozen embeddings, 4-fold piece-stratified CV (clean folds). These are the first VALID numbers on clean folds.

| Dimension | Aria-Embedding (512d) | Aria-Base (1536d) | MuQ mean-pooled (1024d) |
|-----------|----------------------|-------------------|------------------------|
| dynamics | 65.8% | 62.5% | **72.4%** |
| timing | 55.8% | 58.0% | **67.5%** |
| pedaling | 58.6% | 60.7% | **66.6%** |
| articulation | 54.2% | 54.3% | **54.7%** |
| phrasing | 57.9% | 54.8% | **60.9%** |
| interpretation | 61.2% | 62.3% | **63.9%** |
| **Overall** | **59.6%** | **59.6%** | **62.2%** |

Error correlation (phi): **0.043** -- near-zero. Models make completely independent mistakes, validating dual-encoder use. (Pre-2026-05-27 this was framed as "fusion viable"; the same finding now justifies parallel streams, since independent errors mean both encoders contribute genuinely independent information whether you combine them with learned gates or expose both to the teacher LLM. See [[project_parallel_streams_decision]].)

**Confound check (required on every experiment):** Aria-only skill discrimination on T5 val. If Aria discriminates skill buckets (above 50% chance), the signal is musical. If MuQ discriminates but Aria doesn't, MuQ may be exploiting audio quality as a shortcut.

MuQ dominates all dimensions from frozen embeddings. This is expected: MuQ was pretrained on 160K hours of audio for music understanding tasks, while Aria was pretrained on MIDI for generation/identity tasks (not quality). The key finding is that Aria has quality signal (significantly above 50% chance) despite never being trained for quality, and its errors are independent from MuQ's.

### Historical MuQ continued-pretraining plan

Before fine-tuning on PercePiano, apply symmetric contrastive pretraining to MuQ so that its embeddings become quality-aware (not just content-aware). This mirrors what Aria's SimCSE contrastive stage does for symbolic.

**Approach:**
- NT-Xent contrastive loss on PercePiano pairs with known quality ordering
- Positive pairs: same piece, different performer (quality-varying)
- Negative pairs: different pieces
- Curriculum: easy negatives first (different composers), then hard (same composer, different piece)
- Training: 20-30 epochs on top of frozen MuQ, adapting only LoRA layers + pooling head
- Goal: reduce error correlation with Aria by making audio embeddings explicitly quality-sensitive (improves per-stream score reliability; pre-2026-05-27 framed as "before fusion")

This is symmetric with Aria's contrastive pretraining -- both encoders get quality-aware contrastive training before fine-tuning.

### Historical audio experiments

| Experiment | Effort | Expected Impact |
|-----------|--------|-----------------|
| Retrain A1-Max on clean folds | High | Establish valid baselines |
| Quality-aware contrastive pretraining | Medium | More reliable per-stream quality scores |
| Multi-head attention pooling (6 heads, one per dim) | Medium | +2-3% pairwise |
| Multi-scale temporal modeling (hierarchical pooling) | Medium | +2-4% pairwise |
| Competition data (T2) integration | Medium | +3-5% pairwise |
| Per-dimension loss weighting (by MuQ R2) | Low | +1-2% on strong dims |

---

## Current symbolic encoder: MoonBeam-839M (chosen 2026-08-03, #138 Phase 0)

MoonBeam-839M (`guozixunnicolas/moonbeam-midi-foundation-model`, Apache 2.0)
replaced Aria as the symbolic encoder after a frozen bake-off on the **Transkun
domain** -- the domain the difficulty head actually trains on, rather than the
clean PSyllabus MIDIs the earlier Aria numbers were measured on. Scope: this is
the encoder for MIREX Track A difficulty (#138). It does not reopen the closed
MuQ/Aria parallel-stream program; #139 still owns the successor architecture.

**Composer-disjoint 5-fold x 5 seeds, RidgeCV, n=900 Transkun MIDIs across 900
distinct composers, all arms on byte-identical folds:**

| encoder | pooling | tau-c | std |
|---|---|---|---|
| Aria-medium | EOS-position (its shipped scheme) | 0.7790 | 0.0030 |
| Aria-medium | mean over tokens | 0.7702 | 0.0030 |
| **MoonBeam-839M** | **mean over tokens** | **0.8257** | 0.0018 |
| MoonBeam-839M | last token | 0.7827 | 0.0016 |

Paired bootstrap over pieces (2000 resamples): `moonbeam_mean - aria_eos` =
+0.0540 [+0.0379, +0.0710]; under **matched** pooling `moonbeam_mean -
aria_mean` = **+0.0619 [+0.0446, +0.0787]**, P(diff<=0) = 0.000.

**The pooling confound was tested and refuted.** MoonBeam gains +0.046 from
mean-over-tokens vs last-token, which first read as the bake-off measuring
pooling rather than the backbone. The control -- same Aria checkpoint, same
300-note chunks, both poolings from one forward pass, `eos_pool` reproducing the
deployed `get_global_embedding_from_midi` path bit-exactly -- showed mean
pooling does **not** help Aria (-0.0079, CI straddles 0). The pooling gain is
backbone-specific, and under matched pooling MoonBeam's lead grows. Plausible
mechanism (not measured): `aria-medium-embedding` is contrastively trained so
its EOS position already IS the pooled global vector, while MoonBeam is a plain
causal LM whose difficulty signal is distributed across tokens.

**Do not compare 0.8257 across harnesses.** It is RidgeCV + seeded
composer-disjoint folds (`bakeoff_cv.py`), not comparable to #137's 37-feature
anchor 0.7929/0.7971 (LightGBM + GroupKFold, `tk_ablation.py`), to the old
frozen-Aria 0.744 (clean-MIDI, n=600), or to the train-on-test 0.824.

Harness: `model/src/claim_measurement/difficulty/{run_bakeoff,extract,
moonbeam_extract_script,aria_pooling_backbone}.py`. Weights and the fork pinned
at commit `4e2c015` live in `model/data/weights/moonbeam/` (gitignored).

**Same-folds gate (#149 Phase 1, 2026-08-03):** on these same bake-off folds, the
37 hand-engineered difficulty features alone score **0.8048** tau-c -- so
MoonBeam's real lead over the reference feature set is **+0.024**, not the
headline +0.047 vs Aria. 0.8048 is the #149 Phase 1 gate a fine-tuned encoder
must clear. As of 2026-08-04 the Phase 1 LoRA harness is merged and CPU-verified,
but **no fine-tune has been run and the gate is unmeasured**; operator steps live
in `docs/mirex/phase1-lora-runbook.md`.
Canonical source: `docs/mirex/track-a-difficulty-prediction.md`.

---

## Historical symbolic encoder: Aria (superseded 2026-08-03)

Retained as the #138 bake-off reference arm and as a paper artifact -- the
`model/src/model_improvement/aria_*.py` modules are released by the arXiv paper
and implement the reference arm above, so they are **not dead code**.

Aria (EleutherAI, 2025, 650M-param LLaMA-architecture, 820K piano MIDI
pretraining, Apache 2.0) was adopted 2026-03-18 to replace all custom symbolic
encoders (S1/S2/S2H/S3), eliminating the need for a from-scratch symbolic
foundation model. It reached SOTA on 6 MIR benchmarks (genre, form, composer,
performer, period, emotion) and, via `delta = z_perf - z_score` over dual
performance/score MIDI input, gave immediate score conditioning. Frozen linear
probe (2026-03-19): 59.6% pairwise, error correlation phi=0.043 vs MuQ
(near-zero, validating dual-encoder use). Independent LoRA fine-tune
(2026-06-26, T1-only, fluidsynth-proxy audio): mean pairwise 0.6988, R2=0.194;
the full multi-tier run never happened before the program closed.

Superseded 2026-08-03 by MoonBeam-839M (#138 Phase 0) on Transkun-domain
tau-c. See the MoonBeam section above for the current encoder and
`docs/model/04-north-star.md` for where symbolic scoring sits in the pipeline.

AMT-survival validation (methodology, not the encoder choice, is what's load
bearing): ByteDance AMT vs ground-truth MIDI on studio audio showed 0% pairwise
drop; on YouTube mediocre audio, 79.9% A1-vs-S2 cross-encoder agreement (all
dimensions > 72%). This generalizes to any symbolic encoder, including Aria --
the AMT bottleneck sits upstream of encoder choice.

---

## Historical parallel-stream architecture (superseded 2026-08-03)

Parallel streams (MuQ audio stream + Aria symbolic stream + deterministic
MPM-style extraction, all three feeding the teacher LLM as independent
signals, no learned fusion gates) replaced gated fusion on 2026-05-27. Gated
fusion was retired because the ISMIR paper's fused score underperformed
audio-only (R2 0.524 vs 0.537, error correlation r=0.738 -- both streams
failed on the same samples); Aria's pretraining scale closed that asymmetry
and dropped error correlation to phi=0.043, which justified exposing
disagreement to the teacher LLM directly rather than routing it through
learned gates.

This architecture, its diagrams, and its training protocol are superseded
along with Aria (see above) -- MoonBeam-839M is scoped to MIREX Track A
difficulty only (#138) and does not reopen the parallel-stream program.
Current pipeline framing lives in `docs/model/04-north-star.md`; the
retirement decision trail is #127 (audio-teacher pivot) and #139 (successor
architecture).

---

## Cross-Modality Comparison (ALL NUMBERS INVALID -- fold leak)

Retained for relative comparison only. All models used the same leaked folds.

| Rank | Model | Modality | Pairwise (INVALID) | R2 (INVALID) |
|------|-------|----------|---------------------|--------------|
| 1 | **A1-Max (ensemble)** | **Audio** | **80.8%** | **0.50** |
| 2 | A1-Max (single fold mean) | Audio | 78.7% | 0.16 |
| 3 | A1 (LoRA) | Audio | 73.9% | 0.40 |
| 4 | A2 (Staged) | Audio | 71.4% | 0.42 |
| 5 | S2 (GNN) | Symbolic | 71.3% | 0.32 |
| 6 | S2H (Hetero GNN) | Symbolic | 70.2% | 0.36 |
| 7 | S3 (CNN+Trans) | Symbolic | 70.0% | 0.37 |
| 8 | A3 (Full Unfreeze) | Audio | 69.9% | 0.28 |
| 9 | S1 (Transformer) | Symbolic | 68.4% | 0.33 |

---

## LEGACY: Custom Symbolic Encoders (Superseded by Aria)

Custom symbolic encoders (S1 Transformer on REMI, S2 GNN on score graph, S2H heterogeneous GNN, S3 CNN+Transformer on continuous features) were superseded by Aria on 2026-03-18 and archived on 2026-04-21. All were trained from scratch on ~24K sequences, creating a pretraining-scale asymmetry with MuQ (160K hours) that made dual-encoder combination fail in the ISMIR experiments (error correlation r=0.738). Aria's 820K MIDI pretraining closes that gap, which is what made dual-encoder approaches (now: parallel streams, 2026-05-27; previously: gated fusion) viable.

Best leaked-fold results for context only: S2 71.3% / S2H 70.2% / S3 70.0% / S1 68.4% pairwise. Code lives at `model/archive/model_improvement/{graph.py,symbolic_encoders.py}` and archived S2 GNN dataset/collate code under `model/archive/` (see `model/archive/README.md`). Not imported by any active pipeline.

---

## Historical product interpretation

### Teaching Moment Selection -- Workable

Teaching moment selection gates on worst-dimension `deviation < 0` (student below their own baseline) with a positive-moment fallback, running in `teaching_moments.rs` (WASM) after HF scores return. Ranking quality matters more than absolute accuracy.

### Student Model / Blind Spot Detection -- Workable With Smoothing

Blind spot detection compares relative dimension deviations. Depends on ranking consistency more than absolute R2. Student model uses exponential moving averages (alpha=0.3) across sessions, smoothing per-chunk noise.

### LLM Teacher Prompt -- Sufficient

The LLM receives structured context like `"pedaling": 0.35, baseline: 0.62`. Whether true score is 0.35 or 0.42 matters less than "pedaling is significantly below baseline." Model provides relative signal.

### Progress Tracking -- Noisy but Usable

The leaked-fold R2~0.50 did not license a product claim. The clean-fold value is 0.336, and real-practice validity remains unproven.

### Score Conditioning -- Immediate with Aria

Aria encodes both performance MIDI and score MIDI natively. Score conditioning is available from day one of Aria integration, not deferred to a future phase. This fixes the dynamics inversion (rho=-0.917) because quality becomes relative: pp when score says pp = HIGH quality, pp when score says ff = LOW quality.

Training data: reference-anchored on MAESTRO (ranking signal from multiple performers of the same piece, no new annotation needed).

---

## Layer 1 Validation Results (2026-03-11)

Code: `model/src/model_improvement/layer1_validation.py`, `midi_comparison.py`, `feedback_assessment.py`
Notebook: `model/notebooks/model_improvement/04_layer1_validation.ipynb`

### Experiment 1: Competition Correlation -- PASS

A1 scores on 2,293 Chopin 2021 competition segments (11 performers) correlate with expert placement.

| Aggregation | rho | p-value | Gate |
|-------------|-----|---------|------|
| mean | +0.704 | 0.016 | PASS |
| min | +0.654 | 0.029 | PASS |
| median | +0.248 | 0.463 | INVESTIGATE |

Per-dimension (mean aggregation):

| Dimension | rho | p-value |
|-----------|-----|---------|
| dynamics | -0.917 | 0.0001 |
| timing | -0.590 | 0.056 |
| pedaling | +0.887 | 0.0003 |
| articulation | +0.292 | 0.383 |
| phrasing | +0.803 | 0.003 |
| interpretation | +0.169 | 0.620 |

Pedaling and phrasing are strongest predictors. Dynamics is inverted -- captures "amount" not "appropriateness." Score conditioning via Aria delta will fix this.

### Experiment 2: AMT Degradation -- PASS

ByteDance piano transcription vs ground-truth MIDI on 50 MAESTRO recordings (107 pairs): **0.0% pairwise accuracy drop.** All per-dimension drops < 4%.

YouTube follow-up (50 mediocre recordings, 1,225 pairs): **79.9% A1-vs-S2 agreement.** All dimensions > 72%.

### Experiment 3: Dynamic Range -- DIAGNOSTIC

| Comparison | Cohen's d |
|-----------|-----------|
| Intermediate vs Professional | 0.47 |
| Advanced vs Professional | 0.47 |
| Advanced vs Intermediate | 0.15 |

Separates skill levels at group level. Usable for within-student tracking, not absolute classification.

### Experiment 4: MIDI-as-Context -- SKIP (raw stats), REVISIT (bar-aligned facts)

LLM judge: A wins 55%, B wins 45% (below 55% BORDERLINE threshold). Raw MIDI stats add "false precision." But bar-aligned passage-specific context is a fundamentally different input. "Velocity MAE = 15" is noise. "Crescendo in bars 12-16 only reaches mf, score asks for ff" is teacher-language.

Phase 1 of the pipeline roadmap (see `04-north-star.md`) builds the correct version: a bar-aligned musical analysis engine that produces structured facts per passage, combining AMT output with score comparison and reference performance statistics. This is the highest-leverage improvement in the entire roadmap.

---

## Verification Criteria (for any future experiment)

1. 4-fold piece-stratified CV, same folds as `model/data/labels/percepiano/folds.json` (CLEAN folds, post-leak fix)
2. Pairwise accuracy (primary), R2 (secondary), robustness score_drop_pct (veto at >15%)
3. Per-dimension breakdown reported
4. Bootstrap CI on pairwise accuracy difference vs A1 baseline
6. Error correlation between audio and symbolic encoders (target: r < 0.5 for dual-encoder viability; same gate threshold pre- and post-2026-05-27 architecture pivot)
