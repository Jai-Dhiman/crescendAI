# Aria Symbolic Encoder Validation Experiment

**Date:** 2026-03-18
**Status:** Design approved, pending implementation

## Goal

Determine whether Aria's (EleutherAI, 650M params, pretrained on 820K piano MIDIs) frozen representations capture piano performance **quality** signal -- not just compositional or performer identity. This is the critical validation gate before committing to Aria as CrescendAI's symbolic encoder (replacing S2 GNN).

Secondary goal: compare ByteDance AMT vs Aria-AMT on real YouTube recordings to inform AMT selection for the production pipeline.

## Context

- CrescendAI evaluates piano performance quality across 6 dimensions: dynamics, timing, pedaling, articulation, phrasing, interpretation
- Previous symbolic encoder (S2 GNN, 71.3% pairwise) trained from scratch on 24K graphs -- results INVALID due to fold leak
- Aria was pretrained on 820K piano MIDIs, SOTA on 6 MIR benchmarks (91.6% pianist identification from frozen embeddings)
- The question: Does Aria capture QUALITY, not just identity?

## Experiment Design

### Experiment Matrix

2 Aria variants evaluated on PercePiano with existing ByteDance AMT MIDIs:

| Set | Aria Variant | Dimensionality | Pooling |
|-----|-------------|----------------|---------|
| A | aria-medium-embedding | 512 | EOS token (built-in) |
| B | aria-medium-base | 1536 | Last token |

Plus a separate AMT comparison on 51 YouTube recordings (descriptive, no quality labels).

### Stage 0: AMT Comparison on YouTube Audio

**Input:** 51 YouTube wav files in `data/evals/youtube_amt/`

**Process:**
1. Run ByteDance AMT on all 51 files -> `data/midi/amt/youtube_bytedance/`
2. Run Aria-AMT on all 51 files -> `data/midi/amt/youtube_aria/`
3. Compare MIDI statistics: note count, mean velocity, velocity variance, onset density, pedal events
4. Extract Aria-embedding (512-dim) from both AMT outputs per recording
5. Compute cosine similarity per recording between the two AMT-derived embeddings

**Output:** Descriptive statistics table. Flag any recordings where cosine sim < 0.8 (major transcription disagreement).

### Stage 1: Embedding Extraction

**Input:** 1,202 PercePiano MIDI files in `data/midi/percepiano/` (ByteDance AMT)

**Process:**
1. Tokenize each MIDI with Aria's AbsTokenizer
2. Run forward pass through aria-medium-embedding -> 512-dim from EOS token
3. Run forward pass through aria-medium-base -> 1536-dim from last token
4. Save as `data/embeddings/percepiano/aria_embedding.pt` and `aria_base.pt`

**Format:** `{segment_id: tensor}` dict, matching existing `muq_embeddings.pt` convention.

### Stage 2: Linear Probe Evaluation

**Input:** Embedding sets A and B + `composite_labels.json` + `folds.json` (4 clean piece-stratified folds)

**Probe architecture:**
```
Input: frozen embedding (512 or 1536-dim)
  -> Linear(dim_in, 6)
  -> 6 scores (dynamics, timing, pedaling, articulation, phrasing, interpretation)
```

**Training:**
- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- Loss: MSE
- Epochs: 100, early stopping (patience=10, monitor val pairwise accuracy)
- Batch size: full dataset
- No data augmentation

**Metrics per fold (reusing existing MetricsSuite):**
- `pairwise_accuracy()` -- primary, excludes ambiguous pairs (|diff| < 0.05)
- `regression_r2()` -- secondary
- `per_dimension_breakdown()` -- 6-dim pairwise breakdown

**Reporting:** 4-fold mean +/- std. Side-by-side comparison table: Set A vs Set B vs MuQ baseline.

### Stage 3: Error Correlation

**Input:** Pairwise predictions from MuQ linear probe and best-performing Aria linear probe

**Process:**
1. For each validation pair across all folds, record correct/incorrect for both models
2. Compute phi coefficient (binary correlation)

**Interpretation:**
- phi < 0.50: models make different mistakes, fusion is promising
- phi > 0.70: models are redundant, fusion adds little value

## Decision Thresholds

| Result | Interpretation | Next Step |
|--------|---------------|-----------|
| Pairwise > 60% | Quality signal confirmed | Proceed to Phase B (contrastive pretraining + fine-tuning) |
| Pairwise 55-60% | Marginal | Try LoRA fine-tuning before deciding |
| Pairwise ~50% | No quality signal | Reconsider Aria strategy |
| Error corr < 0.50 | Complementary to MuQ | Fusion is promising |
| Error corr > 0.70 | Redundant with MuQ | Fusion unlikely to help |

## File Layout

New files only (no existing code modified):

```
model/src/model_improvement/
  aria_embeddings.py       # Stage 1: Tokenize + extract embeddings (both variants)
  aria_linear_probe.py     # Stage 2 + 3: Linear probe + error correlation
  aria_amt_compare.py      # Stage 0: AMT comparison on YouTube audio

model/data/embeddings/percepiano/
  aria_embedding.pt        # Set A: embedding variant, 512-dim
  aria_base.pt             # Set B: base variant, 1536-dim

model/data/midi/amt/
  youtube_bytedance/       # ByteDance AMT of YouTube audio (new)
  youtube_aria/            # Aria-AMT of YouTube audio (new)
```

## Dependencies

**New (via `uv add`):**
- `aria` (git clone from EleutherAI) -- AbsTokenizer + model architecture
- `safetensors` -- explicit (may already be transitive)

**Model weights (via `huggingface-cli download` to `data/weights/`):**
- `loubb/aria-medium-embedding` (~2.5GB)
- `loubb/aria-medium-base` (~2.5GB)
- `loubb/aria-amt` (~2.5GB)

**Already satisfied:**
- `transformers>=4.30.0`, `torch>=2.0.0`, `mido>=1.3.0`, `scipy`

## Compute Requirements

- No GPU required (650M params, short 8-bar MIDI segments)
- CPU forward pass: ~0.5s/segment (embedding), ~1s/segment (base)
- Total extraction: ~10-20 min for 1,202 segments x 2 variants
- AMT on 51 YouTube files: ~5-10 min
- Linear probe training: seconds

## Scope Exclusions

- **Skill bucket test:** Blocked on YouTube audio downloads + AMT. Deferred to T5 curation.
- **AMT comparison on PercePiano:** PercePiano is MIDI-only (no real audio). AMT comparison is only meaningful on real recordings.
- **LoRA fine-tuning:** Only if linear probe results are marginal (55-60%). Separate experiment.
- **Fusion architecture:** Designed in Phase D after validation confirms complementarity.
