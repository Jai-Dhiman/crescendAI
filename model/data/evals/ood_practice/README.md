# Out-of-Distribution Practice Testing Set

This directory holds a small corpus of practice-domain recordings captured
outside the piece-stratified fold structure (`data/labels/percepiano/folds.json`).
It exists so the harness can ever report practice-domain performance, which
piece-stratified CV cannot measure.

## Status

Scaffold only — no recordings committed. Populate when you have 30+ clips.

## Capture protocol

**Device.** Phone (iPhone or similar) in a practice room. No studio mic.
**Environment.** Your actual practice space, with your actual piano. Room
reverb, nearby noise (page turns, pedal thumps, creaky bench) is desirable —
that's the distribution we need to measure on.

**Content mix.** Target roughly:
- 50% weak playing (early stages of learning a passage, many stops/restarts)
- 30% mid playing (piece known but not performance-ready)
- 20% strong playing (piece you'd play for a friend)

Avoid recording only your best takes. The harness needs weak signal.

**Length.** 15-30 seconds per clip. Trim so only one skill level is in each clip.

**Minimum for first report.** 30 clips. The harness returns
`{"skipped": "empty_ood_dataset"}` below that.

## Label schema

Single-ordinal 1-5 (same scale as T5 YouTube Skill labels) — NOT per-dimension.
This matches existing labeling capacity under the solo-dev constraint; per-dim
relabeling is out of scope (see `docs/model/06-label-quality.md`).

### `labels.json` format

```json
{
  "practice_2026_04_20_001": {
    "ordinal": 2,
    "scores": [0.35, 0.40, 0.30, 0.35, 0.33, 0.32]
  },
  "practice_2026_04_20_002": {
    "ordinal": 4,
    "scores": [0.72, 0.68, 0.70, 0.74, 0.71, 0.70]
  }
}
```

- `ordinal`: integer 1-5 (single bucket you assign while labeling)
- `scores`: derived 6-vector in `[0, 1]`, same derivation as
  `data/labels/composite/composite_labels.json` (ordinal broadcast across the
  6 dims with a small calibration offset)

## Embedding extraction

Embeddings live in `embeddings/{segment_id}.pt` as `[T, D]` float tensors. Use
the existing MuQ extraction pipeline (see `scripts/extract_muq_embeddings.py`)
pointing `--input-dir` at the raw WAV directory for this OOD set.

## Running the OOD test

```python
from pathlib import Path
import torch
from model_improvement.ood_harness import OODDataset, run_ood_test
from src.paths import Evals

ds = OODDataset(
    cache_dir=Evals.root / "ood_practice" / "embeddings",
    labels_path=Evals.root / "ood_practice" / "labels.json",
)

result = run_ood_test(
    model=my_model,
    ood_dataset=ds,
    encode_fn=lambda m, inp, mask: m.encode(inp, mask),
    compare_fn=lambda m, z_a, z_b: m.compare(z_a, z_b),
    predict_fn=lambda m, inp, mask: m.predict_scores(inp, mask),
)
```

The result dict carries every field `evaluate_model()` emits plus `n_samples`
and `ood_source`. Compare clean-fold `pairwise_mean` against OOD `pairwise`;
degradation > 15 pp is a domain-shift red flag (target for year 2: degradation
< 10 pp after practice-distribution augmentation lands).

## What this set is NOT

- Not a substitute for PercePiano per-dim labels — has none.
- Not used in training or fold selection — strictly held-out.
- Not rater-diverse — single labeler (you). Drift is tracked separately via
  `scripts/t5_label_consistency.py`.
