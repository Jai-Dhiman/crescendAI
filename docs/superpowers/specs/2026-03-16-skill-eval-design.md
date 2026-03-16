# Skill Eval -- Model Quality Validation Across Skill Levels

> **Status:** Design spec. Tests whether MuQ A1-Max dimension scores meaningfully predict pianist skill level, and quantifies quality loss from faster inference configs.

## Problem

The A1-Max model (80.8% pairwise accuracy) was trained and evaluated on PercePiano -- advanced-level performances rendered via Pianoteq. Before optimizing inference latency (e.g., single fold, skip AMT) or trusting the model's scores for real-world feedback, we need ground truth: do the scores actually separate a beginner from an advanced player on real YouTube audio?

## Goals

1. Validate that model scores correlate with pianist skill level on real (non-synthesized) recordings
2. Quantify per-dimension signal strength (which dimensions best predict skill?)
3. Measure quality loss from faster inference configs to make informed latency/quality tradeoffs

## Non-Goals

- No manual listening or labeling (metadata-derived labels only)
- No model retraining (tests existing A1-Max and variants)
- No pipeline eval (STOP classifier, teaching moments, LLM feedback are separate)
- No AMT accuracy eval (already validated at 79.9%)
- No expansion beyond 2 pieces (future work if results are promising)

---

## Design

### Section 1: Data Collection

**Two pieces, controlled for difficulty:**
- **Fur Elise (Beethoven)** -- massive beginner-to-intermediate YouTube coverage
- **Nocturne Op. 9 No. 2 (Chopin)** -- massive intermediate-to-professional coverage

**Target:** 50 recordings per piece, 100 total.

**Skill buckets (metadata-derived):**

| Bucket | Label | Metadata signals |
|--------|-------|-----------------|
| 1 | Beginner | "beginner," "first year," "learning," "grade 1-2," "progress" |
| 2 | Early intermediate | "grade 3-4," "2-3 years," "ABRSM 3-4," "RCM 3-4" |
| 3 | Intermediate | "grade 5-6," "ABRSM 5-6," no qualifier (default for unlabeled non-pro channels) |
| 4 | Advanced | "grade 7-8," "ABRSM 7-8," "diploma," "conservatory," music school recitals |
| 5 | Professional | Known concert pianists, competition performances, verified professional channels |

**Target distribution:** ~10 recordings per bucket per piece. Fur Elise will skew toward buckets 1-3. Op. 9 No. 2 will skew toward buckets 3-5. Together they cover the full range.

**Search strategy per piece:**
- `"{piece}" beginner piano` -- targets bucket 1
- `"{piece}" piano progress` -- targets buckets 1-2
- `"{piece}" piano` (sorted by view count, skip top professionals) -- targets bucket 3
- `"{piece}" piano recital` -- targets buckets 3-4
- `"{piece}" piano concert` / known artist names -- targets bucket 5

**Filtering criteria:**
- Solo piano only (no duets, accompaniment, MIDI playback)
- Real recording (no synthetic visualizer-only)
- Audible piano audio (not tutorial talking or sheet music display)
- Duration within expected range (Fur Elise: 2-5 min, Op. 9 No. 2: 3.5-6 min)
- No duplicate performances (same pianist, same take re-uploaded to different channels)

**Labeling approach:** Automated keyword matching on title + description. Keywords are matched case-insensitively. When no keyword matches and the channel is not a known professional, assign bucket 3 (intermediate) as default. The manifest includes `label_rationale` for each assignment so labels can be spot-checked.

**Known confound:** Beginner recordings tend to have worse audio quality (phone mic, room reverb) than professional recordings (studio mastered). The model may partially measure recording quality rather than playing quality. This is acknowledged as a limitation -- the eval measures whether the model's scores correlate with skill level *in the conditions users will actually record in*, which is the relevant product question.

**Robustness:** The manifest tracks download status. Videos that become unavailable after manifest creation are excluded from analysis. Minimum 40 successfully downloaded recordings per piece required to proceed.

**Disk space:** ~3 GB for 100 WAV files (24kHz mono, 3-6 min each). Results JSONs add ~100 MB per config. Total ~5 GB across all configs.

**Output:** YAML manifest per piece at `model/data/skill_eval/{piece_id}/manifest.yaml`:

```yaml
piece: fur_elise
recordings:
  - video_id: abc123
    title: "Fur Elise - 6 months progress"
    channel: "Piano Learner"
    duration_seconds: 180
    skill_bucket: 1
    label_rationale: "title contains 'progress', '6 months'"
  - video_id: def456
    title: "Beethoven Fur Elise - Lang Lang"
    channel: "Deutsche Grammophon"
    duration_seconds: 195
    skill_bucket: 5
    label_rationale: "known concert pianist, professional label channel"
```

**Script:** `apps/evals/model/skill_eval/collect.py`
- Uses `yt-dlp --dump-json` for metadata search and keyword-based skill labeling
- Downloads audio via `yt-dlp -x --audio-format wav`
- Stores audio at `model/data/skill_eval/{piece_id}/audio/{video_id}.wav`
- Adds 2-second delay between downloads to avoid YouTube throttling

### Section 2: Inference & Scoring

**Processing per recording:**
1. Chunk audio into 15-second segments (reuse `audio_chunker.py` from `apps/inference/`)
2. Run each chunk through inference (reuse `run_inference_on_chunk()` from `eval_runner.py`)
3. Aggregate per-recording scores: mean across all chunks, weighted equally (last chunk may be shorter but still gets equal weight -- simpler and sufficient for this eval)
4. Minimum 2 successful chunks per recording to include in analysis

**Cross-package imports:** The scripts add `apps/inference/` to `sys.path` to import `audio_chunker`, `models.loader`, `models.inference`, `models.transcription`, and `preprocessing.audio`. This matches the approach used by `eval_runner.py`.

**Inference configs -- two categories:**

*Quality variants (different predictions):*

| Config | Description | What changes |
|--------|------------|-------------|
| `ensemble_4fold` | Full A1-Max (baseline) | All 4 fold heads, MPS, AMT |
| `single_fold_0` | Fold 0 only | 1 head, different scores |
| `single_fold_best` | Best val-pairwise fold (fold 0, 77.7% pairwise) | 1 head, different scores |

*Latency-only variants (identical or near-identical predictions):*

| Config | Description | Notes |
|--------|------------|-------|
| `no_amt` | Skip AMT transcription | MuQ scores are independent of AMT -- predictions identical to ensemble. Only measures AMT's share of latency. |
| `cpu_only` | Force CPU | Scores near-identical (float precision differences). Measures MPS vs CPU latency. |

Quality variants run on the full 100 recordings. Latency-only variants run on a 10-recording subset (enough to measure timing, no need for full correlation analysis since scores don't change).

**ModelCache singleton handling:** `ModelCache` in `models/loader.py` is a singleton. To switch between configs, `run_inference.py` resets `ModelCache._instance = None` and `ModelCache._initialized = False` before each config, then calls `initialize()` with the appropriate fold subset. For single-fold configs, after initialization, it trims `cache.muq_heads` to keep only the target fold.

**Per-fold validation accuracies** (from `03-encoders.md`):

| Fold | Pairwise Acc |
|------|-------------|
| 0 | 77.7% |
| 1 | 70.3% |
| 2 | 76.6% |
| 3 | 71.1% |

`single_fold_best` uses fold 0 (77.7%).

**Output per config:** `model/data/skill_eval/{config_name}/{piece_id}/results.json`:

```json
{
    "config": "ensemble_4fold",
    "piece": "fur_elise",
    "recordings": [
        {
            "video_id": "abc123",
            "skill_bucket": 1,
            "chunk_count": 12,
            "mean_scores": {"dynamics": 0.45, "timing": 0.38, ...},
            "per_chunk_scores": [
                {"chunk_index": 0, "predictions": {...}, "processing_time_ms": 31000}
            ],
            "mean_processing_time_ms": 31000
        }
    ]
}
```

**Script:** `apps/evals/model/skill_eval/run_inference.py`
- Takes `--config` and `--piece` flags
- Caches results per recording (skip if already inferred for this config)
- Can run all configs sequentially or a single config for quick iteration
- Recordings that fail to chunk or have fewer than 2 successful chunks are logged and excluded

### Section 3: Analysis & Metrics

**Primary metrics (does the model predict skill?):**

| Metric | What it measures | Computation |
|--------|-----------------|-------------|
| Spearman rho (overall) | Do mean scores rank-order with skill? | Correlation between mean(6 dims) and bucket label (1-5), computed per-piece |
| Spearman rho (per-dim) | Which dimensions best predict skill? | Per-dimension correlation with bucket label |
| Spearman rho (best dim) | What's the ceiling for a single dimension? | Best single-dimension correlation (avoids dilution from noisy dims) |
| Bucket separation (Cohen's d) | Can the model tell coarse levels apart? | Effect size between merged buckets: (1+2) vs (3) vs (4+5). Merging gives ~20 per group, enough for descriptive effect sizes. |
| Confusion rate | How often does lower-skill outscore higher-skill? | % of cross-bucket pairs with inverted scores, with permutation-test baseline |

**Config comparison metrics (single-fold quality variants only):**

| Metric | What it measures |
|--------|-----------------|
| Rank correlation vs ensemble | Does single-fold rank recordings the same way as 4-fold? |
| Score MAE vs ensemble | Raw score shift (tells you if calibration changes) |
| Spearman rho delta | Does single-fold lose skill correlation? |

All metrics reported with 95% bootstrap confidence intervals (1000 resamples) to distinguish real differences from noise at n=50.

**Visualization:** Summary plot per piece -- mean score (y-axis) vs skill bucket (x-axis) with error bars, one line per config. Saved to `model/data/skill_eval/figures/`.

**Script:** `apps/evals/model/skill_eval/analyze.py`
- Reads results JSONs from all configs
- Prints metrics table to stdout
- Saves plots to figures directory

**Example output:**
```
=== Fur Elise (50 recordings) ===
Overall Spearman rho: 0.72 (p<0.001)
  dynamics:      0.58
  timing:        0.81
  pedaling:      0.65
  articulation:  0.74
  phrasing:      0.69
  interpretation: 0.51

Bucket separation (Cohen's d):
  beginner vs early-int:  1.2
  early-int vs int:       0.6
  int vs advanced:        0.4
  advanced vs pro:        0.3

Config comparison (vs ensemble_4fold):
  single_fold_0:    rank_corr=0.97, score_MAE=0.02, rho_delta=-0.03
  single_fold_best: rank_corr=0.98, score_MAE=0.01, rho_delta=-0.01

Latency summary (10-recording subset):
  ensemble_4fold:   31.0s/chunk (baseline)
  single_fold_best: 12.1s/chunk (-61%)
  no_amt:           22.3s/chunk (-28%)
  cpu_only:         45.2s/chunk (+46%)
```

---

## Files

| File | Purpose | Type |
|------|---------|------|
| `apps/evals/model/skill_eval/collect.py` | YouTube search, metadata parsing, audio download, manifest generation | New |
| `apps/evals/model/skill_eval/run_inference.py` | Run inference configs on downloaded audio, cache results | New |
| `apps/evals/model/skill_eval/analyze.py` | Compute metrics, generate plots | New |
| `model/data/skill_eval/fur_elise/manifest.yaml` | Recording manifest for Fur Elise | Generated |
| `model/data/skill_eval/nocturne_op9no2/manifest.yaml` | Recording manifest for Op. 9 No. 2 | Generated |
| `model/data/skill_eval/{config}/{piece}/results.json` | Inference results per config per piece | Generated |
| `model/data/skill_eval/figures/` | Analysis plots | Generated |

---

## Execution Order

1. **Collect:** Run `collect.py` for both pieces. Review manifests -- spot-check ~10 label assignments per piece.
2. **Infer (baseline):** Run `run_inference.py --config ensemble_4fold` on both pieces. ~8 hours per piece on M4 MPS at 31s/chunk (run overnight). Can also parallelize by running both pieces simultaneously if memory allows.
3. **Analyze (baseline):** Run `analyze.py`. Check if scores correlate with skill at all. If rho < 0.3, the model doesn't work for this and the rest of the eval is moot.
4. **Infer (quality variants):** Run `single_fold_0` and `single_fold_best` on both pieces (~3 hours each at ~12s/chunk).
5. **Infer (latency variants):** Run `no_amt` and `cpu_only` on 10-recording subset (~30 min each).
6. **Analyze (comparison):** Run `analyze.py` with all configs. Determine which configs preserve quality.

---

## Success Criteria

- **Model works:** Overall Spearman rho > 0.5 on at least one piece
- **Dimensions differentiate:** At least 3 of 6 dimensions have rho > 0.4
- **Single-fold viable:** Rank correlation vs ensemble > 0.95 (can safely cut latency by ~3x)
- **Bucket separation:** Cohen's d > 0.5 for at least the beginner-vs-intermediate boundary
