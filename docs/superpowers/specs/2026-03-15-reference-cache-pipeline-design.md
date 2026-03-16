# Reference Cache Data Pipeline Design

**Date:** 2026-03-15
**Goal:** Generate reference performance profiles from MAESTRO recordings for all matched pieces in the score library (244 pieces in score_library/), and upload them to R2 so the API analysis engine can use them.

## Pipeline Overview

Three CLI subcommands run sequentially with a human review step between match and generate:

```
match    → maestro_asap_matches.csv (candidate mapping)
             ↓ human reviews CSV, marks approved/rejected
generate → references/v1/{piece_id}.json (one per piece)
upload   → wrangler r2 object put to SCORES bucket
```

Data flow:

```
MAESTRO CSV (1276 recordings)
  + ASAP piece IDs (244 pieces)
  → fuzzy match → candidate CSV (~100-200 matches)
  → human review → approved CSV
  → DTW alignment + validation (coverage >= 75%, DTW cost logged)
  → ReferenceProfile JSONs (one per piece, multiple performers aggregated)
  → wrangler r2 object put → R2 SCORES bucket (references/v1/)
```

## 1. Fuzzy Matching (`match` command)

### Composer Normalization

Map MAESTRO `canonical_composer` to ASAP composer prefix via last-name case-insensitive comparison. The set of known ASAP composers is extracted from score library filenames (first segment before `.`).

Example mappings:
- "Frederic Chopin" → "chopin"
- "Johann Sebastian Bach" → "bach"
- "Ludwig van Beethoven" → "beethoven"

This immediately filters the search space: a Chopin recording only matches Chopin pieces.

### Title Normalization

Both MAESTRO titles and ASAP piece ID components are normalized:
- Lowercase, strip accents (unicodedata NFKD)
- Opus references: "Op. 10", "op.10", "Opus 10" → "op_10"
- Number extraction: "No. 3", "No.3", "Nr. 3" → "3"
- Common prefix stripping: "Piano Sonata" → "sonata", "Etude"/"Etudes" → "etude"
- Catalog numbers: "BWV 846" → "bwv_846", "K. 331" → "k_331", "D. 899" → "d_899"

### Scoring

For each MAESTRO entry within a composer group, compute token overlap + bigram Dice similarity against each ASAP piece's title components. Output the best match with a confidence score.

### Multi-piece Detection

Flag MAESTRO titles containing range indicators ("No. 13-24", "Nos. 1-6", "Books I & II") as `multi_piece=true`. These still get matched to their best single-piece candidate but are flagged for closer human review.

### Output

Primary: `maestro_asap_matches.csv`
```
maestro_composer, maestro_title, midi_filename, duration_s,
asap_piece_id, asap_title, confidence, multi_piece, status
```

`status` starts blank. Reviewer fills in `approved` or `rejected`. Rows with confidence >= 0.8 and `multi_piece=false` are strong candidates for batch approval.

Secondary: `unmatched_maestro.csv` listing entries below minimum confidence threshold (0.2) against any ASAP piece. Useful for identifying coverage gaps.

## 2. Generation (`generate` command)

### Per-piece Processing

1. Group approved CSV rows by ASAP piece_id to collect all MAESTRO MIDI paths per piece.
2. Load score JSON from `data/score_library/{piece_id}.json` for ground truth note events and bar structure.
3. For each MAESTRO MIDI:
   - Parse with `mido`
   - Extract note onsets (pitch, time, velocity) and pedal CC64 events
   - Run onset+pitch DTW alignment against the score
   - **Validation gate:**
     - **Coverage (primary):** `len(aligned_bars) / len(score["bars"])` where `aligned_bars` is the set of score bar numbers that received at least one aligned note onset from the DTW output. Empty bars in the score (bars with only rests) count toward the denominator -- this is conservative and acceptable because MAESTRO pieces are complete performances. Reject if coverage < 75%.
     - **DTW cost (logged, not enforced on first run):** The `dtw-python` library's `normalizedDistance` (total cost divided by path length). Log the value for all recordings. After first run, review the distribution and set an absolute ceiling for subsequent runs.
   - If passed: compute per-bar `BarStats` (velocity mean/std, onset deviation mean/std, pedal duration, pedal changes, note duration ratio)
4. Aggregate across performers: for each bar, average per-performer stats, track performer_count.
5. Write `ReferenceProfile` JSON to `data/reference_profiles/references/v1/{piece_id}.json`.

### First-run Strategy

Since DTW cost distributions are unknown:
- Apply coverage gate (75%) -- interpretable without calibration
- Log DTW costs but do not reject on cost
- After first run, review cost distribution, set threshold, re-run with both gates

### Generation Report

After all pieces, write `generation_report.csv`:
```
piece_id, total_recordings, passed_validation, rejected_coverage,
rejected_dtw_cost, performer_count, mean_coverage, mean_dtw_cost
```

## 3. Upload (`upload` command)

Wrapper around `wrangler r2 object put`. For each JSON in the output directory:

```bash
wrangler r2 object put crescendai-bucket/references/v1/{piece_id}.json \
  --file=data/reference_profiles/references/v1/{piece_id}.json \
  --content-type=application/json
```

- Sequential execution (50-150 files)
- Log each upload success/failure
- Raise on first failure (no silent skips)
- Summary: "Uploaded N files"

## 4. CLI Interface

All three commands as subcommands of `reference_cache.py`:

```bash
# Step 1: Generate candidate matches
uv run python -m src.score_library.reference_cache match \
  --maestro-csv data/maestro_cache/maestro-v3.0.0.csv \
  --score-dir data/score_library \
  --output data/reference_profiles/maestro_asap_matches.csv

# ... human reviews CSV, fills in status column ...

# Step 2: Generate reference profiles
uv run python -m src.score_library.reference_cache generate \
  --matches data/reference_profiles/maestro_asap_matches.csv \
  --maestro-dir data/maestro_cache \
  --score-dir data/score_library \
  --output-dir data/reference_profiles/references/v1

# Step 3: Upload to R2
uv run python -m src.score_library.reference_cache upload \
  --source-dir data/reference_profiles/references/v1 \
  --bucket crescendai-bucket \
  --prefix references/v1
```

## 5. Manual Validation

After upload, spot-check 3-5 known pieces:
1. Download from R2: `wrangler r2 object get crescendai-bucket/references/v1/{piece_id}.json`
2. Verify: performer_count > 1, velocity ranges plausible (40-120 MIDI velocity), pedal patterns exist for Romantic pieces, bar count matches score total_bars

## JSON Schema Contract

The output JSON must match the Rust consumer's `ReferenceProfile` / `ReferenceBar` structs exactly. Field names, types, and nullability:

```json
{
  "piece_id": "string",
  "performer_count": "u32 (>= 1)",
  "bars": [
    {
      "bar_number": "u32 (>= 1)",
      "velocity_mean": "f64",
      "velocity_std": "f64",
      "onset_deviation_mean_ms": "f64",
      "onset_deviation_std_ms": "f64",
      "pedal_duration_mean_beats": "f64 | null",
      "pedal_changes": "u32 | null (non-negative)",
      "note_duration_ratio_mean": "f64",
      "performer_count": "u32 (>= 1)"
    }
  ]
}
```

All numeric values must be non-negative (the Python generation step must validate this before serialization). The Rust consumer deserializes `pedal_changes` as `Option<u32>`, so a negative value would cause a deserialization failure.

## Single-Performer References

Pieces with only 1 validated MAESTRO recording still get a reference profile (performer_count=1). The consumer analysis engine does not gate on performer_count, so these profiles will produce reference comparisons. However, the `velocity_std` for a single performer represents intra-bar variance, not inter-performer consensus -- the "reference range" will be tighter than it would be with multiple performers.

Mitigation: the generation_report.csv flags these pieces. During manual validation, decide whether to keep or exclude single-performer references. For the first run, include them (some reference data is better than none), but log a warning.

## Error Handling Policy

- **match command:** Raise on malformed CSV or missing score files. Log and skip individual MAESTRO entries that fail normalization (with a count in stdout summary).
- **generate command:** Per-recording exceptions (MIDI parse failure, DTW failure) are logged to generation_report.csv with the specific error, not silently swallowed. The piece continues processing with remaining recordings. If ALL recordings for a piece fail, no JSON is written and the piece appears in the report with `passed_validation=0`.
- **upload command:** Raise on first wrangler failure. No silent skips.

## Idempotency

The `generate` command overwrites existing JSON files for a given piece_id. Re-running with a corrected CSV produces fresh output. This is intentional -- the CSV is the source of truth, and the JSONs are derived artifacts.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Multi-piece recordings | Map + validate | Attempt matching but reject via DTW quality gate (coverage < 75%) |
| Matching algorithm | Two-pass hybrid | Algorithmic fuzzy match produces candidates; human reviews CSV before generation |
| DTW validation | Coverage + DTW cost | Coverage (75%) as primary gate; DTW cost logged first run, enforced after calibration |
| Upload mechanism | wrangler CLI | Simple, no new dependencies, sufficient for 50-150 files |
| Review artifact | CSV file | Human-reviewable, version-controllable, permanent audit trail |

## Files Modified

- `model/src/score_library/reference_cache.py` -- near-complete rewrite. The existing code has a flat `main()` with no subcommands, a scaffold `_find_maestro_midis_for_piece` that prints "not yet implemented", no validation gates, and no CSV workflow. The rewrite replaces the CLI with three subcommands (`match`, `generate`, `upload`), adds the fuzzy matching engine, CSV-based workflow, validation gates, and generation reporting. The existing DTW alignment logic (`align_to_score`) and `BarStats`/`ReferenceProfile` dataclasses are preserved as-is.

## Output Artifacts

- `data/reference_profiles/maestro_asap_matches.csv` -- match candidates (committed after review)
- `data/reference_profiles/unmatched_maestro.csv` -- unmatched recordings
- `data/reference_profiles/generation_report.csv` -- generation quality metrics
- `data/reference_profiles/references/v1/{piece_id}.json` -- reference profiles (uploaded to R2, not committed)
