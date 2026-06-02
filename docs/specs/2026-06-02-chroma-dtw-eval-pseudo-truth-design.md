# Chroma-DTW Eval Harness Rework: Practice-Corpus + AMT-Pseudo-Truth

**Goal:** Repoint the chroma-DTW eval harness primary scalar from MAESTRO+parangonar gold-truth at 50ms tolerance to practice-corpus + AMT-pseudo-truth at +/-1.5s tolerance, with an on-disk pseudo-truth cache so `just chroma-eval-verify` runs in <=120s without calling AMT inline.

**First baseline piece:** `bach_prelude_c_wtc1` (Bach Prelude in C, BWV 846). Single 120bpm tempo marking (see `model/data/scores/bach.prelude.bwv_846.json`). Constant-tempo score → score `onset_seconds` are the score-audio-time axis directly; no partitura projection is needed. 21 approved practice videos already labeled in `model/data/evals/practice_eval/bach_prelude_c_wtc1/candidates.yaml`.

**Not in scope:**
- Variable-tempo score support — see "Variable-tempo score support (future)" below.
- Score sourcing for the other 14 practice_eval pieces beyond bach_prelude (opportunistic fur_elise pull is task C4, time-boxed 30 min, skipped if not findable).
- HF AMT endpoint deploy (local AMT is sufficient for cache regen).
- DTW improvements (those run via `/autoresearch` after this rework ships).
- AMT model itself (we measure DTW under realistic AMT noise; AMT changes are a separate research track).
- Wiring an end-to-end `--corpus` real-run path beyond the practice corpus; the fixture-based smoke path and the practice-corpus path are the only two supported.

## Variable-tempo score support (future)

This rework assumes single-tempo scores for the first baseline (bach prelude has one 120bpm marking). When a 2nd piece with variable tempo (e.g., chopin pieces with rubato/tempo markings) is added, one of the following must be implemented:
1. **Per-piece tempo map reconciliation:** build a `(beat → score_audio_sec_dtw)` map from the score JSON's `tempo_markings`; project a synthetic perf-na using that piece-specific map.
2. **Fixed-tempo score chroma renderer:** render the DTW score chroma at a fixed tempo, ignoring score tempo markings (loses musical intent; not recommended).
3. **Beat-space cache (RECOMMENDED):** cache `(perf_audio_sec, score_beat)` pairs; the verify CLI converts `predicted_score_frame → score_audio_sec_dtw → score_beat` via the score JSON's `tempo_markings`, and matches against the cached score_beat.

Option 3 is the recommended path because it decouples the cache from any specific tempo assumption; the conversion lives in the consumer (verify), the cache stores the most stable axis (beats). Implement when the 2nd piece lands.

## Problem

The harness shipped 2026-05-31 (`3305babf`) computes its primary scalar over MAESTRO Disklavier audio + parangonar gold-truth at +/-50ms. Production audio is amateur phone-recorded practice, not studio captures. The reproduced DTW failures (teleport, silence-lock, cost-not-confidence) all happened on amateur YouTube audio; optimizing against MAESTRO does not move the metric users experience. The pilot in `docs/implementation/2026-06-01-amt-pseudo-truth-pilot.md` validated that AMT-pseudo-truth on practice audio has an internal jitter floor of ~60-80ms (well below 1.5s) and structural variability that swamps a 50ms tolerance regardless of DTW quality. Until the harness pivots, every `/autoresearch` iteration would optimize the wrong target.

The pilot script also calls AMT inline. At one-shot pilot scale (3 clips x ~5 min) that is fine; at `/autoresearch` scale (hundreds of iterations) it is unworkable.

## Solution (from the user's perspective)

`just chroma-eval-verify` keeps the same contract: one float on stdout, exit 0 iff no guard regressed, sidecar JSON for diagnostics. Internally it now compares DTW-predicted audio time to AMT-pseudo-truth audio time on practice audio, tolerant to +/-1.5s, with the AMT step pre-computed and cached on disk. A new `just amt-regen-pseudo-truth` recipe rebuilds the cache when the practice corpus grows or the pinned AMT checkpoint bumps. Day-to-day `/autoresearch` candidates never touch AMT.

## Design

### Primary scalar

For each chunk in the manifest:
1. Run DTW on the chunk's audio chroma against the score; receive `BarMapChroma` including `score_frame_per_audio_frame: Vec<u32>` (added in commit `758ff953`).
2. For the chunk's last audio frame, look up the predicted score frame from that vector; convert to predicted score time via the score-frame rate, then to predicted audio time via the pseudo-truth's inverse interpolation (score_sec -> audio_sec is monotone since pseudo-truth pairs are running-max).
3. Pseudo-truth provides the ground-truth audio_sec at that same score_sec.
4. Error = `abs(predicted_audio_sec - pseudo_audio_sec)`. Chunk passes iff error <= 1.5s.

Primary = percent of practice chunks passing.

### Pseudo-truth cache (deep module)

- **Interface:** `load_pseudo_truth(piece_id, video_id, *, audio_sha256, amt_checkpoint_hash, score_sha256, parangonar_version, cache_root) -> PseudoTruth` and `write_pseudo_truth(piece_id, video_id, payload, cache_root) -> Path` and `cache_path(cache_root, piece_id, video_id) -> Path` (PUBLIC, used by amt_regen and chunk_sampler). `PseudoTruth` carries `perf_audio_sec: np.ndarray`, `score_audio_sec: np.ndarray` (since bach prelude is single-tempo, these are score seconds, not divs), `measure_table: list[dict]`, and the metadata used as the cache key. Helpers `audio_sec_to_score_sec(t)` + inverse `score_sec_to_audio_sec(s)` expose monotone interpolation.
- **Cache key:** `(audio_sha256, amt_checkpoint_hash, score_sha256, parangonar_version)`. All four are stored in the JSON body and checked on load; any mismatch raises `PseudoTruthMismatchError`. Score hash + parangonar version close the silent-staleness vector flagged in the challenge review.
- **Hides:** on-disk JSON schema, cache-key construction, schema validation, hash-mismatch invalidation, atomic write via tempfile-rename.
- **Tested through:** writes payload via the public writer, reads it back via the public loader, asserts round-trip equality; loader raises `PseudoTruthMissingError` when the file is absent; loader raises `PseudoTruthMismatchError` when any of the four key fields disagrees with the file's stored key.

### `amt_regen` orchestrator (deep module)

- **Interface:** `regenerate_pseudo_truth(piece_id, video_id, *, score_path, audio_path, amt_url, amt_checkpoint_hash, parangonar_version, cache_root, force=False) -> RegenResult`. CLI wrapper at `python -m chroma_dtw_eval.amt_regen --piece ... --video-id ... --amt-url ...`. `just amt-regen-pseudo-truth piece video_id` is a thin shim.
- **Score loading:** the bach prelude JSON (`model/data/scores/bach.prelude.bwv_846.json`) is loaded directly — no partitura, no MXL→JSON build. The orchestrator iterates `bars[].notes[]` and builds a synthetic score note-array with `onset_sec`, `pitch`, and `onset_beat` (computed from `onset_tick / ticks_per_beat`, where `ticks_per_beat` is inferred from the JSON: the first bar's `start_tick` is 0, the second bar's `start_tick` is 4 beats later in 4/4, so `ticks_per_beat = (bars[1].start_tick - bars[0].start_tick) / 4`). `score_audio_sec` is read directly from each matched note's `onset_seconds` field (single-tempo identity, no projection bpm). The score's declared bpm (120 for bach prelude) is used only to provide parangonar's matcher with a `onset_beat` axis.
- **Hides:** 27s non-overlapping chunking, base64 WAV encoding, AMT POST + skip-on-tokenizer-error per chunk, AMT note concatenation with `chunk_idx * 27` offset, JSON score load + measure-table extraction, parangonar `AutomaticNoteMatcher` invocation, label=='match' filtering, monotonic running-max on score_audio_sec, idempotence check via cache-key compare.
- **Error discipline:** `requests.RequestException` on the AMT POST is fatal (`AmtRegenError`); a documented AMT tokenizer-error payload (200 with `error` field) is the only condition that skips a chunk. Minimum coverage gate: if final matched-pairs count < 100 OR `len(matched_pairs) / len(amt_notes) < 0.5`, raise `LowCoverageError` rather than writing a sparse pseudo-truth.
- **Tested through:** an integration test that drives the orchestrator with a stub AMT URL (a local `httpserver` fixture returning canned `midi_notes`) against a tiny JSON score fixture and asserts the cache file is created with the expected key, that a second call with the same inputs is a no-op, and that the loader returns interpolation arrays matching the orchestrator's in-memory output.
- Reads `model/config/amt_version.json` for the committed `amt_checkpoint_hash` AND `parangonar_version`. If AMT `/health` exposes a `checkpoint_hash` field, the orchestrator refuses to write when they disagree (`AmtCheckpointMismatchError`). Today's `/health` does not expose it, so the orchestrator records the committed hash unconditionally and the cross-check is exercised only when AMT is upgraded to publish it.

**Note (implementation):** the legacy `gold_truth_builder.py` module had a latent variable-tempo bug (`bpm=100.0` hardcoded projection at line 86). Audit deferred — module is being removed in Group A.

### Chunk sampler (internals rewritten, public surface stable)

- **Interface (unchanged):** `sample_chunks(pieces, n_per_piece, chunk_len_s, seed) -> list[Chunk]`. `PieceSpec` and `Chunk` dataclasses stay; `Chunk` gains an optional `video_id: str | None` field defaulting to `None` so existing callers do not break.
- **New helper:** `sample_practice_chunks(corpus_root, cache_root, n_per_piece, chunk_len_s, seed) -> list[Chunk]` enumerates `practice_eval/<piece>/candidates.yaml` (filter `approved: true`), cross-references that `pseudo_truth/<piece>/<video_id>.json` exists for each candidate, and stratifies positions within each clip into the existing 5 buckets (`intro`/`early`/`middle`/`late`/`cadence`). Skips clips with missing pseudo-truth (raises `PseudoTruthCoverageError` listing the gaps when zero clips qualify for a piece).
- **Tested through:** given a fake `practice_eval/<piece>/candidates.yaml` + matching `pseudo_truth/<piece>/<video_id>.json`, the helper returns a deterministic chunk list with the expected count and position distribution; given a missing pseudo-truth file, the helper raises.

### Metric aggregator (modified)

- **Interface drift:** `aggregate(results, baseline, *, frame_rate_hz, tolerance_s)` replaces `tolerance_ms`; the unit is seconds throughout, primary tolerance defaulting to 1.5. `ChunkResult.error_seconds: Optional[float]` replaces `error_frames` for the practice-pseudo path; existing fields (`bar_distance_from_forward`, `silence_loud_failure`) keep their semantics for G1/G3/G5.
- **G4 repurposed → consecutive-chunk continuity guard:** `practice_compose.py` (synthetic-MAESTRO composition) is still deleted. A new G4-named guard replaces it: for the same `(piece_id, video_id)`, consecutive sampled chunks at offsets `t_n`, `t_{n+1}` must satisfy `|predicted_score_audio_sec(n+1) - predicted_score_audio_sec(n) - (t_{n+1} - t_n)| <= 5.0` seconds. Built from the practice corpus itself; no synthetic data needed. Catches the stitch-regression class the previous G4 caught.
- **G1/G2:** unchanged formulas; G2's positive-label condition becomes `error_seconds > tolerance_s` (was frames). G2 regression threshold scales by `sqrt(50 / max(n_chunks, 1))` capped at 4× to widen tolerance when chunk counts are small.
- **G3/G5:** unchanged.
- **Tested through:** synthesize a `ChunkResult` list mixing `practice` (pass + fail by tolerance), `silence`, `amateur`, and `real_practice`; assert primary matches the percentage of practice passes; assert G2 is finite when costs split the error labels; assert that exceeding the primary baseline plus epsilon does NOT regress while dropping below DOES regress; assert the consecutive-chunk continuity guard fires when predicted score-audio-sec for adjacent chunks diverges by > 5s from the audio-timeline delta.

### Verify CLI (contract unchanged)

- Same flags: `--baseline`, `--fixtures`, `--corpus`, `--sidecar`.
- Stdout: one float (the primary) on a single line; nothing else.
- Stderr: when the practice manifest has < 2 pieces, emit a `WARNING: smoke-only baseline (n=<k> piece(s)); /autoresearch dispatch deferred until ≥ 2 pieces have scores` line. Documents n=1 as smoke signal, not research signal.
- Exit: 0 iff no guard regressed; nonzero otherwise.
- Sidecar JSON: `{primary, guards{g1,g2,g3,g5}, baseline, regressed, n_chunks, error_seconds_distribution: {p50, p90, p95, max, mean}, tolerance_sensitivity: {0.5: pct, 1.0: pct, 1.5: pct, 2.0: pct, 3.0: pct}}` — distribution + tolerance sweep derived from per-chunk error_seconds for tolerance calibration.
- **Cache key construction:** the CLI computes the real `audio_sha256` (via `hashlib.sha256` on the wav file) for each chunk's `(piece_id, video_id)` and reads `amt_checkpoint_hash` + `parangonar_version` + `score_sha256` from `model/config/amt_version.json` + the score JSON before calling `load_pseudo_truth`. No `"z"*16` placeholders in the real path.
- Internals: when `--corpus` is given and points to `model/data/evals/`, the CLI calls `sample_practice_chunks` + per-chunk DTW + per-chunk pseudo-truth lookup; when `--fixtures` is given, it keeps the existing simulated-fixture path with G4-synthetic and `stitch_error_frames` references stripped.
- Baseline file at `model/data/evals/chroma_dtw/baseline.json` is rewritten to `{primary, guards: {g1, g2, g3, g4, g5}}` where g4 now stores the consecutive-chunk continuity pass-pct. One re-baseline commit follows the metric switch.

### Modules removed

- `model/src/chroma_dtw_eval/gold_truth_builder.py` + `model/tests/chroma_dtw_eval/test_gold_truth_builder.py`
- `model/src/chroma_dtw_eval/practice_compose.py` + `model/tests/chroma_dtw_eval/test_practice_compose.py`
- `model/scripts/amt_pseudo_truth_pilot.py` (superseded by `amt_regen.py`)

### Modules untouched

- `chroma_cache.py`, `dtw_runner.py`, `silence_synth.py`, `ratchet.py`
- Rust crate `apps/api/src/wasm/score-analysis/` (warping path field already present)

## Modules (deep / shallow audit)

| Module | Interface | Hides | Depth |
|---|---|---|---|
| `pseudo_truth_cache.py` | `load_pseudo_truth`, `write_pseudo_truth`, `cache_path`, `PseudoTruth.audio_sec_to_score_sec`, `PseudoTruth.score_sec_to_audio_sec` | JSON schema, 4-field cache-key (audio_sha256, amt_checkpoint_hash, score_sha256, parangonar_version), monotone-arrays invariant, atomic write, hash-mismatch detection | DEEP |
| `amt_regen.py` | `regenerate_pseudo_truth(...)` + `python -m chroma_dtw_eval.amt_regen` CLI | 27s chunking, base64 encoding, AMT POST (RequestException fatal; only documented tokenizer-error skips), AMT-note concatenation with offset, direct bach JSON score load (single-tempo), measure-table extraction, parangonar invocation, label-filter + running-max, min-coverage gate (matched≥100, match-rate≥0.5), checkpoint mismatch | DEEP |
| `chunk_sampler.py` (extended) | `sample_chunks`, new `sample_practice_chunks` | YAML enumeration, pseudo-truth coverage check, per-clip duration discovery, position-bucket stratification | DEEP |
| `metric_aggregator.py` (modified) | `aggregate(results, baseline, *, frame_rate_hz, tolerance_s) -> Metrics` | Per-kind aggregation, G2 AUC, regression thresholds | DEEP |
| `verify.py` (modified) | `python -m chroma_dtw_eval.verify --baseline ... --corpus ...` | Real DTW dispatch, pseudo-truth lookup-and-interpolate, sidecar write | DEEP |

All five hide substantial mechanism behind small APIs; none are shallow.

## Verification Architecture

- **Canonical success state:** `just chroma-eval-verify` exits 0 with a single float on stdout, within 120s wallclock on M4. Sidecar JSON validates against schema (`primary` float, `guards` object with exactly `g1`/`g2`/`g3`/`g5`, `regressed` list, `n_chunks` int). The committed `baseline.json` reflects a real measurement on `bach_prelude_c_wtc1` after the pivot.
- **Automated check:** `cd model && uv run python -m chroma_dtw_eval.verify --baseline data/evals/chroma_dtw/baseline.json --corpus data/evals/`. CI/local invocation `just chroma-eval-verify` wraps it.
- **Harness already exists:** the shipped verify CLI smoke test (`model/tests/chroma_dtw_eval/test_verify_cli_smoke.py`) is the in-process harness. It must be updated to assert (a) the new sidecar schema (no `g4`), (b) stdout is still exactly one float, (c) exit 0 on a known-good fixture. This is part of the metric_aggregator task group.

## File Changes

| File | Change | Type |
|---|---|---|
| `model/src/chroma_dtw_eval/pseudo_truth_cache.py` | Cache reader/writer + dataclass + interpolators | New |
| `model/tests/chroma_dtw_eval/test_pseudo_truth_cache.py` | Round-trip + missing + mismatch | New |
| `model/src/chroma_dtw_eval/amt_regen.py` | Orchestrator + CLI | New |
| `model/tests/chroma_dtw_eval/test_amt_regen.py` | Integration with stub AMT server | New |
| `model/config/amt_version.json` | Committed `{amt_checkpoint_hash, ...}` | New |
| `Justfile` | Add `chroma-eval-verify`, `chroma-eval-ratchet`, `amt-regen-pseudo-truth` recipes (none exist today despite CLAUDE.md claim) | Modify |
| `model/src/chroma_dtw_eval/chunk_sampler.py` | Add `sample_practice_chunks`, extend `Chunk` with `video_id` | Modify |
| `model/tests/chroma_dtw_eval/test_chunk_sampler.py` | Add practice-sampler tests | Modify |
| `model/src/chroma_dtw_eval/metric_aggregator.py` | Switch to seconds, drop G4, add `practice` kind | Modify |
| `model/tests/chroma_dtw_eval/test_metric_aggregator.py` | Replace G4 case with practice cases | Modify |
| `model/src/chroma_dtw_eval/verify.py` | Wire practice path through `sample_practice_chunks` + pseudo-truth | Modify |
| `model/tests/chroma_dtw_eval/test_verify_cli_smoke.py` | Drop g4 expectations | Modify |
| `model/data/evals/chroma_dtw/baseline.json` | Rewrite (no `g4`) after metric switch | Modify |
| `model/src/chroma_dtw_eval/gold_truth_builder.py` | Remove | Delete |
| `model/tests/chroma_dtw_eval/test_gold_truth_builder.py` | Remove | Delete |
| `model/src/chroma_dtw_eval/practice_compose.py` | Remove | Delete |
| `model/tests/chroma_dtw_eval/test_practice_compose.py` | Remove | Delete |
| `model/scripts/amt_pseudo_truth_pilot.py` | Remove (superseded) | Delete |

## Open Questions

- Q: Does `/health` on AMT need to be extended to publish `checkpoint_hash` now? Default: no — record the committed hash unconditionally; cross-check fires only when the field appears. Tracked as future hook in `amt_regen.py` comment.
- Q: How does the practice path discover clip duration? Default: from the cached pseudo-truth file's last `perf_audio_sec` entry (saves a `soundfile.info` call on every clip during sampling).
