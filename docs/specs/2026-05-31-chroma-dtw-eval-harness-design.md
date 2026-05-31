# Chroma-DTW Score-Follower Eval Harness Design

**Goal:** Provide a single CLI that scores the production chroma-DTW score follower on a frozen multi-piece test set and exits non-zero if any guard regresses, so `/autoresearch` can keep-or-revert DTW changes mechanically.

**Not in scope:**
- Any change to `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs` or its callers (the DTW is the system being measured, not modified).
- Rehearsal-acquisition logic (multi-chunk agreement) — measured by G5 but not implemented.
- AMT note-level symbolic alignment (future eval extension).
- Decoupling chroma alignment from the AMT gate in `apps/api/src/do/session-brain.ts`.
- Rebuilding `chroma_feature` (we re-use `apps/inference/muq/chroma.py`).
- Re-downloading the MAESTRO audio subset — the harness consumes whatever is already on disk and skips with explicit exit code if the subset is missing.

## Problem

The production chroma-DTW score follower has three reproduced failure modes documented in `docs/implementation/2026-05-31-chroma-dtw-eval-pivot.md`:

1. **Teleport.** Amateur 15s opening of Ballade 1 aligned to bar 261/264.
2. **Cost is not confidence.** Correct (0.174) and wrong (0.214) alignments overlap.
3. **Silence locks.** rms~0.01 chunks produce confident-looking wrong alignments (cost 0.049 → wrong bar 48).

The first `/brainstorm` and `/plan` cycle (see `docs/specs/2026-05-28-continuity-aware-chroma-follower-design.md` and `docs/plans/2026-05-28-continuity-aware-chroma-follower.md`) was halted at Task 0 because the proposed fix could not be validated beyond a single fixture. The blocker is the absence of an eval, not the absence of fixes. Until a per-commit verify command exists that surfaces all three failures on a frozen multi-piece corpus, no DTW change can be honestly accepted or rejected.

## Solution (from the user's perspective)

A single command:

```
just chroma-eval-verify
```

returns one number on stdout (the primary scalar — % of MAESTRO chunks aligned within 50ms), exits 0 if all guards stay within their ratcheted baselines, exits non-zero if any guard regresses. A sidecar JSON (`model/data/evals/chroma_dtw/last_run.json`) carries per-chunk results and per-guard deltas for human review. A second command:

```
just chroma-eval-ratchet
```

is run by a human after manual review to update the committed baseline file (`model/data/evals/chroma_dtw/baseline.json`). The verify command never writes the baseline.

`/autoresearch` consumes the stdout number as its metric and the exit code as its keep-or-revert gate.

## Design

Architecture: **primary scalar + hard guards, not a weighted composite.** The failure modes are non-commensurable — teleport-rate and ms-error live in different units and trade off in non-physical ways. /autoresearch keeps a change only if (primary moves up OR stays flat within CI) AND no guard regresses against its baseline. Each guard is a single percent or AUC; baselines are committed JSON updated by an explicit human command.

**Primary scalar.** % of frozen MAESTRO chunks where `|predicted_frame - gold_frame| <= 50ms`. Frame-level rather than bar-level (sharper gradient, no staircase artefacts near bar boundaries). 50ms matches Paper 1's primary comparator and is perceptually meaningful. Test set ~900 chunks (~30 pieces × ~30 chunks), position-stratified into intro/early/middle/late/cadence-zone, biased toward the ~17 pieces present in `model/data/evals/skill_eval/` so primary and amateur guards co-vary on the same repertoire.

**Ground truth.** parangonar on the (n)ASAP subset (already at `model/data/raw/asap/`). MAESTRO audio↔MAESTRO MIDI is zero-error (Disklavier simultaneous capture); (n)ASAP gives MIDI↔score at ~6ms; chained → audio↔score in ms, well under 50ms tolerance.

**Guards** (all baseline-ratcheted; first run sets the baseline, no absolute thresholds):

- **G1 — Teleport rate.** % of stateless cold-start chunks landing > 5 bars from the forward-track alignment at the same time index. Dataset: 50 chunks across 5 amateur recordings × 5 skill buckets, frozen subset of `model/data/evals/skill_eval/`. Predicate: `pct(stateless_bar - forward_bar > 5) <= baseline + 1pp`.
- **G2 — Confidence calibration.** AUC of `dtw_cost` as a predictor of `|error| > 50ms` over the 900 primary-set chunks. Predicate: `auc >= baseline_auc - 0.02`.
- **G3 — Silence robustness.** Phase 1 (no abstention exists): % of synthetic-silence chunks producing `|error| > 5s` must stay `>= baseline - 1pp`. The metric is "loud failure rate on silence" — confidently locking to garbage is the bug, so a higher number is worse, but Phase 1 has no other lever, so we ratchet it as a fingerprint and let `/autoresearch` move it down once an abstention path lands. When a `chunk_result.status == "abstain"` field appears in the dtw_runner output, the harness switches Phase 2 predicate: `pct(status == "abstain") >= baseline - 1pp`. Dataset: 50 zero/low-noise/low-RMS chunks.
- **G4 — Synthetic practice composition.** MAESTRO chunks stitched into repeats/restarts/mid-piece jumps/partial plays with known source-frame of each stitch. Predicate: `pct(|err| <= 50ms) >= baseline - 1pp` (one number aggregated across all stitch types).
- **G5 — Real-practice self-consistency.** Same predicate as G1 but on `model/data/evals/practice_eval/<piece>/candidates.yaml` rows with `approved: true`. No ms-truth — pure self-consistency.

**Runtime budget.** ≤ 120s wallclock on M4. Achieved by caching chroma to disk (idempotent on hash of audio file + chroma params); the DTW call itself is fast (≤ 50ms per 15s chunk in the existing Rust binary). 900 + 50 + 50 + 50 + 50 = 1100 chunks at ~50ms each is 55s, leaving headroom for IO and aggregation.

**Stateless DTW invocation.** `dtw_runner` shells out to a small Rust binary (new — `apps/api/src/wasm/score-analysis/src/bin/dtw_chunk_cli.rs`) that calls the existing `chroma_dtw_native` with audio chroma read from stdin and score bars read from a JSON file path. This is the only Rust addition; it exposes the existing native function over a subprocess boundary so the Python harness does not need maturin/PyO3. The binary lives in the same crate and is rebuilt by `cargo build --release` whenever the DTW changes — so /autoresearch can keep editing `chroma_dtw.rs` and the verify command automatically picks up the new binary.

## Modules

| Module | Interface | Hides | Depth |
|---|---|---|---|
| `gold_truth_builder` | `build_gold_map(maestro_midi_path, asap_score_dir) -> GoldMap` | parangonar invocation, MAESTRO→ASAP lookup, ms-tight audio↔score frame map, on-disk caching keyed by (midi hash, score hash) | DEEP |
| `chunk_sampler` | `sample_chunks(corpus_spec, policy, seed) -> ChunkManifest` | position stratification, frozen seed handling, manifest JSON schema with per-chunk gold offsets, intro/early/middle/late/cadence-zone bucketing | DEEP |
| `chroma_cache` | `get_chroma(audio_path, params) -> ChromaArray` | hash-keyed file naming, chroma extraction via `apps/inference/muq/chroma.py`, atomic write-then-rename, miss → compute → cache | DEEP |
| `dtw_runner` | `run_dtw(audio_chroma, score_bars_path, frame_rate, decim) -> DtwResult` | subprocess management of `dtw_chunk_cli`, stdin marshalling, stderr → exception, JSON parse | DEEP |
| `metric_aggregator` | `aggregate(per_chunk_results, baseline) -> Metrics` | primary scalar formula, all 5 guard formulas, baseline-delta diff, JSON sidecar shape | DEEP |
| `verify_cli` | `python -m chroma_dtw_eval.verify` | argparse, single-number stdout, exit-code policy, baseline file I/O | SHALLOW by design — it is the assembly of the deep modules and nothing else; thin glue is correct here |
| `dtw_chunk_cli` (Rust) | stdin: raw float32 chroma; argv: score json, frame_rate, decim; stdout: JSON `{bar_min, bar_max, cost, bar_per_frame, predicted_score_frame}` | release-mode Rust binary calling existing `chroma_dtw_native` plus the trivial path→frame conversion | DEEP — wraps the production function unchanged, isolates the WASM-free boundary |

All Python modules live under `model/src/chroma_dtw_eval/`. Tests under `model/tests/chroma_dtw_eval/`. The single Rust binary lives under `apps/api/src/wasm/score-analysis/src/bin/dtw_chunk_cli.rs` (same crate as the DTW, no new Cargo manifest).

**Tested through:** Each module is tested through its public function only. No tests reach into private helpers, file-layout details, or struct internals. Integration tests exercise the verify CLI end-to-end against a tiny committed fixture (3 chunks total) so CI does not depend on the full MAESTRO download.

## Verification Architecture

- **Canonical success state:** `python -m chroma_dtw_eval.verify` returns one float on stdout (the primary scalar in [0.0, 100.0]), exits 0, and writes `model/data/evals/chroma_dtw/last_run.json` with the full per-chunk breakdown plus baseline deltas. With the committed `baseline.json` recording the first-run numbers, running verify a second time on unchanged code must reproduce the same primary scalar (±0 — chroma is cached, DTW is deterministic) and exit 0.
- **Automated check:** A single `pytest` integration test (`tests/chroma_dtw_eval/test_verify_cli_smoke.py`) invokes the CLI on a tiny committed fixture (`model/data/evals/chroma_dtw_fixtures/` — 3 chunks: 1 MAESTRO-with-gold, 1 amateur, 1 synthetic-silence) and asserts (a) stdout parses as a float in [0, 100], (b) exit code is 0 when run against a freshly-written baseline, (c) exit code is non-zero when the baseline has guards set above current measurements.
- **Harness:** Task Group 0 (below) builds the committed-fixture tier of the eval before any other module. This is the smallest harness that exercises every deep module end-to-end. The MAESTRO/skill/practice corpora are wired in subsequent groups; nothing depends on those datasets being present until Group D.

## File Changes

| File | Change | Type |
|---|---|---|
| `docs/specs/2026-05-31-chroma-dtw-eval-harness-design.md` | this design | New |
| `docs/plans/2026-05-31-chroma-dtw-eval-harness.md` | implementation plan | New |
| `model/src/chroma_dtw_eval/__init__.py` | package marker | New |
| `model/src/chroma_dtw_eval/gold_truth_builder.py` | parangonar gold-map builder + cache | New |
| `model/src/chroma_dtw_eval/chunk_sampler.py` | frozen chunk manifest + position stratification | New |
| `model/src/chroma_dtw_eval/chroma_cache.py` | hash-keyed chroma cache | New |
| `model/src/chroma_dtw_eval/dtw_runner.py` | subprocess wrapper over `dtw_chunk_cli` | New |
| `model/src/chroma_dtw_eval/metric_aggregator.py` | primary + 5 guards + baseline delta | New |
| `model/src/chroma_dtw_eval/verify.py` | CLI entry point (`python -m chroma_dtw_eval.verify`) | New |
| `model/src/chroma_dtw_eval/ratchet.py` | CLI to update committed baseline (`python -m chroma_dtw_eval.ratchet`) | New |
| `model/src/chroma_dtw_eval/silence_synth.py` | injected-silence chunk generator | New |
| `model/src/chroma_dtw_eval/practice_compose.py` | synthetic practice composition (repeats/restarts/jumps) | New |
| `model/tests/chroma_dtw_eval/__init__.py` | test pkg marker | New |
| `model/tests/chroma_dtw_eval/test_chroma_cache.py` | unit-level via public API | New |
| `model/tests/chroma_dtw_eval/test_chunk_sampler.py` | unit-level via public API | New |
| `model/tests/chroma_dtw_eval/test_gold_truth_builder.py` | unit-level via public API | New |
| `model/tests/chroma_dtw_eval/test_dtw_runner.py` | subprocess integration via public API | New |
| `model/tests/chroma_dtw_eval/test_metric_aggregator.py` | per-guard formula via public API | New |
| `model/tests/chroma_dtw_eval/test_silence_synth.py` | generator via public API | New |
| `model/tests/chroma_dtw_eval/test_practice_compose.py` | generator via public API | New |
| `model/tests/chroma_dtw_eval/test_verify_cli_smoke.py` | end-to-end via subprocess | New |
| `model/tests/chroma_dtw_eval/test_ratchet_cli.py` | ratchet CLI behaviour | New |
| `model/data/evals/chroma_dtw_fixtures/README.md` | describes the 3-chunk committed fixture | New |
| `model/data/evals/chroma_dtw_fixtures/manifest.json` | tiny committed chunk manifest used by smoke test | New |
| `model/data/evals/chroma_dtw_fixtures/score_bars.json` | small score for the fixture | New |
| `model/data/evals/chroma_dtw_fixtures/audio_chunks/` | 3 tiny chroma `.bin` files | New |
| `model/data/evals/chroma_dtw/baseline.json` | committed baseline (written by ratchet, NOT verify) | New |
| `apps/api/src/wasm/score-analysis/src/bin/dtw_chunk_cli.rs` | release-mode binary wrapping `chroma_dtw_native` | New |
| `apps/api/src/wasm/score-analysis/Cargo.toml` | add `[[bin]]` entry for `dtw_chunk_cli` | Modify |
| `justfile` | add `chroma-eval-verify` and `chroma-eval-ratchet` targets | Modify |

## Open Questions

- **Q:** Is the parangonar API surface stable enough to lock to one version, or should we pin a range?
  **Default:** Pin the version present in `model/pyproject.toml` today (1.x). If parangonar breaks across a minor bump, the gold-map cache key bumps and ground truth is recomputed.
- **Q:** Should `dtw_chunk_cli` accept score bars on stdin alongside chroma to avoid temp files in the verify path?
  **Default:** No — score bars are read from a path argv. Path-based avoids re-serializing 264-bar scores 900 times; the score file is cached on disk by `chunk_sampler` once per piece per run.
- **Q:** What happens when the MAESTRO audio subset is partially missing?
  **Default:** `verify` exits with a distinct non-zero code (3 — "corpus incomplete") and stdout reports the missing fraction; this is treated as a setup error, not a guard regression. /autoresearch should be configured to abort the loop if exit code 3 appears (not auto-revert).
- **Q:** Do we run G1/G5 on chroma cached at the same params as the primary set, or recompute?
  **Default:** Same chroma cache, same params, keyed by `(audio file sha256, hop_target=50Hz, n_fft librosa default, sr=24000 — matching MuQ source)`. One cache, many consumers.
