# Pipeline Element Audit (2026-05-31)

## Context

End-to-end per-element investigation of the practice pipeline on real recordings (T5 bucket-labeled YouTube data + a clean Lang Lang BWV 846 prelude). Goal: ground-truth verdicts on what works, what's broken, and where the moat actually fails, before committing to fix work.

Recordings tested locally:
- K545 beginner (ord 1) + advanced (ord 5) — same-piece skill contrast, MuQ ground truth
- Chopin Waltz C♯m "first week" (messy beginner practice, Tier-2/unknown-piece path)
- Lang Lang BWV 846 prelude — catalog piece, moat path

## Scorecard

| # | Element | Verdict | Evidence |
|---|---|---|---|
| 1 | Local rig (MuQ + AMT boot) | Works | Smoke tests pass on MPS |
| 2 | MuQ scoring | Weak on consumer audio | Compressed 0.49–0.60 range; 4/6 dims tied/wrong on same-piece skill pair |
| 3 | AMT transcription | Works, accurate | Zero false accidentals on a C-major piece; latency scales with note density (4.8s sparse / 53s dense) |
| 4 | Piece-ID (fingerprint) | Fails for arpeggiated pieces, gracefully | BWV 846 prelude: 1 trigram hit, tied with wrong pieces; DTW gate rejects → Tier 2 |
| 5 | DTW chroma alignment | Algorithm-tested via roundtrip fixture | Real-audio test deferred (needs MuQ chroma extraction) |
| 6 | Bar-analysis Tier 2 | Works | Real per-dim facts from 109 AMT notes |
| 7 | Bar-analysis Tier 1 | Works, score-referenced | "Softer than notated", "legato 2.72× score", etc., given correct alignment |
| 8 | ModeDetector | Works | Silent in Tier-3 (correct); detects drilling within 2 chunks given Tier-1 inputs |
| 9 | PassageLoopDetector | Works (existing unit test) | Same gating pattern as ModeDetector |
| 10 | Teacher synthesis | ASCF-grade given Tier-1 input | Bach-aware bar-specific feedback, no score leakage |
| 11 | Artifact tool_use | Works | `assign_segment_loop` emitted with correct piece_id, bar range, dimension, rationale |
| 12 | Eval harness (DO-level) | Wiring works; synthesis short-circuit blocks numerical baseline | `eval_practice.py` runs end-to-end but returns 0-char synthesis |

## The reframe

Every element downstream of piece-ID is healthy. The teacher produces bar-specific, piece-aware, artifact-emitting feedback the moment it receives Tier-1 facts. The locked ASCF baseline of 1.387/3.0 is measuring the teacher on the wrong distribution of inputs — Tier 2 dominates because piece-ID fails for most pieces. **The "specific feedback" problem is a piece-ID problem, not a teacher problem.**

## Open issues (prioritized)

### Issue 1 — Pitch-trigram piece-ID is structurally weak for harmonically-repetitive pieces

**Severity:** High. Gates the entire Tier-1 bar-aligned moat (DTW alignment, bar-level facts, segment loops, play_passage, bar-specific teacher feedback) for any fingerprint-hostile piece.

**Evidence:**
- BWV 846 prelude: only **11 distinct pitch trigrams** in the index vs the fugue's **89**.
- 109-note window from a clean Lang Lang performance: prelude recalled with **1 hit**, tied with Chopin Ballade 1, Chopin Ballade 2, Debussy.
- Rerank cosine put Chopin/Debussy above the correct piece. DTW gate correctly rejected → "unknown" / Tier 2.

**Root cause:** Arpeggiated broken-chord music has few unique pitch trigrams and the ones it has (C-E-G, E-G-C) are harmonically generic. Any AMT pitch slip breaks an exact trigram match.

**Impact:** Mode/loop detection, segment loops, play_passage, and bar-specific teacher feedback are silently disabled for these pieces. User has no signal that bar-level feedback was unavailable.

**Repro:**
```
NOTES_JSON=model/data/results/pipeline_test/bwv846_langlang_40to65s_notes.json \
NGRAM_INDEX=model/data/fingerprints/ngram_index.json \
RERANK_FEATURES=model/data/fingerprints/rerank_features.json \
SCORE_JSON=model/data/scores/bach.prelude.bwv_846.json \
EXPECTED_PIECE=bach.prelude.bwv_846 \
cargo test -p piece-identify identify_real_recording -- --nocapture
```

**Fix directions to explore:**
- Replace pitch trigrams with **interval trigrams** (transpose-invariant, more discriminative for harmonic patterns).
- Add a **chroma/CQT-based n-gram fingerprint** as a parallel recall channel, combined with the symbolic one.
- Broaden DTW confirmation tolerance for low-recall pieces, OR add a "low-confidence + high-DTW-match" combined gate.
- Characterize piece-ID across the full catalog first: a fugue or melodic piece (89+ trigrams) may already work fine; the fix may only need to target the worst quartile.

### Issue 2 — MuQ weak on consumer (home-recorded) audio

**Severity:** Medium-High. Known and on the Model v2 retrain roadmap; this audit empirically confirms the gap.

**Evidence:**
- K545 beginner (T5 ord 1) vs advanced (T5 ord 5) — same piece. Chunk-level medians: 4/6 dims tied or *inverted* (advanced rates *lower* than beginner on dynamics/timing).
- Output range across both recordings, all dims: compressed to 0.49–0.60. Within-recording chunk std (~0.01–0.04) is comparable to between-recording delta. Classic regression-to-prior signature on out-of-distribution input.

**Impact:** Skill discrimination on real-user audio is near-random. Downstream teacher feedback inherits this noise.

**Fix:** Model v2 retrain on T2+T5 with consumer audio coverage (already planned).

**Repro:**
```
cd apps/inference
uv run muq_chunk_compare.py \
  --checkpoint-dir ../../model/data/checkpoints/a1_max_sweep/A1max_r32_L7-12_ls0.1 \
  --label beginner_ord1 ../../model/data/results/pipeline_test/k545_beginner_ord1_meowrjlmmjE.wav \
  --label advanced_ord5 ../../model/data/results/pipeline_test/k545_advanced_ord5_tvwDf0Y83eo.wav
```

### Issue 3 — Eval-mode synthesis short-circuits, producing empty output

**Severity:** High. Blocks the production-shape DO-level eval baseline (`eval_practice.py`). Synthesis-quality eval (`run_eval.py`) is unaffected and remains usable.

**Evidence:**
- `eval_practice.py --scenarios t5 --max-recordings 1` produces `synthesis (0 chars)` on both Pass A (with `piece_query`) and Pass B (zero-config).
- Synthesis latency **46 ms** — a real Anthropic Sonnet call is 1–3 s. The path is short-circuiting before the LLM call.
- Accumulator metric `chunks_with_audio: 0` for all chunks (eval-mode chunks bypass R2 audio fetch; `hasAudio` flag in the timeline appears never set true).
- Judges score the empty synthesis 0 across all four rubric dimensions, so all metrics come out 0.000.

**Repro:**
```
# (requires fresh CF auth — see Issue 4)
just api  # boot wrangler dev at :8787
cd apps/evals
uv run python -m pipeline.practice_eval.eval_practice --scenarios t5 --max-recordings 1
# observe "synthesis (0 chars)" in stdout
# report at apps/evals/reports/practice_eval_details.json
```

**Fix locus — three candidate code paths in `apps/api/src/do/session-brain.ts`:**
1. `handleEvalChunk` (~line 1036) — may never populate the timeline `hasAudio` flag, tripping a downstream "no audio → skip synthesis" guard.
2. `runSynthesisAndPersist` — may early-return when the accumulator has no top-moments (the typical eval state after few chunks).
3. The `if (state.isEvalSession)` branches at lines 1286 and 1355 — may take a path that skips the LLM call entirely.

**Debug approach (~30 min):** Add `console.log(JSON.stringify({stage, state_snapshot}))` traces in `runSynthesisAndPersist` and `handleEvalChunk`, re-run the smoke, read the wrangler runtime log to see exactly where it bails. Then patch the responsible branch.

### Issue 4 — Wrangler dev gated on fresh Cloudflare auth (operational fragility)

**Severity:** Low for established devs; medium for first-time contributors and CI.

**Evidence:** First boot of `just api` failed at `/accounts/.../workers/subdomain/edge-preview` (CF API call). `env.AI` binding is `remote` and requires a valid `CLOUDFLARE_API_TOKEN`. Wrangler refuses to start when the AI binding's remote proxy session can't be established.

**Fix directions:**
- Document `wrangler login` (or a fresh `CLOUDFLARE_API_TOKEN`) as a prereq in `apps/evals/EVAL_CHECKLIST.md`.
- OR wire a local stub for the `env.AI` binding so eval can run without remote auth (eval doesn't currently use the Workers AI subagent for synthesis — Anthropic is direct).

## What's confirmed healthy (do not touch)

For future readers / fix planners: these are *not* issues, despite the user's intuition that "everything is broken." Listed so they're not accidentally targeted:

- AMT transcription (accurate pitch detection; latency scales correctly with note density, not duration).
- Bar-analysis engine — Tier 2 *and* Tier 1, given correct inputs.
- ModeDetector + PassageLoopDetector — both correctly silent in Tier-3 (no piece-ID) and correctly active in Tier-1.
- Teacher synthesis — produces ASCF-grade output given Tier-1 facts. Doesn't leak scores. Adapts piece-specific knowledge ("Bach's counterpoint…").
- Tool_use artifact emission — `assign_segment_loop` emitted with correct piece_id, bar range, dimension, and pedagogical rationale.
- DTW gate behavior — correctly *rejected* a wrong piece-ID candidate (Haydn for a Bach prelude). Safety gate works as designed.
- Eval harness wiring — `eval=true` WebSocket mode, debug auth, chunk delivery, judge pipeline, report writer all functional end-to-end. Only the synthesis path inside is short-circuiting.

## Eval baseline status

| Path | Status | Notes |
|---|---|---|
| **Production-shape DO eval** (`eval_practice.py`) | Set up, blocked on Issue 3 | Once Issue 3 is fixed, run `python -m pipeline.practice_eval.eval_practice --scenarios t5` for the full baseline |
| **Synthesis-quality eval** (`run_eval.py`) | Unaffected, runnable now | Per `apps/evals/EVAL_CHECKLIST.md` — produces the same metric type as the locked ASCF 1.387 |
| **Per-element baselines** | Available now | See test harnesses below |

For the "come back in 2 months" comparison, the working options today are: (a) per-element cargo + bun harnesses, (b) the `run_eval.py` synthesis-quality run. The DO-level integration baseline becomes the third option once Issue 3 is fixed.

## Test artifacts created in this audit

Harnesses (test-only, no production impact):

| Path | What it tests |
|---|---|
| `apps/api/src/wasm/piece-identify/src/real_recording_test.rs` (+ `#[cfg(test)] mod` in `lib.rs`) | piece-ID three-stage pipeline on real AMT notes + index |
| `apps/api/src/wasm/score-analysis/src/real_recording_test.rs` (+ `#[cfg(test)] mod` in `lib.rs`) | Bar-analysis Tier 1 and Tier 2 on real AMT notes + score |
| `apps/api/test-pipeline-mode.ts` (bun) | ModeDetector on real chunk score stream + synthetic Tier-1 stream |
| `apps/api/test-pipeline-teacher.ts` (bun) | Teacher A/B (Tier-2 vs Tier-1 brief) + artifact tool_use, via direct Anthropic fetch |
| `apps/inference/muq_chunk_compare.py` | Per-chunk MuQ scoring across recordings, per-dim median/std, optional JSON dump |
| `apps/inference/amt_to_json.py` | Single-chunk AMT transcription to JSON + pitch-class histogram sanity check |

Test recordings under `model/data/results/pipeline_test/`:
- `chopin_waltz_cs_minor_firstweek.wav` — Tier-2 messy practice
- `k545_beginner_ord1_meowrjlmmjE.wav` + `k545_advanced_ord5_tvwDf0Y83eo.wav` — same-piece skill contrast (MuQ ground truth)
- `bwv846_prelude_langlang_gVah1cr3pU0.wav` + 20s and 40-65s slices + AMT notes JSONs — catalog-piece moat tests
- `chopin_per_chunk.json` — MuQ per-chunk scores for ModeDetector input

Doc fixes:
- `docs/apps/00-status.md` — corrected AMT status to LOCAL ONLY (prod `AMT_ENDPOINT` unset, deploy deferred pre-beta).

## Suggested next-session priorities

1. **Fix Issue 3** (eval-mode synthesis short-circuit). ~30 min. Unblocks the real DO-level baseline number, which is the artifact needed for the 2-month regression check.
2. **Run the full `eval_practice` baseline** across all four T5 scenarios. Persist the JSONL report as `apps/evals/reports/baseline_pre_improvements_2026-MM-DD.jsonl`. This is the artifact to compare against after improvements land.
3. **Spike on Issue 1** (interval-trigram or chroma fingerprint). Independent of other fixes. Could move piece-ID hit rate from "fails on prelude" to "works for most catalog pieces" — unlocking the entire Tier-1 moat in one stroke.
4. (Deferred) Real-audio DTW alignment test — needs MuQ chroma extraction added to the smoke path. Existing roundtrip cargo test covers the algorithm.

## Related

- `docs/apps/00-status.md` — Apps implementation dashboard (AMT line corrected as part of this audit).
- `apps/evals/EVAL_CHECKLIST.md` — Eval runbook (synthesis-quality path).
- `MEMORY.md` — durable session findings.
