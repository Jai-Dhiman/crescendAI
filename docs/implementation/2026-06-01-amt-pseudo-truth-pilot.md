# AMT-pseudo-truth pilot for practice-corpus eval

**Status:** PILOT COMPLETE (2026-06-01). Three clips of chopin_ballade_1 transcribed via local AMT, aligned via parangonar, analyzed for noise floor. Findings drive the harness rework that follows.

**Picks up from:** `docs/implementation/2026-05-31-chroma-dtw-eval-pivot.md` (harness shipped) + brainstorm pivot to practice-distribution primary scalar.

---

## 1. Why this pilot existed

The shipped harness assumes MAESTRO + parangonar gold-truth as primary scalar (50ms tolerance). User pushed back: production audio is amateur phone-recorded practice, not Disklavier studio captures. Measuring DTW changes on MAESTRO optimizes a metric users never experience. The chroma-DTW failures reproduced earlier (teleport, silence-lock, cost-not-confidence) all happened on amateur YouTube audio, not MAESTRO.

Practice videos have no ms-truth (no Disklavier MIDI capture). To use them as primary, the pseudo-truth path is: AMT transcribes audio → MIDI → parangonar aligns MIDI to score → audio↔score map. This couples DTW measurement to AMT noise — feature, not bug, since production already routes AMT notes to the teacher LLM as ground truth.

Pilot question: **is AMT-pseudo-truth precise enough to drive an /autoresearch loop, and at what tolerance?**

---

## 2. Pipeline as built

`model/scripts/amt_pseudo_truth_pilot.py`:

1. Read WAV (yt-dlp produced 24kHz mono).
2. Split into 27s non-overlapping chunks. (27s, not 30s: AMT internally pads to 30s; sending 27s leaves 3s of zero-pad so the model emits clean note terminations and avoids the tokenizer's `('on', 67), ('onset', 30000), ('off', 67)` boundary failure.)
3. POST each chunk to `localhost:8001/transcribe` as base64-encoded WAV. ffmpeg inside `decode_webm_to_pcm` auto-detects WAV regardless of the `.webm` tempfile extension. Skip-on-error per chunk (~8% of chunks hit the boundary bug despite 27s).
4. Concatenate notes across chunks, offset each by `chunk_idx * 27`.
5. Build AMT performance note array (numpy structured with `onset_sec`, `pitch`, etc.).
6. Load score via `partitura.load_score()`. Extract bar measure table from `part.iter_all(pt.score.Measure)`.
7. `parangonar.AutomaticNoteMatcher()(score_na, amt_perf_na)`. Score side wants `onset_beat` (raw note array); performance side wants `onset_sec`. Returns `List[Dict]` with `label/score_id/performance_id`.
8. Filter to `label=='match'`. Build `(amt_onset_sec, score_onset_div)` pairs. Enforce monotonic score_div via running-max.
9. For any audio time `t`, `np.interp(t, perf_sec, score_div)` → score_div → bar number via measure table.

`model/scripts/amt_pseudo_truth_analyze.py`: bootstrap-style noise estimation (drop 10/20/30% of matched pairs, re-interpolate at fixed anchors, measure stddev of bar positions) + per-30s-window bar-rate distribution.

---

## 3. Findings (chopin_ballade_1, 3 clips, all data)

| clip | dur | AMT notes | matched | vs AMT | vs score | median bars/30s | min–max |
|---|---|---|---|---|---|---|---|
| MKbOHysTOE8 | 299s | 2305 | 1883 | 82% | 36% | 12.12 | 7.6–21.2 |
| kNZEQP0hiwE | 573s | 3774 | 3774 | 93% | 73% | 12.40 | 5.6–31.2 |
| 2cEhqylETzY | 715s | 5282 | 4756 | 90% | 92% | 9.12 | 5.6–22.7 |

**Internal jitter is tight.** Bootstrap dropouts (10/20/30%) move bar positions by stddev ≤ 0.025 bars at all anchors. At ~2.6 s/bar Chopin tempo that's ~60-80ms equivalent. Compatible with the original spec's 50ms tolerance **when the alignment is right**.

**Structural variability is larger.** Per-30s-window bar-rate spreads 5.5× from min to max within a single clip. Some windows have 2-3× the median rate. Investigating further: cross-clip behavior at common audio anchors shows two clips with similar overall tempo (median 12.1–12.4 bars/30s) diverge by 5-8 bars after one clip hits a rate spike, while clip 3 (slower performer, median 9.1) stays smooth.

**Cross-clip divergence at common audio anchors:**

| audio time | clip 1 bar | clip 2 bar | clip 3 bar |
|---|---|---|---|
| 30s | 7.0 | 7.6 | 6.2 |
| 90s | 29.2 | 29.3 | 20.3 |
| 180s | 76.8 | 71.6 | 49.6 |
| 270s | 113.7 | 105.3 | 81.9 |

Clip 3 is consistently behind — genuinely slower performance, ~2.79 s/bar avg vs ~2.5 s/bar. Pseudo-truth correctly tracks per-clip tempo. Most of the cross-clip "noise" is real performer variation, not measurement noise.

---

## 4. Implications for the metric

The original 50ms primary tolerance is **structurally incompatible** with AMT-pseudo-truth at scale. It would be dominated by structural drift and chunk-boundary AMT misses, regardless of DTW quality.

**Decision (this pilot's deliverable):** primary scalar tolerance widens to **±1.5s of audio time**.

Reasoning:
- Well above internal jitter (~60-80ms) and AMT FP16 vs FP32 numerical drift (~10-30ms).
- ~1 bar at Chopin tempo, ~0.5 bars at Bach Invention tempo, ~3 bars at Czerny etudes — scales naturally across pieces.
- Still tight enough that catastrophic DTW failures (teleport, silence-lock, multi-bar drift) clearly fail.
- Time-domain matches the existing G3 silence guard.

Guards reshuffle:
- G1 (teleport on amateur) — STAYS, dataset shifts from skill_eval to practice_eval where overlap exists.
- G2 (cost AUC vs error) — STAYS, error definition shifts from 50ms to 1.5s.
- G3 (silence robustness) — STAYS, synthetic silence dataset unchanged.
- G4 (synthetic MAESTRO composition) — DROPPED. Requires MAESTRO ground truth that's out of scope.
- G5 (real-practice self-consistency) — STAYS, now serves clips where pseudo-truth isn't available.

---

## 5. AMT in the loop: it isn't

Calling AMT in-line during eval was fine for the pilot (3 clips × ~5 min each = 15 min). It does not work for /autoresearch (hundreds of iterations per day on a ~30-piece × ~30-chunk corpus).

**Solution: pseudo-truth cache + regen command.** Same shape as the existing `chroma_cache`:

```
[one-time, when corpus changes or AMT checkpoint bumps]
audio_clip + AMT_checkpoint_hash
    -> production HF AMT (or local for dev)
    -> raw AMT notes (cached as JSON)
    -> parangonar align to score
    -> (audio_sec, score_div) pairs (cached as JSON)
    -> commit cache index to git, store payloads under data/

[every /autoresearch iteration]
candidate DTW change
    -> just chroma-eval-verify
    -> reads cached pseudo-truth, runs DTW only
    -> single number on stdout, ~60s wall, $0 to AMT
```

Cache key: `(audio_sha256, amt_checkpoint_hash)`. When AMT checkpoint bumps, cache invalidates by hash and a regen command rebuilds it.

This decouples eval iteration speed from AMT deploy choice. Local CPU AMT for dev convenience; production HF AMT as authoritative source of cached pseudo-truth. The eval never calls either.

---

## 6. AMT deploy direction (decided, separate work)

CPU-only CF Container plan was the wrong shape for AMT specifically — encoder is non-autoregressive (CPU-fine) but decoder is autoregressive with KV cache (scales badly on CPU). At realistic load, GPU is both faster and cheaper:

- **HF Inference Endpoint, L4 ($0.80/hr)** — same SDK as MuQ, same AI Gateway routing pattern. ~200-500ms per chunk vs 5-30s on CPU.
- Per-user-hour cost ~$0.008 at 100 concurrent users.

Drop the CF Container plan for AMT specifically. Keep `apps/inference/amt/amt_local_server.py` for dev. The HF deploy is a separate work item; the harness rework does not block on it (local AMT regenerates the cache fine for pre-production work).

---

## 7. What ships next

See plan + spec to be created via `/plan`. High-level scope:

1. **Cache infrastructure** — `model/src/chroma_dtw_eval/pseudo_truth_cache.py` keyed by `(audio_sha256, amt_checkpoint_hash)`; idempotent; explicit invalidation on hash mismatch.
2. **Regen command** — `just amt-regen-pseudo-truth` reads audio, calls AMT (URL configurable), runs parangonar, writes cache. Reuses the pilot's chunking + skip-on-error logic.
3. **Primary metric switch** — `metric_aggregator.aggregate()` primary becomes practice bar/time-tolerance; tolerance widens to 1.5s; G4 removed.
4. **Corpus switch** — `chunk_sampler` stratifies over practice_eval (not MAESTRO) + skill_eval overlap; ~30 chunks per piece × pieces-with-scores.
5. **Score sourcing** — DEFERRED as parallel data work. We have chopin_ballade_1; baseline initially on that piece; expand as scores get sourced.
6. **AMT checkpoint pinning** — `model/config/amt_version.json` (or equivalent) committed; regen command refuses to write cache if AMT endpoint reports a different checkpoint.

Out of scope:
- MAESTRO + ASAP path (removed; `gold_truth_builder.py` becomes dead code or gets deleted).
- HF AMT endpoint deploy (separate work; local AMT regenerates the cache fine).
- AMT itself (we measure DTW under realistic AMT noise; AMT improvements are a separate research track).

---

## 8. Pilot artifacts committed alongside

- `model/scripts/download_practice_eval.py` — yt-dlp wrapper, idempotent, per-clip error logging.
- `model/scripts/amt_pseudo_truth_pilot.py` — single-clip pipeline; will be refactored into the regen command during the rework.
- `model/scripts/amt_pseudo_truth_analyze.py` — bootstrap noise floor estimator; throwaway after harness rework but kept for now as the pilot's analysis basis.
- `model/data/evals/practice_eval_pseudo/chopin_ballade_1/<video_id>/` — per-clip cache (`amt_notes.json`, `matched_pairs.json`, `bar_map.json`, `report.json`). Will be relocated/restructured by the rework.
- Audio under `model/data/evals/practice_eval/<piece>/audio/` is gitignored; regen with the download script.
