# Piece-ID AMT Stage-0 Feasibility Design

**Goal:** Produce a defensible KILL/TUNE/PROCEED verdict (and, if PROCEED, a named winning matcher + query window) for symbolic note-to-note piece identification, measured locally on the 16 `practice_eval` recordings (amateur AMT) against the 254-piece score catalog, before any Rust/WASM is written.

**Not in scope:**
- Any Rust/WASM change (`apps/api/src/wasm/piece-identify/`), the fingerprint generator (`just fingerprint`), or `session-brain.ts` rewiring. That is Phase 1, gated on this verdict, and gets its own `/plan`.
- MuQ/Aria contrastive embedding channel (2-year open-set extension).
- Audio acquisition (yt-dlp) — audio is already cached for all 16 slugs.
- The dead audio-chroma harness (`query_chroma.py`, `score_chroma.py`, `matchers/chord_ngram.py`, `matchers/twodft.py`) — superseded, see File Changes.

## Problem

Issue #21 proved the chroma KILL was a **cross-modal artifact**: matching real-audio chroma (MuQ) against note-derived score chroma caps at recall@10 ~0.26, while same-distribution self-query hits 0.999. Harmony discriminates; the modality gap is fatal. The current production pipeline (`apps/api/src/wasm/piece-identify/src/ngram.rs`) is monophonic pitch-trigram recall that fails on arpeggiated/polyphonic music (BWV 846 -> 1 tied trigram hit, DTW-rejected to unknown).

Piece-ID is already fed AMT notes (`apps/api/src/do/session-brain.ts:688`, `perfNotes = amtResult.value.notes`). Matching **symbolic-to-symbolic** (AMT notes vs score notes) eliminates the cross-modal gap by construction. But this is unproven on real amateur AMT. We must measure it before committing to a Rust rebuild.

Two facts the existing harness gets wrong for this task:
1. It is **chroma-of-audio end-to-end** (`query_set.py` -> `audio_to_chroma` -> `(12, frames)`); the killed design. The new task is note-based.
2. Its query model fingerprints whole recordings; production decides on a **short, arbitrary-start window** matched against full-piece catalog entries (`session-brain.ts:893-903` accumulates a note count, then matches one chunk). A whole-vs-whole recall number is unachievable in production.

**Data gap (verified):** cached AMT notes (`practice_eval_pseudo/<slug>/<video_id>/amt_notes.json`) exist for only **1 of 16 slugs** (`chopin_ballade_1`). The other 15 must be transcribed via the local AMT server before the bake-off can run. Audio is cached for all 16.

## Solution (from the researcher's perspective)

Run `python -m piece_id_eval.bakeoff` locally (after `just amt` is up for the one-time transcription step). It:
1. Loads AMT notes for all 16 slugs' approved recordings (transcribing the 15 missing ones into the cache on first run).
2. For each recording, samples arbitrary-start windows at swept lengths {15, 30, 60, 90s, full} x N starts.
3. Ranks each window against the 254-piece note catalog with four matchers (C1 note-chroma, C2 landmark hash, C3 subsequence-DTW ceiling, C4 chroma-sequence DTW).
4. Computes recall@{1,5,10} per (matcher, window length); a leave-one-out open-set false-accept / true-accept curve; and a corruption-ablation curve (deletion/insertion/jitter).
5. Emits a metrics table, a per-contestant + per-window verdict via the gate, a sidecar JSON, and Trackio logs.

Output answers: does recall recover from 0.26 toward the note self-query ceiling, which contestant wins, at what window length, and how robust each is to note errors.

## Design

**Approach:** a pure-Python bake-off harness under `model/src/piece_id_eval/`, note-based throughout. Symbolic `Note` is the single substrate; queries and catalog are both `list[Note]`, so every matcher is graded apples-to-apples. The four contestants are swappable behind one `Matcher.rank(query: list[Note]) -> list[Ranked]` interface; the bake-off is that interface exercised four ways across the swept windows and the corruption grid.

**Key decisions and trade-offs:**
- **Note-based, not chroma-of-audio.** Eliminates the cross-modal gap that caused the KILL. Trade-off: depends on AMT transcription quality (the biggest risk) — which is exactly what the 16 amateur recordings measure.
- **Key-dependent everywhere; no transposition invariance.** Issue #21 + memory establish OTI measurably hurts (students play the written key). C2's token carries absolute `pc_anchor` (not a bare interval, which would be transpose-invariant). C1/C4 use un-normalized (key-dependent) pitch-class chroma. No OTI variants are built.
- **Ordinal (event-index) time in the landmark token, not seconds/beats.** Tempo-invariant by construction (a student at half speed yields the same note ordering), no beat tracker required. Trade-off: discards absolute duration; acceptable because tempo robustness is a hard requirement and beat-tracking amateur playing is unreliable.
- **Precision-dominated gate.** A wrong lock poisons the whole session (score-following aligns to the wrong score); "unknown" degrades to Tier-3 (today's behavior, zero regression). So the gate requires false-accept <= 0.05 even at the cost of true-accept >= 0.60. Negatives via leave-one-out (drop the true piece from the catalog; correct answer becomes "unknown") — 16 free negatives, no new audio.
- **Swept window, not a fixed guess.** The recall-vs-window curve is the artifact that picks the production window; pre-committing to one length would assume the answer the experiment exists to measure.
- **Corruption ablation as secondary diagnostic.** The 16 recordings are majority-amateur (covers real AMT error), but a controlled deletion/insertion/jitter grid is the contestant-differentiator (histogram C1/C4 degrade gracefully; landmark C2 loses every token a wrong note touches) and probes severities beyond the sample.

**Reuse vs rebuild:** reuse `Ranked` (NamedTuple), `metrics.recall_at_k`/`mrr`, the Trackio logging pattern, and `chroma_dtw_eval.amt_regen._transcribe_clip`/`_read_wav_16k_mono` for transcription. Rebuild the matcher set (the chroma matchers are dead) and the query loader (notes, not audio-chroma). Re-threshold `decision.py`.

## Modules

All under `model/src/piece_id_eval/`.

- **`notes.py`** — symbolic note substrate + loaders.
  - Interface: `Note(NamedTuple: onset, offset, pitch, velocity)`; `load_amt_notes(path) -> list[Note]`; `load_score_notes(path) -> list[Note]`.
  - Hides: the two on-disk JSON shapes (AMT `{onset,offset,pitch,velocity}` vs score `bars[].notes[].{onset_seconds,duration_seconds,pitch,velocity}`), sorting by onset, empty/invalid-note rejection.
  - Tested through: the two public loaders on committed fixtures.

- **`transcribe.py`** — AMT-notes cache builder (Task Group 0 prerequisite).
  - Interface: `ensure_amt_notes(audio_path, out_path, amt_url, force=False) -> Path`; CLI `python -m piece_id_eval.transcribe --slugs ...`.
  - Hides: 16k-mono wav read, AMT chunking + HTTP retries (delegated to `amt_regen`), idempotent cache write.
  - Tested through: `ensure_amt_notes` against a stub AMT HTTP server (real 15-slug run is an operational step, not a unit test).

- **`windowing.py`** — arbitrary-start subsequence sampler.
  - Interface: `sample_windows(notes, window_seconds, n_starts, seed) -> list[list[Note]]` (`window_seconds=None` -> single full-piece window).
  - Hides: uniform start-offset sampling, short-recording fallback, deterministic seeding.
  - Tested through: `sample_windows` output counts/extents on synthetic notes.

- **`note_chroma.py`** — note-derived chroma features (for C1/C4).
  - Interface: `chroma_vector(notes) -> np.ndarray (12,)` (key-dependent, L2-normalized); `chroma_sequence(notes, frame_seconds) -> np.ndarray (12, T)`.
  - Hides: pitch-class binning, framing, normalization.
  - Tested through: both functions on synthetic notes with known pitch classes.

- **`corruption.py`** — synthetic note degradation.
  - Interface: `corrupt_notes(notes, deletion_rate, insertion_rate, jitter_seconds, seed) -> list[Note]`.
  - Hides: RNG, the three independent corruption modes, pitch/onset bounds.
  - Tested through: `corrupt_notes` statistics at known rates/seeds.

- **`matchers/` (rebuilt)** — the four contestants behind one note-based protocol.
  - `base.py`: `Matcher` Protocol `rank(query: list[Note]) -> list[Ranked]`; reuse `Ranked`.
  - `note_chroma_matcher.py` (C1): cosine over key-dependent chroma vectors.
  - `landmark.py` (C2): tokens `(pc_anchor, interval, ordinal_gap)`, inverted hit-count index.
  - `dtw_ceiling.py` (C3, replaces chroma version): subsequence onset-ordered pitch DTW.
  - `chroma_seq_dtw.py` (C4): subsequence DTW over note-derived chroma sequences.
  - Each: interface `__init__(catalog: dict[str, list[Note]])` + `rank`. Hides indexing/DTW/normalization.
  - Tested through: `rank` — score self-query recall@1 == 1.0 on a small synthetic catalog (per matcher).

- **`open_set.py`** — leave-one-out false-accept / true-accept.
  - Interface: `operating_points(in_catalog_results, loo_results, thresholds) -> list[OperatingPoint]`; `best_point(points, max_fa, min_ta) -> OperatingPoint | None`.
  - Hides: threshold sweep, FA/TA computation, dominated-point selection.
  - Tested through: the two functions on synthetic ranked results.

- **`decision.py` (re-thresholded)** — the gate.
  - Interface: `decide(dtw_ceiling_recall10, best_indexable_recall10, open_set_ok_flag) -> str` (existing signature; new constants).
  - Hides: PROCEED/KILL/TUNE thresholds (recall@10 >= 0.85; FA <= 0.05 @ TA >= 0.60; KILL if ceiling recall@10 < 0.70).
  - Tested through: `decide` truth table.

- **`bakeoff.py`** — orchestrator + CLI (replaces `cli.py` for this design).
  - Interface: `run(catalog, recordings, window_lengths, corruption_grid, ...) -> BakeoffReport`; CLI `python -m piece_id_eval.bakeoff`.
  - Hides: window x matcher x corruption iteration, aggregation, sidecar + Trackio emission.
  - Tested through: `run` on a 2-piece synthetic catalog producing a populated `BakeoffReport`; CLI smoke with `--no-track`.

## Verification Architecture

- **Canonical success state:** `python -m piece_id_eval.bakeoff` completes and prints `VERDICT: KILL|TUNE|PROCEED`, with a per-(matcher, window) recall table, a leave-one-out FA/TA line, and corruption curves; a sidecar JSON is written; if PROCEED, the report names the winning matcher + window length.
- **Automated check:** `cd model && uv run pytest tests/piece_id_eval/` (all rebuilt tests pass). Plus a synthetic-catalog `bakeoff.run` integration test that asserts a populated report.
- **Harness (Task Group 0):** YES — two foundations the feature is built on, both buildable first:
  1. `notes.py` (the substrate every module imports) with fixture-based loader tests.
  2. `transcribe.py` with a stub-AMT test, then the **operational run** populating `amt_notes.json` for the 15 missing slugs (`just amt` + `python -m piece_id_eval.transcribe --slugs <15>`). The bake-off cannot run until this cache exists; the build agent must complete it before the final integration run.
- **Per-matcher golden check:** every matcher's first test is score self-query recall@1 == 1.0 on a synthetic catalog — the note-domain analogue of the 0.999 ceiling, proving plumbing before AMT noise enters.

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/piece_id_eval/notes.py` | `Note` type + AMT/score loaders | New |
| `model/src/piece_id_eval/transcribe.py` | AMT-notes cache builder + CLI | New |
| `model/src/piece_id_eval/windowing.py` | arbitrary-start window sampler | New |
| `model/src/piece_id_eval/note_chroma.py` | note-derived chroma features | New |
| `model/src/piece_id_eval/corruption.py` | synthetic note degradation | New |
| `model/src/piece_id_eval/matchers/base.py` | note-based `Matcher` protocol (was chroma `np.ndarray`) | Modify |
| `model/src/piece_id_eval/matchers/note_chroma_matcher.py` | C1 | New |
| `model/src/piece_id_eval/matchers/landmark.py` | C2 | New |
| `model/src/piece_id_eval/matchers/dtw_ceiling.py` | C3 (note-based; replaces chroma body) | Modify |
| `model/src/piece_id_eval/matchers/chroma_seq_dtw.py` | C4 | New |
| `model/src/piece_id_eval/matchers/chord_ngram.py` | dead chroma matcher | Delete |
| `model/src/piece_id_eval/matchers/twodft.py` | dead chroma matcher | Delete |
| `model/src/piece_id_eval/matchers/__init__.py` | export the four note matchers | Modify |
| `model/src/piece_id_eval/open_set.py` | leave-one-out FA/TA | New |
| `model/src/piece_id_eval/decision.py` | re-threshold (FA<=0.05@TA>=0.60) | Modify |
| `model/src/piece_id_eval/bakeoff.py` | orchestrator + CLI | New |
| `model/tests/piece_id_eval/test_notes.py` | loader behavior | New |
| `model/tests/piece_id_eval/test_transcribe.py` | stub-AMT cache write | New |
| `model/tests/piece_id_eval/test_windowing.py` | window sampling | New |
| `model/tests/piece_id_eval/test_note_chroma.py` | chroma features | New |
| `model/tests/piece_id_eval/test_corruption.py` | degradation stats | New |
| `model/tests/piece_id_eval/test_matchers.py` | four matchers (replaces chroma version) | Modify |
| `model/tests/piece_id_eval/test_open_set.py` | FA/TA | New |
| `model/tests/piece_id_eval/test_decision.py` | new thresholds (replaces old) | Modify |
| `model/tests/piece_id_eval/test_bakeoff.py` | synthetic-catalog integration + CLI smoke | New |
| `model/tests/piece_id_eval/test_query_chroma.py` | dead (audio-chroma) | Delete |
| `model/tests/piece_id_eval/test_score_chroma.py` | dead (audio-chroma) | Delete |
| `model/tests/piece_id_eval/test_query_set.py` | dead (audio-chroma) | Delete |
| `model/tests/piece_id_eval/test_cli_smoke.py` | superseded by test_bakeoff.py | Delete |
| `model/tests/piece_id_eval/test_report.py` | superseded (report folded into bakeoff) | Delete |
| `model/src/piece_id_eval/query_chroma.py` | dead (audio-chroma) | Delete |
| `model/src/piece_id_eval/score_chroma.py` | dead (audio-chroma) | Delete |
| `model/src/piece_id_eval/query_set.py` | dead (audio-chroma loader) | Delete |
| `model/src/piece_id_eval/cli.py` | replaced by bakeoff.py | Delete |
| `model/src/piece_id_eval/report.py` | folded into bakeoff.py | Delete |

## Open Questions

- Q: Landmark fan-out K (number of target notes paired per anchor) and max ordinal_gap. Default: K=5 target notes, ordinal_gap in 1..5; revisit only if C2 underperforms its self-query ceiling.
- Q: C4 chroma frame size. Default: 0.5s frames for `chroma_sequence`.
- Q: Number of window start offsets per recording per length. Default: 5 (5 lengths x 5 starts = 25 queries/recording).
- Q: DTW subsequence cost normalization for C3/C4. Default: total path cost divided by query length; rank by negative normalized cost.
- Q: Which recording per slug if multiple approved. Default: use all approved recordings that have cached audio; each contributes windows.
