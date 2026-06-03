# Chroma-Identification Feasibility Harness Design

**Goal:** One offline command measures whether harmonic (chroma) content can identify the correct piano piece from a growing catalog — printing recall@K / MRR / open-set metrics and a pre-registered `KILL | TUNE | PROCEED` verdict — so we know whether to build a chroma recall channel in Rust/WASM *before* writing any.

**Not in scope:**
- Any Rust/WASM channel (separate downstream issue, gated on a `PROCEED` verdict).
- `session-brain` / Durable Object integration.
- The symbolic/Aria recall channel (added when AMT deploys).
- AMT deployment.
- Issue #10's candidate-gate band-aid (stays as the cheap fallback).
- Production R2/D1 catalog upload (the harness reads local `model/data/scores/`).

## Problem

Piece identification is the cascade's weak link. Today's recall stage
(`apps/api/src/wasm/piece-identify/src/ngram.rs`) keys on pitch trigrams, which
fail on arpeggiated/harmonic piano music: BWV 846 produces one tied trigram hit
that DTW then rejects (documented in
`docs/implementation/2026-05-31-issue1-interval-trigrams-debug.md`). The proposed
fix is a chroma/harmonic-progression recall channel that also survives Tier 3
(chroma comes from the deployed MuQ endpoint, `apps/inference/muq/chroma.py`, not
from undeployed AMT).

That channel rests on one unproven claim: **harmony discriminates pieces where
melody fails.** Building a multi-day Rust channel on an unvalidated hypothesis
repeats the interval-trigram dead end. The cheapest thing that validates (or
kills) the hypothesis is an offline Python measurement over the real 254-piece
catalog — which doubles as the identification metric the eventual channel will be
tuned against.

## Solution (from the user's perspective)

The user runs `just piece-id-feasibility`. It downloads real audio for the 16
labeled `practice_eval` pieces, extracts production-identical chroma, searches
each query window against synthetic chroma fingerprints of the full catalog via
three matchers, and prints:

- a per-matcher table of recall@{1,5,10} and MRR;
- an open-set accept-curve (out-of-catalog false-accept vs in-catalog
  true-accept across confidence thresholds);
- a single final line `VERDICT: KILL` / `TUNE` / `PROCEED`.

The verdict tells the user whether to commit to building the Rust channel, keep
iterating the representation in Python, or abandon the chroma direction.

## Design

The harness mirrors the cascade's separation of concerns: a **library side**
(synthetic chroma fingerprints built from catalog score MIDI, reusing the exact
construction in `apps/api/src/wasm/score-analysis/src/chroma_dtw.rs`) and a
**query side** (real-audio chroma via the production extractor
`apps/inference/muq/chroma.py::chroma_feature`). The asymmetry — synthetic score
chroma vs. real performance chroma — *is the bet under test*, so the harness
measures it directly rather than assuming it away.

Three matchers are compared, deliberately:

- **DtwCeilingMatcher** — subsequence chroma-DTW of the query against each of the
  254 catalog pieces (slow, offline-only). This is the **discrimination ceiling /
  hypothesis-kill diagnostic**: if even the ideal matcher can't separate pieces,
  the direction is dead and no index will save it.
- **ChordNgramMatcher** — quantize each chroma window to a discrete chord token
  (12-bit pitch-class mask), form token n-grams, invert into a hit-counting
  index. The WASM-buildable, inverted-index recall (Shazam-analog for harmony).
- **TwoDFTMatcher** — 2D-Fourier-magnitude embedding of the chromagram, matched
  by cosine. The fixed-dim, ANN-able recall, inherently time-shift-invariant and
  (with pitch-axis magnitude) key-invariant.

Transposition invariance (OTI / pitch-axis magnitude) is a per-matcher boolean
variant, not a separate module — students mostly play in the written key, so it
is a secondary axis.

**Open-set probe via holdout:** rather than sourcing extra "not in catalog"
recordings, the harness holds a configurable set of the 16 pieces *out of the
searchable library* while still issuing their queries. Those queries are
structurally tagged `is_in_catalog=False`; any confident match against them is a
false-accept. This reuses real labeled audio and guarantees a genuine open-set
condition.

**Headline metric is recall@10**, because the chroma fingerprint is a *recall*
stage feeding rerank + DTW-confirm downstream — it must put the true piece in a
short verifiable list, not win outright. recall@1 and MRR are diagnostics.

**Pre-registered decision rule** (computed by a directly-tested pure function):

- `KILL` if `DtwCeilingMatcher` recall@10 < 0.70.
- `PROCEED` if some indexable matcher (ChordNgram or TwoDFT) reaches
  recall@10 >= 0.85 **and** an open-set threshold exists with out-of-catalog
  false-accept <= 0.10 at in-catalog true-accept >= 0.75.
- `TUNE` otherwise.

**Trade-offs chosen:** Python over Rust (5 representation variants in the time
one WASM rewrite takes; the result is a go/no-go, not production code). Synthetic
score chroma over rendered audio (no audio for 254 catalog pieces exists; the
synthetic-vs-real gap is the thing being measured). Holdout open-set over sourced
negatives (reuses labeled audio, zero extra data). `partitura` is **not** needed
here — catalog scores are already parsed JSON; no MIDI/MusicXML parsing occurs in
the harness.

## Modules

All under `model/src/piece_id_eval/`, mirroring the structure of the precedent
harness `model/src/chroma_dtw_eval/` (explicit errors, JSON-not-pickle, paths
anchored to `__file__`).

### `score_chroma.py` — catalog fingerprint primitive
- **Interface:** `build_score_chroma(notes, frame_rate_hz) -> np.ndarray (12, N)`;
  `load_catalog_score_chroma(score_path, frame_rate_hz) -> np.ndarray`.
- **Hides:** the per-note pitch-class accumulation, 1e-3 floor, and per-column
  L2-normalization that exactly mirror `chroma_dtw.rs::build_score_chroma`.
- **Tested through:** public functions on a 2-note toy → known column placement
  and unit norm. **DEEP.**

### `query_chroma.py` — production-identical audio chroma
- **Interface:** `audio_to_chroma(wav_path) -> (np.ndarray (12, N), frame_rate_hz)`;
  `window_chroma(chroma, frame_rate_hz, window_seconds, hop_seconds) -> list[np.ndarray]`.
- **Hides:** audio load, the reused `chroma_feature` call (imported from
  `apps/inference/muq/chroma.py`), byte→array unpacking, and audio-intrinsic
  windowing (no score bars — the piece is unknown).
- **Tested through:** a synthesized pure tone → dominant pitch-class bin; window
  count for a known-length signal. **DEEP.**

### `query_set.py` — labeled query corpus
- **Interface:** `LabeledQueryWindow(query_id, piece_id, is_in_catalog, chroma)`;
  `QuerySet.load(...) -> LoadResult(windows, excluded)`.
- **Hides:** `candidates.yaml` parsing, yt-dlp acquisition on cache miss,
  reuse of existing `pipeline_test/*.wav`, slug→catalog-id mapping via
  `eval_piece_map.json`, holdout tagging, and explicit exclusion accounting.
- **Tested through:** a fixture `candidates.yaml` + pre-cached local wav →
  correct labels, `is_in_catalog` flags, and excluded-count (no network in
  tests). **DEEP.**

### `matchers/` — the three recall families (one interface)
- **Interface:** `Matcher` protocol: `name: str`, `rank(query) -> list[(piece_id, score)]`
  (descending, higher = better). Impls `DtwCeilingMatcher`, `ChordNgramMatcher`,
  `TwoDFTMatcher`, each constructed from `{piece_id: score_chroma}` + an `oti` flag.
- **Hides:** subsequence DTW; chord-token quantization + inverted n-gram index;
  2D-FFT-magnitude embedding + cosine ANN.
- **Tested through:** a toy catalog where a query derived from one piece's own
  chroma ranks that piece #1 (unit-level circularity). **DEEP.**

### `metrics.py` — scoring
- **Interface:** `recall_at_k(rankings, k)`, `mrr(rankings)`,
  `open_set_curve(in_catalog, out_catalog, thresholds)`, `open_set_ok(curve, max_fa, min_ta)`.
- **Hides:** rank scanning, reciprocal-rank averaging, threshold sweep.
- **Tested through:** toy ranked lists with known answers. **DEEP.**

### `decision.py` — the gate
- **Interface:** `decide(dtw_recall10, best_indexable_recall10, open_set_ok_flag) -> str`.
- **Hides:** the pre-registered rule thresholds.
- **Tested through:** fixture metric sets → correct `KILL/TUNE/PROCEED`. **DEEP**
  (small surface, load-bearing logic).

### `report.py` — orchestration
- **Interface:** `EvalReport.run(query_windows, catalog_chromas, matchers, holdout_ids, thresholds) -> ReportResult`.
- **Hides:** running every matcher over in/out-of-catalog windows, aggregating
  all metrics, applying the decision rule.
- **Tested through:** an integration run on a toy 3-piece catalog with
  score-rendered queries → near-perfect recall and a deterministic verdict
  (circularity sanity). **DEEP.**

### `cli.py` — entry point
- **Interface:** `python -m piece_id_eval.cli [...]`; prints metrics table +
  `VERDICT: X`; exit 0 on success.
- **Hides:** arg wiring, catalog/query/matcher construction, Trackio logging
  (suppressible via `--no-track`), sidecar JSON.
- **Tested through:** subprocess smoke on toy fixtures → stdout contains a
  verdict line, exit 0. **DEEP.**

## Verification Architecture

- **Canonical success state:** the harness is itself a measurement tool, so
  "correct" means the *code* behaves correctly, not that the research verdict is
  any particular value. Correct behavior = (a) the decision-rule function returns
  the right verdict for fixture metrics; (b) recall/MRR are computed correctly on
  a toy catalog with known answers; (c) score-rendered queries against their own
  fingerprints yield near-perfect recall (circularity sanity); (d) a tagged
  out-of-catalog query yields below-threshold confidence.
- **Automated check:** `cd model && uv run pytest tests/piece_id_eval/` (unit +
  integration), plus a CLI subprocess smoke on toy fixtures.
- **Harness:** No separate Task Group 0 — the feature *is* a verification
  harness; its self-verification lives in the toy-fixture test suite above. The
  real-data run (`just piece-id-feasibility`) produces the research verdict,
  which is an output to interpret, not a code pass/fail.

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/piece_id_eval/__init__.py` | package marker | New |
| `model/src/piece_id_eval/score_chroma.py` | score-MIDI → synthetic chroma | New |
| `model/src/piece_id_eval/query_chroma.py` | audio → production chroma + windows | New |
| `model/src/piece_id_eval/acquire.py` | yt-dlp audio acquisition (cache-miss only) | New |
| `model/src/piece_id_eval/query_set.py` | labeled query corpus + holdout/open-set | New |
| `model/src/piece_id_eval/matchers/__init__.py` | matcher exports | New |
| `model/src/piece_id_eval/matchers/base.py` | `Matcher` protocol + `Ranked` type | New |
| `model/src/piece_id_eval/matchers/dtw_ceiling.py` | subsequence chroma-DTW ceiling | New |
| `model/src/piece_id_eval/matchers/chord_ngram.py` | chord-token n-gram inverted index | New |
| `model/src/piece_id_eval/matchers/twodft.py` | 2D-FFT-magnitude embedding | New |
| `model/src/piece_id_eval/metrics.py` | recall@k, MRR, open-set sweep | New |
| `model/src/piece_id_eval/decision.py` | pre-registered KILL/TUNE/PROCEED rule | New |
| `model/src/piece_id_eval/report.py` | orchestration + ReportResult | New |
| `model/src/piece_id_eval/cli.py` | CLI entry, table + verdict, Trackio | New |
| `model/data/evals/piece_id/eval_piece_map.json` | 16 slug → catalog piece_id mapping (already exists, shipped by #23) | Existing |
| `model/tests/piece_id_eval/*` | behavior tests (per module) | New |
| `Justfile` | add `piece-id-feasibility` recipe | Modify |

## Open Questions

- Q: Window length (1s vs 2s) and hop for query chroma?
  Default: 2.0s window, 1.0s hop (≈ MuQ chunk scale; tunable via CLI flags, so
  the feasibility run can sweep without code change).
- Q: Chord-token quantization threshold (which pitch classes are "on")?
  Default: binarize each chroma column at its own mean; token = 12-bit mask;
  OTI canonicalization = min over 12 cyclic rotations. Tunable in the matcher.
- Q: 2DFT embedding size?
  Default: 12 × 50 low-time-frequency block, L2-normalized. Tunable.
- Q: Which pieces to hold out for the open-set probe on the real run?
  Default: a CLI `--holdout` list; the feasibility run holds out ~3 of the 16 and
  rotates if a stronger open-set read is needed. Does not affect harness code.
