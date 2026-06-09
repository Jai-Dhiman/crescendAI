# Piece-ID Phase-1 Certified Gate Design

**Goal:** Replace the legacy 3-stage piece-ID pipeline (ngram → rerank → DTW-confirm) with the Phase-0-CERTIFIED 2-stage gate (chroma recall → pitch-only chord-Jaccard elastic-DTW margin gate) in production Rust/WASM + the catalog artifact, so a practice session locks to the correct score only when the margin clears the certified threshold and otherwise stays `unknown` (Tier-3) instead of locking to a wrong piece.

**Not in scope:**
- The `session-brain.ts` cross-chunk note-buffer rewire — specified here as the **DEFERRED slice**, BLOCKED-ON-#28 (which is rewriting `session-brain.ts`); it lands as a follow-up after #28 merges. No `session-brain.ts` edits in this PR.
- Recall **certification** (TA capped at the 16 amateur recordings) — recall failure is graceful (miss → `unknown` → Tier-3) and does not gate this work.
- Production deploy / R2 upload to the hosted bucket (`upload.py`). Pre-beta, local-first: "done" = local green (`cargo test` + `vitest` + `wrangler dev` manual click-through).
- Any re-introduction of a timing/IOI term (Phase-0 proved it HURTS — pitch-only is the certified config).
- `text_match.rs` (free-text query matching) — left untouched.

## Problem

`apps/api/src/wasm/piece-identify/` implements a 3-stage pipeline — `ngram_recall` (`ngram.rs`) → `rerank_candidates` (`rerank.rs`) → `dtw_confirm` (`dtw_confirm.rs`, a monophonic L1 subsequence DTW). Phase-0 (#26, Stage-0 through Stage-0f) falsified this design's open-set behavior:

- **C1 note-chroma ranks well** (R@10=0.938) but **cannot reject** a harmonically-similar wrong piece on its own (wrong-piece cosine 0.952–0.994 overlaps correct 0.960–0.999). A wrong lock poisons the whole session (score-following aligns to the wrong score).
- The cheap monophonic DTW (`dtw_confirm.rs`, = the Stage-0b instrument) **FAILED** as an open-set gate (best TA=0.31 @ FA=0): a fixed-window monophonic L1 slide with no chord handling underperforms chroma even as a ranker (R@1=0.625 < chroma 0.875).

Phase-0 then **CERTIFIED** a different gate (Stage-0c → Stage-0f): chroma recall (top-5) → **pitch-only chord-Jaccard elastic subsequence-DTW margin** gate. The false-accept (wrong-lock) axis is certified across 322 diverse out-of-catalog real-performance works (foreign-composer fa=0/97; same-composer fa=1/225; both upper-95%-CI ≤ 0.0133 ≤ 0.05). Production still runs the falsified pipeline. This spec ports the certified gate.

## Solution (from the user's perspective)

During a practice session, once enough notes have accumulated, the system attempts to identify the piece. It locks to a catalog piece **only** when the certified margin gate is confident (margin between the best and 2nd-best candidate ≥ 0.0935); otherwise it stays in Tier-3 ("unknown") and never mis-locks. No user-visible UI change in this PR — the change is the identification *decision quality* and the WASM/artifact substrate. The DEFERRED slice wires the accumulated cross-chunk buffer into this gate.

## Design

### Frozen algorithm (ported verbatim from the Phase-0 reference — do NOT redesign)

Reference (present on this branch): `model/src/piece_id_eval/stage0c_elastic_dtwgate.py` (`_notes_to_events`, `_elastic_cost`, `ElasticGate`), `model/src/piece_id_eval/note_chroma.py` (`chroma_vector`), `model/src/piece_id_eval/matchers/note_chroma_matcher.py` (`rank`), `model/src/piece_id_eval/stage0f_hard_ood_certify.py` (`_score_candidate`, `_W_TIME=0.0`).

1. **Chroma vector** (key-dependent): 12-bin pitch-class histogram, each note adds its `velocity` to bin `pitch % 12`, then L2-normalize. (`note_chroma.py:chroma_vector`.)
2. **Recall**: rank all catalog pieces by `dot(query_chroma, piece_chroma)` (cosine, since vectors are L2-normalized) descending; take top-K = 5. (`note_chroma_matcher.py:rank`.)
3. **Events**: collapse note onsets within 50 ms into chord-events; each event is a 12-bit pitch-class **set** (bit `i` = pc `i` present). Pitch-only ⇒ **no onset/IOI data retained**. (`stage0c.py:_notes_to_events` with `w_time=0`.)
4. **Elastic cost** (one query vs one candidate): local cost between two events = Jaccard **distance** of their pitch-class sets = `1 - |A∩B| / |A∪B|`. Subsequence DTW (free start/end, 3-direction min) embeds the **shorter** event-sequence as rows in the longer; cost = `min(accumulated last row) / shorter_event_count`. (`stage0c.py:_elastic_cost`, `w_pitch=1, w_time=0`.)
5. **Margin gate**: compute elastic cost of the query against each of the chroma top-5; sort ascending; `margin = second_best_cost − best_cost`. **Accept (lock to best)** iff `margin ≥ PIECE_ID_MARGIN_THRESHOLD`; else **unknown**. (`stage0f.py:_score_candidate`.)

**Certified operating point (frozen constant):** `PIECE_ID_MARGIN_THRESHOLD = 0.0935`. At this point Stage-0f reports TA=0.875, fa_loo=0, fa_ood=0.0044 (upper-95%-CI 0.0133, certifiable). Source: `model/data/evals/piece_id/stage0f_hard_ood_certify_results.json` (`chosen_point.threshold = 0.09353`) corroborated by `stage0c_..._results.json` full/`v3_topk_margin` best point 0.0969.

### Key decisions (locked)

- **Chroma recall in Rust/WASM, not TS.** One WASM export `identify_piece` runs the whole chain. Rationale: single source of truth with the Python reference (the port-fidelity test pins both), the artifact already loads into WASM, recall runs once/session off the hot path so Rust↔TS perf is irrelevant. Trade-off rejected: a TS recall would split the algorithm across two languages and double the surfaces the parity test must cover.
- **Artifact crosses the WASM boundary as a JSON string**, parsed by `serde_json` inside Rust — not `serde_wasm_bindgen` on a ~2–4 MB parsed object. Cleaner boundary, faster deserialize, matches "once per session."
- **The legacy 3-stage pipeline is deleted, not extended.** `ngram.rs`, `rerank.rs`, `dtw_confirm.rs`, `real_recording_test.rs` are removed; `ngram_recall` / `compute_rerank_features` / `rerank_candidates` / `dtw_confirm` exports removed from `lib.rs`. `text_match.rs` + `match_piece_text` untouched.
- **Artifact is versioned `v2`** (`fingerprint/v2/piece_index.json`) to avoid clobbering the live `v1` keys and to make the DO's load path an explicit cutover (DEFERRED slice).
- **Subsequence-DTW port replicates librosa exactly:** free-start row-0 = `C[0,:]`, free-end `min(D[last,:])`, 3-direction min — which the existing `subsequence_dtw` already does — PLUS transpose-shorter-to-rows and normalize-by-shorter (the two behaviors the legacy version lacked).

### Phase-1 artifact schema (`fingerprint/v2/piece_index.json`)

```json
{
  "version": "v2",
  "onset_tol_ms": 50,
  "pieces": [
    {
      "piece_id": "chopin.ballades.23",
      "composer": "Chopin",
      "title": "Ballade No. 1 in G minor, Op. 23",
      "chroma": [0.0, 0.13, ...],          // 12 floats, L2-normalized velocity-weighted pc histogram
      "events": [2049, 16, 4, ...]          // sequence of u16 12-bit pc-set masks (onsets collapsed within 50ms)
    }
  ]
}
```

This single file replaces the legacy `ngram_index.json` + `rerank_features.json` + the never-generated `catalog.json` (it carries `piece_id`/`composer`/`title` metadata too).

## Modules

### `chroma.rs` (NEW, Rust/WASM)
- **Interface:** `pub fn chroma_vector(notes: &[PerfNote]) -> [f64; 12]`; `pub fn rank_top_k(query: &[f64; 12], catalog: &[(usize, [f64; 12])], k: usize) -> Vec<usize>` (returns catalog indices, cosine-desc).
- **Hides:** velocity-weighted pitch-class accumulation, L2 normalization, dot-product ranking, top-k selection.
- **Tested through:** public fns via `cargo test` (known notes → expected vector; known catalog → expected ranked indices).
- **Depth:** DEEP (simple numeric interface; hides the recall stage).

### `gate.rs` (NEW, Rust/WASM — supersedes `dtw_confirm.rs`)
- **Interface:** `pub fn notes_to_events(notes: &[PerfNote], onset_tol_s: f64) -> Vec<u16>`; `pub fn elastic_cost(q: &[u16], r: &[u16]) -> f64`; `pub fn margin_gate(query_events: &[u16], candidate_event_lists: &[&[u16]], threshold: f64) -> GateDecision` where `GateDecision { best_index: usize, margin: f64, locked: bool }`.
- **Hides:** 50 ms onset bucketing into 12-bit masks, Jaccard local cost, subsequence-DTW (transpose-shorter-to-rows, free start/end, normalize-by-shorter), best/2nd-best margin computation, threshold decision.
- **Tested through:** public fns via `cargo test` (hand-computed Jaccard cost; margin/lock decisions).
- **Depth:** DEEP (the entire certified gate behind three small fns).

### `lib.rs` `identify_piece` (NEW export; MODIFY `lib.rs`)
- **Interface:** `#[wasm_bindgen] pub fn identify_piece(notes_js: JsValue, artifact_json: &str, margin_threshold: f64) -> Result<JsValue, JsValue>` → `{ piece_id, composer, title, margin, locked } | null`. Returns `null` when query has < 2 events or artifact has < 2 pieces.
- **Hides:** artifact JSON parse, chroma recall (top-5), per-candidate elastic cost, margin gate, result marshaling.
- **Tested through:** `cargo test` integration (notes + small artifact → expected decision) and a workerd `vitest` test through `wasm-bridge.ts`.
- **Depth:** DEEP (the public face of the whole feature; one call, full decision).

### `build_piece_index` (MODIFY `model/src/score_library/fingerprint.py`)
- **Interface:** `build_piece_index(scores_dir: Path, onset_tol_s: float = 0.05) -> dict` → the `fingerprint/v2/piece_index.json` structure above.
- **Hides:** score-JSON traversal, chroma computation (mirrors `note_chroma.chroma_vector`), event quantization (mirrors `stage0c._notes_to_events` pc-set masks), metadata extraction.
- **Tested through:** Python test over a tiny fixture scores dir (asserts schema + chroma L2-norm + event masks).
- **Depth:** DEEP.

### `export_parity_fixtures.py` (NEW, `model/src/piece_id_eval/`) — verification harness
- **Interface:** `python -m piece_id_eval.export_parity_fixtures` → writes `model/data/evals/piece_id/parity_fixtures.json`.
- **Hides:** loading the 16 in-catalog recordings + a sample of OOD MAESTRO works, running the FROZEN Stage-0c/0f gate to compute per-query `expected_margin` / `expected_best_piece_id` / `expected_locked` at threshold 0.0935, and emitting the same `events`/`chroma` inputs the WASM will see.
- **Tested through:** a Python test asserting the exporter reproduces the certified point (≥14/16 in-catalog full-piece queries `expected_locked=true`; OOD queries `expected_locked=false`).
- **Depth:** DEEP.

### `identifyPiece` (MODIFY `apps/api/src/services/wasm-bridge.ts`)
- **Interface:** `export function identifyPiece(notes: PerfNote[], artifactJson: string, marginThreshold?: number): IdentifyResult | null` with `IdentifyResult { piece_id: string; composer: string; title: string; margin: number; locked: boolean }`.
- **Hides:** the `serde_wasm_bindgen` call boundary; replaces `ngramRecall`/`rerankCandidates`/`dtwConfirm` wrappers.
- **Tested through:** workerd `vitest` (real WASM) — lock on in-catalog fixture, unknown on OOD fixture.
- **Depth:** SHALLOW-by-design (a thin typed forwarder, the established `wasm-bridge.ts` pattern). Justified: matches the existing bridge convention; the depth lives in the WASM module it fronts.

## Verification Architecture

- **Canonical success state:** for every frozen fixture query, the Rust/WASM `identify_piece` reproduces the Python reference gate's `locked` decision exactly AND its `margin` within tolerance `1e-4`; at threshold 0.0935 in-catalog fixtures LOCK (TA point estimate 0.875 ⇒ ≥14/16) and OOD fixtures stay `unknown`.
- **Automated check:** `cargo test -p piece-identify parity` (primary correctness gate, native f64 precision) + `bun run test` workerd vitest for the bridge boundary + `bun run test:scripts` is unaffected. A new `just test-piece-id` recipe runs the Rust tests (currently un-wired).
- **Harness (Task Group 0, buildable before the feature):** `export_parity_fixtures.py` freezes the Python reference's decisions+margins into a committed `parity_fixtures.json`. This is the golden file the Rust parity test asserts against. Built and committed first; the Rust parity test is then written to fail (no `identify_piece` yet) and driven green by the port.

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/piece_id_eval/export_parity_fixtures.py` | Freeze Python-reference decisions/margins into golden fixtures | New |
| `model/data/evals/piece_id/parity_fixtures.json` | Committed golden fixtures (the harness output) | New |
| `apps/api/src/wasm/piece-identify/src/chroma.rs` | Chroma vector + top-k recall | New |
| `apps/api/src/wasm/piece-identify/src/gate.rs` | Events + Jaccard elastic cost + margin gate | New |
| `apps/api/src/wasm/piece-identify/src/lib.rs` | Add `identify_piece` export; remove legacy exports + modules | Modify |
| `apps/api/src/wasm/piece-identify/src/types.rs` | Add `PieceArtifact`, `PieceIndex`, `IdentifyResult`; keep `PerfNote`, `CatalogEntry` | Modify |
| `apps/api/src/wasm/piece-identify/src/parity_test.rs` | Rust parity test reading `parity_fixtures.json` | New |
| `apps/api/src/wasm/piece-identify/src/ngram.rs` | Legacy recall — delete | Delete |
| `apps/api/src/wasm/piece-identify/src/rerank.rs` | Legacy rerank — delete | Delete |
| `apps/api/src/wasm/piece-identify/src/dtw_confirm.rs` | Legacy monophonic DTW — delete (superseded by `gate.rs`) | Delete |
| `apps/api/src/wasm/piece-identify/src/real_recording_test.rs` | Legacy 3-stage test — delete | Delete |
| `model/src/score_library/fingerprint.py` | Replace ngram/rerank builders with `build_piece_index` | Modify |
| `model/src/score_library/cli.py` | `fingerprint` subcommand emits `piece_index.json` | Modify |
| `Justfile` | `fingerprint` emits v2 artifact; add `seed-fingerprint` (local R2) + `test-piece-id` (cargo test) | Modify |
| `apps/api/src/services/wasm-bridge.ts` | Replace 3 legacy wrappers with `identifyPiece` | Modify |
| `apps/api/src/services/wasm-bridge.workerd.test.ts` | Replace `ngramRecall` real-WASM test with `identifyPiece` lock/unknown | Modify |
| `apps/api/src/services/wasm-bridge.test.ts` | Replace mocked `ngramRecall` forwarding test with `identifyPiece` | Modify |

### DEFERRED slice (BLOCKED-ON-#28, spec only — no edits this PR)

| File | Change (follow-up) |
|------|--------|
| `apps/api/src/do/session-brain.schema.ts` | Add `identificationNoteBuffer: z.array(perfNoteSchema).default([])` (Zod-versioned), cap at `MAX_IDENTIFICATION_BUFFER` notes |
| `apps/api/src/do/session-brain.ts` | Replace `tryIdentifyPiece(perfNotes,...)` + `identificationNoteCount` count with: append chunk notes to buffer; once `buffer.length >= MIN_NOTES_FOR_IDENTIFICATION`, fetch `fingerprint/v2/piece_index.json` text, call `wasm.identifyPiece(buffer, artifactJson, 0.0935)`; on `locked` set `pieceLocked` + `pieceIdentification`; else stay unknown (Tier-3). Re-verify line numbers at build time. |

## Open Questions

- Q: How many OOD works to include in `parity_fixtures.json`?  Default: all 16 in-catalog recordings (full-piece) + 12 OOD MAESTRO works sampled deterministically (seed 42) from `stage0f_hard_ood_analysis.json`'s scored set, balanced foreign/same-composer — enough to exercise both lock and unknown without a multi-GB fixture.
- Q: Does the DEFERRED slice land in this PR if #28 merges before review?  Default: NO — keep this PR's scope to the WASM+artifact+harness so it can ship independently; the rewire is a separate follow-up branched off post-#28 `main`, closing #26 only when the rewire lands. This PR's body says `Closes #26` only if the rewire is included; otherwise it references #26 and the rewire PR closes it.
