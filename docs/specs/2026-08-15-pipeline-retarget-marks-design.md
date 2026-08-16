# Pipeline Retarget: Transkun-first -> Mock Scorer -> MPM Marks Design

**Goal:** A local practice session produces baseline-gated marks with
MPM-grounded evidence end-to-end over the WebSocket, with zero MuQ references
on the request path.

**Not in scope:**
- Training or serving the real MoonBeam 6-dim head (mock scorer ships; the
  serving decision — local Docker container on the `apps/inference/amt`
  pattern — is recorded for when the head exists; issue #162 STATE 2026-08-15).
- Chunk-to-bar alignment. MuQ's chroma was the aligner's only input; it is
  NOT ported to the AMT container (user decision 2026-08-15: bar anchoring
  is unreachable in the live app until #160 wires piece context, so
  preserving it preserves nothing reachable). All sessions run the existing
  Tier-2 path; marks carry timestamp anchors. Bar anchoring returns as a
  dedicated follower issue (entry criteria: #119 HMM matcher + #160),
  behind the unchanged `BarMap` contract. The WASM aligner
  (`alignChunkNotes`/`alignChunkChroma`, #64) stays in the tree, dormant.
- Piece-picker surface and live piece ladder rungs 1-2 reachability (#160).
- Removing the chat UI, `chunk_processed`, `observation`, `synthesis`, or
  `session_summary` events from the web client (#164). The client keeps
  handling them; the server stops emitting `observation` for real sessions.
- Retargeting the teaching-knowledge eval to marks (follow-up; eval sessions
  keep the observation stream, user decision 2026-08-15).
- iOS.

## Problem

The DO session brain (`apps/api/src/do/session-brain.ts`) depends on the MuQ
HF endpoint twice on the hot path: `callMuqEndpoint` supplies the 6-dim
scores AND the chroma frames that drive tier-1 alignment
(`wasm.alignChunkNotes`, session-brain.ts:877). The epic-#154 target
architecture (docs/apps/02-pipeline.md) kills MuQ, but today:

- No server emits the `mark` WS event; the #157/#158 frontend contract
  (`apps/web/src/lib/mark.ts`, `practice-api.ts:87`) has zero server-side
  producers.
- The #163 baseline gate (`apps/api/src/services/student-baseline.ts`) is
  implemented, pure, and has no caller; `BaselineState` has no persistence.
- MPM per-bar feature extraction exists only in docs.
- The session-end output is LLM prose (`synthesis`/`session_summary`), not
  the target "marks + one verdict + one carry-forward"; #159's SessionReview
  renders fixtures because no server event carries a verdict.

## Solution (from the user's perspective)

The student records a session. While playing, nothing is delivered. At a
>=20s pause, marks land on the timeline canvas (client-side pause gating in
`pause-state.ts` already holds them; timestamp anchors until the follower
issue restores bars). On stop, the review shows one verdict sentence, the
session's marks, one carry-forward, and any disclosed no-comment zones.
Every mark's evidence line is a deterministic MPM statement ("around 2:41:
rushed — local tempo up 9%, uneven note lengths"), not model prose.

## Design

Chunk flow after this change (`finalizeChunk`):

1. `callAmtEndpoint` (Transkun) is the only inference call. MuQ client
   code, `MUQ_ENDPOINT`, and `apps/inference/muq/` are deleted, and with
   them the chroma input to `wasm.alignChunkNotes` — the tier-1 branch and
   the `chunk_bar_map` emission are removed; every chunk takes the existing
   Tier-2 path (`wasm.analyzeTier2`).
2. No-comment zone check: AMT rejected OR fewer than `MIN_NOTES_FOR_COMMENT
   = 10` transcribed notes -> record a zone (chunk index + reason), emit no
   mark, exclude the chunk from baseline samples and scoring. Proxy chosen
   because Transkun exposes no confidence signal (02-pipeline.md open
   question 2).
3. `scoreChunk` (new `services/scoring.ts`) produces the 6-dim scores — a
   deterministic pure-TS mock derived from MIDI statistics, behind the exact
   interface the future MoonBeam container call will implement.
4. MPM features (new `services/mpm-features.ts`): deterministic per-window
   features over the chunk's transcribed notes: local tempo and rubato
   (inter-onset-interval statistics), velocity mean/range/slope,
   articulation ratio (sounded duration / IOI), pedal overlap fraction.
   Each feature set renders to one evidence sentence. The interface
   accepts an optional bar-scoped input so the future follower issue adds
   per-bar granularity without an interface change. Guardrail #19:
   evidence only, never model inputs.
5. Mark gate (new `services/mark-gate.ts`, pure): per-dimension chunk
   samples vs the student's `BaselineState` band (longMean +/- halfWidth;
   within-session median reference when no persisted state — same
   cold-start-equals-fallback path as 02-pipeline.md section 7), requiring
   `firePersistence`-consecutive out-of-band chunks; deduped against marks
   already emitted for the same dimension; at most one mark per chunk.
   Taxonomy: below band = `needs_work`, above band = `strong`;
   `missed_opportunity` is reserved for score-marking comparisons and is
   not produced in this issue. Anchor candidates are timestamp-only in
   this issue (chunkIndex * 15s + offset); the client's `resolveAnchor`
   remains the single brander and will accept bars again when the
   follower issue lands.
6. Emission: `{type: "mark", mark}` per gate-passing candidate, immediately
   (client pause-gates display).
7. Session end (alarm path): V6 synthesis runs unchanged; the DO maps the
   artifact to a new `{type: "review", verdict, carry_forward,
   no_comment_zones}` event (verdict = `artifact.headline`, carry_forward =
   `artifact.next_session_focus`). Synthesis failure emits `{type:
   "review", verdict: null, ...}` — loud, the client renders its retry
   state. Then the DO folds the session's per-dimension samples through
   `updateBaseline` and persists `BaselineState` to a new
   `student_profiles.baseline_state` jsonb column. `observation` events are
   emitted only when `isEvalSession` (locked eval stays comparable).

Trade-offs: mock scores mean mark *content* is placeholder until the real
head lands, but every seam (scorer interface, gate, evidence, contract) is
real and tested. Timestamp-only anchors trade away bar-anchored marks —
acceptable because no live session can obtain a score context until #160,
so nothing reachable regresses; the follower issue restores bars behind
unchanged contracts. TS-side MPM extraction (not WASM) keeps the feature
layer cheap to iterate and unit-test.

## Modules

- **`services/scoring.ts`** — Interface: `scoreChunk(perfNotes, perfPedal):
  DimScores`. Hides: the entire scoring implementation (today a seeded
  deterministic mock; later a MoonBeam container call — callers cannot
  tell). Tested through: unit tests on determinism + range. DEEP.
- **`services/mpm-features.ts`** — Interface:
  `extractMpmFeatures(input): MpmFeatureSet[]` + `evidenceFor(set,
  dimension): string`. Hides: tempo/rubato decomposition, articulation and
  pedal math, windowing, prose rendering. Tested through: unit tests on
  synthetic note sets. DEEP.
- **`services/mark-gate.ts`** — Interface: `proposeMarks(input):
  MarkCandidate[]` (pure, no I/O, no clock). Hides: band computation,
  persistence counting, dedupe, taxonomy, anchor degradation. Tested
  through: unit tests over crafted score/baseline sequences. DEEP.
- **`services/baseline-store.ts`** — Interface:
  `loadBaselineState(db, studentId)` / `saveBaselineState(db, studentId,
  state)`. Hides: jsonb round-trip + `BaselineStateSchema` validation
  (loud `ValidationError` on corrupt rows). Shallow-ish by necessity (a
  store); justified: it is the single owner of the persistence format.
- **`do/session-brain.ts` wiring** — Modify only: swap inference calls,
  insert zone check, scorer, MPM, gate, mark emission, review event,
  baseline fold. Tested through: existing workerd integration tests +
  new ones exercising the WS surface.

## Verification Architecture

- Canonical success state: a workerd integration test drives a session
  (chunk_ready with fixture audio bytes stubbed at the AMT-client seam ->
  end_session) and observes on the WS: >=1 `mark` event whose
  `mark.evidence` matches an MPM evidence pattern and whose anchor is
  timestamp-typed, then exactly one `review` event with non-null `verdict`
  and `carry_forward`; and `student_profiles.baseline_state` is written.
- Automated checks: `just check-api` (0 errors), `just test-api` scoped to
  `apps/api` vitest (workerd + node), and
  `rg -il muq apps/api/src apps/api/wrangler.toml` returning no
  request-path hits (grep gate encoded as a test/step, not prose).
- Harness: buildable — Task Group 0 adds the failing end-to-end workerd
  test above (red until the final wiring group lands). Manual verification
  afterward: the user's deferred click-through now covers live marks via
  `just dev` (AMT container running locally).

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/services/inference.ts` | delete MuQ client/parse (`callMuqEndpoint`, `parseMuqResponse`, `MuqResult`, chroma fields) | Modify |
| `apps/api/src/services/scoring.ts` | mock 6-dim scorer behind stable interface | New |
| `apps/api/src/services/scoring.test.ts` | determinism/range tests | New |
| `apps/api/src/services/mpm-features.ts` | per-bar/per-window features + evidence prose | New |
| `apps/api/src/services/mpm-features.test.ts` | unit tests | New |
| `apps/api/src/services/mark-gate.ts` | pure candidate gate | New |
| `apps/api/src/services/mark-gate.test.ts` | unit tests | New |
| `apps/api/src/services/baseline-store.ts` | BaselineState jsonb load/save | New |
| `apps/api/src/db/schema/students.ts` | + `baselineState` jsonb column | Modify |
| `apps/api/src/db/migrations/0005_baseline_state.sql` | ALTER TABLE student_profiles ADD COLUMN baseline_state jsonb | New |
| `apps/api/src/do/session-brain.ts` | wiring per Design 1-7; remove tier-1 branch + `chunk_bar_map` emission | Modify |
| `apps/api/src/do/session-brain.schema.ts` | state: emittedMarks, sessionSamples, noCommentZones; outgoing mark/review schemas | Modify |
| `apps/api/src/do/session-brain.marks.test.ts` | Group-0 end-to-end harness + wiring tests | New |
| `apps/api/wrangler.toml` | remove MUQ_ENDPOINT | Modify |
| `apps/inference/muq/` | delete directory | Delete |
| `apps/web/src/lib/practice-api.ts` | + `review` event in PracticeWsEvent | Modify |
| `apps/web/src/hooks/usePracticeSession.ts` | handle `review` -> verdict/carryForward/noCommentZones state | Modify |
| `docs/apps/02-pipeline.md` | resolve open questions 1-2 (serving decision, confidence proxy); record stage-4 alignment as deferred to the follower issue | Modify |

## Open Questions

- Q: Should `MUQ_CHUNK_MS` and other MuQ-named internals be renamed?
  Default: yes, rename to `CHUNK_MS` etc. — the success criterion is zero
  MuQ references on the request path, and a grep gate enforces it.
- Q: `missed_opportunity` taxonomy unproduced this issue. Default: accepted
  — it requires score-marking comparison (a later contract milestone,
  02-pipeline.md "content upgrades").
