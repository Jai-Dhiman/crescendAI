# Ground Molecule Layer Design

**Goal:** V6 Phase-1 diagnosis molecules produce findings from real signal fetched server-side (not LLM-fabricated args), the conversing model sees only a compact per-chunk signal summary, and the glm 131K context overflow disappears as a side effect.

**Not in scope:**
- articulation-clarity-check molecule (requires score-articulation capability not yet available)
- phrasing-arc-analysis grounding (not in the 6 required molecules)
- eval routing, full-artifact judge, legacy synthesize() / HARNESS_V6_ENABLED deletion (all remain in #28)
- Baseline lock in #28 (locked after this ships)
- Molecule calling score-derived dynamics range when score lacks dynamics text
- Any changes to the iOS or web app surfaces

---

## Problem

`phase1.ts:41` inlines the full session digest as `JSON.stringify(ctx.digest, null, 2)`.

For a 10-chunk session the digest contains:
- Raw `midi_notes` arrays from every `EnrichedChunk` (no `bar` field — bar lives in `alignment`)
- Raw `pedal_cc` and `alignment` arrays

This raw dump is ~136K tokens on a 10-chunk session, which exceeds glm-4.7-flash's 131K context and causes a hard HTTP 413. 94% of the token cost is the raw per-note arrays.

Beyond the overflow, the 8 current molecules have `invoke` signatures that require ~7 fields that are not present in the digest at the time the LLM sees it (`muq_scores`, `midi_notes` with per-note `bar`, `pedal_cc`, `alignment`, `session_means_*`, `past_diagnoses`, `piece_id`, `now_ms`). The LLM either fabricates these or skips molecules entirely. The diagnostic layer is ungrounded.

Three additional breakages:
1. `EnrichedChunk.midi_notes` has no `bar` field; `bar` lives in `alignment[i].bar` keyed by `perf_index`. The `extract-bar-range-signals` atom already expects per-note `bar` but currently receives notes without it — molecules that rely on bar-filtered notes silently operate on the wrong data.
2. `PastDiagnosisRecord` in the DO does not select `id` or `pieceId`, so the `artifact_id` and `piece_id` fields that molecules need are missing.
3. `fetch-student-baseline` returns null when history < 3 sessions; three molecules throw on null, so thin-history students get no diagnoses.

---

## Solution (from the user's perspective)

After this ships, a student who completes a practice session gets a V6 SynthesisArtifact that includes molecule-grounded diagnoses. The Phase-1 prompt the LLM sees is a compact signal summary (a few hundred tokens), not a 136K raw dump. Molecules never fabricate inputs and never throw on first-time students; instead they fall back gracefully to a within-session baseline. The HTTP 413 disappears.

---

## Design

### Two deep modules

**`buildGroundedDigest(input: SynthesisInput, deps: GroundedDigestDeps): GroundedDigest`**

A pure synthesis-time adapter. Takes the full `SynthesisInput` (which has `EnrichedChunk[]`, `PastDiagnosisRecord[]`, `SessionHistoryRecord[]`, `baselines`, `COHORT_TABLES`) and produces:

- `chunks_adapted`: each `EnrichedChunk` annotated with `chunk_id = 'chunk:' + chunkIndex` and midi_notes with per-note `bar` injected via the `alignment.perf_index` join
- `mono_notes_per_bar`: derived from aligned notes, grouped by bar for cross-modal check
- `now_ms`: `Date.now()` at digest build time
- `cohort`: cohort mean and stddev per dimension, derived from `COHORT_TABLES` percentiles as `mean = p50`, `stddev = max(0.01, p84 - p50)`
- `past_diagnoses_grounded`: reshaped from the DO query to include `artifact_id` and `piece_id`
- `session_means`: per-dimension last-10-session means from a NEW DB query (`AVG(dimension_score) GROUP BY session_id` over `observations` table), or within-session fallback
- `within_session_means`: per-dimension means computed from this session's chunk scores (always present)
- `compact_signal_summary`: a short human-readable string summarising per-chunk MuQ scores and bar coverage (renders into Phase-1 prompt instead of raw digest)

The `deps` parameter carries the DB handle and student ID needed for the session-history query. `buildGroundedDigest` is called inside `synthesizeV6` before constructing the `HookContext`.

**`resolveMoleculeContext(digest: GroundedDigest, bar_range: [number, number] | null, scopeHint: 'stop_moment' | 'passage' | 'session'): ResolvedMoleculeContext`**

Called by each molecule's `invoke` after it extracts selectors from `ctx.digest`. Returns a `ResolvedMoleculeContext` containing:

- `bundle`: `SignalBundle` from `extractBarRangeSignals` (bar-range filtered; full-session if `bar_range` null)
- `baseline`: tiered — `Baseline` from `fetchStudentBaseline` if session_means length >= 3, otherwise synthesised from `within_session_means` so molecules never get null
- `cohort`: per-dimension `{mean, stddev}` from the grounded digest
- `past_diagnoses`: the grounded past-diagnosis records
- `piece_id`: from digest
- `now_ms`: from digest

### Molecule refactor

Each of the 6 grounded molecules becomes a thin wrapper:
1. LLM supplies only selectors: `bar_range`, `scope`, `evidence_refs` (the LLM knows these from the compact summary it sees)
2. `invoke(input, ctx)` calls `resolveMoleculeContext(ctx.digest as GroundedDigest, input.bar_range, input.scope)` to get all real data
3. Existing atom calls are unchanged — they now receive real data instead of fabricated data

The `input_schema` for each molecule is reduced to just `{bar_range, scope, evidence_refs}`. The LLM no longer supplies signal arrays.

### Phase-1 prompt change

`phase1.ts:41` changes from:
```
`Session digest:\n${JSON.stringify(ctx.digest, null, 2)}\n\n` + binding.procedurePrompt
```
to:
```
`Session summary:\n${(ctx.digest as GroundedDigest).compact_signal_summary}\n\n` + binding.procedurePrompt
```

### `extract-bar-range-signals` as optional read-tool

`extractBarRangeSignals` is added to the `OnSessionEnd` compound's tool list as an optional read-only tool, so the LLM can fetch raw bar-range signals for any passage if needed. The grounded digest provides it, so no fabrication is needed.

### `compound-registry.ts` procedurePrompt update

The `SESSION_SYNTHESIS_PROCEDURE` prompt is updated to remove the instruction "supply the bar range and signal data from the digest" and replace it with "supply only bar_range, scope, and evidence_refs — the molecule fetches all signal data server-side."

### `ALL_MOLECULES` cleanup

`articulationClarityCheck` is removed from `ALL_MOLECULES`. `phrasingArcAnalysis` remains (it uses cohort_table_phrasing which is available in the grounded digest). The 6 grounded molecules are:
1. pedal-triage
2. tempo-stability-triage
3. rubato-coaching
4. voicing-diagnosis
5. dynamic-range-audit
6. cross-modal-contradiction-check (3 score-independent arms only: timing-drift, pedal-ratio, dynamics-range; articulation arm is removed)

### Cross-modal articulation arm removal

The articulation arm in `crossModalContradictionCheck` requires `score_articulation_per_bar` (score-derived). With grounding, the molecule no longer receives this field, so the arm is removed. The remaining 3 arms (timing-drift-vs-MuQ, pedal-ratio-vs-MuQ, dynamics-range-vs-MuQ) are score-independent and remain.

### DO past-diagnoses query fix

The DO's diagnosis query at `session-brain.ts ~1674-1688` must also select `id` (→ `artifact_id`) and `pieceId`. `PastDiagnosisRecord` in `teacher.ts:73-80` is extended with `id: string` and `pieceId: string | null`.

### Tiered baseline

`resolveMoleculeContext` implements tiered baseline:
- If `session_means[dim].length >= 3`: use `fetchStudentBaseline` → returns a real multi-session `Baseline`
- Else: synthesise `Baseline` from `within_session_means[dim]` with `n_sessions = 1` and a safe stddev (0.1 floor)

Molecules receive a non-null `Baseline` in all cases and never throw on thin history.

---

## Modules

### `buildGroundedDigest`

**Interface:** `buildGroundedDigest(input: SynthesisInput, deps: { db: Db; studentId: string }): Promise<GroundedDigest>`

**Hides:**
- The `alignment.perf_index` join to annotate each MIDI note with its `bar`
- The DB query for per-session dimension means (`AVG(dimension_score) GROUP BY session_id`)
- COHORT_TABLES percentile arithmetic (`p50` as mean, `p84-p50` as stddev)
- `PastDiagnosisRecord` → `GroundedPastDiagnosis` reshaping (adds `artifact_id`, `piece_id`)
- `within_session_means` computation from chunk MuQ scores
- `compact_signal_summary` rendering (per-chunk one-liner: bar coverage + 6-vector)

**Tested through:** `buildGroundedDigest` public function called with synthetic `SynthesisInput` + mock DB; assert `GroundedDigest` fields.

**Depth verdict:** DEEP

---

### `resolveMoleculeContext`

**Interface:** `resolveMoleculeContext(digest: GroundedDigest, bar_range: [number, number] | null): Promise<ResolvedMoleculeContext>`

**Hides:**
- Calling `extractBarRangeSignals` to produce the bar-range `SignalBundle`
- Tiered baseline logic (multi-session vs within-session)
- Selecting cohort stats, past_diagnoses, piece_id, now_ms from digest

**Tested through:** `resolveMoleculeContext` public function with synthetic `GroundedDigest`; assert `ResolvedMoleculeContext` fields including baseline tier.

**Depth verdict:** DEEP

---

## Verification Architecture

- **Canonical success state:** A 10-chunk synthetic session's Phase-1 prompt string has serialized JSON length under 500K characters (well under 131K tokens). Each of the 6 molecules invoked with `(selectors, ctx-with-grounded-digest)` returns a valid `DiagnosisArtifact`. Insufficient-history path returns a neutral artifact, not a throw. Empty `bar_range` path returns a neutral artifact.
- **Automated check:** `bun run test --run apps/api/src` — all new unit tests pass, all pre-existing tests stay green. Typecheck net-new errors = 0 (baseline = 20 pre-existing errors).
- **Harness:** Unit tests are buildable. Overflow regression is a string-length assertion on the rendered prompt. Live integration (real glm run with no 413) is a `/review`-stage manual check.

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/harness/loop/grounded-digest.ts` | New module: `GroundedDigest` type + `buildGroundedDigest` | New |
| `apps/api/src/harness/loop/grounded-digest.test.ts` | Unit tests for `buildGroundedDigest` | New |
| `apps/api/src/harness/loop/resolve-molecule-context.ts` | New module: `ResolvedMoleculeContext` type + `resolveMoleculeContext` | New |
| `apps/api/src/harness/loop/resolve-molecule-context.test.ts` | Unit tests for `resolveMoleculeContext` | New |
| `apps/api/src/harness/skills/molecules/pedal-triage.ts` | Refactor: selectors-only input_schema, self-fetch via `resolveMoleculeContext` | Modify |
| `apps/api/src/harness/skills/molecules/pedal-triage.test.ts` | Rewrite tests for new (selectors, ctx) contract | Modify |
| `apps/api/src/harness/skills/molecules/tempo-stability-triage.ts` | Refactor: selectors-only input_schema, self-fetch | Modify |
| `apps/api/src/harness/skills/molecules/tempo-stability-triage.test.ts` | Rewrite tests for new contract | Modify |
| `apps/api/src/harness/skills/molecules/rubato-coaching.ts` | Refactor: selectors-only input_schema, self-fetch | Modify |
| `apps/api/src/harness/skills/molecules/rubato-coaching.test.ts` | Rewrite tests for new contract | Modify |
| `apps/api/src/harness/skills/molecules/voicing-diagnosis.ts` | Refactor: selectors-only input_schema, self-fetch | Modify |
| `apps/api/src/harness/skills/molecules/voicing-diagnosis.test.ts` | Rewrite tests for new contract | Modify |
| `apps/api/src/harness/skills/molecules/dynamic-range-audit.ts` | Refactor: selectors-only input_schema, self-fetch | Modify |
| `apps/api/src/harness/skills/molecules/dynamic-range-audit.test.ts` | Rewrite tests for new contract | Modify |
| `apps/api/src/harness/skills/molecules/cross-modal-contradiction-check.ts` | Refactor: selectors-only input_schema, remove articulation arm, self-fetch | Modify |
| `apps/api/src/harness/skills/molecules/cross-modal-contradiction-check.test.ts` | Rewrite tests for new contract | Modify |
| `apps/api/src/harness/skills/molecules/index.ts` | Remove `articulationClarityCheck` from `ALL_MOLECULES` | Modify |
| `apps/api/src/harness/loop/compound-registry.ts` | Update `SESSION_SYNTHESIS_PROCEDURE` prompt; add `extractBarRangeSignals` to tools | Modify |
| `apps/api/src/harness/loop/phase1.ts` | Replace raw digest dump with `compact_signal_summary` | Modify |
| `apps/api/src/services/teacher.ts` | Add `id`/`pieceId` to `PastDiagnosisRecord`; call `buildGroundedDigest` in `synthesizeV6` | Modify |
| `apps/api/src/do/session-brain.ts` | Add `id` and `pieceId` to the past-diagnoses SELECT query | Modify |

---

## Open Questions

- Q: Should `buildGroundedDigest` keep `chunks` in the digest for use as the `extract-bar-range-signals` read-tool input?  Default: Yes — the adapted `chunks_adapted` array is in `GroundedDigest` and is passed through `ctx.digest` so the tool can use it directly.
