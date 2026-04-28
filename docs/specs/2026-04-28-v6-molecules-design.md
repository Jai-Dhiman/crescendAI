# V6 Molecules Design

**Goal:** Implement all 9 molecule ToolDefinition objects so the V6 harness loop's Phase 1 tool registry has concrete diagnosis and exercise skills that chain atom functions deterministically.

**Not in scope:**
- Compound wiring (Plan 4)
- LLM calls or network I/O of any kind
- Changes to atom implementations (Plan 2)
- Changes to the loop infrastructure (Plan 1)
- New artifact schemas

## Problem

`compound-registry.ts` bindings have no molecule tools registered. Molecules are the mid-tier diagnostic layer — they translate raw atom outputs (z-scores, velocity curves, onset drift arrays) into structured `DiagnosisArtifact` and `ExerciseArtifact` objects the compound can reason over. Without them the LLM sees only low-level atom results and must do the pedagogical reasoning inline, which is untrainable and inconsistent.

## Solution

Each molecule is a `ToolDefinition` object in `apps/api/src/harness/skills/molecules/`. Its `invoke()` function receives a pre-extracted signal bundle (all data materialized from the digest by the caller), calls atom `invoke()` functions directly with that data, applies the branching logic specified in the molecule's markdown spec, and returns a validated `DiagnosisArtifact` or `ExerciseArtifact`. No LLM, no network, no DB — pure deterministic compute, same contract as atoms.

## Design

**Key decision: molecules take a flat signal bundle, not a session ID.** Atoms are pure compute functions that accept their full data as input. Molecules call atoms by constructing the atom's input inline from the bundle. This keeps molecules testable in isolation without mocking any network boundary. The caller (compound) is responsible for extracting signals from `HookContext.digest` before invoking the molecule.

**Dimension index convention (matches atoms plan):** `[dynamics=0, timing=1, pedaling=2, articulation=3, phrasing=4, interpretation=5]`. All molecules reference `muq_scores[DIM_INDEX[dimension]]` with this constant object.

**Severity from z-score:** All diagnosis molecules use the same bucket mapping from the voicing-diagnosis spec: `|z| in [1.0, 1.5) → 'minor'`, `[1.5, 2.0) → 'moderate'`, `>= 2.0 → 'significant'`.

**finding_type branching:** Each molecule spec documents explicit branching — early `return { finding_type: 'neutral' }` paths vs full `DiagnosisArtifact` composition for `'issue'` and `'strength'` cases. The test verifies the issue path; neutral branching is verified by testing the condition just outside the threshold.

**exercise-proposal is the only action molecule.** It receives a `DiagnosisArtifact` as direct input (Option B contract) and returns an `ExerciseArtifact`. It calls `fetchSimilarPastObservation.invoke()` to check for a recent duplicate. Mapping from `primary_dimension + severity` to `exercise_type` is a fixed lookup table defined in the spec.

## Modules

### 8 Diagnosis Molecules

| Molecule | Interface | Hides | Depth |
|----------|-----------|-------|-------|
| `voicing-diagnosis` | `invoke(VoicingInput) → DiagnosisArtifact` | Velocity projection into top/bass voices; z-score branching; confidence from past-observation match | DEEP |
| `pedal-triage` | `invoke(PedalInput) → DiagnosisArtifact` | over/under/timing subtype classification; confidence bump from repeat observation | DEEP |
| `rubato-coaching` | `invoke(RubatoInput) → DiagnosisArtifact` | IOI correlation + net drift convergence; rushed vs dragged subtype | DEEP |
| `phrasing-arc-analysis` | `invoke(PhrasingInput) → DiagnosisArtifact` | Peak detection in velocity curve; drift convergence check; strength vs issue branching | DEEP |
| `tempo-stability-triage` | `invoke(TempoInput) → DiagnosisArtifact` | Monotonic drift fraction; slowing/rushing/unstable subtype | DEEP |
| `dynamic-range-audit` | `invoke(DynamicInput) → DiagnosisArtifact` | Observed velocity range vs score markings; score_marking_type classification | DEEP |
| `articulation-clarity-check` | `invoke(ArticulationInput) → DiagnosisArtifact` | Per-bar mismatch fraction; legato vs staccato direction logic | DEEP |
| `cross-modal-contradiction-check` | `invoke(CrossModalInput) → DiagnosisArtifact` | Four cross-modal pair checks; max-delta winner selection | DEEP |

### 1 Action Molecule

| Molecule | Interface | Hides | Depth |
|----------|-----------|-------|-------|
| `exercise-proposal` | `invoke(ExerciseInput) → ExerciseArtifact` | dimension+severity → exercise_type mapping; estimated_minutes by severity; action_binding construction | DEEP |

### Index Barrel

`apps/api/src/harness/skills/molecules/index.ts` — re-exports all 9 ToolDefinition objects and one `ALL_MOLECULES: ToolDefinition[]` array.

**Interface:** `export const ALL_MOLECULES: ToolDefinition[]`
**Depth:** SHALLOW — thin aggregation barrel, same pattern as `atoms/index.ts`

## Shared Input Types

Each molecule's invoke() accepts a typed input object. These types are defined inline in each molecule file:

```typescript
// voicing-diagnosis.ts
type VoicingInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  session_means_dynamics: number[]; cohort_table_dynamics: { p: number; value: number }[]
  past_diagnoses: { artifact_id: string; session_id: string; created_at: number; primary_dimension: string; bar_range: [number,number]|null; piece_id: string }[]
  piece_id: string; now_ms: number
}

// pedal-triage.ts
type PedalInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  pedal_cc: { time_ms: number; value: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number|null; bar: number }[]
  harmony_changes: { bar: number; time_ms: number }[]
  session_means_pedaling: number[]
  past_diagnoses: { artifact_id: string; session_id: string; created_at: number; primary_dimension: string; bar_range: [number,number]|null; piece_id: string }[]
  piece_id: string; now_ms: number
}

// rubato-coaching.ts
type RubatoInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number|null; bar: number }[]
  session_means_timing: number[]
  piece_id: string; now_ms: number
}

// phrasing-arc-analysis.ts
type PhrasingInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number|null; bar: number }[]
  cohort_table_phrasing: { p: number; value: number }[]
  piece_id: string; now_ms: number
}

// tempo-stability-triage.ts
type TempoInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number|null; bar: number }[]
  session_means_timing: number[]
  piece_id: string; now_ms: number
}

// dynamic-range-audit.ts
type DynamicInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  score_marking_type: 'wide' | 'medium' | 'narrow' | 'none'
  session_means_dynamics: number[]; cohort_table_dynamics: { p: number; value: number }[]
  piece_id: string; now_ms: number
}

// articulation-clarity-check.ts
type ArticulationInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  mono_notes_per_bar: { bar: number; notes: { onset_ms: number; duration_ms: number }[] }[]
  score_articulation_per_bar: { bar: number; articulation: 'slur' | 'staccato' | 'detache' }[]
  cohort_table_articulation: { p: number; value: number }[]
  piece_id: string; now_ms: number
}

// cross-modal-contradiction-check.ts
type CrossModalInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]
  muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  pedal_cc: { time_ms: number; value: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number|null; bar: number }[]
  mono_notes_per_bar: { bar: number; notes: { onset_ms: number; duration_ms: number }[] }[]
  score_articulation_per_bar: { bar: number; articulation: 'slur' | 'staccato' | 'detache' }[]
  cohort_baselines: { [dim: string]: { mean: number; stddev: number } }
  piece_id: string; now_ms: number
}

// exercise-proposal.ts
type ExerciseInput = {
  diagnosis: DiagnosisArtifact
  diagnosis_ref: string
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  past_diagnoses: { artifact_id: string; session_id: string; created_at: number; primary_dimension: string; bar_range: [number,number]|null; piece_id: string }[]
  piece_id: string; now_ms: number
}
```

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/harness/skills/molecules/voicing-diagnosis.ts` | Voicing imbalance molecule | New |
| `apps/api/src/harness/skills/molecules/voicing-diagnosis.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/molecules/pedal-triage.ts` | Pedal subtype molecule | New |
| `apps/api/src/harness/skills/molecules/pedal-triage.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/molecules/rubato-coaching.ts` | Rubato detection molecule | New |
| `apps/api/src/harness/skills/molecules/rubato-coaching.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/molecules/phrasing-arc-analysis.ts` | Phrase arc molecule | New |
| `apps/api/src/harness/skills/molecules/phrasing-arc-analysis.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/molecules/tempo-stability-triage.ts` | Tempo drift molecule | New |
| `apps/api/src/harness/skills/molecules/tempo-stability-triage.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/molecules/dynamic-range-audit.ts` | Dynamic range molecule | New |
| `apps/api/src/harness/skills/molecules/dynamic-range-audit.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/molecules/articulation-clarity-check.ts` | Articulation mismatch molecule | New |
| `apps/api/src/harness/skills/molecules/articulation-clarity-check.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/molecules/cross-modal-contradiction-check.ts` | Cross-modal contradiction molecule | New |
| `apps/api/src/harness/skills/molecules/cross-modal-contradiction-check.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/molecules/exercise-proposal.ts` | Exercise action molecule | New |
| `apps/api/src/harness/skills/molecules/exercise-proposal.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/molecules/index.ts` | Barrel: ALL_MOLECULES | New |
| `apps/api/src/harness/skills/molecules/index.test.ts` | Barrel behavior test | New |

## Open Questions

- Q: Should `cross-modal-contradiction-check` use per-molecule cohort baselines or a flat z-score for all four pair checks?
  Default: Use the `cohort_baselines` map in the input — the caller passes the cohort mean/stddev for each dimension so the molecule can compute z-scores without calling `fetch-student-baseline` (cross-modal is inherently cohort-relative, not student-relative).
- Q: Should `exercise-proposal` throw when `finding_type !== 'issue'`?
  Default: Yes — throw `Error('exercise-proposal: diagnosis must have finding_type "issue"')` to enforce the Option B contract explicitly.
