# V6 Atoms Design

**Goal:** Implement all 15 atom ToolDefinition objects so the V6 harness loop's Phase 1 has a populated tool registry and can collect real DiagnosisArtifacts.

**Not in scope:**
- Molecule implementations (Plan 3)
- Wiring atoms into compound-registry (Plan 4)
- LLM calls or network I/O of any kind
- Changes to the loop infrastructure (already shipped in Plan 1)
- The `WASM` STOP classifier — the TS atom ports its math, not its binary

## Problem

`compound-registry.ts` has `tools: []` for the OnSessionEnd binding. Phase 1 calls Anthropic with zero tools, so it immediately emits `phase1_done` with `toolCallCount: 0`. Phase 2 receives `diagnoses: []` and writes a SynthesisArtifact with empty `focus_areas`, empty `strengths`, empty `diagnosis_refs`, and a hollow `headline`. The feature flag `HARNESS_V6_ENABLED` cannot be turned on until Plans 2–4 land.

## Solution

Each atom is a `ToolDefinition` object: `{ name, description, input_schema, invoke }`. The `invoke` function is a pure deterministic compute function — no network calls, no DB queries. "Fetch" atoms receive pre-materialized data in their input payload (the caller extracts it from `HookContext.digest` before calling). The LLM invokes atoms via the Phase 1 tool loop, which already handles retry, error capture, and result collection.

Atoms live in `apps/api/src/harness/skills/atoms/`. An index barrel re-exports all 15 as `ToolDefinition[]`.

## Design

**Key decision: "fetch" atoms are data-in / compute-out.** Atoms run inside a Cloudflare Worker with no ambient DB binding. The `HookContext.digest` already contains all materialized session cache data (muq scores, AMT transcription, pedal CC, score alignment, student baselines, session history). The LLM calls `fetch-student-baseline` with the pre-materialized session-means array it finds in the digest — the atom computes mean+stddev from that array. This keeps atoms testable in isolation with zero mocking.

**classify-stop-moment ports the Rust STOP classifier.** The coefficients in `stop.rs` are the ground truth: `SCALER_MEAN`, `SCALER_STD`, `WEIGHTS`, `BIAS`. The TS implementation applies StandardScaler → dot product → sigmoid, identical to the Rust path. The same coefficients are also in `apps/config/stop_config.json` (per the comment in stop.rs).

**align-performance-to-score implements a simplified DTW.** Full Rust DTW is in the WASM module; this TS version is semantically equivalent but uses JS arrays. Cost function: `|perf_onset - score_expected_onset| (normalized) + 100 * (pitch_mismatch)`. Unaligned notes (cost > 500) get `score_index: -1`.

## Modules

### 15 Atom Modules

Each exports one named constant: a `ToolDefinition`. The depth verdict is DEEP for all — a simple `invoke(input)` interface hides all numeric/algorithmic logic.

| Atom | Interface | Hides | Depth |
|------|-----------|-------|-------|
| `classify-stop-moment` | `invoke({scores: number[6]}) → number` | StandardScaler + logistic regression from stop.rs | DEEP |
| `compute-dimension-delta` | `invoke({dimension, current, baseline}) → number` | z-score formula; stddev-zero guard | DEEP |
| `compute-ioi-correlation` | `invoke({notes}) → number\|null` | Pearson r; null guard for n<4 aligned notes | DEEP |
| `compute-key-overlap-ratio` | `invoke({notes}) → number` | per-pair overlap ratio; mean; n<3 guard | DEEP |
| `compute-onset-drift` | `invoke({notes}) → OnsetDrift[]` | abs + signed drift per note | DEEP |
| `compute-pedal-overlap-ratio` | `invoke({notes, pedal_cc}) → number` | CC64 interval integration over note windows | DEEP |
| `compute-velocity-curve` | `invoke({bar_range, notes}) → VelocityCurve[]` | per-bar mean + p90; exact bar count validation | DEEP |
| `align-performance-to-score` | `invoke({perf_notes, score_notes}) → Alignment[]` | simplified DTW; unaligned sentinel -1 | DEEP |
| `detect-passage-repetition` | `invoke({attempts}) → RepetitionList[]` | 50% bar-overlap grouping; attempt count filter | DEEP |
| `extract-bar-range-signals` | `invoke({bar_range, chunks}) → SignalBundle` | chunk filter by bar coverage overlap; dedup; projection | DEEP |
| `fetch-reference-percentile` | `invoke({dimension, score, cohort_table}) → number` | linear interpolation between percentile buckets | DEEP |
| `fetch-session-history` | `invoke({sessions, window_days, now_ms}) → SessionHistory` | date-window filter; descending sort | DEEP |
| `fetch-similar-past-observation` | `invoke({dimension, piece_id, bar_range, past_diagnoses, now_ms}) → PastObservation\|null` | similarity = 0.5*piece_match + 0.5*bar_overlap; threshold 0.5 | DEEP |
| `fetch-student-baseline` | `invoke({dimension, session_means}) → Baseline\|null` | rolling mean+stddev; n<3 guard | DEEP |
| `prioritize-diagnoses` | `invoke({diagnoses}) → DiagnosisArtifact[]` | severity/confidence/dimension rank tuple; strengths-last | DEEP |

### Index Barrel

`apps/api/src/harness/skills/atoms/index.ts` — re-exports all 15 ToolDefinition objects and one `ALL_ATOMS: ToolDefinition[]` array for the registry wiring in Plan 4.

**Interface:** `export const ALL_ATOMS: ToolDefinition[]`
**Hides:** nothing (thin barrel)
**Depth:** SHALLOW — justified; its only role is aggregation for the registry

## Shared Types

These types are defined inline in the atoms that produce them, and imported by molecules:

```typescript
// In align-performance-to-score.ts
export type Alignment = { perf_index: number; score_index: number; expected_onset_ms: number | null; bar: number }

// In compute-velocity-curve.ts
export type VelocityCurve = { bar: number; mean_velocity: number; p90_velocity: number }

// In compute-onset-drift.ts
export type OnsetDrift = { note_index: number; drift_ms: number; signed: number }

// In extract-bar-range-signals.ts
export type MidiNote = { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar?: number }
export type CcEvent = { time_ms: number; value: number }
export type SignalBundle = { muq_scores: number[][]; midi_notes: MidiNote[]; pedal_cc: CcEvent[]; alignment: Alignment[] }

// In detect-passage-repetition.ts
export type RepetitionEntry = { bar_range: [number, number]; attempt_count: number; first_attempt_ms: number; last_attempt_ms: number }

// In fetch-student-baseline.ts
export type Baseline = { dimension: string; mean: number; stddev: number; n_sessions: number }

// In fetch-session-history.ts
export type SessionHistory = { sessions: { session_id: string; created_at: number; synthesis: unknown; diagnoses: unknown[] }[] }

// In fetch-similar-past-observation.ts
export type PastObservation = { artifact_id: string; session_id: string; days_ago: number; similarity_score: number }
```

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/harness/skills/atoms/align-performance-to-score.ts` | DTW alignment atom | New |
| `apps/api/src/harness/skills/atoms/align-performance-to-score.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/classify-stop-moment.ts` | STOP probability atom | New |
| `apps/api/src/harness/skills/atoms/classify-stop-moment.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/compute-dimension-delta.ts` | z-score atom | New |
| `apps/api/src/harness/skills/atoms/compute-dimension-delta.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/compute-ioi-correlation.ts` | Pearson r atom | New |
| `apps/api/src/harness/skills/atoms/compute-ioi-correlation.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/compute-key-overlap-ratio.ts` | Articulation overlap atom | New |
| `apps/api/src/harness/skills/atoms/compute-key-overlap-ratio.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/compute-onset-drift.ts` | Timing drift atom | New |
| `apps/api/src/harness/skills/atoms/compute-onset-drift.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/compute-pedal-overlap-ratio.ts` | Pedal overlap atom | New |
| `apps/api/src/harness/skills/atoms/compute-pedal-overlap-ratio.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/compute-velocity-curve.ts` | Velocity curve atom | New |
| `apps/api/src/harness/skills/atoms/compute-velocity-curve.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/detect-passage-repetition.ts` | Repetition detection atom | New |
| `apps/api/src/harness/skills/atoms/detect-passage-repetition.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/extract-bar-range-signals.ts` | Signal extraction atom | New |
| `apps/api/src/harness/skills/atoms/extract-bar-range-signals.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/fetch-reference-percentile.ts` | Cohort percentile atom | New |
| `apps/api/src/harness/skills/atoms/fetch-reference-percentile.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/fetch-session-history.ts` | Session history atom | New |
| `apps/api/src/harness/skills/atoms/fetch-session-history.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/fetch-similar-past-observation.ts` | Past observation lookup atom | New |
| `apps/api/src/harness/skills/atoms/fetch-similar-past-observation.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/fetch-student-baseline.ts` | Rolling baseline atom | New |
| `apps/api/src/harness/skills/atoms/fetch-student-baseline.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/prioritize-diagnoses.ts` | Diagnosis ranking atom | New |
| `apps/api/src/harness/skills/atoms/prioritize-diagnoses.test.ts` | Behavior test | New |
| `apps/api/src/harness/skills/atoms/index.ts` | Barrel export: ALL_ATOMS | New |

## Open Questions

- Q: Should atoms validate their input schema at runtime or trust callers?
  Default: Validate at runtime with a thrown `Error` if required fields are missing or wrong type. Atoms run inside the tool-call loop which already catches and surfaces errors.
- Q: Does `compute-ioi-correlation` treat aligned vs unaligned notes differently?
  Default: Only include notes where `expected_onset_ms` is not null (i.e., `score_index !== -1`) in the IOI computation.
