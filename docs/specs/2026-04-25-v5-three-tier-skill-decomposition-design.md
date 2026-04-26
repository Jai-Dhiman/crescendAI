# V5: Three-Tier Skill Decomposition Design

**Goal:** Replace CrescendAI's monolithic teacher prompt with a structured catalog of 28 markdown skill files (15 atoms / 9 molecules / 4 compounds) plus a typed-artifact contract layer, so that V6 can build the agent loop against stable interfaces and so that the Qwen finetune has clean per-skill training targets.

**Not in scope:**
- The agent loop runtime (V6).
- Middleware hooks (`before_model`, `wrap_tool_call`, etc.) — V6.
- Wiring `tool_call_spec` referenced by `ExerciseArtifact.action_binding` to actual tools — V6/V8.
- Student memory longitudinal logic — V7.
- Any modification to existing runtime code in `apps/api/src/services/` (`prompts.ts`, `practice-mode.ts`, `teacher.ts`, `synthesis.ts`, etc.) or in `apps/api/src/wasm/score-analysis/src/` (`stop.rs`, `score_follower.rs`).
- Replacing or rewriting any prompts that ship today.

## Problem

The current teacher pipeline (`apps/api/src/services/teacher.ts` ~16KB, `apps/api/src/services/prompts.ts` ~8.5KB, `apps/api/src/services/synthesis.ts` ~3KB, `apps/api/src/services/practice-mode.ts` ~11.6KB) encodes pedagogical reasoning as one large prompt with embedded conditional branches and inline output schemas. Three concrete failure modes:

1. **Untrainable.** The Qwen finetune (per `MEMORY.md` → `project_teacher_model_finetuning.md`) requires per-skill input/output specs to ground atomic-RL reward functions. A monolithic prompt produces composite reward signals that overfit to task structure (per *Skill Design* concept page in the Mahler wiki).
2. **Unevaluable.** The Phase 1 dual-judge plan (per `MEMORY.md` → `project_teaching_knowledge_eval.md`) needs narrow rubrics scoring individual pedagogical moves. There is currently no narrow surface to score; the prompt produces one teacher message that tangles diagnosis, exercise prescription, and tone.
3. **Uninspectable.** Changing teacher behavior today requires editing ~30KB of TypeScript and shipping a deploy. *Natural Language Harnesses* (NLAH) prescribes markdown-first skill files diffed and reviewed as artifacts, with the runtime providing only mechanical execution.

The harness anchor doc (`docs/harness.md`) names V5 as a NOW-priority deliverable that gates Qwen data collection, Phase 1 judge design, and the V6 agent loop.

## Solution (from the user's perspective)

There is no end-user-visible behavior change in V5. V5 is the foundation V6 builds on. After V5:

- A pedagogy reviewer can read `docs/harness/skills/molecules/voicing-diagnosis.md` and understand exactly when that move fires, what it consumes, and what artifact it writes — without reading any TypeScript.
- A test asserts every skill file conforms to its tier's composition rules; a malformed skill file fails CI before merge.
- The three durable artifact types (`DiagnosisArtifact`, `ExerciseArtifact`, `SynthesisArtifact`) have machine-checked Zod schemas; V6 will produce instances of these types and validate against the same schemas.
- The Qwen data-collection effort can target individual molecules and use their typed I/O as training-example structure.

## Design

### Approach

**Three-tier skill catalog** with strict composition rules:

- **Atoms (15)** — narrow, near-deterministic primitives. No calls to other skills. Closely mirror existing tool-function shapes (e.g., `apps/api/src/wasm/score-analysis/src/stop.rs` is the implementation that the `classify-stop-moment` atom describes the contract of).
- **Molecules (9)** — chain 2-10 atoms; one pedagogical move each. Consume signals AND prior-molecule artifacts (Option B, locked in brainstorm). No molecule-to-molecule calls.
- **Compounds (4)** — orchestrate molecules; one per hook (`OnSessionEnd`, `OnRecordingActive`, `OnWeeklyReview`, `OnPieceDetected-first-time`); single teacher-facing write per compound.

**Three typed artifact types** form the entire cross-skill interface:

- `DiagnosisArtifact` — written by all 8 diagnosis molecules + `cross-modal-contradiction-check`; consumed by `exercise-proposal` and `session-synthesis`.
- `ExerciseArtifact` — written by `exercise-proposal`; consumed by `session-synthesis` and `weekly-review`.
- `SynthesisArtifact` — written by all 4 compounds; consumed by V6 client / V7 student memory (out of V5 scope).

### Key decisions and rationale

| Decision | Choice | Why this over alternatives |
|----------|--------|---------------------------|
| Molecule input contract | Option B (signals + prior-molecule artifacts via compound) | NLAH durable-artifacts primitive; lets `exercise-proposal` consume diagnosis without re-deriving it. Option A (signals only) under-uses NLAH; Option C (fold exercise into diagnosis) violates atomic-RL principle. |
| Compound overlap policy | Session-synthesis reads live-companion's stop-moment artifacts as inputs | Per *Multi-Agents: What's Actually Working*: review-style agents perform better on a clean upstream artifact than on raw context. Avoids duplicate diagnosis work. |
| Live-companion artifact shape | Same `DiagnosisArtifact` type, scope-discriminated (`scope: stop_moment`) | One type beats two parallel types; session-synthesis aggregates a list of the same struct. |
| Diagnosis severity | 3-level enum (`minor`/`moderate`/`significant`) | Continuous values lose teacher-action information; 3 buckets match how a teacher prioritizes. Raw continuous value retrievable via `evidence_refs`. |
| Diagnosis dimension shape | `primary_dimension` scalar + `dimensions` list | Scalar enables capability-routing by dimension and grounds future per-dimension exercise recommendations; list preserves multi-dim findings. |
| Diagnosis "no-issue" outputs | Always emit; `finding_type: [issue, strength, neutral]` | Strengths drive teacher encouragement at synthesis time. Naming "negative" deliberately avoided (a strength is positive for the student). |
| Exercise type vocabulary | Fixed enum + `exercise_subtype` string escape hatch | Enum drives capability-routing for action tools; subtype absorbs nuance without breaking Qwen training stability. |
| Exercise → diagnosis link | Reference by id + frozen `diagnosis_summary` denorm | Reference is single source of truth (matches enrichment-cache pattern from *How to grep video*); summary is rendered without a join. |
| Exercise per-call cardinality | One exercise per `exercise-proposal` call | Atomic-RL training target stays clean; compound orchestrates progression by calling N times. |
| Synthesis structure | Hybrid C: structured fields source-of-truth + headline-as-projection | Serves both the student (teacher voice) and weekly-review (pattern mining). Headline derived LAST from structured fields. |
| Synthesis scope | `synthesis_scope: [session, weekly, piece_onboarding]` | One artifact type, scope-discriminated, with scope-conditional contracts. Mirrors DiagnosisArtifact's scope pattern. |
| Validator placement | `apps/api/src/harness/` (Option 1) | Schemas will be consumed by V6 runtime artifact validation; sharing one location avoids later refactor. |
| Test granularity | Per-skill test files in `__catalog__/` | Enables build-agent parallelism (15 atom tasks fan out as 15 subagents). |

### Composition rules (machine-enforced by the validator)

- Atoms: `depends_on: []` MUST be empty.
- Molecules: every entry in `depends_on` MUST be the `name` of an existing atom file. `reads.artifacts` MAY list `DiagnosisArtifact` (only `exercise-proposal` does so in V5).
- Compounds: every entry in `depends_on` MUST be the `name` of an existing molecule file OR an existing atom file (utility reads). `triggered_by` MUST be present (atoms and molecules MUST NOT have it).
- Every skill file MUST have YAML frontmatter that parses against the tier-specific frontmatter schema.
- Every skill file MUST have all 5 body sections present (`When-to-fire`, `When-NOT-to-fire`, `Procedure`, `Concrete example`, `Post-conditions`).

### Catalog (final)

**Atoms (15):** `compute-velocity-curve`, `compute-pedal-overlap-ratio`, `compute-onset-drift`, `compute-dimension-delta`, `fetch-student-baseline`, `fetch-reference-percentile`, `fetch-similar-past-observation`, `align-performance-to-score`, `classify-stop-moment`, `extract-bar-range-signals`, `compute-ioi-correlation`, `compute-key-overlap-ratio`, `detect-passage-repetition`, `prioritize-diagnoses`, `fetch-session-history`.

**Molecules (9):** `voicing-diagnosis`, `pedal-triage`, `rubato-coaching`, `phrasing-arc-analysis`, `tempo-stability-triage`, `dynamic-range-audit`, `articulation-clarity-check`, `exercise-proposal`, `cross-modal-contradiction-check`.

**Compounds (4):** `session-synthesis` (`OnSessionEnd`), `live-practice-companion` (`OnRecordingActive`), `weekly-review` (`OnWeeklyReview`), `piece-onboarding` (`OnPieceDetected`).

## Modules

### `apps/api/src/harness/artifacts/diagnosis.ts`
- **Interface:** `DiagnosisArtifact` Zod schema; exported TypeScript type via `z.infer`; const tuples for the 6 dimensions, 3 severity levels, 3 scope values, 3 finding types, 3 confidence levels.
- **Hides:** the invariant that `primary_dimension ∈ dimensions`; the 200-char cap on `one_sentence_finding`; the rule that `bar_range` MAY be `null` only when `scope: session`; the rule that `evidence_refs` MUST be non-empty.
- **Tested through:** Zod parse on a fixture set of 6 valid + 8 invalid examples (one invalid per refinement).

### `apps/api/src/harness/artifacts/exercise.ts`
- **Interface:** `ExerciseArtifact` Zod schema; exported TS type; const tuple for the 6 exercise types.
- **Hides:** the conditional that `action_binding` MUST be populated when `exercise_type ∈ {segment_loop, isolated_hands, pedal_isolation}`; the 400-char `instruction` cap; the 200-char `success_criterion` cap; the 1-15 `estimated_minutes` range; the `target_dimension ∈ 6dim` enum; the `bar_range` required-non-null invariant.
- **Tested through:** Zod parse on 6 valid + 6 invalid fixtures (each refinement).

### `apps/api/src/harness/artifacts/synthesis.ts`
- **Interface:** `SynthesisArtifact` Zod schema; exported TS type; const tuple for the 3 synthesis scopes.
- **Hides:** the conditional that `recurring_pattern` MUST be populated when `synthesis_scope: weekly`; the conditional that all `focus_areas[i].severity` MUST equal `minor` when `synthesis_scope: piece_onboarding`; the array max constraints (`strengths` ≤ 2, `focus_areas` ≤ 3, `proposed_exercises` ≤ 3); the 300-500 char `headline` range; the `dominant_dimension ∈ 6dim` enum.
- **Tested through:** Zod parse on 5 valid + 7 invalid fixtures.

### `apps/api/src/harness/artifacts/index.ts`
- **Interface:** Re-exports `DiagnosisArtifact`, `ExerciseArtifact`, `SynthesisArtifact` schemas + types + a discriminator `ArtifactName = 'DiagnosisArtifact' | 'ExerciseArtifact' | 'SynthesisArtifact'` and a lookup map `artifactSchemas: Record<ArtifactName, ZodSchema>`.
- **Hides:** Nothing — this is a barrel.
- **Tested through:** A single test asserting `artifactSchemas` has exactly the 3 expected keys (catches accidental drift if a fourth artifact type is added without updating the discriminator).

### `apps/api/src/harness/skills/validator.ts`
- **Interface:** `validateSkill(filePath: string): Promise<ValidationResult>` where `ValidationResult = { valid: boolean; errors: string[] }`. Also exports `validateCatalog(rootDir: string): Promise<CatalogValidationResult>` for cross-file checks (e.g., that every name listed in a molecule's `depends_on` resolves to a real atom file).
- **Hides:** YAML frontmatter parse, the per-tier frontmatter Zod schemas (atom-frontmatter, molecule-frontmatter, compound-frontmatter — three separate schemas keyed by the `tier` field), tier composition rule checks, body-section presence checks via markdown header scan, `writes` field artifact-name resolution against `artifactSchemas`, and `depends_on` cross-file resolution for `validateCatalog`.
- **Tested through:** A behavior test suite (`validator.test.ts`) using fixture markdown files. Per-skill catalog tests under `__catalog__/` exercise the same public API on the actual deliverables.

### Each skill markdown file (28 total)
- **Interface:** YAML frontmatter (tier-specific schema) + 5 body sections (`When-to-fire`, `When-NOT-to-fire`, `Procedure`, `Concrete example`, `Post-conditions`).
- **Hides:** Pedagogical reasoning for that move; trigger phrasing; negative boundaries; concrete worked example.
- **Tested through:** Per-skill test (`__catalog__/<tier>-<name>.test.ts`) calling `validateSkill('docs/harness/skills/<tier>/<name>.md')`.

## File Changes

| File | Change | Type |
|------|--------|------|
| `apps/api/src/harness/artifacts/diagnosis.ts` | Zod schema for DiagnosisArtifact | New |
| `apps/api/src/harness/artifacts/diagnosis.test.ts` | Behavior tests for the schema | New |
| `apps/api/src/harness/artifacts/exercise.ts` | Zod schema for ExerciseArtifact | New |
| `apps/api/src/harness/artifacts/exercise.test.ts` | Behavior tests | New |
| `apps/api/src/harness/artifacts/synthesis.ts` | Zod schema for SynthesisArtifact | New |
| `apps/api/src/harness/artifacts/synthesis.test.ts` | Behavior tests | New |
| `apps/api/src/harness/artifacts/index.ts` | Barrel + `artifactSchemas` map | New |
| `apps/api/src/harness/artifacts/index.test.ts` | Asserts the 3-keys invariant | New |
| `apps/api/src/harness/skills/validator.ts` | `validateSkill` + `validateCatalog` | New |
| `apps/api/src/harness/skills/validator.test.ts` | Behavior tests via fixture skill files | New |
| `apps/api/src/harness/skills/__fixtures__/*.md` | Fixture skill files for validator tests | New |
| `apps/api/src/harness/skills/__catalog__/atom-<name>.test.ts` × 15 | One per atom, validates the real skill file | New |
| `apps/api/src/harness/skills/__catalog__/molecule-<name>.test.ts` × 9 | One per molecule | New |
| `apps/api/src/harness/skills/__catalog__/compound-<name>.test.ts` × 4 | One per compound | New |
| `docs/harness/skills/atoms/<name>.md` × 15 | The 15 atom skill files | New |
| `docs/harness/skills/molecules/<name>.md` × 9 | The 9 molecule skill files | New |
| `docs/harness/skills/compounds/<name>.md` × 4 | The 4 compound skill files | New |
| `docs/harness/skills/atoms/README.md` | Replace candidate list with final list of 15 | Modify |
| `docs/harness/skills/molecules/README.md` | Replace candidate list with final list of 9 | Modify |
| `docs/harness/skills/compounds/README.md` | Replace candidate list with final list of 4 | Modify |

**Total: 10 TS files (3 artifact schemas + 1 barrel + 1 validator + 4 schema tests + 1 validator test) + 1 fixtures directory of 4 markdown files + 28 catalog test files + 28 skill markdown files + 3 README updates = 73 file operations.**

## Open Questions

- **Q:** Should the validator also check that every signal name a molecule references in `reads.signals` matches a known signal type emitted by V1 (MuQ 6-dim, AMT midi_notes, STOP probability, score-alignment)?
  **Default:** No for V5. Signal names are free-text descriptions of contract; V6 will introduce a typed signal registry. Adding a registry now expands V5 scope into V1/V6 territory.

- **Q:** Should `tool_call_spec` (the type of `ExerciseArtifact.action_binding`) be schemafied in V5 or left as `z.unknown()`?
  **Default:** Left as `z.unknown()` with a TODO comment in the schema file. The actual tool registry lives in V6/V8 and the schema would either be premature or have to be re-cut. The action_binding's per-exercise-type required-vs-not contract IS enforced in V5 — only the inner shape is deferred.

- **Q:** Should the spec require examples that are *real* (audio segment, real bar numbers from a real piece) or synthetic?
  **Default:** Synthetic but plausible. Real examples require coordinating with model/data team and risk leaking student data into spec docs. Synthetic examples with realistic numbers (bar 12-16 of an unspecified Chopin Ballade) are sufficient for the contract test, which only asserts that the example block exists.
