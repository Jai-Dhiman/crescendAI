# V2 Canonical Entity Schema Design

**Goal:** Define the three-layer (Content / Entity / Fact) typed schema that makes every reference in the harness — to a student, piece, session, exercise, signal, observation, or claim — collapse to a canonical row before any agent reasons about it, so V5 / V6 / V7 build on stable interfaces instead of free-text strings and untyped JSON.

**Not in scope:**
- SQL migrations for entity-mention or evidence normalization. Three additive migrations are *named* in the migration-path section as future work; they are not written here.
- Promoting Bar to a row type. Bar stays a composite-key addressable type until mutable per-bar state appears.
- Redefining V5's three artifact schemas (`DiagnosisArtifact`, `ExerciseArtifact`, `SynthesisArtifact`). They slot into Layer 1 as `schema_name` values; V2 references them, V2 does not redefine them.
- The `tool_call_spec` shape inside V5's `ExerciseArtifact.action_binding` — V5/V6/V8 territory.
- Wiring `EvidenceRef` resolution into existing V5 artifact consumers. V5 keeps its current shapes; V2 is additive.
- Any modification to `apps/api/src/services/`, `apps/api/src/wasm/`, `apps/api/src/db/schema/`, `apps/api/src/do/`.
- Master `apps/api/src/harness/index.ts` barrel — V5 spec already names it; V2 ships per-subdir indices only to avoid a merge collision between concurrent plans.

## Problem

The harness anchor doc (`docs/harness.md`) names V2 (entity schema) as a NOW-priority deliverable that gates V5 (skill catalog), V6 (agent loop), and V7 (student memory). Three concrete failure modes today:

1. **Free-text signal references.** `docs/specs/2026-04-25-v5-three-tier-skill-decomposition-design.md` Open Question 1 (line 143) explicitly defers a typed signal registry: "Signal names are free-text descriptions of contract; V6 will introduce a typed signal registry." Today every skill that reads a signal types it as a string; nothing catches a typo or a stale name.
2. **Untyped evidence pointers.** V5's `DiagnosisArtifact.evidence_refs` (line 86 of V5 spec) is "required non-empty" but the *shape* of a ref is undefined. Today's `synthesized_facts.evidence` column (`apps/api/src/db/schema/memory.ts:24`) is a `TEXT` JSON array of observation IDs only — no signal refs, no artifact refs, no type checker.
3. **Untyped entity mentions.** `synthesized_facts.entities` is a `jsonb` column (`apps/api/src/db/schema/memory.ts:30`) with no schema. The two-clocks framing in `docs/apps/03-memory-system.md` (lines 14-46) and the existing Three Layers subsection (lines 49-85) both name "resolved identities" as the substrate, but no spec defines what those identities are or how they are resolved.

Without V2, every agent trajectory through the context graph re-fights identity resolution and re-derives signal-name conventions in tokens and latency. From the Mahler wiki's *Context Graphs* page: "Every problem-directed agent trajectory through a well-layered graph is cheap; every trajectory through an un-resolved graph pays the identity-resolution cost again."

## Solution (from the user's perspective)

There is no end-user-visible behavior change in V2. V2 is the substrate V5 / V6 / V7 build on. After V2:

- A V5 skill author writing a molecule reads `docs/harness/entities.md` and knows exactly which signal types exist, exactly what an `EvidenceRef` looks like, and exactly which six entities can appear in `entityMentions[]`.
- V5's `ExerciseArtifact.action_binding` and `DiagnosisArtifact.evidence_refs` types tighten from `unknown` / free-text to typed unions imported from `apps/api/src/harness/content/`.
- V6's agent loop has a typed signal registry to validate `reads.signals` against.
- V7's longitudinal memory plan has a `Fact` schema with `validAt` / `invalidAt` / `evidence[]` already defined, so V7 spec writes against it instead of reinventing it.
- Every Zod schema in `apps/api/src/harness/{entities,content,facts}/` parses fixtures at test time; a malformed schema fails CI before merge.

## Design

### Approach

**Pointer-based three-layer spec.** Layer-1 storage stays distributed (signals in `chunk_results` / DO state / R2; observations in `observations`; artifacts in places V5 will name). The *address* is normalized via an `EvidenceRef` discriminated union. The *data* is not.

This is the load-bearing decision. Storage normalization (a unified `content` table) is a costly migration with no urgency; pointer normalization gives the type-safety benefit with zero migrations on day one and stays additive.

### Three layers

```
┌────────────────────────────────────────────────────────────────────┐
│ Layer 3: Facts (temporal assertions with evidence)                 │
│   Fact { validAt, invalidAt, entityMentions[], evidence[], ... }   │
│       │                                                             │
│       ▼ evidence[] :: EvidenceRef[]                                 │
├────────────────────────────────────────────────────────────────────┤
│ Layer 1: Content (immutable emissions, distributed storage)        │
│   Signal (chunk_results, DO state, R2) — model emissions           │
│   Observation (observations table) — pipeline LLM emissions        │
│   Artifact (V5 artifacts/* via schema_name) — skill emissions      │
│       │                                                             │
│       ▼ mentions Layer-2 entities                                   │
├────────────────────────────────────────────────────────────────────┤
│ Layer 2: Entities (resolved identities)                            │
│   Student, Piece, Movement, Bar, Session, Exercise                 │
└────────────────────────────────────────────────────────────────────┘
```

Per the Mahler wiki *Context Graphs* page, identity resolution lives at Layer 2 only. Layer 1 is content-addressed (no resolution needed — `{producer, schema_name, input_ref}` is unique by construction). Layer 3 is synthesized claims, not identities.

### Six Layer-2 entities

| Entity | Canonical key | Existing source | Resolution |
|---|---|---|---|
| Student | `apple_user_id` | `students.studentId` | Already enforced by Sign in with Apple |
| Piece | `piece_id` (`composer.catalogue_type_opus_number.piece_number`) | `pieces.pieceId` | Already done in `apps/api/src/wasm/piece-identify/` |
| Movement | `{piece_id, movement_index}` (TYPE, not row) | n/a — composite key only | Key match |
| Bar | `{piece_id, movement_index, bar_number}` (TYPE, not row) | n/a — composite key only | Key match |
| Session | `id` UUID | `sessions.id` | Server-issued, no resolution |
| Exercise | `id` UUID | `exercises.id` | Server-issued; dedup-on-insert by `{title, source}` |

Movement and Bar are intentionally *types*, not rows. A Chopin Ballade has ~260 bars; 100 pieces is ~50–100K bar rows on day one with no per-row mutable state. Promotion to rows is reversible (composite key IS the primary key — additive migration). Until a skill needs mutable per-bar state, the address-only form is sufficient.

### Layer-1 ContentRow + schema_name discriminator

Three concrete `schema_name` discriminators exist in V2:

```
ContentRow ::= Signal | Observation | Artifact
Signal.schema_name ::= 'MuQQuality' | 'AMTTranscription' | 'StopMoment' | 'ScoreAlignment'
Observation.schema_name ::= 'Observation'  (one shape today)
Artifact.schema_name ::= 'DiagnosisArtifact' | 'ExerciseArtifact' | 'SynthesisArtifact'
                         (defined in V5; V2 references the names)
```

V2 defines:
- `SignalSchema` — Zod discriminated union over the four signal `schema_name` values, with each variant carrying its own payload (MuQ 6-dim vector; AMT midi_notes + pedals; STOP probability + dimension; score alignment offsets).
- `ObservationSchema` — Zod schema mirroring `apps/api/src/db/schema/observations.ts` (lines 12-39), with `dimension` constrained to the 6-dim enum and `framing` constrained to `correction | recognition | encouragement | question`.
- `ArtifactRowSchema` — Zod base schema with `schema_name: string`, `schema_version: number`, `producer: string`, `created_at: timestamp`, and a `payload: unknown` slot. The `payload` is *intentionally* `unknown` here; V5's three concrete artifact schemas (`DiagnosisArtifact` etc.) live in `apps/api/src/harness/artifacts/` and are referenced — not redefined — by name.

### EvidenceRef discriminated union (load-bearing)

```typescript
EvidenceRef ::=
  | { kind: 'signal',      chunk_id: string, schema_name: SignalSchemaName, row_id: string }
  | { kind: 'observation', observation_id: string }
  | { kind: 'artifact',    artifact_id: string,  schema_name: ArtifactSchemaName }
```

This is the *only* type Layer 3 carries to point at Layer 1. It does **not** require a unified content table. Resolution helpers (`resolveSignal`, `resolveObservation`, `resolveArtifact`) live in their respective `apps/api/src/harness/content/*.ts` files and abstract the lookup; the union itself is storage-agnostic.

### Layer-3 Fact schema

```typescript
Fact = {
  id: string                    // UUID
  studentId: string             // FK to students
  factText: string              // natural-language assertion
  assertionType: 'recurring_issue' | 'recent_breakthrough' | 'student_reported' | 'piece_status' | 'baseline_shift'
  dimension?: SixDim            // optional
  validAt: Timestamp            // when fact became true in the world
  invalidAt: Timestamp | null   // when superseded; null = active
  entityMentions: EntityRef[]   // non-empty
  evidence: EvidenceRef[]       // non-empty
  trend?: 'improving' | 'stable' | 'declining' | 'new'
  confidence: 'high' | 'medium' | 'low'
  sourceType: 'synthesized' | 'student_reported' | 'inferred'
  createdAt: Timestamp
  expiredAt: Timestamp | null   // system clock — when row was superseded in DB
}
```

Bi-temporal: `validAt`/`invalidAt` is the world clock; `createdAt`/`expiredAt` is the system clock. Mirrors today's `synthesized_facts` columns (`apps/api/src/db/schema/memory.ts:11-42`) but tightens the `entities` and `evidence` jsonb/text fields into typed Zod schemas.

Invariants enforced by Zod refinements (each tested by one invalid fixture):
1. `entityMentions.length >= 1`
2. `evidence.length >= 1`
3. `invalidAt === null || invalidAt >= validAt`
4. `expiredAt === null || expiredAt >= createdAt`

### EntityRef discriminated union

```typescript
EntityRef ::=
  | { kind: 'student',  studentId: string }
  | { kind: 'piece',    pieceId: string }
  | { kind: 'movement', pieceId: string, movementIndex: number }
  | { kind: 'bar',      pieceId: string, movementIndex: number, barNumber: number }
  | { kind: 'session',  sessionId: string }
  | { kind: 'exercise', exerciseId: string }
```

### Identity resolution rules

| Entity | Rule | Notes |
|---|---|---|
| Student | `apple_user_id` is canonical | Sign in with Apple guarantees uniqueness; no merge needed |
| Piece | `pieceIdFromCatalogue({composer, catalogue_type, opus_number, piece_number})` returns the canonical `piece_id`; collision = same piece | `apps/api/src/wasm/piece-identify/` already implements; V2 documents the contract |
| Movement | `{piece_id, movement_index}` collision = same movement | Composite-key match |
| Bar | `{piece_id, movement_index, bar_number}` collision = same bar | Composite-key match |
| Session | UUID; no resolution | Server-issued at session start |
| Exercise | UUID at row level; dedup-on-insert by `{title, source}` lowercased-trimmed | `apps/api/src/db/schema/exercises.ts` does not enforce today; V2 exposes `exerciseDedupKey()` for callers to use; DB constraint is named-future-work |

### V5 alignment

V5's three artifact schemas live in `apps/api/src/harness/artifacts/{diagnosis,exercise,synthesis}.ts` and export `artifactSchemas: Record<ArtifactName, ZodSchema>` (V5 spec line 100). V2 mirrors that pattern:

- `apps/api/src/harness/entities/index.ts` exports `entityRefSchemas: Record<EntityKind, ZodSchema>` and `entityRefSchema` (the union).
- `apps/api/src/harness/content/index.ts` exports `contentSchemas: Record<ContentSchemaName, ZodSchema>`, `evidenceRefSchema`, and the `EvidenceRef` / `ContentRow` types.
- `apps/api/src/harness/facts/index.ts` exports `factSchema` and the `Fact` type.

Once V5 ships, the master `apps/api/src/harness/index.ts` barrel (V5's responsibility) re-exports both. V2 does not write that file to avoid concurrent-edit collision.

V5's deferrals close cleanly:
- Open Question 1 (signal registry): V2 ships `SignalSchemaName` enum and `SignalSchema`. V6 can validate `reads.signals` against `Object.keys(contentSchemas).filter(k => k in SignalSchemaName)`.
- `DiagnosisArtifact.evidence_refs: z.array(z.unknown())` → V5 narrows to `z.array(evidenceRefSchema)` once V2 lands. V5 spec change is a one-line import; not part of V2 plan.

### Migration path (named, not implemented)

Three additive migrations are documented in `entities.md` as future work; V2 writes none of them:

1. **`fact_entity_mentions` join table.** Today `synthesized_facts.entities` is unindexed jsonb. Future migration: extract to a join table with `(fact_id, entity_kind, entity_key)` rows for indexed lookup. Trigger: when "find all facts mentioning piece X" exceeds ~50ms via jsonb scan.
2. **`fact_evidence` join table.** Today `synthesized_facts.evidence` is a TEXT JSON array of observation IDs only. Future migration: extract to a join table with `(fact_id, evidence_kind, evidence_id, signal_schema_name?)` rows. Trigger: when signal-typed evidence enters facts (V6 agent loop).
3. **`signals` table** (optional). Today signals live distributed across `chunk_results` / DO state / R2. Future migration: a unified `signals` table for non-chunk-derived signals (e.g., session-level aggregates the V6 agent loop emits). Trigger: when a skill emits a signal that has no natural home in `chunk_results`.

All three are additive — `EvidenceRef` and `EntityRef` shapes do not change when these migrations land. Resolution helpers absorb the storage shift.

### Doc edit to `docs/apps/03-memory-system.md`

The existing Three Layers subsection (lines 49-85) is good. Two surgical changes:

1. Replace the inline "Mapping to Existing Tables" sub-table (lines 78-85) with a one-line citation: `See docs/harness/entities.md for the full layer mapping, schemas, and identity-resolution rules.`
2. Update the parenthetical `(V2 deliverable)` on line 85 to a present-tense link.

The "Two Clocks" subsection above (lines 14-46) is unchanged. The rest of the file is unchanged.

## Modules

### `apps/api/src/harness/entities/student.ts`
- **Interface:** `StudentSchema: ZodSchema`, `Student` type, `resolveStudent(input: { appleUserId: string }): { studentId: string }`.
- **Hides:** the rule that `studentId === appleUserId` (no separate canonical ID), the mirroring of `apps/api/src/db/schema/students.ts`.
- **Tested through:** `safeParse` on 1 valid + 1 invalid fixture (missing `appleUserId`); `resolveStudent` returns the canonical key.

### `apps/api/src/harness/entities/piece.ts`
- **Interface:** `PieceSchema`, `MovementRefSchema`, `BarRefSchema`, `Piece` / `MovementRef` / `BarRef` types, `pieceIdFromCatalogue(input: { composer, catalogueType, opusNumber, pieceNumber }): string`.
- **Hides:** the canonical-key construction (`composer.catalogueType_opusNumber.pieceNumber`), the rule that movement/bar are addressable types not rows, the mirroring of `apps/api/src/db/schema/catalog.ts`.
- **Tested through:** `safeParse` on 1 valid Piece + 1 invalid (missing `barCount`); `safeParse` on valid `MovementRef` and `BarRef`; `pieceIdFromCatalogue` returns expected string for a known input.

### `apps/api/src/harness/entities/session.ts`
- **Interface:** `SessionSchema`, `Session` type.
- **Hides:** nullable-end invariant (`endedAt === null || endedAt >= startedAt`), accumulator-shape opacity, mirroring of `apps/api/src/db/schema/sessions.ts`.
- **Tested through:** `safeParse` on 1 valid + 1 invalid (`endedAt < startedAt`).

### `apps/api/src/harness/entities/exercise.ts`
- **Interface:** `ExerciseSchema`, `Exercise` type, `exerciseDedupKey(input: { title: string, source: string }): string`.
- **Hides:** the dedup canonicalization (`title.trim().toLowerCase() + '|' + source.trim().toLowerCase()`), mirroring of `apps/api/src/db/schema/exercises.ts`.
- **Tested through:** `safeParse` on 1 valid + 1 invalid (empty `title`); `exerciseDedupKey` returns same string for whitespace/case variants of the same `{title, source}`.

### `apps/api/src/harness/entities/index.ts`
- **Interface:** `EntityRef` discriminated union (Zod + TS type) with 6 variants, `entityRefSchema`, `entityRefSchemas: Record<EntityKind, ZodSchema>`, `EntityKind` enum.
- **Hides:** the discriminator-key convention (`kind`), the registry-map shape that mirrors V5's `artifactSchemas` precedent.
- **Tested through:** `safeParse` on 1 valid `EntityRef` of each `kind` (6 valid fixtures) + 1 invalid (`kind: 'unknown'`).

### `apps/api/src/harness/content/signal.ts`
- **Interface:** `SignalSchema` (Zod discriminated union over 4 signal schema names), `Signal` type, `SignalSchemaName` enum (`MuQQuality`, `AMTTranscription`, `StopMoment`, `ScoreAlignment`).
- **Hides:** the four signal-payload variants (MuQ 6-dim vector; AMT midi_notes + pedals; STOP probability + dimension; score alignment offsets), the rule that `chunk_id` is the input_ref, the rule that `producer_version` is required.
- **Tested through:** `safeParse` on 1 valid signal of each `schema_name` (4 valid fixtures) + 1 invalid (`schema_name: 'unknown'`) + 1 invalid (MuQ payload missing one of the 6 dimensions).

### `apps/api/src/harness/content/observation.ts`
- **Interface:** `ObservationSchema`, `Observation` type.
- **Hides:** the 6-dimension enum, the framing enum, the mirroring of `apps/api/src/db/schema/observations.ts`.
- **Tested through:** `safeParse` on 1 valid + 1 invalid (`dimension: 'unknown'`).

### `apps/api/src/harness/content/artifact.ts`
- **Interface:** `ArtifactRowSchema` (Zod base), `ArtifactRow` type, `ArtifactSchemaName` enum imported from `apps/api/src/harness/artifacts/` (V5).
- **Hides:** the rule that `payload: unknown` here is intentional (V5 narrows it via per-artifact schema lookup), the `schema_name`-as-discriminator pattern.
- **Tested through:** `safeParse` on 1 valid + 1 invalid (missing `schema_name`). V2 does NOT test V5 artifact-payload conformance — V5 owns those tests.

### `apps/api/src/harness/content/index.ts`
- **Interface:** `EvidenceRef` discriminated union (Zod + TS type), `evidenceRefSchema`, `ContentRow` union (Signal | Observation | ArtifactRow), `contentSchemas: Record<ContentSchemaName, ZodSchema>`.
- **Hides:** the registry mapping each `schema_name` to its Zod parser, the storage-agnostic pointer pattern, the convention that `EvidenceRef` is the only type Layer 3 carries.
- **Tested through:** `safeParse` on 1 valid `EvidenceRef` of each `kind` (3 valid fixtures) + 1 invalid (`kind: 'unknown'`); `contentSchemas` registry has expected keys.

### `apps/api/src/harness/facts/fact.ts`
- **Interface:** `FactSchema`, `Fact` type, `AssertionType` enum (`recurring_issue` | `recent_breakthrough` | `student_reported` | `piece_status` | `baseline_shift`).
- **Hides:** the four bi-temporal invariants (entityMentions non-empty; evidence non-empty; `invalidAt >= validAt` when set; `expiredAt >= createdAt` when set), the dimension and framing enums.
- **Tested through:** `safeParse` on 1 valid + 4 invalid (one per refinement).

### `apps/api/src/harness/facts/index.ts`
- **Interface:** Re-exports `FactSchema`, `Fact`, `AssertionType`.
- **Hides:** Nothing — barrel.
- **Tested through:** Implicitly by `facts/fact.test.ts` (no separate test).

### `docs/harness/entities.md`
- **Interface:** Markdown doc — six sections per the V2 brainstorm: three-layer diagram, six entity Zod definitions, identity resolution rules per entity, fact-layer schema, evidence-chain example, migration path.
- **Hides:** Nothing — it's a reference doc.
- **Tested through:** Not auto-tested. Cited by V5/V6/V7 specs; correctness enforced indirectly because every Zod definition shown in the doc maps to a tested file in `apps/api/src/harness/`.

### `docs/apps/03-memory-system.md` (modify)
- **Interface:** Existing memory-system doc. Two surgical edits per the "Doc edit" subsection above.
- **Hides:** N/A — doc.
- **Tested through:** Not auto-tested.

## File Changes

| File | Change | Type |
|---|---|---|
| `docs/specs/2026-04-25-v2-canonical-entity-schema-design.md` | This spec | New (committed in plan-skill step) |
| `docs/plans/2026-04-25-v2-canonical-entity-schema.md` | TDD impl plan | New (committed in plan-skill step) |
| `docs/harness/entities.md` | The V2 reference doc | New |
| `docs/apps/03-memory-system.md` | Surgical edits to the existing Three Layers subsection (lines 78-85) | Modify |
| `apps/api/src/harness/entities/student.ts` + `.test.ts` | Student Zod schema + behavior test | New |
| `apps/api/src/harness/entities/piece.ts` + `.test.ts` | Piece + Movement + Bar Zod schemas + behavior test | New |
| `apps/api/src/harness/entities/session.ts` + `.test.ts` | Session Zod schema + behavior test | New |
| `apps/api/src/harness/entities/exercise.ts` + `.test.ts` | Exercise Zod schema + behavior test | New |
| `apps/api/src/harness/entities/index.ts` + `.test.ts` | EntityRef union + registry + behavior test | New |
| `apps/api/src/harness/content/signal.ts` + `.test.ts` | Signal Zod discriminated union + registry + behavior test | New |
| `apps/api/src/harness/content/observation.ts` + `.test.ts` | Observation Zod schema + behavior test | New |
| `apps/api/src/harness/content/artifact.ts` + `.test.ts` | ArtifactRow Zod base + behavior test | New |
| `apps/api/src/harness/content/index.ts` + `.test.ts` | EvidenceRef union + ContentRow + contentSchemas registry + behavior test | New |
| `apps/api/src/harness/facts/fact.ts` + `.test.ts` | Fact Zod schema + AssertionType enum + behavior test | New |
| `apps/api/src/harness/facts/index.ts` | Facts barrel | New |

**Total: 1 doc spec + 1 doc edit + 10 TS source files + 9 TS test files + 1 barrel = 22 files (excluding the spec and plan files themselves).**

## Open Questions

- **Q:** Should V2 ship a runtime `validateContentRow(row, schema_name)` helper that does the `contentSchemas` lookup-and-parse, or is exposing the registry map sufficient?
  **Default:** Registry map only. Adding a validator helper is a thin wrapper that V5/V6 can write at point of use; shipping it in V2 commits to an API surface no consumer has asked for. Add when V5 or V6 first imports it.

- **Q:** Should `Movement` carry a `title` field (e.g., "Allegro maestoso") in `MovementRefSchema`, or is the index alone enough?
  **Default:** Index only. Title is presentational metadata; it lives on the score data fetched from R2 (`/api/scores/:pieceId/data`) and is not part of *identity*. Including it in the ref would mean two refs with same `(piece_id, movement_index)` but different titles are non-equal — a bug.

- **Q:** Should `EvidenceRef.kind = 'signal'` carry `producer_version` in addition to `schema_name`, so that an evidence chain pinned to model v2 doesn't get confused with model v3 emissions on the same chunk?
  **Default:** Not in V2. The `chunk_results` row already carries `producer_version` and `row_id` resolves the version. Adding it to the ref bloats every Fact's evidence array. Revisit when model v3 ships and the first evidence-chain ambiguity appears.
