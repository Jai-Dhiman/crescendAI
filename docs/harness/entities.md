> **Status (2026-04-26):** V2 spec landed. Six entity schemas, ContentRow base, EvidenceRef union, and Fact schema implemented in `apps/api/src/harness/`. Three additive migrations named below as future work.

## Three-Layer Diagram

The harness organizes every piece of information in a practice session into three layers. Layer 1 (Content) holds immutable emissions from the pipeline: model signals, LLM-generated observations, and V5 skill artifacts. Layer 2 (Entities) holds resolved identities — the six canonical entity types that Content and Facts reference by key. Layer 3 (Facts) holds synthesized temporal assertions about a student, each pointing back into Layer 1 via an `EvidenceRef` union and into Layer 2 via an `EntityRef` union. Identity resolution lives at Layer 2 only; Layer 1 is content-addressed by `{producer, schema_name, input_ref}` and needs no resolution; Layer 3 is synthesized claims, not identities.

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

## Canonical Entity Types

### Student

The top-level learner identity. `studentId` equals the Apple User ID issued by Sign in with Apple — no separate canonical ID.

Canonical key: `studentId`

```typescript
// apps/api/src/harness/entities/student.ts
export const StudentSchema = z.object({
  studentId: z.string().min(1),
  inferredLevel: z.string().nullable().optional(),
  baselineDynamics: z.number().nullable().optional(),
  baselineTiming: z.number().nullable().optional(),
  baselinePedaling: z.number().nullable().optional(),
  baselineArticulation: z.number().nullable().optional(),
  baselinePhrasing: z.number().nullable().optional(),
  baselineInterpretation: z.number().nullable().optional(),
  baselineSessionCount: z.number().int().nonnegative(),
  explicitGoals: z.string().nullable().optional(),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
});
```

File: `apps/api/src/harness/entities/student.ts`

---

### Piece

A catalogued score. The canonical key is constructed from catalogue metadata, not a database sequence.

Canonical key: `pieceId` (`composer.catalogueType_opNumber.pieceNumber`)

```typescript
// apps/api/src/harness/entities/piece.ts
export const PieceSchema = z.object({
  pieceId: z.string().min(1),
  composer: z.string().min(1),
  title: z.string().min(1),
  keySignature: z.string().nullable().optional(),
  timeSignature: z.string().nullable().optional(),
  tempoBpm: z.number().int().nullable().optional(),
  barCount: z.number().int().positive(),
  durationSeconds: z.number().nullable().optional(),
  noteCount: z.number().int().positive(),
  pitchRangeLow: z.number().int().nullable().optional(),
  pitchRangeHigh: z.number().int().nullable().optional(),
  hasTimeSigChanges: z.boolean(),
  hasTempoChanges: z.boolean(),
  source: z.string(),
  opusNumber: z.number().int().nullable().optional(),
  pieceNumber: z.number().int().nullable().optional(),
  catalogueType: z.string().nullable().optional(),
  createdAt: z.string().datetime(),
});
```

File: `apps/api/src/harness/entities/piece.ts`

---

### Movement

An addressable movement within a piece. Movement is a *type*, not a database row — it has no mutable per-movement state today, and the composite key is sufficient for addressing. Promotion to a row is a future additive migration (same key becomes the primary key).

Canonical key: `{ pieceId, movementIndex }`

```typescript
// apps/api/src/harness/entities/piece.ts
export const MovementRefSchema = z.object({
  pieceId: z.string().min(1),
  movementIndex: z.number().int().nonnegative(),
});
```

File: `apps/api/src/harness/entities/piece.ts`

---

### Bar

An addressable bar within a movement. Bar is a *type*, not a database row. A Chopin Ballade has ~260 bars; at 100 pieces that is 50–100K rows with no per-row mutable state today. The composite key is sufficient for addressing. Promotion to a row is a future additive migration.

Canonical key: `{ pieceId, movementIndex, barNumber }`

```typescript
// apps/api/src/harness/entities/piece.ts
export const BarRefSchema = z.object({
  pieceId: z.string().min(1),
  movementIndex: z.number().int().nonnegative(),
  barNumber: z.number().int().nonnegative(),
});
```

File: `apps/api/src/harness/entities/piece.ts`

---

### Session

A single practice session. UUID is server-issued at session start; no resolution step needed.

Canonical key: `id` (UUID)

```typescript
// apps/api/src/harness/entities/session.ts
export const SessionSchema = z
  .object({
    id: z.string().uuid(),
    studentId: z.string().min(1),
    startedAt: z.string().datetime(),
    endedAt: z.string().datetime().nullable(),
    avgDynamics: z.number().nullable().optional(),
    avgTiming: z.number().nullable().optional(),
    avgPedaling: z.number().nullable().optional(),
    avgArticulation: z.number().nullable().optional(),
    avgPhrasing: z.number().nullable().optional(),
    avgInterpretation: z.number().nullable().optional(),
    observationsJson: z.unknown().nullable().optional(),
    chunksSummaryJson: z.unknown().nullable().optional(),
    conversationId: z.string().nullable().optional(),
    accumulatorJson: z.unknown().nullable().optional(),
    needsSynthesis: z.boolean(),
  })
  .refine(
    (s) => s.endedAt === null || Date.parse(s.endedAt) >= Date.parse(s.startedAt),
    { message: "endedAt must be >= startedAt" },
  );
```

File: `apps/api/src/harness/entities/session.ts`

---

### Exercise

A practice exercise. UUID at row level; deduplication on insert uses `exerciseDedupKey({ title, source })`.

Canonical key: `id` (UUID); dedup key: `title.trim().toLowerCase() + '|' + source.trim().toLowerCase()`

```typescript
// apps/api/src/harness/entities/exercise.ts
export const ExerciseSchema = z.object({
  id: z.string().uuid(),
  title: z.string().min(1),
  description: z.string(),
  instructions: z.string(),
  difficulty: z.string(),
  category: z.string(),
  repertoireTags: z.unknown().nullable().optional(),
  notationContent: z.string().nullable().optional(),
  notationFormat: z.string().nullable().optional(),
  midiContent: z.string().nullable().optional(),
  source: z.string().min(1),
  variantsJson: z.unknown().nullable().optional(),
  createdAt: z.string().datetime(),
});
```

File: `apps/api/src/harness/entities/exercise.ts`

## Identity Resolution Rules

| Entity | Rule | Notes |
|---|---|---|
| Student | `apple_user_id` is canonical | Sign in with Apple guarantees uniqueness; no merge needed |
| Piece | `pieceIdFromCatalogue({ composer, catalogueType, opusNumber, pieceNumber })` returns the canonical `piece_id`; collision = same piece | Implemented in `apps/api/src/wasm/piece-identify/`; V2 documents the contract |
| Movement | `{ piece_id, movement_index }` collision = same movement | Composite-key match; no row lookup |
| Bar | `{ piece_id, movement_index, bar_number }` collision = same bar | Composite-key match; no row lookup |
| Session | UUID; no resolution | Server-issued at session start |
| Exercise | UUID at row level; dedup-on-insert by `{ title, source }` lowercased-trimmed | `exerciseDedupKey()` in `apps/api/src/harness/entities/exercise.ts` exposes the contract; DB unique constraint is named future work |

**Piece resolution detail.** `pieceIdFromCatalogue` constructs `composer.catalogue_op_opusNumber.pieceNumber` after lowercasing and trimming both inputs. The piece-identify pipeline in `apps/api/src/wasm/piece-identify/` calls this function after DTW + N-gram fingerprint matching; two pieces that produce the same key are the same piece regardless of how the caller named them.

**Exercise dedup contract.** `exerciseDedupKey({ title, source })` returns `title.trim().toLowerCase() + '|' + source.trim().toLowerCase()`. Callers must check this key before insert; rows with the same key are the same exercise. A DB unique constraint on this computed column is named future work (see Migration Path).

## Fact Layer Schema

```typescript
// apps/api/src/harness/facts/fact.ts
export const ASSERTION_TYPE = [
  "recurring_issue",
  "recent_breakthrough",
  "student_reported",
  "piece_status",
  "baseline_shift",
] as const;

export const factSchema = z
  .object({
    id: z.string().uuid(),
    studentId: z.string().min(1),
    factText: z.string().min(1),
    assertionType: z.enum(ASSERTION_TYPE),
    dimension: z.enum(SIX_DIM).nullable().optional(),
    validAt: z.string().datetime(),
    invalidAt: z.string().datetime().nullable(),
    entityMentions: z.array(entityRefSchema).min(1),
    evidence: z.array(evidenceRefSchema).min(1),
    trend: z
      .enum(["improving", "stable", "declining", "new"])
      .nullable()
      .optional(),
    confidence: z.enum(["high", "medium", "low"]),
    sourceType: z.enum(["synthesized", "student_reported", "inferred"]),
    createdAt: z.string().datetime(),
    expiredAt: z.string().datetime().nullable(),
  })
  .refine(
    (f) =>
      f.invalidAt === null ||
      Date.parse(f.invalidAt) >= Date.parse(f.validAt),
    { message: "invalidAt must be >= validAt", path: ["invalidAt"] },
  )
  .refine(
    (f) =>
      f.expiredAt === null ||
      Date.parse(f.expiredAt) >= Date.parse(f.createdAt),
    { message: "expiredAt must be >= createdAt", path: ["expiredAt"] },
  );
```

File: `apps/api/src/harness/facts/fact.ts`

**Invariants enforced by Zod refinements** (each tested by one invalid fixture):

- `entityMentions.length >= 1` — every fact must mention at least one entity; a free-floating claim with no entity anchor is not a valid fact.
- `evidence.length >= 1` — every fact must have at least one piece of evidence; unsupported claims cannot enter the Fact layer.
- `invalidAt === null || invalidAt >= validAt` — a fact's world-clock end cannot precede its world-clock start.
- `expiredAt === null || expiredAt >= createdAt` — a fact's system-clock supersession cannot precede its system-clock creation.

**`ASSERTION_TYPE` enum values:**

| Value | Meaning |
|---|---|
| `recurring_issue` | A pattern the student exhibits repeatedly across sessions (e.g., over-pedaling in slow movements) |
| `recent_breakthrough` | A positive shift observed in recent sessions (e.g., dynamics range improved markedly) |
| `student_reported` | A claim the student stated directly in conversation (e.g., "I want to work on Chopin this week") |
| `piece_status` | The student's current relationship to a specific piece (e.g., learning, polishing, shelved) |
| `baseline_shift` | A statistically significant change in one of the six baseline dimensions |

## Evidence Chain Example

Scenario: the synthesis layer has determined that a student recurrently over-pedals in slow movements. The resulting Fact's `evidence[]` array traces back to three concrete Layer-1 items:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440001",
  "studentId": "apple.user.abc123",
  "factText": "Student recurrently over-pedals in slow movements, particularly in the A-flat major section.",
  "assertionType": "recurring_issue",
  "dimension": "pedaling",
  "validAt": "2026-04-20T14:00:00Z",
  "invalidAt": null,
  "entityMentions": [
    { "kind": "student", "studentId": "apple.user.abc123" },
    { "kind": "movement", "pieceId": "chopin.op_op_28.15", "movementIndex": 0 }
  ],
  "evidence": [
    {
      "kind": "observation",
      "observation_id": "550e8400-e29b-41d4-a716-446655440002"
    },
    {
      "kind": "signal",
      "chunk_id": "chunk_session7_seg3",
      "schema_name": "MuQQuality",
      "row_id": "550e8400-e29b-41d4-a716-446655440003"
    },
    {
      "kind": "signal",
      "chunk_id": "chunk_session7_seg3",
      "schema_name": "StopMoment",
      "row_id": "550e8400-e29b-41d4-a716-446655440004"
    }
  ],
  "trend": "stable",
  "confidence": "high",
  "sourceType": "synthesized",
  "createdAt": "2026-04-26T09:00:00Z",
  "expiredAt": null
}
```

**How each evidence link traces back to a chunk:**

- `kind: 'observation'` — `observation_id` resolves to a row in the `observations` table. That row carries a `session_id` and the chunk offset where the observation was generated. The pipeline LLM emitted this observation after seeing the MuQ and STOP signals from this chunk.
- `kind: 'signal', schema_name: 'MuQQuality'` — `chunk_id` is the primary lookup key for the `chunk_results` table (or DO state). `row_id` disambiguates when multiple MuQ results exist for the same chunk (e.g., different producer versions). The MuQ 6-dim vector shows depressed `pedaling` score for this chunk.
- `kind: 'signal', schema_name: 'StopMoment'` — same `chunk_id`; `row_id` points to the STOP classifier result. The STOP probability exceeded the threshold with `dimension: 'pedaling'`, which triggered the observation above and corroborates the MuQ signal.

All three evidence items share the same `chunk_id`, meaning the full evidence chain for this Fact is anchored to a single 15-second audio chunk from session 7.

## Migration Path

Three additive migrations are named here as future work. V2 writes none of them. `EvidenceRef` and `EntityRef` shapes do not change when these migrations land — resolution helpers absorb the storage shift.

### 1. `fact_entity_mentions` join table

**What:** Extract `synthesized_facts.entities` (currently unindexed `jsonb`) into a join table with rows `(fact_id, entity_kind, entity_key)`.

**Trigger:** When "find all facts mentioning piece X" exceeds ~50ms via jsonb scan. At low fact counts, the jsonb scan is fast enough. As longitudinal memory grows across hundreds of sessions, the join table becomes necessary for the V7 agent loop to do efficient entity-scoped fact retrieval.

**Additive nature:** The `EvidenceRef` union and `entityRefSchema` shapes are unchanged. The `fact_entity_mentions` table is populated by a migration script reading the existing `entities` jsonb column; consumers switch to a JOIN query instead of an in-process jsonb parse. No schema-level change to `factSchema`.

**Which V2 type absorbs the shift:** `EntityRef` discriminated union (`apps/api/src/harness/entities/index.ts`). The join table rows use the same `(kind, key)` structure as `EntityRef`; the resolution helper switches from jsonb parse to a parameterized query.

### 2. `fact_evidence` join table

**What:** Extract `synthesized_facts.evidence` (currently a `TEXT` JSON array of observation IDs only) into a join table with rows `(fact_id, evidence_kind, evidence_id, signal_schema_name?)`.

**Trigger:** When signal-typed evidence enters Facts — i.e., when the V6 agent loop begins writing Facts whose `evidence[]` includes `kind: 'signal'` entries. Today only observation IDs appear; once the V6 loop emits signal-backed facts, indexed lookup by `evidence_kind` and `signal_schema_name` becomes necessary.

**Additive nature:** `evidenceRefSchema` and `EvidenceRef` are unchanged. The join table is populated by a migration script; resolution helpers switch to a parameterized JOIN. No schema-level change to `factSchema`.

**Which V2 type absorbs the shift:** `EvidenceRef` discriminated union (`apps/api/src/harness/content/index.ts`). The join table columns map 1:1 to the three `EvidenceRef` variants.

### 3. `signals` table (optional)

**What:** A unified `signals` table for non-chunk-derived signals — e.g., session-level aggregates the V6 agent loop emits.

**Trigger:** When a skill emits a signal that has no natural home in `chunk_results`. Today all signals are chunk-derived and live in `chunk_results` / DO state / R2. If the V6 agent loop begins emitting session-level or cross-session signals (e.g., a trend signal computed over the last 10 sessions), those have no existing storage location.

**Additive nature:** `SignalSchema` and `EvidenceRef.kind = 'signal'` are unchanged. The `signals` table adds a new storage backend; the resolution helper for `kind: 'signal'` gains a fallback lookup path: try `chunk_results` first, then `signals`. No schema-level change to `factSchema` or `evidenceRefSchema`.

**Which V2 type absorbs the shift:** `SignalSchema` discriminated union (`apps/api/src/harness/content/signal.ts`). New signal schema names added by V6 extend the existing enum; existing `EvidenceRef` pointers using those names route to the new table.

## Related

- `docs/harness.md` — anchor doc; V2 named at line 152
- `docs/apps/03-memory-system.md` — Two Clocks + Three Layers preamble (cites this doc)
- `docs/specs/2026-04-25-v2-canonical-entity-schema-design.md` — V2 design spec
- `docs/specs/2026-04-25-v5-three-tier-skill-decomposition-design.md` — V5 spec; V5's three artifact schemas slot into Layer 1 via `schema_name`
- `apps/api/src/harness/entities/` — entity Zod schemas
- `apps/api/src/harness/content/` — Layer-1 schemas + EvidenceRef
- `apps/api/src/harness/facts/` — Fact Layer-3 schema
