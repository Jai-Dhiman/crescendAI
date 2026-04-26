# V2 Canonical Entity Schema Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Define the three-layer (Content / Entity / Fact) typed schema that makes every reference in the harness — to a student, piece, session, exercise, signal, observation, or claim — collapse to a canonical row before any agent reasons about it.

**Spec:** docs/specs/2026-04-25-v2-canonical-entity-schema-design.md

**Style:** Follow `apps/api/TS_STYLE.md`. Zod 3, Vitest 3, `@cloudflare/vitest-pool-workers`. No emojis. Explicit exception handling. No backwards-compat shims. Test commands run from `apps/api/`.

## Task Groups

```
Group A (parallel — leaf entity schemas):       Tasks 1, 2, 3, 4
Group B (depends on A — entity barrel):         Task 5
Group C (parallel — leaf content schemas):      Tasks 6, 7, 8
Group D (depends on C — content barrel):        Task 9
Group E (depends on B and D — fact schema):     Task 10
Group F (depends on E — reference doc):         Task 11
Group G (depends on F — memory-system edit):    Task 12
```

A task in Group A and a task in Group C can also run concurrently — they touch disjoint subdirs. Tasks within Group A do not touch overlapping files; same for Group C. Tasks 5, 9, 10, 11, 12 are each their own group (single task).

---

## Task 1: Student entity schema

**Group:** A (parallel with Tasks 2, 3, 4)

**Behavior being verified:** A Student row mirroring `apps/api/src/db/schema/students.ts` parses; a missing `studentId` fails parse; `resolveStudent` canonicalizes `appleUserId` to `studentId`.

**Interface under test:** `StudentSchema.safeParse(input)`, `resolveStudent({ appleUserId })`.

**Files:**
- Create: `apps/api/src/harness/entities/student.ts`
- Test: `apps/api/src/harness/entities/student.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/entities/student.test.ts
import { describe, expect, test } from "vitest";
import { StudentSchema, resolveStudent } from "./student";

describe("StudentSchema", () => {
	test("parses a valid Student row", () => {
		const result = StudentSchema.safeParse({
			studentId: "apple:user:abc123",
			inferredLevel: "intermediate",
			baselineDynamics: 0.6,
			baselineTiming: 0.55,
			baselinePedaling: 0.5,
			baselineArticulation: 0.62,
			baselinePhrasing: 0.58,
			baselineInterpretation: 0.6,
			baselineSessionCount: 12,
			explicitGoals: null,
			createdAt: "2026-04-01T00:00:00.000Z",
			updatedAt: "2026-04-25T00:00:00.000Z",
		});
		expect(result.success).toBe(true);
	});

	test("fails parse when studentId is missing", () => {
		const result = StudentSchema.safeParse({
			inferredLevel: "intermediate",
			baselineSessionCount: 0,
			createdAt: "2026-04-01T00:00:00.000Z",
			updatedAt: "2026-04-25T00:00:00.000Z",
		});
		expect(result.success).toBe(false);
	});
});

describe("resolveStudent", () => {
	test("canonicalizes appleUserId to studentId", () => {
		expect(resolveStudent({ appleUserId: "apple:user:abc123" })).toEqual({
			studentId: "apple:user:abc123",
		});
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun x vitest run src/harness/entities/student.test.ts
```
Expected: FAIL — `Cannot find module './student'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/entities/student.ts
import { z } from "zod";

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
	createdAt: z.string(),
	updatedAt: z.string(),
});

export type Student = z.infer<typeof StudentSchema>;

export function resolveStudent(input: { appleUserId: string }): {
	studentId: string;
} {
	return { studentId: input.appleUserId };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun x vitest run src/harness/entities/student.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/entities/student.ts apps/api/src/harness/entities/student.test.ts && git commit -m "feat(harness): V2 Student entity Zod schema + resolver"
```

---

## Task 2: Piece entity + Movement/Bar types

**Group:** A (parallel with Tasks 1, 3, 4)

**Behavior being verified:** A Piece row mirroring `apps/api/src/db/schema/catalog.ts` parses; a Piece missing `barCount` fails; valid `MovementRef` and `BarRef` parse; `pieceIdFromCatalogue` returns `composer.catalogueType_opusNumber.pieceNumber` for a known input.

**Interface under test:** `PieceSchema`, `MovementRefSchema`, `BarRefSchema`, `pieceIdFromCatalogue`.

**Files:**
- Create: `apps/api/src/harness/entities/piece.ts`
- Test: `apps/api/src/harness/entities/piece.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/entities/piece.test.ts
import { describe, expect, test } from "vitest";
import {
	BarRefSchema,
	MovementRefSchema,
	PieceSchema,
	pieceIdFromCatalogue,
} from "./piece";

describe("PieceSchema", () => {
	test("parses a valid Piece row", () => {
		const result = PieceSchema.safeParse({
			pieceId: "chopin.etudes_op_25.1",
			composer: "Chopin",
			title: "Etude Op. 25 No. 1",
			keySignature: "Ab major",
			timeSignature: "4/4",
			tempoBpm: 104,
			barCount: 49,
			durationSeconds: 145,
			noteCount: 1240,
			pitchRangeLow: 36,
			pitchRangeHigh: 96,
			hasTimeSigChanges: false,
			hasTempoChanges: true,
			source: "asap",
			opusNumber: 25,
			pieceNumber: 1,
			catalogueType: "etudes",
			createdAt: "2026-04-01T00:00:00.000Z",
		});
		expect(result.success).toBe(true);
	});

	test("fails parse when barCount is missing", () => {
		const result = PieceSchema.safeParse({
			pieceId: "chopin.etudes_op_25.1",
			composer: "Chopin",
			title: "Etude Op. 25 No. 1",
			noteCount: 1240,
			hasTimeSigChanges: false,
			hasTempoChanges: true,
			source: "asap",
			createdAt: "2026-04-01T00:00:00.000Z",
		});
		expect(result.success).toBe(false);
	});
});

describe("MovementRefSchema", () => {
	test("parses a valid MovementRef", () => {
		const result = MovementRefSchema.safeParse({
			pieceId: "beethoven.sonatas_op_27.2",
			movementIndex: 0,
		});
		expect(result.success).toBe(true);
	});
});

describe("BarRefSchema", () => {
	test("parses a valid BarRef", () => {
		const result = BarRefSchema.safeParse({
			pieceId: "chopin.etudes_op_25.1",
			movementIndex: 0,
			barNumber: 47,
		});
		expect(result.success).toBe(true);
	});
});

describe("pieceIdFromCatalogue", () => {
	test("returns composer.catalogueType_opusNumber.pieceNumber for a known input", () => {
		const id = pieceIdFromCatalogue({
			composer: "Chopin",
			catalogueType: "etudes",
			opusNumber: 25,
			pieceNumber: 1,
		});
		expect(id).toBe("chopin.etudes_op_25.1");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun x vitest run src/harness/entities/piece.test.ts
```
Expected: FAIL — `Cannot find module './piece'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/entities/piece.ts
import { z } from "zod";

export const PieceSchema = z.object({
	pieceId: z.string().min(1),
	composer: z.string().min(1),
	title: z.string().min(1),
	keySignature: z.string().nullable().optional(),
	timeSignature: z.string().nullable().optional(),
	tempoBpm: z.number().int().nullable().optional(),
	barCount: z.number().int().nonnegative(),
	durationSeconds: z.number().nullable().optional(),
	noteCount: z.number().int().nonnegative(),
	pitchRangeLow: z.number().int().nullable().optional(),
	pitchRangeHigh: z.number().int().nullable().optional(),
	hasTimeSigChanges: z.boolean(),
	hasTempoChanges: z.boolean(),
	source: z.string(),
	opusNumber: z.number().int().nullable().optional(),
	pieceNumber: z.number().int().nullable().optional(),
	catalogueType: z.string().nullable().optional(),
	createdAt: z.string(),
});

export type Piece = z.infer<typeof PieceSchema>;

export const MovementRefSchema = z.object({
	pieceId: z.string().min(1),
	movementIndex: z.number().int().nonnegative(),
});

export type MovementRef = z.infer<typeof MovementRefSchema>;

export const BarRefSchema = z.object({
	pieceId: z.string().min(1),
	movementIndex: z.number().int().nonnegative(),
	barNumber: z.number().int().nonnegative(),
});

export type BarRef = z.infer<typeof BarRefSchema>;

export function pieceIdFromCatalogue(input: {
	composer: string;
	catalogueType: string;
	opusNumber: number;
	pieceNumber: number;
}): string {
	const composer = input.composer.trim().toLowerCase();
	const catalogue = input.catalogueType.trim().toLowerCase();
	return `${composer}.${catalogue}_op_${input.opusNumber}.${input.pieceNumber}`;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun x vitest run src/harness/entities/piece.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/entities/piece.ts apps/api/src/harness/entities/piece.test.ts && git commit -m "feat(harness): V2 Piece entity + MovementRef + BarRef + pieceIdFromCatalogue"
```

---

## Task 3: Session entity schema

**Group:** A (parallel with Tasks 1, 2, 4)

**Behavior being verified:** A Session row mirroring `apps/api/src/db/schema/sessions.ts` parses; a Session with `endedAt < startedAt` fails parse.

**Interface under test:** `SessionSchema.safeParse(input)`.

**Files:**
- Create: `apps/api/src/harness/entities/session.ts`
- Test: `apps/api/src/harness/entities/session.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/entities/session.test.ts
import { describe, expect, test } from "vitest";
import { SessionSchema } from "./session";

describe("SessionSchema", () => {
	test("parses a valid Session row", () => {
		const result = SessionSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			studentId: "apple:user:abc123",
			startedAt: "2026-04-25T10:00:00.000Z",
			endedAt: "2026-04-25T10:30:00.000Z",
			avgDynamics: 0.6,
			avgTiming: 0.55,
			avgPedaling: 0.5,
			avgArticulation: 0.62,
			avgPhrasing: 0.58,
			avgInterpretation: 0.6,
			observationsJson: null,
			chunksSummaryJson: null,
			conversationId: null,
			accumulatorJson: null,
			needsSynthesis: false,
		});
		expect(result.success).toBe(true);
	});

	test("parses a valid in-progress Session with endedAt null", () => {
		const result = SessionSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			studentId: "apple:user:abc123",
			startedAt: "2026-04-25T10:00:00.000Z",
			endedAt: null,
			needsSynthesis: false,
		});
		expect(result.success).toBe(true);
	});

	test("fails parse when endedAt is before startedAt", () => {
		const result = SessionSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			studentId: "apple:user:abc123",
			startedAt: "2026-04-25T10:30:00.000Z",
			endedAt: "2026-04-25T10:00:00.000Z",
			needsSynthesis: false,
		});
		expect(result.success).toBe(false);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun x vitest run src/harness/entities/session.test.ts
```
Expected: FAIL — `Cannot find module './session'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/entities/session.ts
import { z } from "zod";

export const SessionSchema = z
	.object({
		id: z.string().uuid(),
		studentId: z.string().min(1),
		startedAt: z.string(),
		endedAt: z.string().nullable(),
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

export type Session = z.infer<typeof SessionSchema>;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun x vitest run src/harness/entities/session.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/entities/session.ts apps/api/src/harness/entities/session.test.ts && git commit -m "feat(harness): V2 Session entity Zod schema with bi-temporal refinement"
```

---

## Task 4: Exercise entity schema + dedup key

**Group:** A (parallel with Tasks 1, 2, 3)

**Behavior being verified:** An Exercise row mirroring `apps/api/src/db/schema/exercises.ts` parses; an Exercise with empty `title` fails; `exerciseDedupKey` returns the same string for whitespace/case variants of the same `{title, source}`.

**Interface under test:** `ExerciseSchema.safeParse(input)`, `exerciseDedupKey({ title, source })`.

**Files:**
- Create: `apps/api/src/harness/entities/exercise.ts`
- Test: `apps/api/src/harness/entities/exercise.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/entities/exercise.test.ts
import { describe, expect, test } from "vitest";
import { ExerciseSchema, exerciseDedupKey } from "./exercise";

describe("ExerciseSchema", () => {
	test("parses a valid Exercise row", () => {
		const result = ExerciseSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			title: "Slow-hands voicing on bar 12-16",
			description: "Right-hand melody isolation",
			instructions: "Play right hand only at half tempo, exaggerate top voice",
			difficulty: "intermediate",
			category: "voicing",
			repertoireTags: null,
			notationContent: null,
			notationFormat: null,
			midiContent: null,
			source: "molecule:voicing-diagnosis",
			variantsJson: null,
			createdAt: "2026-04-01T00:00:00.000Z",
		});
		expect(result.success).toBe(true);
	});

	test("fails parse when title is empty", () => {
		const result = ExerciseSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			title: "",
			description: "desc",
			instructions: "ins",
			difficulty: "intermediate",
			category: "voicing",
			source: "molecule:voicing-diagnosis",
			createdAt: "2026-04-01T00:00:00.000Z",
		});
		expect(result.success).toBe(false);
	});
});

describe("exerciseDedupKey", () => {
	test("returns same key for whitespace and case variants", () => {
		const a = exerciseDedupKey({
			title: "Slow-Hands Voicing",
			source: "Molecule:Voicing-Diagnosis",
		});
		const b = exerciseDedupKey({
			title: "  slow-hands voicing  ",
			source: "molecule:voicing-diagnosis",
		});
		expect(a).toBe(b);
	});

	test("returns different keys for different titles", () => {
		const a = exerciseDedupKey({ title: "A", source: "x" });
		const b = exerciseDedupKey({ title: "B", source: "x" });
		expect(a).not.toBe(b);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun x vitest run src/harness/entities/exercise.test.ts
```
Expected: FAIL — `Cannot find module './exercise'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/entities/exercise.ts
import { z } from "zod";

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
	createdAt: z.string(),
});

export type Exercise = z.infer<typeof ExerciseSchema>;

export function exerciseDedupKey(input: {
	title: string;
	source: string;
}): string {
	const t = input.title.trim().toLowerCase();
	const s = input.source.trim().toLowerCase();
	return `${t}|${s}`;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun x vitest run src/harness/entities/exercise.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/entities/exercise.ts apps/api/src/harness/entities/exercise.test.ts && git commit -m "feat(harness): V2 Exercise entity Zod schema + dedup key"
```

---

## Task 5: Entities barrel + EntityRef discriminated union

**Group:** B (sequential, depends on Group A — imports types from Tasks 1-4)

**Behavior being verified:** Each of the six EntityRef variants (`student | piece | movement | bar | session | exercise`) parses; an unknown `kind` fails; the `entityRefSchemas` registry has exactly the six expected keys.

**Interface under test:** `entityRefSchema.safeParse(input)`, `entityRefSchemas` registry shape.

**Files:**
- Create: `apps/api/src/harness/entities/index.ts`
- Test: `apps/api/src/harness/entities/index.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/entities/index.test.ts
import { describe, expect, test } from "vitest";
import { entityRefSchema, entityRefSchemas } from "./index";

describe("entityRefSchema", () => {
	test.each([
		["student", { kind: "student", studentId: "apple:user:abc" }],
		["piece", { kind: "piece", pieceId: "chopin.etudes_op_25.1" }],
		[
			"movement",
			{ kind: "movement", pieceId: "chopin.etudes_op_25.1", movementIndex: 0 },
		],
		[
			"bar",
			{
				kind: "bar",
				pieceId: "chopin.etudes_op_25.1",
				movementIndex: 0,
				barNumber: 47,
			},
		],
		[
			"session",
			{ kind: "session", sessionId: "11111111-2222-3333-4444-555555555555" },
		],
		[
			"exercise",
			{ kind: "exercise", exerciseId: "11111111-2222-3333-4444-555555555555" },
		],
	])("parses a valid EntityRef of kind %s", (_label, ref) => {
		const result = entityRefSchema.safeParse(ref);
		expect(result.success).toBe(true);
	});

	test("fails parse for an unknown kind", () => {
		const result = entityRefSchema.safeParse({ kind: "alien", x: 1 });
		expect(result.success).toBe(false);
	});
});

describe("entityRefSchemas registry", () => {
	test("has exactly the six expected keys", () => {
		expect(Object.keys(entityRefSchemas).sort()).toEqual([
			"bar",
			"exercise",
			"movement",
			"piece",
			"session",
			"student",
		]);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun x vitest run src/harness/entities/index.test.ts
```
Expected: FAIL — `Cannot find module './index'` or missing exports.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/entities/index.ts
import { z } from "zod";

export * from "./student";
export * from "./piece";
export * from "./session";
export * from "./exercise";

const studentRef = z.object({
	kind: z.literal("student"),
	studentId: z.string().min(1),
});

const pieceRef = z.object({
	kind: z.literal("piece"),
	pieceId: z.string().min(1),
});

const movementRef = z.object({
	kind: z.literal("movement"),
	pieceId: z.string().min(1),
	movementIndex: z.number().int().nonnegative(),
});

const barRef = z.object({
	kind: z.literal("bar"),
	pieceId: z.string().min(1),
	movementIndex: z.number().int().nonnegative(),
	barNumber: z.number().int().nonnegative(),
});

const sessionRef = z.object({
	kind: z.literal("session"),
	sessionId: z.string().uuid(),
});

const exerciseRef = z.object({
	kind: z.literal("exercise"),
	exerciseId: z.string().uuid(),
});

export const entityRefSchema = z.discriminatedUnion("kind", [
	studentRef,
	pieceRef,
	movementRef,
	barRef,
	sessionRef,
	exerciseRef,
]);

export type EntityRef = z.infer<typeof entityRefSchema>;
export type EntityKind = EntityRef["kind"];

export const entityRefSchemas = {
	student: studentRef,
	piece: pieceRef,
	movement: movementRef,
	bar: barRef,
	session: sessionRef,
	exercise: exerciseRef,
} as const;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun x vitest run src/harness/entities/index.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/entities/index.ts apps/api/src/harness/entities/index.test.ts && git commit -m "feat(harness): V2 EntityRef union + entities barrel + entityRefSchemas registry"
```

---

## Task 6: Signal Layer-1 schema (4-variant discriminated union)

**Group:** C (parallel with Tasks 7, 8)

**Behavior being verified:** Each of the four signal variants (`MuQQuality | AMTTranscription | StopMoment | ScoreAlignment`) parses with valid payload; an unknown `schema_name` fails; a MuQQuality payload missing one of the six dimensions fails.

**Interface under test:** `SignalSchema.safeParse(input)`.

**Files:**
- Create: `apps/api/src/harness/content/signal.ts`
- Test: `apps/api/src/harness/content/signal.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/content/signal.test.ts
import { describe, expect, test } from "vitest";
import { SignalSchema } from "./signal";

const baseHeaders = {
	chunk_id: "chunk-abc",
	producer: "muq-handler",
	producer_version: "muq.v2.1",
	created_at: "2026-04-25T10:00:00.000Z",
};

describe("SignalSchema", () => {
	test("parses a valid MuQQuality signal", () => {
		const result = SignalSchema.safeParse({
			...baseHeaders,
			schema_name: "MuQQuality",
			payload: {
				dynamics: 0.6,
				timing: 0.55,
				pedaling: 0.5,
				articulation: 0.62,
				phrasing: 0.58,
				interpretation: 0.6,
			},
		});
		expect(result.success).toBe(true);
	});

	test("fails when MuQQuality payload is missing a dimension", () => {
		const result = SignalSchema.safeParse({
			...baseHeaders,
			schema_name: "MuQQuality",
			payload: {
				dynamics: 0.6,
				timing: 0.55,
				pedaling: 0.5,
				articulation: 0.62,
				phrasing: 0.58,
			},
		});
		expect(result.success).toBe(false);
	});

	test("parses a valid AMTTranscription signal", () => {
		const result = SignalSchema.safeParse({
			...baseHeaders,
			schema_name: "AMTTranscription",
			payload: {
				midi_notes: [
					{ pitch: 60, onset_ms: 0, offset_ms: 500, velocity: 80 },
				],
				pedals: [{ onset_ms: 0, offset_ms: 1200, type: "sustain" }],
			},
		});
		expect(result.success).toBe(true);
	});

	test("parses a valid StopMoment signal", () => {
		const result = SignalSchema.safeParse({
			...baseHeaders,
			schema_name: "StopMoment",
			payload: {
				probability: 0.82,
				dimension: "pedaling",
				bar_range: { start: 12, end: 16 },
			},
		});
		expect(result.success).toBe(true);
	});

	test("parses a valid ScoreAlignment signal", () => {
		const result = SignalSchema.safeParse({
			...baseHeaders,
			schema_name: "ScoreAlignment",
			payload: {
				alignments: [
					{ chunk_offset_ms: 100, score_offset_ms: 95, confidence: 0.9 },
				],
			},
		});
		expect(result.success).toBe(true);
	});

	test("fails for an unknown schema_name", () => {
		const result = SignalSchema.safeParse({
			...baseHeaders,
			schema_name: "Unknown",
			payload: {},
		});
		expect(result.success).toBe(false);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun x vitest run src/harness/content/signal.test.ts
```
Expected: FAIL — `Cannot find module './signal'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/content/signal.ts
import { z } from "zod";

const SIX_DIM = [
	"dynamics",
	"timing",
	"pedaling",
	"articulation",
	"phrasing",
	"interpretation",
] as const;

export const SignalSchemaName = {
	MuQQuality: "MuQQuality",
	AMTTranscription: "AMTTranscription",
	StopMoment: "StopMoment",
	ScoreAlignment: "ScoreAlignment",
} as const;

export type SignalSchemaName =
	(typeof SignalSchemaName)[keyof typeof SignalSchemaName];

const headers = z.object({
	chunk_id: z.string().min(1),
	producer: z.string().min(1),
	producer_version: z.string().min(1),
	created_at: z.string(),
});

const muqPayload = z.object({
	dynamics: z.number(),
	timing: z.number(),
	pedaling: z.number(),
	articulation: z.number(),
	phrasing: z.number(),
	interpretation: z.number(),
});

const amtPayload = z.object({
	midi_notes: z.array(
		z.object({
			pitch: z.number().int(),
			onset_ms: z.number(),
			offset_ms: z.number(),
			velocity: z.number().int(),
		}),
	),
	pedals: z.array(
		z.object({
			onset_ms: z.number(),
			offset_ms: z.number(),
			type: z.string(),
		}),
	),
});

const stopPayload = z.object({
	probability: z.number().min(0).max(1),
	dimension: z.enum(SIX_DIM),
	bar_range: z
		.object({ start: z.number().int(), end: z.number().int() })
		.optional(),
});

const scoreAlignPayload = z.object({
	alignments: z.array(
		z.object({
			chunk_offset_ms: z.number(),
			score_offset_ms: z.number(),
			confidence: z.number(),
		}),
	),
});

const muq = headers.extend({
	schema_name: z.literal("MuQQuality"),
	payload: muqPayload,
});

const amt = headers.extend({
	schema_name: z.literal("AMTTranscription"),
	payload: amtPayload,
});

const stop = headers.extend({
	schema_name: z.literal("StopMoment"),
	payload: stopPayload,
});

const align = headers.extend({
	schema_name: z.literal("ScoreAlignment"),
	payload: scoreAlignPayload,
});

export const SignalSchema = z.discriminatedUnion("schema_name", [
	muq,
	amt,
	stop,
	align,
]);

export type Signal = z.infer<typeof SignalSchema>;

export const signalSchemas = {
	MuQQuality: muq,
	AMTTranscription: amt,
	StopMoment: stop,
	ScoreAlignment: align,
} as const;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun x vitest run src/harness/content/signal.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/content/signal.ts apps/api/src/harness/content/signal.test.ts && git commit -m "feat(harness): V2 Signal Layer-1 4-variant discriminated union"
```

---

## Task 7: Observation Layer-1 schema

**Group:** C (parallel with Tasks 6, 8)

**Behavior being verified:** An Observation row mirroring `apps/api/src/db/schema/observations.ts` parses; an Observation with `dimension: 'unknown'` fails parse.

**Interface under test:** `ObservationSchema.safeParse(input)`.

**Files:**
- Create: `apps/api/src/harness/content/observation.ts`
- Test: `apps/api/src/harness/content/observation.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/content/observation.test.ts
import { describe, expect, test } from "vitest";
import { ObservationSchema } from "./observation";

describe("ObservationSchema", () => {
	test("parses a valid Observation row", () => {
		const result = ObservationSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			studentId: "apple:user:abc",
			sessionId: "22222222-3333-4444-5555-666666666666",
			chunkIndex: 4,
			dimension: "pedaling",
			observationText: "Pedal held a bit long into the next harmony at bar 14.",
			elaborationText: null,
			reasoningTrace: null,
			framing: "correction",
			dimensionScore: 0.42,
			studentBaseline: 0.55,
			pieceContext: null,
			learningArc: "mid-learning",
			isFallback: false,
			createdAt: "2026-04-25T10:05:00.000Z",
			messageId: null,
			conversationId: null,
		});
		expect(result.success).toBe(true);
	});

	test("fails parse when dimension is not one of the 6 known", () => {
		const result = ObservationSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			studentId: "apple:user:abc",
			sessionId: "22222222-3333-4444-5555-666666666666",
			dimension: "unknown",
			observationText: "x",
			isFallback: false,
			createdAt: "2026-04-25T10:05:00.000Z",
		});
		expect(result.success).toBe(false);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun x vitest run src/harness/content/observation.test.ts
```
Expected: FAIL — `Cannot find module './observation'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/content/observation.ts
import { z } from "zod";

const SIX_DIM = [
	"dynamics",
	"timing",
	"pedaling",
	"articulation",
	"phrasing",
	"interpretation",
] as const;

const FRAMING = [
	"correction",
	"recognition",
	"encouragement",
	"question",
] as const;

export const ObservationSchema = z.object({
	id: z.string().uuid(),
	studentId: z.string().min(1),
	sessionId: z.string().uuid(),
	chunkIndex: z.number().int().nullable().optional(),
	dimension: z.enum(SIX_DIM),
	observationText: z.string().min(1),
	elaborationText: z.string().nullable().optional(),
	reasoningTrace: z.string().nullable().optional(),
	framing: z.enum(FRAMING).nullable().optional(),
	dimensionScore: z.number().nullable().optional(),
	studentBaseline: z.number().nullable().optional(),
	pieceContext: z.string().nullable().optional(),
	learningArc: z.string().nullable().optional(),
	isFallback: z.boolean(),
	createdAt: z.string(),
	messageId: z.string().nullable().optional(),
	conversationId: z.string().nullable().optional(),
});

export type Observation = z.infer<typeof ObservationSchema>;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun x vitest run src/harness/content/observation.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/content/observation.ts apps/api/src/harness/content/observation.test.ts && git commit -m "feat(harness): V2 Observation Layer-1 schema"
```

---

## Task 8: Artifact Layer-1 base schema

**Group:** C (parallel with Tasks 6, 7)

**Behavior being verified:** An ArtifactRow with a `schema_name` and `payload` parses; an ArtifactRow missing `schema_name` fails parse. V2 does NOT validate payload contents — that is V5's responsibility.

**Interface under test:** `ArtifactRowSchema.safeParse(input)`.

**Files:**
- Create: `apps/api/src/harness/content/artifact.ts`
- Test: `apps/api/src/harness/content/artifact.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/content/artifact.test.ts
import { describe, expect, test } from "vitest";
import { ArtifactRowSchema } from "./artifact";

describe("ArtifactRowSchema", () => {
	test("parses a valid ArtifactRow with opaque payload", () => {
		const result = ArtifactRowSchema.safeParse({
			artifact_id: "11111111-2222-3333-4444-555555555555",
			schema_name: "DiagnosisArtifact",
			schema_version: 1,
			producer: "molecule:voicing-diagnosis",
			created_at: "2026-04-25T10:05:00.000Z",
			payload: { anything: "v5 owns this" },
		});
		expect(result.success).toBe(true);
	});

	test("fails parse when schema_name is missing", () => {
		const result = ArtifactRowSchema.safeParse({
			artifact_id: "11111111-2222-3333-4444-555555555555",
			schema_version: 1,
			producer: "molecule:voicing-diagnosis",
			created_at: "2026-04-25T10:05:00.000Z",
			payload: {},
		});
		expect(result.success).toBe(false);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun x vitest run src/harness/content/artifact.test.ts
```
Expected: FAIL — `Cannot find module './artifact'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/content/artifact.ts
import { z } from "zod";

export const ArtifactRowSchema = z.object({
	artifact_id: z.string().uuid(),
	schema_name: z.string().min(1),
	schema_version: z.number().int().positive(),
	producer: z.string().min(1),
	created_at: z.string(),
	payload: z.unknown(),
});

export type ArtifactRow = z.infer<typeof ArtifactRowSchema>;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun x vitest run src/harness/content/artifact.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/content/artifact.ts apps/api/src/harness/content/artifact.test.ts && git commit -m "feat(harness): V2 ArtifactRow Layer-1 base schema (V5 narrows payload)"
```

---

## Task 9: Content barrel + EvidenceRef discriminated union + contentSchemas registry

**Group:** D (sequential, depends on Group C — imports Signal/Observation/Artifact schemas)

**Behavior being verified:** Each of the three EvidenceRef variants (`signal | observation | artifact`) parses; an unknown `kind` fails; the `contentSchemas` registry has exactly the four signal-name keys plus `Observation` and `ArtifactRow`.

**Interface under test:** `evidenceRefSchema.safeParse(input)`, `contentSchemas` registry shape.

**Files:**
- Create: `apps/api/src/harness/content/index.ts`
- Test: `apps/api/src/harness/content/index.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/content/index.test.ts
import { describe, expect, test } from "vitest";
import { contentSchemas, evidenceRefSchema } from "./index";

describe("evidenceRefSchema", () => {
	test.each([
		[
			"signal",
			{
				kind: "signal",
				chunk_id: "chunk-abc",
				schema_name: "MuQQuality",
				row_id: "row-1",
			},
		],
		[
			"observation",
			{
				kind: "observation",
				observation_id: "11111111-2222-3333-4444-555555555555",
			},
		],
		[
			"artifact",
			{
				kind: "artifact",
				artifact_id: "11111111-2222-3333-4444-555555555555",
				schema_name: "DiagnosisArtifact",
			},
		],
	])("parses a valid EvidenceRef of kind %s", (_label, ref) => {
		const result = evidenceRefSchema.safeParse(ref);
		expect(result.success).toBe(true);
	});

	test("fails parse for an unknown kind", () => {
		const result = evidenceRefSchema.safeParse({ kind: "alien", x: 1 });
		expect(result.success).toBe(false);
	});
});

describe("contentSchemas registry", () => {
	test("has exactly the six expected keys", () => {
		expect(Object.keys(contentSchemas).sort()).toEqual([
			"AMTTranscription",
			"ArtifactRow",
			"MuQQuality",
			"Observation",
			"ScoreAlignment",
			"StopMoment",
		]);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun x vitest run src/harness/content/index.test.ts
```
Expected: FAIL — `Cannot find module './index'` or missing exports.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/content/index.ts
import { z } from "zod";
import { ArtifactRowSchema } from "./artifact";
import { ObservationSchema } from "./observation";
import { SignalSchemaName, signalSchemas } from "./signal";

export * from "./signal";
export * from "./observation";
export * from "./artifact";

const signalRef = z.object({
	kind: z.literal("signal"),
	chunk_id: z.string().min(1),
	schema_name: z.enum([
		SignalSchemaName.MuQQuality,
		SignalSchemaName.AMTTranscription,
		SignalSchemaName.StopMoment,
		SignalSchemaName.ScoreAlignment,
	]),
	row_id: z.string().min(1),
});

const observationRef = z.object({
	kind: z.literal("observation"),
	observation_id: z.string().uuid(),
});

const artifactRef = z.object({
	kind: z.literal("artifact"),
	artifact_id: z.string().uuid(),
	schema_name: z.string().min(1),
});

export const evidenceRefSchema = z.discriminatedUnion("kind", [
	signalRef,
	observationRef,
	artifactRef,
]);

export type EvidenceRef = z.infer<typeof evidenceRefSchema>;

export const contentSchemas = {
	...signalSchemas,
	Observation: ObservationSchema,
	ArtifactRow: ArtifactRowSchema,
} as const;

export type ContentSchemaName = keyof typeof contentSchemas;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun x vitest run src/harness/content/index.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/content/index.ts apps/api/src/harness/content/index.test.ts && git commit -m "feat(harness): V2 EvidenceRef union + content barrel + contentSchemas registry"
```

---

## Task 10: Fact Layer-3 schema + facts barrel

**Group:** E (sequential, depends on Groups B and D — imports `entityRefSchema` and `evidenceRefSchema`)

**Behavior being verified:** A valid Fact parses; four invalid Facts each fail one refinement (entityMentions empty; evidence empty; `invalidAt < validAt`; `expiredAt < createdAt`).

**Interface under test:** `factSchema.safeParse(input)`.

**Files:**
- Create: `apps/api/src/harness/facts/fact.ts`
- Create: `apps/api/src/harness/facts/index.ts`
- Test: `apps/api/src/harness/facts/fact.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/facts/fact.test.ts
import { describe, expect, test } from "vitest";
import { factSchema } from "./fact";

const validFact = {
	id: "11111111-2222-3333-4444-555555555555",
	studentId: "apple:user:abc",
	factText: "Student recurrently over-pedals in slow movements.",
	assertionType: "recurring_issue",
	dimension: "pedaling",
	validAt: "2026-04-01T00:00:00.000Z",
	invalidAt: null,
	entityMentions: [{ kind: "student", studentId: "apple:user:abc" }],
	evidence: [
		{
			kind: "observation",
			observation_id: "22222222-3333-4444-5555-666666666666",
		},
	],
	trend: "stable",
	confidence: "high",
	sourceType: "synthesized",
	createdAt: "2026-04-15T00:00:00.000Z",
	expiredAt: null,
};

describe("factSchema", () => {
	test("parses a valid Fact", () => {
		const result = factSchema.safeParse(validFact);
		expect(result.success).toBe(true);
	});

	test("fails when entityMentions is empty", () => {
		const result = factSchema.safeParse({ ...validFact, entityMentions: [] });
		expect(result.success).toBe(false);
	});

	test("fails when evidence is empty", () => {
		const result = factSchema.safeParse({ ...validFact, evidence: [] });
		expect(result.success).toBe(false);
	});

	test("fails when invalidAt is before validAt", () => {
		const result = factSchema.safeParse({
			...validFact,
			invalidAt: "2026-03-01T00:00:00.000Z",
		});
		expect(result.success).toBe(false);
	});

	test("fails when expiredAt is before createdAt", () => {
		const result = factSchema.safeParse({
			...validFact,
			expiredAt: "2026-03-01T00:00:00.000Z",
		});
		expect(result.success).toBe(false);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun x vitest run src/harness/facts/fact.test.ts
```
Expected: FAIL — `Cannot find module './fact'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/facts/fact.ts
import { z } from "zod";
import { entityRefSchema } from "../entities";
import { evidenceRefSchema } from "../content";

const SIX_DIM = [
	"dynamics",
	"timing",
	"pedaling",
	"articulation",
	"phrasing",
	"interpretation",
] as const;

export const ASSERTION_TYPE = [
	"recurring_issue",
	"recent_breakthrough",
	"student_reported",
	"piece_status",
	"baseline_shift",
] as const;

export type AssertionType = (typeof ASSERTION_TYPE)[number];

export const factSchema = z
	.object({
		id: z.string().uuid(),
		studentId: z.string().min(1),
		factText: z.string().min(1),
		assertionType: z.enum(ASSERTION_TYPE),
		dimension: z.enum(SIX_DIM).nullable().optional(),
		validAt: z.string(),
		invalidAt: z.string().nullable(),
		entityMentions: z.array(entityRefSchema).min(1),
		evidence: z.array(evidenceRefSchema).min(1),
		trend: z
			.enum(["improving", "stable", "declining", "new"])
			.nullable()
			.optional(),
		confidence: z.enum(["high", "medium", "low"]),
		sourceType: z.enum(["synthesized", "student_reported", "inferred"]),
		createdAt: z.string(),
		expiredAt: z.string().nullable(),
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

export type Fact = z.infer<typeof factSchema>;
```

```typescript
// apps/api/src/harness/facts/index.ts
export * from "./fact";
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun x vitest run src/harness/facts/fact.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/facts/fact.ts apps/api/src/harness/facts/index.ts apps/api/src/harness/facts/fact.test.ts && git commit -m "feat(harness): V2 Fact Layer-3 schema with bi-temporal refinements"
```

---

## Task 11: Reference doc — `docs/harness/entities.md`

**Group:** F (sequential, depends on Group E — cites all V2 schemas)

**Behavior being verified:** This is a doc task. The verification is structural: the file exists at the named path, contains the six required H2 sections from the brainstorm, and cites every schema file shipped in Tasks 1-10.

**Interface under test:** Markdown structure (no runtime test).

**Files:**
- Create: `docs/harness/entities.md`

- [ ] **Step 1: Write the doc**

Create `docs/harness/entities.md` with the following six H2 sections, in order. Each section's content comes directly from the spec (`docs/specs/2026-04-25-v2-canonical-entity-schema-design.md`); the doc is the user-facing reference, the spec is the design rationale.

Required sections:

1. `## Three-Layer Diagram` — copy the ASCII diagram from the spec's "Three layers" subsection. Add a one-paragraph preamble citing the Mahler wiki *Context Graphs* page.
2. `## Canonical Entity Types` — the six Layer-2 entity Zod schemas. For each entity, include: a one-line description, the canonical key, a TypeScript code block (paste the actual exported Zod schema from `apps/api/src/harness/entities/<file>.ts`), and the file path under `apps/api/src/harness/entities/`.
3. `## Identity Resolution Rules` — the table from the spec's "Identity resolution rules" subsection. For Piece, link to `apps/api/src/wasm/piece-identify/`. For Exercise, document the `exerciseDedupKey` contract.
4. `## Fact Layer Schema` — paste the `factSchema` Zod definition from `apps/api/src/harness/facts/fact.ts`. Document each of the four refinements as a bulleted invariant. List the five `assertionType` enum values with one-line descriptions each.
5. `## Evidence Chain Example` — write a worked example of the fact "student recurrently over-pedals in slow movements" with three concrete `evidence[]` entries: one `kind: 'observation'` pointing at a UUID, one `kind: 'signal'` with `schema_name: 'MuQQuality'`, one `kind: 'signal'` with `schema_name: 'StopMoment'`. Show how each link traces back to a chunk_id.
6. `## Migration Path` — the three named-but-not-implemented additive migrations from the spec: `fact_entity_mentions` join table; `fact_evidence` join table; optional `signals` table. For each, list (a) the trigger condition, (b) the additive nature (existing schemas unchanged), (c) which V2 type absorbs the storage shift.

The doc opens with a status line:

```markdown
> **Status (2026-04-26):** V2 spec landed. Six entity schemas, ContentRow base, EvidenceRef union, and Fact schema implemented in `apps/api/src/harness/`. Three additive migrations named below as future work.
```

The doc closes with a "Related" section:

```markdown
## Related

- `docs/harness.md` — anchor doc; V2 named at line 152
- `docs/apps/03-memory-system.md` — Two Clocks + Three Layers preamble (cites this doc)
- `docs/specs/2026-04-25-v2-canonical-entity-schema-design.md` — V2 design spec
- `docs/specs/2026-04-25-v5-three-tier-skill-decomposition-design.md` — V5 spec; V5's three artifact schemas slot into Layer 1 via `schema_name`
- `apps/api/src/harness/entities/` — entity Zod schemas
- `apps/api/src/harness/content/` — Layer-1 schemas + EvidenceRef
- `apps/api/src/harness/facts/` — Fact Layer-3 schema
```

- [ ] **Step 2: Verify the file structurally**

```bash
test -f docs/harness/entities.md && \
  grep -c "^## Three-Layer Diagram" docs/harness/entities.md && \
  grep -c "^## Canonical Entity Types" docs/harness/entities.md && \
  grep -c "^## Identity Resolution Rules" docs/harness/entities.md && \
  grep -c "^## Fact Layer Schema" docs/harness/entities.md && \
  grep -c "^## Evidence Chain Example" docs/harness/entities.md && \
  grep -c "^## Migration Path" docs/harness/entities.md
```
Expected: each `grep -c` returns `1`. If any returns `0`, that section is missing — add it.

- [ ] **Step 3: Commit**

```bash
git add docs/harness/entities.md && git commit -m "docs(harness): V2 canonical entity schema reference"
```

---

## Task 12: Tightening edit to `docs/apps/03-memory-system.md`

**Group:** G (sequential, depends on Group F — cites entities.md)

**Behavior being verified:** The "Mapping to Existing Tables" sub-table (currently lines 78-85) is replaced by a one-line citation; the `(V2 deliverable)` parenthetical on line 85 becomes a link. The "Two Clocks" subsection (lines 14-46) and the rest of the file are unchanged.

**Interface under test:** File diff (no runtime test).

**Files:**
- Modify: `docs/apps/03-memory-system.md`

- [ ] **Step 1: Make the edits**

Two surgical changes via the Edit tool:

**Edit 1** — Replace the "Mapping to Existing Tables" sub-block. Find this block (currently lines 77-85):

```markdown
### Mapping to Existing Tables

| Layer | Current Tables | Status |
|---|---|---|
| Content | `chunks`, `chunk_results`, signal emissions from HF | Exists but evidence linkage is implicit |
| Entity | `students`, `pieces`, `sessions`, `exercises` | Exists for students/pieces/sessions; `bars`, `movements` not yet first-class |
| Fact | `observations` (episode capture), `synthesized_facts` (pattern capture) | Exists; evidence chains stored in `reasoning_trace` JSON but not indexed |

The V2 harness work formalizes the evidence chain as a queryable column, not a blob, and adds `bars` / `movements` as first-class entities. See `docs/harness/entities.md` (V2 deliverable).
```

Replace with:

```markdown
### Mapping to Existing Tables

See [`docs/harness/entities.md`](../harness/entities.md) for the full layer mapping, six canonical entity Zod schemas, identity-resolution rules, the `EvidenceRef` discriminated union, the `Fact` Layer-3 schema with bi-temporal refinements, and the named additive migrations for `fact_entity_mentions`, `fact_evidence`, and a future unified `signals` table.
```

**Edit 2** — None (the `(V2 deliverable)` parenthetical was inside the block deleted by Edit 1; the new one-line citation is the link).

- [ ] **Step 2: Verify the edits**

```bash
grep -c "fact_entity_mentions" docs/apps/03-memory-system.md
grep -c "Mapping to Existing Tables" docs/apps/03-memory-system.md
test ! "$(grep -c "V2 deliverable" docs/apps/03-memory-system.md)" = "1" || (echo "V2 deliverable parenthetical still present — Edit 1 did not land cleanly" && exit 1)
```
Expected: first `grep -c` returns `1` (citation landed); second `grep -c` returns `1` (heading kept); third command does not error.

- [ ] **Step 3: Commit**

```bash
git add docs/apps/03-memory-system.md && git commit -m "docs(memory): cite docs/harness/entities.md for Three Layers mapping"
```

---

## Task summary

| # | Task | Group | New files | Modified files |
|---|---|---|---|---|
| 1 | Student schema | A | 2 | 0 |
| 2 | Piece + Movement + Bar | A | 2 | 0 |
| 3 | Session schema | A | 2 | 0 |
| 4 | Exercise schema + dedup | A | 2 | 0 |
| 5 | Entities barrel + EntityRef | B | 2 | 0 |
| 6 | Signal Layer-1 union | C | 2 | 0 |
| 7 | Observation Layer-1 | C | 2 | 0 |
| 8 | ArtifactRow Layer-1 base | C | 2 | 0 |
| 9 | Content barrel + EvidenceRef | D | 2 | 0 |
| 10 | Fact Layer-3 + barrel | E | 3 | 0 |
| 11 | Reference doc | F | 1 | 0 |
| 12 | Memory-system citation | G | 0 | 1 |
| | **Total** | | **22** | **1** |

Each task is one test (or one structural verification for docs) + one implementation + one commit. No task bundles multiple tests before implementation. Every test exercises the Zod schema's public `.safeParse()` interface — no mocks of internal collaborators, no private-method calls, no internal-state assertions.
