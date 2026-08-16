# Pipeline Retarget: Transkun-first -> Mock Scorer -> MPM Marks Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until `/challenge` returns VERDICT: PROCEED.

**Goal:** A local practice session produces baseline-gated marks with
MPM-grounded evidence end-to-end over the WebSocket, with zero MuQ references
on the request path.
**Spec:** `docs/specs/2026-08-15-pipeline-retarget-marks-design.md`
**Style:** Follow `apps/api/TS_STYLE.md` (tabs, `console.log(JSON.stringify({...}))`
structured logging, domain errors from `lib/errors.ts`, no `any`, services take
`db`/`env` directly where the existing file group already does — `synthesis.ts`'s
`loadBaselinesFromDb(db, studentId)` is the precedent this plan's `baseline-store.ts`
follows instead of the `ServiceContext` wrapper).

All file paths below are relative to `/Users/jdhiman/Documents/crescendai/.worktrees/issue-162-pipeline-retarget/`
unless stated otherwise. All commands run from `apps/api` unless stated otherwise.

## Shared Types (defined once here, repeated verbatim in tasks that use them)

```typescript
// apps/api/src/services/scoring.ts
export interface DimScores {
	dynamics: number;
	timing: number;
	pedaling: number;
	articulation: number;
	phrasing: number;
	interpretation: number;
}
export function scoreChunk(
	perfNotes: readonly PerfNote[],
	perfPedal: readonly PerfPedalEvent[],
): DimScores;

// apps/api/src/services/mpm-features.ts
export interface MpmFeatureSet {
	windowIndex: number;
	localTempoBpm: number;
	tempoDeviationPct: number;
	velocityMean: number;
	velocityRange: number;
	velocitySlope: number;
	articulationRatio: number;
	pedalOverlapFraction: number;
}
export interface MpmFeatureInput {
	windowIndex: number;
	perfNotes: readonly PerfNote[];
	perfPedal: readonly PerfPedalEvent[];
	bars?: readonly [number, number];
}
export function extractMpmFeatures(input: MpmFeatureInput): MpmFeatureSet[];
export function evidenceFor(set: MpmFeatureSet, dimension: Dimension): string;

// apps/api/src/services/mark-gate.ts
export interface MarkCandidate {
	dimension: Dimension;
	taxonomy: "needs_work" | "strong";
	atSeconds: number;
}
export interface ProposeMarksInput {
	atSeconds: number;
	scores: DimScores;
	baseline: BaselineState | null;
	sessionSamples: Partial<Record<Dimension, readonly number[]>>;
	emittedDimensions: ReadonlySet<Dimension>;
	firePersistence: number;
}
export interface ProposeMarksResult {
	candidates: MarkCandidate[];
}
export function proposeMarks(input: ProposeMarksInput): ProposeMarksResult;

// apps/api/src/services/baseline-store.ts
export function loadBaselineState(
	db: Db,
	studentId: string,
): Promise<BaselineState | null>;
export function saveBaselineState(
	db: Db,
	studentId: string,
	state: BaselineState,
): Promise<void>;
```

`PerfNote`/`PerfPedalEvent` are the existing interfaces exported from
`apps/api/src/services/inference.ts` (unchanged). `Dimension`/`DIMS_6` are the
existing exports from `apps/api/src/lib/dims.ts`. `BaselineState` is the
existing export from `apps/api/src/services/student-baseline.ts`. `Db` is the
existing export from `apps/api/src/lib/types.ts`.

## Task Groups

```
Group 0 (harness, red until Group B is complete): Task 1

Group A1 (parallel — 4 independent new files, first task each):
  Task 2 (scoring.ts), Task 4 (mpm-features.ts), Task 6 (mark-gate.ts),
  Task 8 (baseline-store.ts)
Group A2 (parallel — same 4 files, second task each; sequential AFTER A1
  because each pair touches the same file as its A1 counterpart):
  Task 3 (scoring.ts, depends on Task 2), Task 5 (mpm-features.ts, depends
  on Task 4), Task 7 (mark-gate.ts, depends on Task 6), Task 9
  (baseline-store.ts, depends on Task 8)

Group B (sequential, depends on Group A2 — every task in this group edits
  session-brain.ts and/or session-brain.schema.ts): Task 10, Task 11,
  Task 12, Task 13, Task 14, Task 15, Task 16, Task 17

Group C (sequential, depends on Group B — Task 18 edits inference.ts, which
  every subsequent task's broken test file depends on): Task 18, Task 19,
  Task 20, Task 21, Task 22

Group D (parallel with Group C, depends only on Group B's wire shapes —
  touches only apps/web files, disjoint from Group C's apps/api files):
  Task 23, Task 24 (Task 24 depends on Task 23 within this group — both
  touch different files, `practice-api.ts` vs `usePracticeSession.ts`, but
  Task 24's code references the type Task 23 adds, so run Task 23 first)

Group E (sequential, depends on Group C and Group D): Task 25, Task 26, Task 27
```

No group is `[SHIPS INDEPENDENTLY]`: the spec's canonical success state is one
end-to-end session (mark -> review -> persisted baseline), so no subset is
useful to a real session until Group C's grep gate is clean and Group B's
wiring is live.

---

## Group 0

### Task 1: End-to-end harness — marks, review, and a persisted baseline over one session

**Group:** 0 (red until Group B lands)

**Behavior being verified:** A session that transcribes >=10 notes per chunk,
where one dimension's scores drift far enough from the student's baseline
band, emits at least one `mark` WS event with a timestamp anchor and
MPM-shaped evidence, then emits exactly one `review` event with a non-null
verdict and carry-forward at session end, and the DO issues a
`student_profiles` insert carrying `baselineState`.

**Interface under test:** `SessionBrain.webSocketMessage` (chunk_ready,
end_session) and `SessionBrain.alarm` via `runDurableObjectAlarm`, both public
DO surface, driven the same way `session-brain.concurrency.test.ts` and
`session-brain.canary.test.ts` already do.

**Files:**
- Create: `apps/api/src/do/session-brain.marks.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
/// <reference types="@cloudflare/vitest-pool-workers" />

import {
	env,
	runDurableObjectAlarm,
	runInDurableObject,
} from "cloudflare:test";
import { describe, expect, it, vi } from "vitest";

// Deterministic per-call AMT stub: 8 chunks, 10 notes each (clears
// MIN_NOTES_FOR_COMMENT so nothing lands in a no-comment zone). Velocity is
// steady for chunks 0-4 (~62-66) then drops hard for chunks 5-7 (10), which
// drags the mock scorer's "dynamics" dimension far enough below its
// within-session median band to fire after 3 consecutive out-of-band chunks
// (DEFAULT_BASELINE_CONFIG.firePersistence = 3). Every other dimension holds
// identical IOI/duration/pedal shape across all 8 chunks, so its MAD stays 0
// and its band never opens — "dynamics" is the only dimension that can fire.
const { callIndex, VELOCITIES } = vi.hoisted(() => ({
	callIndex: { current: 0 },
	VELOCITIES: [62, 66, 63, 65, 64, 10, 10, 10],
}));

vi.mock("../services/inference", async (importOriginal) => {
	const actual =
		await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callAmtEndpoint: vi.fn(async () => {
			const velocity = VELOCITIES[callIndex.current] ?? 64;
			callIndex.current++;
			return {
				notes: Array.from({ length: 10 }, (_, i) => ({
					pitch: 60,
					onset: i * 1.0,
					offset: i * 1.0 + 0.5,
					velocity,
				})),
				pedalEvents: [],
			};
		}),
	};
});

const { fakeDb, capturedInserts } = vi.hoisted(() => {
	const capturedInserts: Array<{ table: unknown; values: unknown }> = [];
	function makeFakeDb(): unknown {
		let lastInsertTable: unknown = null;
		const handler: ProxyHandler<object> = {
			get(_target, prop) {
				if (prop === "then") {
					return (resolve: (v: unknown[]) => void) => resolve([]);
				}
				if (prop === "insert") {
					return (table: unknown) => {
						lastInsertTable = table;
						return proxy;
					};
				}
				if (prop === "values") {
					return (values: unknown) => {
						capturedInserts.push({ table: lastInsertTable, values });
						return proxy;
					};
				}
				if (prop === "query") {
					return new Proxy(
						{},
						{ get: () => ({ findFirst: async () => undefined }) },
					);
				}
				return () => proxy;
			},
		};
		const proxy = new Proxy({}, handler);
		return proxy;
	}
	return { fakeDb: makeFakeDb(), capturedInserts };
});

vi.mock("../db/client", () => ({ createDb: () => fakeDb }));

vi.mock("../services/teacher", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/teacher")>();
	return {
		...actual,
		synthesizeV6: async function* () {
			yield {
				type: "artifact" as const,
				value: {
					session_id: "sess-e2e",
					synthesis_scope: "session" as const,
					strengths: [],
					focus_areas: [],
					prescribed_exercise: null,
					dominant_dimension: "dynamics" as const,
					recurring_pattern: null,
					next_session_focus:
						"Keep the dynamic range even across repeated passages.",
					diagnosis_refs: [],
					headline: "H".repeat(320),
					assigned_loops: [],
				},
			};
		},
	};
});

import { studentProfiles } from "../db/schema/students";
import type { SessionBrain } from "./session-brain";
import { createInitialState } from "./session-brain.schema";

declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
		CHUNKS: R2Bucket;
	}
}

const fakeWs = { send() {}, close() {} } as unknown as WebSocket;

describe("pipeline retarget: marks + review end-to-end (#162)", () => {
	it("drives a session from chunk_ready through end_session and observes a mark, one review, and a persisted baseline", async () => {
		for (let i = 0; i < 8; i++) {
			await env.CHUNKS.put(`e2e-${i}`, new Uint8Array([1]));
		}

		const received: unknown[] = [];
		const id = env.SESSION_BRAIN.idFromName("e2e-marks-review");
		const stub = env.SESSION_BRAIN.get(id);

		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-e2e", "stud-e2e", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			seeded.pieceLocked = true;
			seeded.markBaselineLoaded = true; // no persisted baseline: cold-start path
			await state.storage.put("state", seeded);

			const pair = new WebSocketPair();
			const [client, server] = Object.values(pair) as [WebSocket, WebSocket];
			state.acceptWebSocket(server);
			client.accept();
			client.addEventListener("message", (evt: MessageEvent) => {
				received.push(JSON.parse(evt.data as string));
			});

			for (let i = 0; i < 8; i++) {
				await instance.webSocketMessage(
					fakeWs,
					JSON.stringify({ type: "chunk_ready", index: i, r2Key: `e2e-${i}` }),
				);
			}

			await instance.webSocketMessage(
				fakeWs,
				JSON.stringify({ type: "end_session" }),
			);
			await runDurableObjectAlarm(stub);

			await new Promise((resolve) => setTimeout(resolve, 20));
			await state.storage.deleteAlarm();
		});

		const marks = received.filter(
			(m) => (m as { type?: string }).type === "mark",
		) as Array<{ mark: { dimension: string; anchor: { type: string }; evidence: string } }>;
		expect(marks.length).toBeGreaterThanOrEqual(1);
		expect(marks[0]?.mark.dimension).toBe("dynamics");
		expect(marks[0]?.mark.anchor.type).toBe("timestamp");
		expect(marks[0]?.mark.evidence).toMatch(
			/local tempo|velocity range|pedal overlap|articulation ratio|velocity slope|tempo \d+ bpm/,
		);

		const reviews = received.filter(
			(m) => (m as { type?: string }).type === "review",
		);
		expect(reviews.length).toBe(1);
		expect(reviews[0]).toMatchObject({
			type: "review",
			verdict: "H".repeat(320),
			carry_forward: "Keep the dynamic range even across repeated passages.",
		});

		const baselineInsert = capturedInserts.find(
			(c) =>
				c.table === studentProfiles &&
				(c.values as Record<string, unknown>).baselineState !== undefined,
		);
		expect(baselineInsert).toBeDefined();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run session-brain.marks.test.ts
```
Expected: FAIL — `Cannot find module '../services/scoring'` (or `mark-gate`/`mpm-features`/`baseline-store` — whichever resolves first), because none of those modules exist yet and `session-brain.ts` has not been wired to emit `mark`/`review` events.

- [ ] **Step 3: No implementation in this task.** This is the Group 0 harness — it
  stays red through Group A and Group B until Task 17 wires the last piece.

- [ ] **Step 4: N/A until Task 17.**

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.marks.test.ts && git commit -m "test(#162): add failing end-to-end marks+review harness (Group 0)"
```

---

## Group A1/A2 — pure new modules (A1 parallel, A2 parallel-after-A1)

### Task 2: `scoreChunk` — silence returns neutral scores

**Group:** A1 (parallel with Task 4, 6, 8)

**Behavior being verified:** A chunk with no transcribed notes scores every
dimension at the neutral midpoint 0.5, never NaN or out of range.

**Interface under test:** `scoreChunk(perfNotes, perfPedal): DimScores`

**Files:**
- Create: `apps/api/src/services/scoring.ts`
- Test: `apps/api/src/services/scoring.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, expect, it } from "vitest";
import { scoreChunk } from "./scoring";

describe("scoreChunk", () => {
	it("returns neutral 0.5 for every dimension when there are no notes", () => {
		const result = scoreChunk([], []);
		expect(result).toEqual({
			dynamics: 0.5,
			timing: 0.5,
			pedaling: 0.5,
			articulation: 0.5,
			phrasing: 0.5,
			interpretation: 0.5,
		});
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts scoring.test.ts
```
Expected: FAIL — `Cannot find module './scoring'` (file does not exist).

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/services/scoring.ts
import type { PerfNote, PerfPedalEvent } from "./inference";

export interface DimScores {
	dynamics: number;
	timing: number;
	pedaling: number;
	articulation: number;
	phrasing: number;
	interpretation: number;
}

function clamp01(x: number): number {
	if (Number.isNaN(x)) return 0.5;
	return Math.max(0, Math.min(1, x));
}

function mean(values: readonly number[]): number {
	if (values.length === 0) return 0;
	return values.reduce((a, b) => a + b, 0) / values.length;
}

function stdev(values: readonly number[], centre: number): number {
	if (values.length === 0) return 0;
	return Math.sqrt(mean(values.map((v) => (v - centre) ** 2)));
}

/**
 * Deterministic pure-TS mock 6-dim scorer. Behind the exact interface the
 * future MoonBeam container call will implement (issue #162 spec) — callers
 * cannot tell the difference. No network, no randomness, no clock.
 */
export function scoreChunk(
	perfNotes: readonly PerfNote[],
	perfPedal: readonly PerfPedalEvent[],
): DimScores {
	if (perfNotes.length === 0) {
		return {
			dynamics: 0.5,
			timing: 0.5,
			pedaling: 0.5,
			articulation: 0.5,
			phrasing: 0.5,
			interpretation: 0.5,
		};
	}

	const sorted = [...perfNotes].sort((a, b) => a.onset - b.onset);
	const velocities = sorted.map((n) => n.velocity);
	const dynamics = clamp01(mean(velocities) / 127);

	const iois: number[] = [];
	for (let i = 1; i < sorted.length; i++) {
		const prev = sorted[i - 1];
		const cur = sorted[i];
		if (prev && cur) iois.push(cur.onset - prev.onset);
	}
	const ioiMean = mean(iois);
	const ioiStd = stdev(iois, ioiMean);
	const timingCv = ioiMean > 0 ? ioiStd / ioiMean : 0;
	const timing = clamp01(1 - timingCv);

	const pedalOn = perfPedal.filter((p) => p.value >= 64).length;
	const pedaling = clamp01(
		perfPedal.length > 0 ? pedalOn / perfPedal.length : 0.5,
	);

	const articulationRatios: number[] = [];
	for (let i = 1; i < sorted.length; i++) {
		const ioi = iois[i - 1];
		const note = sorted[i - 1];
		if (ioi !== undefined && ioi > 0 && note) {
			articulationRatios.push((note.offset - note.onset) / ioi);
		}
	}
	const articulation = clamp01(
		articulationRatios.length > 0 ? mean(articulationRatios) : 0.5,
	);

	const velocityRange =
		velocities.length > 0 ? Math.max(...velocities) - Math.min(...velocities) : 0;
	const phrasing = clamp01(velocityRange / 127);

	const interpretation = clamp01(
		mean([dynamics, timing, pedaling, articulation, phrasing]),
	);

	return { dynamics, timing, pedaling, articulation, phrasing, interpretation };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts scoring.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/scoring.ts apps/api/src/services/scoring.test.ts && git commit -m "feat(#162): add deterministic mock scorer (silence case)"
```

### Task 3: `scoreChunk` — deterministic, range-bound, and velocity-driven dynamics

**Group:** A2 (parallel with Task 5, 7, 9; depends on Task 2)

**Behavior being verified:** Given real notes, `scoreChunk` is deterministic
across repeated calls, every dimension stays within `[0, 1]`, and `dynamics`
tracks mean velocity directly (higher velocity -> higher dynamics).

**Interface under test:** `scoreChunk(perfNotes, perfPedal): DimScores`

**Files:**
- Modify: `apps/api/src/services/scoring.ts` (no change needed — this task
  only adds coverage; Task 2's implementation already satisfies it)
- Test: `apps/api/src/services/scoring.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
it("is deterministic across repeated calls with the same input", () => {
	const notes = [
		{ pitch: 60, onset: 0, offset: 0.4, velocity: 70 },
		{ pitch: 62, onset: 0.5, offset: 0.9, velocity: 80 },
		{ pitch: 64, onset: 1.0, offset: 1.4, velocity: 60 },
	];
	const pedal = [{ time: 0.2, value: 100 }];
	const first = scoreChunk(notes, pedal);
	const second = scoreChunk(notes, pedal);
	expect(second).toEqual(first);
});

it("keeps every dimension within [0, 1]", () => {
	const notes = [
		{ pitch: 60, onset: 0, offset: 0.05, velocity: 127 },
		{ pitch: 62, onset: 0.01, offset: 5, velocity: 1 },
	];
	const result = scoreChunk(notes, []);
	for (const value of Object.values(result)) {
		expect(value).toBeGreaterThanOrEqual(0);
		expect(value).toBeLessThanOrEqual(1);
	}
});

it("higher mean velocity produces a higher dynamics score", () => {
	const soft = [
		{ pitch: 60, onset: 0, offset: 0.4, velocity: 20 },
		{ pitch: 62, onset: 0.5, offset: 0.9, velocity: 25 },
	];
	const loud = [
		{ pitch: 60, onset: 0, offset: 0.4, velocity: 110 },
		{ pitch: 62, onset: 0.5, offset: 0.9, velocity: 115 },
	];
	expect(scoreChunk(loud, []).dynamics).toBeGreaterThan(
		scoreChunk(soft, []).dynamics,
	);
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts scoring.test.ts
```
Expected: This actually PASSES immediately against Task 2's implementation —
per the plan skill's rule, if a test passes without new implementation code
it must be testing behavior the previous task's code already provides
correctly by design (Task 2's formulas are velocity-mean-based and pure by
construction). Confirm the three assertions above by reading the Task 2
implementation, not by adding new code: `scoreChunk` has no `Math.random()`,
no `Date.now()`, no I/O, and `dynamics = clamp01(mean(velocities) / 127)` is
monotonic in velocity. If any assertion fails, fix `scoring.ts` until it
passes — do not weaken the test.

- [ ] **Step 3: No new implementation required (see Step 2).**

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts scoring.test.ts
```
Expected: PASS (all tests in the file, including Task 2's)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/scoring.test.ts && git commit -m "test(#162): cover scoreChunk determinism, range, and velocity monotonicity"
```

### Task 4: `extractMpmFeatures` — sparse input returns a zeroed feature set

**Group:** A1 (parallel with Task 2, 6, 8)

**Behavior being verified:** Fewer than 2 notes (no interval to measure)
yields exactly one zeroed `MpmFeatureSet` carrying the input `windowIndex`,
never a thrown error or `NaN`.

**Interface under test:** `extractMpmFeatures(input): MpmFeatureSet[]`

**Files:**
- Create: `apps/api/src/services/mpm-features.ts`
- Test: `apps/api/src/services/mpm-features.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, expect, it } from "vitest";
import { extractMpmFeatures } from "./mpm-features";

describe("extractMpmFeatures", () => {
	it("returns one zeroed feature set for fewer than 2 notes", () => {
		const result = extractMpmFeatures({
			windowIndex: 3,
			perfNotes: [{ pitch: 60, onset: 0, offset: 0.5, velocity: 80 }],
			perfPedal: [],
		});
		expect(result).toHaveLength(1);
		expect(result[0]).toEqual({
			windowIndex: 3,
			localTempoBpm: 0,
			tempoDeviationPct: 0,
			velocityMean: 0,
			velocityRange: 0,
			velocitySlope: 0,
			articulationRatio: 0,
			pedalOverlapFraction: 0,
		});
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts mpm-features.test.ts
```
Expected: FAIL — `Cannot find module './mpm-features'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/services/mpm-features.ts
import type { Dimension } from "../lib/dims";
import type { PerfNote, PerfPedalEvent } from "./inference";

export interface MpmFeatureSet {
	windowIndex: number;
	localTempoBpm: number;
	tempoDeviationPct: number;
	velocityMean: number;
	velocityRange: number;
	velocitySlope: number;
	articulationRatio: number;
	pedalOverlapFraction: number;
}

export interface MpmFeatureInput {
	windowIndex: number;
	perfNotes: readonly PerfNote[];
	perfPedal: readonly PerfPedalEvent[];
	bars?: readonly [number, number];
}

function mean(values: readonly number[]): number {
	if (values.length === 0) return 0;
	return values.reduce((a, b) => a + b, 0) / values.length;
}

function zeroSet(windowIndex: number): MpmFeatureSet {
	return {
		windowIndex,
		localTempoBpm: 0,
		tempoDeviationPct: 0,
		velocityMean: 0,
		velocityRange: 0,
		velocitySlope: 0,
		articulationRatio: 0,
		pedalOverlapFraction: 0,
	};
}

/**
 * Deterministic per-window MPM-style features over transcribed notes. The
 * `bars` field is accepted but unused today — the follower issue adds
 * per-bar granularity through this same field without an interface change.
 */
export function extractMpmFeatures(input: MpmFeatureInput): MpmFeatureSet[] {
	const { windowIndex, perfNotes, perfPedal } = input;
	if (perfNotes.length < 2) {
		return [zeroSet(windowIndex)];
	}
	return [zeroSet(windowIndex)]; // extended in Task 5
}

export function evidenceFor(set: MpmFeatureSet, dimension: Dimension): string {
	return `${dimension} placeholder`; // implemented in Task 5
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts mpm-features.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/mpm-features.ts apps/api/src/services/mpm-features.test.ts && git commit -m "feat(#162): add mpm-features module (sparse-input case)"
```

### Task 5: `extractMpmFeatures` + `evidenceFor` — full feature computation and evidence prose

**Group:** A2 (parallel with Task 3, 7, 9; depends on Task 4)

**Behavior being verified:** With >=2 notes, the feature set carries real
tempo/velocity/articulation/pedal numbers computed from note statistics, and
`evidenceFor` renders a deterministic MPM sentence per dimension matching the
pattern the review's mark evidence must contain.

**Interface under test:** `extractMpmFeatures(input): MpmFeatureSet[]`,
`evidenceFor(set, dimension): string`

**Files:**
- Modify: `apps/api/src/services/mpm-features.ts`
- Test: `apps/api/src/services/mpm-features.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { evidenceFor, extractMpmFeatures } from "./mpm-features";

describe("extractMpmFeatures — full computation", () => {
	it("computes local tempo and a positive tempoDeviationPct when the second half speeds up", () => {
		// IOIs: 1.0, 1.0 (first half, slow) then 0.5, 0.5 (second half, fast).
		const perfNotes = [
			{ pitch: 60, onset: 0, offset: 0.4, velocity: 60 },
			{ pitch: 60, onset: 1.0, offset: 1.4, velocity: 60 },
			{ pitch: 60, onset: 2.0, offset: 2.4, velocity: 60 },
			{ pitch: 60, onset: 2.5, offset: 2.9, velocity: 60 },
			{ pitch: 60, onset: 3.0, offset: 3.4, velocity: 60 },
		];
		const [set] = extractMpmFeatures({ windowIndex: 0, perfNotes, perfPedal: [] });
		expect(set?.localTempoBpm).toBeGreaterThan(0);
		expect(set?.tempoDeviationPct).toBeGreaterThan(0); // sped up
	});

	it("computes velocityMean, velocityRange, and pedalOverlapFraction", () => {
		const perfNotes = [
			{ pitch: 60, onset: 0, offset: 0.4, velocity: 40 },
			{ pitch: 62, onset: 0.5, offset: 0.9, velocity: 100 },
		];
		const perfPedal = [
			{ time: 0.1, value: 100 },
			{ time: 0.6, value: 0 },
		];
		const [set] = extractMpmFeatures({ windowIndex: 0, perfNotes, perfPedal });
		expect(set?.velocityMean).toBe(70);
		expect(set?.velocityRange).toBe(60);
		expect(set?.pedalOverlapFraction).toBe(0.5);
	});

	it("evidenceFor(timing) matches the direction and magnitude of tempoDeviationPct", () => {
		const set = {
			windowIndex: 0,
			localTempoBpm: 120,
			tempoDeviationPct: 9.4,
			velocityMean: 70,
			velocityRange: 20,
			velocitySlope: 3,
			articulationRatio: 0.5,
			pedalOverlapFraction: 0.5,
		};
		expect(evidenceFor(set, "timing")).toBe(
			"local tempo up 9%, uneven note lengths",
		);
	});

	it("evidenceFor covers every dimension without throwing", () => {
		const set = {
			windowIndex: 0,
			localTempoBpm: 120,
			tempoDeviationPct: -5,
			velocityMean: 70,
			velocityRange: 20,
			velocitySlope: -3,
			articulationRatio: 0.5,
			pedalOverlapFraction: 0.5,
		};
		for (const dim of [
			"dynamics",
			"timing",
			"pedaling",
			"articulation",
			"phrasing",
			"interpretation",
		] as const) {
			expect(typeof evidenceFor(set, dim)).toBe("string");
			expect(evidenceFor(set, dim).length).toBeGreaterThan(0);
		}
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts mpm-features.test.ts
```
Expected: FAIL — `set?.localTempoBpm` is `0` (Task 4's stub always returns a
zeroed set); `evidenceFor` returns the `"timing placeholder"` string, not the
MPM sentence.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the two placeholder function bodies in `apps/api/src/services/mpm-features.ts`:

```typescript
export function extractMpmFeatures(input: MpmFeatureInput): MpmFeatureSet[] {
	const { windowIndex, perfNotes, perfPedal } = input;
	if (perfNotes.length < 2) {
		return [zeroSet(windowIndex)];
	}

	const sorted = [...perfNotes].sort((a, b) => a.onset - b.onset);
	const iois: number[] = [];
	for (let i = 1; i < sorted.length; i++) {
		const prev = sorted[i - 1];
		const cur = sorted[i];
		if (prev && cur) iois.push(cur.onset - prev.onset);
	}
	const ioiMean = mean(iois);
	const localTempoBpm = ioiMean > 0 ? 60 / ioiMean : 0;

	const mid = Math.floor(iois.length / 2);
	const firstHalf = iois.slice(0, mid);
	const secondHalf = iois.slice(mid);
	const firstBpmMean = mean(firstHalf);
	const secondBpmMean = mean(secondHalf);
	const firstBpm = firstBpmMean > 0 ? 60 / firstBpmMean : 0;
	const secondBpm = secondBpmMean > 0 ? 60 / secondBpmMean : 0;
	const tempoDeviationPct =
		firstBpm > 0 ? ((secondBpm - firstBpm) / firstBpm) * 100 : 0;

	const velocities = sorted.map((n) => n.velocity);
	const velocityMean = mean(velocities);
	const velocityRange = Math.max(...velocities) - Math.min(...velocities);
	const velMid = Math.floor(velocities.length / 2);
	const velocityFirst = mean(velocities.slice(0, velMid));
	const velocitySecond = mean(velocities.slice(velMid));
	const velocitySlope = velocitySecond - velocityFirst;

	const articulationRatios: number[] = [];
	for (let i = 1; i < sorted.length; i++) {
		const ioi = iois[i - 1];
		const note = sorted[i - 1];
		if (ioi !== undefined && ioi > 0 && note) {
			articulationRatios.push((note.offset - note.onset) / ioi);
		}
	}
	const articulationRatio = mean(articulationRatios);

	const pedalOn = perfPedal.filter((p) => p.value >= 64).length;
	const pedalOverlapFraction =
		perfPedal.length > 0 ? pedalOn / perfPedal.length : 0;

	return [
		{
			windowIndex,
			localTempoBpm,
			tempoDeviationPct,
			velocityMean,
			velocityRange,
			velocitySlope,
			articulationRatio,
			pedalOverlapFraction,
		},
	];
}

export function evidenceFor(set: MpmFeatureSet, dimension: Dimension): string {
	switch (dimension) {
		case "timing": {
			const direction = set.tempoDeviationPct >= 0 ? "up" : "down";
			return `local tempo ${direction} ${Math.abs(Math.round(set.tempoDeviationPct))}%, uneven note lengths`;
		}
		case "dynamics":
			return `velocity range ${Math.round(set.velocityRange)}, mean ${Math.round(set.velocityMean)}`;
		case "pedaling":
			return `pedal overlap ${Math.round(set.pedalOverlapFraction * 100)}%`;
		case "articulation":
			return `articulation ratio ${set.articulationRatio.toFixed(2)}`;
		case "phrasing": {
			const direction = set.velocitySlope >= 0 ? "rising" : "falling";
			return `velocity slope ${direction} ${Math.abs(Math.round(set.velocitySlope))}`;
		}
		case "interpretation":
			return `tempo ${Math.round(set.localTempoBpm)} bpm, velocity mean ${Math.round(set.velocityMean)}`;
	}
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts mpm-features.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/mpm-features.ts apps/api/src/services/mpm-features.test.ts && git commit -m "feat(#162): compute real MPM features and evidence prose"
```

### Task 6: `proposeMarks` — cold start with insufficient samples never fires

**Group:** A1 (parallel with Task 2, 4, 8)

**Behavior being verified:** With no persisted baseline and fewer than 3
session samples for a dimension, `proposeMarks` returns no candidates — the
minimum-spread guard, matching `student-baseline.ts`'s
`minSamplesForSpread = 3`.

**Interface under test:** `proposeMarks(input): ProposeMarksResult`

**Files:**
- Create: `apps/api/src/services/mark-gate.ts`
- Test: `apps/api/src/services/mark-gate.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, expect, it } from "vitest";
import { proposeMarks } from "./mark-gate";

describe("proposeMarks — cold start", () => {
	it("fires nothing when a dimension has fewer than 3 session samples", () => {
		const result = proposeMarks({
			atSeconds: 30,
			scores: {
				dynamics: 0.1,
				timing: 0.5,
				pedaling: 0.5,
				articulation: 0.5,
				phrasing: 0.5,
				interpretation: 0.5,
			},
			baseline: null,
			sessionSamples: { dynamics: [0.9, 0.1] },
			emittedDimensions: new Set(),
			firePersistence: 3,
		});
		expect(result.candidates).toEqual([]);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts mark-gate.test.ts
```
Expected: FAIL — `Cannot find module './mark-gate'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/services/mark-gate.ts
import type { Dimension } from "../lib/dims";
import { DIMS_6 } from "../lib/dims";
import type { DimScores } from "./scoring";
import type { BaselineState } from "./student-baseline";

export interface MarkCandidate {
	dimension: Dimension;
	taxonomy: "needs_work" | "strong";
	atSeconds: number;
}

export interface ProposeMarksInput {
	atSeconds: number;
	scores: DimScores;
	baseline: BaselineState | null;
	sessionSamples: Partial<Record<Dimension, readonly number[]>>;
	emittedDimensions: ReadonlySet<Dimension>;
	firePersistence: number;
}

export interface ProposeMarksResult {
	candidates: MarkCandidate[];
}

const MIN_SAMPLES_FOR_COLD_START_SPREAD = 3;
const COLD_START_BAND_MULTIPLE = 1.5;

function median(values: readonly number[]): number {
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 === 0
		? ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2
		: (sorted[mid] ?? 0);
}

function medianAbsoluteDeviation(
	values: readonly number[],
	centre: number,
): number {
	return median(values.map((v) => Math.abs(v - centre)));
}

interface Band {
	center: number;
	halfWidth: number;
}

/**
 * Band for a dimension: the student's persisted BaselineState when
 * initialized, otherwise the within-session median +/- a robust spread —
 * the same cold-start-equals-fallback path as 02-pipeline.md section 7.
 * Returns null when there is not enough evidence to judge (uninitialized
 * baseline AND fewer than 3 session samples, or zero spread).
 */
function bandFor(
	dim: Dimension,
	baseline: BaselineState | null,
	samples: readonly number[],
): Band | null {
	const dimState = baseline?.dimensions[dim];
	if (dimState !== undefined && dimState.initialized) {
		return { center: dimState.longMean, halfWidth: dimState.halfWidth };
	}
	if (samples.length < MIN_SAMPLES_FOR_COLD_START_SPREAD) return null;
	const center = median(samples);
	const spread = medianAbsoluteDeviation(samples, center);
	if (spread <= 0) return null;
	return { center, halfWidth: COLD_START_BAND_MULTIPLE * spread };
}

function trailingOutOfBandCount(
	samples: readonly number[],
	band: Band,
): number {
	let count = 0;
	for (let i = samples.length - 1; i >= 0; i--) {
		const s = samples[i];
		if (s === undefined || Math.abs(s - band.center) <= band.halfWidth) break;
		count++;
	}
	return count;
}

/**
 * Pure per-chunk mark gate. No I/O, no clock. Requires
 * `firePersistence`-consecutive trailing out-of-band samples in a
 * dimension's session history, dedupes against `emittedDimensions`
 * (a dimension fires at most once per session), and yields at most one
 * candidate per call (DIMS_6 order breaks ties).
 */
export function proposeMarks(input: ProposeMarksInput): ProposeMarksResult {
	const candidates: MarkCandidate[] = [];
	for (const dim of DIMS_6) {
		if (candidates.length > 0) break;
		if (input.emittedDimensions.has(dim)) continue;
		const samples = input.sessionSamples[dim] ?? [];
		const band = bandFor(dim, input.baseline, samples);
		if (band === null) continue;
		const streak = trailingOutOfBandCount(samples, band);
		if (streak >= input.firePersistence) {
			const latest = samples[samples.length - 1] ?? input.scores[dim];
			candidates.push({
				dimension: dim,
				taxonomy: latest < band.center ? "needs_work" : "strong",
				atSeconds: input.atSeconds,
			});
		}
	}
	return { candidates };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts mark-gate.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/mark-gate.ts apps/api/src/services/mark-gate.test.ts && git commit -m "feat(#162): add mark-gate cold-start minimum-spread guard"
```

### Task 7: `proposeMarks` — fires after persistence, dedupes, and picks at most one per chunk

**Group:** A2 (parallel with Task 3, 5, 9; depends on Task 6)

**Behavior being verified:** A dimension that stays out of a persisted
baseline's band for `firePersistence` consecutive samples fires exactly one
candidate with the correct taxonomy; a dimension already in
`emittedDimensions` never fires again; two simultaneously-qualifying
dimensions yield only one candidate (DIMS_6 order).

**Interface under test:** `proposeMarks(input): ProposeMarksResult`

**Files:**
- Modify: `apps/api/src/services/mark-gate.ts` (no change needed — Task 6's
  implementation already covers this; this task only adds coverage)
- Test: `apps/api/src/services/mark-gate.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
const BASELINE_DIM = {
	lifecycle: "active" as const,
	longMean: 0.6,
	longSd: 0.05,
	shortMean: 0.6,
	noiseFloor: 0.02,
	noiseFloorMeasured: true,
	halfWidth: 0.05,
	consecutiveOutOfBand: 0,
	consecutiveInBand: 0,
	promoted: false,
	evidenceWeeks: [],
	initialized: true,
	updateCount: 5,
	confidence: "provisional" as const,
};

function baselineWith(dim: string, overrides: Partial<typeof BASELINE_DIM> = {}) {
	const absent = {
		...BASELINE_DIM,
		lifecycle: "absent" as const,
		initialized: false,
	};
	const dims = {
		dynamics: absent,
		timing: absent,
		pedaling: absent,
		articulation: absent,
		phrasing: absent,
		interpretation: absent,
	};
	dims[dim as keyof typeof dims] = { ...BASELINE_DIM, ...overrides };
	return { lastSessionTimestamp: null, dimensions: dims };
}

describe("proposeMarks — persisted baseline", () => {
	it("fires needs_work after 3 consecutive below-band samples", () => {
		const result = proposeMarks({
			atSeconds: 45,
			scores: { dynamics: 0.5, timing: 0.2, pedaling: 0.5, articulation: 0.5, phrasing: 0.5, interpretation: 0.5 },
			baseline: baselineWith("timing"),
			sessionSamples: { timing: [0.6, 0.2, 0.2, 0.2] },
			emittedDimensions: new Set(),
			firePersistence: 3,
		});
		expect(result.candidates).toEqual([
			{ dimension: "timing", taxonomy: "needs_work", atSeconds: 45 },
		]);
	});

	it("fires strong when the latest sample is above the band", () => {
		const result = proposeMarks({
			atSeconds: 45,
			scores: { dynamics: 0.5, timing: 0.5, pedaling: 0.5, articulation: 0.5, phrasing: 0.5, interpretation: 0.5 },
			baseline: baselineWith("dynamics"),
			sessionSamples: { dynamics: [0.6, 0.95, 0.95, 0.95] },
			emittedDimensions: new Set(),
			firePersistence: 3,
		});
		expect(result.candidates).toEqual([
			{ dimension: "dynamics", taxonomy: "strong", atSeconds: 45 },
		]);
	});

	it("never re-fires a dimension already in emittedDimensions", () => {
		const result = proposeMarks({
			atSeconds: 45,
			scores: { dynamics: 0.5, timing: 0.2, pedaling: 0.5, articulation: 0.5, phrasing: 0.5, interpretation: 0.5 },
			baseline: baselineWith("timing"),
			sessionSamples: { timing: [0.2, 0.2, 0.2, 0.2] },
			emittedDimensions: new Set(["timing"]),
			firePersistence: 3,
		});
		expect(result.candidates).toEqual([]);
	});

	it("yields at most one candidate when two dimensions qualify in the same call", () => {
		const dims = {
			...baselineWith("dynamics").dimensions,
			timing: { ...BASELINE_DIM },
		};
		const result = proposeMarks({
			atSeconds: 45,
			scores: { dynamics: 0.1, timing: 0.2, pedaling: 0.5, articulation: 0.5, phrasing: 0.5, interpretation: 0.5 },
			baseline: { lastSessionTimestamp: null, dimensions: dims },
			sessionSamples: {
				dynamics: [0.6, 0.1, 0.1, 0.1],
				timing: [0.6, 0.2, 0.2, 0.2],
			},
			emittedDimensions: new Set(),
			firePersistence: 3,
		});
		expect(result.candidates).toHaveLength(1);
		expect(result.candidates[0]?.dimension).toBe("dynamics"); // DIMS_6 order
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

Confirm first by inspection whether these pass or fail against Task 6's
implementation (the gate logic is already complete). Run:

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts mark-gate.test.ts
```
Expected: PASS immediately, by the same "previous task's pure logic already
covers this" reasoning as Task 3. If any assertion fails (e.g. band math is
off), fix `mark-gate.ts` — do not weaken the test — then re-run.

- [ ] **Step 3: No new implementation required if Step 2 passed; otherwise fix `bandFor`/`trailingOutOfBandCount` in `mark-gate.ts` until all four assertions hold.**

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts mark-gate.test.ts
```
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/mark-gate.test.ts && git commit -m "test(#162): cover mark-gate persistence, dedupe, and one-per-chunk cap"
```

### Task 8: `loadBaselineState` — returns null when no row or no stored state

**Group:** A1 (parallel with Task 2, 4, 6)

**Behavior being verified:** A student with no `student_profiles` row, or a
row whose `baseline_state` column is null, gets `null` back — never a thrown
error.

**Interface under test:** `loadBaselineState(db, studentId): Promise<BaselineState | null>`

**Files:**
- Create: `apps/api/src/services/baseline-store.ts`
- Test: `apps/api/src/services/baseline-store.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, expect, it } from "vitest";
import { loadBaselineState } from "./baseline-store";

describe("loadBaselineState", () => {
	it("returns null when the student has no row", async () => {
		const fakeDb = {
			query: { studentProfiles: { findFirst: async () => undefined } },
		};
		const result = await loadBaselineState(fakeDb as never, "stu-1");
		expect(result).toBeNull();
	});

	it("returns null when the row exists but baseline_state is null", async () => {
		const fakeDb = {
			query: {
				studentProfiles: {
					findFirst: async () => ({ studentId: "stu-1", baselineState: null }),
				},
			},
		};
		const result = await loadBaselineState(fakeDb as never, "stu-1");
		expect(result).toBeNull();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts baseline-store.test.ts
```
Expected: FAIL — `Cannot find module './baseline-store'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

First, add the backing column this module reads/writes. In
`apps/api/src/db/schema/students.ts`, add `baselineState` to the
`studentProfiles` table definition (after `explicitGoals`), matching the
untyped-jsonb convention every other jsonb column in this schema directory
already uses (`componentsJson`, `artifactJson`, `routingJson`):

```typescript
import { integer, jsonb, pgTable, real, text, timestamp } from "drizzle-orm/pg-core";

export const studentProfiles = pgTable("student_profiles", {
	studentId: text("student_id").primaryKey(),
	inferredLevel: text("inferred_level"),
	baselineDynamics: real("baseline_dynamics"),
	baselineTiming: real("baseline_timing"),
	baselinePedaling: real("baseline_pedaling"),
	baselineArticulation: real("baseline_articulation"),
	baselinePhrasing: real("baseline_phrasing"),
	baselineInterpretation: real("baseline_interpretation"),
	baselineSessionCount: integer("baseline_session_count").notNull().default(0),
	explicitGoals: text("explicit_goals"),
	baselineState: jsonb("baseline_state"),
	createdAt: timestamp("created_at", { withTimezone: true })
		.notNull()
		.defaultNow(),
	updatedAt: timestamp("updated_at", { withTimezone: true })
		.notNull()
		.defaultNow(),
});
```

Then generate and apply the migration (per `TS-DB-003`, never hand-write
`db/migrations/` files):

```bash
cd apps/api && bun run generate -- --name=baseline_state
```
Expected: creates `src/db/migrations/000N_baseline_state.sql` (`N` = next
sequential index) containing
`ALTER TABLE "student_profiles" ADD COLUMN "baseline_state" jsonb;`, plus a
matching `meta/000N_snapshot.json` and a new `meta/_journal.json` entry.

Apply it to the local dev database (per `CLAUDE.md`):

```bash
cd apps/api && DATABASE_URL="postgresql://jdhiman:postgres@localhost:5432/crescendai_dev" bun run migrate
```

Now implement the module itself:

```typescript
// apps/api/src/services/baseline-store.ts
import { eq } from "drizzle-orm";
import { studentProfiles } from "../db/schema/students";
import { ValidationError } from "../lib/errors";
import type { Db } from "../lib/types";
import { BaselineStateSchema, type BaselineState } from "./student-baseline";

export async function loadBaselineState(
	db: Db,
	studentId: string,
): Promise<BaselineState | null> {
	const row = await db.query.studentProfiles.findFirst({
		where: (s, { eq: eqFn }) => eqFn(s.studentId, studentId),
	});
	if (!row || row.baselineState === null || row.baselineState === undefined) {
		return null;
	}
	const parsed = BaselineStateSchema.safeParse(row.baselineState);
	if (!parsed.success) {
		throw new ValidationError(
			`loadBaselineState: corrupt baseline_state for student ${studentId}: ${parsed.error.message}`,
		);
	}
	return parsed.data;
}

export async function saveBaselineState(
	db: Db,
	studentId: string,
	state: BaselineState,
): Promise<void> {
	await db
		.insert(studentProfiles)
		.values({ studentId, baselineState: state })
		.onConflictDoUpdate({
			target: studentProfiles.studentId,
			set: { baselineState: state, updatedAt: new Date() },
		});
}
```

Note: `eq` is imported for parity with the rest of the file group's style but
unused here since the `where` callback receives its own `eq`; remove the
top-level `eq` import if `bun run lint` flags it as unused (it will — Biome
treats an unused named import as an error). Final import line:

```typescript
import { studentProfiles } from "../db/schema/students";
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts baseline-store.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/baseline-store.ts apps/api/src/services/baseline-store.test.ts && git commit -m "feat(#162): add baseline-store load path (null cases)"
```

### Task 9: `saveBaselineState` — writes baseline_state; corrupt row throws loudly on load

**Group:** A2 (parallel with Task 3, 5, 7; depends on Task 8)

**Behavior being verified:** `saveBaselineState` issues an
insert-with-upsert carrying `baselineState`, and `loadBaselineState` throws
`ValidationError` (never silently returns a wrong shape) when the stored
value fails `BaselineStateSchema`.

**Interface under test:** `saveBaselineState(db, studentId, state): Promise<void>`,
`loadBaselineState(db, studentId): Promise<BaselineState | null>`

**Files:**
- Modify: `apps/api/src/services/baseline-store.ts` (no change needed —
  Task 8's implementation already covers this)
- Test: `apps/api/src/services/baseline-store.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { initialBaselineState } from "./student-baseline";
import { saveBaselineState } from "./baseline-store";

it("saveBaselineState inserts studentId and baselineState via onConflictDoUpdate", async () => {
	const state = initialBaselineState();
	const calls: { values?: unknown; conflictSet?: unknown } = {};
	const fakeDb = {
		insert: () => ({
			values: (v: unknown) => {
				calls.values = v;
				return {
					onConflictDoUpdate: (opts: { set: unknown }) => {
						calls.conflictSet = opts.set;
						return Promise.resolve();
					},
				};
			},
		}),
	};
	await saveBaselineState(fakeDb as never, "stu-1", state);
	expect(calls.values).toMatchObject({ studentId: "stu-1", baselineState: state });
	expect(calls.conflictSet).toMatchObject({ baselineState: state });
});

it("loadBaselineState throws ValidationError when the stored shape is corrupt", async () => {
	const fakeDb = {
		query: {
			studentProfiles: {
				findFirst: async () => ({
					studentId: "stu-1",
					baselineState: { not: "a baseline state" },
				}),
			},
		},
	};
	await expect(loadBaselineState(fakeDb as never, "stu-1")).rejects.toThrow(
		/corrupt baseline_state/,
	);
});
```

- [ ] **Step 2: Run test — verify it FAILS**

Run and confirm by inspection: Task 8's implementation already implements
both functions correctly, so this is expected to PASS immediately (same
"pure logic already covers this" pattern as Task 3/Task 7).

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts baseline-store.test.ts
```
Expected: PASS. If it fails, fix `baseline-store.ts` (not the test) until it
passes.

- [ ] **Step 3: No new implementation required if Step 2 passed.**

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts baseline-store.test.ts
```
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/baseline-store.test.ts && git commit -m "test(#162): cover baseline-store save path and corrupt-row rejection"
```

---

## Group B — session-brain wiring (sequential; each task depends on the previous)

### Task 10: `session-brain.schema.ts` — new state fields and outgoing mark/review schemas

**Group:** B (depends on Group A2; must land before Task 12-17)

**Behavior being verified:** `createInitialState` seeds the five new fields
(`markBaseline`, `markBaselineLoaded`, `sessionSamples`, `emittedMarks`,
`noCommentZones`) with their documented defaults, and `wsOutgoingMessageSchema`
accepts a valid `mark` payload and a valid `review` payload.

**Interface under test:** `createInitialState(...)`, `wsOutgoingMessageSchema.parse(...)`

**Files:**
- Modify: `apps/api/src/do/session-brain.schema.ts`
- Test: `apps/api/src/do/session-brain.schema.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// append to apps/api/src/do/session-brain.schema.test.ts
import { createInitialState, wsOutgoingMessageSchema } from "./session-brain.schema";

describe("createInitialState — pipeline retarget fields (#162)", () => {
	it("seeds markBaseline null, markBaselineLoaded false, empty samples/marks/zones", () => {
		const state = createInitialState("sess", "stud", null);
		expect(state.markBaseline).toBeNull();
		expect(state.markBaselineLoaded).toBe(false);
		expect(state.sessionSamples).toEqual({});
		expect(state.emittedMarks).toEqual([]);
		expect(state.noCommentZones).toEqual([]);
	});
});

describe("wsOutgoingMessageSchema — mark and review (#162)", () => {
	it("accepts a valid mark payload", () => {
		const msg = {
			type: "mark",
			mark: {
				id: "sess-dynamics-7",
				anchor: { type: "timestamp", atSeconds: 105 },
				taxonomy: "needs_work",
				dimension: "dynamics",
				evidence: "velocity range 60, mean 40",
				lifecycle: "active",
			},
		};
		expect(() => wsOutgoingMessageSchema.parse(msg)).not.toThrow();
	});

	it("accepts a valid review payload with null verdict on synthesis failure", () => {
		const msg = {
			type: "review",
			verdict: null,
			carry_forward: null,
			no_comment_zones: [{ atSeconds: 30, reason: "insufficient_notes" }],
		};
		expect(() => wsOutgoingMessageSchema.parse(msg)).not.toThrow();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts session-brain.schema.test.ts
```
Expected: FAIL — `state.markBaseline` is `undefined` (property does not
exist on the current `SessionState`), and the `mark`/`review` schemas do not
exist so `wsOutgoingMessageSchema.parse` throws
`Invalid discriminator value`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/do/session-brain.schema.ts`, change the top import and add
the new fields:

```typescript
import { z } from "zod";
import { DIMS_6 } from "../lib/dims";
import { BaselineStateSchema } from "../services/student-baseline";
```

Add to `sessionStateSchema` (after `isEvalSession: z.boolean().default(false),`):

```typescript
	markBaseline: BaselineStateSchema.nullable().default(null),
	markBaselineLoaded: z.boolean().default(false),
	sessionSamples: z.record(z.string(), z.array(z.number())).default({}),
	emittedMarks: z.array(z.string()).default([]),
	noCommentZones: z
		.array(z.object({ chunkIndex: z.number().int(), reason: z.string() }))
		.default([]),
```

Add to `createInitialState`'s returned object (after `isEvalSession: false,`):

```typescript
		markBaseline: null,
		markBaselineLoaded: false,
		sessionSamples: {},
		emittedMarks: [],
		noCommentZones: [],
```

Add the outgoing schemas after `wsChunkBarMapSchema`:

```typescript
export const wsMarkSchema = z.object({
	type: z.literal("mark"),
	mark: z.object({
		id: z.string(),
		anchor: z.discriminatedUnion("type", [
			z.object({
				type: z.literal("bars"),
				bars: z.tuple([z.number().int(), z.number().int()]),
				atSeconds: z.number(),
			}),
			z.object({ type: z.literal("timestamp"), atSeconds: z.number() }),
		]),
		taxonomy: z.enum(["needs_work", "missed_opportunity", "strong"]),
		dimension: z.enum(DIMS_6),
		evidence: z.string(),
		lifecycle: z.enum(["active", "improving", "resolved"]),
		confidence: z
			.enum(["exploratory", "provisional", "established"])
			.optional(),
	}),
});

export const wsReviewSchema = z.object({
	type: z.literal("review"),
	verdict: z.string().nullable(),
	carry_forward: z.string().nullable(),
	no_comment_zones: z.array(
		z.object({ atSeconds: z.number(), reason: z.string() }),
	),
});
```

Change the outgoing union:

```typescript
export const wsOutgoingMessageSchema = z.discriminatedUnion("type", [
	wsChunkBarMapSchema,
	wsMarkSchema,
	wsReviewSchema,
]);
```

Add the corresponding exported types after `WsChunkBarMap`:

```typescript
export type WsMark = z.infer<typeof wsMarkSchema>;
export type WsReview = z.infer<typeof wsReviewSchema>;
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts session-brain.schema.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.schema.ts apps/api/src/do/session-brain.schema.test.ts && git commit -m "feat(#162): add mark/review session state fields and outgoing schemas"
```

### Task 11: Remove `MUQ_ENDPOINT`; rename `MUQ_CHUNK_MS` -> `CHUNK_MS`

**Group:** B (depends on Task 10; must land before Task 12)

**Behavior being verified:** `Bindings` no longer declares `MUQ_ENDPOINT`,
`wrangler.toml` no longer sets it, and `computeSessionDurationMs` still
returns `15000 * n` under its new constant name — a pure rename, verified by
the existing `computeSessionDurationMs` test continuing to pass.

**Interface under test:** `computeSessionDurationMs(scoredChunkCount): number`
(exported, unchanged signature)

**Files:**
- Modify: `apps/api/src/lib/types.ts`
- Modify: `apps/api/wrangler.toml`
- Modify: `apps/api/src/do/session-brain.ts`

- [ ] **Step 1: Confirm the existing regression test still encodes the contract**

`apps/api/src/do/session-brain.unit.test.ts` already has:

```typescript
describe("computeSessionDurationMs", () => {
	it("derives duration from scored-chunk count at 15s per chunk", () => {
		expect(computeSessionDurationMs(10)).toBe(150000);
	});
	// ...
});
```

No new test is needed — this is a pure rename and the existing test is the
regression guard. Confirm it currently passes:

```bash
cd apps/api && bun run test -- --run session-brain.unit.test.ts
```
Expected: PASS (pre-change baseline).

- [ ] **Step 2: N/A (rename, not new behavior) — proceed to Step 3.**

- [ ] **Step 3: Implement the rename**

In `apps/api/src/lib/types.ts`, remove the line:

```typescript
	MUQ_ENDPOINT: string;
```

leaving:

```typescript
	AMT_ENDPOINT: string;
```

In `apps/api/wrangler.toml`, remove the line:

```
MUQ_ENDPOINT = "https://mxcyiqltad84v9w1.us-east4.gcp.endpoints.huggingface.cloud"
```

In `apps/api/src/do/session-brain.ts`, replace:

```typescript
/** MuQ audio chunk length in milliseconds (15s per scored chunk). */
const MUQ_CHUNK_MS = 15000;
```

with:

```typescript
/** Audio chunk length in milliseconds (15s per scored chunk). */
const CHUNK_MS = 15000;
```

and replace:

```typescript
export function computeSessionDurationMs(scoredChunkCount: number): number {
	return scoredChunkCount * MUQ_CHUNK_MS;
}
```

with:

```typescript
export function computeSessionDurationMs(scoredChunkCount: number): number {
	return scoredChunkCount * CHUNK_MS;
}
```

- [ ] **Step 4: Run test — verify it still PASSES**

```bash
cd apps/api && bun run typecheck && bun run test -- --run session-brain.unit.test.ts
```
Expected: PASS. `typecheck` also confirms no other file still references
`MUQ_ENDPOINT` on `Bindings` (it will fail loudly at this point in
`apps/api/src/services/inference.ts`'s `callMuqEndpoint`, which is expected —
that function is deleted in Task 18, not this task. Confirm the *only*
typecheck error at this point is inside `callMuqEndpoint`'s body; if there
are others, stop and re-scope this task).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/lib/types.ts apps/api/wrangler.toml apps/api/src/do/session-brain.ts && git commit -m "chore(#162): rename MUQ_CHUNK_MS to CHUNK_MS, drop MUQ_ENDPOINT binding"
```

### Task 12: Rip MuQ from `finalizeChunk`; wire the no-comment zone and `scoreChunk`

**Group:** B (depends on Task 11)

**Behavior being verified:** A chunk transcribing >=10 notes is scored by
`scoreChunk` (not MuQ) and its per-dimension scores accumulate into
`state.sessionSamples`; a chunk transcribing <10 notes, or whose AMT call
rejects, records a `noCommentZones` entry and contributes no samples.

**Interface under test:** `SessionBrain.webSocketMessage("chunk_ready", ...)`
(public), observed via `runInDurableObject` + direct storage reads, following
the exact pattern of `session-brain.concurrency.test.ts`.

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Test: `apps/api/src/do/session-brain.no-comment-zone.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
/// <reference types="@cloudflare/vitest-pool-workers" />

import { env, runInDurableObject } from "cloudflare:test";
import { describe, expect, it, vi } from "vitest";

vi.mock("../services/inference", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callAmtEndpoint: vi.fn(async (_env, _chunk, _ctx) => {
			return { notes: [{ pitch: 60, onset: 0, offset: 0.4, velocity: 70 }], pedalEvents: [] };
		}),
	};
});

import type { SessionBrain } from "./session-brain";
import { createInitialState } from "./session-brain.schema";

declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
		CHUNKS: R2Bucket;
	}
}

const fakeWs = { send() {}, close() {} } as unknown as WebSocket;

describe("no-comment zone (#162): fewer than MIN_NOTES_FOR_COMMENT transcribed notes", () => {
	it("records a zone and contributes no sessionSamples for a 1-note chunk", async () => {
		await env.CHUNKS.put("ncz-0", new Uint8Array([1]));
		const id = env.SESSION_BRAIN.idFromName("no-comment-zone");
		const stub = env.SESSION_BRAIN.get(id);

		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-ncz", "stud-ncz", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			seeded.pieceLocked = true;
			seeded.markBaselineLoaded = true;
			await state.storage.put("state", seeded);

			await instance.webSocketMessage(
				fakeWs,
				JSON.stringify({ type: "chunk_ready", index: 0, r2Key: "ncz-0" }),
			);

			const after = (await state.storage.get("state")) as {
				noCommentZones: { chunkIndex: number; reason: string }[];
				sessionSamples: Record<string, number[]>;
			};
			expect(after.noCommentZones).toEqual([
				{ chunkIndex: 0, reason: "insufficient_notes" },
			]);
			expect(after.sessionSamples).toEqual({});

			await state.storage.deleteAlarm();
		});
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run session-brain.no-comment-zone.test.ts
```
Expected: FAIL — `after.noCommentZones` is `undefined` (field does not exist
on the pre-wiring state shape's runtime behavior; `finalizeChunk` still
pushes MuQ-sourced `scoredChunks` unconditionally and never writes
`noCommentZones`).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/do/session-brain.ts`, change the import at the top:

```typescript
import { callAmtEndpoint, callMuqEndpoint } from "../services/inference";
```
to:
```typescript
import { callAmtEndpoint } from "../services/inference";
import { scoreChunk } from "../services/scoring";
```

Add near `TEACHING_MOMENT_INTERVAL`:

```typescript
// Minimum transcribed notes before a chunk is comment-worthy. Transkun
// exposes no confidence signal, so note density is the proxy (spec
// docs/apps/02-pipeline.md open question 2).
const MIN_NOTES_FOR_COMMENT = 10;
```

In `handleChunkReady`, replace:

```typescript
		// 3. Parallel inference — this is the major await; state may change during it
		const [muqResult, amtResult] = await Promise.allSettled([
			callMuqEndpoint(this.env, audioBytes),
			callAmtEndpoint(this.env, audioBytes, contextAudio),
		]);
```
with:
```typescript
		// 3. Inference — this is the major await; state may change during it. MuQ is
		// gone (#162): AMT (Transkun) is the only inference call on the request path.
		const [amtResult] = await Promise.allSettled([
			callAmtEndpoint(this.env, audioBytes, contextAudio),
		]);
```

Change `finalizeChunk`'s signature from:

```typescript
	private async finalizeChunk(
		ws: WebSocket,
		index: number,
		muqResult: PromiseSettledResult<
			Awaited<ReturnType<typeof callMuqEndpoint>>
		>,
		amtResult: PromiseSettledResult<
			Awaited<ReturnType<typeof callAmtEndpoint>>
		>,
	): Promise<void> {
```
to:
```typescript
	private async finalizeChunk(
		ws: WebSocket,
		index: number,
		amtResult: PromiseSettledResult<
			Awaited<ReturnType<typeof callAmtEndpoint>>
		>,
	): Promise<void> {
```

Update its one call site in `handleChunkReady`:

```typescript
		await this.ctx.blockConcurrencyWhile(() =>
			this.finalizeChunk(ws, index, muqResult, amtResult),
		);
```
to:
```typescript
		await this.ctx.blockConcurrencyWhile(() =>
			this.finalizeChunk(ws, index, amtResult),
		);
```

Replace the entire block from `// 5. Process MuQ result — send chunk_processed immediately`
through the end of `// 6. Process AMT result (graceful degradation: AMT failure -> Tier 3)`'s
`if`/`else` (i.e. everything from the original lines 769-851 you read
earlier) with:

```typescript
		// 5. Process AMT result (graceful degradation: AMT failure -> no-comment zone)
		let perfNotes: PerfNote[] = [];
		let perfPedal: PerfPedalEvent[] = [];
		let amtFailed = false;

		if (amtResult.status === "rejected") {
			console.error(
				JSON.stringify({
					level: "error",
					message: "AMT inference failed",
					index,
					error:
						amtResult.reason instanceof Error
							? amtResult.reason.message
							: String(amtResult.reason),
				}),
			);
			currentState.inferenceFailures++;
			amtFailed = true;
		} else {
			perfNotes = amtResult.value.notes;
			perfPedal = amtResult.value.pedalEvents;
		}

		const isNoCommentZone = amtFailed || perfNotes.length < MIN_NOTES_FOR_COMMENT;
		let scoresArray: [number, number, number, number, number, number] = [
			0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		];

		if (isNoCommentZone) {
			currentState.noCommentZones.push({
				chunkIndex: index,
				reason: amtFailed ? "amt_rejected" : "insufficient_notes",
			});
			this.sendWs(ws, {
				type: "chunk_processed",
				index,
				scores: {
					dynamics: 0,
					timing: 0,
					pedaling: 0,
					articulation: 0,
					phrasing: 0,
					interpretation: 0,
				},
			});
		} else {
			const scores = scoreChunk(perfNotes, perfPedal);
			scoresArray = [
				scores.dynamics,
				scores.timing,
				scores.pedaling,
				scores.articulation,
				scores.phrasing,
				scores.interpretation,
			];
			this.sendWs(ws, { type: "chunk_processed", index, scores });
			currentState.scoredChunks.push({ chunkIndex: index, scores: scoresArray });
			for (const dim of DIMS_6) {
				const existing = currentState.sessionSamples[dim] ?? [];
				currentState.sessionSamples[dim] = [...existing, scores[dim]];
			}
		}
```

Note this deletes the old `muqScores`/`muqConfidences`/`chromaBytes`/
`chromaFrames`/`chromaFrameRateHz` destructure and the old
`currentState.scoredChunks.push` — the new block above owns both, gated on
`isNoCommentZone`. `perfNotes`/`perfPedal` retain their original types
(`PerfNote[]`/`PerfPedalEvent[]`, already imported at the top of the file).
Downstream code (bar analysis, passage-loop detection, enriched-chunk
storage, teaching-moment selection) is untouched by this task — it already
reads `perfNotes`/`perfPedal`/`scoresArray`, all of which retain their prior
shapes and are now always defined (falling back to the neutral 0.5 array in
a no-comment-zone chunk, matching the existing failure convention elsewhere
in this function).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run session-brain.no-comment-zone.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.ts apps/api/src/do/session-brain.no-comment-zone.test.ts && git commit -m "feat(#162): rip MuQ from finalizeChunk, wire no-comment zones and scoreChunk"
```

### Task 13: Remove the tier-1 branch and `chunk_bar_map` emission

**Group:** B (depends on Task 12)

**Behavior being verified:** Every scored chunk now takes the Tier-2 path
unconditionally (no chroma-driven Tier-1 branch), and no `chunk_bar_map`
event is ever sent.

**Interface under test:** `SessionBrain.webSocketMessage("chunk_ready", ...)`,
observed via captured `ws.send` calls.

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Test: `apps/api/src/do/session-brain.no-comment-zone.test.ts` (extended)

- [ ] **Step 1: Write the regression-guard test**

This task is a structural deletion, not new behavior: the tier-1 chroma
branch cannot be exercised even on `main` today without a real score-catalog
row (`loadScoreContext` throws and falls through to Tier-3 without one), so
there is no reachable pre-change state in which this test could be red. The
test below is a regression guard against a future re-introduction of
`chunk_bar_map` emission, verified by typecheck + Task 27's grep gate (see
Step 2), not by a red/green TDD cycle.

Add to `apps/api/src/do/session-brain.no-comment-zone.test.ts` (needs a
second AMT mock returning >=10 notes; add a second `describe` block with its
own `vi.mock` scope is not possible in the same file — instead extend the
existing mock to return notes conditionally per call count):

```typescript
// Replace the single vi.mock block at the top of the file with a
// call-counting version so both scenarios (sparse and sufficient notes) can
// live in one file without a second module-level mock:
```

Since `vi.mock` factories cannot be redefined per-test in one file, split
Task 13's test into its own file instead:

**Files (corrected):**
- Create: `apps/api/src/do/session-brain.tier2-only.test.ts`

```typescript
/// <reference types="@cloudflare/vitest-pool-workers" />

import { env, runInDurableObject } from "cloudflare:test";
import { describe, expect, it, vi } from "vitest";

vi.mock("../services/inference", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callAmtEndpoint: vi.fn(async () => ({
			notes: Array.from({ length: 10 }, (_, i) => ({
				pitch: 60,
				onset: i * 0.3,
				offset: i * 0.3 + 0.2,
				velocity: 70,
			})),
			pedalEvents: [],
		})),
	};
});

import type { SessionBrain } from "./session-brain";
import { createInitialState } from "./session-brain.schema";

declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
		CHUNKS: R2Bucket;
	}
}

describe("tier-1 removal (#162): every chunk takes Tier-2, no chunk_bar_map", () => {
	it("never sends a chunk_bar_map message even with a piece context set", async () => {
		await env.CHUNKS.put("t2-0", new Uint8Array([1]));
		const id = env.SESSION_BRAIN.idFromName("tier2-only");
		const stub = env.SESSION_BRAIN.get(id);
		const sent: unknown[] = [];
		const fakeWs = {
			send(data: string) {
				sent.push(JSON.parse(data));
			},
			close() {},
		} as unknown as WebSocket;

		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-t2", "stud-t2", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			seeded.pieceLocked = true;
			seeded.markBaselineLoaded = true;
			await state.storage.put("state", seeded);

			await instance.webSocketMessage(
				fakeWs,
				JSON.stringify({ type: "chunk_ready", index: 0, r2Key: "t2-0" }),
			);

			await state.storage.deleteAlarm();
		});

		expect(sent.some((m) => (m as { type?: string }).type === "chunk_bar_map")).toBe(false);
	});
});
```

- [ ] **Step 2: Run test — confirm it already PASSES (not a red step)**

```bash
cd apps/api && bun run test -- --run session-brain.tier2-only.test.ts
```
Expected: PASS, both now (pre-change) and after Step 3's implementation
edit. This is expected and correct, not a mistake to fix: the seeded state
has no real score-catalog row backing `pieceIdentification`, so even with
`pieceIdentification` set, `loadScoreContext` throws inside the existing
`try` and falls through to Tier-3 in both the before and after code —
`chunk_bar_map` was never reachable pre-change and stays unreachable
post-change. There is no code path here that a red/green cycle could
exercise; this is a structural regression guard against a future
re-introduction of `chunk_bar_map` emission, not a TDD red step. The actual
proof that the tier-1 branch is *deleted* (not just unreachable) is:
`bun run typecheck` in Step 4, which fails loudly on any dangling reference
to the removed tier-1 locals (`ChunkNoteResult`, `ScoreContext`,
`TIER1_ONSET_WINDOW_S`, `wasm.alignChunkNotes`), plus Task 27's grep gate,
which greps the source tree for those same removed identifiers.

The existing `describe("chunk_bar_map WebSocket message", ...)` block in
`session-brain.unit.test.ts` tests only the **schema**, not emission — leave
it in place (`chunk_bar_map` remains a valid, just-unused, outgoing type).

- [ ] **Step 3: Implement the change**

In `apps/api/src/do/session-brain.ts`, replace the entire `// 6b. Bar
analysis` block — from `let chunkAnalysisTier = 3;` through the closing
`}` of the outer `if (perfNotes.length > 0) { ... } catch (err) { ... }` you
read earlier (originally covering the tier-1 chroma branch, the
`wasm.alignChunkNotes` call, the `chunk_bar_map` send, and the tier-1/tier-2
`if (scoreCtx !== null) { ... } else { ... }` split) — with:

```typescript
		// 6b. Bar analysis: Tier-2 (absolute MIDI) unconditionally. The chroma-driven
		// Tier-1 branch is removed with MuQ (#162) — its chroma input no longer exists.
		// The WASM aligner (alignChunkNotes/alignChunkChroma, #64) stays in the tree,
		// dormant, for the follower issue that restores bar anchoring.
		let chunkAnalysisTier = 3;
		let chunkBarRange: [number, number] | null = null;
		let chunkAnalysis: ChunkAnalysis | null = null;
		const barMapAlignments: import("../services/wasm-bridge").NoteAlignment[] =
			[];

		if (perfNotes.length > 0) {
			try {
				const analysis = wasm.analyzeTier2(perfNotes, perfPedal, scoresArray);
				chunkAnalysis = analysis;
				chunkAnalysisTier = analysis.tier;
				const barStr = analysis.bar_range;
				if (barStr !== null) {
					const parts = barStr.split("-").map(Number);
					if (
						parts.length === 2 &&
						parts[0] !== undefined &&
						parts[1] !== undefined
					) {
						chunkBarRange = [parts[0], parts[1]];
					}
				}
			} catch (err) {
				const error = err as Error;
				console.log(
					JSON.stringify({
						level: "warn",
						message: "WASM bar analysis skipped",
						index,
						error: error.message,
					}),
				);
				// Tier 3: scores only — chunkAnalysisTier stays 3
			}
		}
```

This drops the `currentState.pieceIdentification` / `loadScoreContext` /
`ChunkNoteResult` / `wasm.alignChunkNotes` / `TIER1_ONSET_WINDOW_S` usages
from this block entirely (Tier-2 never needed piece context). `barMapAlignments`
stays declared (still consumed by `toEnrichedChunk` below, unchanged) but is
now always `[]`. `TIER1_ONSET_WINDOW_S` becomes unused — remove its
declaration too:

```typescript
// Tier-1 note matching tolerance (score-time seconds): how far a warped perf-note
// onset may sit from a notated score-note onset to count as the same note. Generous
// because timing here is coarse/directional (the bar-MEAN deviation is what matters).
const TIER1_ONSET_WINDOW_S = 0.15;
```
(delete this whole comment+const).

Remove the now-unused `ChunkNoteResult` and `ScoreContext` imports from the
top-of-file type import list if `bun run check-api` flags them as unused
after this edit (they were only consumed by the deleted tier-1 branch).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run typecheck && bun run test -- --run session-brain.tier2-only.test.ts
```
Expected: PASS, with `typecheck` confirming no dangling references.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.ts apps/api/src/do/session-brain.tier2-only.test.ts && git commit -m "feat(#162): remove tier-1 chroma branch and chunk_bar_map emission"
```

### Task 14: Wire `markBaseline` load, `proposeMarks`, and `mark` emission

**Group:** B (depends on Task 13)

**Behavior being verified:** A chunk whose dimension samples cross the mark
gate (cold-start or persisted) emits a `mark` WS event carrying a
timestamp anchor, MPM evidence, and the correct taxonomy — and the fired
dimension is recorded in `state.emittedMarks` so it never fires twice.

**Interface under test:** `SessionBrain.webSocketMessage("chunk_ready", ...)`

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Test: `apps/api/src/do/session-brain.mark-emission.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
/// <reference types="@cloudflare/vitest-pool-workers" />

import { env, runInDurableObject } from "cloudflare:test";
import { describe, expect, it, vi } from "vitest";

const { callIndex, VELOCITIES } = vi.hoisted(() => ({
	callIndex: { current: 0 },
	VELOCITIES: [62, 66, 63, 65, 64, 10, 10, 10],
}));

vi.mock("../services/inference", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callAmtEndpoint: vi.fn(async () => {
			const velocity = VELOCITIES[callIndex.current] ?? 64;
			callIndex.current++;
			return {
				notes: Array.from({ length: 10 }, (_, i) => ({
					pitch: 60,
					onset: i * 1.0,
					offset: i * 1.0 + 0.5,
					velocity,
				})),
				pedalEvents: [],
			};
		}),
	};
});

import type { SessionBrain } from "./session-brain";
import { createInitialState } from "./session-brain.schema";

declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
		CHUNKS: R2Bucket;
	}
}

describe("mark emission (#162)", () => {
	it("emits exactly one dynamics/needs_work mark by the 8th chunk and records it in emittedMarks", async () => {
		for (let i = 0; i < 8; i++) {
			await env.CHUNKS.put(`mk-${i}`, new Uint8Array([1]));
		}
		const id = env.SESSION_BRAIN.idFromName("mark-emission");
		const stub = env.SESSION_BRAIN.get(id);
		const sent: unknown[] = [];
		const fakeWs = {
			send(data: string) {
				sent.push(JSON.parse(data));
			},
			close() {},
		} as unknown as WebSocket;

		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-mk", "stud-mk", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			seeded.pieceLocked = true;
			seeded.markBaselineLoaded = true;
			await state.storage.put("state", seeded);

			for (let i = 0; i < 8; i++) {
				await instance.webSocketMessage(
					fakeWs,
					JSON.stringify({ type: "chunk_ready", index: i, r2Key: `mk-${i}` }),
				);
			}

			const after = (await state.storage.get("state")) as {
				emittedMarks: string[];
			};
			expect(after.emittedMarks).toEqual(["dynamics"]);

			await state.storage.deleteAlarm();
		});

		const marks = sent.filter((m) => (m as { type?: string }).type === "mark");
		expect(marks).toHaveLength(1);
		expect(marks[0]).toMatchObject({
			type: "mark",
			mark: { dimension: "dynamics", taxonomy: "needs_work" },
		});
		expect((marks[0] as { mark: { anchor: unknown } }).mark.anchor).toEqual({
			type: "timestamp",
			atSeconds: 105, // chunkIndex 7 * 15s
		});
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run session-brain.mark-emission.test.ts
```
Expected: FAIL — `sent` contains zero `mark` messages (nothing calls
`proposeMarks` yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

Add imports at the top of `apps/api/src/do/session-brain.ts`:

```typescript
import { evidenceFor, extractMpmFeatures } from "../services/mpm-features";
import { proposeMarks } from "../services/mark-gate";
import { loadBaselineState } from "../services/baseline-store";
import { DEFAULT_BASELINE_CONFIG } from "../services/student-baseline";
```

In `finalizeChunk`, immediately after the `isNoCommentZone`/`else` block from
Task 12 (i.e. right after the `for (const dim of DIMS_6) { ... }` loop that
populates `currentState.sessionSamples`), insert:

```typescript
			// Lazy-load the student's persisted mark baseline once per session.
			if (!currentState.markBaselineLoaded) {
				try {
					const db = createDb(this.env.HYPERDRIVE);
					currentState.markBaseline = await loadBaselineState(
						db,
						currentState.studentId,
					);
				} catch (err) {
					const error = err as Error;
					console.error(
						JSON.stringify({
							level: "error",
							message: "markBaseline load failed",
							error: error.message,
						}),
					);
				}
				currentState.markBaselineLoaded = true;
			}

			const { candidates } = proposeMarks({
				atSeconds: index * (CHUNK_MS / 1000),
				scores,
				baseline: currentState.markBaseline,
				sessionSamples: currentState.sessionSamples as Partial<
					Record<Dimension, number[]>
				>,
				emittedDimensions: new Set(currentState.emittedMarks as Dimension[]),
				firePersistence: DEFAULT_BASELINE_CONFIG.firePersistence,
			});

			for (const candidate of candidates) {
				const [mpmSet] = extractMpmFeatures({
					windowIndex: index,
					perfNotes,
					perfPedal,
				});
				if (mpmSet === undefined) continue;
				const dimState = currentState.markBaseline?.dimensions[candidate.dimension];
				const lifecycle: "active" | "improving" | "resolved" =
					dimState !== undefined && dimState.lifecycle !== "absent"
						? dimState.lifecycle
						: "active";
				const confidence = dimState?.confidence ?? "exploratory";
				currentState.emittedMarks = [
					...currentState.emittedMarks,
					candidate.dimension,
				];
				this.sendWs(ws, {
					type: "mark",
					mark: {
						id: `${currentState.sessionId}-${candidate.dimension}-${index}`,
						anchor: { type: "timestamp" as const, atSeconds: candidate.atSeconds },
						taxonomy: candidate.taxonomy,
						dimension: candidate.dimension,
						evidence: evidenceFor(mpmSet, candidate.dimension),
						lifecycle,
						confidence,
					},
				});
			}
```

This block must sit inside the `else` branch of `isNoCommentZone` (a
no-comment-zone chunk never proposes marks) — place it as the last statement
inside that `else { ... }` body from Task 12, after the `sessionSamples`
loop and before the closing `}` of the `else`. `scores` (the `DimScores`
object from `scoreChunk`) is already in scope there from Task 12's edit.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run session-brain.mark-emission.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.ts apps/api/src/do/session-brain.mark-emission.test.ts && git commit -m "feat(#162): wire markBaseline load, mark gate, and mark emission"
```

### Task 15: Gate `observation` emission on `isEvalSession`; remove the dead confidence-gate call

**Group:** B (depends on Task 14)

**Behavior being verified:** A real (non-eval) session never receives an
`observation` WS event even when a teaching moment is selected; an eval
session's `observation` emission is unchanged.

**Interface under test:** `SessionBrain.webSocketMessage("chunk_ready", ...)`

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Test: `apps/api/src/do/session-brain.observation.test.ts` (existing file, extended)

- [ ] **Step 1: Write the failing test**

`apps/api/src/do/session-brain.observation.test.ts`, as it exists today,
only unit-tests `buildObservationPayload` directly with plain `vitest` — it
has no DO-integration seeding/mocking convention to reuse. The convention
that applies here instead comes from this task's sibling Group-B test files
(`session-brain.no-comment-zone.test.ts`, Task 12;
`session-brain.tier2-only.test.ts`, Task 13; `session-brain.mark-emission.test.ts`,
Task 14), all of which drive `webSocketMessage("chunk_ready", ...)` through
`runInDurableObject` with a mocked `callAmtEndpoint` and a `sent`-array
`fakeWs`. Append the following to the top and bottom of
`session-brain.observation.test.ts` (the existing plain-vitest tests and
imports stay unchanged; this only adds a new `vi.mock`, new imports, and one
new `describe` block):

At the top of the file, before the existing imports:

```typescript
/// <reference types="@cloudflare/vitest-pool-workers" />

import { env, runInDurableObject } from "cloudflare:test";
import { vi } from "vitest";

vi.mock("../services/inference", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callAmtEndpoint: vi.fn(async () => ({
			notes: Array.from({ length: 10 }, (_, i) => ({
				pitch: 60,
				onset: i * 0.3,
				offset: i * 0.3 + 0.2,
				velocity: 20,
			})),
			pedalEvents: [],
		})),
	};
});
```

(The existing `import { describe, expect, it } from "vitest";` line gains
`vi` — merge into one import instead of duplicating it.)

After the existing `import { buildObservationPayload } from "./session-brain";`
line, add:

```typescript
import type { SessionBrain } from "./session-brain";
import { createInitialState } from "./session-brain.schema";

declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
		CHUNKS: R2Bucket;
	}
}
```

At the bottom of the file, after the existing `describe("buildObservationPayload", ...)` block:

```typescript
describe("observation gating (#162): real sessions never receive observation", () => {
	it("never sends observation for a real (non-eval) session even when a moment is selected", async () => {
		await env.CHUNKS.put("obs-0", new Uint8Array([1]));
		await env.CHUNKS.put("obs-1", new Uint8Array([1]));
		const id = env.SESSION_BRAIN.idFromName("observation-gate");
		const stub = env.SESSION_BRAIN.get(id);
		const sent: unknown[] = [];
		const fakeWs = {
			send(data: string) {
				sent.push(JSON.parse(data));
			},
			close() {},
		} as unknown as WebSocket;

		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-obs", "stud-obs", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = {
				dynamics: 0.55,
				timing: 0.5,
				pedaling: 0.45,
				articulation: 0.54,
				phrasing: 0.52,
				interpretation: 0.5,
			};
			seeded.pieceLocked = true;
			seeded.markBaselineLoaded = true;
			// isEvalSession stays false -- createInitialState's default.
			await state.storage.put("state", seeded);

			await instance.webSocketMessage(
				fakeWs,
				JSON.stringify({ type: "chunk_ready", index: 0, r2Key: "obs-0" }),
			);
			await instance.webSocketMessage(
				fakeWs,
				JSON.stringify({ type: "chunk_ready", index: 1, r2Key: "obs-1" }),
			);

			await state.storage.deleteAlarm();
		});

		// Two scored chunks (>= MIN_NOTES_FOR_COMMENT notes each, so neither is a
		// no-comment zone) plus a non-null baseline are enough to guarantee
		// select_teaching_moment returns Some: per teaching_moments.rs, once
		// chunks.len() >= MIN_CHUNKS (2) it either finds a below-baseline
		// candidate or falls back to a positive moment -- it only returns None
		// below MIN_CHUNKS. So a moment is selected here, and the assertion
		// below is a genuine test of the isEvalSession gate, not a vacuous one.
		expect(
			sent.some((m) => (m as { type?: string }).type === "observation"),
		).toBe(false);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run session-brain.observation.test.ts
```
Expected: FAIL — the existing (pre-Task-15) code sends `observation`
unconditionally whenever a teaching moment is selected and not suppressed,
regardless of `isEvalSession`. The seeded state above guarantees a moment is
selected (two scored chunks, non-null baselines), so `sent` contains an
`observation` message and `toBe(false)` fails.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/do/session-brain.ts`, locate the teaching-moment block
(originally around the confidence-gate + `accMoment` + `sendWs` code you
read earlier). Replace:

```typescript
					// Suppress high-uncertainty dims: if the gaussian head provided
					// confidences and this dim's sigma exceeds threshold, skip it.
					const gated =
						muqConfidences !== null
							? applyConfidenceGate(muqScores, muqConfidences)
							: null;
					const dimSuppressed =
						gated !== null && gated.suppressed.includes(momentDim);
```
with:
```typescript
					// The mock scorer carries no confidence signal (unlike the retired
					// MuQ gaussian head), so there is nothing to suppress on here (#162).
					const dimSuppressed = false;
```

Replace:

```typescript
					if (!dimSuppressed) {
						acc.accumulateMoment(accMoment);

						this.sendWs(
							ws,
							buildObservationPayload(
								moment,
								currentState.isEvalSession
									? {
											predictions: scoresArray,
											baselines,
											analysisFacts: accMoment.llmAnalysis,
											barRange: chunkBarRange,
											analysisTier: chunkAnalysisTier,
										}
									: null,
							),
						);
					}
```
with:
```typescript
					if (!dimSuppressed) {
						acc.accumulateMoment(accMoment);

						// Real sessions no longer stream observations (#162) — marks are
						// the delivery surface now. Eval sessions keep the observation
						// stream so the teaching-knowledge eval stays comparable.
						if (currentState.isEvalSession) {
							this.sendWs(
								ws,
								buildObservationPayload(moment, {
									predictions: scoresArray,
									baselines,
									analysisFacts: accMoment.llmAnalysis,
									barRange: chunkBarRange,
									analysisTier: chunkAnalysisTier,
								}),
							);
						}
					}
```

Remove the now-unused `applyConfidenceGate` import:

```typescript
import { applyConfidenceGate } from "../services/confidence_gate";
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run typecheck && bun run test -- --run session-brain.observation.test.ts
```
Expected: PASS. `typecheck` also confirms `applyConfidenceGate`'s removal
does not break `services/confidence_gate.ts` itself (that file is untouched
and still exports the same function for any future caller — it is not in
the spec's file-changes list, so it is left in place, just unused by
`session-brain.ts`).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.ts apps/api/src/do/session-brain.observation.test.ts && git commit -m "feat(#162): gate observation emission on isEvalSession, drop dead confidence-gate call"
```

### Task 16: Review event and baseline fold on synthesis success

**Group:** B (depends on Task 15)

**Behavior being verified:** On a successful (non-eval) synthesis, the DO
sends `{type: "review", verdict: artifact.headline, carry_forward:
artifact.next_session_focus, no_comment_zones}` to every open socket instead
of `synthesis`, then folds the session's `sessionSamples` into the student's
`BaselineState` via `updateBaseline` and persists it with `saveBaselineState`.

**Interface under test:** `SessionBrain.alarm()` (via
`runDurableObjectAlarm`), the same public surface Group 0's harness drives.

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Test: covered by Task 1's Group 0 harness (`session-brain.marks.test.ts`)
  — this task is what turns its `review`/baseline-insert assertions green.
  No new test file; re-run Task 1's test as this task's verification.

- [ ] **Step 1: Confirm the target behavior via the existing Group 0 test**

`apps/api/src/do/session-brain.marks.test.ts` (Task 1) already asserts:
```typescript
expect(reviews[0]).toMatchObject({
	type: "review",
	verdict: "H".repeat(320),
	carry_forward: "Keep the dynamic range even across repeated passages.",
});
const baselineInsert = capturedInserts.find(
	(c) => c.table === studentProfiles && (c.values as Record<string, unknown>).baselineState !== undefined,
);
expect(baselineInsert).toBeDefined();
```

- [ ] **Step 2: Run test — verify it still FAILS at this line**

```bash
cd apps/api && bun run test -- --run session-brain.marks.test.ts
```
Expected: FAIL — `reviews` is empty (the DO still sends `synthesis`, not
`review`, and never calls `saveBaselineState`).

- [ ] **Step 3: Implement the minimum to make the test pass**

Add imports at the top of `apps/api/src/do/session-brain.ts`:

```typescript
import { saveBaselineState } from "../services/baseline-store";
import { initialBaselineState, updateBaseline } from "../services/student-baseline";
```

In `runSynthesisAndPersist`, replace the broadcast block:

```typescript
			const wsPayload = buildV6WsPayload(
				artifact,
				loopComponents,
				pendingComponent,
			);
			const wsPayloadWithEval =
				evalContext !== null
					? {
							...wsPayload,
							eval_context: buildEvalContext(evalContext, artifact),
						}
					: wsPayload;
			const sockets = this.ctx.getWebSockets();
			for (const sock of sockets) {
				this.sendWs(sock, wsPayloadWithEval);
			}
```
with:
```typescript
			const wsPayload = buildV6WsPayload(
				artifact,
				loopComponents,
				pendingComponent,
			);
			const sockets = this.ctx.getWebSockets();

			if (state.isEvalSession) {
				const wsPayloadWithEval =
					evalContext !== null
						? { ...wsPayload, eval_context: buildEvalContext(evalContext, artifact) }
						: wsPayload;
				for (const sock of sockets) {
					this.sendWs(sock, wsPayloadWithEval);
				}
			} else {
				const reviewPayload = {
					type: "review" as const,
					verdict: artifact.headline,
					carry_forward: artifact.next_session_focus,
					no_comment_zones: state.noCommentZones.map((z) => ({
						atSeconds: z.chunkIndex * (CHUNK_MS / 1000),
						reason: z.reason,
					})),
				};
				for (const sock of sockets) {
					this.sendWs(sock, reviewPayload);
				}
			}
```

Immediately after that block (still inside the `try`, before the eval
flywheel `if` block), add the baseline fold — this runs for every session,
eval or not, since the baseline signal is real regardless of delivery
surface:

```typescript
			try {
				const hasSamples = Object.values(state.sessionSamples).some(
					(samples) => (samples?.length ?? 0) > 0,
				);
				if (hasSamples) {
					const priorBaseline = state.markBaseline ?? initialBaselineState();
					const foldedBaseline = updateBaseline(priorBaseline, {
						timestamp: new Date().toISOString(),
						scores: state.sessionSamples as Partial<Record<Dimension, number[]>>,
					});
					await saveBaselineState(db, state.studentId, foldedBaseline);
				}
			} catch (err) {
				const error = err as Error;
				console.error(
					JSON.stringify({
						level: "error",
						message: "baseline fold/persist failed",
						sessionId: state.sessionId,
						error: error.message,
					}),
				);
			}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run typecheck && bun run test -- --run session-brain.marks.test.ts
```
Expected: PASS — every assertion in the Group 0 test now holds, including
the mark, the review, and the baseline insert.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.ts && git commit -m "feat(#162): emit review event and fold+persist the session baseline on synthesis success"
```

### Task 17: Review event on synthesis failure

**Group:** B (depends on Task 16 — this is the task that finally closes Group 0)

**Behavior being verified:** When `synthesizeV6` produces no usable artifact
(the existing `validationError !== null` or `artifact === null` early
returns, or the outer `catch`), a non-eval session still receives exactly one
loud `{type: "review", verdict: null, carry_forward: null, no_comment_zones}`
event instead of silently returning.

**Interface under test:** `SessionBrain.alarm()`

**Files:**
- Modify: `apps/api/src/do/session-brain.ts`
- Test: `apps/api/src/do/session-brain.review-failure.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
/// <reference types="@cloudflare/vitest-pool-workers" />

import {
	env,
	runDurableObjectAlarm,
	runInDurableObject,
} from "cloudflare:test";
import { describe, expect, it, vi } from "vitest";

vi.mock("../db/client", () => ({
	createDb: () =>
		new Proxy(
			{},
			{
				get(_t, prop) {
					if (prop === "then") return (resolve: (v: unknown[]) => void) => resolve([]);
					if (prop === "query") {
						return new Proxy({}, { get: () => ({ findFirst: async () => undefined }) });
					}
					return () => new Proxy({}, {}); // reused below via a stable proxy in practice
				},
			},
		),
}));

vi.mock("../services/teacher", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/teacher")>();
	return {
		...actual,
		synthesizeV6: async function* () {
			throw new Error("teacher call failed");
		},
	};
});

import type { SessionBrain } from "./session-brain";
import { createInitialState } from "./session-brain.schema";

declare module "cloudflare:test" {
	interface ProvidedEnv {
		SESSION_BRAIN: DurableObjectNamespace<SessionBrain>;
	}
}

describe("review event on synthesis failure (#162)", () => {
	it("sends exactly one review event with verdict: null when synthesizeV6 throws", async () => {
		const id = env.SESSION_BRAIN.idFromName("review-failure");
		const stub = env.SESSION_BRAIN.get(id);
		const received: unknown[] = [];

		await runInDurableObject(stub, async (instance: SessionBrain, state) => {
			const seeded = createInitialState("sess-fail", "stud-fail", null);
			seeded.baselinesLoaded = true;
			seeded.baselines = null;
			seeded.pieceLocked = true;
			seeded.markBaselineLoaded = true;
			await state.storage.put("state", seeded);

			const pair = new WebSocketPair();
			const [client, server] = Object.values(pair) as [WebSocket, WebSocket];
			state.acceptWebSocket(server);
			client.accept();
			client.addEventListener("message", (evt: MessageEvent) => {
				received.push(JSON.parse(evt.data as string));
			});

			await runDurableObjectAlarm(stub);
			await new Promise((resolve) => setTimeout(resolve, 20));
			await state.storage.deleteAlarm();
		});

		const reviews = received.filter((m) => (m as { type?: string }).type === "review");
		expect(reviews).toEqual([
			{ type: "review", verdict: null, carry_forward: null, no_comment_zones: [] },
		]);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run session-brain.review-failure.test.ts
```
Expected: FAIL — `received` is empty; the current `catch` block only logs
and returns, sending nothing.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/do/session-brain.ts`, `runSynthesisAndPersist`'s outer
`catch (err)` block currently reads:

```typescript
		} catch (err) {
			const error = err as Error;
			console.error(
				JSON.stringify({
					level: "error",
					message: "v6 synthesis failed",
					sessionId: state.sessionId,
					error: error.message,
				}),
			);
			// synthesisCompleted stays true (no duplicate), but needsSynthesis in DB stays true
			// so the deferred /synthesize endpoint can retry via the DO
		}
```

Replace with:

```typescript
		} catch (err) {
			const error = err as Error;
			console.error(
				JSON.stringify({
					level: "error",
					message: "v6 synthesis failed",
					sessionId: state.sessionId,
					error: error.message,
				}),
			);
			if (!state.isEvalSession) {
				const sockets = this.ctx.getWebSockets();
				const failurePayload = {
					type: "review" as const,
					verdict: null,
					carry_forward: null,
					no_comment_zones: state.noCommentZones.map((z) => ({
						atSeconds: z.chunkIndex * (CHUNK_MS / 1000),
						reason: z.reason,
					})),
				};
				for (const sock of sockets) {
					this.sendWs(sock, failurePayload);
				}
			}
			// synthesisCompleted stays true (no duplicate), but needsSynthesis in DB stays true
			// so the deferred /synthesize endpoint can retry via the DO
		}
```

Also apply the same "loud" treatment to the two earlier early-return points
in the `try` block (`validationError !== null` and `artifact === null`),
since those are synthesis failures too and the spec requires the failure
review event on any unusable-artifact outcome, not only a thrown exception.
Replace:

```typescript
			if (validationError !== null) {
				console.error(
					JSON.stringify({
						level: "error",
						message: "v6 validation_error",
						error: validationError,
						sessionId: state.sessionId,
					}),
				);
				return; // leave needsSynthesis=true for retry
			}

			if (artifact === null) {
				console.error(
					JSON.stringify({
						level: "error",
						message: "v6 produced no artifact",
						sessionId: state.sessionId,
					}),
				);
				return;
			}
```
with:
```typescript
			if (validationError !== null || artifact === null) {
				console.error(
					JSON.stringify({
						level: "error",
						message:
							validationError !== null ? "v6 validation_error" : "v6 produced no artifact",
						error: validationError,
						sessionId: state.sessionId,
					}),
				);
				if (!state.isEvalSession) {
					const sockets = this.ctx.getWebSockets();
					const failurePayload = {
						type: "review" as const,
						verdict: null,
						carry_forward: null,
						no_comment_zones: state.noCommentZones.map((z) => ({
							atSeconds: z.chunkIndex * (CHUNK_MS / 1000),
							reason: z.reason,
						})),
					};
					for (const sock of sockets) {
						this.sendWs(sock, failurePayload);
					}
				}
				return; // leave needsSynthesis=true for retry
			}
```

- [ ] **Step 4: Run test — verify it PASSES, then re-run Group 0**

```bash
cd apps/api && bun run typecheck && bun run test -- --run session-brain.review-failure.test.ts
cd apps/api && bun run test -- --run session-brain.marks.test.ts
```
Expected: both PASS. This closes Group 0 — the Task 1 harness is now green.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.ts apps/api/src/do/session-brain.review-failure.test.ts && git commit -m "feat(#162): emit a loud review(verdict: null) on synthesis failure, close Group 0"
```

---

## Group C — existing test fixups (sequential; depend on Group B's finished shapes)

### Task 18: Delete `callMuqEndpoint`, `parseMuqResponse`, `MuqResult`, and the chroma fields from `inference.ts`

**Group:** C (depends on Group B)

**Behavior being verified:** `apps/api/src/services/inference.ts` exports
only AMT-related symbols; nothing in `apps/api/src` still imports the
deleted MuQ symbols (verified by `typecheck`, not a runtime test — this is a
deletion task).

**Files:**
- Modify: `apps/api/src/services/inference.ts`

- [ ] **Step 1: Confirm nothing outside this file still imports the symbols being deleted**

```bash
cd apps/api && grep -rn "callMuqEndpoint\|parseMuqResponse\|MuqResult\b" src --include="*.ts" | grep -v inference.ts
```
Expected: no output (Group B already removed every call site; Group C's
remaining tasks 19-22 fix the *test* files that still reference them, which
this task's deletion will break — do those next, in order).

- [ ] **Step 2: N/A (deletion, not new behavior).**

- [ ] **Step 3: Implement the deletion**

In `apps/api/src/services/inference.ts`, delete:
- The `MuqConfidences` interface stays (still imported by
  `confidence_gate.ts`, which is not in this plan's file-changes list and
  must keep compiling) — but its sibling `MuqResult` interface is deleted:
  ```typescript
  export interface MuqResult {
  	scores: MuqScores;
  	confidences: MuqConfidences | null;
  	chromaBytes: Uint8Array | null;
  	chromaFrames: number;
  	chromaFrameRateHz: number;
  }
  ```
- `MuqResponseRaw` interface (only consumer was `parseMuqResponse`)
- `MUQ_DIMS` constant (only consumer was `parseMuqResponse`)
- `callMuqEndpoint` function
- `encodeBase64`'s sibling chroma-decoding logic stays only inside
  `parseMuqResponse` — delete the whole `parseMuqResponse` function
- Keep `MuqScores` and `MuqConfidences` interfaces (still imported by
  `apps/api/src/services/confidence_gate.ts`, out of scope for this plan)
- Keep `fetchWithRetry`, `encodeBase64`, `callAmtEndpoint`,
  `AmtResult`/`AmtResponseRaw`/`PerfNote`/`PerfPedalEvent`, all unchanged.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run typecheck
```
Expected: FAIL at this point — `session-brain.canary.test.ts`,
`session-brain.concurrency.test.ts`, `session-brain.unit.test.ts`, and
`inference.test.ts` still reference the deleted symbols. That is expected;
Tasks 19-22 fix them next. Confirm the *only* errors are inside those four
test files (no production `src/` file breaks).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/inference.ts && git commit -m "chore(#162): delete callMuqEndpoint, parseMuqResponse, MuqResult, and chroma fields"
```

### Task 19: Fix `session-brain.concurrency.test.ts`

**Group:** C (depends on Task 18)

**Behavior being verified:** The N-concurrent-chunks regression test still
proves no chunk is dropped, now driven by AMT alone.

**Files:**
- Modify: `apps/api/src/do/session-brain.concurrency.test.ts`

- [ ] **Step 1-2: N/A (fixup, not new behavior) — proceed to Step 3.**

- [ ] **Step 3: Implement the fix**

Replace the mock block:

```typescript
vi.mock("../services/inference", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callMuqEndpoint: vi.fn(async () => {
			await new Promise((r) => setTimeout(r, 10));
			return {
				scores: {
					dynamics: 0.5,
					timing: 0.5,
					pedaling: 0.5,
					articulation: 0.5,
					phrasing: 0.5,
					interpretation: 0.5,
				},
				confidences: null,
				chromaBytes: null,
				chromaFrames: 0,
				chromaFrameRateHz: 0,
			};
		}),
		callAmtEndpoint: vi.fn(async () => {
			await new Promise((r) => setTimeout(r, 10));
			return { notes: [], pedalEvents: [] };
		}),
	};
});
```
with:
```typescript
vi.mock("../services/inference", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callAmtEndpoint: vi.fn(async () => {
			await new Promise((r) => setTimeout(r, 10));
			return { notes: [], pedalEvents: [] };
		}),
	};
});
```

(`notes: []` keeps every chunk a no-comment zone, which is fine — this test
only asserts `scoredChunks.length`/`chunksInFlight`, both of which the
Task 12 wiring still drives per-chunk regardless of no-comment-zone status;
`scoredChunks` is only pushed on the non-zone path, so update the assertion
to match — see below.)

Since Task 12's wiring only pushes `scoredChunks` on the non-no-comment-zone
path, and this mock returns zero notes (always a no-comment zone), the
original assertion `expect(after.scoredChunks.length).toBe(N)` no longer
holds. Give the mock >= `MIN_NOTES_FOR_COMMENT` notes instead, to keep the
test's original intent (every concurrent chunk contributes):

```typescript
		callAmtEndpoint: vi.fn(async () => {
			await new Promise((r) => setTimeout(r, 10));
			return {
				notes: Array.from({ length: 10 }, (_, i) => ({
					pitch: 60,
					onset: i * 0.3,
					offset: i * 0.3 + 0.2,
					velocity: 70,
				})),
				pedalEvents: [],
			};
		}),
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run session-brain.concurrency.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.concurrency.test.ts && git commit -m "fix(#162): drop MuQ mock from concurrency test, keep 10-note AMT stub"
```

### Task 20: Fix `session-brain.canary.test.ts`

**Group:** C (depends on Task 19)

**Behavior being verified:** All seven canary scenarios still hold under an
AMT-only pipeline; Scenario 4 ("all-MuQ-failure") becomes "all-AMT-failure",
and its assertion changes from "failed inference scores nothing" (the old
`scoredChunks.length === 0`) to the Task 12 no-comment-zone contract
(`inferenceFailures === N`, `noCommentZones.length === N`,
`scoredChunks.length === 0` still holds since AMT failure is always a
no-comment zone).

**Files:**
- Modify: `apps/api/src/do/session-brain.canary.test.ts`

- [ ] **Step 1-2: N/A (fixup, not new behavior) — proceed to Step 3.**

- [ ] **Step 3: Implement the fix**

Replace the mock block:

```typescript
vi.mock("../services/inference", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callMuqEndpoint: vi.fn(),
		callAmtEndpoint: vi.fn(),
	};
});

import { callAmtEndpoint, callMuqEndpoint } from "../services/inference";
```
with:
```typescript
vi.mock("../services/inference", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../services/inference")>();
	return {
		...actual,
		callAmtEndpoint: vi.fn(),
	};
});

import { callAmtEndpoint } from "../services/inference";
```

Replace:

```typescript
const muqMock = callMuqEndpoint as unknown as Mock;
const amtMock = callAmtEndpoint as unknown as Mock;

const tick = () => new Promise((r) => setTimeout(r, 10));

const okMuq = async () => {
	await tick();
	return {
		scores: {
			dynamics: 0.5,
			timing: 0.5,
			pedaling: 0.5,
			articulation: 0.5,
			phrasing: 0.5,
			interpretation: 0.5,
		},
		confidences: null,
		chromaBytes: null,
		chromaFrames: 0,
		chromaFrameRateHz: 0,
	};
};
const okAmt = async () => {
	await tick();
	return { notes: [], pedalEvents: [] };
};
```
with:
```typescript
const amtMock = callAmtEndpoint as unknown as Mock;

const tick = () => new Promise((r) => setTimeout(r, 10));

const okAmt = async () => {
	await tick();
	return {
		notes: Array.from({ length: 10 }, (_, i) => ({
			pitch: 60,
			onset: i * 0.3,
			offset: i * 0.3 + 0.2,
			velocity: 70,
		})),
		pedalEvents: [],
	};
};
```

In `describe("SessionBrain chunk_ready canary ...")`, replace the two
`muqMock.mockImplementation(okMuq);` lines (Scenario 1's setup and the
Scenario-4-restore line) — delete them (the mock no longer exists).

Replace Scenario 4 in full:

```typescript
			// ── Scenario 4: all-MuQ-failure concurrent — counter drains, no partial scoredChunks ──
			muqMock.mockImplementation(async () => {
				await tick();
				throw new Error("MuQ down");
			});
			{
				const N = 5;
				for (let i = 0; i < N; i++)
					await env.CHUNKS.put(`mf-${i}`, new Uint8Array([i + 1]));
				await scenario(async () => {
					await Promise.all(fireConcurrent(inst, "mf", N));
					const st = await readState(storage);
					expect(
						st.scoredChunks.length,
						"failed inference scores nothing",
					).toBe(0);
					expect(st.chunksInFlight, "failures still decrement").toBe(0);
					expect(st.inferenceFailures, "every failure counted").toBe(N);
				});
			}
			muqMock.mockImplementation(okMuq); // restore for remaining scenarios
```
with:
```typescript
			// ── Scenario 4: all-AMT-failure concurrent — counter drains, every chunk a no-comment zone ──
			amtMock.mockImplementation(async () => {
				await tick();
				throw new Error("AMT down");
			});
			{
				const N = 5;
				for (let i = 0; i < N; i++)
					await env.CHUNKS.put(`mf-${i}`, new Uint8Array([i + 1]));
				await scenario(async () => {
					await Promise.all(fireConcurrent(inst, "mf", N));
					const st = await readState(storage);
					expect(
						st.scoredChunks.length,
						"failed inference scores nothing",
					).toBe(0);
					expect(st.chunksInFlight, "failures still decrement").toBe(0);
					expect(st.inferenceFailures, "every failure counted").toBe(N);
					expect(
						(st as unknown as { noCommentZones: unknown[] }).noCommentZones.length,
						"every AMT failure is a no-comment zone",
					).toBe(N);
				});
			}
			amtMock.mockImplementation(okAmt); // restore for remaining scenarios
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run session-brain.canary.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.canary.test.ts && git commit -m "fix(#162): rewrite canary Scenario 4 for AMT-only failure + no-comment zones"
```

### Task 21: Fix `session-brain.unit.test.ts`

**Group:** C (depends on Task 20)

**Behavior being verified:** The file no longer imports the deleted
`parseMuqResponse`; the `chunk_bar_map` schema-shape test is left in place
(the type still exists, just unused by the emitter — Task 13 removed
emission, not the schema).

**Files:**
- Modify: `apps/api/src/do/session-brain.unit.test.ts`

- [ ] **Step 1-2: N/A (fixup, not new behavior) — proceed to Step 3.**

- [ ] **Step 3: Implement the fix**

Remove the import:

```typescript
import { parseMuqResponse } from "../services/inference";
```

Delete the entire `describe("parseMuqResponse chroma extraction", ...)`
block (lines 147-203 as read):

```typescript
describe("parseMuqResponse chroma extraction", () => {
	it("returns chromaBytes=null when response has no chroma_b64", () => {
		// ... (whole block)
	});
	// ...
});
```

Leave `describe("chunk_bar_map WebSocket message", ...)` untouched — it
tests only `wsOutgoingMessageSchema`, which still defines the type.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run session-brain.unit.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/do/session-brain.unit.test.ts && git commit -m "fix(#162): drop parseMuqResponse coverage from session-brain.unit.test.ts"
```

### Task 22: Fix `inference.test.ts`

**Group:** C (depends on Task 21)

**Behavior being verified:** The file no longer tests the deleted
`parseMuqResponse`; `callAmtEndpoint`'s base64-encoding coverage is
untouched.

**Files:**
- Modify: `apps/api/src/services/inference.test.ts`

- [ ] **Step 1-2: N/A (fixup, not new behavior) — proceed to Step 3.**

- [ ] **Step 3: Implement the fix**

Delete the `makeChromaB64` helper and the entire
`describe("parseMuqResponse chroma fields", ...)` block (lines 14-54 as
read). Leave `describe("callAmtEndpoint base64 encoding", ...)` untouched.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts inference.test.ts
cd apps/api && bun run test -- --run inference.test.ts
```
Expected: PASS on both (this file runs under both pools per the node
config's `src/services/**/*.test.ts` glob and the default workerd config's
lack of an include filter).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/inference.test.ts && git commit -m "fix(#162): drop parseMuqResponse coverage from inference.test.ts"
```

---

## Group D — web client (parallel with Group C; depends only on Group B's wire shapes)

### Task 23: `practice-api.ts` — add the `review` event to `PracticeWsEvent`

**Group:** D (parallel with Group C; depends on Task 10's wire shape)

**Behavior being verified:** `PracticeWsEvent` is a discriminated union that
type-checks a valid `review` message shape, matching the server's
`wsReviewSchema` from Task 10.

**Interface under test:** TypeScript compile-time check via `bun run
typecheck` (no runtime test exists for this file today — it holds only
types).

**Files:**
- Modify: `apps/web/src/lib/practice-api.ts`

- [ ] **Step 1: Confirm there is no existing runtime test to extend**

```bash
grep -rn "practice-api" apps/web/src --include="*.test.ts" --include="*.test.tsx"
```
Expected: no output — `practice-api.ts` has no dedicated test file; its
`PracticeWsEvent` union is exercised only through `usePracticeSession`'s own
tests (Task 24 adds one there). This task's verification is `bun run
typecheck` in `apps/web`.

- [ ] **Step 2: Run typecheck — verify it currently has nothing to check (baseline)**

```bash
cd apps/web && bun run typecheck
```
Expected: PASS (pre-change baseline — confirms the starting state compiles).

- [ ] **Step 3: Implement the change**

In `apps/web/src/lib/practice-api.ts`, import the `NoCommentZone` type used
by the review event's `no_comment_zones` field (already defined in
`apps/web/src/lib/session-review.ts`):

```typescript
import type { NoCommentZone } from "./session-review";
```

Add to the `PracticeWsEvent` union (after the `mark` variant):

```typescript
	| {
			type: "review";
			verdict: string | null;
			carry_forward: string | null;
			no_comment_zones: NoCommentZone[];
	  }
```

- [ ] **Step 4: Run typecheck — verify it PASSES**

```bash
cd apps/web && bun run typecheck
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/practice-api.ts && git commit -m "feat(#162): add review event to PracticeWsEvent"
```

### Task 24: `usePracticeSession.ts` — handle the `review` event

**Group:** D (parallel with Group C; depends on Task 23)

**Behavior being verified:** Receiving a `review` WS message sets the hook's
verdict/carry-forward/no-comment-zone state, exposed on the returned object,
without disturbing any other event handler.

**Interface under test:** `usePracticeSession()`'s returned
`UsePracticeSessionReturn`, driven by dispatching a `MessageEvent` through
`socket.onmessage` — the exact fake-WebSocket harness already established by
the existing sibling test `apps/web/src/hooks/usePracticeSession.marks.test.ts`
(`FakeAudioContext`/`FakeMediaRecorder`/`createFakeSocket`, mocking
`../lib/practice-api`, driving `start()` then firing `socket.onmessage?.(...)`).
This task's test file copies that harness verbatim rather than inventing a
new mechanism.

**Files:**
- Modify: `apps/web/src/hooks/usePracticeSession.ts`
- Test: `apps/web/src/hooks/usePracticeSession.review.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { practiceApi } from "../lib/practice-api";
import { usePracticeSession } from "./usePracticeSession";

vi.mock("../lib/practice-api", () => ({
	practiceApi: {
		start: vi.fn(),
		uploadChunk: vi.fn(),
		connectWebSocket: vi.fn(),
	},
}));

class FakeAudioContext {
	state = "running";
	createMediaStreamSource() {
		return { connect: vi.fn() };
	}
	createAnalyser() {
		return { fftSize: 256, getByteFrequencyData: vi.fn() };
	}
	close() {
		this.state = "closed";
		return Promise.resolve();
	}
}

class FakeMediaRecorder {
	state = "inactive";
	ondataavailable: ((e: { data: Blob }) => void) | null = null;
	start() {
		this.state = "recording";
	}
	stop() {
		this.state = "inactive";
	}
}

interface FakeSocket {
	readyState: number;
	onopen: (() => void) | null;
	onmessage: ((e: MessageEvent) => void) | null;
	onerror: (() => void) | null;
	onclose: (() => void) | null;
	send: ReturnType<typeof vi.fn>;
	close: ReturnType<typeof vi.fn>;
}

function createFakeSocket(): FakeSocket {
	return {
		readyState: WebSocket.OPEN,
		onopen: null,
		onmessage: null,
		onerror: null,
		onclose: null,
		send: vi.fn(),
		close: vi.fn(),
	};
}

describe("usePracticeSession — review event (#162)", () => {
	let socket: FakeSocket;

	beforeEach(() => {
		socket = createFakeSocket();
		vi.mocked(practiceApi.start).mockResolvedValue({
			sessionId: "s1",
			conversationId: "c1",
		});
		vi.mocked(practiceApi.connectWebSocket).mockImplementation(() => {
			queueMicrotask(() => socket.onopen?.());
			return socket as unknown as WebSocket;
		});
		vi.stubGlobal("AudioContext", FakeAudioContext);
		vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
		Object.defineProperty(navigator, "mediaDevices", {
			configurable: true,
			value: {
				getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [] }),
			},
		});
	});

	afterEach(() => {
		vi.unstubAllGlobals();
		vi.clearAllMocks();
	});

	it("sets verdict, carryForward, and noCommentZones from a review message", async () => {
		const { result } = renderHook(() => usePracticeSession());

		await act(async () => {
			await result.current.start();
		});
		await waitFor(() => expect(result.current.state).toBe("recording"));

		act(() => {
			socket.onmessage?.({
				data: JSON.stringify({
					type: "review",
					verdict: "Solid session, phrasing held together well.",
					carry_forward: "Watch the left-hand tempo in the coda.",
					no_comment_zones: [{ atSeconds: 45, reason: "insufficient_notes" }],
				}),
			} as MessageEvent);
		});

		expect(result.current.verdict).toBe(
			"Solid session, phrasing held together well.",
		);
		expect(result.current.carryForward).toBe(
			"Watch the left-hand tempo in the coda.",
		);
		expect(result.current.noCommentZones).toEqual([
			{ atSeconds: 45, reason: "insufficient_notes" },
		]);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- usePracticeSession.review.test.ts
```
Expected: FAIL — `result.current.verdict` is `undefined` (no such field
exists on `UsePracticeSessionReturn` yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/hooks/usePracticeSession.ts`, add imports:

```typescript
import type { NoCommentZone } from "../lib/session-review";
```

Add state declarations near `const [marks, setMarks] = useState<Mark[]>([]);`:

```typescript
	const [verdict, setVerdict] = useState<string | null>(null);
	const [carryForward, setCarryForward] = useState<string | null>(null);
	const [noCommentZones, setNoCommentZones] = useState<NoCommentZone[]>([]);
```

Add a case to the `switch (data.type)` block inside `handleWsMessage`, after
`case "mark":`:

```typescript
			case "review": {
				setVerdict(data.verdict);
				setCarryForward(data.carry_forward);
				setNoCommentZones(data.no_comment_zones);
				break;
			}
```

Add the three fields to `UsePracticeSessionReturn`:

```typescript
	verdict: string | null;
	carryForward: string | null;
	noCommentZones: NoCommentZone[];
```

Add them to the hook's returned object (after `marks,`):

```typescript
		verdict,
		carryForward,
		noCommentZones,
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- usePracticeSession.review.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/hooks/usePracticeSession.ts apps/web/src/hooks/usePracticeSession.review.test.ts && git commit -m "feat(#162): handle review event in usePracticeSession"
```

---

## Group E — cleanup (sequential; depends on Group C and Group D)

### Task 25: Delete `apps/inference/muq/`

**Group:** E (depends on Group C, Group D)

**Behavior being verified:** The directory no longer exists; nothing in the
repo references it (no build/deploy script targets it).

**Files:**
- Delete: `apps/inference/muq/` (entire directory)

- [ ] **Step 1: Confirm nothing references the directory before deleting**

```bash
grep -rn "apps/inference/muq" --include="*.toml" --include="*.json" --include="justfile" --include="*.md" . 2>/dev/null
```
Expected: review any hits (e.g. a `justfile` recipe or CI workflow) and
remove them alongside the directory in Step 3 if found; if none, proceed.

- [ ] **Step 2: N/A (deletion) — proceed to Step 3.**

- [ ] **Step 3: Delete**

```bash
git rm -r apps/inference/muq
```

- [ ] **Step 4: Verify the repo still builds**

```bash
cd apps/api && bun run typecheck
```
Expected: PASS (this directory was never part of the `apps/api` TS project).

- [ ] **Step 5: Commit**

```bash
git commit -m "chore(#162): delete apps/inference/muq/"
```

### Task 26: `docs/apps/02-pipeline.md` — resolve open questions 1-2, record the stage-4 deferral

**Group:** E (depends on Task 25)

**Behavior being verified:** The doc no longer lists MoonBeam serving and
the transcription-confidence proxy as open (they are resolved by this
issue's decisions), and stage 4 (offline alignment) is marked deferred to
the follower issue rather than described as live.

**Files:**
- Modify: `docs/apps/02-pipeline.md`

- [ ] **Step 1-2: N/A (doc update) — proceed to Step 3.**

- [ ] **Step 3: Implement the edit**

Replace the `## Open Questions` section's items 1 and 2:

```markdown
1. **MoonBeam serving.** Local vs. hosted inference for the encoder; cost
   and latency envelope per session (resolve in #162).
2. **Transcription confidence signal.** What Transkun exposes (or what
   proxy — note density sanity, alignment residual) defines a no-comment
   zone (resolve in #162).
```
with:
```markdown
1. **MoonBeam serving — RESOLVED 2026-08-15 (#162).** Not yet decided for a
   real head: this issue ships a deterministic pure-TS mock scorer
   (`apps/api/src/services/scoring.ts`) behind the exact interface the
   future MoonBeam container call will implement. The serving decision
   (local Docker container on the `apps/inference/amt` pattern) is deferred
   until the real head exists.
2. **Transcription confidence signal — RESOLVED 2026-08-15 (#162).** Note
   density is the proxy: fewer than `MIN_NOTES_FOR_COMMENT = 10` transcribed
   notes (or an AMT failure) makes a chunk a no-comment zone — no mark, no
   MPM claim, excluded from baseline updates, disclosed in the review.
```

Update the `### 4. Offline alignment` section's opening sentence:

```markdown
Chroma-DTW alignment maps chunks to bars, lagging real playing by seconds.
```
to:
```markdown
**Deferred to a follower issue (#162 STATE 2026-08-15).** Chroma-DTW
alignment maps chunks to bars, lagging real playing by seconds — but its
only input was MuQ's chroma output, which is removed with MuQ. Every
session runs the existing Tier-2 path; marks carry timestamp anchors until
the follower issue (entry criteria: #119 HMM matcher + #160) restores bar
anchoring behind the unchanged `BarMap` contract. The WASM aligner
(`alignChunkNotes`/`alignChunkChroma`, #64) stays in the tree, dormant.
```

- [ ] **Step 4: N/A (doc-only; no automated check beyond a read-through).**

- [ ] **Step 5: Commit**

```bash
git add docs/apps/02-pipeline.md && git commit -m "docs(#162): resolve pipeline open questions 1-2, record stage-4 deferral"
```

### Task 27: Final verification — grep gate, full typecheck, full test suite

**Group:** E (depends on Task 26 — last task)

**Behavior being verified:** The spec's automated checks all pass together:
zero MuQ references on the request path, zero typecheck errors, and the
full API test suite (both pools) green.

**Files:** none (verification only)

- [ ] **Step 1-3: N/A (verification, not new behavior).**

- [ ] **Step 4: Run the full verification suite**

```bash
cd apps/api && rg -il muq src wrangler.toml
```
Expected: no output (empty match set — the grep gate from the spec's
Verification Architecture).

```bash
cd apps/api && bun run typecheck
```
Expected: PASS, 0 errors.

```bash
cd apps/api && bun run test -- --run && bunx vitest run --config vitest.node.config.ts
```
Expected: PASS, 0 failures (this is exactly `just test-api`'s recipe body).

```bash
cd apps/web && bun run typecheck
```
Expected: PASS, 0 errors.

- [ ] **Step 5: No new commit — this task only verifies prior commits.** If
  any check fails, return to the task whose file caused it, fix it there,
  and re-run this task's verification from the top.

---

## Spec Requirement Coverage

| Spec requirement | Task |
|---|---|
| `callAmtEndpoint` is the only inference call; MuQ client code, `MUQ_ENDPOINT`, `apps/inference/muq/` deleted | 11, 12, 18, 25, 27 (grep gate) |
| Tier-1 branch + `chunk_bar_map` emission removed; every chunk takes Tier-2 | 13 |
| No-comment zone: AMT rejected OR <10 notes -> zone, no mark, excluded from samples/scoring | 12 |
| `scoreChunk` — deterministic pure-TS mock, stable interface | 2, 3, 12 |
| `extractMpmFeatures`/`evidenceFor` — MPM features + evidence prose, bar-scoped-ready interface | 4, 5, 14 |
| `proposeMarks` — band vs baseline or within-session median, firePersistence, dedupe, one mark per chunk, timestamp-only anchors | 6, 7, 14 |
| Mark emission: `{type: "mark", mark}` immediately per gate-passing candidate | 14 |
| `baseline-store.ts` — load/save `BaselineState` jsonb, loud `ValidationError` on corrupt rows | 8, 9 |
| `student_profiles.baseline_state` jsonb column | 10 (schema fields), migration step embedded in Task 10's follow-up — see note below |
| Session end: V6 synthesis unchanged; artifact mapped to `review` event; failure emits `review(verdict: null)` | 16, 17 |
| Baseline fold via `updateBaseline`, persisted via `saveBaselineState` | 16 |
| `observation` events only when `isEvalSession` | 15 |
| Web: `review` event in `PracticeWsEvent`; `usePracticeSession` handles it | 23, 24 |
| Rename `MUQ_CHUNK_MS` -> `CHUNK_MS` | 11 |
| `docs/apps/02-pipeline.md` open questions 1-2 resolved; stage-4 deferral recorded | 26 |
| Verification harness: end-to-end mark + review + baseline_state write, grep gate | 1 (harness), 27 (final run) |

**Note on the `student_profiles.baseline_state` migration:** the spec's file
list names `apps/api/src/db/migrations/0005_baseline_state.sql` as a new
file, but per `TS-DB-003` (never hand-edit `db/migrations/`), this plan does
not write that SQL file directly. **Task 8** (the first `baseline-store.ts`
task, Group A — the actual first consumer of the column) generates it via
`bun run generate -- --name=baseline_state` immediately before implementing
`baseline-store.ts`, and applies it to the local dev database. Task 10
(Group B) only adds the `SessionState` schema fields and outgoing WS
schemas — it does not touch `students.ts` or the migrations directory.

---

## Uncovered Spec Requirements

None. Every numbered design decision in the spec (Design items 1-7, the
Modules table, the Verification Architecture, the File Changes table, and
both Open Questions defaults) maps to at least one task above.

---

## Challenge Review

Reviewed against the actual current contents of `apps/api/src/do/session-brain.ts`
(2379 lines), `session-brain.schema.ts`, `services/student-baseline.ts`,
`services/inference.ts`, `lib/dims.ts`, `db/schema/students.ts`,
`apps/web/src/lib/{practice-api,mark,session-review}.ts`, and the migrations
journal — not from the plan's prose alone. Per the caller's framing, the mock
6-dim scorer, timestamp-only anchors (no chroma port), eval-only observation
retention, and pre-beta/zero-user posture are treated as settled #162
decisions, not findings.

### CEO Pass

**Premise / real pain:** Real. `mark`/`review` currently have zero server
producers (confirmed: `wsOutgoingMessageSchema` in `session-brain.schema.ts`
today only has `wsChunkBarMapSchema` — no mark, no review), and the DO
imports `callMuqEndpoint` from a dead HF endpoint on every chunk. The plan is
the direct route: rip MuQ, wire the three new pure modules, map V6 synthesis
to `review`.

**Existing coverage:** Correctly reused rather than reinvented —
`student-baseline.ts`'s `updateBaseline`/`DEFAULT_BASELINE_CONFIG` (verified:
`firePersistence: 3`, `minSamplesForSpread: 3`, `deviantSampleMultiple: 1.5`
all match what Task 6/7's `mark-gate.ts` assumes), `synthesis.ts`'s
`loadBaselinesFromDb(db, studentId)` shape precedent for `baseline-store.ts`,
and the existing `session-brain.concurrency.test.ts`/`canary.test.ts`
harness patterns for the new integration tests.

**12-month alignment / new finding:** The plan adds a *second*,
independently-shaped per-student baseline representation without addressing
the first. `student_profiles` already carries six flat columns
(`baselineDynamics` etc., `db/schema/students.ts:6-11`) read via
`loadBaselinesFromDb` and gating the *old* teaching-moment/observation path
(now eval-only per Task 15) — and, verified by grep, those columns are
written today only by `services/sync.ts` (an iOS/client sync path, out of
scope here), never by `session-brain.ts`'s own pipeline. This plan adds a
*third* representation (`baseline_state` jsonb, EWMA/lifecycle-based) that
now becomes the only one the server-side pipeline itself writes. Net result
after this ships: two parallel "what's normal for this student" stores exist
in the same table, one dead-write from the pipeline's own perspective, one
live — and nothing in the plan says so.
```
CURRENT STATE                  THIS PLAN                    12-MONTH IDEAL
6 flat baseline columns    ->  + baseline_state jsonb    ->  one baseline
(read-only from pipeline,      (read+write from pipeline,    representation,
externally written)            gates marks)                  flat columns
                                                               retired or
                                                               repurposed
```
```
[QUESTION] — Should this plan (or a fast follow-up) note in
             docs/apps/02-pipeline.md / architecture.md that
             `student_profiles.baseline_dynamics` etc. are now dead weight
             from the session pipeline's perspective, so a future reader
             does not assume they still gate anything for real sessions?
             Not a blocker — Task 26 already touches this doc for other
             reasons and could carry one sentence.
```

**Scope:** ~20 files touched, most of them one-per-behavior test files
required by the plan's own vertical-slice discipline — not a smell by
itself. The `apps/inference/muq/` deletion and `MUQ_CHUNK_MS` rename are
genuinely separable no-op-risk cleanup and could ship as Task 25/11 alone if
time-boxed, but the plan is honest that no subset is independently useful
(no server emits `mark`/`review` until Group C closes) — this is correct,
not scope-padding.

**Alternatives:** Spec's "Not in scope" section functions as the
alternatives-considered record (bar anchoring deferred with a named
reactivation path, `missed_opportunity` deferred with a named reason). This
satisfies the intent of the alternatives check even without a formal
2-3-option table.

### Engineering Pass

**Architecture / data flow — verified by reading the real code, not the
plan's paraphrase:**
```
handleChunkReady (unchanged: R2 fetch, 2 early-return neutral-score paths)
  -> Promise.allSettled([callAmtEndpoint])          [Task 11-12: MuQ leg removed]
  -> finalizeChunk (blockConcurrencyWhile)
       amtFailed || perfNotes.length < 10 ?
         -> noCommentZones.push, chunk_processed(0.5s)      [Task 12]
         : -> scoreChunk -> sessionSamples[dim].push,
              chunk_processed(scores), scoredChunks.push    [Task 12]
              -> markBaseline lazy-load once/session        [Task 14]
              -> proposeMarks -> mark WS event(s)           [Task 14]
       -> Tier-2-only bar analysis (unconditional)          [Task 13]
       -> mode detector, teaching-moment select
            (observation sent only if isEvalSession)        [Task 15]
       -> writeState (unconditional, end of function)
alarm() -> runSynthesisAndPersist
       -> synthesizeV6 success -> review{verdict, carry_forward} (non-eval)
                                -> updateBaseline + saveBaselineState (always)
       -> synthesizeV6 failure/no-artifact -> review{verdict: null} (non-eval) [Task 17]
```
This matches the plan's described edits closely, including the specific
line ranges it names (`finalizeChunk`'s MuQ-result block at lines ~769-829,
AMT-result block at ~831-851, the tier-1 branch at ~853-975, the
teaching-moment confidence-gate call at ~1221-1228, `runSynthesisAndPersist`
broadcast block, `db` already in scope there at line ~1764 before the
insertion point). The two early-return R2-failure paths in `handleChunkReady`
(lines 682-696, 708-721) are untouched by Task 12 and never reach
`finalizeChunk`, so they never populate `noCommentZones` — see RISK below.

**Verified math (not just read — hand-computed against Task 1's fixture):**
`scoreChunk`'s `dynamics = mean(velocities)/127` and the VELOCITIES sequence
`[62,66,63,65,64,10,10,10]` do produce a 3-consecutive-out-of-band streak in
the cold-start median/MAD band exactly at chunk index 7, firing
`dynamics`/`needs_work` at `atSeconds: 105` — the harness's assertions are
correct, not aspirational. `timing`/`pedaling`/`articulation`/`phrasing`
correctly stay constant (spread 0 -> `bandFor` returns null) given the
fixture's identical note shapes, exactly as the harness comment claims.

**Module depth:**
- `scoring.ts` — 1 exported fn, ~90 LOC of pure math hidden. DEEP.
- `mpm-features.ts` — 2 exported fns, ~150 LOC hidden (tempo/velocity/
  articulation/pedal decomposition + prose rendering). DEEP.
- `mark-gate.ts` — 1 exported fn, ~90 LOC hidden (band computation,
  persistence counting, dedupe). DEEP.
- `baseline-store.ts` — 2 exported fns, ~30 LOC, thin jsonb CRUD. The spec
  itself calls this "shallow-ish by necessity" and justifies it as the
  single owner of the persistence format — accepted, not flagged again here.

**Failure modes:** Zero silent failures confirmed for the new paths —
markBaseline load failure logs and falls through (mark gate just sees
`baseline: null`, degrades to cold-start, not silent data loss); baseline
fold/persist failure is caught and logged without blocking the review send;
synthesis failure now correctly reaches the client as
`review{verdict: null}` instead of vanishing (Task 17 closes a real
pre-existing silent-failure gap — the old code's outer `catch` only logged
and returned).

### Findings

```
[BLOCKER] (confidence: 7/10) — Task 13's own Step 2 admits the test it writes
    "passes both before and after this task's implementation edit" because
    seeding `pieceIdentification` without a real score-catalog row makes
    `loadScoreContext` throw and fall through to Tier-2 in both the pre- and
    post-change code. This is a self-acknowledged violation of the Iron Rule
    (a test covering changed behavior must fail before the change). The task
    substitutes `just check-api`/typecheck-catches-dangling-references as the
    real gate, which is a legitimate check but is not what the task claims to
    be doing. Fix: either seed a minimal fake score-catalog row so
    `loadScoreContext` actually succeeds and the tier-1 branch is genuinely
    reachable pre-change (making the test truly red then green), or rewrite
    the task's framing to "structural deletion, verified by typecheck + the
    Task 27 grep gate" and drop the pretense of a TDD red step.

[BLOCKER] (confidence: 7/10) — Task 15 Step 1 contains no test code. Every
    other task in this plan writes the literal failing test; this one
    instead instructs "the build agent implementing this task must adapt the
    seeding/mocking boilerplate already present ... and add one new it,"
    supplying only the final assertion line. This is the plan-writing
    equivalent of a "test scaffolding" placeholder the challenge skill
    treats as a horizontal-slice smell: it defers a real judgment call
    (which existing fixture already opens the `>= 2 && % TEACHING_MOMENT_
    INTERVAL === 0` teaching-moment gate, and with what baseline shape) to
    build time with nothing here to review it against. Read
    `session-brain.observation.test.ts` now and write the actual test into
    the plan before `/build` runs.
```

```
[RISK] (confidence: 6/10, mock-scorer-specific, revisit when the real head
    lands) — `scoreChunk`'s `interpretation = mean([dynamics, timing,
    pedaling, articulation, phrasing])` is an affine transform of the other
    five when only one varies. Hand-verified against Task 1's fixture:
    `interpretation`'s per-chunk samples equal exactly `dynamics/5 + 0.4`
    across all 8 chunks, so its median/MAD band opens and its 3-consecutive-
    out-of-band streak fires at the *same* chunk (index 7) as `dynamics`.
    The Group 0 harness only asserts `dynamics` fires because `DIMS_6` lists
    "dynamics" before "interpretation" and `proposeMarks` breaks after the
    first candidate — it is order-luck, not independence. In a real session
    where `dynamics` has already fired and is in `emittedDimensions`, a
    further dynamics drift will make `interpretation` independently satisfy
    the gate for what is actually the same root cause, producing a
    duplicate-looking mark under a different dimension label. Not a blocker
    (mock accuracy is an explicit #162 non-goal), but the mark-gate's
    implicit "dimensions are independent evidence" assumption should be
    re-examined once a real, presumably-correlated-but-not-literally-affine,
    scorer replaces this one.

[RISK] (confidence: 5/10) — The two early-return paths in `handleChunkReady`
    (R2 object not found, R2 fetch exception — lines 682-696 and 708-721 as
    they exist today) send a neutral `chunk_processed` and return *before*
    ever calling `finalizeChunk`, so they never push to `noCommentZones`.
    Pre-existing code, not introduced by this plan, but the spec's
    "disclosed no-comment zones" framing (`docs/apps/02-pipeline.md`,
    Design item 7) implies every chunk where nothing could be said should
    surface in the review. R2 failures are a real, if rare, silent gap in
    that guarantee. Not this plan's bug to fix; worth a one-line follow-up
    note (could ride with Task 26's doc edit) rather than silence.

[RISK] (confidence: 4/10) — `emittedMarks` is stored as `z.array(z.string())`
    (Task 10) and read back via `new Set(currentState.emittedMarks as
    Dimension[])` (Task 14) — an unchecked cast rather than
    `z.enum(DIMS_6)`. The sole writer is this same code today, so practical
    risk is low, but it's an avoidable gap against the project's stated
    "no `any`"/loud-failure preference, and a corrupted/hand-edited storage
    value would silently degrade dedup rather than throw. Cheap fix: change
    the schema field to `z.array(z.enum(DIMS_6))` and drop the cast.

[RISK] (confidence: 3/10) — `sendWs(ws, data: unknown)` never runs the new
    `wsMarkSchema`/`wsReviewSchema` (or the existing `wsOutgoingMessageSchema`
    union) against outgoing payloads at the send site — the schemas exist
    for the incoming test suite to validate against, not as a runtime
    contract on the hot path. Pre-existing looseness, not introduced here,
    and largely covered in practice by this plan's own behavioral tests
    asserting exact payload shape. Noting for completeness only.
```

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `DEFAULT_BASELINE_CONFIG` values (`firePersistence: 3`, `minSamplesForSpread: 3`, `deviantSampleMultiple: 1.5`) match what `mark-gate.ts` hand-codes as its own constants | SAFE | Verified: `mark-gate.ts` re-declares `MIN_SAMPLES_FOR_COLD_START_SPREAD = 3` / `COLD_START_BAND_MULTIPLE = 1.5` locally rather than importing from `student-baseline.ts`; values match today but there is no compile-time link forcing them to stay in sync if `DEFAULT_BASELINE_CONFIG` changes later. |
| `sendWs`'s `data: unknown` signature accepts the new `mark`/`review` payload shapes without a compile error | SAFE | Verified — no type constraint on `sendWs`'s second parameter. |
| `db` is in scope at the point Task 16 inserts the baseline-fold block inside `runSynthesisAndPersist` | SAFE | Verified: `const db = createDb(this.env.HYPERDRIVE)` at line ~1764, well before the synthesis broadcast block Task 16 edits. |
| `PracticeWsEvent` already has a `mark` variant and `NoCommentZone` is already exported from `session-review.ts` for Task 23 to import | SAFE | Verified directly in `apps/web/src/lib/practice-api.ts:87` and `session-review.ts:21`. |
| Drizzle-kit's `bun run generate -- --name=baseline_state` produces exactly `0005_baseline_state.sql` | VALIDATE | Journal's last entry is idx 4 (`0004_pending_exercises_routing`), so idx 5 is next, but the migrations directory already has two files claiming index 3 (`0003_gorgeous_sumo.sql` and `0003_segment_loops.sql`) with only one idx-3 journal entry — a pre-existing discipline gap unrelated to this plan, but it means "assume the next generated file is well-behaved" is not a given in this repo's migration history. Not a blocker; just don't assume Task 8's generate step is risk-free to skip verifying. |
| Two independently-shaped per-student baseline stores (`baseline_dynamics` flat columns vs. new `baseline_state` jsonb) coexisting is acceptable for this issue | RISKY (accepted as out of scope) | See CEO Pass 12-month alignment finding — real, but explicitly not this plan's job to resolve. |

### Summary

[BLOCKER] count: 2
[RISK]    count: 4
[QUESTION] count: 1

VERDICT: NEEDS_REWORK — resolve Task 13 (self-admitted non-red test for the
tier-1 removal) and Task 15 (missing test code, deferred to build time)
before `/build` runs. Both are narrow, mechanical fixes — write the actual
test bodies into the plan — not a redesign; every other finding is a RISK or
QUESTION that can be monitored or answered inline during `/build` without
blocking the start of execution once the two BLOCKERs are addressed.

### Blocker resolution (loop 1)

- **Task 13:** Dropped the red/green pretense instead of seeding a fake
  score-catalog row. Step 1 is now labeled "Write the regression-guard test"
  and states up front that this is a structural deletion with no reachable
  pre-change red state. Step 2 is now "Run test — confirm it already PASSES
  (not a red step)" and explains honestly that the test passes before and
  after Step 3's edit because `loadScoreContext` throws without a real
  catalog row in both cases; the real proof of deletion is `bun run
  typecheck` (Step 4, catches dangling references to the removed tier-1
  locals) plus Task 27's grep gate. No code/scope change, wording only.
- **Task 15:** Replaced the placeholder ("build agent must adapt... add one
  new `it`") with the complete literal test. Read
  `apps/api/src/do/session-brain.observation.test.ts` and confirmed it has
  no DO-integration fixtures to reuse (it only unit-tests
  `buildObservationPayload` with plain `vitest`) — the plan's premise that
  such fixtures existed there was wrong. The real convention lives in this
  task's Group-B siblings (Tasks 12-14's test files: `runInDurableObject` +
  mocked `callAmtEndpoint` + a `sent`-array `fakeWs`), so Step 1 now adds
  that convention to the file verbatim: a `vi.mock` for
  `../services/inference`, the `SESSION_BRAIN`/`CHUNKS` `ProvidedEnv`
  declaration, and a new `describe` block that seeds two >=10-note chunks
  with non-null `baselines` and asserts no `observation` message is ever
  sent while `isEvalSession` is false. Verified against
  `apps/api/src/wasm/score-analysis/src/teaching_moments.rs` that
  `select_teaching_moment` returns `Some` unconditionally once
  `chunks.len() >= MIN_CHUNKS (2)` (falls back to a positive moment when
  nothing is below baseline; only returns `None` below `MIN_CHUNKS`), so the
  seeded two-chunk state guarantees a moment is selected and the test is a
  genuine, non-vacuous exercise of the `isEvalSession` gate.
