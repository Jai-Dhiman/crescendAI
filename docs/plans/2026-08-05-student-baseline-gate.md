# Student Baseline Gate Implementation Plan

> **For the build agent:** Dispatch each task in sequence (all tasks touch the
> same two files, so there is no parallel group). Do NOT start execution until
> `/challenge` returns VERDICT: PROCEED.

**Goal:** Decide, per dimension, whether a deviation in a student's playing is
worth marking — firing on repeated evidence, staying quiet on single-observation
noise, and retiring symmetrically when the deviation persistently returns to
normal.
**Spec:** `docs/specs/2026-08-05-student-baseline-gate-design.md`
**Style:** `apps/api/TS_STYLE.md` is normative. No `any` (TS-TYPE-001);
services are stateless pure functions (TS-SVC-002); services never import
`HTTPException` (TS-ERR-001); tabs for indentation, double quotes (biome).

---

## Every task's verification command

Unless a task states otherwise, verify with:

```bash
cd apps/api && bun run test:scripts -- student-baseline
```

This runs `vitest --config vitest.node.config.ts student-baseline`, which
matches `src/services/**/*.test.ts` under Node — the fast loop for this
module. The final task additionally runs the workerd pool, typecheck, and
lint.

## A note on "crude, then refined" steps

Several pairs of tasks below share one mechanism (e.g. "fires on repeated
evidence" and "stays quiet on a single observation" are the same threshold
comparison read two ways). To keep every task's Step 1 test genuinely
failing before its Step 3 implementation — never trivially passing — some
tasks land a **deliberately narrow** implementation (e.g. "any deviant sample
fires immediately," no threshold), and the very next task's test exploits
that narrowness to force the real, general mechanism. Each such task's Step 3
says so explicitly in a comment, and the plan calls out which later task
replaces it. This is standard TDD triangulation, not a shortcut: the code
that ships at the end of the sequence is the same either way, and every step
in between is independently tested and committed.

---

## Task 0: `runSequence` harness fails before the module exists

**Group:** sequential (first)

**Behavior being verified:** a synthetic-sequence test harness exists and can
call `updateBaseline`/`initialBaselineState` — before either exists.
**Interface under test:** `updateBaseline`, `initialBaselineState` (imported,
not yet defined).

**Files:**
- Create: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, expect, it } from "vitest";
import {
	type BaselineState,
	type SessionSamples,
	initialBaselineState,
	updateBaseline,
} from "./student-baseline";

function runSequence(sessions: SessionSamples[]): BaselineState[] {
	const trace: BaselineState[] = [];
	let state = initialBaselineState();
	for (const session of sessions) {
		state = updateBaseline(state, session);
		trace.push(state);
	}
	return trace;
}

// Symmetric jitter (+/-0.01 around 0.5, equal counts each side) so the median
// absolute deviation equals every point's own deviation -- no point exceeds
// DEVIANT_SAMPLE_MULTIPLE * MAD, so a clean cluster never self-triggers.
const CLUSTER = [0.49, 0.51, 0.49, 0.51, 0.49, 0.51];

describe("runSequence harness", () => {
	it("folds sessions through updateBaseline and traces lifecycle", () => {
		const trace = runSequence([
			{ timestamp: "2026-01-01T00:00:00Z", scores: { pedaling: CLUSTER } },
		]);
		expect(trace).toHaveLength(1);
		expect(trace[0].dimensions.pedaling.lifecycle).toBe("absent");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — module resolution error, e.g. `Failed to resolve import
"./student-baseline" from "src/services/student-baseline.test.ts"` /
`Cannot find module './student-baseline'`. `apps/api/src/services/student-baseline.ts`
does not exist yet. This is the required "harness fails before the feature"
proof — a harness that passes here would prove nothing.

- [ ] **Step 3: Implement the minimum to make the test pass**

This step is intentionally deferred to Task 1, which creates
`student-baseline.ts` for the first time. Task 0 has no Step 3 of its own —
its only job is to prove the harness fails first. Do not create any file in
this task.

- [ ] **Step 4: Run test — verify it still FAILS (unchanged)**

Same command, same failure. Confirms Task 0 introduced no implementation.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.test.ts && git commit -m "test(baseline): add runSequence harness, failing on missing module"
```

---

## Task 1: Cold start — same call shape as any other session, and the state round-trips through JSON

**Group:** sequential (depends on Task 0)

**Behavior being verified:** `updateBaseline` never returns `null`; a fresh
baseline behaves exactly like any other session; the returned `BaselineState`
is plain JSON that re-validates against `BaselineStateSchema`.
**Interface under test:** `initialBaselineState`, `updateBaseline`,
`BaselineStateSchema`.

**Files:**
- Create: `apps/api/src/services/student-baseline.ts`
- Modify: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

Add to `student-baseline.test.ts`, and add `BaselineStateSchema` to the
existing import from `"./student-baseline"`:

```typescript
import {
	type BaselineState,
	BaselineStateSchema,
	type SessionSamples,
	initialBaselineState,
	updateBaseline,
} from "./student-baseline";
```

```typescript
describe("cold start", () => {
	it("uses the same call shape as any other session, and round-trips through JSON", () => {
		const state = initialBaselineState();
		const result = updateBaseline(state, {
			timestamp: "2026-01-01T00:00:00Z",
			scores: { phrasing: CLUSTER },
		});
		expect(result.dimensions.phrasing.lifecycle).toBe("absent");
		expect(typeof result.dimensions.phrasing.noiseFloor).toBe("number");
		const roundTripped = JSON.parse(JSON.stringify(result));
		expect(() => BaselineStateSchema.parse(roundTripped)).not.toThrow();
		expect(roundTripped).toEqual(result);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — same module-resolution error as Task 0 (`student-baseline.ts`
still does not exist).

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `apps/api/src/services/student-baseline.ts`:

```typescript
import { z } from "zod";
import { DIMS_6, type Dimension } from "../lib/dims";

// ---------------------------------------------------------------------------
// student-baseline — the single gate deciding whether a deviation in a
// student's playing is worth marking. See docs/specs/2026-08-05-student-
// baseline-gate-design.md for the full design rationale.
// ---------------------------------------------------------------------------

export type Lifecycle = "absent" | "active" | "improving" | "resolved";

export interface SessionSamples {
	/** ISO 8601 timestamp for this session. */
	timestamp: string;
	/** Raw per-dimension sample scores observed during this session. */
	scores: Partial<Record<Dimension, readonly number[]>>;
}

export interface BaselineConfig {
	shortHalfLifeSessions: number;
	longHalfLifeSessions: number;
	minBandSdFraction: number;
	firePersistence: number;
	improvingPersistence: number;
	retirePersistence: number;
	promotionDistinctWeeks: number;
	maxWithinSessionContribution: number;
	minSamplesForSpread: number;
	deviantSampleMultiple: number;
}

export const DEFAULT_BASELINE_CONFIG: BaselineConfig = {
	shortHalfLifeSessions: 4,
	longHalfLifeSessions: 20,
	minBandSdFraction: 0.2,
	firePersistence: 3,
	improvingPersistence: 2,
	retirePersistence: 3,
	promotionDistinctWeeks: 2,
	maxWithinSessionContribution: 3,
	minSamplesForSpread: 3,
	deviantSampleMultiple: 1.5,
};

const DimensionStateSchema = z.object({
	lifecycle: z.enum(["absent", "active", "improving", "resolved"]),
	longMean: z.number(),
	longSd: z.number(),
	shortMean: z.number(),
	noiseFloor: z.number(),
	consecutiveOutOfBand: z.number().int().min(0),
	consecutiveInBand: z.number().int().min(0),
	promoted: z.boolean(),
	evidenceWeeks: z.array(z.string()),
	initialized: z.boolean(),
	updateCount: z.number().int().min(0),
});

// An explicit object (one key per DIMS_6 entry), not z.record: z.record infers
// values as possibly-undefined on access, which would force every caller and
// every internal read to null-check a key that this module always populates.
export const BaselineStateSchema = z.object({
	lastSessionTimestamp: z.string().nullable(),
	dimensions: z.object({
		dynamics: DimensionStateSchema,
		timing: DimensionStateSchema,
		pedaling: DimensionStateSchema,
		articulation: DimensionStateSchema,
		phrasing: DimensionStateSchema,
		interpretation: DimensionStateSchema,
	}),
});

export type BaselineState = z.infer<typeof BaselineStateSchema>;
export type DimensionBaselineState = z.infer<typeof DimensionStateSchema>;

/** A fresh baseline: every dimension absent, no evidence folded in yet. */
export function initialBaselineState(): BaselineState {
	const dimensions = {} as Record<Dimension, DimensionBaselineState>;
	for (const dim of DIMS_6) {
		dimensions[dim] = {
			lifecycle: "absent",
			longMean: 0,
			longSd: 0,
			shortMean: 0,
			noiseFloor: 0,
			consecutiveOutOfBand: 0,
			consecutiveInBand: 0,
			promoted: false,
			evidenceWeeks: [],
			initialized: false,
			updateCount: 0,
		};
	}
	return { lastSessionTimestamp: null, dimensions };
}

/**
 * Pure fold: (state, session) -> state. No clock, no randomness, no I/O.
 * NOTE: this is a deliberately minimal first cut -- it establishes the call
 * shape and return type only. It does not yet validate input (Tasks 2-5) or
 * fold any evidence (Tasks 6+); every dimension simply passes through
 * unchanged.
 */
export function updateBaseline(
	state: BaselineState,
	session: SessionSamples,
	_config: BaselineConfig = DEFAULT_BASELINE_CONFIG,
): BaselineState {
	return { lastSessionTimestamp: session.timestamp, dimensions: state.dimensions };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: PASS (2 tests: the Task 0 harness test and this task's cold-start
test).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.ts apps/api/src/services/student-baseline.test.ts && git commit -m "feat(baseline): add BaselineState schema and pass-through updateBaseline"
```

---

## Task 2: Throws on an unknown dimension

**Group:** sequential (depends on Task 1)

**Behavior being verified:** `updateBaseline` throws rather than silently
ignoring a session keyed by a dimension outside `DIMS_6`.
**Interface under test:** `updateBaseline`.

**Files:**
- Modify: `apps/api/src/services/student-baseline.ts`
- Modify: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
describe("explicit failures", () => {
	it("throws on an unknown dimension", () => {
		const state = initialBaselineState();
		expect(() =>
			updateBaseline(state, {
				timestamp: "2026-01-01T00:00:00Z",
				scores: { not_a_dimension: CLUSTER } as never,
			}),
		).toThrow(/unknown dimension/);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — `updateBaseline` never throws today (Task 1's implementation
is a pure pass-through), so `expect(...).toThrow(...)` fails with
`AssertionError: expected [Function] to throw error matching /unknown
dimension/ but it didn't throw at all` (approximately; exact vitest wording
may vary, but no throw occurs).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `student-baseline.ts`, add before `initialBaselineState` is fine, but the
convention here is directly above `updateBaseline`:

```typescript
function validateSession(state: BaselineState, session: SessionSamples): void {
	for (const dimension of Object.keys(session.scores)) {
		if (!(DIMS_6 as readonly string[]).includes(dimension)) {
			throw new Error(`updateBaseline: unknown dimension "${dimension}"`);
		}
	}
}
```

Change `updateBaseline`'s body to call it first:

```typescript
export function updateBaseline(
	state: BaselineState,
	session: SessionSamples,
	_config: BaselineConfig = DEFAULT_BASELINE_CONFIG,
): BaselineState {
	validateSession(state, session);
	return { lastSessionTimestamp: session.timestamp, dimensions: state.dimensions };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.ts apps/api/src/services/student-baseline.test.ts && git commit -m "feat(baseline): throw on an unknown dimension"
```

---

## Task 3: Throws on a non-finite score

**Group:** sequential (depends on Task 2)

**Behavior being verified:** `updateBaseline` throws on `NaN`/`Infinity`
sample scores instead of silently folding them in.
**Interface under test:** `updateBaseline`.

**Files:**
- Modify: `apps/api/src/services/student-baseline.ts`
- Modify: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

Add inside the existing `describe("explicit failures", ...)` block:

```typescript
	it("throws on a non-finite score", () => {
		const state = initialBaselineState();
		expect(() =>
			updateBaseline(state, {
				timestamp: "2026-01-01T00:00:00Z",
				scores: { pedaling: [0.5, Number.NaN, 0.5] },
			}),
		).toThrow(/non-finite/);
	});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — `validateSession` only checks dimension names today; a
`NaN` score passes through silently, so `toThrow(/non-finite/)` fails with no
throw observed.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `validateSession` in `student-baseline.ts`:

```typescript
function validateSession(state: BaselineState, session: SessionSamples): void {
	for (const [dimension, samples] of Object.entries(session.scores)) {
		if (!(DIMS_6 as readonly string[]).includes(dimension)) {
			throw new Error(`updateBaseline: unknown dimension "${dimension}"`);
		}
		for (const score of samples ?? []) {
			if (!Number.isFinite(score)) {
				throw new Error(
					`updateBaseline: non-finite score ${score} for dimension "${dimension}"`,
				);
			}
		}
	}
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.ts apps/api/src/services/student-baseline.test.ts && git commit -m "feat(baseline): throw on a non-finite score"
```

---

## Task 4: Throws on an unparseable timestamp

**Group:** sequential (depends on Task 3)

**Behavior being verified:** `updateBaseline` throws when `session.timestamp`
cannot be parsed as a date.
**Interface under test:** `updateBaseline`.

**Files:**
- Modify: `apps/api/src/services/student-baseline.ts`
- Modify: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

Add inside `describe("explicit failures", ...)`:

```typescript
	it("throws on an unparseable timestamp", () => {
		const state = initialBaselineState();
		expect(() =>
			updateBaseline(state, {
				timestamp: "not-a-date",
				scores: { pedaling: CLUSTER },
			}),
		).toThrow(/unparseable timestamp/);
	});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — `validateSession` never inspects the timestamp today, so no
throw occurs; `toThrow(/unparseable timestamp/)` fails.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add a timestamp check at the top of `validateSession`:

```typescript
function validateSession(state: BaselineState, session: SessionSamples): void {
	const timestampMs = Date.parse(session.timestamp);
	if (Number.isNaN(timestampMs)) {
		throw new Error(
			`updateBaseline: unparseable timestamp "${session.timestamp}"`,
		);
	}
	for (const [dimension, samples] of Object.entries(session.scores)) {
		if (!(DIMS_6 as readonly string[]).includes(dimension)) {
			throw new Error(`updateBaseline: unknown dimension "${dimension}"`);
		}
		for (const score of samples ?? []) {
			if (!Number.isFinite(score)) {
				throw new Error(
					`updateBaseline: non-finite score ${score} for dimension "${dimension}"`,
				);
			}
		}
	}
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.ts apps/api/src/services/student-baseline.test.ts && git commit -m "feat(baseline): throw on an unparseable session timestamp"
```

---

## Task 5: Throws on an out-of-order session timestamp

**Group:** sequential (depends on Task 4)

**Behavior being verified:** `updateBaseline` throws when a session's
timestamp precedes the last folded session's timestamp.
**Interface under test:** `updateBaseline`.

**Files:**
- Modify: `apps/api/src/services/student-baseline.ts`
- Modify: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

Add inside `describe("explicit failures", ...)`:

```typescript
	it("throws on an out-of-order session timestamp", () => {
		const state = updateBaseline(initialBaselineState(), {
			timestamp: "2026-01-05T00:00:00Z",
			scores: { pedaling: CLUSTER },
		});
		expect(() =>
			updateBaseline(state, {
				timestamp: "2026-01-01T00:00:00Z",
				scores: { pedaling: CLUSTER },
			}),
		).toThrow(/precedes/);
	});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — `validateSession` never compares against
`state.lastSessionTimestamp` today, so no throw occurs; `toThrow(/precedes/)`
fails.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add the ordering check to `validateSession`, right after the unparseable-
timestamp check:

```typescript
function validateSession(state: BaselineState, session: SessionSamples): void {
	const timestampMs = Date.parse(session.timestamp);
	if (Number.isNaN(timestampMs)) {
		throw new Error(
			`updateBaseline: unparseable timestamp "${session.timestamp}"`,
		);
	}
	if (state.lastSessionTimestamp !== null) {
		const lastMs = Date.parse(state.lastSessionTimestamp);
		if (timestampMs < lastMs) {
			throw new Error(
				`updateBaseline: session timestamp ${session.timestamp} precedes last folded session ${state.lastSessionTimestamp}`,
			);
		}
	}
	for (const [dimension, samples] of Object.entries(session.scores)) {
		if (!(DIMS_6 as readonly string[]).includes(dimension)) {
			throw new Error(`updateBaseline: unknown dimension "${dimension}"`);
		}
		for (const score of samples ?? []) {
			if (!Number.isFinite(score)) {
				throw new Error(
					`updateBaseline: non-finite score ${score} for dimension "${dimension}"`,
				);
			}
		}
	}
}
```

`validateSession` is now complete for the rest of this plan; no further task
modifies it.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.ts apps/api/src/services/student-baseline.test.ts && git commit -m "feat(baseline): throw on an out-of-order session timestamp"
```

---

## Task 6: Session 1 fires from repeated within-session evidence

**Group:** sequential (depends on Task 5)

**Behavior being verified:** a single sitting with >=3 samples that deviate
from that session's own centre is enough evidence to fire, with no history.
**Interface under test:** `updateBaseline`.

**Files:**
- Modify: `apps/api/src/services/student-baseline.ts`
- Modify: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
const CLUSTER_WITH_3_OUTLIERS = [
	0.49, 0.51, 0.49, 0.51, 0.49, 0.51, 0.1, 0.1, 0.1,
];

describe("session 1 within-session evidence", () => {
	it("fires immediately from >=3 deviant samples in the first sitting", () => {
		const trace = runSequence([
			{
				timestamp: "2026-01-01T00:00:00Z",
				scores: { pedaling: CLUSTER_WITH_3_OUTLIERS },
			},
		]);
		expect(trace[0].dimensions.pedaling.lifecycle).toBe("active");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — `updateBaseline` never inspects sample values yet (Task 1's
pass-through), so `lifecycle` stays `"absent"`: `AssertionError: expected
'absent' to be 'active'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add `median`/`medianAbsoluteDeviation` helpers and a `foldDimension` function
to `student-baseline.ts` (median absolute deviation is robust to the
minority-outlier case this gate targets — a plain population standard
deviation would be inflated by the very outliers it's trying to catch):

```typescript
function median(values: readonly number[]): number {
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 === 0
		? (sorted[mid - 1] + sorted[mid]) / 2
		: sorted[mid];
}

/** Median absolute deviation — robust to the minority-outlier case this gate targets. */
function medianAbsoluteDeviation(
	values: readonly number[],
	centre: number,
): number {
	return median(values.map((v) => Math.abs(v - centre)));
}

/**
 * NOTE: deliberately crude for this task -- any deviant sample fires
 * immediately, ignoring persistence. Task 7 replaces this with the real
 * FIRE_PERSISTENCE-threshold counter mechanism.
 */
function foldDimension(
	prior: DimensionBaselineState,
	samples: readonly number[],
	config: BaselineConfig,
): DimensionBaselineState {
	const sessionCentre = median(samples);
	let withinSessionDeviants = 0;
	if (samples.length >= config.minSamplesForSpread) {
		const mad = medianAbsoluteDeviation(samples, sessionCentre);
		if (mad > 0) {
			const threshold = config.deviantSampleMultiple * mad;
			for (const s of samples) {
				if (Math.abs(s - sessionCentre) > threshold) withinSessionDeviants += 1;
			}
		}
	}
	const lifecycle = withinSessionDeviants > 0 ? "active" : prior.lifecycle;
	return { ...prior, lifecycle };
}
```

Update `updateBaseline` to fold each dimension's samples (this also renames
the unused `_config` parameter to `config`, now that it is used):

```typescript
export function updateBaseline(
	state: BaselineState,
	session: SessionSamples,
	config: BaselineConfig = DEFAULT_BASELINE_CONFIG,
): BaselineState {
	validateSession(state, session);
	const dimensions = { ...state.dimensions };
	for (const [dimension, samples] of Object.entries(session.scores)) {
		if (!samples || samples.length === 0) continue;
		const dim = dimension as Dimension;
		dimensions[dim] = foldDimension(dimensions[dim], samples, config);
	}
	return { lastSessionTimestamp: session.timestamp, dimensions };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: PASS (7 tests). Confirm no regression: the cold-start test (Task 1,
`CLUSTER` has 0 deviants since every point's deviation equals the MAD
exactly, never exceeding it) still reports `"absent"`.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.ts apps/api/src/services/student-baseline.test.ts && git commit -m "feat(baseline): fire on within-session deviant samples (crude threshold)"
```

---

## Task 7: A single deviant observation stays quiet

**Group:** sequential (depends on Task 6)

**Behavior being verified:** one deviant sample alone must not fire — this
forces the real persistence-counter mechanism, replacing Task 6's crude
"any deviant fires" rule.
**Interface under test:** `updateBaseline`.

**Files:**
- Modify: `apps/api/src/services/student-baseline.ts`
- Modify: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

Add inside `describe("session 1 within-session evidence", ...)`:

```typescript
const CLUSTER_WITH_1_OUTLIER = [0.49, 0.51, 0.49, 0.51, 0.49, 0.51, 0.1];
```

```typescript
	it("stays quiet on a single deviant observation", () => {
		const trace = runSequence([
			{
				timestamp: "2026-01-01T00:00:00Z",
				scores: { pedaling: CLUSTER_WITH_1_OUTLIER },
			},
		]);
		expect(trace[0].dimensions.pedaling.lifecycle).toBe("absent");
	});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — Task 6's crude rule fires on any deviant count > 0. With 1
deviant sample, `withinSessionDeviants = 1 > 0`, so `lifecycle` becomes
`"active"`: `AssertionError: expected 'active' to be 'absent'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `foldDimension` in `student-baseline.ts` with the real persistence
counter (`consecutiveOutOfBand`/`consecutiveInBand`), capped and thresholded
per the spec's `MAX_WITHIN_SESSION_CONTRIBUTION` and `FIRE_PERSISTENCE`:

```typescript
function foldDimension(
	prior: DimensionBaselineState,
	samples: readonly number[],
	config: BaselineConfig,
): DimensionBaselineState {
	const sessionCentre = median(samples);
	let withinSessionDeviants = 0;
	if (samples.length >= config.minSamplesForSpread) {
		const mad = medianAbsoluteDeviation(samples, sessionCentre);
		if (mad > 0) {
			const threshold = config.deviantSampleMultiple * mad;
			for (const s of samples) {
				if (Math.abs(s - sessionCentre) > threshold) withinSessionDeviants += 1;
			}
		}
		withinSessionDeviants = Math.min(
			withinSessionDeviants,
			config.maxWithinSessionContribution,
		);
	}

	const contribution = withinSessionDeviants;
	let consecutiveOutOfBand = prior.consecutiveOutOfBand;
	let consecutiveInBand = prior.consecutiveInBand;
	if (contribution > 0) {
		consecutiveOutOfBand += contribution;
		consecutiveInBand = 0;
	} else {
		consecutiveInBand += 1;
		consecutiveOutOfBand = 0;
	}

	let lifecycle = prior.lifecycle;
	if (lifecycle === "absent" && consecutiveOutOfBand >= config.firePersistence) {
		lifecycle = "active";
	}

	return { ...prior, lifecycle, consecutiveOutOfBand, consecutiveInBand };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: PASS (8 tests). Confirm no regression: Task 6's 3-outlier test
still fires (`withinSessionDeviants` capped at 3 == `firePersistence`, so
`consecutiveOutOfBand` reaches 3 in one session).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.ts apps/api/src/services/student-baseline.test.ts && git commit -m "feat(baseline): gate within-session firing on the real persistence counter"
```

---

## Task 8: Persistent across-session deviation fires

**Group:** sequential (depends on Task 7)

**Behavior being verified:** a deviation that holds across
`FIRE_PERSISTENCE` (3) consecutive sessions fires, even with zero
within-session deviants in any of them.
**Interface under test:** `updateBaseline`.

**Files:**
- Modify: `apps/api/src/services/student-baseline.ts`
- Modify: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
describe("across-session evidence", () => {
	it("fires on persistent deviation across FIRE_PERSISTENCE sessions", () => {
		const shifted = [0.79, 0.81, 0.79, 0.81, 0.79, 0.81];
		const trace = runSequence([
			{ timestamp: "2026-01-01T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-02T00:00:00Z", scores: { dynamics: shifted } },
			{ timestamp: "2026-01-03T00:00:00Z", scores: { dynamics: shifted } },
			{ timestamp: "2026-01-04T00:00:00Z", scores: { dynamics: shifted } },
		]);
		expect(trace[0].dimensions.dynamics.lifecycle).toBe("absent");
		expect(trace[1].dimensions.dynamics.lifecycle).toBe("absent");
		expect(trace[2].dimensions.dynamics.lifecycle).toBe("absent");
		expect(trace[3].dimensions.dynamics.lifecycle).toBe("active");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — `foldDimension` only ever looks at within-session deviants
today; `shifted` has zero within-session deviants (it is a tight symmetric
cluster), so `consecutiveOutOfBand` never moves and `lifecycle` stays
`"absent"` at every step: `AssertionError: expected 'absent' to be 'active'`
at `trace[3]`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add `alphaFromHalfLife`/`ewma` helpers, and extend `foldDimension` with a
**deliberately crude** across-session signal: any nonzero gap between this
session's own centre and the long-run mean counts as out-of-band, with no
band width yet (that arrives in Tasks 9-10):

```typescript
function alphaFromHalfLife(halfLifeSessions: number): number {
	return 1 - 2 ** (-1 / halfLifeSessions);
}

function ewma(
	prev: number,
	value: number,
	alpha: number,
	initialized: boolean,
): number {
	if (!initialized) return value;
	return prev + alpha * (value - prev);
}
```

```typescript
function foldDimension(
	prior: DimensionBaselineState,
	samples: readonly number[],
	config: BaselineConfig,
): DimensionBaselineState {
	const alphaLong = alphaFromHalfLife(config.longHalfLifeSessions);
	const sessionCentre = median(samples);

	let withinSessionDeviants = 0;
	if (samples.length >= config.minSamplesForSpread) {
		const mad = medianAbsoluteDeviation(samples, sessionCentre);
		if (mad > 0) {
			const threshold = config.deviantSampleMultiple * mad;
			for (const s of samples) {
				if (Math.abs(s - sessionCentre) > threshold) withinSessionDeviants += 1;
			}
		}
		withinSessionDeviants = Math.min(
			withinSessionDeviants,
			config.maxWithinSessionContribution,
		);
	}

	const longMean = ewma(prior.longMean, sessionCentre, alphaLong, prior.initialized);
	// NOTE: deliberately crude for this task -- ANY nonzero gap between this
	// session's own centre and the pre-update long-run mean counts as
	// out-of-band, with no band width yet. Task 9 introduces the real band
	// (noiseFloor/longSd); Task 10 replaces this raw-centre comparison with
	// the smoothed short EWMA the spec calls for.
	const acrossSessionOutOfBand = prior.initialized && sessionCentre !== prior.longMean;

	const contribution = withinSessionDeviants + (acrossSessionOutOfBand ? 1 : 0);
	let consecutiveOutOfBand = prior.consecutiveOutOfBand;
	let consecutiveInBand = prior.consecutiveInBand;
	if (contribution > 0) {
		consecutiveOutOfBand += contribution;
		consecutiveInBand = 0;
	} else {
		consecutiveInBand += 1;
		consecutiveOutOfBand = 0;
	}

	let lifecycle = prior.lifecycle;
	if (lifecycle === "absent" && consecutiveOutOfBand >= config.firePersistence) {
		lifecycle = "active";
	}

	return {
		...prior,
		lifecycle,
		longMean,
		consecutiveOutOfBand,
		consecutiveInBand,
		initialized: true,
	};
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: PASS (9 tests). Confirm no regression on Tasks 6-7's within-session
tests (they never reach the across-session branch since `withinSessionDeviants`
alone already satisfies their assertions).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.ts apps/api/src/services/student-baseline.test.ts && git commit -m "feat(baseline): fire on persistent across-session deviation (crude threshold)"
```

---

## Task 9: The band narrows monotonically under consistent evidence

**Group:** sequential (depends on Task 8)

**Behavior being verified:** the band's effective half-width (bias-corrected
`noiseFloor`/`longSd`) shrinks as more consistent sessions accumulate — the
behavioural proof that cold start has no session-count branch: a wide band
from sparse evidence narrows smoothly, rather than a fixed rule flipping at
session 3.
**Interface under test:** `updateBaseline` (reads `noiseFloor`, `longSd`,
`updateCount` off the returned state).

**Files:**
- Modify: `apps/api/src/services/student-baseline.ts`
- Modify: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

Add `DEFAULT_BASELINE_CONFIG` to the existing import from
`"./student-baseline"`:

```typescript
import {
	type BaselineState,
	BaselineStateSchema,
	DEFAULT_BASELINE_CONFIG,
	type SessionSamples,
	initialBaselineState,
	updateBaseline,
} from "./student-baseline";
```

```typescript
describe("band width", () => {
	it("narrows the band monotonically under consistent evidence", () => {
		// Same call shape every session -- no session-count branch anywhere in
		// updateBaseline. The narrowing is read off the *effective*, bias-
		// corrected band width, not the raw EWMA (which a constant CLUSTER
		// input would hold flat from session 1 and prove nothing).
		const sessions: SessionSamples[] = Array.from({ length: 6 }, (_, i) => ({
			timestamp: `2026-01-0${i + 1}T00:00:00Z`,
			scores: { phrasing: CLUSTER },
		}));
		const trace = runSequence(sessions);
		const halfWidths = trace.map((s) => {
			const d = s.dimensions.phrasing;
			const alphaShort =
				1 - 2 ** (-1 / DEFAULT_BASELINE_CONFIG.shortHalfLifeSessions);
			const alphaLong =
				1 - 2 ** (-1 / DEFAULT_BASELINE_CONFIG.longHalfLifeSessions);
			const weightShort = 1 - (1 - alphaShort) ** d.updateCount;
			const weightLong = 1 - (1 - alphaLong) ** d.updateCount;
			return Math.max(
				d.noiseFloor / weightShort,
				DEFAULT_BASELINE_CONFIG.minBandSdFraction * (d.longSd / weightLong),
			);
		});
		for (let i = 1; i < halfWidths.length; i++) {
			expect(halfWidths[i]).toBeLessThanOrEqual(halfWidths[i - 1] + 1e-9);
		}
		expect(halfWidths[halfWidths.length - 1]).toBeLessThan(halfWidths[0]);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — `noiseFloor`, `longSd`, and `updateCount` are never updated
by `foldDimension` today (they stay at their `initialBaselineState()`
defaults of `0`). `weightShort`/`weightLong` compute
`1 - (1 - alpha) ** 0 = 0`, so every `halfWidths` entry is `0 / 0 = NaN`. The
final assertion `expect(NaN).toBeLessThan(NaN)` throws
`AssertionError: expected NaN to be less than NaN`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add `biasCorrected` to `student-baseline.ts`, and extend `foldDimension` to
compute and carry `noiseFloor`, `longSd`, and `updateCount` — **not yet
consulted by the out-of-band decision**, which still uses Task 8's crude
raw-centre comparison:

```typescript
/**
 * Bias-corrects a fresh EWMA (same trick as Adam's moment estimates): with few
 * updates the raw EWMA is dominated by its own bootstrap value, so dividing by
 * the accumulated weight inflates it. That inflation is what makes the band
 * wide on session 1 and narrow smoothly as `updateCount` grows -- a formula
 * applied uniformly every session, not a branch on how many sessions exist.
 */
function biasCorrected(
	rawEwma: number,
	alpha: number,
	updateCount: number,
): number {
	if (updateCount <= 0) return rawEwma;
	const weight = 1 - (1 - alpha) ** updateCount;
	return rawEwma / weight;
}
```

```typescript
function foldDimension(
	prior: DimensionBaselineState,
	samples: readonly number[],
	config: BaselineConfig,
): DimensionBaselineState {
	const alphaShort = alphaFromHalfLife(config.shortHalfLifeSessions);
	const alphaLong = alphaFromHalfLife(config.longHalfLifeSessions);
	const sessionCentre = median(samples);

	let withinSessionDeviants = 0;
	if (samples.length >= config.minSamplesForSpread) {
		const mad = medianAbsoluteDeviation(samples, sessionCentre);
		if (mad > 0) {
			const threshold = config.deviantSampleMultiple * mad;
			for (const s of samples) {
				if (Math.abs(s - sessionCentre) > threshold) withinSessionDeviants += 1;
			}
		}
		withinSessionDeviants = Math.min(
			withinSessionDeviants,
			config.maxWithinSessionContribution,
		);
	}

	const noiseFloorSample =
		samples.length >= config.minSamplesForSpread
			? medianAbsoluteDeviation(samples, sessionCentre)
			: prior.noiseFloor;
	const noiseFloor = ewma(prior.noiseFloor, noiseFloorSample, alphaShort, prior.initialized);

	const deviation = prior.initialized ? sessionCentre - prior.longMean : 0;
	const longSd = ewma(prior.longSd, Math.abs(deviation), alphaLong, prior.initialized);

	const longMean = ewma(prior.longMean, sessionCentre, alphaLong, prior.initialized);
	const updateCount = prior.updateCount + 1;

	// Still Task 8's crude decision -- noiseFloor/longSd/updateCount are
	// computed and carried on state (proving the band narrows) but not yet
	// consulted here. Task 10 wires them into the real decision.
	const acrossSessionOutOfBand = prior.initialized && sessionCentre !== prior.longMean;

	const contribution = withinSessionDeviants + (acrossSessionOutOfBand ? 1 : 0);
	let consecutiveOutOfBand = prior.consecutiveOutOfBand;
	let consecutiveInBand = prior.consecutiveInBand;
	if (contribution > 0) {
		consecutiveOutOfBand += contribution;
		consecutiveInBand = 0;
	} else {
		consecutiveInBand += 1;
		consecutiveOutOfBand = 0;
	}

	let lifecycle = prior.lifecycle;
	if (lifecycle === "absent" && consecutiveOutOfBand >= config.firePersistence) {
		lifecycle = "active";
	}

	return {
		...prior,
		lifecycle,
		longMean,
		longSd,
		noiseFloor,
		consecutiveOutOfBand,
		consecutiveInBand,
		initialized: true,
		updateCount,
	};
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.ts apps/api/src/services/student-baseline.test.ts && git commit -m "feat(baseline): track bias-corrected noiseFloor and longSd for a narrowing band"
```

---

## Task 10: A single off session stays quiet

**Group:** sequential (depends on Task 9)

**Behavior being verified:** one atypical session amid otherwise-consistent
sessions must not fire — this forces replacing Task 8's crude raw-centre
comparison with the real band (bias-corrected `noiseFloor`/`longSd`) and the
smoothed short EWMA the spec calls for ("the short EWMA sitting outside the
band").
**Interface under test:** `updateBaseline`.

**Files:**
- Modify: `apps/api/src/services/student-baseline.ts`
- Modify: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

Add inside `describe("across-session evidence", ...)`:

```typescript
	it("stays quiet on a single off session amid consistent sessions", () => {
		const mildlyOff = [0.59, 0.61, 0.59, 0.61, 0.59, 0.61];
		const trace = runSequence([
			{ timestamp: "2026-01-01T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-02T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-03T00:00:00Z", scores: { dynamics: mildlyOff } },
			{ timestamp: "2026-01-04T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-05T00:00:00Z", scores: { dynamics: CLUSTER } },
			{ timestamp: "2026-01-06T00:00:00Z", scores: { dynamics: CLUSTER } },
		]);
		for (const s of trace) {
			expect(s.dimensions.dynamics.lifecycle).toBe("absent");
		}
	});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — Task 8/9's crude rule counts ANY nonzero gap between the raw
session centre and the pre-update long-run mean as out-of-band, with no
threshold. After the `mildlyOff` session nudges `longMean` away from exactly
`0.5`, every subsequent `CLUSTER` session's centre (`0.5`) also differs from
the now-slightly-shifted `longMean`, so out-of-band contributions keep
accumulating and the dimension fires by the 5th session:
`AssertionError: expected 'active' to be 'absent'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace `foldDimension` in `student-baseline.ts` with the real decision:
compute `shortMean`, and gate on the bias-corrected band half-width instead
of a raw nonzero-gap check.

```typescript
function foldDimension(
	prior: DimensionBaselineState,
	samples: readonly number[],
	config: BaselineConfig,
): DimensionBaselineState {
	const alphaShort = alphaFromHalfLife(config.shortHalfLifeSessions);
	const alphaLong = alphaFromHalfLife(config.longHalfLifeSessions);
	const sessionCentre = median(samples);

	let withinSessionDeviants = 0;
	if (samples.length >= config.minSamplesForSpread) {
		const mad = medianAbsoluteDeviation(samples, sessionCentre);
		if (mad > 0) {
			const threshold = config.deviantSampleMultiple * mad;
			for (const s of samples) {
				if (Math.abs(s - sessionCentre) > threshold) withinSessionDeviants += 1;
			}
		}
		withinSessionDeviants = Math.min(
			withinSessionDeviants,
			config.maxWithinSessionContribution,
		);
	}

	const noiseFloorSample =
		samples.length >= config.minSamplesForSpread
			? medianAbsoluteDeviation(samples, sessionCentre)
			: prior.noiseFloor;
	const noiseFloor = ewma(prior.noiseFloor, noiseFloorSample, alphaShort, prior.initialized);

	const deviation = prior.initialized ? sessionCentre - prior.longMean : 0;
	const longSd = ewma(prior.longSd, Math.abs(deviation), alphaLong, prior.initialized);

	const shortMean = ewma(prior.shortMean, sessionCentre, alphaShort, prior.initialized);
	const longMean = ewma(prior.longMean, sessionCentre, alphaLong, prior.initialized);

	const updateCount = prior.updateCount + 1;
	const effectiveNoiseFloor = biasCorrected(noiseFloor, alphaShort, updateCount);
	const effectiveLongSd = biasCorrected(longSd, alphaLong, updateCount);
	const halfWidth = Math.max(
		effectiveNoiseFloor,
		config.minBandSdFraction * effectiveLongSd,
	);
	const acrossSessionOutOfBand = Math.abs(shortMean - longMean) > halfWidth;

	const contribution = withinSessionDeviants + (acrossSessionOutOfBand ? 1 : 0);
	let consecutiveOutOfBand = prior.consecutiveOutOfBand;
	let consecutiveInBand = prior.consecutiveInBand;
	if (contribution > 0) {
		consecutiveOutOfBand += contribution;
		consecutiveInBand = 0;
	} else {
		consecutiveInBand += 1;
		consecutiveOutOfBand = 0;
	}

	let lifecycle = prior.lifecycle;
	if (lifecycle === "absent" && consecutiveOutOfBand >= config.firePersistence) {
		lifecycle = "active";
	}

	return {
		...prior,
		lifecycle,
		longMean,
		longSd,
		shortMean,
		noiseFloor,
		consecutiveOutOfBand,
		consecutiveInBand,
		initialized: true,
		updateCount,
	};
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: PASS (11 tests). Confirm no regression: re-run Task 8's persistent-
deviation test — the 0.3-magnitude `shifted` cluster is far enough outside
even the inflated early-session band to still fire by the 4th session (index
3).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.ts apps/api/src/services/student-baseline.test.ts && git commit -m "feat(baseline): gate across-session firing on the real band and short EWMA"
```

---

## Task 11: Symmetric retirement — active -> improving -> resolved

**Group:** sequential (depends on Task 10)

**Behavior being verified:** persistent return-to-band softens `active` to
`improving`, then retires to `resolved` — the same persistence-counter
mechanism read in reverse.
**Interface under test:** `updateBaseline`.

**Files:**
- Modify: `apps/api/src/services/student-baseline.ts`
- Modify: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
describe("symmetric retirement", () => {
	it("walks active -> improving -> resolved on persistent return-to-band", () => {
		const shifted = [0.79, 0.81, 0.79, 0.81, 0.79, 0.81];
		const sessions: SessionSamples[] = [
			{ timestamp: "2026-01-01T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-01-02T00:00:00Z", scores: { timing: shifted } },
			{ timestamp: "2026-01-03T00:00:00Z", scores: { timing: shifted } },
			{ timestamp: "2026-01-04T00:00:00Z", scores: { timing: shifted } }, // fires -> active
			{ timestamp: "2026-02-01T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-02T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-03T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-04T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-05T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-06T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-07T00:00:00Z", scores: { timing: CLUSTER } }, // -> improving
			{ timestamp: "2026-02-08T00:00:00Z", scores: { timing: CLUSTER } }, // -> resolved
		];
		const trace = runSequence(sessions);
		expect(trace[3].dimensions.timing.lifecycle).toBe("active");
		expect(trace[10].dimensions.timing.lifecycle).toBe("improving");
		expect(trace[11].dimensions.timing.lifecycle).toBe("resolved");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — `foldDimension`'s lifecycle switch only handles
`"absent" -> "active"` today; once `active`, nothing moves it to
`"improving"` or `"resolved"` regardless of `consecutiveInBand`:
`AssertionError: expected 'active' to be 'improving'` at `trace[10]`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `foldDimension`, replace the lifecycle `if` with a full switch handling
`active` and `improving` (not yet `resolved`, which stays terminal until
Task 12):

```typescript
	let lifecycle = prior.lifecycle;
	if (lifecycle === "absent") {
		if (consecutiveOutOfBand >= config.firePersistence) lifecycle = "active";
	} else if (lifecycle === "active") {
		if (consecutiveInBand >= config.improvingPersistence) lifecycle = "improving";
	} else if (lifecycle === "improving") {
		if (consecutiveInBand >= config.retirePersistence) lifecycle = "resolved";
	}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.ts apps/api/src/services/student-baseline.test.ts && git commit -m "feat(baseline): retire active dimensions through improving to resolved"
```

---

## Task 12: Recurrence after resolved returns lifecycle to active

**Group:** sequential (depends on Task 11)

**Behavior being verified:** after a dimension resolves, persistent
out-of-band evidence reopens it as `active` (recurrence), and the same
applies if it recurs while still `improving`.
**Interface under test:** `updateBaseline`.

**Files:**
- Modify: `apps/api/src/services/student-baseline.ts`
- Modify: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
describe("recurrence", () => {
	it("returns lifecycle to active after a resolved dimension recurs", () => {
		const shifted = [0.79, 0.81, 0.79, 0.81, 0.79, 0.81];
		const sessions: SessionSamples[] = [
			{ timestamp: "2026-01-01T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-01-02T00:00:00Z", scores: { timing: shifted } },
			{ timestamp: "2026-01-03T00:00:00Z", scores: { timing: shifted } },
			{ timestamp: "2026-01-04T00:00:00Z", scores: { timing: shifted } }, // fires -> active
			{ timestamp: "2026-02-01T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-02T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-03T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-04T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-05T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-06T00:00:00Z", scores: { timing: CLUSTER } },
			{ timestamp: "2026-02-07T00:00:00Z", scores: { timing: CLUSTER } }, // -> improving
			{ timestamp: "2026-02-08T00:00:00Z", scores: { timing: CLUSTER } }, // -> resolved
			{ timestamp: "2026-03-01T00:00:00Z", scores: { timing: shifted } },
			{ timestamp: "2026-03-02T00:00:00Z", scores: { timing: shifted } },
			{ timestamp: "2026-03-03T00:00:00Z", scores: { timing: shifted } }, // recurs -> active
		];
		const trace = runSequence(sessions);
		expect(trace[11].dimensions.timing.lifecycle).toBe("resolved");
		expect(trace[14].dimensions.timing.lifecycle).toBe("active");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — the lifecycle switch has no `"resolved"` branch today, so
`lifecycle` stays `"resolved"` forever regardless of `consecutiveOutOfBand`:
`AssertionError: expected 'resolved' to be 'active'` at `trace[14]`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Extend the lifecycle switch in `foldDimension` to its final, complete form —
adding the `"improving" -> "active"` recurrence-while-improving branch and
the `"resolved" -> "active"` branch:

```typescript
	let lifecycle = prior.lifecycle;
	if (lifecycle === "absent") {
		if (consecutiveOutOfBand >= config.firePersistence) lifecycle = "active";
	} else if (lifecycle === "active") {
		if (consecutiveInBand >= config.improvingPersistence) lifecycle = "improving";
	} else if (lifecycle === "improving") {
		if (consecutiveOutOfBand >= config.firePersistence) {
			lifecycle = "active";
		} else if (consecutiveInBand >= config.retirePersistence) {
			lifecycle = "resolved";
		}
	} else if (lifecycle === "resolved") {
		if (consecutiveOutOfBand >= config.firePersistence) lifecycle = "active";
	}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: PASS (13 tests). Confirm no regression on Task 11's test (the
`"improving" -> "active"` branch is never triggered by that sequence, since
it never re-fires while improving).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.ts apps/api/src/services/student-baseline.test.ts && git commit -m "feat(baseline): recur to active from resolved or improving"
```

---

## Task 13: Promotion after out-of-band evidence in >=2 distinct ISO weeks

**Group:** sequential (depends on Task 12)

**Behavior being verified:** a dimension is promoted to a durable habit only
once it has recorded out-of-band evidence while `active` in at least
`PROMOTION_DISTINCT_WEEKS` (2) distinct ISO weeks; evidence confined to a
single week does not promote. Both branches of this one gate are asserted in
a single test, matching the issue's single success criterion ("recurs across
weeks... not one week").
**Interface under test:** `updateBaseline`.

**Files:**
- Modify: `apps/api/src/services/student-baseline.ts`
- Modify: `apps/api/src/services/student-baseline.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
describe("promotion", () => {
	it("promotes only once out-of-band evidence while active spans >=2 distinct ISO weeks", () => {
		const shifted = [0.79, 0.81, 0.79, 0.81, 0.79, 0.81];
		const trace = runSequence([
			{ timestamp: "2026-01-05T00:00:00Z", scores: { articulation: CLUSTER } }, // Mon wk02
			{ timestamp: "2026-01-06T00:00:00Z", scores: { articulation: shifted } }, // wk02
			{ timestamp: "2026-01-07T00:00:00Z", scores: { articulation: shifted } }, // wk02
			{ timestamp: "2026-01-08T00:00:00Z", scores: { articulation: shifted } }, // wk02, fires
			{ timestamp: "2026-01-13T00:00:00Z", scores: { articulation: shifted } }, // wk03, more evidence
		]);
		expect(trace[3].dimensions.articulation.lifecycle).toBe("active");
		expect(trace[3].dimensions.articulation.promoted).toBe(false);
		expect(trace[4].dimensions.articulation.promoted).toBe(true);
	});

	it("does not promote when all evidence falls inside a single ISO week", () => {
		const shifted = [0.79, 0.81, 0.79, 0.81, 0.79, 0.81];
		const trace = runSequence([
			{ timestamp: "2026-01-05T00:00:00Z", scores: { articulation: CLUSTER } },
			{ timestamp: "2026-01-06T00:00:00Z", scores: { articulation: shifted } },
			{ timestamp: "2026-01-07T00:00:00Z", scores: { articulation: shifted } },
			{ timestamp: "2026-01-08T00:00:00Z", scores: { articulation: shifted } },
		]);
		expect(trace[3].dimensions.articulation.lifecycle).toBe("active");
		expect(trace[3].dimensions.articulation.promoted).toBe(false);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: FAIL — `foldDimension` never touches `evidenceWeeks`/`promoted`
today, so `promoted` stays `false` from `initialBaselineState()` at every
step: `AssertionError: expected false to be true` on the first test's
`trace[4].dimensions.articulation.promoted` assertion.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add an `isoWeek` helper to `student-baseline.ts`, thread `timestamp` into
`foldDimension`, and compute `evidenceWeeks`/`promoted` at the end of the
fold:

```typescript
function isoWeek(timestamp: string): string {
	const date = new Date(timestamp);
	const target = new Date(
		Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()),
	);
	const dayNumber = (target.getUTCDay() + 6) % 7;
	target.setUTCDate(target.getUTCDate() - dayNumber + 3);
	const firstThursday = new Date(Date.UTC(target.getUTCFullYear(), 0, 4));
	const week =
		1 +
		Math.round(
			((target.getTime() - firstThursday.getTime()) / 86400000 -
				3 +
				((firstThursday.getUTCDay() + 6) % 7)) /
				7,
		);
	return `${target.getUTCFullYear()}-W${String(week).padStart(2, "0")}`;
}
```

Change `foldDimension`'s signature to accept `timestamp`, and add the
promotion block at the end, right before the `return`:

```typescript
function foldDimension(
	prior: DimensionBaselineState,
	samples: readonly number[],
	timestamp: string,
	config: BaselineConfig,
): DimensionBaselineState {
	// ...unchanged body through the lifecycle switch (Task 12)...

	let evidenceWeeks = prior.evidenceWeeks;
	let promoted = prior.promoted;
	if (lifecycle === "active" && contribution > 0) {
		const week = isoWeek(timestamp);
		if (!evidenceWeeks.includes(week)) {
			evidenceWeeks = [...evidenceWeeks, week];
		}
		if (evidenceWeeks.length >= config.promotionDistinctWeeks) {
			promoted = true;
		}
	}

	return {
		...prior,
		lifecycle,
		longMean,
		longSd,
		shortMean,
		noiseFloor,
		consecutiveOutOfBand,
		consecutiveInBand,
		promoted,
		evidenceWeeks,
		initialized: true,
		updateCount,
	};
}
```

Update the call site in `updateBaseline` to pass `session.timestamp`:

```typescript
export function updateBaseline(
	state: BaselineState,
	session: SessionSamples,
	config: BaselineConfig = DEFAULT_BASELINE_CONFIG,
): BaselineState {
	validateSession(state, session);
	const dimensions = { ...state.dimensions };
	for (const [dimension, samples] of Object.entries(session.scores)) {
		if (!samples || samples.length === 0) continue;
		const dim = dimension as Dimension;
		dimensions[dim] = foldDimension(
			dimensions[dim],
			samples,
			session.timestamp,
			config,
		);
	}
	return { lastSessionTimestamp: session.timestamp, dimensions };
}
```

`student-baseline.ts` is now complete and matches the module described in
the spec's Modules section.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test:scripts -- student-baseline
```
Expected: PASS (15 tests: all tasks 0-13).

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/student-baseline.ts apps/api/src/services/student-baseline.test.ts && git commit -m "feat(baseline): promote habits after out-of-band evidence in 2+ distinct ISO weeks"
```

---

## Task 14: Full verification — both runtimes, typecheck, lint

**Group:** sequential (depends on Task 13; final task)

**Behavior being verified:** the module and its test suite are clean in both
Vitest configurations (workerd pool and Node), and the whole `apps/api`
package still typechecks and lints, per the spec's Verification Architecture
section.

**Files:** none changed — this task only runs commands and records output.

- [ ] **Step 1: Run the full suite**

```bash
cd apps/api && bun run test && bun run test:scripts && bun run typecheck && bun run lint
```

- [ ] **Step 2: Interpret the results**

- `bun run test` (workerd pool, `vitest.config.ts`) must show
  `src/services/student-baseline.test.ts` passing all 15 tests, alongside the
  rest of the existing suite.
- `bun run test:scripts` (Node, `vitest.node.config.ts`) must show the same
  15 tests passing.
- `bun run typecheck` must show no errors attributable to
  `src/services/student-baseline.ts` or `.test.ts`. **Known pre-existing,
  unrelated failures**: `src/services/wasm-bridge.ts` reports two
  `TS2307: Cannot find module '../wasm/.../pkg/...'` errors because the WASM
  packages have not been built in this checkout (`bun run build:wasm` is a
  separate, unrelated prerequisite — see `docs/standards/rules.json` /
  the `check-api needs build:wasm first` gotcha). Confirm no *other*
  typecheck errors appear; do not attempt to fix the WASM errors as part of
  this issue.
- `bun run lint` must show no errors attributable to `student-baseline.ts` or
  `.test.ts`. **Known pre-existing, unrelated findings**: `biome check`
  reports errors/warnings elsewhere in `apps/api/src` (e.g.
  `wasm-bridge.workerd.test.ts` non-null-assertion findings) that predate
  this issue. If `bun run lint` exits non-zero, isolate the two new files
  with `bunx biome check src/services/student-baseline.ts
  src/services/student-baseline.test.ts` and confirm that command alone
  reports zero errors and zero warnings before treating the task as done.

- [ ] **Step 3: No commit**

This task makes no file changes, so there is nothing to commit. If Step 2
finds a genuine regression in the two new files (not a pre-existing,
unrelated failure), fix it, re-run the affected earlier task's test command
to confirm, and commit the fix with a message referencing which task's
implementation it corrects — then re-run Step 1 in full before considering
the plan complete.

---

## Task Groups

All 15 tasks (0-14) are **sequential** — every task modifies
`apps/api/src/services/student-baseline.ts` and/or
`apps/api/src/services/student-baseline.test.ts`, so there is no pair of
tasks that can run as a parallel group. This is a single vertical slice
built up one behavior at a time; dispatch tasks to a build agent one at a
time, in order, verifying each before moving to the next.

**Decouple check:** the plan produces exactly one deep module
(`student-baseline.ts`) with one test file. No task group ships independent
user value on its own — the module has no caller yet (wiring it into a
pipeline or route is explicitly out of scope per the spec's "Not in scope"
section and is follow-up work on #162). The whole plan is one unit that
lands when Task 14 is green.
