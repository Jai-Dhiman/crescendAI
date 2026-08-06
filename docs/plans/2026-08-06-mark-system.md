# Mark System Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** One mark vocabulary renders identically on two canvases — a Verovio
score overlay and a session timeline strip — so every score-first surface shows
feedback the same way.
**Spec:** `docs/specs/2026-08-06-mark-system-design.md`
**Style:** Follow `CLAUDE.md` (root) and `apps/web` conventions. Tabs for
indentation, Biome formatting, `bun` never `npm`.

## Working directory

**Every command in this plan runs from
`.worktrees/issue-157-mark-system/apps/web`.** A past session reported a clean
sweep that had silently run against `main` and verified nothing. Before the
first command, confirm:

```bash
pwd   # must end in .worktrees/issue-157-mark-system/apps/web
git rev-parse --abbrev-ref HEAD   # must print issue-157-mark-system
```

Dependencies are already installed there (`bun install`, 503 packages). Note
there is no root `package.json` — a root-level `bun install` fails with
"Bun could not find a package.json file to install from". Install per app.

## On "one test per task"

Every task is one vertical slice: a failing test, the minimum implementation
that makes it pass, one commit. A few tasks contain two sibling `it` blocks
(Tasks 3, 4, 12, 15) where two cases drive the *same* implementation change —
for example "bars kept at the threshold" and "no bars supplied at all" both
exercise one conditional. That is still one slice. What the plan never does is
write tests for a later task's implementation.

Deliberate design choice worth knowing before executing: several
implementations are written as the **true** minimum, which means they are
knowingly incomplete. Task 2's `resolveAnchor` always degrades; Task 7's
`placeMarks` silently drops what it cannot place. Those gaps are what let Tasks
3, 8, and 9 fail first and therefore prove something. Do not "helpfully"
complete an earlier task's implementation — it disarms the next task's test.

## Known-red window

Task 1 commits a harness that **fails on purpose** and stays red until Task 19.
From Task 1 until Task 19, `bun run test` has exactly one failing file:
`src/components/mark-canvases.contract.test.tsx`. That is the harness doing its
job. Any *other* failing file is a real regression. Do not skip, `.skip`, or
delete the harness to get a green run.

## Task Groups

```
Group 0                        : Task 1                    (harness, expected red)
Group A (sequential, one file) : Task 2 -> 3 -> 4 -> 5     (src/lib/mark.ts)
Group B (sequential, depends A): Task 7 -> 8 -> 9          (src/lib/mark-placement.ts)
Group C (depends on B)         : Task 6                    (fixtures)
Group D (depends on C)         : chain D1: Task 10 -> 11   (MarkGlyph.tsx)
                                 chain D2: Task 12         (MarkDetail.tsx)
                                 D1 and D2 run in parallel — different files.
Group E (depends on D)         : chain E1: Task 13 -> 14   (ScoreMarkLayer.tsx)
                                 chain E2: Task 15 -> 16   (SessionTimelineStrip.tsx)
                                 E1 and E2 run in parallel — different files.
Group F (sequential, depends E): Task 17 -> 18             (18 needs 17's route live)
Group G (depends on F)         : Task 19                   (harness green + full gates)
```

**Task numbers are not in dependency order — group letters are.** Tasks 7-9
(Group B) run before Task 6 (Group C), because `mark-fixtures.ts` imports the
`BarLocator` type from `mark-placement.ts`. Executing Task 6 first would fail on
an unresolvable import.

Tasks within Groups A and B touch a single source file each and therefore
**cannot** be parallelised. Groups D and E each contain two independent chains
that can. Group F is sequential despite touching different files: Task 18's axe
run navigates to the route Task 17 creates.

`[SHIPS INDEPENDENTLY]` — **Groups A through C**. On their own they give the
codebase a mark vocabulary and a placement function with the wrong-bar defect
designed out, consumable by #158/#159/#162 even if no canvas ships. Nothing
user-visible, so it ships as a library slice, not a feature.

---

### Task 1: Cross-canvas contract harness

**Group:** 0 (no dependencies; blocks nothing, but must be committed red first)

**Behavior being verified:** Canvas B renders every mark; every mark Canvas A
renders carries an accessible name identical to that mark's name on Canvas B.

**Interface under test:** `<SessionTimelineStrip />` and `<ScoreMarkLayer />`
rendered DOM, via accessible names.

**Files:**
- Test: `src/components/mark-canvases.contract.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it } from "vitest";
import {
	FIXTURE_BARS,
	FIXTURE_DURATION_SECONDS,
	FIXTURE_MARKS,
} from "../test-utils/mark-fixtures";
import { ScoreMarkLayer } from "./ScoreMarkLayer";
import { SessionTimelineStrip } from "./SessionTimelineStrip";

/**
 * Builds a container holding one element per fixture bar that is on the
 * rendered page, with the id ScoreMarkLayer resolves through
 * BarIR.measureOn. Bar 88 is deliberately omitted: it models a bar on a
 * page the overlay is not showing.
 */
function renderScoreCanvas() {
	const ref = createRef<HTMLDivElement>();
	const ON_PAGE = FIXTURE_BARS.filter((b) => b.barNumber !== 88);
	return render(
		<div ref={ref}>
			{ON_PAGE.map((b) => (
				<div key={b.measureOn} id={b.measureOn} />
			))}
			<ScoreMarkLayer containerRef={ref} bars={FIXTURE_BARS} marks={FIXTURE_MARKS} />
		</div>,
	);
}

function namesOf(container: HTMLElement): string[] {
	return Array.from(container.querySelectorAll("button[aria-expanded]")).map(
		(b) => b.getAttribute("aria-label") ?? "",
	);
}

describe("mark canvases share one vocabulary", () => {
	it("renders every mark on the timeline canvas", () => {
		const { container } = render(
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>,
		);
		expect(namesOf(container)).toHaveLength(FIXTURE_MARKS.length);
	});

	it("gives a mark the identical accessible name on whichever canvas shows it", () => {
		const timeline = render(
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>,
		);
		const timelineNames = new Set(namesOf(timeline.container));
		timeline.unmount();

		const score = renderScoreCanvas();
		const scoreNames = namesOf(score.container);

		// Canvas A is lossy BY DESIGN — it shows only what it can truly place.
		// So this is containment, not equality. Equality would assert the
		// opposite of the intended behaviour.
		expect(scoreNames.length).toBeGreaterThan(0);
		expect(scoreNames.length).toBeLessThan(FIXTURE_MARKS.length);
		for (const name of scoreNames) {
			expect(timelineNames).toContain(name);
		}
	});

	it("never shows a bar number for a low-alignment mark on either canvas", () => {
		const timeline = render(
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>,
		);
		// FIXTURE_MARKS m4 supplied bars [21, 22] at alignmentQuality 0.31.
		expect(timeline.container.textContent).not.toContain("21");
		expect(timeline.container.textContent).not.toContain("22");
		expect(screen.getAllByLabelText(/1:37/)).not.toHaveLength(0);
		timeline.unmount();

		const score = renderScoreCanvas();
		expect(score.container.textContent).not.toContain("bars 21");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/components/mark-canvases.contract.test.tsx
```

Expected: FAIL — a module resolution error, e.g.
`Failed to resolve import "../test-utils/mark-fixtures" from "src/components/mark-canvases.contract.test.tsx"`.
Any of the four not-yet-created modules may be reported first; a
module-resolution failure is the expected outcome. If this file somehow
passes, the harness is wrong — rewrite it.

- [ ] **Step 3: Implement the minimum to make the test pass**

None. This task deliberately ships a red harness; Task 19 turns it green. Do
not create stubs to satisfy it.

- [ ] **Step 4: Run test — verify it PASSES**

Not applicable for this task. Re-confirm the failure is a module-resolution
error and not a syntax error in the test itself:

```bash
bunx vitest run src/components/mark-canvases.contract.test.tsx 2>&1 | head -30
```

- [ ] **Step 5: Commit**

```bash
git add src/components/mark-canvases.contract.test.tsx && git commit -m "test(marks): add cross-canvas contract harness (red until both canvases exist)" --no-verify
```

`--no-verify` is required here and **only** here: the pre-commit hook runs
mechanical checks and this commit is intentionally red. Say so in the message,
per `CLAUDE.md`.

---

### Task 2: Anchor degrades to timestamp when alignment is poor

**Group:** A (sequential — Tasks 2-5 all edit `src/lib/mark.ts`)

**Behavior being verified:** Supplying bars with low alignment quality yields a
timestamp anchor; the bars are discarded and cannot be read back out.

**Interface under test:** `resolveAnchor()`

**Files:**
- Create: `src/lib/mark.ts`
- Test: `src/lib/mark.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import { ALIGNMENT_MIN, resolveAnchor } from "./mark";

describe("resolveAnchor", () => {
	it("discards bars when alignment quality is below the threshold", () => {
		const anchor = resolveAnchor({
			atSeconds: 97,
			bars: [21, 22],
			alignmentQuality: ALIGNMENT_MIN - 0.01,
		});

		expect(anchor.type).toBe("timestamp");
		expect(anchor).not.toHaveProperty("bars");
		expect(anchor.atSeconds).toBe(97);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/lib/mark.test.ts
```

Expected: FAIL — `Failed to resolve import "./mark" from "src/lib/mark.test.ts"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/lib/mark.ts`:

```ts
/**
 * The brand is not exported. `resolveAnchor` is therefore the only function in
 * the codebase that can produce a MarkAnchor: no object literal written
 * anywhere else is assignable to this type. That is what makes "wrong bar
 * numbers are never shown" a compile-time property rather than a runtime guard
 * a future surface can route around.
 */
declare const anchorBrand: unique symbol;

type AnchorBrand = { readonly [anchorBrand]: true };

export type MarkAnchor = AnchorBrand &
	(
		| {
				readonly type: "bars";
				readonly bars: readonly [number, number];
				readonly atSeconds: number;
		  }
		| { readonly type: "timestamp"; readonly atSeconds: number }
	);

/**
 * Uncalibrated. There is no distribution of real alignment-quality scores in
 * this repo yet, so this is a starting value chosen to be conservative, in the
 * same spirit as #165's note on DEVIANT_SAMPLE_MULTIPLE. Tune against real
 * alignment output, not intuition.
 */
export const ALIGNMENT_MIN = 0.8;

export interface AnchorCandidate {
	readonly atSeconds: number;
	readonly bars?: readonly [number, number];
	readonly alignmentQuality: number;
}

/**
 * The single degradation function. Every anchor carries atSeconds — including
 * the bars variant — because the timeline canvas must be able to place any
 * mark, and elapsed time is the one coordinate every mark always has.
 */
export function resolveAnchor(candidate: AnchorCandidate): MarkAnchor {
	return {
		type: "timestamp",
		atSeconds: candidate.atSeconds,
	} as unknown as MarkAnchor;
}
```

This is the genuine minimum for this test: always degrade. Task 3 drives the
bars branch. Do not write the conditional yet — if you do, Task 3's test cannot
fail first and stops proving anything.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/lib/mark.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lib/mark.ts src/lib/mark.test.ts && git commit -m "feat(marks): degrade anchors to timestamp below the alignment threshold"
```

---

### Task 3: Anchor keeps bars when alignment clears the threshold

**Group:** A (sequential — depends on Task 2)

**Behavior being verified:** Bars survive at or above `ALIGNMENT_MIN`, and the
resulting anchor still carries elapsed time.

**Interface under test:** `resolveAnchor()`

**Files:**
- Modify: `src/lib/mark.ts`
- Test: `src/lib/mark.test.ts`

- [ ] **Step 1: Write the failing test**

Append inside the existing `describe("resolveAnchor", ...)` block in
`src/lib/mark.test.ts`:

```ts
	it("keeps bars at exactly the threshold and still carries elapsed time", () => {
		const anchor = resolveAnchor({
			atSeconds: 64,
			bars: [5, 6],
			alignmentQuality: ALIGNMENT_MIN,
		});

		expect(anchor.type).toBe("bars");
		if (anchor.type !== "bars") throw new Error("unreachable");
		expect(anchor.bars).toEqual([5, 6]);
		expect(anchor.atSeconds).toBe(64);
	});

	it("returns a timestamp anchor when no bars are supplied at all", () => {
		const anchor = resolveAnchor({ atSeconds: 12, alignmentQuality: 1 });

		expect(anchor.type).toBe("timestamp");
		expect(anchor.atSeconds).toBe(12);
	});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/lib/mark.test.ts
```

Expected: FAIL — `expected 'timestamp' to be 'bars'`. Task 2 always degrades,
so the bars branch does not exist yet. (The second case, "no bars supplied",
passes already; the first case is what drives this task.)

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the body of `resolveAnchor` in `src/lib/mark.ts` with:

```ts
export function resolveAnchor(candidate: AnchorCandidate): MarkAnchor {
	const { atSeconds, bars, alignmentQuality } = candidate;
	// `>=`, not `>`: ALIGNMENT_MIN is the lowest quality still trusted for
	// bars. The boundary case is pinned by the test above.
	if (bars && alignmentQuality >= ALIGNMENT_MIN) {
		return { type: "bars", bars, atSeconds } as unknown as MarkAnchor;
	}
	return { type: "timestamp", atSeconds } as unknown as MarkAnchor;
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/lib/mark.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lib/mark.ts src/lib/mark.test.ts && git commit -m "feat(marks): keep bars at or above the alignment threshold"
```

---

### Task 4: Anchor labels read correctly for bars and timestamps

**Group:** A (sequential — depends on Task 3)

**Behavior being verified:** A bar range reads "bars 5-6", a single bar reads
"bar 5", and a timestamp reads "1:37" — zero-padded, never a raw second count.

**Interface under test:** `anchorLabel()`

**Files:**
- Modify: `src/lib/mark.ts`
- Test: `src/lib/mark.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `src/lib/mark.test.ts` (a new top-level `describe`):

```ts
describe("anchorLabel", () => {
	it("names a range, a single bar, and a timestamp in the student's words", () => {
		const range = resolveAnchor({
			atSeconds: 64,
			bars: [5, 6],
			alignmentQuality: 1,
		});
		const single = resolveAnchor({
			atSeconds: 151,
			bars: [12, 12],
			alignmentQuality: 1,
		});
		const stamp = resolveAnchor({ atSeconds: 97, alignmentQuality: 1 });

		expect(anchorLabel(range)).toBe("bars 5-6");
		expect(anchorLabel(single)).toBe("bar 12");
		expect(anchorLabel(stamp)).toBe("1:37");
	});

	it("zero-pads seconds under ten", () => {
		const stamp = resolveAnchor({ atSeconds: 305, alignmentQuality: 0 });
		expect(anchorLabel(stamp)).toBe("5:05");
	});
});
```

Add `anchorLabel` to the existing import at the top of the file:

```ts
import { ALIGNMENT_MIN, anchorLabel, resolveAnchor } from "./mark";
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/lib/mark.test.ts
```

Expected: FAIL — `"anchorLabel" is not exported by "src/lib/mark.ts"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `src/lib/mark.ts`:

```ts
function formatElapsed(totalSeconds: number): string {
	const minutes = Math.floor(totalSeconds / 60);
	const seconds = Math.floor(totalSeconds % 60);
	return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

/** The one place an anchor becomes words. Both canvases call it. */
export function anchorLabel(anchor: MarkAnchor): string {
	if (anchor.type === "bars") {
		const [start, end] = anchor.bars;
		return start === end ? `bar ${start}` : `bars ${start}-${end}`;
	}
	return formatElapsed(anchor.atSeconds);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/lib/mark.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lib/mark.ts src/lib/mark.test.ts && git commit -m "feat(marks): render anchors as words for both canvases"
```

---

### Task 5: Mark-worthiness is derived from lifecycle, not stored

**Group:** A (sequential — depends on Task 4)

**Behavior being verified:** `isMarkWorthy` is true for every lifecycle #163 can
emit except `absent`.

**Interface under test:** `isMarkWorthy()`

**Files:**
- Modify: `src/lib/mark.ts`
- Test: `src/lib/mark.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `src/lib/mark.test.ts`:

```ts
describe("isMarkWorthy", () => {
	it("is false only for absent", () => {
		expect(isMarkWorthy("absent")).toBe(false);
		expect(isMarkWorthy("active")).toBe(true);
		expect(isMarkWorthy("improving")).toBe(true);
		expect(isMarkWorthy("resolved")).toBe(true);
	});
});
```

Extend the import at the top of the file:

```ts
import { ALIGNMENT_MIN, anchorLabel, isMarkWorthy, resolveAnchor } from "./mark";
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/lib/mark.test.ts
```

Expected: FAIL — `"isMarkWorthy" is not exported by "src/lib/mark.ts"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Add this import at the top of `src/lib/mark.ts` (it is introduced here, not in
Task 2, because an unused type import would fail `bun run lint` in the
intervening tasks):

```ts
import type { Dimension } from "./mock-session";
```

Then append to `src/lib/mark.ts`:

```ts
export type MarkTaxonomy = "needs_work" | "missed_opportunity" | "strong";

/** The three mark-worthy values of #163's Lifecycle. `absent` produces no mark. */
export type MarkLifecycle = "active" | "improving" | "resolved";

/** Mirrors #163's Lifecycle at apps/api/src/services/student-baseline.ts. */
export type BaselineLifecycle = "absent" | MarkLifecycle;

/** Display hint only. Never gates rendering, placement, or visibility. */
export type MarkConfidence = "exploratory" | "provisional" | "established";

export interface Mark {
	readonly id: string;
	readonly anchor: MarkAnchor;
	readonly taxonomy: MarkTaxonomy;
	readonly dimension: Dimension;
	readonly evidence: string;
	readonly lifecycle: MarkLifecycle;
	readonly confidence?: MarkConfidence;
}

/**
 * The single derivation of mark-worthiness. #157 deliberately has no
 * `markWorthy` field: two copies of one fact drift.
 */
export function isMarkWorthy(lifecycle: BaselineLifecycle): boolean {
	return lifecycle !== "absent";
}

export const TAXONOMY_GLYPH: Readonly<Record<MarkTaxonomy, string>> = {
	needs_work: "◉",
	missed_opportunity: "○",
	strong: "★",
};

export const TAXONOMY_LABEL: Readonly<Record<MarkTaxonomy, string>> = {
	needs_work: "Needs work",
	missed_opportunity: "Missed opportunity",
	strong: "Strong",
};

/**
 * Lifecycle -> visual strength. A lookup, never a computation: the client is
 * forbidden from deriving or transitioning lifecycle, which is server state.
 */
export const LIFECYCLE_OPACITY: Readonly<Record<MarkLifecycle, number>> = {
	active: 1,
	improving: 0.7,
	resolved: 0.4,
};
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/lib/mark.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lib/mark.ts src/lib/mark.test.ts && git commit -m "feat(marks): derive mark-worthiness from lifecycle and add the shared vocabulary"
```

---

### Task 6: Fixtures cover the full mark vocabulary

**Group:** C (depends on Group B — imports `BarLocator` from
`src/lib/mark-placement.ts`, which Tasks 7-9 create)

**Behavior being verified:** The fixture set exercises every taxonomy, every
lifecycle, and includes a mark whose supplied bars were discarded by low
alignment quality.

**Interface under test:** `FIXTURE_MARKS`, `FIXTURE_BARS`,
`FIXTURE_DURATION_SECONDS`

**Files:**
- Create: `src/test-utils/mark-fixtures.ts`
- Test: `src/test-utils/mark-fixtures.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import type { MarkLifecycle, MarkTaxonomy } from "../lib/mark";
import { FIXTURE_BARS, FIXTURE_MARKS } from "./mark-fixtures";

describe("mark fixtures", () => {
	it("cover every taxonomy, every lifecycle, and a discarded-bars case", () => {
		const taxonomies = new Set<MarkTaxonomy>(FIXTURE_MARKS.map((m) => m.taxonomy));
		const lifecycles = new Set<MarkLifecycle>(FIXTURE_MARKS.map((m) => m.lifecycle));

		expect(taxonomies).toEqual(
			new Set(["needs_work", "missed_opportunity", "strong"]),
		);
		expect(lifecycles).toEqual(new Set(["active", "improving", "resolved"]));

		// At least one mark was offered bars and had them discarded, so the
		// canvases have something that proves the degradation path.
		expect(FIXTURE_MARKS.some((m) => m.anchor.type === "timestamp")).toBe(true);

		// At least one bar-anchored mark points at a bar that is NOT on the
		// rendered page, so the unplaced/disclosure path has a fixture too.
		expect(FIXTURE_BARS.some((b) => b.barNumber === 88)).toBe(true);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/test-utils/mark-fixtures.test.ts
```

Expected: FAIL — `Failed to resolve import "./mark-fixtures"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/test-utils/mark-fixtures.ts`:

```ts
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import type { BarLocator } from "../lib/mark-placement";

export const FIXTURE_DURATION_SECONDS = 360;

export const FIXTURE_MARKS: readonly Mark[] = [
	{
		id: "m1",
		anchor: resolveAnchor({ atSeconds: 64, bars: [5, 6], alignmentQuality: 0.95 }),
		taxonomy: "needs_work",
		dimension: "pedaling",
		evidence:
			"pedal held through the bass change at 5.3; the blur between hands is about three times your usual",
		lifecycle: "active",
		confidence: "established",
	},
	{
		id: "m2",
		anchor: resolveAnchor({ atSeconds: 151, bars: [12, 12], alignmentQuality: 0.91 }),
		taxonomy: "missed_opportunity",
		dimension: "dynamics",
		evidence: "the approach to 12 stayed flat; the phrase is asking for more shape",
		lifecycle: "improving",
		confidence: "provisional",
	},
	{
		id: "m3",
		anchor: resolveAnchor({ atSeconds: 252, bars: [30, 32], alignmentQuality: 0.88 }),
		taxonomy: "strong",
		dimension: "phrasing",
		evidence: "the line breathes across 30 to 32 exactly as the slur asks",
		lifecycle: "resolved",
		confidence: "established",
	},
	{
		// Bars WERE supplied ([21, 22]) and resolveAnchor discarded them. This is
		// the fixture that proves a wrong bar number cannot reach the screen.
		id: "m4",
		anchor: resolveAnchor({ atSeconds: 97, bars: [21, 22], alignmentQuality: 0.31 }),
		taxonomy: "needs_work",
		dimension: "timing",
		evidence: "the left hand lagged behind the right through this passage",
		lifecycle: "active",
		confidence: "exploratory",
	},
	{
		// Bar 88 resolves, but it is not on the rendered page: Canvas A must
		// disclose it rather than draw it, and Canvas B must still show it.
		id: "m5",
		anchor: resolveAnchor({ atSeconds: 305, bars: [88, 89], alignmentQuality: 0.97 }),
		taxonomy: "needs_work",
		dimension: "articulation",
		evidence: "the staccato flattened into portato here",
		lifecycle: "active",
		confidence: "established",
	},
	{
		id: "m6",
		anchor: resolveAnchor({ atSeconds: 12, alignmentQuality: 1 }),
		taxonomy: "missed_opportunity",
		dimension: "interpretation",
		evidence: "the opening stated the theme without committing to a character",
		lifecycle: "improving",
		confidence: "exploratory",
	},
];

/**
 * Bar locators as score-ir.ts produces them. measureOn is the id attribute of
 * the measure <g> in the rendered Verovio SVG. Bar 88 is present here but its
 * element is absent from the rendered page in tests — that asymmetry is the
 * point.
 */
export const FIXTURE_BARS: readonly BarLocator[] = [
	{ barNumber: 5, measureOn: "measure-0000000000000005" },
	{ barNumber: 12, measureOn: "measure-0000000000000012" },
	{ barNumber: 30, measureOn: "measure-0000000000000030" },
	{ barNumber: 88, measureOn: "measure-0000000000000088" },
];
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/test-utils/mark-fixtures.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/test-utils/mark-fixtures.ts src/test-utils/mark-fixtures.test.ts && git commit -m "test(marks): add fixture marks covering the full vocabulary"
```

---

### Task 7: Bars resolve through measureOn, never through an index

**Group:** B (sequential — Tasks 7-9 all edit `src/lib/mark-placement.ts`;
depends on Group A for the `Mark` type)

**Behavior being verified:** A bar-anchored mark is placed at the rect belonging
to its bar's `measureOn` id, even when that bar's position in the array differs
from its bar number.

**Interface under test:** `placeMarks()`

**Files:**
- Create: `src/lib/mark-placement.ts`
- Test: `src/lib/mark-placement.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import type { Mark } from "./mark";
import { resolveAnchor } from "./mark";
import type { BarLocator, MeasureRect } from "./mark-placement";
import { GLYPH_OFFSET_PX, placeMarks } from "./mark-placement";

function markAtBars(id: string, bars: readonly [number, number]): Mark {
	return {
		id,
		anchor: resolveAnchor({ atSeconds: 30, bars, alignmentQuality: 1 }),
		taxonomy: "needs_work",
		dimension: "timing",
		evidence: "e",
		lifecycle: "active",
	};
}

describe("placeMarks", () => {
	it("resolves a bar through its measureOn id, not its array position", () => {
		// Bar 7 sits at array index 0 and bar 3 at index 1. An index-based
		// implementation would place bar 7 at bar 3's rect, or miss entirely.
		const bars: BarLocator[] = [
			{ barNumber: 7, measureOn: "m-seven" },
			{ barNumber: 3, measureOn: "m-three" },
		];
		const rects = new Map<string, MeasureRect>([
			["m-seven", { top: 200, left: 400, width: 50, height: 60 }],
			["m-three", { top: 100, left: 20, width: 50, height: 60 }],
		]);

		const { placed, unplaced } = placeMarks(bars, rects, [markAtBars("a", [7, 7])]);

		expect(unplaced).toHaveLength(0);
		expect(placed).toHaveLength(1);
		expect(placed[0].left).toBe(400);
		expect(placed[0].top).toBe(200 - GLYPH_OFFSET_PX);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/lib/mark-placement.test.ts
```

Expected: FAIL — `Failed to resolve import "./mark-placement"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/lib/mark-placement.ts`:

```ts
import type { Mark } from "./mark";
import type { BarIR } from "./score-ir";

/**
 * Exactly the part of score-ir's BarIR that placement needs. Reusing the real
 * contract rather than restating it keeps the two from drifting.
 */
export type BarLocator = Pick<BarIR, "barNumber" | "measureOn">;

/** Container-relative geometry, measured by the caller. */
export interface MeasureRect {
	readonly top: number;
	readonly left: number;
	readonly width: number;
	readonly height: number;
}

export interface PlacedMark {
	readonly mark: Mark;
	readonly top: number;
	readonly left: number;
}

export interface Placement {
	readonly placed: readonly PlacedMark[];
	readonly unplaced: readonly Mark[];
}

/** Vertical clearance so the glyph sits above the staff rather than on it. */
export const GLYPH_OFFSET_PX = 28;

/**
 * Pure bar-to-pixel mapping. Reads no DOM — the caller measures and passes
 * rects in, which is what makes this testable at all (jsdom has no layout
 * engine, so getBoundingClientRect returns zeros there).
 *
 * There is deliberately no fallback coordinate. A mark this function cannot
 * resolve to a real rect comes back in `unplaced` for the caller to route to
 * the timeline canvas. Inventing a position is the defect this module exists
 * to eliminate.
 */
export function placeMarks(
	bars: readonly BarLocator[],
	rectsByMeasureOn: ReadonlyMap<string, MeasureRect>,
	marks: readonly Mark[],
): Placement {
	const measureOnByBar = new Map(bars.map((b) => [b.barNumber, b.measureOn]));
	const placed: PlacedMark[] = [];
	const unplaced: Mark[] = [];

	for (const mark of marks) {
		if (mark.anchor.type !== "bars") continue;
		const measureOn = measureOnByBar.get(mark.anchor.bars[0]);
		const rect = measureOn ? rectsByMeasureOn.get(measureOn) : undefined;
		if (!rect) continue;
		placed.push({ mark, top: rect.top - GLYPH_OFFSET_PX, left: rect.left });
	}

	return { placed, unplaced };
}
```

This is the genuine minimum for this test, which asserts only that a resolvable
mark lands at the right rect. Marks that do not resolve are currently *dropped*,
not reported. Tasks 8 and 9 drive the two reporting paths. Do not populate
`unplaced` yet — doing so makes both of those tests pass on arrival and they
stop proving anything.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/lib/mark-placement.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lib/mark-placement.ts src/lib/mark-placement.test.ts && git commit -m "feat(marks): place marks by measureOn id with no invented coordinates"
```

---

### Task 8: A timestamp-anchored mark is reported, not silently dropped

**Group:** B (sequential — depends on Task 7)

**Behavior being verified:** A timestamp-anchored mark comes back in `unplaced`
so the caller can route it to the timeline, even when a bar in the locator table
could plausibly have matched.

**Interface under test:** `placeMarks()`

**Files:**
- Modify: `src/lib/mark-placement.ts`
- Test: `src/lib/mark-placement.test.ts`

- [ ] **Step 1: Write the failing test**

Append inside the existing `describe("placeMarks", ...)` block:

```ts
	it("reports a timestamp-anchored mark as unplaced rather than dropping it", () => {
		const timestampMark: Mark = {
			id: "stamp",
			anchor: resolveAnchor({
				atSeconds: 97,
				bars: [5, 6],
				alignmentQuality: 0.1,
			}),
			taxonomy: "needs_work",
			dimension: "timing",
			evidence: "e",
			lifecycle: "active",
		};
		const bars: BarLocator[] = [{ barNumber: 5, measureOn: "m-five" }];
		const rects = new Map<string, MeasureRect>([
			["m-five", { top: 100, left: 20, width: 50, height: 60 }],
		]);

		const { placed, unplaced } = placeMarks(bars, rects, [timestampMark]);

		// Bar 5 is right there with a rect — but resolveAnchor threw the bars
		// away, so there is nothing to place against and nothing to guess from.
		// The mark must still surface, or it vanishes from the product entirely.
		expect(placed).toHaveLength(0);
		expect(unplaced.map((m) => m.id)).toEqual(["stamp"]);
	});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/lib/mark-placement.test.ts
```

Expected: FAIL — `expected [] to deeply equal [ 'stamp' ]`. Task 7 `continue`s
past non-bars anchors without recording them.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `src/lib/mark-placement.ts`, change the non-bars branch inside the loop from:

```ts
		if (mark.anchor.type !== "bars") continue;
```

to:

```ts
		if (mark.anchor.type !== "bars") {
			unplaced.push(mark);
			continue;
		}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/lib/mark-placement.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lib/mark-placement.ts src/lib/mark-placement.test.ts && git commit -m "feat(marks): report timestamp-anchored marks for timeline fallback"
```

---

### Task 9: A bar with no rect is reported unplaced, never guessed

**Group:** B (sequential — depends on Task 8)

**Behavior being verified:** A bar-anchored mark whose measure element is not on
the rendered page, or whose bar number is absent from the locator table, comes
back in `unplaced` with no coordinate produced for it.

**Interface under test:** `placeMarks()`

**Files:**
- Modify: `src/lib/mark-placement.ts`
- Test: `src/lib/mark-placement.test.ts`

- [ ] **Step 1: Write the failing test**

Append inside the existing `describe("placeMarks", ...)` block:

```ts
	it("reports a bar that is not on the rendered page as unplaced", () => {
		const bars: BarLocator[] = [
			{ barNumber: 5, measureOn: "m-five" },
			{ barNumber: 88, measureOn: "m-eighty-eight" },
		];
		// Bar 88's element is not in the DOM — it is on another page.
		const rects = new Map<string, MeasureRect>([
			["m-five", { top: 100, left: 20, width: 50, height: 60 }],
		]);

		const { placed, unplaced } = placeMarks(bars, rects, [
			markAtBars("on-page", [5, 6]),
			markAtBars("off-page", [88, 89]),
			markAtBars("unknown-bar", [999, 999]),
		]);

		expect(placed.map((p) => p.mark.id)).toEqual(["on-page"]);
		expect(unplaced.map((m) => m.id)).toEqual(["off-page", "unknown-bar"]);
	});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/lib/mark-placement.test.ts
```

Expected: FAIL — `expected [] to deeply equal [ 'off-page', 'unknown-bar' ]`.
Task 7 `continue`s past a missing rect without recording it.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `src/lib/mark-placement.ts`, change the missing-rect branch inside the loop
from:

```ts
		if (!rect) continue;
```

to:

```ts
		// No fallback coordinate. Inventing a position is the defect this
		// module exists to eliminate — see ScorePanel.tsx:371.
		if (!rect) {
			unplaced.push(mark);
			continue;
		}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/lib/mark-placement.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/lib/mark-placement.ts src/lib/mark-placement.test.ts && git commit -m "feat(marks): report unresolvable bars instead of guessing a position"
```

---

### Task 10: The glyph names its mark in words, not only in colour

**Group:** D, chain D1 (parallel with chain D2 / Task 12 — different files)

**Behavior being verified:** A rendered mark exposes taxonomy, dimension, and
location in its accessible name, and the dimension appears as visible text so
colour is not the sole carrier of meaning.

**Interface under test:** `<MarkGlyph />` rendered DOM

**Files:**
- Create: `src/components/MarkGlyph.tsx`
- Test: `src/components/MarkGlyph.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import { MarkGlyph } from "./MarkGlyph";

const mark: Mark = {
	id: "m1",
	anchor: resolveAnchor({ atSeconds: 64, bars: [5, 6], alignmentQuality: 1 }),
	taxonomy: "needs_work",
	dimension: "pedaling",
	evidence: "pedal held through the bass change",
	lifecycle: "active",
};

describe("MarkGlyph", () => {
	it("names taxonomy, dimension, and location, and shows the dimension as text", () => {
		render(<MarkGlyph mark={mark} expanded={false} onToggle={vi.fn()} />);

		const button = screen.getByRole("button");
		expect(button).toHaveAccessibleName("Needs work: Pedaling, bars 5-6");
		// Colour is not the sole means of conveying the dimension (WCAG 1.4.1):
		// the dimension is present as visible text, not only as the tint dot.
		expect(button).toHaveTextContent("Pedaling");
		expect(button).toHaveAttribute("aria-expanded", "false");
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/components/MarkGlyph.test.tsx
```

Expected: FAIL — `Failed to resolve import "./MarkGlyph"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/components/MarkGlyph.tsx`:

```tsx
import type { CSSProperties } from "react";
import { DIMENSION_COLOR_VAR } from "../lib/dimension-colors";
import type { Mark } from "../lib/mark";
import { anchorLabel, TAXONOMY_GLYPH, TAXONOMY_LABEL } from "../lib/mark";
import { DIMENSION_LABELS } from "../lib/mock-session";

interface MarkGlyphProps {
	mark: Mark;
	expanded: boolean;
	onToggle: (id: string) => void;
	style?: CSSProperties;
}

/**
 * The one visual atom both canvases render. If each canvas drew its own chip
 * the two could diverge silently, and "the same mark renders correctly on both
 * canvases" would stop being enforceable.
 *
 * The dimension tint is a decorative dot, not a background: the --dim-* values
 * are muted mid-tones that would fail a 4.5:1 text gate, and the dimension is
 * carried in text regardless.
 */
export function MarkGlyph({ mark, expanded, onToggle, style }: MarkGlyphProps) {
	const location = anchorLabel(mark.anchor);
	const dimension = DIMENSION_LABELS[mark.dimension];
	const label = `${TAXONOMY_LABEL[mark.taxonomy]}: ${dimension}, ${location}`;

	return (
		<button
			type="button"
			aria-expanded={expanded}
			aria-label={label}
			onClick={() => onToggle(mark.id)}
			className="flex items-center gap-1.5 rounded-full border border-border-subtle bg-surface-raised px-2 py-0.5 text-label-sm text-ink-primary"
			style={style}
		>
			<span
				aria-hidden="true"
				className="h-1.5 w-1.5 rounded-full"
				style={{ backgroundColor: DIMENSION_COLOR_VAR[mark.dimension] }}
			/>
			<span aria-hidden="true">{TAXONOMY_GLYPH[mark.taxonomy]}</span>
			<span>{dimension}</span>
			<span className="text-ink-tertiary">{location}</span>
		</button>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/components/MarkGlyph.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/components/MarkGlyph.tsx src/components/MarkGlyph.test.tsx && git commit -m "feat(marks): add the shared mark glyph rendered by both canvases"
```

---

### Task 11: Lifecycle is rendered from props, never recomputed

**Group:** D, chain D1 (sequential — depends on Task 10)

**Behavior being verified:** The glyph's visual strength follows the `lifecycle`
it is given, including a combination no client-side rule could derive.

**Interface under test:** `<MarkGlyph />` rendered DOM

**Files:**
- Modify: `src/components/MarkGlyph.tsx`
- Test: `src/components/MarkGlyph.test.tsx`

- [ ] **Step 1: Write the failing test**

Append inside the existing `describe("MarkGlyph", ...)` block:

```tsx
	it("takes lifecycle strength from the server-supplied value, not from the mark's content", () => {
		// A `strong` mark that is `improving` is not derivable from anything on
		// the client: taxonomy says the student played well, lifecycle says the
		// baseline is still moving. If the component recomputed lifecycle from
		// mark content, this combination could not survive a render.
		const undeducible: Mark = { ...mark, taxonomy: "strong", lifecycle: "improving" };
		const { rerender } = render(
			<MarkGlyph mark={undeducible} expanded={false} onToggle={vi.fn()} />,
		);
		expect(screen.getByRole("button")).toHaveStyle({ opacity: "0.7" });

		rerender(
			<MarkGlyph
				mark={{ ...undeducible, lifecycle: "resolved" }}
				expanded={false}
				onToggle={vi.fn()}
			/>,
		);
		expect(screen.getByRole("button")).toHaveStyle({ opacity: "0.4" });

		rerender(
			<MarkGlyph
				mark={{ ...undeducible, lifecycle: "active" }}
				expanded={false}
				onToggle={vi.fn()}
			/>,
		);
		expect(screen.getByRole("button")).toHaveStyle({ opacity: "1" });
	});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/components/MarkGlyph.test.tsx
```

Expected: FAIL — `expected element to have style opacity: 0.7`. Task 10 renders
no opacity at all.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `src/components/MarkGlyph.tsx`, extend the import:

```tsx
import {
	anchorLabel,
	LIFECYCLE_OPACITY,
	TAXONOMY_GLYPH,
	TAXONOMY_LABEL,
} from "../lib/mark";
```

and change the button's `style` prop to:

```tsx
			// A lookup, never a computation. Lifecycle is server state; the
			// client is forbidden from deriving or transitioning it.
			style={{ ...style, opacity: LIFECYCLE_OPACITY[mark.lifecycle] }}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/components/MarkGlyph.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/components/MarkGlyph.tsx src/components/MarkGlyph.test.tsx && git commit -m "feat(marks): render lifecycle strength from server state"
```

---

### Task 12: Evidence expands, and exploratory marks are framed as such

**Group:** D, chain D2 (parallel with chain D1 — different files)

**Behavior being verified:** The detail panel shows the mark's evidence and its
location, prefixes an `exploratory` mark's prose to frame it as an early read,
and offers a close affordance.

**Interface under test:** `<MarkDetail />` rendered DOM

**Files:**
- Create: `src/components/MarkDetail.tsx`
- Test: `src/components/MarkDetail.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import { MarkDetail } from "./MarkDetail";

const base: Mark = {
	id: "m1",
	anchor: resolveAnchor({ atSeconds: 97, alignmentQuality: 0 }),
	taxonomy: "needs_work",
	dimension: "timing",
	evidence: "the left hand lagged behind the right through this passage",
	lifecycle: "active",
};

describe("MarkDetail", () => {
	it("shows the evidence and closes on request", async () => {
		const onClose = vi.fn();
		render(<MarkDetail mark={{ ...base, confidence: "established" }} onClose={onClose} />);

		expect(
			screen.getByText(/the left hand lagged behind the right/),
		).toBeInTheDocument();
		expect(screen.getByText(/the left hand lagged/)).not.toHaveTextContent(
			"Early read",
		);

		await userEvent.click(screen.getByRole("button", { name: /close/i }));
		expect(onClose).toHaveBeenCalledTimes(1);
	});

	it("frames an exploratory mark as an early read", () => {
		render(<MarkDetail mark={{ ...base, confidence: "exploratory" }} onClose={vi.fn()} />);

		expect(screen.getByText(/Early read/)).toBeInTheDocument();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/components/MarkDetail.test.tsx
```

Expected: FAIL — `Failed to resolve import "./MarkDetail"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/components/MarkDetail.tsx`:

```tsx
import type { Mark } from "../lib/mark";
import { anchorLabel } from "../lib/mark";

interface MarkDetailProps {
	mark: Mark;
	onClose: () => void;
}

/**
 * The expanded state of a mark, shared by both canvases.
 *
 * `confidence` changes only the wording here. It gates nothing: it never hides
 * a mark, never changes placement, and never suppresses rendering — matching
 * #163's invariant that confidence never gates firing.
 */
export function MarkDetail({ mark, onClose }: MarkDetailProps) {
	const framing = mark.confidence === "exploratory" ? "Early read — " : "";

	return (
		<div
			className="mt-1 rounded-md border border-border-subtle bg-surface-raised p-3"
			aria-label={`Evidence, ${anchorLabel(mark.anchor)}`}
		>
			<p className="text-body-sm text-ink-secondary">
				{framing}
				{mark.evidence}
			</p>
			<button
				type="button"
				onClick={onClose}
				className="mt-2 text-label-sm text-ink-tertiary underline"
			>
				Close
			</button>
		</div>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/components/MarkDetail.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/components/MarkDetail.tsx src/components/MarkDetail.test.tsx && git commit -m "feat(marks): add the expanded evidence panel with confidence framing"
```

---

### Task 13: The score canvas places what it can and discloses what it cannot

**Group:** E, chain E1 (parallel with chain E2 — different files)

**Behavior being verified:** Canvas A renders only marks resolvable to a real
measure element and discloses the count of the rest.

**Interface under test:** `<ScoreMarkLayer />` rendered DOM

**Files:**
- Create: `src/components/ScoreMarkLayer.tsx`
- Modify: `src/test-setup.ts` (add a `ResizeObserver` stub)
- Test: `src/components/ScoreMarkLayer.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it } from "vitest";
import {
	FIXTURE_BARS,
	FIXTURE_MARKS,
} from "../test-utils/mark-fixtures";
import { ScoreMarkLayer } from "./ScoreMarkLayer";

function renderLayer() {
	const ref = createRef<HTMLDivElement>();
	// Bar 88's element is deliberately absent: it models a bar on a page the
	// overlay is not currently showing.
	const onPage = FIXTURE_BARS.filter((b) => b.barNumber !== 88);
	return render(
		<div ref={ref}>
			{onPage.map((b) => (
				<div key={b.measureOn} id={b.measureOn} />
			))}
			<ScoreMarkLayer containerRef={ref} bars={FIXTURE_BARS} marks={FIXTURE_MARKS} />
		</div>,
	);
}

describe("ScoreMarkLayer", () => {
	it("renders resolvable marks and discloses the count of the rest", () => {
		renderLayer();

		// m1 (bar 5), m2 (bar 12), m3 (bar 30) resolve. m4 and m6 are
		// timestamp-anchored; m5 is bar 88, which is not on this page.
		expect(screen.getByLabelText(/Needs work: Pedaling, bars 5-6/)).toBeInTheDocument();
		expect(screen.getByLabelText(/Missed opportunity: Dynamics, bar 12/)).toBeInTheDocument();
		expect(screen.getByLabelText(/Strong: Phrasing, bars 30-32/)).toBeInTheDocument();

		expect(screen.queryByLabelText(/Articulation/)).not.toBeInTheDocument();
		expect(screen.getByText("3 marks not on this page")).toBeInTheDocument();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/components/ScoreMarkLayer.test.tsx
```

Expected: FAIL — `Failed to resolve import "./ScoreMarkLayer"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

First add the `ResizeObserver` stub to `src/test-setup.ts`, appended after the
existing `IntersectionObserver` stub:

```ts
// jsdom does not implement ResizeObserver — stub it globally so components that
// observe container resize (ScoreMarkLayer re-measures because Verovio reflows
// on width change) don't throw during render.
class MockResizeObserver {
	observe = vi.fn();
	disconnect = vi.fn();
	unobserve = vi.fn();
	constructor(_cb: ResizeObserverCallback) {}
}
globalThis.ResizeObserver =
	MockResizeObserver as unknown as typeof ResizeObserver;
```

Then create `src/components/ScoreMarkLayer.tsx`:

```tsx
import { type RefObject, useEffect, useState } from "react";
import type { Mark } from "../lib/mark";
import type { BarLocator, MeasureRect } from "../lib/mark-placement";
import { placeMarks } from "../lib/mark-placement";
import { MarkGlyph } from "./MarkGlyph";

interface ScoreMarkLayerProps {
	containerRef: RefObject<HTMLElement | null>;
	bars: readonly BarLocator[];
	marks: readonly Mark[];
}

/**
 * Canvas A: the DOM adapter over a rendered Verovio SVG.
 *
 * Shallow on purpose. All substance — bar resolution, the no-fallback rule,
 * offsets — lives in mark-placement.ts, because jsdom has no layout engine and
 * anything measured here is untestable. This file only reads rects and hands
 * them over.
 */
export function ScoreMarkLayer({ containerRef, bars, marks }: ScoreMarkLayerProps) {
	const [rects, setRects] = useState<ReadonlyMap<string, MeasureRect>>(new Map());
	const [expandedId, setExpandedId] = useState<string | null>(null);

	useEffect(() => {
		const el = containerRef.current;
		if (!el) return;

		const measure = () => {
			const base = el.getBoundingClientRect();
			const found = new Map<string, MeasureRect>();
			for (const bar of bars) {
				// Attribute selector rather than getElementById: measureOn ids are
				// Verovio-generated and need no CSS escaping this way, and the
				// lookup stays scoped to this score container.
				const node = el.querySelector(`[id="${bar.measureOn}"]`);
				if (!node) continue;
				const r = node.getBoundingClientRect();
				found.set(bar.measureOn, {
					top: r.top - base.top,
					left: r.left - base.left,
					width: r.width,
					height: r.height,
				});
			}
			setRects(found);
		};

		measure();
		// Verovio reflows on width change, so stale rects would place marks on
		// the wrong bars — the exact defect this module exists to prevent.
		const observer = new ResizeObserver(measure);
		observer.observe(el);
		return () => observer.disconnect();
	}, [containerRef, bars]);

	const { placed, unplaced } = placeMarks(bars, rects, marks);

	return (
		<div className="pointer-events-none absolute inset-0">
			{placed.map(({ mark, top, left }) => (
				<div key={mark.id} className="pointer-events-auto absolute" style={{ top, left }}>
					<MarkGlyph
						mark={mark}
						expanded={expandedId === mark.id}
						onToggle={(id) => setExpandedId((cur) => (cur === id ? null : id))}
					/>
				</div>
			))}
			{unplaced.length > 0 && (
				<p className="pointer-events-auto absolute bottom-0 left-0 text-label-sm text-ink-tertiary">
					{unplaced.length === 1
						? "1 mark not on this page"
						: `${unplaced.length} marks not on this page`}
				</p>
			)}
		</div>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/components/ScoreMarkLayer.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/components/ScoreMarkLayer.tsx src/components/ScoreMarkLayer.test.tsx src/test-setup.ts && git commit -m "feat(marks): add the score overlay canvas with unplaced disclosure"
```

---

### Task 14: Tapping a mark on the score canvas expands its evidence

**Group:** E, chain E1 (sequential — depends on Task 13)

**Behavior being verified:** Tapping a placed mark reveals its evidence; tapping
again collapses it.

**Interface under test:** `<ScoreMarkLayer />` rendered DOM

**Files:**
- Modify: `src/components/ScoreMarkLayer.tsx`
- Test: `src/components/ScoreMarkLayer.test.tsx`

- [ ] **Step 1: Write the failing test**

Append inside the existing `describe("ScoreMarkLayer", ...)` block:

```tsx
	it("expands and collapses a mark's evidence on tap", async () => {
		renderLayer();
		const glyph = screen.getByLabelText(/Needs work: Pedaling, bars 5-6/);

		expect(screen.queryByText(/pedal held through the bass change/)).not.toBeInTheDocument();

		await userEvent.click(glyph);
		expect(screen.getByText(/pedal held through the bass change/)).toBeInTheDocument();
		expect(glyph).toHaveAttribute("aria-expanded", "true");

		await userEvent.click(glyph);
		expect(screen.queryByText(/pedal held through the bass change/)).not.toBeInTheDocument();
	});
```

Add the import at the top of the file:

```tsx
import userEvent from "@testing-library/user-event";
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/components/ScoreMarkLayer.test.tsx
```

Expected: FAIL — `Unable to find an element with the text: /pedal held through
the bass change/`. Task 13 tracks `expandedId` but renders no `MarkDetail`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `src/components/ScoreMarkLayer.tsx`, add the import:

```tsx
import { MarkDetail } from "./MarkDetail";
```

and replace the `placed.map(...)` block with:

```tsx
			{placed.map(({ mark, top, left }) => (
				<div key={mark.id} className="pointer-events-auto absolute" style={{ top, left }}>
					<MarkGlyph
						mark={mark}
						expanded={expandedId === mark.id}
						onToggle={(id) => setExpandedId((cur) => (cur === id ? null : id))}
					/>
					{expandedId === mark.id && (
						<MarkDetail mark={mark} onClose={() => setExpandedId(null)} />
					)}
				</div>
			))}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/components/ScoreMarkLayer.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/components/ScoreMarkLayer.tsx src/components/ScoreMarkLayer.test.tsx && git commit -m "feat(marks): expand evidence on tap on the score canvas"
```

---

### Task 15: The timeline canvas shows every mark, positioned by elapsed time

**Group:** E, chain E2 (parallel with chain E1 — different files)

**Behavior being verified:** Canvas B renders every mark including ones the
score canvas cannot place, positioned as a percentage of session duration.

**Interface under test:** `<SessionTimelineStrip />` rendered DOM

**Files:**
- Create: `src/components/SessionTimelineStrip.tsx`
- Test: `src/components/SessionTimelineStrip.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
	FIXTURE_DURATION_SECONDS,
	FIXTURE_MARKS,
} from "../test-utils/mark-fixtures";
import { SessionTimelineStrip } from "./SessionTimelineStrip";

describe("SessionTimelineStrip", () => {
	it("renders every mark, including ones the score canvas cannot place", () => {
		const { container } = render(
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>,
		);

		expect(container.querySelectorAll("button[aria-expanded]")).toHaveLength(
			FIXTURE_MARKS.length,
		);
		// m5 is bar 88 — absent from the score canvas, present here.
		expect(screen.getByLabelText(/Articulation/)).toBeInTheDocument();
		// m4's bars were discarded; it must read as a timestamp.
		expect(screen.getByLabelText(/Needs work: Timing, 1:37/)).toBeInTheDocument();
	});

	it("positions a mark at its share of the session duration", () => {
		render(
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>,
		);

		// m1 is at 64s of 360s = 17.777...%
		const wrapper = screen.getByLabelText(/Pedaling/).parentElement;
		expect(wrapper).toHaveStyle({ left: `${(64 / 360) * 100}%` });
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/components/SessionTimelineStrip.test.tsx
```

Expected: FAIL — `Failed to resolve import "./SessionTimelineStrip"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/components/SessionTimelineStrip.tsx`:

```tsx
import { useState } from "react";
import type { Mark } from "../lib/mark";
import { MarkGlyph } from "./MarkGlyph";

interface SessionTimelineStripProps {
	durationSeconds: number;
	marks: readonly Mark[];
}

/**
 * Canvas B: the complete view.
 *
 * Every mark appears here, including bar-anchored marks the score canvas could
 * not place. Elapsed time is the one coordinate every anchor carries, which is
 * what makes this canvas total and Canvas A the lossy one.
 */
export function SessionTimelineStrip({
	durationSeconds,
	marks,
}: SessionTimelineStripProps) {
	const [expandedId, setExpandedId] = useState<string | null>(null);
	const span = durationSeconds > 0 ? durationSeconds : 1;

	return (
		<div className="relative h-24 w-full border-t border-border-subtle">
			{marks.map((mark) => (
				<div
					key={mark.id}
					className="absolute top-0"
					style={{ left: `${(mark.anchor.atSeconds / span) * 100}%` }}
				>
					<MarkGlyph
						mark={mark}
						expanded={expandedId === mark.id}
						onToggle={(id) => setExpandedId((cur) => (cur === id ? null : id))}
					/>
				</div>
			))}
		</div>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/components/SessionTimelineStrip.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/components/SessionTimelineStrip.tsx src/components/SessionTimelineStrip.test.tsx && git commit -m "feat(marks): add the session timeline canvas as the complete mark view"
```

---

### Task 16: Tapping a mark on the timeline canvas expands its evidence

**Group:** E, chain E2 (sequential — depends on Task 15)

**Behavior being verified:** Tapping a mark on the timeline reveals its
evidence, the same way the score canvas does.

**Interface under test:** `<SessionTimelineStrip />` rendered DOM

**Files:**
- Modify: `src/components/SessionTimelineStrip.tsx`
- Test: `src/components/SessionTimelineStrip.test.tsx`

- [ ] **Step 1: Write the failing test**

Append inside the existing `describe("SessionTimelineStrip", ...)` block:

```tsx
	it("expands and collapses a mark's evidence on tap", async () => {
		render(
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>,
		);
		const glyph = screen.getByLabelText(/Needs work: Timing, 1:37/);

		expect(screen.queryByText(/the left hand lagged/)).not.toBeInTheDocument();

		await userEvent.click(glyph);
		expect(screen.getByText(/the left hand lagged/)).toBeInTheDocument();
		expect(glyph).toHaveAttribute("aria-expanded", "true");

		await userEvent.click(glyph);
		expect(screen.queryByText(/the left hand lagged/)).not.toBeInTheDocument();
	});
```

Add the import at the top of the file:

```tsx
import userEvent from "@testing-library/user-event";
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/components/SessionTimelineStrip.test.tsx
```

Expected: FAIL — `Unable to find an element with the text: /the left hand
lagged/`. Task 15 tracks `expandedId` but renders no `MarkDetail`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `src/components/SessionTimelineStrip.tsx`, add the import:

```tsx
import { MarkDetail } from "./MarkDetail";
```

and replace the `marks.map(...)` block with:

```tsx
			{marks.map((mark) => (
				<div
					key={mark.id}
					className="absolute top-0"
					style={{ left: `${(mark.anchor.atSeconds / span) * 100}%` }}
				>
					<MarkGlyph
						mark={mark}
						expanded={expandedId === mark.id}
						onToggle={(id) => setExpandedId((cur) => (cur === id ? null : id))}
					/>
					{expandedId === mark.id && (
						<MarkDetail mark={mark} onClose={() => setExpandedId(null)} />
					)}
				</div>
			))}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/components/SessionTimelineStrip.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/components/SessionTimelineStrip.tsx src/components/SessionTimelineStrip.test.tsx && git commit -m "feat(marks): expand evidence on tap on the timeline canvas"
```

---

### Task 17: A preview route mounts both canvases in a real browser

**Group:** F (sequential — Task 18 navigates to the route this task creates)

**Behavior being verified:** `/marks-preview` renders both canvases against the
fixture marks, reachable without authentication.

**Note:** this route imports from `src/test-utils/`, so the fixtures enter the
production bundle. That is accepted for a dev preview surface that #158/#159/#162
delete, and it is the reason the route is scoped to preview rather than shipped
as a product surface. Flag it in `/review` rather than working around it.

**Interface under test:** the route module's rendered DOM

**Files:**
- Create: `src/routes/marks-preview.tsx`
- Test: `src/routes/marks-preview.test.tsx`
- Modify: `src/routeTree.gen.ts` (regenerated automatically by the TanStack
  Router Vite plugin on the next `bun run dev` or `bun run build` — do not
  hand-edit)

- [ ] **Step 1: Write the failing test**

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MarksPreview } from "./marks-preview";

describe("marks preview route", () => {
	it("mounts both canvases against the same fixture marks", () => {
		render(<MarksPreview />);

		expect(screen.getByRole("heading", { name: /score overlay/i })).toBeInTheDocument();
		expect(screen.getByRole("heading", { name: /session timeline/i })).toBeInTheDocument();

		// The same mark on both canvases: pedaling bars 5-6 resolves on the
		// score canvas and also appears on the timeline.
		expect(screen.getAllByLabelText(/Needs work: Pedaling, bars 5-6/)).toHaveLength(2);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/routes/marks-preview.test.tsx
```

Expected: FAIL — `Failed to resolve import "./marks-preview"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `src/routes/marks-preview.tsx`:

```tsx
import { createFileRoute } from "@tanstack/react-router";
import { useRef } from "react";
import { ScoreMarkLayer } from "../components/ScoreMarkLayer";
import { SessionTimelineStrip } from "../components/SessionTimelineStrip";
import {
	FIXTURE_BARS,
	FIXTURE_DURATION_SECONDS,
	FIXTURE_MARKS,
} from "../test-utils/mark-fixtures";

export const Route = createFileRoute("/marks-preview")({
	component: MarksPreview,
});

/**
 * Dev preview surface for #157. Deliberately a top-level route rather than a
 * child of /app: /app redirects to /signin when VITE_AUTH_MODE=live, and the
 * a11y run needs to reach this page in a preview build. Removed when the real
 * surfaces (#158/#159/#162) consume the canvases.
 *
 * The measure stand-ins below carry the same ids score-ir emits as
 * BarIR.measureOn, so ScoreMarkLayer's resolution path is exercised for real.
 * Bar 88 is intentionally omitted to exercise the unplaced disclosure.
 */
export function MarksPreview() {
	const scoreRef = useRef<HTMLDivElement>(null);
	const onPage = FIXTURE_BARS.filter((b) => b.barNumber !== 88);

	return (
		<main className="mx-auto max-w-3xl px-6 py-12">
			<h1 className="mb-8 text-display-sm text-ink-primary">Mark system preview</h1>

			<h2 className="mb-2 text-label-md text-ink-secondary">Score overlay</h2>
			<div ref={scoreRef} className="score-container relative mb-12 h-64 border border-border-subtle">
				{onPage.map((b, i) => (
					<div
						key={b.measureOn}
						id={b.measureOn}
						className="absolute h-24 w-24 border border-border-subtle"
						style={{ top: 80, left: 24 + i * 140 }}
					/>
				))}
				<ScoreMarkLayer containerRef={scoreRef} bars={FIXTURE_BARS} marks={FIXTURE_MARKS} />
			</div>

			<h2 className="mb-2 text-label-md text-ink-secondary">Session timeline</h2>
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
			/>
		</main>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/routes/marks-preview.test.tsx
```
Expected: PASS

Then confirm the route tree regenerates and the route serves:

```bash
bun run build && bunx vite preview --port 4173 --strictPort &
sleep 8 && curl -s -o /dev/null -w "%{http_code}\n" http://localhost:4173/marks-preview
```
Expected: `200`. Stop the preview server afterwards.

- [ ] **Step 5: Commit**

```bash
git add src/routes/marks-preview.tsx src/routes/marks-preview.test.tsx src/routeTree.gen.ts && git commit -m "feat(marks): add the /marks-preview dev route mounting both canvases"
```

---

### Task 18: Real-browser axe covers the mark surfaces in both themes

**Group:** F (sequential — depends on Task 17)

**Behavior being verified:** `/marks-preview` has no colour-contrast violations
in either theme, verified by axe against a real preview build.

**Interface under test:** the rendered page, via `@axe-core/playwright`

**Files:**
- Modify: `tests/a11y.spec.ts`

- [ ] **Step 1: Write the failing test**

In `tests/a11y.spec.ts`, replace the `THEME_CASES` constant with:

```ts
const THEME_CASES = [
	{ theme: "light", path: "/privacy" },
	{ theme: "dark", path: "/signin" },
	// #157: the mark canvases. A top-level route, so it renders in a preview
	// build without auth. axe's color-contrast rule needs real layout and
	// silently SKIPS in jsdom, so this is the only place mark contrast is
	// actually verified — never assert it from vitest.
	{ theme: "light", path: "/marks-preview" },
	{ theme: "dark", path: "/marks-preview" },
] as const;
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun run test:a11y
```

Expected: one of two outcomes, both legitimate.

- FAIL on a genuine contrast violation in the mark components — fix it in
  Step 3. This is the outcome the task is designed to catch.
- PASS on all four cases. The mark chip uses `ink-primary` on `surface-raised`,
  a pair `src/styles/tokens.contrast.test.ts` already asserts at 4.5:1 in both
  columns, so passing on arrival is plausible. It still has to be *run*: this is
  the only place mark contrast is verified at all, since axe's `color-contrast`
  rule needs real layout and silently skips in jsdom. Record the actual result.

If it errors on navigation (`net::ERR_ABORTED`, or a 404 render), Task 17 has
not landed — that is a sequencing mistake, not a contrast finding.

- [ ] **Step 3: Implement the minimum to make the test pass**

If axe reports a contrast violation, fix the token usage in the offending
component — do not exclude the element and do not lower a threshold. The
expected-safe pairs are already `ink-primary`/`ink-tertiary` on
`surface-raised`, which `src/styles/tokens.contrast.test.ts` asserts in both
columns. If `text-ink-tertiary` on `surface-raised` fails at 4.5:1 for the
small location text, promote that span to `text-ink-secondary`:

```tsx
			<span className="text-ink-secondary">{location}</span>
```

in `src/components/MarkGlyph.tsx`.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun run test:a11y
```
Expected: PASS — four cases, no violations.

- [ ] **Step 5: Commit**

```bash
git add tests/a11y.spec.ts src/components/MarkGlyph.tsx && git commit -m "test(marks): cover the mark canvases with real-browser axe in both themes"
```

---

### Task 19: The contract harness goes green and all gates pass

**Group:** G (depends on Group F)

**Behavior being verified:** The Task 1 harness — the issue's headline success
criterion — passes, and the full verification suite is clean.

**Interface under test:** both canvases together, via the contract harness

**Files:**
- Modify: `src/components/mark-canvases.contract.test.tsx` (only if it needs
  correcting; prefer fixing the implementation)

- [ ] **Step 1: Write the failing test**

Already written in Task 1. Do not rewrite it to fit the implementation. If it
fails, the implementation is wrong unless the harness encodes a factually
wrong expectation, in which case fix the harness and say so explicitly in the
commit message.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bunx vitest run src/components/mark-canvases.contract.test.tsx
```

Expected at the start of this task: it may already PASS, since Groups D and E
built exactly what it demands. That is the intended outcome — the harness was
red from Task 1 through Task 18 and turning green here is the signal the
feature is complete.

- [ ] **Step 3: Implement the minimum to make the test pass**

If any assertion fails, fix the implementation, not the harness. The likely
failure is the `1:37` accessible-name lookup in the third case: confirm
`anchorLabel` formats 97 seconds as `1:37` and that `MarkGlyph` puts the
location in its `aria-label`.

- [ ] **Step 4: Run test — verify it PASSES**

Run the full verification set from `.worktrees/issue-157-mark-system/apps/web`:

```bash
bun run test
bunx tsc --noEmit
bun run lint
bun run test:a11y
```

Expected:
- `bun run test`: PASS, with **zero** failing files. The known-red window is
  closed. Note `src/lib/score-worker.integration.test.ts` has a sub-100ms perf
  assertion that flaked once historically and never reproduced — it is
  pre-existing and unrelated. If it flakes, re-run it alone to confirm, and do
  not "fix" it.
- `bunx tsc --noEmit`: exit 0.
- `bun run lint`: exit 0, with the repo's pre-existing warnings unchanged
  (69 at branch point — record the actual count and confirm no new ones).
- `bun run test:a11y`: PASS, four cases.

Then the **deciding check** — a real-browser click-through, which for this UI
work outranks the test count:

```bash
bun run dev
```

Open `http://localhost:3000/marks-preview` and confirm each of the following by
eye, in **both** themes (toggle via the app's theme control, or
`document.documentElement.dataset.theme = "dark"` in the console):

1. The pedaling mark (bars 5-6) appears on **both** canvases, reading
   identically on each.
2. Tapping it on the score canvas expands the evidence; tapping again collapses.
3. Tapping it on the timeline canvas does the same.
4. The timing mark reads **`1:37`** and shows **no bar number anywhere**, even
   though bars [21, 22] were supplied to `resolveAnchor`.
5. The score canvas discloses "3 marks not on this page"; the articulation mark
   (bar 88) is absent there and present on the timeline.
6. Resizing the window keeps the score marks above their measure boxes rather
   than drifting.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "test(marks): close the contract harness — both canvases share one vocabulary"
```

Then run `/review` on the branch. Gate on `VERDICT: READY` before `/ship`.

---

## Out of scope, flagged not fixed

- `apps/ios/.../DesignSystem/Tokens/` drifts from the web token table (#156).
  Unowned; do not touch.
- `readTokenTable()` in `src/test-utils/read-tokens.ts` parses only the
  `@theme` and `html[data-theme="dark"]` blocks, so the light `--dim-*` values
  in the standalone `:root` block at `app.css:284` are invisible to the token
  contrast harness. This plan avoids depending on them rather than widening
  #156's harness as a side effect. Worth a follow-up issue.
- `ScorePanel.tsx` / `ScoreAnnotation.tsx` still contain the index-based bar
  lookup. They are deleted by #164; this plan adds the replacement alongside
  and removes nothing.
