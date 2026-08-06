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

## Prerequisites — do these before Task 1

A fresh worktree does **not** inherit enough to run the verification commands.
Skipping this makes `bunx tsc --noEmit` report **209 errors** that look exactly
like pre-existing repo breakage and are not. Three subagents have already been
caught by a thinner version of this.

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-157-mark-system

# 1. Per-app deps. There is NO root package.json — a root-level `bun install`
#    fails with "Bun could not find a package.json file to install from".
(cd apps/web && bun install)

# 2. apps/api deps are ALSO required, even though this issue never edits
#    apps/api. apps/web/tsconfig.json includes **/*.ts and the web app imports
#    types from apps/api, so tsc follows those imports into apps/api sources.
#    skipLibCheck does not skip them — they are .ts, not .d.ts.
(cd apps/api && bun install)

# 3. The WASM pkg/ dirs are gitignored build artifacts a new worktree does not
#    get. Copy, do NOT rebuild.
cp -R /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/piece-identify/pkg \
      apps/api/src/wasm/piece-identify/
cp -R /Users/jdhiman/Documents/crescendai/apps/api/src/wasm/score-analysis/pkg \
      apps/api/src/wasm/score-analysis/
```

Confirm the clean baseline before writing any code. These numbers were measured
on this branch at `db843868` and are what "unchanged" means later:

```bash
cd apps/web
bunx tsc --noEmit ; echo "exit=$?"     # expect: no output, exit=0
bun run lint      ; echo "exit=$?"     # expect: exit=0, 107 warnings, 23 infos, 151 files
bun run test                            # expect: 1 failed | 215 passed (216); see below
```

**The one baseline failure is real and expected.**
`src/lib/score-worker.integration.test.ts > IR walk cost (parseScoreIR alone) is
under 100ms` fails under full-suite load and passes 9/9 when run alone. It is a
load-dependent perf assertion, unrelated to this issue. Do not fix it, do not
skip it. The disambiguator is:

```bash
bunx vitest run src/lib/score-worker.integration.test.ts   # expect: 9 passed
```

Note this corrects a belief carried in the session handoff that the test "flaked
once historically and never reproduced" — it reproduces reliably under load on
this machine.

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
From Task 1 until Task 19, **two** commands are expected to be red, and both
recover only when Task 19 closes the window:

**1. `bun run test`** has exactly one failing file:
`src/components/mark-canvases.contract.test.tsx`. Any *other* failing file is a
real regression (excluding the known `score-worker` perf flake described in
Prerequisites).

**2. `bunx tsc --noEmit` exits 2**, with all errors confined to that same
harness file — unresolved imports of `../test-utils/mark-fixtures`,
`./ScoreMarkLayer`, and `./SessionTimelineStrip`, plus implicit-`any` callback
parameters that are downstream of those unresolved imports (with `FIXTURE_BARS`
unresolved, the `(b)` params have nothing to infer from).

This second consequence was discovered during Task 2, not anticipated when the
plan was written. It matters because most tasks' Step 4 says "confirm
`bunx tsc --noEmit` exits 0", which is **not achievable** until Task 19.

**How every task from here to Task 18 must check typecheck instead:**

```bash
bunx tsc --noEmit 2>&1 | grep -v "mark-canvases.contract.test.tsx"
```

Expect **no output**. Any line that survives that filter is a real error you
introduced. If errors appear under `../api/...`, the worktree Prerequisites
regressed — go re-do them rather than reporting the errors as pre-existing.

Do not skip, `.skip`, or delete the harness to get a green run, and do not add
stubs to satisfy its imports.

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
Group F (sequential, depends E): Task 17 -> 20 -> 18       (each needs the prior's route state)
Group G (depends on F)         : Task 19                   (harness green + full gates)
```

Task 20 is numbered last but executes **between** 17 and 18: it was added after
the challenge review, and renumbering would have invalidated every
cross-reference in the plan. Group letters and this ordering line are
authoritative, not the numbers.

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

**Then immediately format the file and amend.** Bypassing pre-commit also
bypasses the Biome formatter, which leaves `bun run lint` at **exit 1** with a
`format` error on this file. `lint-web` is a pre-push blocking gate in this
repo, so an unformatted harness breaks a gate for reasons that have nothing to
do with the intentional-red test semantics. This was missed on the first build
pass and caught by the controller after Task 5:

```bash
bunx biome format --write src/components/mark-canvases.contract.test.tsx
git add src/components/mark-canvases.contract.test.tsx && git commit --amend --no-edit --no-verify
```

Confirm `bun run lint` returns to exit 0 with the baseline 107 warnings / 23
infos, and that the harness is still red with a module-resolution error.

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

### Task 20: A real Verovio score proves the measureOn chain end to end

**Group:** F (sequential — runs after Task 17, before Task 18)

**Behavior being verified:** Against a **real** Verovio engraving of a real
piece, a bar-anchored mark renders positioned over that bar's actual measure
element — proving `BarIR.measureOn` really is the SVG element's `id` at render
time, not merely by construction in `score-ir.ts`.

**Interface under test:** `/marks-preview` in a real browser

**Why Playwright and not vitest:** this is the one claim jsdom structurally
cannot check. Rendering a real score needs the Verovio WASM toolkit, a worker,
and a layout engine; jsdom has none of the three, and its
`getBoundingClientRect()` returns zeros. A real browser has all three. Every
other test in this plan uses stand-in `<div id="measure-...">` elements, which
verify the resolution *logic* against correctly-shaped ids but can never verify
that Verovio emits those ids.

**Files:**
- Modify: `src/routes/marks-preview.tsx` (add the real-piece section)
- Create: `tests/marks.spec.ts`
- Create: `playwright.marks.config.ts`
- Modify: `package.json` (add the `test:marks` script)

- [ ] **Step 1: Write the failing test**

Create `playwright.marks.config.ts`:

```ts
import { defineConfig } from "@playwright/test";

export default defineConfig({
	testMatch: ["tests/marks.spec.ts"],
	use: {
		headless: true,
		baseURL: "http://localhost:4173",
	},
	// Verovio WASM init plus a real score load is slow on a cold preview build.
	timeout: 120000,
	webServer: {
		command: "bun run build && bunx vite preview --port 4173 --strictPort",
		port: 4173,
		reuseExistingServer: !process.env.CI,
		timeout: 180000,
	},
});
```

Create `tests/marks.spec.ts`:

```ts
import { expect, test } from "@playwright/test";

test("a mark sits over its real measure on a real Verovio engraving", async ({
	page,
}) => {
	await page.goto("/marks-preview");

	const realScore = page.locator("[data-testid='real-score']");
	// Verovio emits <g class="measure" id="..."> once the toolkit has rendered.
	const measures = realScore.locator("g.measure");
	await expect(measures.first()).toBeVisible({ timeout: 90000 });

	// The preview anchors its real-score mark to the FIRST bar the IR reports,
	// so the element it resolves to must exist and the glyph must be visible.
	const glyph = realScore.locator("button[aria-expanded]").first();
	await expect(glyph).toBeVisible();

	// The load-bearing assertion: the glyph's box overlaps the measure element
	// it claims to mark. A wrong-bar or invented position fails here.
	const markedId = await glyph.getAttribute("data-measure-on");
	expect(markedId).toBeTruthy();
	const target = realScore.locator(`g.measure[id="${markedId}"]`);
	await expect(target).toHaveCount(1);

	const glyphBox = await glyph.boundingBox();
	const targetBox = await target.boundingBox();
	expect(glyphBox).not.toBeNull();
	expect(targetBox).not.toBeNull();
	if (!glyphBox || !targetBox) throw new Error("unreachable");

	// Horizontal overlap: the glyph starts within the measure's horizontal span.
	expect(glyphBox.x).toBeGreaterThanOrEqual(targetBox.x - 1);
	expect(glyphBox.x).toBeLessThanOrEqual(targetBox.x + targetBox.width);
	// Vertical: the glyph sits ABOVE the staff, by GLYPH_OFFSET_PX.
	expect(glyphBox.y).toBeLessThan(targetBox.y);

	// And the degradation constraint holds on a real score too.
	await expect(realScore).not.toContainText("bars 21");
});
```

Add to `package.json` scripts:

```json
		"test:marks": "playwright test --config playwright.marks.config.ts",
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun run test:marks
```

Expected: FAIL — `expect(locator).toBeVisible()` times out on
`[data-testid='real-score']`, which does not exist yet. If it fails instead on
Verovio never rendering `g.measure`, that is a *finding*, not a test bug: it
would mean the whole `measureOn` premise is wrong. Report it rather than
working around it.

- [ ] **Step 3: Implement the minimum to make the test pass**

Two changes. First, `MarkGlyph` must expose which measure it was placed against
so the test can assert overlap rather than mere presence. In
`src/components/MarkGlyph.tsx`, add an optional prop and forward it:

```tsx
interface MarkGlyphProps {
	mark: Mark;
	expanded: boolean;
	onToggle: (id: string) => void;
	style?: CSSProperties;
	/** The measure element this glyph was placed against, for E2E assertions. */
	measureOn?: string;
}
```

and on the `<button>`:

```tsx
			data-measure-on={measureOn}
```

Then in `src/components/ScoreMarkLayer.tsx`, `placeMarks` already knows the
resolution; pass it through. Change `PlacedMark` in
`src/lib/mark-placement.ts` to carry it:

```ts
export interface PlacedMark {
	readonly mark: Mark;
	readonly top: number;
	readonly left: number;
	readonly measureOn: string;
}
```

and in `placeMarks`, the push becomes:

```ts
		placed.push({
			mark,
			top: rect.top - GLYPH_OFFSET_PX,
			left: rect.left,
			measureOn,
		});
```

(`measureOn` is already in scope and provably non-undefined at that point,
because `rect` was looked up through it.)

In `ScoreMarkLayer`, forward it:

```tsx
			{placed.map(({ mark, top, left, measureOn }) => (
				<div key={mark.id} className="pointer-events-auto absolute" style={{ top, left }}>
					<MarkGlyph
						mark={mark}
						measureOn={measureOn}
						expanded={expandedId === mark.id}
						onToggle={(id) => setExpandedId((cur) => (cur === id ? null : id))}
					/>
					{expandedId === mark.id && (
						<MarkDetail mark={mark} onClose={() => setExpandedId(null)} />
					)}
				</div>
			))}
```

Second, add the real-piece section to `src/routes/marks-preview.tsx`. Add these
imports:

```tsx
import { useEffect, useMemo, useRef, useState } from "react";
import { resolveAnchor, type Mark } from "../lib/mark";
import type { BarLocator } from "../lib/mark-placement";
import { scoreRenderer } from "../lib/score-renderer";
```

and this component, rendered between the synthetic score section and the
timeline:

```tsx
const REAL_PIECE_ID = "chopin.ballades.1";

/**
 * The only place in #157 where marks meet a real Verovio engraving. Everything
 * else uses stand-in divs, which verify the resolution logic against
 * correctly-shaped ids but cannot verify that Verovio emits those ids at all.
 */
function RealScoreSection() {
	const containerRef = useRef<HTMLDivElement>(null);
	const [svg, setSvg] = useState<string | null>(null);
	const [bars, setBars] = useState<readonly BarLocator[]>([]);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		let cancelled = false;
		async function load() {
			try {
				const result = await scoreRenderer.load(REAL_PIECE_ID);
				if (cancelled) return;
				if (result === "failed") {
					setError("Score failed to load");
					return;
				}
				const page = await scoreRenderer.getPage(REAL_PIECE_ID, 1);
				if (cancelled) return;
				// Page 1 only: bars on later pages exercise the unplaced path,
				// which the synthetic section already covers deterministically.
				setBars(
					result.ir.bars
						.filter((b) => b.pageN === 1)
						.map((b) => ({ barNumber: b.barNumber, measureOn: b.measureOn })),
				);
				setSvg(page);
			} catch (e) {
				if (!cancelled) setError(String(e));
			}
		}
		load();
		return () => {
			cancelled = true;
		};
	}, []);

	// Anchor a mark to the first bar the IR actually reports, so this never
	// depends on a hardcoded bar number surviving a re-engraving.
	const marks = useMemo<readonly Mark[]>(() => {
		const first = bars[0];
		if (!first) return [];
		return [
			{
				id: "real-1",
				anchor: resolveAnchor({
					atSeconds: 30,
					bars: [first.barNumber, first.barNumber],
					alignmentQuality: 1,
				}),
				taxonomy: "needs_work",
				dimension: "pedaling",
				evidence: "pedal held through the bass change",
				lifecycle: "active",
				confidence: "established",
			},
		];
	}, [bars]);

	// Injected imperatively into a dedicated child node, matching the
	// established pattern at src/scorehost/score-host.ts:382. Two reasons this
	// is not React's dangerouslySetInnerHTML: it follows the code already in
	// the repo, and it keeps the SVG in a sibling of the mark layer so React
	// never owns or re-reconciles Verovio's DOM.
	//
	// Trust boundary: the markup is Verovio's own output, produced by our
	// worker from copyright-cleared score bytes we fetch. No user-supplied
	// content reaches this string. That is the same boundary score-host.ts
	// already accepts.
	const svgHostRef = useRef<HTMLDivElement>(null);
	useEffect(() => {
		if (svgHostRef.current && svg) svgHostRef.current.innerHTML = svg;
	}, [svg]);

	if (error) {
		return <p className="text-danger">{error}</p>;
	}

	return (
		<div
			data-testid="real-score"
			ref={containerRef}
			className="score-container relative mb-12 min-h-64 border border-border-subtle"
		>
			<div ref={svgHostRef} />
			{!svg && <p className="text-ink-tertiary">Loading score...</p>}
			{svg && <ScoreMarkLayer containerRef={containerRef} bars={bars} marks={marks} />}
		</div>
	);
}
```

and in `MarksPreview`'s JSX, between the synthetic section and the timeline:

```tsx
			<h2 className="mb-2 text-label-md text-ink-secondary">
				Score overlay (real engraving)
			</h2>
			<RealScoreSection />
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bunx vitest run src/lib/mark-placement.test.ts src/components src/routes
bunx tsc --noEmit
bun run test:marks
```

Expected: PASS. The vitest run is included because `PlacedMark` gained a field —
Tasks 7-9's assertions use `.map((p) => p.mark.id)` and `placed[0].left`, so
they are unaffected, but confirm rather than assume.

- [ ] **Step 5: Commit**

```bash
git add src/routes/marks-preview.tsx src/components/MarkGlyph.tsx src/components/ScoreMarkLayer.tsx src/lib/mark-placement.ts tests/marks.spec.ts playwright.marks.config.ts package.json && git commit -m "test(marks): prove the measureOn chain against a real Verovio engraving"
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

Expected, measured against the Prerequisites baseline:

- `bun run test`: **1 failed | N passed**, where the single failure is
  `src/lib/score-worker.integration.test.ts > IR walk cost (parseScoreIR alone)
  is under 100ms` — the known load-dependent perf flake, unchanged from the
  branch-point baseline. The known-red window from Task 1 is closed: the
  contract harness must now be green. Any failure other than the perf
  assertion is a real regression. Confirm the flake with:

  ```bash
  bunx vitest run src/lib/score-worker.integration.test.ts   # expect: 9 passed
  ```

  Do not "fix" it and do not skip it.
- `bunx tsc --noEmit`: exit 0, no output. (If this reports errors in
  `../api/...`, the Prerequisites were skipped — go back and do them rather
  than reporting the errors as pre-existing.)
- `bun run lint`: exit 0, with **no new warnings** relative to the branch-point
  baseline of 107 warnings / 23 infos across 151 files. Compare the delta, not
  the absolute number — the absolute drifts with unrelated merges.
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

---

## Challenge Review

Reviewed 2026-08-06. Every finding below marked 9/10 or 10/10 was verified by
running a command in this worktree, not by reasoning about likelihood — per the
#163 lesson that a P0 survived four gates because everyone argued about whether
an input could occur instead of constructing it.

### CEO Pass

**Premise.** Not re-litigated; the design is approved (#154 brainstorm
2026-08-04). The plan matches the spec's goal exactly and does not drift into
#158/#159/#162/#164 territory. The one scope addition beyond the spec's
"no route wiring" line is `/marks-preview`, which the user explicitly chose
this session as the a11y and click-through mount.

**Existing coverage.** `ScorePanel.tsx:347-375` + `ScoreAnnotation.tsx` already
do a version of this. The plan correctly reuses the *technique* and rejects the
*components*, and correctly identifies that `ScorePanel`'s bar lookup is broken.
`score-ir.ts` already supplies `BarIR{barNumber, measureOn, pageN}`, and the
plan consumes it rather than inventing a parallel mapping — this is the single
best decision in the plan.

**Scope.** 13 new files, 19 tasks. Above the 8-file complexity smell threshold.
Checked whether it can be smaller: `MarkDetail` could fold into `MarkGlyph`, and
`mark-placement.ts` could live inside `ScoreMarkLayer`. Both were rejected for
good reasons stated in the spec (shared by two canvases; jsdom has no layout
engine). The count is earned, not padded.

**Minimum viable version.** Groups A-C alone (Tasks 2-9) give the vocabulary and
the placement function — the entire correctness bet — with no UI. If time
collapsed, that is what to keep. The plan already marks it
`[SHIPS INDEPENDENTLY]`.

**12-month alignment.**

```
CURRENT STATE                    THIS PLAN                  12-MONTH IDEAL
marks exist only as chat-era  →  one vocabulary, two    →   every surface renders
observation badges welded to     canvases, fixture-fed,     server-produced marks
AppChat, with an index-based     wrong-bar defect made      through one component;
bar bug and no timeline canvas   unrepresentable            no chat shell remains
```

Moves toward the ideal. The one debt it creates is `/marks-preview` importing
`src/test-utils/`, which puts fixtures in the production bundle — already flagged
in Task 17 and deleted with the route.

### Engineering Pass

#### [BLOCKER] (confidence: 10/10) — the plan's typecheck command fails with 209 errors in a fresh worktree

`bunx tsc --noEmit` from `apps/web` does **not** check only `apps/web`.
`apps/web/tsconfig.json` sets `"include": ["**/*.ts", "**/*.tsx"]` and the app
imports types from `apps/api`, so tsc follows those imports into `apps/api`
sources. `skipLibCheck` does not skip them — they are `.ts`, not `.d.ts`.

Verified in this worktree, in order:

1. Fresh worktree, `apps/web` deps only: **5 errors**, including the two
   `TS2307` on `wasm-bridge.ts:196-197` the handoff warned about. The gitignored
   `pkg/` dirs are not inherited by a new worktree.
2. After `cp -R` of both `pkg/` dirs from the primary checkout: **209 errors** —
   the WASM errors cleared and exposed a second layer, every `apps/api`
   dependency unresolvable because `apps/api` had no `node_modules`.
3. After `bun install` in `apps/api`: **0 errors, exit 0.**

This is exactly the failure mode that already burned three subagents, who each
ran the command, saw errors, and called them pre-existing. Running a command
proves errors are real; it does not prove they are pre-existing.

**Required change:** add a Prerequisites section to the plan, before Task 1,
with all three setup steps and the expected clean baseline.

#### [BLOCKER] (confidence: 9/10) — Task 19's "zero failing files" is unachievable as written

Verified: `bun run test` on the untouched branch is **1 failed | 215 passed
(216 tests), 1 failed | 54 passed (55 files)**. The failure is
`src/lib/score-worker.integration.test.ts > IR walk cost (parseScoreIR alone) is
under 100ms`.

Run in isolation, that same file is **9 passed / 9**, in 26s.

So the flake is real and load-dependent, and it **does** reproduce — which
corrects the standing belief (carried in the session handoff and repeated in
Task 19) that it "flaked once historically and never reproduced." A build agent
told to expect zero failures will either chase a phantom regression or declare
the plan failed.

**Required change:** state the true baseline in Task 19 and make the isolation
re-run the disambiguator, not an optional afterthought.

#### [BLOCKER] (confidence: 9/10) — the lint baseline number in Task 19 is wrong

Task 19 says "the repo's pre-existing warnings unchanged (69 at branch point)."
Verified actual `apps/web` baseline: **exit 0, 107 warnings, 23 infos, 151 files
checked.** The 69 figure belongs to a different scope (it appears in session
memory for `apps/api`) and was transplanted. An agent asked to confirm 69 will
conclude it introduced 38 warnings.

**Required change:** correct to 107 warnings / 23 infos, and phrase the check as
"no *new* warnings" rather than an absolute count, since the absolute number
drifts with unrelated merges.

#### [RISK] (confidence: 9/10) — jsdom proves which marks render, never where

Verified by probe: every `getBoundingClientRect()` in jsdom returns
`{top:0,left:0,width:0,height:0}`. A probe replicating `ScoreMarkLayer`'s effect
resolved 2 of 3 measure elements, all with zero rects, rendered the two
resolvable marks, omitted the third, and produced the correct "1 not on this
page" disclosure.

So Task 13's test genuinely verifies **which** marks render and that the
disclosure count is right — the behaviourally important half. It cannot verify
coordinates at all. The arithmetic *is* covered, by Tasks 7-9 against synthetic
non-zero rects. The genuinely untested surface is the four-line coordinate
translation inside the effect (`r.top - base.top`).

This validates the spec's claim that `placeMarks`' purity buys real testability
rather than relocating untested code: what got relocated into the untestable
adapter is 4 lines, not the policy. **Watch:** click-through step 6 (resize) is
the only check on those 4 lines. Do not drop it under time pressure.

#### [RISK] (confidence: 8/10) — no automated test ever renders a real Verovio SVG

Every measure element in this plan — in the contract harness, in Task 13, and in
`/marks-preview` itself — is a stand-in `<div id="measure-...">`. The plan
therefore proves the resolution chain works against ids *shaped like*
`BarIR.measureOn`, not that Verovio and `score-ir` actually put those ids in the
DOM at render time.

The supporting evidence is strong but indirect: `score-ir.ts:185` builds
`measureByMeasureOn` from the timemap, `extractMeasureNoteMap` (`:150`) collects
measure ids from the SVG by `class="...measure..."`, and `:242` writes
`measureOn: measureId` — so the two are equal by construction. Strong is not the
same as observed.

**Fallback:** the click-through in Task 19 is currently against stand-in divs
too, so nothing in the plan closes this. See the QUESTION below.

#### [RISK] (confidence: 6/10) — the known-red window spans 18 tasks

Task 1 commits a deliberately failing harness with `--no-verify` and it stays red
until Task 19. Per session memory, `pre-push` blocks on
`check-api`/`lint-api`/`check-web`/`lint-web` but tests are opt-in, so this
should not block anything mechanically. The real cost is that an interrupted
session leaves a branch that reads as broken. The plan's "Known-red window"
section mitigates this and should stay prominent. Acceptable — the alternative
(landing the harness last) would mean the headline success criterion is written
after the code that satisfies it, which is worse.

#### Module Depth Audit

| Module | Interface | Hidden | Verdict |
|---|---|---|---|
| `lib/mark.ts` | 3 fns, 6 types, 4 const tables (~13 names) | brand, threshold, `m:ss` formatting, plural rule, universality of `atSeconds` | **DEEP** — but the widest interface in the plan; watch const-table sprawl as later issues add taxonomy metadata |
| `lib/mark-placement.ts` | 1 fn, 4 types, 1 const | the whole `barNumber -> measureOn -> rect` chain, the no-fallback policy, page filtering as emergent behaviour | **DEEP** |
| `components/MarkGlyph.tsx` | 1 component, 4 props | glyph selection, tint lookup, lifecycle opacity, accessible-name composition | **MEDIUM**, justified — it is the mechanism by which "one vocabulary" is enforceable |
| `components/MarkDetail.tsx` | 1 component, 2 props | confidence framing, anchor labelling | **MEDIUM** |
| `components/ScoreMarkLayer.tsx` | 1 component, 3 props | DOM lookup, coordinate translation, resize re-measure, disclosure | **SHALLOW by design**, justified in the spec — the substance was deliberately pushed into `mark-placement.ts` so it could be tested |
| `components/SessionTimelineStrip.tsx` | 1 component, 2 props | time→percent positioning, completeness guarantee | **MEDIUM** |

No shallow-module smell beyond the one that is deliberate and argued.

#### Test Philosophy Audit

Every test in the plan exercises a public interface: exported functions, or
rendered DOM through accessible names and visible text. No test mocks an
internal collaborator (`vi.fn()` appears only as a *prop* — a callback the
component's contract requires, not a stubbed collaborator). No test asserts on
internal state. No test calls a private method. No shape-only tests.

Task 11 deserves specific credit: rendering a `strong` mark with lifecycle
`improving` is a combination no client-side rule could derive, which is a real
test of "lifecycle comes from server state" rather than a restatement of it.

#### Vertical Slice Audit

Checked every task for fail-first honesty after the plan's own self-review pass:

- Task 2 → 3: Task 2's `resolveAnchor` always degrades, so Task 3's bars case
  genuinely fails first. **Sound.**
- Task 7 → 8 → 9: Task 7 `continue`s past unplaceable marks without recording
  them, so both reporting paths genuinely fail first. **Sound.**
- Task 10 → 11: Task 10 renders no opacity, so Task 11 genuinely fails first.
  **Sound.**

On the question of whether deliberately-incomplete intermediates are a trap for
a build agent that "helpfully" completes them: they are, and the plan already
names the trap explicitly in its "On one test per task" section. That is the
right mitigation — the alternative (writing complete implementations early) makes
three tests pass on arrival and prove nothing. Keep the warning prominent.

Tasks 3, 4, 12, and 15 each contain two sibling `it` blocks driving one
implementation change. This is within the spirit of one-slice-per-task and the
plan documents it. Not flagged.

#### Test Coverage

```
[+] src/lib/mark.ts
    ├── resolveAnchor()
    │   ├── [TESTED] ★★★ below threshold, at threshold, no bars — Tasks 2, 3
    │   └── [GAP]        alignmentQuality NaN / negative / >1 — no test
    ├── anchorLabel()
    │   ├── [TESTED] ★★★ range, single bar, timestamp, zero-pad — Task 4
    │   └── [GAP]        atSeconds >= 3600 renders "60:00" not "1:00:00"
    └── isMarkWorthy()
        └── [TESTED] ★★  all four lifecycle values — Task 5

[+] src/lib/mark-placement.ts
    └── placeMarks()
        ├── [TESTED] ★★★ resolves by measureOn not index — Task 7
        ├── [TESTED] ★★★ timestamp anchor reported — Task 8
        ├── [TESTED] ★★★ off-page and unknown bar reported — Task 9
        └── [GAP]        duplicate barNumber in locator table (Map keeps last)

[+] src/components/ScoreMarkLayer.tsx
    ├── [TESTED] ★★  which marks render + disclosure count — Task 13
    ├── [TESTED] ★★  tap to expand/collapse — Task 14
    └── [GAP]        coordinate translation (untestable in jsdom — see RISK)

[+] src/components/SessionTimelineStrip.tsx
    ├── [TESTED] ★★★ completeness + timestamp label + position — Task 15
    ├── [TESTED] ★★  tap to expand/collapse — Task 16
    └── [GAP]        durationSeconds = 0 (guarded by `span`, not asserted)
```

The three GAPs are all low-severity: none is on an auth, payment, or data-mutation
path, and the plan's own product constraint (60s soft auto-stop) makes the
one-hour label gap unreachable in practice. Recording them, not blocking on them.

#### Failure Modes

No async operations, no I/O, no persistence, no user input reaching SQL, shell,
or an LLM prompt. This slice is pure rendering over fixture data, so the usual
failure-mode surface is genuinely absent rather than overlooked.

The one silent-failure candidate is real and correctly handled: a mark that
cannot be placed is **reported**, not dropped. Tasks 8 and 9 exist precisely to
convert two silent `continue`s into visible `unplaced` entries, and Task 13
asserts the count reaches the screen. Zero silent failures.

`ResizeObserver` disconnect on unmount is present. No transaction boundaries to
consider.

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| Discriminated-union narrowing survives intersection with a brand | **SAFE** | Verified: scratch file typechecked clean, including both `@ts-expect-error` assertions |
| A raw literal cannot forge a `MarkAnchor` | **SAFE** | Verified: `@ts-expect-error` on the forged literal held |
| `as unknown as MarkAnchor` is required (plain `as` insufficient) | **SAFE** | Verified in the same scratch file; confined to two lines |
| `BarIR.measureOn` equals the SVG measure element's `id` | **VALIDATE** | True by construction at `score-ir.ts:242`, but never observed against real Verovio output — see RISK |
| jsdom returns zero rects, so placement still "succeeds" | **SAFE** | Verified by probe: 2 rects resolved, all zeros, marks rendered |
| `containerRef.current` is populated in a child's `useEffect` | **SAFE** | Verified by probe: the null-guard branch never logged |
| `toHaveStyle({left: "17.777...%"})` matches | **SAFE** | Verified by probe |
| `noUnusedLocals` makes an unused type import an error, not a warning | **SAFE** | Confirmed in `apps/web/tsconfig.json`; justifies moving the `Dimension` import to Task 5 |
| `bun run lint` baseline is 69 warnings | **RISKY** | False — verified 107 warnings / 23 infos |
| `bun run test` baseline is fully green | **RISKY** | False — verified 1/216 failing under load, passes in isolation |
| A fresh worktree can run `bunx tsc --noEmit` after `bun install` in `apps/web` | **RISKY** | False — needs `apps/api` deps and the WASM `pkg/` copy as well |
| `/app`-free top-level routes render in a preview build without auth | **VALIDATE** | `routes/app.tsx:12` gates only `/app`; `/marks-preview` is top-level, but not yet observed serving 200 |

### Open Questions raised by this review

- **Q: Should `/marks-preview` render a real Verovio score instead of stand-in
  divs?** Default: keep stand-ins for the automated tests (they make the
  off-page case constructible, which a real score does not), but add a
  real-piece section to the route via `scoreRenderer.load()` + `getIR()` so the
  Task 19 click-through observes marks landing on actual engraving. Without
  this, nothing in #157 ever proves the feature works on a real score.
- **Q: How does the brand survive a JSON round-trip when #158 adds a backend?**
  `JSON.parse` yields an unbranded object that is not assignable to
  `MarkAnchor`, which is correct and desirable. The boundary must therefore
  transmit `alignmentQuality` and re-run `resolveAnchor` client-side rather than
  casting past the brand. Recording it now so a later issue does not reach for
  `as unknown as MarkAnchor` at the deserialization boundary and quietly delete
  the guarantee.

### Summary

[BLOCKER] count: 3 — all in the verification preamble, all verified by running
commands, all mechanically fixable without touching the architecture.
[RISK] count: 3
[QUESTION] count: 2

The architecture, module decomposition, test philosophy, and vertical-slice
discipline all hold up. Every blocker is about the plan describing an
environment and a baseline that do not match this worktree.

VERDICT: NEEDS_REWORK — (1) the typecheck command needs a Prerequisites section
covering the WASM `pkg/` copy and `apps/api` deps, without which it reports 209
errors; (2) Task 19's "zero failing files" contradicts a verified 1/216 baseline
failure; (3) Task 19's "69 warnings" contradicts a verified 107.

---

## Post-Challenge Resolution

The challenge review returned NEEDS_REWORK on three blockers. All three were in
the plan's description of the environment, not its architecture. Resolved:

1. **Typecheck prerequisites** — added the "Prerequisites — do these before
   Task 1" section. Verified sequence: fresh worktree reports 5 errors; after
   copying both WASM `pkg/` dirs, 209; after `bun install` in `apps/api`, 0.
2. **Test baseline** — Task 19 now expects the one known load-dependent perf
   failure rather than zero failures, with the isolation re-run as the
   disambiguator. This corrects the handoff's "never reproduced": it reproduces
   reliably under full-suite load.
3. **Lint baseline** — corrected from 69 to the verified 107 warnings / 23
   infos, and reframed as "no *new* warnings" so the check survives unrelated
   merges.

Both open questions are resolved:

- **Real Verovio score:** ACCEPTED. Added **Task 20**, which puts a real
  `chopin.ballades.1` engraving on `/marks-preview` and asserts in Playwright
  that a mark's box overlaps the actual `g.measure` element it names. This
  closes the review's largest honesty gap and moves the `BarIR.measureOn`
  presumption from VALIDATE to a genuinely tested claim. Synthetic stand-ins
  stay for the vitest tests, because the off-page/unplaced case cannot be
  constructed deterministically against a real score.
- **Brand across a JSON round-trip:** RECORDED for #158, no change here.
  `JSON.parse` yields an unbranded object that is not assignable to
  `MarkAnchor` — which is the guarantee working, not a bug. The wire format must
  therefore carry `alignmentQuality` and the client must re-run `resolveAnchor`
  at the deserialization boundary. A later issue reaching for
  `as unknown as MarkAnchor` there would silently delete the whole property.

### Residual risks carried into build

- jsdom cannot verify placement coordinates (verified: all rects are zeros).
  Mitigated by Tasks 7-9 against synthetic non-zero rects, and now by Task 20 in
  a real browser.
- `/marks-preview` imports `src/test-utils/`, putting fixtures in the production
  bundle. Accepted for a preview route that #158/#159/#162 delete.
- The known-red window spans Tasks 1-18. Documented; do not resolve it by
  deleting or skipping the harness.

VERDICT: PROCEED

---

## Build Progress (recovery state)

Authoritative record of which tasks have landed. A resuming session should start
at the first task NOT listed here. Per-step `- [ ]` checkboxes inside task bodies
are NOT maintained — they are textually identical across all 20 tasks and bulk
editing them is unreliable. This log is the recovery state.

| Task | Group | Observed Step-2 failure (proof the test bit) | Commit | Reviews |
|---|---|---|---|---|
| 1 — contract harness | 0 | `Failed to resolve import "../test-utils/mark-fixtures"` (vite:import-analysis, 0 tests collected) | `d5076641` | PASS (merged spec+quality: verbatim file, no impl code) |
| 2 — anchor degrades to timestamp | A | `Failed to resolve import "./mark"` | `bb126408` | see Group A note |
| 3 — bars kept above threshold | A | `AssertionError: expected 'timestamp' to be 'bars'` (mark.test.ts:24) | `bf868f9a` | see Group A note |
| 4 — anchorLabel | A | `TypeError: anchorLabel is not a function` (mark.test.ts:52) | `e45a5806` | see Group A note |
| 5 — isMarkWorthy + vocabulary | A | `TypeError: isMarkWorthy is not a function` (mark.test.ts:65) | `40bdb1a0` | see Group A note |

**Group A status: all four tasks landed, 6/6 tests green in `src/lib/mark.test.ts`.**
Cumulative Group A review still outstanding — Tasks 2-5 all edit the same two
files sequentially, so per-task review would race the next task's edits and
produce false findings. Review the cumulative diff `bb126408..40bdb1a0` task by
task instead.

### Review findings carried forward

- Task 1, MINOR, not fixed: `expect(screen.getAllByLabelText(/1:37/)).not.toHaveLength(0)`
  is redundant — `getAllByLabelText` throws when nothing matches, so the length
  assertion can never observe zero. The throw is the real assertion, so the test
  still verifies the behaviour. Left as-is rather than churning a deliberately
  red harness.
