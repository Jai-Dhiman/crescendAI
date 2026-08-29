# Home + Piece Page Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent
> per task). Do NOT start execution until /challenge returns
> VERDICT: PROCEED.

**Goal:** A student opens `/app` and sees the pieces they have actually
practiced, can add a piece from the score library, and can open any piece to
see its score marked up across sessions.
**Spec:** `docs/specs/2026-08-27-home-piece-page-design.md`
**Style:** `apps/api/TS_STYLE.md` (API), `apps/web/WEB_STYLE.md` (web),
`docs/standards/rules.json` for rule status. Root and `apps/CLAUDE.md` apply.

---

## Standing constraints for every task

These are violations the reviewer will reject; they are repeated here because
a subagent may only read its own task.

- **`bun`, never `npm`.** Web tests: `cd apps/web && bun run test`. API tests:
  `cd apps/api && bun run test`.
- **Never run bare `bun run migrate` in `apps/api`** -- it targets the hosted
  production database. Local migrations only, with the explicit URL:
  `DATABASE_URL="postgresql://jdhiman:postgres@localhost:5432/crescendai_dev" bun run migrate`.
- **[TS-DB-003]** Never hand-write or hand-edit a file in
  `apps/api/src/db/migrations/`. Generate with `bun run generate`.
- **[WEB-STATE-002]** Do NOT add `useMemo`/`useCallback`. Default is no
  memoization; neither a downstream dependency-array consumer nor a measured
  perf need exists anywhere in this plan.
- **[WEB-STATE-001]** Never `useEffect` + `useState` to derive a value that is
  a pure function of props/state. Compute during render.
- **[WEB-QUERY-001]** Never write a `queryKey` array literal in a route or
  component. Use the exported `queryOptions` objects.
- **[WEB-QUERY-002]** Never put server data in zustand.
- **[WEB-UI-002]** No Tailwind arbitrary bracket values. Allowed colour
  tokens: `surface-page`, `surface-raised`, `surface-sunken`, `ink-primary`,
  `ink-secondary`, `ink-tertiary`, `border-subtle`, `border-strong`, `accent`,
  `on-accent`, `score-canvas`, `danger`, `warn`.
- **[WEB-A11Y-004]** Do not introduce a new colour/token pairing. Reuse the
  pairings already shipped in `routes/app.tsx`, `ScoreStand`, `MarkGlyph`.
- **[WEB-A11Y-003]** Every interactive target has a >= 32px hit area
  (`py-2` + `text-body-sm` or larger clears it); decorative icons are
  `aria-hidden`.
- **[WEB-A11Y-001]** Add no animation in this plan.
- **[WEB-ERR-001]** Never `catch` an API error and return `[]`/`null`.
- **No new `.md` files** beyond this plan and its spec.
- **Do not modify** `ScoreStand`, `ScoreMarkLayer`, `MarkGlyph`,
  `PieceLessMode`, `SessionReview`, `PracticeMode`, `lib/mark.ts`,
  `lib/piece-ladder.ts`, or `routes/practice-preview.tsx`.

---

## Task Groups

```
Group 0 (harness, must pass first):  Task 1
Group A (parallel):                  Task 2, Task 3, Task 4, Task 5
Group B (sequential, after A):       Task 6, then Task 7
Group C (parallel, after B):         Task 8, Task 9
Group D (single, after C):           Task 10
Group E (parallel, after D):         Task 11, Task 12, Task 13, Task 14
Group F (single, after E):           Task 15
Group G (parallel, after F):         Task 16, Task 17
```

**Why Task 10 is alone in its own group:** it creates the `RepertoireEntry`
and `PieceSessionSummary` types in `lib/api.ts` that Tasks 11 and 13 import.
Running them alongside it would have those tasks compiling against types that
do not exist yet.

- **Group A `[SHIPS INDEPENDENTLY]`** -- nothing user-visible, but the two
  pure web modules and the schema column are independently mergeable.
- **Group C `[SHIPS INDEPENDENTLY]`** -- with A+B+C merged, the API can record
  a session's piece and serve a repertoire. Usable by any client before any UI
  exists.
- Groups B, D, F are single-file or single-task and run sequentially: B's two
  tasks edit the same service file, D is a prerequisite for E's types, and F
  restructures the route tree that G's two pages mount into.

---

## Task 1: Piece-page fixtures -- marks aging across two sessions

**Group:** 0 (must pass before any other group is dispatched)

**Behavior being verified:** The fixture set contains marks from two distinct
sessions in which the same passage's mark progresses through lifecycle states,
so the piece page's "marks aging across >= 2 sessions" criterion is assertable.
**Interface under test:** the exported constants of
`apps/web/src/lib/piece-page-fixtures.ts`

**Files:**
- Create: `apps/web/src/lib/piece-page-fixtures.ts`
- Test: `apps/web/src/lib/piece-page-fixtures.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/piece-page-fixtures.test.ts
import { describe, expect, it } from "vitest";
import {
	PIECE_PAGE_ALL_MARKS,
	PIECE_PAGE_MARKS_SPAN_SECONDS,
	PIECE_PAGE_SESSIONS,
} from "./piece-page-fixtures";

describe("piece page fixtures", () => {
	it("carries at least two dated sessions whose marks span all three lifecycle states", () => {
		expect(PIECE_PAGE_SESSIONS.length).toBeGreaterThanOrEqual(2);

		// Distinct sessions, oldest first.
		const ids = PIECE_PAGE_SESSIONS.map((s) => s.sessionId);
		expect(new Set(ids).size).toBe(ids.length);
		const times = PIECE_PAGE_SESSIONS.map((s) =>
			Date.parse(s.startedAt),
		);
		expect(times).toEqual([...times].sort((a, b) => a - b));

		// The accumulated view spans every lifecycle state, which is what the
		// piece page renders as the fade.
		const lifecycles = new Set(
			PIECE_PAGE_SESSIONS.flatMap((s) => s.marks.map((m) => m.lifecycle)),
		);
		expect(lifecycles).toEqual(new Set(["active", "improving", "resolved"]));
	});

	it("tracks one passage improving across the two sessions", () => {
		const [older, newer] = PIECE_PAGE_SESSIONS;
		const olderPedaling = older.marks.find((m) => m.dimension === "pedaling");
		const newerPedaling = newer.marks.find((m) => m.dimension === "pedaling");
		expect(olderPedaling?.lifecycle).toBe("active");
		expect(newerPedaling?.lifecycle).toBe("resolved");
	});

	it("anchors every mark through resolveAnchor, so bar anchors carry atSeconds", () => {
		for (const session of PIECE_PAGE_SESSIONS) {
			for (const mark of session.marks) {
				expect(typeof mark.anchor.atSeconds).toBe("number");
			}
		}
	});

	it("spans far enough to hold every mark, so none is clamped off the strip", () => {
		const latest = Math.max(
			...PIECE_PAGE_ALL_MARKS.map((m) => m.anchor.atSeconds),
		);
		expect(PIECE_PAGE_MARKS_SPAN_SECONDS).toBeGreaterThanOrEqual(latest);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && bun run test src/lib/piece-page-fixtures.test.ts
```
Expected: FAIL -- `Failed to resolve import "./piece-page-fixtures"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/web/src/lib/piece-page-fixtures.ts
import type { Mark } from "./mark";
import { resolveAnchor } from "./mark";

/**
 * Fixture data for the piece page (#160). Marks have no persisted store yet
 * -- they live only in the DO and the session WebSocket -- so "the score with
 * marks accumulated across sessions" is fixture-driven by design, exactly as
 * #160's success criterion words it. The mark-persistence follow-up replaces
 * this module's consumers, not its shape.
 *
 * Every anchor is built by resolveAnchor rather than an object literal: the
 * MarkAnchor brand symbol is unexported precisely so that "a wrong bar number
 * cannot reach the screen" is a compile-time property, and a fixture casting
 * its way past it would be the first hole in that guarantee.
 *
 * Leaf module with no component imports, matching practice-preview-fixtures.ts
 * -- see that file's comment for why (Playwright's loader is not Vite and
 * cannot evaluate lib/config.ts's import.meta.env.PROD access).
 */

export interface FixtureSession {
	readonly sessionId: string;
	/** ISO 8601 */
	readonly startedAt: string;
	readonly marks: readonly Mark[];
}

/** Oldest first. */
export const PIECE_PAGE_SESSIONS: readonly FixtureSession[] = [
	{
		sessionId: "fixture-session-1",
		startedAt: "2026-08-11T18:00:00.000Z",
		marks: [
			{
				id: "pp-1-pedaling",
				anchor: resolveAnchor({
					atSeconds: 64,
					bars: [5, 6],
					alignmentQuality: 1,
				}),
				taxonomy: "needs_work",
				dimension: "pedaling",
				evidence: "pedal held through the bass change at 5.3",
				lifecycle: "active",
			},
			{
				id: "pp-1-dynamics",
				anchor: resolveAnchor({ atSeconds: 120, alignmentQuality: 0 }),
				taxonomy: "missed_opportunity",
				dimension: "dynamics",
				evidence: "the closing diminuendo flattened out early",
				lifecycle: "active",
			},
		],
	},
	{
		sessionId: "fixture-session-2",
		startedAt: "2026-08-18T18:00:00.000Z",
		marks: [
			{
				id: "pp-2-pedaling",
				anchor: resolveAnchor({
					atSeconds: 61,
					bars: [5, 6],
					alignmentQuality: 1,
				}),
				taxonomy: "strong",
				dimension: "pedaling",
				evidence: "the bass change is clean now",
				lifecycle: "resolved",
			},
			{
				id: "pp-2-dynamics",
				anchor: resolveAnchor({ atSeconds: 118, alignmentQuality: 0 }),
				taxonomy: "needs_work",
				dimension: "dynamics",
				evidence: "the diminuendo reaches further, still peaks early",
				lifecycle: "improving",
			},
		],
	},
];

/** Every mark the piece has accumulated, newest session last. */
export const PIECE_PAGE_ALL_MARKS: readonly Mark[] = PIECE_PAGE_SESSIONS.flatMap(
	(s) => s.marks,
);

/**
 * The span the piece page's timeline covers. Derived from the marks rather
 * than hardcoded, so adding a later mark cannot silently push it off the end
 * of the strip -- the clamping bug #157 fixed reappears the moment a mark's
 * atSeconds exceeds the declared duration.
 */
export const PIECE_PAGE_MARKS_SPAN_SECONDS: number = Math.max(
	...PIECE_PAGE_ALL_MARKS.map((m) => m.anchor.atSeconds),
);
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && bun run test src/lib/piece-page-fixtures.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/piece-page-fixtures.ts apps/web/src/lib/piece-page-fixtures.test.ts && git commit -m "test(web): piece-page fixtures with marks aging across two sessions"
```

---

## Task 2: `searchPieces` -- rank catalog pieces against a typed query

**Group:** A (parallel with Task 3, Task 4, Task 5)

**Behavior being verified:** Typing part of a composer's or a work's name
returns the matching catalog pieces, prefix matches before mid-string ones,
and an empty query returns nothing.
**Interface under test:** `searchPieces(pieces, query, limit?)`

**Files:**
- Create: `apps/web/src/lib/piece-search.ts`
- Test: `apps/web/src/lib/piece-search.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/piece-search.test.ts
import { describe, expect, it } from "vitest";
import type { PieceSummary } from "./piece-search";
import { searchPieces } from "./piece-search";

const CATALOG: readonly PieceSummary[] = [
	{ pieceId: "chopin-nocturne-op9-no2", composer: "Chopin", title: "Nocturne Op. 9 No. 2" },
	{ pieceId: "chopin-ballade-1", composer: "Chopin", title: "Ballade No. 1" },
	{ pieceId: "bach-invention-1", composer: "Bach", title: "Invention No. 1" },
	{ pieceId: "faure-nocturne-4", composer: "Fauré", title: "Nocturne No. 4" },
];

describe("searchPieces", () => {
	it("returns nothing for an empty or whitespace query", () => {
		expect(searchPieces(CATALOG, "")).toEqual([]);
		expect(searchPieces(CATALOG, "   ")).toEqual([]);
	});

	it("matches on composer regardless of case", () => {
		const found = searchPieces(CATALOG, "chopin");
		expect(found.map((p) => p.pieceId)).toEqual([
			"chopin-nocturne-op9-no2",
			"chopin-ballade-1",
		]);
	});

	it("matches on title", () => {
		const found = searchPieces(CATALOG, "invention");
		expect(found.map((p) => p.pieceId)).toEqual(["bach-invention-1"]);
	});

	it("ranks a prefix match above a mid-string match", () => {
		// "no" prefixes the title "Nocturne..." but only appears mid-string in
		// "Ballade No. 1" and "Invention No. 1".
		const found = searchPieces(CATALOG, "no");
		expect(found[0].title.startsWith("Nocturne")).toBe(true);
	});

	it("matches accented composer names typed without accents", () => {
		const found = searchPieces(CATALOG, "faure");
		expect(found.map((p) => p.pieceId)).toEqual(["faure-nocturne-4"]);
	});

	it("caps results at the requested limit", () => {
		expect(searchPieces(CATALOG, "n", 2)).toHaveLength(2);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && bun run test src/lib/piece-search.test.ts
```
Expected: FAIL -- `Failed to resolve import "./piece-search"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/web/src/lib/piece-search.ts

/**
 * Structurally the subset of a GET /api/scores row this feature reads.
 * Deliberately not the full `pieces` table type: the search surface has no
 * use for bar counts or pitch ranges, and widening it would couple every
 * caller to catalog schema changes.
 */
export interface PieceSummary {
	readonly pieceId: string;
	readonly composer: string;
	readonly title: string;
}

const DEFAULT_LIMIT = 20;

/**
 * NFD + combining-mark strip, so "faure" matches "Fauré". The catalog is
 * classical repertoire; accented composer names are the rule, not the edge
 * case, and a student typing on a US keyboard cannot produce them.
 */
function normalize(value: string): string {
	return value
		.normalize("NFD")
		.replace(/[̀-ͯ]/g, "")
		.toLowerCase();
}

/**
 * Empty query returns nothing rather than everything: this drives an
 * add-piece surface where showing the entire catalog on focus is noise, and
 * "no query means no results" keeps the caller from needing its own guard.
 */
export function searchPieces(
	pieces: readonly PieceSummary[],
	query: string,
	limit: number = DEFAULT_LIMIT,
): PieceSummary[] {
	const needle = normalize(query.trim());
	if (needle === "") return [];

	const scored: { piece: PieceSummary; rank: number }[] = [];
	for (const piece of pieces) {
		const composer = normalize(piece.composer);
		const title = normalize(piece.title);
		// 0 = prefix match (strongest), 1 = substring, absent = no match.
		let rank: number | null = null;
		if (composer.startsWith(needle) || title.startsWith(needle)) {
			rank = 0;
		} else if (composer.includes(needle) || title.includes(needle)) {
			rank = 1;
		}
		if (rank !== null) scored.push({ piece, rank });
	}

	// Stable sort by rank alone: within a rank the catalog's own ordering
	// (composer, then title -- services/scores.ts orders it that way) is
	// already the order a student expects.
	scored.sort((a, b) => a.rank - b.rank);
	return scored.slice(0, limit).map((s) => s.piece);
}
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && bun run test src/lib/piece-search.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/piece-search.ts apps/web/src/lib/piece-search.test.ts && git commit -m "feat(web): searchPieces ranks catalog pieces by composer and title"
```

---

## Task 3: `describeArc` -- the repertoire card's one line

**Group:** A (parallel with Task 2, Task 4, Task 5)

**Behavior being verified:** A piece's session count and last-practiced date
render as one plain-English line, with relative phrasing near today and an
absolute date further back.
**Interface under test:** `describeArc({ sessionCount, lastPracticedAt, now })`

**Files:**
- Create: `apps/web/src/lib/learning-arc.ts`
- Test: `apps/web/src/lib/learning-arc.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/learning-arc.test.ts
import { describe, expect, it } from "vitest";
import { describeArc } from "./learning-arc";

// `now` is injected, never read from the system clock, so these assertions
// hold in every timezone and need no fake timers.
const NOW = new Date("2026-08-27T12:00:00.000Z");

describe("describeArc", () => {
	it("describes a piece that has never been practiced", () => {
		expect(
			describeArc({ sessionCount: 0, lastPracticedAt: null, now: NOW }),
		).toBe("Not practiced yet");
	});

	it("uses the singular for exactly one session", () => {
		expect(
			describeArc({
				sessionCount: 1,
				lastPracticedAt: "2026-08-27T09:00:00.000Z",
				now: NOW,
			}),
		).toBe("1 session · last practiced today");
	});

	it("says yesterday for the previous day", () => {
		expect(
			describeArc({
				sessionCount: 4,
				lastPracticedAt: "2026-08-26T09:00:00.000Z",
				now: NOW,
			}),
		).toBe("4 sessions · last practiced yesterday");
	});

	it("names the weekday within the past week", () => {
		// 2026-08-23 is a Sunday, 4 days before 2026-08-27.
		expect(
			describeArc({
				sessionCount: 6,
				lastPracticedAt: "2026-08-23T09:00:00.000Z",
				now: NOW,
			}),
		).toBe("6 sessions · last practiced Sunday");
	});

	it("falls back to an absolute date beyond a week", () => {
		expect(
			describeArc({
				sessionCount: 12,
				lastPracticedAt: "2026-07-04T09:00:00.000Z",
				now: NOW,
			}),
		).toBe("12 sessions · last practiced 4 Jul");
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && bun run test src/lib/learning-arc.test.ts
```
Expected: FAIL -- `Failed to resolve import "./learning-arc"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/web/src/lib/learning-arc.ts

export interface LearningArcInput {
	readonly sessionCount: number;
	/** ISO 8601, or null for a piece with no sessions. */
	readonly lastPracticedAt: string | null;
	/**
	 * Injected rather than read from the system clock. Every relative branch
	 * below is then testable without fake timers, and the function cannot
	 * silently change behaviour with the machine's timezone.
	 */
	readonly now: Date;
}

const MS_PER_DAY = 24 * 60 * 60 * 1000;

/** Whole days between two instants, counted in UTC calendar days. */
function daysBetween(from: Date, to: Date): number {
	const fromDay = Date.UTC(
		from.getUTCFullYear(),
		from.getUTCMonth(),
		from.getUTCDate(),
	);
	const toDay = Date.UTC(to.getUTCFullYear(), to.getUTCMonth(), to.getUTCDate());
	return Math.round((toDay - fromDay) / MS_PER_DAY);
}

function describeWhen(last: Date, now: Date): string {
	const days = daysBetween(last, now);
	if (days <= 0) return "today";
	if (days === 1) return "yesterday";
	if (days < 7) {
		return last.toLocaleDateString("en-GB", {
			weekday: "long",
			timeZone: "UTC",
		});
	}
	return last.toLocaleDateString("en-GB", {
		day: "numeric",
		month: "short",
		timeZone: "UTC",
	});
}

/**
 * The repertoire card's single line of history. Extracted from the card so
 * the card carries no date arithmetic and every branch here is testable
 * without rendering anything.
 */
export function describeArc(input: LearningArcInput): string {
	const { sessionCount, lastPracticedAt, now } = input;
	if (sessionCount === 0 || lastPracticedAt === null) {
		return "Not practiced yet";
	}
	const noun = sessionCount === 1 ? "session" : "sessions";
	const when = describeWhen(new Date(lastPracticedAt), now);
	return `${sessionCount} ${noun} · last practiced ${when}`;
}
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && bun run test src/lib/learning-arc.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/learning-arc.ts apps/web/src/lib/learning-arc.test.ts && git commit -m "feat(web): describeArc renders a piece's session history as one line"
```

---

## Task 4: `sessions.piece_id` column and migration

**Group:** A (parallel with Task 2, Task 3, Task 5)

**Behavior being verified:** A practice session can record which catalog piece
it was, and the column survives a real migration against local Postgres.
**Interface under test:** the `sessions` Drizzle table and the generated
migration applied to `crescendai_dev`.

**Files:**
- Modify: `apps/api/src/db/schema/sessions.ts`
- Create: `apps/api/src/db/migrations/0006_*.sql` (**generated**, exact name
  assigned by drizzle-kit)
- Test: `apps/api/src/db/schema/sessions.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/db/schema/sessions.test.ts
import { getTableName } from "drizzle-orm";
import { describe, expect, it } from "vitest";
import { sessions } from "./sessions";

describe("sessions schema", () => {
	it("is the sessions table", () => {
		expect(getTableName(sessions)).toBe("sessions");
	});

	it("carries the piece a session was practising", () => {
		// Nullable by design: a pieceless session is a permanent first-class
		// state (docs/apps/05-ui-system.md#2), not a missing value.
		expect(Object.keys(sessions)).toContain("pieceId");
		expect(sessions.pieceId.notNull).toBe(false);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/api && bun run test src/db/schema/sessions.test.ts
```
Expected: FAIL -- `Property 'pieceId' does not exist on type` at compile, and
at runtime `expect(Object.keys(sessions)).toContain("pieceId")` fails.

- [ ] **Step 3: Implement the minimum to make the test pass**

Edit `apps/api/src/db/schema/sessions.ts`. Add the `pieces` import, the
column, and the index:

```typescript
import {
	boolean,
	index,
	jsonb,
	pgTable,
	real,
	text,
	timestamp,
	uuid,
} from "drizzle-orm/pg-core";
import { pieces } from "./catalog";

export const sessions = pgTable(
	"sessions",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: text("student_id").notNull(),
		startedAt: timestamp("started_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
		endedAt: timestamp("ended_at", { withTimezone: true }),
		// Nullable: a pieceless session is a permanent first-class state, not a
		// missing value. `set null` rather than `cascade` because the catalog is
		// codegen-seeded and a piece leaving it must never delete practice
		// history.
		pieceId: text("piece_id").references(() => pieces.pieceId, {
			onDelete: "set null",
		}),
		avgDynamics: real("avg_dynamics"),
		avgTiming: real("avg_timing"),
		avgPedaling: real("avg_pedaling"),
		avgArticulation: real("avg_articulation"),
		avgPhrasing: real("avg_phrasing"),
		avgInterpretation: real("avg_interpretation"),
		observationsJson: jsonb("observations_json"),
		chunksSummaryJson: jsonb("chunks_summary_json"),
		conversationId: text("conversation_id"),
		accumulatorJson: jsonb("accumulator_json"),
		needsSynthesis: boolean("needs_synthesis").notNull().default(false),
	},
	(t) => [
		index("idx_sessions_student").on(t.studentId, t.startedAt),
		index("idx_sessions_conversation").on(t.conversationId),
		// The repertoire query's access path: every session for one student,
		// grouped by piece.
		index("idx_sessions_student_piece").on(t.studentId, t.pieceId),
	],
);
```

Leave `studentCheckIns` below it untouched.

- [ ] **Step 4: Run test -- verify it PASSES, then generate and apply the migration**

```bash
cd apps/api && bun run test src/db/schema/sessions.test.ts
```
Expected: PASS

Generate the migration (never hand-write it -- **[TS-DB-003]**):

```bash
cd apps/api && bun run generate
```
Expected: a new `src/db/migrations/0006_<name>.sql` plus an updated
`meta/_journal.json` and `meta/0006_snapshot.json`.

Apply it to **local** Postgres only:

```bash
cd apps/api && DATABASE_URL="postgresql://jdhiman:postgres@localhost:5432/crescendai_dev" bun run migrate
```

Verify the column and index actually landed:

```bash
psql "postgresql://jdhiman:postgres@localhost:5432/crescendai_dev" -c "\d sessions" | grep -E "piece_id|idx_sessions_student_piece"
```
Expected: one row naming `piece_id | text` and one naming
`idx_sessions_student_piece`.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/db/schema/sessions.ts apps/api/src/db/schema/sessions.test.ts apps/api/src/db/migrations && git commit -m "feat(api): record which catalog piece a session was practising"
```

---

## Task 5: `listOpenLoopsForPiece` -- the piece page's open drills

**Group:** A (parallel with Task 2, Task 3, Task 4)

**Behavior being verified:** A student's not-yet-finished drills for one piece
are returned oldest first, and drills belonging to another student or another
piece are not.
**Interface under test:** `listOpenLoopsForPiece(db, studentId, pieceId)`

**Files:**
- Modify: `apps/api/src/services/segment-loops.ts`
- Test: `apps/api/src/services/segment-loops.open.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/services/segment-loops.open.test.ts
import { describe, expect, it } from "vitest";
import type { Db } from "../lib/types";
import { listOpenLoopsForPiece } from "./segment-loops";

/**
 * Captures the where-clause the service builds and returns canned rows. The
 * assertion is on the returned drills -- the public result -- not on the SQL;
 * the stub exists only because Drizzle's builder chain has no in-memory
 * backend in this suite.
 */
function makeDbStub(rows: unknown[]) {
	return {
		select: () => ({
			from: () => ({
				where: () => ({
					orderBy: async () => rows,
				}),
			}),
		}),
	} as unknown as Db;
}

describe("listOpenLoopsForPiece", () => {
	it("maps rows to open drills", async () => {
		const db = makeDbStub([
			{
				id: "loop-1",
				barsStart: 5,
				barsEnd: 8,
				dimension: "pedaling",
				requiredCorrect: 5,
				attemptsCompleted: 2,
			},
			{
				id: "loop-2",
				barsStart: 20,
				barsEnd: 24,
				dimension: null,
				requiredCorrect: 5,
				attemptsCompleted: 0,
			},
		]);

		const drills = await listOpenLoopsForPiece(db, "student-a", "piece-x");

		expect(drills).toEqual([
			{
				id: "loop-1",
				barsStart: 5,
				barsEnd: 8,
				dimension: "pedaling",
				requiredCorrect: 5,
				attemptsCompleted: 2,
			},
			{
				id: "loop-2",
				barsStart: 20,
				barsEnd: 24,
				dimension: null,
				requiredCorrect: 5,
				attemptsCompleted: 0,
			},
		]);
	});

	it("returns an empty list when the piece has no open drills", async () => {
		const drills = await listOpenLoopsForPiece(
			makeDbStub([]),
			"student-a",
			"piece-x",
		);
		expect(drills).toEqual([]);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/api && bun run test src/services/segment-loops.open.test.ts
```
Expected: FAIL -- `listOpenLoopsForPiece is not a function` (no such export).

- [ ] **Step 3: Implement the minimum to make the test pass**

Append to `apps/api/src/services/segment-loops.ts`, and add `asc` and
`inArray` to its existing `drizzle-orm` import:

```typescript
/** One drill the student has not finished, as the piece page shows it. */
export interface OpenDrill {
	id: string;
	barsStart: number;
	barsEnd: number;
	dimension: string | null;
	requiredCorrect: number;
	attemptsCompleted: number;
}

/**
 * Every unfinished drill for one piece. Distinct from findActiveForPiece,
 * which answers "is a drill running right now" and is capped at one row:
 * the piece page lists work still outstanding, so it includes `pending`
 * (offered, not yet accepted) alongside `active`.
 */
export async function listOpenLoopsForPiece(
	db: Db,
	studentId: string,
	pieceId: string,
): Promise<OpenDrill[]> {
	const rows = await db
		.select({
			id: segmentLoops.id,
			barsStart: segmentLoops.barsStart,
			barsEnd: segmentLoops.barsEnd,
			dimension: segmentLoops.dimension,
			requiredCorrect: segmentLoops.requiredCorrect,
			attemptsCompleted: segmentLoops.attemptsCompleted,
		})
		.from(segmentLoops)
		.where(
			and(
				eq(segmentLoops.studentId, studentId),
				eq(segmentLoops.pieceId, pieceId),
				inArray(segmentLoops.status, ["pending", "active"]),
			),
		)
		.orderBy(asc(segmentLoops.createdAt));

	return rows;
}
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/api && bun run test src/services/segment-loops.open.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/segment-loops.ts apps/api/src/services/segment-loops.open.test.ts && git commit -m "feat(api): listOpenLoopsForPiece returns a piece's unfinished drills"
```

---

## Task 6: `buildRepertoire` -- session rows become repertoire entries

**Group:** B (sequential; depends on Group A. Task 7 follows this task and
touches the same file.)

**Behavior being verified:** A student's practice sessions collapse into one
entry per piece, carrying a session count and the most recent practice date,
ordered most-recently-practised first.
**Interface under test:** `buildRepertoire(rows)`

**Files:**
- Create: `apps/api/src/services/repertoire.ts`
- Test: `apps/api/src/services/repertoire.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/services/repertoire.test.ts
import { describe, expect, it } from "vitest";
import type { RepertoireRow } from "./repertoire";
import { buildRepertoire } from "./repertoire";

function row(
	pieceId: string,
	composer: string,
	title: string,
	startedAt: string,
): RepertoireRow {
	return { pieceId, composer, title, startedAt: new Date(startedAt) };
}

describe("buildRepertoire", () => {
	it("returns nothing for a student with no sessions", () => {
		expect(buildRepertoire([])).toEqual([]);
	});

	it("collapses repeated sessions of one piece into a single entry", () => {
		const entries = buildRepertoire([
			row("p1", "Chopin", "Nocturne Op. 9 No. 2", "2026-08-11T18:00:00.000Z"),
			row("p1", "Chopin", "Nocturne Op. 9 No. 2", "2026-08-18T18:00:00.000Z"),
			row("p1", "Chopin", "Nocturne Op. 9 No. 2", "2026-08-25T18:00:00.000Z"),
		]);

		expect(entries).toHaveLength(1);
		expect(entries[0]).toEqual({
			pieceId: "p1",
			composer: "Chopin",
			title: "Nocturne Op. 9 No. 2",
			sessionCount: 3,
			lastPracticedAt: "2026-08-25T18:00:00.000Z",
			openMarkCount: null,
		});
	});

	it("takes the most recent session regardless of input order", () => {
		const entries = buildRepertoire([
			row("p1", "Chopin", "Nocturne", "2026-08-25T18:00:00.000Z"),
			row("p1", "Chopin", "Nocturne", "2026-08-11T18:00:00.000Z"),
		]);
		expect(entries[0].lastPracticedAt).toBe("2026-08-25T18:00:00.000Z");
	});

	it("orders pieces by most recently practised first", () => {
		const entries = buildRepertoire([
			row("older", "Bach", "Invention No. 1", "2026-08-01T18:00:00.000Z"),
			row("newer", "Chopin", "Ballade No. 1", "2026-08-26T18:00:00.000Z"),
			row("middle", "Fauré", "Nocturne No. 4", "2026-08-15T18:00:00.000Z"),
		]);
		expect(entries.map((e) => e.pieceId)).toEqual(["newer", "middle", "older"]);
	});

	it("reports openMarkCount as null, never 0", () => {
		// There is no persisted mark store yet. 0 would read to a student as
		// "nothing to work on"; null means "not known", and the card omits it.
		const entries = buildRepertoire([
			row("p1", "Chopin", "Nocturne", "2026-08-25T18:00:00.000Z"),
		]);
		expect(entries[0].openMarkCount).toBeNull();
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/api && bun run test src/services/repertoire.test.ts
```
Expected: FAIL -- `Failed to resolve import "./repertoire"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/services/repertoire.ts

/** One joined sessions-to-pieces row. The input to buildRepertoire. */
export interface RepertoireRow {
	pieceId: string;
	composer: string;
	title: string;
	startedAt: Date;
}

export interface RepertoireEntry {
	pieceId: string;
	composer: string;
	title: string;
	sessionCount: number;
	/** ISO 8601. The most recent session's startedAt. */
	lastPracticedAt: string;
	/**
	 * null until a persisted mark store exists -- never 0. A 0 here reads to a
	 * student as "nothing to work on", which is a claim this system cannot
	 * make yet; null means "not known" and the card omits the count entirely.
	 * The mark-persistence follow-up changes this line and no contract.
	 */
	openMarkCount: number | null;
}

/**
 * The grouping half of the repertoire query, kept pure and separate from the
 * database call. Two reasons, and the second is the real one:
 *
 * 1. It is the deep half. The query is a join; the policy -- one entry per
 *    piece, count the sessions, take the latest, order by recency -- is what
 *    a caller actually depends on.
 * 2. This codebase's route tests have no database. They stub services with
 *    vi.spyOn or hand-build a db.query.X.findFirst object; neither technique
 *    can stub a select().from().leftJoin() chain. Aggregation written inline
 *    would therefore be verifiable only against a live Postgres. As a
 *    function over rows it is testable with real inputs and no mocks.
 */
export function buildRepertoire(
	rows: readonly RepertoireRow[],
): RepertoireEntry[] {
	const byPiece = new Map<string, RepertoireEntry>();

	for (const row of rows) {
		const existing = byPiece.get(row.pieceId);
		if (!existing) {
			byPiece.set(row.pieceId, {
				pieceId: row.pieceId,
				composer: row.composer,
				title: row.title,
				lastPracticedAt: row.startedAt.toISOString(),
				sessionCount: 1,
				openMarkCount: null,
			});
			continue;
		}
		existing.sessionCount += 1;
		// Compare explicitly rather than trusting the query's ORDER BY: the
		// ordering guarantee would then live in two places and could drift.
		if (row.startedAt.toISOString() > existing.lastPracticedAt) {
			existing.lastPracticedAt = row.startedAt.toISOString();
		}
	}

	return [...byPiece.values()].sort((a, b) =>
		b.lastPracticedAt.localeCompare(a.lastPracticedAt),
	);
}
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/api && bun run test src/services/repertoire.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/repertoire.ts apps/api/src/services/repertoire.test.ts && git commit -m "feat(api): buildRepertoire collapses sessions into per-piece entries"
```

---

## Task 7: `listRepertoire` and `listPieceHistory` -- the queries

**Group:** B (sequential, immediately after Task 6 -- same file)

**Behavior being verified:** The service fetches a student's piece-tagged
sessions and returns them as repertoire entries, and returns one piece's
session history with its open drills, raising `NotFoundError` for a piece the
catalog does not have.
**Interface under test:** `listRepertoire(ctx, studentId)`,
`listPieceHistory(ctx, studentId, pieceId)`

**Files:**
- Modify: `apps/api/src/services/repertoire.ts`
- Test: `apps/api/src/services/repertoire.history.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/services/repertoire.history.test.ts
import { describe, expect, it } from "vitest";
import { NotFoundError } from "../lib/errors";
import type { Bindings, Db } from "../lib/types";
import { listPieceHistory } from "./repertoire";

const env = {} as Bindings;

/**
 * Drizzle's builder chain has no in-memory backend in this suite, so the
 * chain is stubbed. Assertions are on the returned PieceHistory -- the public
 * result -- never on which builder methods were called.
 */
function makeDb(opts: {
	piece: { pieceId: string; composer: string; title: string } | null;
	sessions: { sessionId: string; startedAt: Date; endedAt: Date | null }[];
	drills: unknown[];
}): Db {
	let call = 0;
	return {
		select: () => ({
			from: () => ({
				where: () => {
					call += 1;
					// 1st: the piece lookup (limit). 2nd: sessions. 3rd: drills.
					if (call === 1) {
						return {
							limit: async () => (opts.piece ? [opts.piece] : []),
						};
					}
					return { orderBy: async () => (call === 2 ? opts.sessions : opts.drills) };
				},
			}),
		}),
	} as unknown as Db;
}

describe("listPieceHistory", () => {
	it("returns the piece with its sessions newest first and its open drills", async () => {
		const db = makeDb({
			piece: { pieceId: "p1", composer: "Chopin", title: "Nocturne" },
			sessions: [
				{
					sessionId: "s2",
					startedAt: new Date("2026-08-25T18:00:00.000Z"),
					endedAt: new Date("2026-08-25T18:40:00.000Z"),
				},
				{
					sessionId: "s1",
					startedAt: new Date("2026-08-11T18:00:00.000Z"),
					endedAt: null,
				},
			],
			drills: [
				{
					id: "loop-1",
					barsStart: 5,
					barsEnd: 8,
					dimension: "pedaling",
					requiredCorrect: 5,
					attemptsCompleted: 2,
				},
			],
		});

		const history = await listPieceHistory({ db, env }, "student-a", "p1");

		expect(history.pieceId).toBe("p1");
		expect(history.composer).toBe("Chopin");
		expect(history.sessions).toEqual([
			{
				sessionId: "s2",
				startedAt: "2026-08-25T18:00:00.000Z",
				endedAt: "2026-08-25T18:40:00.000Z",
			},
			{
				sessionId: "s1",
				startedAt: "2026-08-11T18:00:00.000Z",
				endedAt: null,
			},
		]);
		expect(history.openDrills).toHaveLength(1);
	});

	it("raises NotFoundError for a piece the catalog does not have", async () => {
		const db = makeDb({ piece: null, sessions: [], drills: [] });
		await expect(
			listPieceHistory({ db, env }, "student-a", "ghost"),
		).rejects.toBeInstanceOf(NotFoundError);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/api && bun run test src/services/repertoire.history.test.ts
```
Expected: FAIL -- `listPieceHistory is not a function` (no such export).

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to the top of `apps/api/src/services/repertoire.ts`:

```typescript
import { and, desc, eq, isNotNull } from "drizzle-orm";
import { pieces } from "../db/schema/catalog";
import { sessions } from "../db/schema/sessions";
import { NotFoundError } from "../lib/errors";
import type { ServiceContext } from "../lib/types";
import type { OpenDrill } from "./segment-loops";
import { listOpenLoopsForPiece } from "./segment-loops";
```

Append below `buildRepertoire`:

```typescript
export interface PieceSessionSummary {
	sessionId: string;
	/** ISO 8601 */
	startedAt: string;
	/** ISO 8601, or null for a session that never ended cleanly. */
	endedAt: string | null;
}

export interface PieceHistory {
	pieceId: string;
	composer: string;
	title: string;
	/** Most recent first. */
	sessions: PieceSessionSummary[];
	openDrills: OpenDrill[];
}

/**
 * Every piece this student has practised. Sessions with a NULL piece_id --
 * pieceless sessions, a permanent first-class state -- are excluded here
 * rather than filtered downstream, so buildRepertoire never has to decide
 * what an entry with no piece would even be.
 */
export async function listRepertoire(
	ctx: ServiceContext,
	studentId: string,
): Promise<RepertoireEntry[]> {
	const rows = await ctx.db
		.select({
			pieceId: pieces.pieceId,
			composer: pieces.composer,
			title: pieces.title,
			startedAt: sessions.startedAt,
		})
		.from(sessions)
		.innerJoin(pieces, eq(sessions.pieceId, pieces.pieceId))
		.where(
			and(eq(sessions.studentId, studentId), isNotNull(sessions.pieceId)),
		);

	return buildRepertoire(rows);
}

export async function listPieceHistory(
	ctx: ServiceContext,
	studentId: string,
	pieceId: string,
): Promise<PieceHistory> {
	const [piece] = await ctx.db
		.select({
			pieceId: pieces.pieceId,
			composer: pieces.composer,
			title: pieces.title,
		})
		.from(pieces)
		.where(eq(pieces.pieceId, pieceId))
		.limit(1);

	if (!piece) throw new NotFoundError("piece", pieceId);

	const rows = await ctx.db
		.select({
			sessionId: sessions.id,
			startedAt: sessions.startedAt,
			endedAt: sessions.endedAt,
		})
		.from(sessions)
		.where(
			and(eq(sessions.studentId, studentId), eq(sessions.pieceId, pieceId)),
		)
		.orderBy(desc(sessions.startedAt));

	const openDrills = await listOpenLoopsForPiece(ctx.db, studentId, pieceId);

	return {
		...piece,
		sessions: rows.map((r) => ({
			sessionId: r.sessionId,
			startedAt: r.startedAt.toISOString(),
			endedAt: r.endedAt ? r.endedAt.toISOString() : null,
		})),
		openDrills,
	};
}
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/api && bun run test src/services/repertoire.history.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/services/repertoire.ts apps/api/src/services/repertoire.history.test.ts && git commit -m "feat(api): listRepertoire and listPieceHistory queries"
```

---

## Task 8: `GET /api/repertoire` and `GET /api/repertoire/:pieceId`

**Group:** C (parallel with Task 9)

**Behavior being verified:** An authenticated student can fetch their
repertoire and one piece's history over HTTP; an unauthenticated caller gets
401.
**Interface under test:** the mounted HTTP routes

**Files:**
- Create: `apps/api/src/routes/repertoire.ts`
- Modify: `apps/api/src/index.ts`
- Test: `apps/api/src/routes/repertoire.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/routes/repertoire.test.ts
import { Hono } from "hono";
import { describe, expect, it, vi } from "vitest";
import type { Bindings, Db, Variables } from "../lib/types";
import { errorHandler } from "../middleware/error-handler";
import * as repertoireService from "../services/repertoire";
import { repertoireRoutes } from "./repertoire";

const noAuthApp = new Hono()
	.onError(errorHandler)
	.route("/api/repertoire", repertoireRoutes);

function makeAuthApp(studentId: string) {
	return new Hono<{ Bindings: Bindings; Variables: Variables }>()
		.use("*", async (c, next) => {
			c.set("studentId", studentId);
			c.set("db", {} as unknown as Db);
			await next();
		})
		.onError(errorHandler)
		.route("/api/repertoire", repertoireRoutes);
}

describe("repertoire routes", () => {
	it("GET /api/repertoire returns 401 without auth", async () => {
		const res = await noAuthApp.request("/api/repertoire");
		expect(res.status).toBe(401);
	});

	it("GET /api/repertoire/:pieceId returns 401 without auth", async () => {
		const res = await noAuthApp.request("/api/repertoire/p1");
		expect(res.status).toBe(401);
	});

	it("GET /api/repertoire returns the student's entries", async () => {
		vi.spyOn(repertoireService, "listRepertoire").mockResolvedValueOnce([
			{
				pieceId: "p1",
				composer: "Chopin",
				title: "Nocturne Op. 9 No. 2",
				sessionCount: 3,
				lastPracticedAt: "2026-08-25T18:00:00.000Z",
				openMarkCount: null,
			},
		]);

		const res = await makeAuthApp("student-a").request("/api/repertoire");
		expect(res.status).toBe(200);
		expect(await res.json()).toEqual({
			entries: [
				{
					pieceId: "p1",
					composer: "Chopin",
					title: "Nocturne Op. 9 No. 2",
					sessionCount: 3,
					lastPracticedAt: "2026-08-25T18:00:00.000Z",
					openMarkCount: null,
				},
			],
		});
	});

	it("GET /api/repertoire/:pieceId maps NotFoundError to 404", async () => {
		const { NotFoundError } = await import("../lib/errors");
		vi.spyOn(repertoireService, "listPieceHistory").mockRejectedValueOnce(
			new NotFoundError("piece", "ghost"),
		);

		const res = await makeAuthApp("student-a").request(
			"/api/repertoire/ghost",
		);
		expect(res.status).toBe(404);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/api && bun run test src/routes/repertoire.test.ts
```
Expected: FAIL -- `Failed to resolve import "./repertoire"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/routes/repertoire.ts
import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import * as repertoireService from "../services/repertoire";

const pieceIdParamSchema = z.object({ pieceId: z.string().min(1) });

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>()
	.get("/", async (c) => {
		requireAuth(c.var.studentId);
		const entries = await repertoireService.listRepertoire(
			{ db: c.var.db, env: c.env },
			c.var.studentId,
		);
		return c.json({ entries });
	})
	.get("/:pieceId", validate("param", pieceIdParamSchema), async (c) => {
		requireAuth(c.var.studentId);
		const { pieceId } = c.req.valid("param");
		const history = await repertoireService.listPieceHistory(
			{ db: c.var.db, env: c.env },
			c.var.studentId,
			pieceId,
		);
		return c.json(history);
	});

export { app as repertoireRoutes };
```

In `apps/api/src/index.ts`, add the import beside the other route imports:

```typescript
import { repertoireRoutes } from "./routes/repertoire";
```

and extend the existing chain (**[TS-HONO-002]** -- append to the chain, never
a separate `app.route(...)` statement) so it reads:

```typescript
	.route("/api/practice", practiceRoutes)
	.route("/api/sessions", sessionsRoutes)
	.route("/api/repertoire", repertoireRoutes);
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/api && bun run test src/routes/repertoire.test.ts && bun run typecheck
```
Expected: PASS, and typecheck clean.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/routes/repertoire.ts apps/api/src/routes/repertoire.test.ts apps/api/src/index.ts && git commit -m "feat(api): GET /api/repertoire and /api/repertoire/:pieceId"
```

---

## Task 9: `POST /api/practice/start` records the piece

**Group:** C (parallel with Task 8)

**Behavior being verified:** Starting a session with a catalog `pieceId`
stores it on the session row; starting with an id the catalog does not have is
rejected rather than stored.
**Interface under test:** `POST /api/practice/start`

**Files:**
- Modify: `apps/api/src/routes/practice.ts`
- Test: `apps/api/src/routes/practice.piece.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/routes/practice.piece.test.ts
import { Hono } from "hono";
import { describe, expect, it, vi } from "vitest";
import type { Bindings, Db, Variables } from "../lib/types";
import { errorHandler } from "../middleware/error-handler";
import { practiceRoutes } from "./practice";

/** Records what was inserted into `sessions` so the test can assert on it. */
function makeDb(captured: { values?: Record<string, unknown> }, piece: unknown) {
	return {
		query: {
			conversations: { findFirst: async () => undefined },
			pieces: { findFirst: async () => piece },
		},
		insert: (table: unknown) => ({
			values: (v: Record<string, unknown>) => {
				const name = String((table as { _?: { name?: string } })._?.name ?? "");
				if (name === "sessions") captured.values = v;
				return {
					returning: async () => [{ id: "00000000-0000-0000-0000-0000000000aa" }],
				};
			},
		}),
	} as unknown as Db;
}

function makeApp(db: Db) {
	return new Hono<{ Bindings: Bindings; Variables: Variables }>()
		.use("*", async (c, next) => {
			c.set("studentId", "student-a");
			c.set("db", db);
			await next();
		})
		.onError(errorHandler)
		.route("/api/practice", practiceRoutes);
}

describe("POST /api/practice/start -- piece identity", () => {
	it("stores a catalog pieceId on the session", async () => {
		const captured: { values?: Record<string, unknown> } = {};
		const db = makeDb(captured, { pieceId: "chopin-nocturne-op9-no2" });

		const res = await makeApp(db).request("/api/practice/start", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ pieceId: "chopin-nocturne-op9-no2" }),
		});

		expect(res.status).toBe(201);
		expect(captured.values?.pieceId).toBe("chopin-nocturne-op9-no2");
	});

	it("rejects a pieceId the catalog does not have, rather than storing it", async () => {
		const captured: { values?: Record<string, unknown> } = {};
		const db = makeDb(captured, undefined);

		const res = await makeApp(db).request("/api/practice/start", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ pieceId: "ghost" }),
		});

		expect(res.status).toBe(404);
		expect(captured.values).toBeUndefined();
	});

	it("still starts a pieceless session when no pieceId is sent", async () => {
		const captured: { values?: Record<string, unknown> } = {};
		const db = makeDb(captured, undefined);

		const res = await makeApp(db).request("/api/practice/start", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({}),
		});

		expect(res.status).toBe(201);
		expect(captured.values?.pieceId).toBeUndefined();
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/api && bun run test src/routes/practice.piece.test.ts
```
Expected: FAIL -- the first case fails with
`expected undefined to be 'chopin-nocturne-op9-no2'`, because `pieceId` is
neither accepted nor stored.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/api/src/routes/practice.ts`, widen the start body schema:

```typescript
const startBodySchema = z.object({
	conversationId: z.string().uuid().optional(),
	pieceId: z.string().min(1).optional(),
});
```

Inside the `/start` handler, after the `conversationId` block resolves and
before the `sessions` insert, validate the piece and pass it through:

```typescript
		// Validated against the catalog rather than trusted: a session row
		// pointing at a piece the catalog lacks would produce a repertoire entry
		// with no composer or title, forcing buildRepertoire to invent one or
		// drop the row. Rejecting here means it never has to decide.
		if (body.pieceId) {
			const piece = await db.query.pieces.findFirst({
				where: (p, { eq: e }) => e(p.pieceId, body.pieceId!),
			});
			if (!piece) throw new NotFoundError("piece", body.pieceId);
		}

		const [session] = await db
			.insert(sessions)
			.values({ studentId, conversationId, pieceId: body.pieceId })
			.returning({ id: sessions.id });
```

`NotFoundError` is already imported in this file.

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/api && bun run test src/routes/practice.piece.test.ts src/routes/practice.test.ts
```
Expected: PASS, including the pre-existing `practice.test.ts` suite.

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/routes/practice.ts apps/api/src/routes/practice.piece.test.ts && git commit -m "feat(api): POST /practice/start records a validated catalog pieceId"
```

---

## Task 10: Web API client and shared query options

**Group:** D (alone -- Tasks 11 and 13 import the types this task creates)

**Behavior being verified:** The browser client fetches the repertoire, a
piece's history, and the score catalog from the right URLs with credentials,
and returns the parsed bodies.
**Interface under test:** `api.repertoire.list()`,
`api.repertoire.history(pieceId)`, `api.scores.list()`

**Files:**
- Modify: `apps/web/src/lib/api.ts`
- Create: `apps/web/src/hooks/useRepertoire.ts`
- Create: `apps/web/src/hooks/usePieceCatalog.ts`
- Test: `apps/web/src/lib/api.repertoire.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/api.repertoire.test.ts
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "./api";

afterEach(() => {
	vi.unstubAllGlobals();
});

function stubFetch(body: unknown) {
	const spy = vi.fn().mockResolvedValue(
		new Response(JSON.stringify(body), {
			status: 200,
			headers: { "Content-Type": "application/json" },
		}),
	);
	vi.stubGlobal("fetch", spy);
	return spy;
}

describe("api.repertoire", () => {
	it("lists the student's repertoire", async () => {
		const entry = {
			pieceId: "p1",
			composer: "Chopin",
			title: "Nocturne Op. 9 No. 2",
			sessionCount: 3,
			lastPracticedAt: "2026-08-25T18:00:00.000Z",
			openMarkCount: null,
		};
		const spy = stubFetch({ entries: [entry] });

		const entries = await api.repertoire.list();

		expect(entries).toEqual([entry]);
		expect(spy.mock.calls[0][0]).toMatch(/\/api\/repertoire$/);
	});

	it("fetches one piece's history", async () => {
		const spy = stubFetch({
			pieceId: "p1",
			composer: "Chopin",
			title: "Nocturne",
			sessions: [],
			openDrills: [],
		});

		const history = await api.repertoire.history("p1");

		expect(history.pieceId).toBe("p1");
		expect(spy.mock.calls[0][0]).toMatch(/\/api\/repertoire\/p1$/);
	});

	it("lists the score catalog", async () => {
		const spy = stubFetch({
			pieces: [
				{ pieceId: "p1", composer: "Chopin", title: "Nocturne Op. 9 No. 2" },
			],
		});

		const pieces = await api.scores.list();

		expect(pieces).toHaveLength(1);
		expect(spy.mock.calls[0][0]).toMatch(/\/api\/scores$/);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && bun run test src/lib/api.repertoire.test.ts
```
Expected: FAIL -- `Cannot read properties of undefined (reading 'list')`
(`api.repertoire` does not exist).

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/lib/api.ts`, add these types above the `api` object:

```typescript
export interface RepertoireEntry {
	pieceId: string;
	composer: string;
	title: string;
	sessionCount: number;
	lastPracticedAt: string;
	openMarkCount: number | null;
}

export interface PieceSessionSummary {
	sessionId: string;
	startedAt: string;
	endedAt: string | null;
}

export interface OpenDrill {
	id: string;
	barsStart: number;
	barsEnd: number;
	dimension: string | null;
	requiredCorrect: number;
	attemptsCompleted: number;
}

export interface PieceHistory {
	pieceId: string;
	composer: string;
	title: string;
	sessions: PieceSessionSummary[];
	openDrills: OpenDrill[];
}
```

Add a `repertoire` block to the `api` object, and a `list` to the existing
`scores` block. `request()` already captures to Sentry and throws `ApiError`
on a non-OK response, which is what **[WEB-ERR-001]** requires -- do not add a
`catch` here:

```typescript
	repertoire: {
		async list(): Promise<RepertoireEntry[]> {
			const body = await request<{ entries: RepertoireEntry[] }>(
				"/api/repertoire",
			);
			return body.entries;
		},

		async history(pieceId: string): Promise<PieceHistory> {
			return request<PieceHistory>(`/api/repertoire/${pieceId}`);
		},
	},
```

and inside `scores`:

```typescript
		async list(): Promise<PieceSummary[]> {
			const body = await request<{ pieces: PieceSummary[] }>("/api/scores");
			return body.pieces;
		},
```

with `import type { PieceSummary } from "./piece-search";` at the top.

Then the two query-options modules. **[WEB-QUERY-001]**: these are the only
places a repertoire or catalog `queryKey` is ever written, mirroring
`hooks/useAuth.ts`'s `authQueryOptions`.

```typescript
// apps/web/src/hooks/useRepertoire.ts
import { queryOptions } from "@tanstack/react-query";
import { api } from "../lib/api";

export const repertoireQueryOptions = queryOptions({
	queryKey: ["repertoire", "list"] as const,
	queryFn: () => api.repertoire.list(),
});

export const pieceHistoryQueryOptions = (pieceId: string) =>
	queryOptions({
		queryKey: ["repertoire", "piece", pieceId] as const,
		queryFn: () => api.repertoire.history(pieceId),
	});
```

```typescript
// apps/web/src/hooks/usePieceCatalog.ts
import { queryOptions } from "@tanstack/react-query";
import { api } from "../lib/api";

// The catalog is codegen-seeded and effectively static within a session, so
// it is fetched once and reused by the add-piece search.
export const pieceCatalogQueryOptions = queryOptions({
	queryKey: ["scores", "catalog"] as const,
	queryFn: () => api.scores.list(),
	staleTime: Number.POSITIVE_INFINITY,
});
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && bun run test src/lib/api.repertoire.test.ts && bun run typecheck
```
Expected: PASS, typecheck clean.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/api.ts apps/web/src/lib/api.repertoire.test.ts apps/web/src/hooks/useRepertoire.ts apps/web/src/hooks/usePieceCatalog.ts && git commit -m "feat(web): repertoire and catalog API client with shared query options"
```

---

## Task 11: `RepertoireCard`

**Group:** E (parallel with Tasks 12, 13, 14; depends on Task 10)

**Behavior being verified:** A card shows a piece's composer, title, and arc
line, omits the mark count when it is unknown, and reports which piece was
opened.
**Interface under test:** the `RepertoireCard` component

**Files:**
- Create: `apps/web/src/components/RepertoireCard.tsx`
- Test: `apps/web/src/components/RepertoireCard.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/RepertoireCard.test.tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { RepertoireEntry } from "../lib/api";
import { RepertoireCard } from "./RepertoireCard";

const NOW = new Date("2026-08-27T12:00:00.000Z");

const ENTRY: RepertoireEntry = {
	pieceId: "chopin-nocturne-op9-no2",
	composer: "Chopin",
	title: "Nocturne Op. 9 No. 2",
	sessionCount: 6,
	lastPracticedAt: "2026-08-26T18:00:00.000Z",
	openMarkCount: null,
};

describe("RepertoireCard", () => {
	it("shows the piece and its arc line", () => {
		render(<RepertoireCard entry={ENTRY} now={NOW} onOpen={vi.fn()} />);
		expect(screen.getByText("Nocturne Op. 9 No. 2")).toBeInTheDocument();
		expect(screen.getByText("Chopin")).toBeInTheDocument();
		expect(
			screen.getByText("6 sessions · last practiced yesterday"),
		).toBeInTheDocument();
	});

	it("omits the mark count when it is unknown, rather than showing zero", () => {
		render(<RepertoireCard entry={ENTRY} now={NOW} onOpen={vi.fn()} />);
		expect(screen.queryByTestId("open-mark-count")).not.toBeInTheDocument();
	});

	it("shows the mark count once one is known", () => {
		render(
			<RepertoireCard
				entry={{ ...ENTRY, openMarkCount: 3 }}
				now={NOW}
				onOpen={vi.fn()}
			/>,
		);
		expect(screen.getByTestId("open-mark-count")).toHaveTextContent(
			"3 open marks",
		);
	});

	it("reports which piece was opened", () => {
		const onOpen = vi.fn();
		render(<RepertoireCard entry={ENTRY} now={NOW} onOpen={onOpen} />);
		fireEvent.click(
			screen.getByRole("button", { name: /Nocturne Op\. 9 No\. 2/ }),
		);
		expect(onOpen).toHaveBeenCalledWith("chopin-nocturne-op9-no2");
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && bun run test src/components/RepertoireCard.test.tsx
```
Expected: FAIL -- `Failed to resolve import "./RepertoireCard"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```tsx
// apps/web/src/components/RepertoireCard.tsx
import type { RepertoireEntry } from "../lib/api";
import { describeArc } from "../lib/learning-arc";

interface RepertoireCardProps {
	entry: RepertoireEntry;
	/** Injected clock, so the arc line is deterministic and timezone-safe. */
	now: Date;
	onOpen: (pieceId: string) => void;
}

/**
 * One piece on the home surface. A callback rather than a <Link> so this
 * renders and is testable without a router context; routes/app.index.tsx owns
 * navigation.
 *
 * openMarkCount renders only when non-null. There is no persisted mark store
 * yet, so the value is null and the count is simply absent -- the system falls
 * silent rather than showing a 0 a student would read as "nothing to work on".
 */
export function RepertoireCard({ entry, now, onOpen }: RepertoireCardProps) {
	return (
		<button
			type="button"
			onClick={() => onOpen(entry.pieceId)}
			// py-3 keeps the target well past the 32px minimum (WEB-A11Y-003).
			className="flex w-full flex-col items-start gap-1 rounded-lg border border-border-subtle bg-surface-raised px-4 py-3 text-left"
		>
			<span className="font-display text-body text-ink-primary">
				{entry.title}
			</span>
			<span className="text-body-sm text-ink-secondary">{entry.composer}</span>
			<span className="text-body-xs text-ink-tertiary">
				{describeArc({
					sessionCount: entry.sessionCount,
					lastPracticedAt: entry.lastPracticedAt,
					now,
				})}
			</span>
			{entry.openMarkCount !== null && (
				<span
					data-testid="open-mark-count"
					className="text-body-xs text-ink-tertiary"
				>
					{entry.openMarkCount} open marks
				</span>
			)}
		</button>
	);
}
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && bun run test src/components/RepertoireCard.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/RepertoireCard.tsx apps/web/src/components/RepertoireCard.test.tsx && git commit -m "feat(web): RepertoireCard shows a piece, its arc, and its open marks"
```

---

## Task 12: `AddPieceSearch`

**Group:** E (parallel with Tasks 11, 13, 14; depends on Task 10)

**Behavior being verified:** Typing filters the score library live and picking
a result reports the chosen piece.
**Interface under test:** the `AddPieceSearch` component

**Files:**
- Create: `apps/web/src/components/AddPieceSearch.tsx`
- Test: `apps/web/src/components/AddPieceSearch.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/AddPieceSearch.test.tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { PieceSummary } from "../lib/piece-search";
import { AddPieceSearch } from "./AddPieceSearch";

const CATALOG: readonly PieceSummary[] = [
	{ pieceId: "chopin-nocturne-op9-no2", composer: "Chopin", title: "Nocturne Op. 9 No. 2" },
	{ pieceId: "bach-invention-1", composer: "Bach", title: "Invention No. 1" },
];

describe("AddPieceSearch", () => {
	it("shows no results before anything is typed", () => {
		render(<AddPieceSearch catalog={CATALOG} onPick={vi.fn()} />);
		expect(screen.queryByRole("button", { name: /Nocturne/ })).not.toBeInTheDocument();
		expect(screen.queryByRole("button", { name: /Invention/ })).not.toBeInTheDocument();
	});

	it("filters the library as the student types", () => {
		render(<AddPieceSearch catalog={CATALOG} onPick={vi.fn()} />);
		fireEvent.change(screen.getByRole("searchbox"), {
			target: { value: "bach" },
		});
		expect(screen.getByRole("button", { name: /Invention No\. 1/ })).toBeInTheDocument();
		expect(screen.queryByRole("button", { name: /Nocturne/ })).not.toBeInTheDocument();
	});

	it("reports the picked piece", () => {
		const onPick = vi.fn();
		render(<AddPieceSearch catalog={CATALOG} onPick={onPick} />);
		fireEvent.change(screen.getByRole("searchbox"), {
			target: { value: "nocturne" },
		});
		fireEvent.click(screen.getByRole("button", { name: /Nocturne Op\. 9 No\. 2/ }));
		expect(onPick).toHaveBeenCalledWith("chopin-nocturne-op9-no2");
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && bun run test src/components/AddPieceSearch.test.tsx
```
Expected: FAIL -- `Failed to resolve import "./AddPieceSearch"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```tsx
// apps/web/src/components/AddPieceSearch.tsx
import { useState } from "react";
import type { PieceSummary } from "../lib/piece-search";
import { searchPieces } from "../lib/piece-search";

interface AddPieceSearchProps {
	catalog: readonly PieceSummary[];
	onPick: (pieceId: string) => void;
}

/**
 * Score-library search for the home surface. The query string is the only
 * client state this feature has; the catalog itself is server state and stays
 * in react-query (WEB-QUERY-002).
 */
export function AddPieceSearch({ catalog, onPick }: AddPieceSearchProps) {
	const [query, setQuery] = useState("");

	// Computed during render, not mirrored into state by an effect
	// (WEB-STATE-001), and deliberately not memoized: nothing downstream takes
	// it as a dependency and the catalog slice is small (WEB-STATE-002).
	const results = searchPieces(catalog, query);

	return (
		<div className="flex flex-col gap-2">
			<input
				type="search"
				value={query}
				onChange={(e) => setQuery(e.target.value)}
				placeholder="Search the score library"
				className="w-full rounded-lg border border-border-subtle bg-surface-raised px-4 py-3 text-body-sm text-ink-primary"
			/>
			{results.length > 0 && (
				<ul className="flex flex-col gap-1">
					{results.map((piece) => (
						<li key={piece.pieceId}>
							<button
								type="button"
								onClick={() => onPick(piece.pieceId)}
								className="flex w-full flex-col items-start rounded-lg px-4 py-2 text-left"
							>
								<span className="text-body-sm text-ink-primary">
									{piece.title}
								</span>
								<span className="text-body-xs text-ink-tertiary">
									{piece.composer}
								</span>
							</button>
						</li>
					))}
				</ul>
			)}
		</div>
	);
}
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && bun run test src/components/AddPieceSearch.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/AddPieceSearch.tsx apps/web/src/components/AddPieceSearch.test.tsx && git commit -m "feat(web): AddPieceSearch filters the score library as you type"
```

---

## Task 13: `PieceSessionHistory`

**Group:** E (parallel with Tasks 11, 12, 14; depends on Task 10)

**Behavior being verified:** A piece's past sessions render as a dated list,
and a piece with no sessions says so rather than rendering an empty box.
**Interface under test:** the `PieceSessionHistory` component

**Files:**
- Create: `apps/web/src/components/PieceSessionHistory.tsx`
- Test: `apps/web/src/components/PieceSessionHistory.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/PieceSessionHistory.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { PieceSessionSummary } from "../lib/api";
import { PieceSessionHistory } from "./PieceSessionHistory";

const SESSIONS: PieceSessionSummary[] = [
	{
		sessionId: "s2",
		startedAt: "2026-08-25T18:00:00.000Z",
		endedAt: "2026-08-25T18:40:00.000Z",
	},
	{
		sessionId: "s1",
		startedAt: "2026-08-11T18:00:00.000Z",
		endedAt: null,
	},
];

describe("PieceSessionHistory", () => {
	it("lists each session with its date", () => {
		render(<PieceSessionHistory sessions={SESSIONS} />);
		expect(screen.getAllByRole("listitem")).toHaveLength(2);
		expect(screen.getByText("25 Aug 2026")).toBeInTheDocument();
		expect(screen.getByText("11 Aug 2026")).toBeInTheDocument();
	});

	it("shows how long a completed session ran", () => {
		render(<PieceSessionHistory sessions={SESSIONS} />);
		expect(screen.getByText("40 min")).toBeInTheDocument();
	});

	it("says so when the piece has no sessions yet", () => {
		render(<PieceSessionHistory sessions={[]} />);
		expect(screen.getByText("No sessions yet")).toBeInTheDocument();
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && bun run test src/components/PieceSessionHistory.test.tsx
```
Expected: FAIL -- `Failed to resolve import "./PieceSessionHistory"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```tsx
// apps/web/src/components/PieceSessionHistory.tsx
import type { PieceSessionSummary } from "../lib/api";

interface PieceSessionHistoryProps {
	sessions: readonly PieceSessionSummary[];
}

/**
 * UTC formatting throughout: session timestamps are stored UTC and this repo's
 * conventions avoid local-clock assumptions, which have caused drift before.
 */
function formatDate(iso: string): string {
	return new Date(iso).toLocaleDateString("en-GB", {
		day: "numeric",
		month: "short",
		year: "numeric",
		timeZone: "UTC",
	});
}

/** null endedAt means the session never ended cleanly -- no duration to show. */
function formatDuration(
	startedAt: string,
	endedAt: string | null,
): string | null {
	if (endedAt === null) return null;
	const minutes = Math.round(
		(Date.parse(endedAt) - Date.parse(startedAt)) / 60000,
	);
	return `${minutes} min`;
}

export function PieceSessionHistory({ sessions }: PieceSessionHistoryProps) {
	if (sessions.length === 0) {
		return <p className="text-body-sm text-ink-tertiary">No sessions yet</p>;
	}

	return (
		<ul className="flex flex-col gap-1">
			{sessions.map((session) => {
				const duration = formatDuration(session.startedAt, session.endedAt);
				return (
					<li
						key={session.sessionId}
						className="flex items-baseline justify-between border-b border-border-subtle py-2"
					>
						<span className="text-body-sm text-ink-primary">
							{formatDate(session.startedAt)}
						</span>
						{duration && (
							<span className="text-body-xs text-ink-tertiary">{duration}</span>
						)}
					</li>
				);
			})}
		</ul>
	);
}
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && bun run test src/components/PieceSessionHistory.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/PieceSessionHistory.tsx apps/web/src/components/PieceSessionHistory.test.tsx && git commit -m "feat(web): PieceSessionHistory lists a piece's past sessions"
```

---

## Task 14: `usePracticeSession` yields a real `confidentGuess`

**Group:** E (parallel with Tasks 11, 12, 13)

**Behavior being verified:** A `piece_identified` WebSocket event becomes a
`ConfidentGuess` the caller can read, instead of a console log.
**Interface under test:** the `confidentGuess` field of
`usePracticeSession()`'s return value

**Files:**
- Modify: `apps/web/src/hooks/usePracticeSession.ts`
- Test: `apps/web/src/hooks/usePracticeSession.piece.test.ts`

Read `apps/web/src/hooks/usePracticeSession.sessionid.test.ts` first and reuse
its harness for driving WebSocket events; do not invent a second one.

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/hooks/usePracticeSession.piece.test.ts
//
// Harness copied from usePracticeSession.sessionid.test.ts -- the established
// way to drive WS events in this codebase. Events arrive through the fake
// socket's onmessage, exactly as they do in the browser; the hook gets no
// test-only export.
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

describe("usePracticeSession -- piece identity", () => {
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

	it("surfaces piece_identified as a ConfidentGuess carrying the catalog pieceId", async () => {
		const { result } = renderHook(() => usePracticeSession());

		expect(result.current.confidentGuess).toBeNull();

		await act(async () => {
			await result.current.start();
		});
		await waitFor(() => expect(result.current.state).toBe("recording"));

		act(() => {
			socket.onmessage?.({
				data: JSON.stringify({
					type: "piece_identified",
					pieceId: "chopin-nocturne-op9-no2",
					composer: "Chopin",
					title: "Nocturne Op. 9 No. 2",
					confidence: 0.92,
					method: "chroma",
				}),
			} as MessageEvent);
		});

		await waitFor(() =>
			expect(result.current.confidentGuess).toEqual({
				pieceId: "chopin-nocturne-op9-no2",
				composer: "Chopin",
				title: "Nocturne Op. 9 No. 2",
				confidence: 0.92,
			}),
		);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && bun run test src/hooks/usePracticeSession.piece.test.ts
```
Expected: FAIL -- `expected undefined to be null` (`confidentGuess` is not on
the returned object).

- [ ] **Step 3: Implement the minimum to make the test pass**

Add the import at the top of `apps/web/src/hooks/usePracticeSession.ts`:

```typescript
import type { ConfidentGuess } from "../lib/piece-ladder";
```

Add the field to `UsePracticeSessionReturn`, next to `marks`:

```typescript
	confidentGuess: ConfidentGuess | null;
```

Add the state beside the other `useState` declarations:

```typescript
	const [confidentGuess, setConfidentGuess] = useState<ConfidentGuess | null>(
		null,
	);
```

Replace the `piece_identified` case (currently at ~line 312) entirely:

```typescript
				case "piece_identified":
					// The event already carries exactly ConfidentGuess's four fields.
					// The hook does NOT decide whether the guess wins --
					// resolvePieceLadderState owns that precedence, and duplicating
					// it here would give the codebase two copies of one policy.
					setConfidentGuess({
						pieceId: data.pieceId,
						composer: data.composer,
						title: data.title,
						confidence: data.confidence,
					});
					break;
```

Add `confidentGuess` to the returned object, next to `marks`:

```typescript
		confidentGuess,
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && bun run test src/hooks/usePracticeSession
```
Expected: PASS, including the four pre-existing `usePracticeSession.*` suites.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/hooks/usePracticeSession.ts apps/web/src/hooks/usePracticeSession.piece.test.ts && git commit -m "feat(web): usePracticeSession surfaces piece_identified as a ConfidentGuess"
```

---

## Task 15: `/app` becomes a layout; practice moves to `/app/practice`

**Group:** F (single, after Group E; Tasks 16 and 17 depend on it)

**Behavior being verified:** The practice surface still works, now at
`/app/practice`, and starts with the piece named by `?pieceId=` already on the
stand -- discharging #158's inbound dependency.
**Interface under test:** the `/app/practice` route component

**Files:**
- Modify: `apps/web/src/routes/app.tsx`
- Modify: `apps/web/src/routes/app.index.tsx`
- Create: `apps/web/src/routes/app.practice.tsx`
- Test: `apps/web/src/routes/app.practice.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/routes/app.practice.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AppPractice } from "./app.practice";

vi.mock("@tanstack/react-router", async (importOriginal) => ({
	...(await importOriginal<typeof import("@tanstack/react-router")>()),
	useNavigate: () => vi.fn(),
}));

// PracticeMode's real subtree pulls in Verovio; the ladder decision this test
// checks is visible from PracticeMode's props alone.
vi.mock("../components/PracticeMode", () => ({
	PracticeMode: (props: { userPickedPieceId: string | null }) => (
		<div
			data-testid="practice-mode"
			data-picked-piece={props.userPickedPieceId ?? "none"}
		/>
	),
}));

describe("AppPractice", () => {
	it("puts the picked piece on the stand", () => {
		render(<AppPractice pieceId="chopin-nocturne-op9-no2" />);
		// The start screen precedes the stand; the piece must survive it.
		screen.getByRole("button", { name: /start practicing/i }).click();
		expect(screen.getByTestId("practice-mode")).toHaveAttribute(
			"data-picked-piece",
			"chopin-nocturne-op9-no2",
		);
	});

	it("starts a pieceless session when no piece was picked", () => {
		render(<AppPractice pieceId={null} />);
		screen.getByRole("button", { name: /start practicing/i }).click();
		expect(screen.getByTestId("practice-mode")).toHaveAttribute(
			"data-picked-piece",
			"none",
		);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && bun run test src/routes/app.practice.test.tsx
```
Expected: FAIL -- `Failed to resolve import "./app.practice"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

Rewrite `apps/web/src/routes/app.tsx` as a layout. It keeps the auth guard
verbatim and gains an `<Outlet/>`; everything else moves out:

```tsx
import { createFileRoute, Outlet, redirect } from "@tanstack/react-router";
import { authQueryOptions } from "../hooks/useAuth";
import { queryClient } from "../lib/query-client";

export const Route = createFileRoute("/app")({
	beforeLoad: async () => {
		if (import.meta.env.VITE_AUTH_MODE !== "live") return;
		try {
			const user = await queryClient.ensureQueryData(authQueryOptions);
			if (!user) {
				throw redirect({ to: "/signin" });
			}
		} catch (err) {
			if (err && typeof err === "object" && "to" in err) throw err;
			throw redirect({ to: "/signin" });
		}
	},
	component: AppLayout,
});

/**
 * The /app shell. Carries the auth guard for every child route -- home, the
 * practice stand, and the piece page -- so the guard is written once rather
 * than repeated per route.
 */
function AppLayout() {
	return <Outlet />;
}
```

Create `apps/web/src/routes/app.practice.tsx` with the surface moved out of
the old `app.tsx`. It is unchanged except that it takes its piece from the
route's search params:

```tsx
import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { PracticeMode } from "../components/PracticeMode";
import { usePracticeSession } from "../hooks/usePracticeSession";
import type { VerdictState } from "../lib/session-review";

export const Route = createFileRoute("/app/practice")({
	validateSearch: (search: Record<string, unknown>) => ({
		pieceId: typeof search.pieceId === "string" ? search.pieceId : null,
	}),
	component: RouteComponent,
});

function RouteComponent() {
	const { pieceId } = Route.useSearch();
	return <AppPractice pieceId={pieceId} />;
}

interface AppPracticeProps {
	/** From ?pieceId=. A search param, not a route param: a pieceless session
	 * is a permanent first-class state, and /app/practice/none would be a lie
	 * about the URL space. */
	pieceId: string | null;
}

/**
 * Score-first practice stand. Exported separately from the route so it can be
 * rendered in tests without a router, matching how routes/index.tsx's page is
 * reached in index.test.tsx.
 *
 * #160 closes #158's inbound dependency here: userPickedPieceId is the piece
 * the student chose on home, and confidentGuess is the live piece_identified
 * result. Both were hardcoded null before.
 */
export function AppPractice({ pieceId }: AppPracticeProps) {
	const [view, setView] = useState<"start" | "practice" | "review">("start");

	const practice = usePracticeSession({
		onSummary: () => setView("review"),
	});

	function handleStart() {
		setView("practice");
		practice.start();
	}

	if (view === "start" && practice.state === "idle") {
		return (
			<div className="flex h-dvh flex-col items-center justify-center gap-6 bg-surface-page px-6">
				<img
					src="/icon_nobackground.png"
					alt=""
					className="w-20 h-20 opacity-50"
				/>
				<button
					type="button"
					onClick={handleStart}
					className="rounded-full bg-accent px-6 py-3 text-body-sm text-on-accent"
				>
					Start practicing
				</button>
			</div>
		);
	}

	const verdict: VerdictState = practice.verdict
		? { status: "ready", text: practice.verdict }
		: practice.state === "error"
			? { status: "failed" }
			: { status: "loading" };

	return (
		<div className="fixed inset-0 z-50 bg-surface-page">
			<PracticeMode
				userPickedPieceId={pieceId}
				confidentGuess={practice.confidentGuess}
				marks={practice.marks}
				elapsedSeconds={practice.elapsedSeconds}
				isPlaying={practice.isPlaying}
				isRecording={practice.state === "recording"}
				onStop={() => practice.stop()}
				phase={view === "review" ? "review" : "practice"}
				sessionId={practice.sessionId ?? undefined}
				verdict={verdict}
				carryForward={
					practice.carryForward ? { text: practice.carryForward } : null
				}
				wsGaps={[]}
				noCommentZones={practice.noCommentZones}
				onExitReview={() => setView("start")}
			/>
		</div>
	);
}
```

Leave `apps/web/src/routes/app.index.tsx` returning `null` for now; Task 16
replaces it. Update only its comment:

```tsx
import { createFileRoute } from "@tanstack/react-router";

// Home renders here (#160, Task 16). app.tsx is the /app layout.
export const Route = createFileRoute("/app/")({
	component: () => null,
});
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && bun run test src/routes/
```
Expected: PASS across every route suite. (Vitest runs with the
`tanstackStart()` plugin disabled and imports `./app.practice` directly, so
this step does not depend on `routeTree.gen.ts`.)

- [ ] **Step 5: Regenerate the route tree**

`createFileRoute("/app/practice")` type-checks only against paths TanStack
Router has already generated into `apps/web/src/routeTree.gen.ts`. That file
is produced by the `tanstackStart()` Vite plugin (`apps/web/vite.config.ts`),
not by a standalone CLI -- this repo has no `tsr`/router-cli binary. Running
the app's own build invokes that plugin and regenerates the file as a build
side effect (verified empirically in this worktree: adding a throwaway route
and running this command updated `routeTree.gen.ts` to include it).

```bash
cd apps/web && bun run build
```
Expected: build succeeds; `git diff apps/web/src/routeTree.gen.ts` shows the
new `/app/practice` route added.

- [ ] **Step 6: Commit the route file with the regenerated route tree**

```bash
git add apps/web/src/routes/app.tsx apps/web/src/routes/app.index.tsx apps/web/src/routes/app.practice.tsx apps/web/src/routes/app.practice.test.tsx apps/web/src/routeTree.gen.ts && git commit -m "refactor(web): /app becomes a layout; practice stand moves to /app/practice"
```

- [ ] **Step 7: Typecheck**

```bash
cd apps/web && bun run typecheck
```
Expected: typecheck clean (this depends on the regenerated
`routeTree.gen.ts` from Step 5 -- do not run this before Step 5/6).

---

## Task 16: Home

**Group:** G (parallel with Task 17; depends on Task 15)

**Behavior being verified:** Home renders one card per practised piece from
real session data, lets the student pick a new piece from the library or start
a pieceless session, and shows an explicit error when the fetch fails.
**Interface under test:** the `AppHome` component

**Files:**
- Modify: `apps/web/src/routes/app.index.tsx`
- Test: `apps/web/src/routes/app.index.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/routes/app.index.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

const navigate = vi.fn();
vi.mock("@tanstack/react-router", async (importOriginal) => ({
	...(await importOriginal<typeof import("@tanstack/react-router")>()),
	useNavigate: () => navigate,
}));

const listRepertoire = vi.fn();
const listScores = vi.fn();
vi.mock("../lib/api", () => ({
	api: {
		repertoire: { list: () => listRepertoire() },
		scores: { list: () => listScores() },
	},
}));

function wrap(ui: ReactNode) {
	const client = new QueryClient({
		defaultOptions: { queries: { retry: false } },
	});
	return render(
		<QueryClientProvider client={client}>{ui}</QueryClientProvider>,
	);
}

const ENTRY = {
	pieceId: "chopin-nocturne-op9-no2",
	composer: "Chopin",
	title: "Nocturne Op. 9 No. 2",
	sessionCount: 6,
	lastPracticedAt: "2026-08-26T18:00:00.000Z",
	openMarkCount: null,
};

async function renderHome() {
	const { AppHome } = await import("./app.index");
	return wrap(<AppHome />);
}

describe("AppHome", () => {
	it("renders one card per practised piece", async () => {
		listRepertoire.mockResolvedValue([
			ENTRY,
			{ ...ENTRY, pieceId: "bach-invention-1", title: "Invention No. 1" },
		]);
		listScores.mockResolvedValue([]);

		await renderHome();

		await waitFor(() =>
			expect(screen.getByText("Nocturne Op. 9 No. 2")).toBeInTheDocument(),
		);
		expect(screen.getByText("Invention No. 1")).toBeInTheDocument();
	});

	it("opens a piece's page when its card is chosen", async () => {
		listRepertoire.mockResolvedValue([ENTRY]);
		listScores.mockResolvedValue([]);

		await renderHome();
		await waitFor(() =>
			expect(screen.getByText("Nocturne Op. 9 No. 2")).toBeInTheDocument(),
		);
		fireEvent.click(screen.getByRole("button", { name: /Nocturne Op\. 9 No\. 2/ }));

		expect(navigate).toHaveBeenCalledWith({
			to: "/app/piece/$pieceId",
			params: { pieceId: "chopin-nocturne-op9-no2" },
		});
	});

	it("starts a piece-picked session straight from the library search", async () => {
		listRepertoire.mockResolvedValue([]);
		listScores.mockResolvedValue([
			{ pieceId: "bach-invention-1", composer: "Bach", title: "Invention No. 1" },
		]);

		await renderHome();
		await waitFor(() => expect(screen.getByRole("searchbox")).toBeInTheDocument());
		fireEvent.change(screen.getByRole("searchbox"), {
			target: { value: "invention" },
		});
		fireEvent.click(screen.getByRole("button", { name: /Invention No\. 1/ }));

		expect(navigate).toHaveBeenCalledWith({
			to: "/app/practice",
			search: { pieceId: "bach-invention-1" },
		});
	});

	it("starts a pieceless session from 'just play'", async () => {
		listRepertoire.mockResolvedValue([]);
		listScores.mockResolvedValue([]);

		await renderHome();
		await waitFor(() =>
			expect(screen.getByRole("button", { name: /just play/i })).toBeInTheDocument(),
		);
		fireEvent.click(screen.getByRole("button", { name: /just play/i }));

		expect(navigate).toHaveBeenCalledWith({
			to: "/app/practice",
			search: { pieceId: null },
		});
	});

	it("shows an explicit error when the repertoire cannot be loaded", async () => {
		listRepertoire.mockRejectedValue(new Error("network down"));
		listScores.mockResolvedValue([]);

		await renderHome();

		// An empty list and a failed fetch must not look alike.
		await waitFor(() =>
			expect(screen.getByText(/couldn't load your pieces/i)).toBeInTheDocument(),
		);
		expect(screen.getByRole("button", { name: /just play/i })).toBeInTheDocument();
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && bun run test src/routes/app.index.test.tsx
```
Expected: FAIL -- `AppHome is not exported from './app.index'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```tsx
// apps/web/src/routes/app.index.tsx
import { useQuery } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { AddPieceSearch } from "../components/AddPieceSearch";
import { RepertoireCard } from "../components/RepertoireCard";
import { pieceCatalogQueryOptions } from "../hooks/usePieceCatalog";
import { repertoireQueryOptions } from "../hooks/useRepertoire";

export const Route = createFileRoute("/app/")({
	component: AppHome,
});

/**
 * Home -- "your music" (docs/apps/05-ui-system.md#1). Exported so it renders
 * in tests without a router, matching routes/index.tsx.
 *
 * Query keys come from the shared queryOptions modules, never a literal here
 * (WEB-QUERY-001).
 */
export function AppHome() {
	const navigate = useNavigate();
	const repertoire = useQuery(repertoireQueryOptions);
	const catalog = useQuery(pieceCatalogQueryOptions);

	const now = new Date();

	function openPiece(pieceId: string) {
		navigate({ to: "/app/piece/$pieceId", params: { pieceId } });
	}

	function practise(pieceId: string | null) {
		navigate({ to: "/app/practice", search: { pieceId } });
	}

	return (
		<div className="mx-auto flex w-full max-w-2xl flex-col gap-8 bg-surface-page px-6 py-10">
			<section className="flex flex-col gap-3">
				<h1 className="font-display text-heading text-ink-primary">
					Your music
				</h1>
				{/* An empty repertoire and a failed fetch must not look alike
				    (WEB-ERR-001). */}
				{repertoire.isError && (
					<p className="text-body-sm text-danger">
						Couldn't load your pieces. Your practice history is still there --
						this is a connection problem.
					</p>
				)}
				{repertoire.data?.map((entry) => (
					<RepertoireCard
						key={entry.pieceId}
						entry={entry}
						now={now}
						onOpen={openPiece}
					/>
				))}
			</section>

			<section className="flex flex-col gap-3">
				<h2 className="font-display text-body text-ink-secondary">
					Add a piece
				</h2>
				{catalog.isError ? (
					<p className="text-body-sm text-danger">
						Couldn't load the score library.
					</p>
				) : (
					<AddPieceSearch
						catalog={catalog.data ?? []}
						onPick={(pieceId) => practise(pieceId)}
					/>
				)}
			</section>

			<button
				type="button"
				onClick={() => practise(null)}
				className="self-start rounded-full border border-border-subtle px-5 py-3 text-body-sm text-ink-secondary"
			>
				Just play
			</button>
		</div>
	);
}
```

`catalog.data ?? []` is a render-time default for a query that has not
resolved, not a swallowed error: the `catalog.isError` branch above renders
the failure explicitly.

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && bun run test src/routes/app.index.test.tsx && bun run typecheck
```
Expected: PASS, typecheck clean.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/routes/app.index.tsx apps/web/src/routes/app.index.test.tsx && git commit -m "feat(web): home renders repertoire from real session data"
```

---

## Task 17: Piece page

**Group:** G (parallel with Task 16; depends on Task 15)

**Behavior being verified:** A piece's page shows its score with every mark
that piece has accumulated across sessions, faded by lifecycle, alongside its
session history and open drills.
**Interface under test:** the `AppPiecePage` component

**Files:**
- Create: `apps/web/src/routes/app.piece.$pieceId.tsx`
- Test: `apps/web/src/routes/app.piece.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/routes/app.piece.test.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { PIECE_PAGE_SESSIONS } from "../lib/piece-page-fixtures";

const navigate = vi.fn();
vi.mock("@tanstack/react-router", async (importOriginal) => ({
	...(await importOriginal<typeof import("@tanstack/react-router")>()),
	useNavigate: () => navigate,
}));

const history = vi.fn();
vi.mock("../lib/api", () => ({
	api: { repertoire: { history: (id: string) => history(id) } },
}));

// The score is Verovio-rendered and network-bound. It is allowed to fail here:
// the marks this test asserts on live on the timeline canvas, which is
// independent of whether a score page ever rendered.
vi.mock("../lib/score-renderer", () => ({
	scoreRenderer: {
		load: vi.fn().mockResolvedValue("failed"),
		getPage: vi.fn(),
	},
}));

function wrap(ui: ReactNode) {
	const client = new QueryClient({
		defaultOptions: { queries: { retry: false } },
	});
	return render(
		<QueryClientProvider client={client}>{ui}</QueryClientProvider>,
	);
}

async function renderPiecePage() {
	const { AppPiecePage } = await import("./app.piece.$pieceId");
	return wrap(<AppPiecePage pieceId="chopin-nocturne-op9-no2" />);
}

describe("AppPiecePage", () => {
	it("shows every mark the piece has accumulated across both sessions, faded by lifecycle", async () => {
		history.mockResolvedValue({
			pieceId: "chopin-nocturne-op9-no2",
			composer: "Chopin",
			title: "Nocturne Op. 9 No. 2",
			sessions: [
				{
					sessionId: "s2",
					startedAt: "2026-08-18T18:00:00.000Z",
					endedAt: "2026-08-18T18:40:00.000Z",
				},
				{
					sessionId: "s1",
					startedAt: "2026-08-11T18:00:00.000Z",
					endedAt: null,
				},
			],
			openDrills: [],
		});

		await renderPiecePage();

		const expected = PIECE_PAGE_SESSIONS.flatMap((s) => s.marks).length;
		await waitFor(() =>
			expect(screen.getAllByTestId("mark-lifecycle-dot")).toHaveLength(expected),
		);

		// The fade itself: all three lifecycle states are on screen at once,
		// which is what "marks aging across sessions" means visually.
		const lifecycles = screen
			.getAllByRole("button")
			.map((b) => b.getAttribute("data-lifecycle"))
			.filter((v): v is string => v !== null);
		expect(new Set(lifecycles)).toEqual(
			new Set(["active", "improving", "resolved"]),
		);
	});

	it("shows the piece's session history", async () => {
		history.mockResolvedValue({
			pieceId: "chopin-nocturne-op9-no2",
			composer: "Chopin",
			title: "Nocturne Op. 9 No. 2",
			sessions: [
				{
					sessionId: "s1",
					startedAt: "2026-08-11T18:00:00.000Z",
					endedAt: "2026-08-11T18:30:00.000Z",
				},
			],
			openDrills: [],
		});

		await renderPiecePage();
		await waitFor(() =>
			expect(screen.getByText("11 Aug 2026")).toBeInTheDocument(),
		);
	});

	it("lists open drills for the piece", async () => {
		history.mockResolvedValue({
			pieceId: "chopin-nocturne-op9-no2",
			composer: "Chopin",
			title: "Nocturne Op. 9 No. 2",
			sessions: [],
			openDrills: [
				{
					id: "loop-1",
					barsStart: 5,
					barsEnd: 8,
					dimension: "pedaling",
					requiredCorrect: 5,
					attemptsCompleted: 2,
				},
			],
		});

		await renderPiecePage();
		await waitFor(() =>
			expect(screen.getByText(/bars 5-8/i)).toBeInTheDocument(),
		);
	});

	it("shows an explicit error when the piece cannot be loaded", async () => {
		history.mockRejectedValue(new Error("network down"));

		await renderPiecePage();
		await waitFor(() =>
			expect(screen.getByText(/couldn't load this piece/i)).toBeInTheDocument(),
		);
	});
});
```

- [ ] **Step 2: Run test -- verify it FAILS**

```bash
cd apps/web && bun run test src/routes/app.piece.test.tsx
```
Expected: FAIL -- `Failed to resolve import "./app.piece.$pieceId"`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```tsx
// apps/web/src/routes/app.piece.$pieceId.tsx
import { useQuery } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { PieceSessionHistory } from "../components/PieceSessionHistory";
import { ScoreStand } from "../components/ScoreStand";
import { SessionTimelineStrip } from "../components/SessionTimelineStrip";
import { pieceHistoryQueryOptions } from "../hooks/useRepertoire";
import {
	PIECE_PAGE_ALL_MARKS,
	PIECE_PAGE_MARKS_SPAN_SECONDS,
} from "../lib/piece-page-fixtures";

export const Route = createFileRoute("/app/piece/$pieceId")({
	component: RouteComponent,
});

function RouteComponent() {
	const { pieceId } = Route.useParams();
	return <AppPiecePage pieceId={pieceId} />;
}

interface AppPiecePageProps {
	pieceId: string;
}

/**
 * The piece page -- "the score across time" (docs/apps/05-ui-system.md#4).
 *
 * The marks are FIXTURE data. Marks have no persisted store yet: they exist
 * only as DO state and WebSocket events within a live session, so there is
 * nothing to query for "every mark this piece has accumulated". #160's
 * success criterion is worded for exactly this ("marks aging across >= 2
 * sessions of fixture data"); the mark-persistence follow-up replaces this
 * one import with a query and changes nothing else on this surface.
 *
 * Session history and open drills, by contrast, are real.
 */
export function AppPiecePage({ pieceId }: AppPiecePageProps) {
	const navigate = useNavigate();
	const history = useQuery(pieceHistoryQueryOptions(pieceId));

	if (history.isError) {
		return (
			<div className="mx-auto w-full max-w-2xl px-6 py-10">
				<p className="text-body-sm text-danger">
					Couldn't load this piece. Your practice history is still there --
					this is a connection problem.
				</p>
			</div>
		);
	}

	return (
		<div className="mx-auto flex w-full max-w-3xl flex-col gap-8 bg-surface-page px-6 py-10">
			<header className="flex flex-col gap-1">
				<h1 className="font-display text-heading text-ink-primary">
					{history.data?.title ?? ""}
				</h1>
				<span className="text-body-sm text-ink-secondary">
					{history.data?.composer ?? ""}
				</span>
			</header>

			{/*
			  * The score itself. It carries NO marks: Canvas A is lossy by
			  * design (docs/apps/05-ui-system.md) -- it can only place marks
			  * whose anchor is `bars` AND whose bars are on the rendered page,
			  * so it can never be the complete across-time view this page owes.
			  */}
			<section className="h-96 rounded-lg border border-border-subtle">
				<ScoreStand
					pieceId={pieceId}
					marks={[]}
					elapsedSeconds={0}
					isRecording={false}
				/>
			</section>

			{/*
			  * Canvas B, the total view: every mark this piece has accumulated,
			  * across every session, faded by lifecycle. Elapsed time is the one
			  * coordinate every anchor carries, which is what makes this canvas
			  * able to show marks the score overlay cannot place.
			  *
			  * durationSeconds spans the whole accumulated history rather than
			  * one session, so a mark late in a long session is not clamped on
			  * top of an early one.
			  */}
			<section className="flex flex-col gap-2">
				<h2 className="font-display text-body text-ink-secondary">
					Marks across sessions
				</h2>
				<SessionTimelineStrip
					durationSeconds={PIECE_PAGE_MARKS_SPAN_SECONDS}
					marks={PIECE_PAGE_ALL_MARKS}
				/>
			</section>

			<button
				type="button"
				onClick={() => navigate({ to: "/app/practice", search: { pieceId } })}
				className="self-start rounded-full bg-accent px-5 py-3 text-body-sm text-on-accent"
			>
				Practice this
			</button>

			<section className="flex flex-col gap-2">
				<h2 className="font-display text-body text-ink-secondary">Sessions</h2>
				<PieceSessionHistory sessions={history.data?.sessions ?? []} />
			</section>

			<section className="flex flex-col gap-2">
				<h2 className="font-display text-body text-ink-secondary">
					Open drills
				</h2>
				{history.data?.openDrills.length ? (
					<ul className="flex flex-col gap-1">
						{history.data.openDrills.map((drill) => (
							<li
								key={drill.id}
								className="flex items-baseline justify-between border-b border-border-subtle py-2 text-body-sm text-ink-primary"
							>
								<span>
									bars {drill.barsStart}-{drill.barsEnd}
								</span>
								<span className="text-body-xs text-ink-tertiary">
									{drill.attemptsCompleted} / {drill.requiredCorrect}
								</span>
							</li>
						))}
					</ul>
				) : (
					<p className="text-body-sm text-ink-tertiary">No open drills</p>
				)}
			</section>
		</div>
	);
}
```

- [ ] **Step 4: Run test -- verify it PASSES**

```bash
cd apps/web && bun run test src/routes/
```
Expected: PASS. (Vitest runs with the `tanstackStart()` plugin disabled and
imports the route module directly, so this step does not depend on
`routeTree.gen.ts`.)

- [ ] **Step 5: Regenerate the route tree**

`createFileRoute("/app/piece/$pieceId")` type-checks only against paths
TanStack Router has already generated into `apps/web/src/routeTree.gen.ts`.
That file is produced by the `tanstackStart()` Vite plugin
(`apps/web/vite.config.ts`), not by a standalone CLI -- this repo has no
`tsr`/router-cli binary. Running the app's own build invokes that plugin and
regenerates the file as a build side effect (verified empirically in this
worktree: adding a throwaway route and running this command updated
`routeTree.gen.ts` to include it; see Task 15 Step 5).

```bash
cd apps/web && bun run build
```
Expected: build succeeds; `git diff apps/web/src/routeTree.gen.ts` shows the
new `/app/piece/$pieceId` route added.

- [ ] **Step 6: Commit the route file with the regenerated route tree**

```bash
git add apps/web/src/routes/app.piece.\$pieceId.tsx apps/web/src/routes/app.piece.test.tsx apps/web/src/routeTree.gen.ts && git commit -m "feat(web): piece page shows marks aging across sessions with history and drills"
```

- [ ] **Step 7: Typecheck**

```bash
cd apps/web && bun run typecheck
```
Expected: typecheck clean (this depends on the regenerated
`routeTree.gen.ts` from Step 5 -- do not run this before Step 5/6).

---

## Final verification (run after every group is merged)

```bash
cd apps/api && bun run test && bun run typecheck && bun run lint
cd ../web && bun run test && bun run typecheck && bun run lint
cd ../web && bun run test:a11y
```

`bun run test:a11y` is the **[WEB-A11Y-004]** gate (enforced). It is the only
place `color-contrast` is checkable -- jsdom's axe skips the rule entirely, so
the suites above make no contrast claim. If it reports a violation on a new
surface, the fix is to use an already-shipped token pairing, not to add an
exemption.

### Manual click-through (human-lit; discharges #158's deferred step)

Cannot be automated: needs a microphone, a live Durable Object, and the
Transkun path.

1. Start local Postgres, `just api`, `just web`.
2. Sign in, land on `/app`.
3. Search the library, pick a piece -> lands on `/app/practice?pieceId=...`.
4. Start recording. Confirm the **score stand renders that piece's score**
   (not `PieceLessMode`) -- the `user-picked` ladder rung, reachable from a
   live session for the first time.
5. Pause >= 20s. Confirm a mark lands on the canvas.
6. Stop. Return to `/app`. Confirm the piece now appears as a repertoire card
   with its session count incremented.
7. Open the card. Confirm the piece page renders marks, session history, and
   any open drills.

Record the outcome as a `STATE:` comment on #160.

---

## Challenge Review

### CEO Pass

**Premise Challenge.** The three gaps in the Problem section are each verified
directly against the tree: `apps/web/src/routes/app.tsx` today hardcodes
`userPickedPieceId={null}` / `confidentGuess={null}` (confirmed by reading the
file), `apps/web/src/hooks/usePracticeSession.ts:312-320` `console.log`s the
`piece_identified` event and discards `data.pieceId` (confirmed), and
`apps/api/src/db/schema/sessions.ts` has no `pieceId` column today (confirmed).
This is real, load-bearing pain, not a proxy problem: without it `/app` cannot
show a repertoire from real data and the ladder's top two rungs are dead code.
No simpler framing was found that gets the same outcome with less surface.

**Scope Check.** The plan matches the spec closely; no drift found. The
hardest problem — making the write path (`sessions.piece_id`) and the ladder
wiring (`confidentGuess`) real, not just adding UI over fixtures — is
attacked directly in Tasks 4, 7, 9, 14, not deferred. The plan explicitly
defers mark persistence and bar-anchored marks to a follow-up, which is the
correct cut: those require alignment work out of scope here. 17 tasks / ~20
files is large but each file is one deep module or one thin leaf; the
File Structure table's own depth ratings hold up under inspection (verified
directly, see Architecture below).

**Twelve-Month Alignment.**
```
CURRENT STATE                     THIS PLAN                        12-MONTH IDEAL
/app is one button; every        sessions carry piece_id;         Every session is piece-aware;
session is pieceless; ladder's   home + piece page render real    marks persist and the piece
top two rungs unreachable;       data; ladder's user-picked and   page shows a true cross-session
marks are DO-only, ephemeral.    confirm-chip rungs reachable.    view without fixtures.
```
Moves toward the ideal without foreclosing it: `openMarkCount: number | null`
and the fixture-sourced piece-page marks are both named exit ramps for the
mark-persistence follow-up (one service line, one import swap), not
one-way doors.

**Alternatives Check.** The spec does not enumerate rejected alternatives
(e.g., storing piece identity on `conversations` instead of `sessions`, or
computing repertoire client-side from existing session-list endpoints). Given
`piece_id` already exists on `segment_loops`/`diagnosis_artifacts`/
`pending_exercises` (confirmed in `apps/api/src/db/schema/`), putting it on
`sessions` too is the obviously consistent choice and likely didn't need
alternatives spelled out, but the spec should say so explicitly for a future
reader.

### Engineering Pass

**Architecture.** Traced end to end and it holds up against the real code:

```
POST /api/practice/start {pieceId?}
   -> validate against pieces (db.query.pieces.findFirst)  [404 if absent]
   -> insert sessions {..., pieceId}
GET /api/repertoire
   -> listRepertoire: join sessions+pieces where pieceId is not null
   -> buildRepertoire (pure): group, count, latest, sort desc
GET /api/repertoire/:pieceId
   -> listPieceHistory: pieces lookup [404 NotFoundError] + sessions + listOpenLoopsForPiece
/app (layout, Outlet) -> /app/ (Home, useQuery(repertoireQueryOptions))
                       -> /app/practice?pieceId= (AppPractice, reads search param)
                       -> /app/piece/$pieceId (AppPiecePage, useQuery(pieceHistoryQueryOptions))
```
All schema/column names used (`sessions.pieceId`, `segmentLoops.barsStart`,
`barsEnd`, `dimension`, `requiredCorrect`, `attemptsCompleted`, `status`,
`createdAt`, `pieces.pieceId/composer/title`) were checked against the actual
Drizzle schemas and match exactly. `errorHandler` maps `NotFoundError` -> 404
and `HTTPException` -> its own response, confirmed in
`apps/api/src/middleware/error-handler.ts`; `requireAuth` throws
`HTTPException(401, ...)`, confirmed in `apps/api/src/middleware/auth-session.ts`.
No SQL/shell/prompt injection surface: all identifiers are Drizzle-typed
columns, `pieceId` is validated against the catalog before it is ever stored
or interpolated into a query.

**Module Depth Audit.**
- `buildRepertoire`/`listRepertoire`/`listPieceHistory`
  (`apps/api/src/services/repertoire.ts`): interface = 3 functions, ~10 fields
  of types; hides a join, a grouping pass, a null-policy, and drill
  aggregation. DEEP, confirmed.
- `searchPieces` (`apps/web/src/lib/piece-search.ts`): 1 function; hides
  normalization, dual-field matching, prefix/substring ranking, capping.
  DEEP, confirmed.
- `describeArc` (`apps/web/src/lib/learning-arc.ts`): 1 function; hides every
  relative-date branch. DEEP, confirmed.
- `listOpenLoopsForPiece` (`apps/api/src/services/segment-loops.ts`): thin
  wrapper over one `select().where().orderBy()` — closer to SHALLOW than the
  other three, but justified: it is the one Drizzle-chain boundary the test
  suite's stub technique can reach, and it is reused by `listPieceHistory`.
  Acceptable.
- `RepertoireCard`/`AddPieceSearch`/`PieceSessionHistory`: SHALLOW by design,
  as the plan itself states, and correctly so — logic is extracted into the
  deep modules above.

**Code Quality.** No `catch(e)` / broad-exception patterns introduced. No
`any`. `[WEB-ERR-001]` is honored: `api.repertoire.*`/`api.scores.list` never
catch-and-return `[]`; `AppHome`/`AppPiecePage` render `isError` explicitly.
One drift from `[WEB-STATE-002]`'s spirit worth flagging under Presumption
Inventory below (`AppHome`'s `now = new Date()` computed fresh every render,
not memoized — acceptable per the plan's own "no measured perf need" rule,
but every render creates a new `Date`, which slightly undercuts
`RepertoireCard`'s `now` prop being presented as an injected/testable clock;
harmless in practice).

**Test Philosophy.** Every test in the plan exercises a public
interface (`buildRepertoire(rows)`, `searchPieces(pieces, query)`, HTTP routes
via `.request()`, rendered components via Testing Library queries). Route
tests mock the *service module* (`vi.spyOn(repertoireService, ...)`), which is
this codebase's established, already-shipped pattern (confirmed: identical to
`exercises.test.ts`'s approach per the spec) — an external-enough boundary for
a route test, not an internal collaborator of the unit under test. No shape-
only tests found; every assertion checks behavior (ordering, filtering,
fallback strings, 404 mapping).

**Vertical Slice Audit.** All 17 tasks follow one-test -> one-implementation
-> one-commit. No task writes multiple tests before any implementation, no
task defers implementation to a later task. Clean.

**Test Coverage Gaps.**
```
[+] apps/api/src/services/repertoire.ts
    ├── buildRepertoire()
    │   ├── [TESTED] ★★★ empty input, grouping, most-recent-wins-any-order, ordering, openMarkCount null — Task 6
    │   └── [GAP]    two pieces with an identical lastPracticedAt (tie-break undefined — see Presumption Inventory)
    ├── listRepertoire()
    │   └── [GAP]    no test at all (only buildRepertoire and listPieceHistory are tested; the join+filter half of listRepertoire has zero coverage — see BLOCKER below)
    └── listPieceHistory()
        ├── [TESTED] ★★  happy path (sessions + drills) — Task 7
        └── [TESTED] ★★  NotFoundError for missing piece — Task 7

[+] apps/api/src/routes/practice.ts (/start)
    ├── [TESTED] ★★ stores pieceId — Task 9
    ├── [TESTED] ★★ 404s an unknown pieceId, does not store — Task 9
    └── [TESTED] ★★ still starts a pieceless session — Task 9

[+] apps/web/src/routes/app.practice.tsx / app.piece.$pieceId.tsx
    └── [GAP] neither route's `Route.validateSearch` / `Route.useParams` wiring is exercised by a route-level test — both test files render the exported plain component (`AppPractice`, `AppPiecePage`) directly, never the `Route` object itself, so a mistake in `validateSearch`'s search-param plumbing (e.g. wrong key) would not be caught by these tests. Low severity: the same gap exists for every route in this codebase already (`routes/index.tsx` pattern), so this is consistent with, not worse than, existing coverage.
```

**Failure Modes.**
- `POST /api/practice/start`: if the `pieces.findFirst` lookup throws (DB
  blip), the whole request 500s before the `sessions` insert — visible via
  `errorHandler`'s catch-all `console.error`, not silent. Acceptable.
- `GET /api/repertoire`/`GET /api/repertoire/:pieceId`: failure surfaces as a
  thrown error -> `errorHandler` -> JSON error response; the web client's
  `request()` throws `ApiError`, and `useQuery`'s `isError` renders an
  explicit message. No silent empty-list path — this was checked directly and
  the design note in the spec (`apps/CLAUDE.md`'s "no client-side fallbacks
  hiding failures") is honored.
- Migration (Task 4): applying `bun run migrate` against local Postgres is a
  single DDL statement (`ADD COLUMN` + index); if it fails partway the task's
  own verification step (`psql \d sessions`) would catch it before commit.
  No partial-state risk beyond what Postgres's own DDL transaction already
  covers.

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `createFileRoute("/app/practice")` and `createFileRoute("/app/piece/$pieceId")` will typecheck against the current `FileRoutesByPath` types without any regeneration step | **RISKY** | Verified: `createFileRoute<TFilePath extends keyof FileRoutesByPath>` (`@tanstack/react-router/dist/esm/fileRoute.d.ts`) constrains the path argument to keys already present in the generated `apps/web/src/routeTree.gen.ts`, which currently has no `/app/practice` or `/app/piece/$pieceId` entries. The router-plugin (`tanstackStart()`) that regenerates this file is explicitly disabled during `vitest` (`!isTest && tanstackStart()` in `vite.config.ts`) and is not invoked by bare `tsc` (`bun run typecheck`). `routeTree.gen.ts` is a tracked, committed file (confirmed via `git log`), not a build artifact regenerated transparently by `tsc`. See BLOCKER below. |
| `db.query.pieces.findFirst` is available on the `Db` type the way `db.query.conversations.findFirst` already is | SAFE | `Db = PostgresJsDatabase<typeof schema>` and `schema` re-exports `pieces` from `catalog.ts`; Drizzle's relational query builder exposes every schema-exported table under `db.query.<exportName>`. No existing call site uses `db.query.pieces` yet, but the mechanism is identical to the already-used `db.query.conversations`. |
| `Object.keys(sessions)` includes `pieceId` and `sessions.pieceId.notNull` is a real boolean property | SAFE | Verified against `drizzle-orm/column.d.ts` (`notNull: boolean` on the column's public type) and an identical existing pattern in `segment-loops.test.ts`. |
| GET `/api/scores` returns `{ pieces: [...] }` with `pieceId`/`composer`/`title` and needs no auth | SAFE | Confirmed by reading `apps/api/src/routes/scores.ts` and `services/scores.ts` directly. |
| `PracticeMode` accepts `confidentGuess: ConfidentGuess | null` (non-optional) and `userPickedPieceId: string | null` | SAFE | Confirmed against `apps/web/src/components/PracticeMode.tsx`'s actual prop interface. |
| The `piece_identified` WS event already carries `pieceId`, `composer`, `title`, `confidence` exactly as `ConfidentGuess` needs | SAFE | Confirmed in `apps/web/src/lib/practice-api.ts`'s event union and `apps/api/src/do/session-brain.ts`'s emit site — fields match exactly. |
| `resolvePieceLadderState` already owns guess-vs-pick precedence, so the hook needn't duplicate it | SAFE | Confirmed: `resolvePieceLadderState` in `piece-ladder.ts` is a pure 3-line precedence function; Task 14 does not reimplement it. |
| Two pieces practiced in the same session-collapsing pass with an identical `lastPracticedAt` millisecond sort deterministically | VALIDATE | `buildRepertoire`'s `Array.prototype.sort` on `String#localeCompare` of equal ISO strings is stable in modern JS engines (spec-guaranteed since ES2019), so ties fall back to Map insertion order — deterministic but untested and not documented as intentional. Low real-world likelihood (sub-ms same-piece ties across *different* pieces), not worth blocking on. |
| `bun run generate` (Task 4) will pick up the schema edit alone and not require a preceding `drizzle-kit` config check | VALIDATE | Not independently re-verified against the actual `drizzle.config.ts` in this pass; the task's own step 4 verification (`psql \d sessions`) catches a failure before commit regardless, so risk is contained to wasted time, not a silent bad state. |

### Summary
[BLOCKER] count: 1
[RISK]    count: 2
[QUESTION] count: 1

**[BLOCKER]** (confidence: 8/10) — Task 15 (`app.practice.tsx`, path
`/app/practice`) and Task 17 (`app.piece.$pieceId.tsx`, path
`/app/piece/$pieceId`) both call `createFileRoute("<new path>")` against paths
that do not exist yet in the checked-in, type-generating
`apps/web/src/routeTree.gen.ts`. `createFileRoute`'s type signature
(`TFilePath extends keyof FileRoutesByPath`) will reject an unrecognized path
literal at compile time. The router-plugin that regenerates this file
(`tanstackStart()` in `vite.config.ts`) is disabled under `vitest`
(`!isTest && tanstackStart()`) and is never invoked by `bun run typecheck`
(bare `tsc`), so neither Task 15's nor Task 17's own "Step 4: run test --
verify it PASSES, ... && bun run typecheck" will produce a green typecheck
without an explicit, currently-missing step to regenerate and commit
`routeTree.gen.ts` (e.g. running `bun run dev`/`bun run build` once, or the
project's `tsr generate` equivalent, before the typecheck gate). **Fix:** add
an explicit sub-step to Task 15 and Task 17 — after creating each new route
file, run the router codegen (confirm the exact command from
`apps/web/package.json`/`vite.config.ts`; none of the existing scripts name
it explicitly, which itself should be resolved) and commit the regenerated
`routeTree.gen.ts` alongside the route file, before running `bun run
typecheck`.

**[RISK]** (confidence: 5/10) — `listRepertoire` (the join+filter half of
`apps/api/src/services/repertoire.ts`) has no test of its own; only its pure
downstream (`buildRepertoire`) and its sibling (`listPieceHistory`) are
tested. A mistake in the join condition or the `isNotNull(sessions.pieceId)`
filter (e.g. an accidental `leftJoin` admitting piece-less rows, or a typo'd
column) would not be caught by any test in this plan. Watch during Task 7's
review; fallback is acceptable given the codebase's stated inability to stub
a `select().from().innerJoin()` chain without a live Postgres — but this
should at minimum be called out as an accepted gap, not left implicit.

**[RISK]** (confidence: 4/10) — `apps/web/src/routes/app.practice.tsx` and
`app.piece.$pieceId.tsx`'s test files render the exported plain component
directly and never exercise the `Route` object's `validateSearch`/`useParams`
wiring, so a bug in the search-param or route-param plumbing itself would
ship untested. Consistent with the rest of the codebase's existing route-test
pattern, so not a regression, but worth a manual click-through check
specifically for `?pieceId=` surviving the round trip (the plan's manual
verification step 3 does cover this at a high level).

**[QUESTION]** — No alternative to "add `piece_id` on `sessions`" is recorded
in the spec (e.g., deriving repertoire from `segment_loops`/
`diagnosis_artifacts`, which already carry `pieceId`, instead of adding a new
column). The chosen design is very likely correct — sessions are the only
table with one row per practice session and per-piece timestamps — but the
spec should say why those other tables were rejected for a future reader,
per the CEO-pass Alternatives Check above.

### Blocker resolution (commit 55ff3f90)

The single BLOCKER above was addressed in the plan body before build: Task 15
and Task 17 each gained an explicit **Step 5: Regenerate the route tree**
(`cd apps/web && bun run build`, which invokes the `tanstackStart()` Vite
plugin), a **Step 6** that commits the regenerated `routeTree.gen.ts` alongside
the route file, and a **Step 7** that runs `bun run typecheck` only after that
regeneration. The two RISKs are accepted and carried into build as documented
gaps: `listRepertoire` has no direct test (no way to stub an `innerJoin` chain
without live Postgres), and neither new route's `validateSearch`/`useParams`
wiring is exercised by a route-level test (consistent with every existing route
in this codebase; covered by the manual click-through instead).

VERDICT: PROCEED_WITH_CAUTION — blocker discharged; the two RISKs above are
accepted, documented gaps, not blockers.
