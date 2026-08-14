# Session Review Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent
> per task). Do NOT start execution until `/challenge` returns
> `VERDICT: PROCEED`.

**Goal:** On stop, the practice canvas transitions in place into a review
state: one qualitative verdict, every session mark consolidated, per-mark
own-audio playback with a tracking cursor, evidence in words, a "work on
this" drill entry, and exactly one carry-forward — with loud failure states.
**Spec:** `docs/specs/2026-08-13-session-review-design.md`
**Style:** Follow `apps/CLAUDE.md` and the existing `apps/web/src` component/
hook conventions (tabs, Tailwind utility classes, `aria-label`/`data-testid`
patterns already established by `ScoreMarkLayer.tsx`/`PlayPassageCard.tsx`).
Use `bun`, never `npm`/`yarn`/`pnpm`. All commands below run from
`apps/web/` unless stated otherwise.

## No separate Task Group 0

The spec's Verification Architecture is per-task TDD (each module's own
vitest file) plus a harness extension to `/practice-preview`. Unlike a
parser or scoring function, there is no golden-fixture file that could
precede all feature work here — the harness *composes* the components this
plan builds (`SessionReview`, `PracticeMode`'s review phase), so it cannot
exist before them. It is therefore the **last** task group (F), not Group 0,
mirroring how #158 built `/practice-preview` alongside its components rather
than ahead of them.

## Task Groups

```
Group A (parallel, no interdependencies): Tasks 1, 2, 3, 4, 5
Group B (parallel, depends on A):          Tasks 6, 7, 8, 9, 10
Group C (depends on B):                    Task 11
Group D (depends on B, C):                 Task 12
Group E (depends on D):                    Task 13
Group F (sequential, depends on E, A3):    Task 14, then Task 15
```

**Decouple check:**
- Group A: `[SHIPS INDEPENDENTLY]` in the sense that each task is a real,
  reviewable unit (pure functions, additive props), but none is
  user-visible alone.
- Groups A-E together are what makes `SessionReview` exist and pass its own
  tests; nothing is user-reachable until Group F wires the
  `/practice-preview` demo, since `AppChat.tsx` wiring to a live session is
  explicitly out of scope (spec, "Not in scope"). The whole plan ships as
  one unit for that reason — there is no earlier point a student could see
  it.

---

## Task 1: `formatWsGap` discloses a connection gap in words

**Group:** A (parallel with 2, 3, 4, 5)

**Behavior being verified:** Given a `WsGap`, `formatWsGap` returns the
disclosure sentence design specifies ("N minutes didn't reach me"),
singular/plural correct.
**Interface under test:** `formatWsGap(gap: WsGap): string`

**Files:**
- Create: `apps/web/src/lib/session-review.ts`
- Test: `apps/web/src/lib/session-review.test.ts`

- [x] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/session-review.test.ts
import { describe, expect, it } from "vitest";
import { formatWsGap } from "./session-review";

describe("formatWsGap", () => {
	it("discloses a multi-minute gap in plural and a one-minute gap in singular", () => {
		expect(
			formatWsGap({ startSeconds: 0, endSeconds: 240 }),
		).toBe("4 minutes didn't reach me");
		expect(
			formatWsGap({ startSeconds: 100, endSeconds: 160 }),
		).toBe("1 minute didn't reach me");
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/lib/session-review.test.ts
```
Expected: FAIL — `Cannot find module './session-review'` (the file does not exist yet).

- [x] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/web/src/lib/session-review.ts

/** Closed union so "loud, never blank" is a compile-time property: an
 * unhandled variant is a type error in VerdictBanner's switch, not a
 * silently-missing render. */
export type VerdictState =
	| { status: "loading" }
	| { status: "ready"; text: string }
	| { status: "failed" };

/** A single object, not an array -- "exactly one carry-forward" is enforced
 * by the shape, the same way mark.ts's branded MarkAnchor enforces "no wrong
 * bar numbers" structurally rather than with a runtime guard. */
export interface CarryForward {
	readonly text: string;
}

export interface WsGap {
	readonly startSeconds: number;
	readonly endSeconds: number;
}

export interface NoCommentZone {
	readonly atSeconds: number;
	readonly reason: string;
}

/** The one place a WS gap becomes words. */
export function formatWsGap(gap: WsGap): string {
	const minutes = Math.round((gap.endSeconds - gap.startSeconds) / 60);
	const unit = minutes === 1 ? "minute" : "minutes";
	return `${minutes} ${unit} didn't reach me`;
}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/lib/session-review.test.ts
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add apps/web/src/lib/session-review.ts apps/web/src/lib/session-review.test.ts
git commit -m "feat(web): add session-review verdict/carry-forward/disclosure types"
```

---

## Task 2: `createPassageQstampSource` bridges real-audio playback time to score qstamp

**Group:** A (parallel with 1, 3, 4, 5)

**Behavior being verified:** Given a bar timeline (bar -> playback seconds)
and a clip's `BarIR[]` (bar -> qstamp range), the bridge interpolates the
qstamp for a given playback second, and degrades to `null` (never a guess)
when playback hasn't started or the bar is unknown.
**Interface under test:** `createPassageQstampSource(barTimeline, bars, getTSec): () => number | null`

**Files:**
- Create: `apps/web/src/lib/passage-cursor.ts`
- Test: `apps/web/src/lib/passage-cursor.test.ts`

- [x] **Step 1: Write the failing test**

```typescript
// apps/web/src/lib/passage-cursor.test.ts
import { describe, expect, it } from "vitest";
import { createPassageQstampSource } from "./passage-cursor";
import type { BarIR } from "./score-ir";

const BAR_TIMELINE = [
	{ bar: 5, tSec: 10 },
	{ bar: 6, tSec: 12 },
];

const BARS: BarIR[] = [
	{
		barNumber: 5,
		measureOn: "m5",
		pageN: 1,
		bbox: { x: 0, y: 0, w: 0, h: 0 },
		noteIds: [],
		qstampStart: 16,
		qstampEnd: 20,
	},
];

describe("createPassageQstampSource", () => {
	it("interpolates qstamp within the bracketing bar, and degrades to null outside known playback", () => {
		const midway = createPassageQstampSource(BAR_TIMELINE, BARS, () => 11);
		expect(midway()).toBe(18); // halfway through tSec [10,12] -> halfway through qstamp [16,20]

		const notStarted = createPassageQstampSource(BAR_TIMELINE, BARS, () => null);
		expect(notStarted()).toBeNull();

		const pastKnownRange = createPassageQstampSource(BAR_TIMELINE, BARS, () => 99);
		expect(pastKnownRange()).toBeNull();
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/lib/passage-cursor.test.ts
```
Expected: FAIL — `Cannot find module './passage-cursor'`

- [x] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/web/src/lib/passage-cursor.ts
import type { BarIR } from "./score-ir";

export interface BarTimelineEntry {
	readonly bar: number;
	readonly tSec: number;
}

/**
 * Bridges PassagePlayer's real-time playback clock (tSec) to ScoreCursor's
 * qstamp clock. Nothing else in the codebase does this: useLoopPlayer's
 * qstampSource comes from a synthesized ClipNote clock with an exact qstamp,
 * not real recorded audio's coarse per-bar barTimeline.
 *
 * Returns null -- never a guessed position -- when playback hasn't started,
 * is past the last known bar boundary, or the resolved bar has no entry in
 * `bars` (a clip IR that doesn't cover the bar the manifest names). This
 * mirrors mark.ts's resolveAnchor: the system falls silent rather than
 * drawing a cursor it isn't sure about.
 */
export function createPassageQstampSource(
	barTimeline: readonly BarTimelineEntry[],
	bars: readonly BarIR[],
	getTSec: () => number | null,
): () => number | null {
	return () => {
		const tSec = getTSec();
		if (tSec === null) return null;

		let lower: BarTimelineEntry | undefined;
		let upper: BarTimelineEntry | undefined;
		for (let i = 0; i < barTimeline.length - 1; i++) {
			const a = barTimeline[i];
			const b = barTimeline[i + 1];
			if (tSec >= a.tSec && tSec <= b.tSec) {
				lower = a;
				upper = b;
				break;
			}
		}
		if (!lower || !upper) return null;

		const span = upper.tSec - lower.tSec;
		const fraction = span > 0 ? (tSec - lower.tSec) / span : 0;

		const barIR = bars.find((b) => b.barNumber === lower.bar);
		if (!barIR) return null;

		return (
			barIR.qstampStart + fraction * (barIR.qstampEnd - barIR.qstampStart)
		);
	};
}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/lib/passage-cursor.test.ts
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add apps/web/src/lib/passage-cursor.ts apps/web/src/lib/passage-cursor.test.ts
git commit -m "feat(web): bridge passage playback time to score qstamp for cursor tracking"
```

---

## Task 3: `usePracticeSession` exposes the live `sessionId`

**Group:** A (parallel with 1, 2, 4, 5)

**Behavior being verified:** After `start()` resolves, the hook's returned
`sessionId` matches the id the server assigned — needed by any surface (like
the future review) that must fetch a passage scoped to this session.
**Interface under test:** `UsePracticeSessionReturn.sessionId`

**Files:**
- Modify: `apps/web/src/hooks/usePracticeSession.ts`
- Test: `apps/web/src/hooks/usePracticeSession.sessionid.test.ts` (new file,
  mirrors the existing `usePracticeSession.marks.test.ts` fixture exactly)

- [x] **Step 1: Write the failing test**

```typescript
// apps/web/src/hooks/usePracticeSession.sessionid.test.ts
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

describe("usePracticeSession sessionId", () => {
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

	it("exposes the server-assigned sessionId once the session starts", async () => {
		const { result } = renderHook(() => usePracticeSession());

		expect(result.current.sessionId).toBeNull();

		await act(async () => {
			await result.current.start();
		});
		await waitFor(() => expect(result.current.state).toBe("recording"));

		expect(result.current.sessionId).toBe("s1");
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/hooks/usePracticeSession.sessionid.test.ts
```
Expected: FAIL — `Property 'sessionId' does not exist on type 'UsePracticeSessionReturn'` (typecheck) or `result.current.sessionId` is `undefined` at runtime.

- [x] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/hooks/usePracticeSession.ts`, add `sessionId` to the return
interface, right after `conversationId`:

```typescript
export interface UsePracticeSessionReturn {
	state: PracticeState;
	elapsedSeconds: number;
	observations: ObservationEvent[];
	marks: Mark[];
	latestScores: DimScores | null;
	summary: string | null;
	error: string | null;
	chunksProcessed: number;
	chunkStates: ChunkState[];
	wsStatus: WsStatus;
	isOnline: boolean;
	isPlaying: boolean;
	energy: number;
	analyserNode: AnalyserNode | null;
	practiceMode: PracticeMode | null;
	start: (conversationId?: string) => Promise<void>;
	stop: () => void;
	setPiece: (query: string) => void;
	observationMessages: RichMessage[];
	conversationId: string | null;
	sessionId: string | null;
	activeLoop: SegmentLoopConfig | null;
}
```

And in the return object at the bottom of `usePracticeSession`, right after
`conversationId: conversationIdRef.current,`:

```typescript
	return {
		state,
		elapsedSeconds,
		observations,
		marks,
		latestScores,
		summary,
		error,
		chunksProcessed,
		chunkStates,
		wsStatus,
		isOnline,
		isPlaying,
		energy,
		analyserNode,
		practiceMode,
		start,
		stop,
		setPiece,
		observationMessages,
		conversationId: conversationIdRef.current,
		sessionId: sessionIdRef.current,
		activeLoop,
	};
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/hooks/usePracticeSession.sessionid.test.ts
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add apps/web/src/hooks/usePracticeSession.ts apps/web/src/hooks/usePracticeSession.sessionid.test.ts
git commit -m "feat(web): expose the live sessionId from usePracticeSession"
```

---

## Task 4: `ScoreMarkLayer` accepts a `renderDetail` override

**Group:** A (parallel with 1, 2, 3, 5)

**Behavior being verified:** When a `renderDetail` prop is supplied,
expanding a mark renders whatever it returns instead of the default
`MarkDetail`; when omitted, existing behavior (default `MarkDetail`) is
unchanged.
**Interface under test:** `ScoreMarkLayerProps.renderDetail`

**Files:**
- Modify: `apps/web/src/components/ScoreMarkLayer.tsx`
- Modify: `apps/web/src/components/ScoreMarkLayer.test.tsx`

- [x] **Step 1: Write the failing test**

Add to `apps/web/src/components/ScoreMarkLayer.test.tsx` (change the
existing `renderLayer` helper to accept an overrides param, and add a new
`it`):

```typescript
import { fireEvent, render, screen } from "@testing-library/react";
import { type ComponentProps, createRef } from "react";
import { describe, expect, it } from "vitest";
import { FIXTURE_BARS, FIXTURE_MARKS } from "../test-utils/mark-fixtures";
import { ScoreMarkLayer } from "./ScoreMarkLayer";

function renderLayer(
	overrides: Partial<ComponentProps<typeof ScoreMarkLayer>> = {},
) {
	const ref = createRef<HTMLDivElement>();
	// Bar 88's element is deliberately absent: it models a bar on a page the
	// overlay is not currently showing.
	const onPage = FIXTURE_BARS.filter((b) => b.barNumber !== 88);
	return render(
		<div ref={ref}>
			{onPage.map((b) => (
				<div key={b.measureOn} id={b.measureOn} />
			))}
			<ScoreMarkLayer
				containerRef={ref}
				bars={FIXTURE_BARS}
				marks={FIXTURE_MARKS}
				{...overrides}
			/>
		</div>,
	);
}
```

...and, in the existing `describe("ScoreMarkLayer", ...)` block, add:

```typescript
	it("uses renderDetail instead of MarkDetail when provided", () => {
		renderLayer({
			renderDetail: (mark, onClose) => (
				<button
					type="button"
					data-testid={`custom-detail-${mark.id}`}
					onClick={onClose}
				>
					custom
				</button>
			),
		});

		fireEvent.click(screen.getByLabelText(/Needs work: Pedaling, bars 5-6/));
		expect(screen.getByTestId("custom-detail-m1")).toBeInTheDocument();
	});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/components/ScoreMarkLayer.test.tsx
```
Expected: FAIL — `Property 'renderDetail' does not exist on type 'IntrinsicAttributes & ScoreMarkLayerProps'` (typecheck), or the custom testid is not found at runtime.

- [x] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/components/ScoreMarkLayer.tsx`:

```typescript
import { type ReactNode, type RefObject, useEffect, useState } from "react";
import type { Mark } from "../lib/mark";
import type { BarLocator, MeasureRect } from "../lib/mark-placement";
import { placeMarks } from "../lib/mark-placement";
import { MarkDetail } from "./MarkDetail";
import { MarkGlyph } from "./MarkGlyph";

interface ScoreMarkLayerProps {
	containerRef: RefObject<HTMLElement | null>;
	bars: readonly BarLocator[];
	marks: readonly Mark[];
	/** Overrides the default evidence-only MarkDetail expansion. SessionReview
	 * uses this to swap in own-audio playback + drill entry without forking
	 * placement/measurement logic. */
	renderDetail?: (mark: Mark, onClose: () => void) => ReactNode;
}
```

And in the render, replace the existing `{expandedId === mark.id && (...)}`
block with:

```tsx
					{expandedId === mark.id &&
						(renderDetail ? (
							renderDetail(mark, () => setExpandedId(null))
						) : (
							<MarkDetail mark={mark} onClose={() => setExpandedId(null)} />
						))}
```

(Add `renderDetail` to the destructured props at the top of the function
alongside `containerRef, bars, marks`.)

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/components/ScoreMarkLayer.test.tsx
```
Expected: PASS (both the existing test and the new one)

- [x] **Step 5: Commit**

```bash
git add apps/web/src/components/ScoreMarkLayer.tsx apps/web/src/components/ScoreMarkLayer.test.tsx
git commit -m "feat(web): let ScoreMarkLayer take a custom mark-detail renderer"
```

---

## Task 5: `SessionTimelineStrip` accepts a `renderDetail` override

**Group:** A (parallel with 1, 2, 3, 4)

**Behavior being verified:** Same contract as Task 4, for the timeline
canvas.
**Interface under test:** `SessionTimelineStripProps.renderDetail`

**Files:**
- Modify: `apps/web/src/components/SessionTimelineStrip.tsx`
- Modify: `apps/web/src/components/SessionTimelineStrip.test.tsx`

- [x] **Step 1: Write the failing test**

Add to `apps/web/src/components/SessionTimelineStrip.test.tsx`:

```typescript
	it("uses renderDetail instead of MarkDetail when provided", () => {
		render(
			<SessionTimelineStrip
				durationSeconds={FIXTURE_DURATION_SECONDS}
				marks={FIXTURE_MARKS}
				renderDetail={(mark, onClose) => (
					<button
						type="button"
						data-testid={`custom-detail-${mark.id}`}
						onClick={onClose}
					>
						custom
					</button>
				)}
			/>,
		);

		fireEvent.click(screen.getByLabelText(/Needs work: Pedaling, bars 5-6/));
		expect(screen.getByTestId("custom-detail-m1")).toBeInTheDocument();
	});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/components/SessionTimelineStrip.test.tsx
```
Expected: FAIL — `renderDetail` is not a valid prop (typecheck), or the
custom testid is not found at runtime.

- [x] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/components/SessionTimelineStrip.tsx`, add the import and
prop:

```typescript
import { type ReactNode, useCallback, useLayoutEffect, useRef, useState } from "react";
import type { Mark } from "../lib/mark";
import type { LaneItem } from "../lib/timeline-lanes";
import { assignLanes, clampToStrip } from "../lib/timeline-lanes";
import { MarkDetail } from "./MarkDetail";
import { MarkGlyph } from "./MarkGlyph";
```

```typescript
interface SessionTimelineStripProps {
	durationSeconds: number;
	marks: readonly Mark[];
	renderDetail?: (mark: Mark, onClose: () => void) => ReactNode;
}
```

Add `renderDetail` to the destructured function params, and replace the
existing:

```tsx
					{expandedId === mark.id && (
						<MarkDetail mark={mark} onClose={() => setExpandedId(null)} />
					)}
```

with:

```tsx
					{expandedId === mark.id &&
						(renderDetail ? (
							renderDetail(mark, () => setExpandedId(null))
						) : (
							<MarkDetail mark={mark} onClose={() => setExpandedId(null)} />
						))}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/components/SessionTimelineStrip.test.tsx
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add apps/web/src/components/SessionTimelineStrip.tsx apps/web/src/components/SessionTimelineStrip.test.tsx
git commit -m "feat(web): let SessionTimelineStrip take a custom mark-detail renderer"
```

---

## Task 6: `MarkPassagePlayback` — cursor-tracked own-audio playback for a mark

**Group:** B (parallel with 7, 8, 9, 10; depends on Task 2)

**Behavior being verified:** Given a bar-anchored mark and a session id, the
component fetches the passage manifest + clip, renders a Play control, and
degrades honestly: a timestamp-only mark (no bars) shows "no passage" without
fetching; a fetch failure shows an error state; an audio-decode failure shows
"Audio unavailable" without crashing.
**Interface under test:** `MarkPassagePlayback({ mark, sessionId, ... })`

**Files:**
- Create: `apps/web/src/components/MarkPassagePlayback.tsx`
- Test: `apps/web/src/components/MarkPassagePlayback.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/MarkPassagePlayback.test.tsx
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import type { PassageManifest } from "../lib/types";

const mockGetPassage = vi.fn();
const mockGetClipPlayback = vi.fn();
const mockPlay = vi.fn();
const mockLoad = vi.fn();
const mockDestroy = vi.fn();

vi.mock("../lib/api", () => ({
	api: {
		sessions: {
			getPassage: (...args: unknown[]) => mockGetPassage(...args),
		},
	},
}));

vi.mock("../lib/score-renderer", () => ({
	scoreRenderer: {
		getClipPlayback: (...args: unknown[]) => mockGetClipPlayback(...args),
	},
}));

vi.mock("../lib/passage-player", () => ({
	PassagePlayer: function MockPassagePlayer() {
		return {
			state: "ready",
			duration: 13,
			load: mockLoad,
			play: mockPlay,
			pause: vi.fn(),
			onTick: () => () => undefined,
			destroy: mockDestroy,
		};
	},
}));

vi.mock("../lib/score-cursor", () => ({
	ScoreCursor: function MockScoreCursor() {
		return { start: vi.fn(), stop: vi.fn() };
	},
}));

beforeEach(() => {
	vi.clearAllMocks();
	mockLoad.mockResolvedValue(undefined);
	globalThis.AudioContext = vi.fn(function MockAudioContext() {
		return { close: vi.fn().mockResolvedValue(undefined) };
	}) as unknown as typeof AudioContext;
});

afterEach(() => {
	cleanup();
});

const barMark: Mark = {
	id: "m1",
	anchor: resolveAnchor({ atSeconds: 20, bars: [5, 6], alignmentQuality: 1 }),
	taxonomy: "needs_work",
	dimension: "pedaling",
	evidence: "test evidence",
	lifecycle: "active",
};

const timestampMark: Mark = {
	id: "m2",
	anchor: resolveAnchor({ atSeconds: 20, alignmentQuality: 0 }),
	taxonomy: "needs_work",
	dimension: "pedaling",
	evidence: "test evidence",
	lifecycle: "active",
};

describe("MarkPassagePlayback", () => {
	it("shows no passage for a timestamp-only mark without fetching", async () => {
		render(<MarkPassagePlayback mark={timestampMark} sessionId="s1" />);

		expect(
			screen.getByText(/No passage audio for this mark/),
		).toBeInTheDocument();
		expect(mockGetPassage).not.toHaveBeenCalled();
	});

	it("fetches and renders a Play control for a bar-anchored mark", async () => {
		const manifest: PassageManifest = {
			source: { kind: "session", sessionId: "s1" },
			pieceId: "chopin-nocturne-op9-no2",
			bars: [5, 6],
			chunks: [],
			startOffsetSec: 0,
			endOffsetSec: 4,
			barTimeline: [
				{ bar: 5, tSec: 0 },
				{ bar: 6, tSec: 4 },
			],
		};
		mockGetPassage.mockResolvedValue(manifest);
		mockGetClipPlayback.mockResolvedValue({
			svg: "<svg data-test='clip'></svg>",
			ir: {
				pieceId: manifest.pieceId,
				verovioVersion: "test",
				pageWidth: 100,
				pages: [],
				bars: [],
				notes: {},
			},
			notes: [],
		});

		render(<MarkPassagePlayback mark={barMark} sessionId="s1" />);

		await waitFor(() =>
			expect(
				screen.getByRole("button", { name: /play passage/i }),
			).toBeInTheDocument(),
		);
		expect(mockGetPassage).toHaveBeenCalledWith("s1", [5, 6]);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/components/MarkPassagePlayback.test.tsx
```
Expected: FAIL — `Cannot find module './MarkPassagePlayback'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```tsx
// apps/web/src/components/MarkPassagePlayback.tsx
import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { DIMENSION_COLOR_VAR } from "../lib/dimension-colors";
import type { Mark } from "../lib/mark";
import { createPassageQstampSource } from "../lib/passage-cursor";
import { PassagePlayer } from "../lib/passage-player";
import { ScoreCursor } from "../lib/score-cursor";
import type { ScoreIR } from "../lib/score-ir";
import { scoreRenderer } from "../lib/score-renderer";
import type { PassageManifest } from "../lib/types";
import { ClipSvg } from "./ClipSvg";

interface MarkPassagePlaybackProps {
	mark: Mark;
	sessionId: string;
	_mockManifest?: PassageManifest;
	_mockClip?: string;
	_mockIr?: ScoreIR;
	_playable?: boolean;
}

/** "no_bars" is distinct from "error": a timestamp-only mark's anchor has no
 * bars to fetch a passage for at all -- that's an honest "unavailable," not
 * a failure, matching mark.ts's silent-degradation precedent. */
type LoadState = "no_bars" | "loading" | "ready" | "audio_error" | "error";

/**
 * The promoted play_passage machinery (was chat-only, `PlayPassageCard.tsx`):
 * own-audio playback of exactly a mark's bars, with a ScoreCursor tracking
 * playback position via passage-cursor.ts's tSec -> qstamp bridge.
 */
export function MarkPassagePlayback({
	mark,
	sessionId,
	_mockManifest,
	_mockClip,
	_mockIr,
	_playable,
}: MarkPassagePlaybackProps) {
	const bars: [number, number] | null =
		mark.anchor.type === "bars"
			? [mark.anchor.bars[0], mark.anchor.bars[1]]
			: null;

	const [loadState, setLoadState] = useState<LoadState>(
		bars ? "loading" : "no_bars",
	);
	const [clipSvg, setClipSvg] = useState<string | null>(null);
	const [manifest, setManifest] = useState<PassageManifest | null>(null);
	const [clipIr, setClipIr] = useState<ScoreIR | null>(null);
	const playerRef = useRef<PassagePlayer | null>(null);
	const ctxRef = useRef<AudioContext | null>(null);
	const containerRef = useRef<HTMLDivElement>(null);
	const currentTSecRef = useRef<number | null>(null);

	useEffect(() => {
		if (!bars) return;

		if (_mockManifest && _mockClip && _mockIr && !_playable) {
			setManifest(_mockManifest);
			setClipSvg(_mockClip);
			setClipIr(_mockIr);
			setLoadState("ready");
			return;
		}

		let cancelled = false;
		(async () => {
			let m: PassageManifest;
			let svg: string;
			let ir: ScoreIR;
			try {
				m = _mockManifest ?? (await api.sessions.getPassage(sessionId, bars));
				if (!_mockManifest && cancelled) return;
				const clip = await scoreRenderer.getClipPlayback(
					m.pieceId,
					bars[0],
					bars[1],
				);
				svg = _mockClip ?? clip.svg;
				ir = _mockIr ?? clip.ir;
				if (cancelled) return;
			} catch (err) {
				console.error("MarkPassagePlayback fetch failed", err);
				if (!cancelled) setLoadState("error");
				return;
			}
			setManifest(m);
			setClipSvg(svg);
			setClipIr(ir);

			try {
				const ctx = new AudioContext();
				ctxRef.current = ctx;
				const player = new PassagePlayer(m, ctx);
				await player.load();
				if (cancelled) {
					player.destroy();
					ctxRef.current = null;
					return;
				}
				player.onTick((tSec) => {
					currentTSecRef.current = tSec;
				});
				playerRef.current = player;
				setLoadState("ready");
			} catch (err) {
				console.error("MarkPassagePlayback audio load failed", err);
				if (!cancelled) setLoadState("audio_error");
			}
		})();
		return () => {
			cancelled = true;
			playerRef.current?.destroy();
			ctxRef.current?.close();
			ctxRef.current = null;
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [sessionId, bars?.[0], bars?.[1], _mockManifest, _mockClip, _mockIr, _playable]);

	useEffect(() => {
		if (!manifest || !clipIr || !containerRef.current) return;
		const qstampSource = createPassageQstampSource(
			manifest.barTimeline,
			clipIr.bars,
			() => currentTSecRef.current,
		);
		const cursor = new ScoreCursor({
			pieceId: manifest.pieceId,
			container: containerRef.current,
			ir: clipIr,
			qstampSource,
		});
		cursor.start();
		return () => cursor.stop();
	}, [manifest, clipIr]);

	if (loadState === "no_bars") {
		return (
			<p className="text-body-sm text-ink-tertiary">
				No passage audio for this mark.
			</p>
		);
	}

	const color = DIMENSION_COLOR_VAR[mark.dimension];

	return (
		<div>
			{loadState === "loading" && (
				<div className="h-10 flex items-center justify-center">
					<div className="w-3.5 h-3.5 rounded-full border-2 border-ink-tertiary/50 border-t-transparent animate-spin" />
				</div>
			)}
			{(loadState === "ready" || loadState === "audio_error") && clipSvg && (
				<div className="px-3 pt-3">
					<div
						ref={containerRef}
						style={{
							position: "relative",
							borderRadius: "6px",
							border: `1.5px solid ${color}40`,
							backgroundColor: "var(--color-score-canvas)",
							overflow: "hidden",
						}}
					>
						<ClipSvg svg={clipSvg} />
					</div>
					{loadState === "ready" ? (
						<button
							type="button"
							aria-label="Play passage"
							onClick={() => void playerRef.current?.play()}
							className="mt-3 px-3 py-1.5 rounded-md border border-border-subtle text-body-sm text-ink-primary hover:bg-surface-raised transition-colors"
						>
							Play
						</button>
					) : (
						<span className="mt-3 inline-block text-body-sm text-ink-tertiary">
							Audio unavailable
						</span>
					)}
				</div>
			)}
			{loadState === "error" && (
				<p className="p-4 text-body-sm text-ink-tertiary">
					couldn't load audio
				</p>
			)}
		</div>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/components/MarkPassagePlayback.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/MarkPassagePlayback.tsx apps/web/src/components/MarkPassagePlayback.test.tsx
git commit -m "feat(web): add cursor-tracked own-audio playback for a mark's passage"
```

---

## Task 7: `VerdictBanner` renders loading/ready/failed with a retry affordance

**Group:** B (parallel with 6, 8, 9, 10; depends on Task 1)

**Behavior being verified:** Each `VerdictState` variant renders distinct,
non-blank content; only the `failed` variant renders a retry button, and
clicking it calls `onRetry`.
**Interface under test:** `VerdictBanner({ verdict, onRetry })`

**Files:**
- Create: `apps/web/src/components/VerdictBanner.tsx`
- Test: `apps/web/src/components/VerdictBanner.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/VerdictBanner.test.tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { VerdictBanner } from "./VerdictBanner";

describe("VerdictBanner", () => {
	it("renders loading copy, the verdict sentence when ready, and an explicit retry state when failed", () => {
		const { rerender } = render(
			<VerdictBanner verdict={{ status: "loading" }} onRetry={() => {}} />,
		);
		expect(
			screen.getByText(/Putting together your review/),
		).toBeInTheDocument();

		rerender(
			<VerdictBanner
				verdict={{ status: "ready", text: "the bass-blur is gone" }}
				onRetry={() => {}}
			/>,
		);
		expect(screen.getByText("the bass-blur is gone")).toBeInTheDocument();

		const onRetry = vi.fn();
		rerender(
			<VerdictBanner verdict={{ status: "failed" }} onRetry={onRetry} />,
		);
		expect(screen.getByRole("alert")).toBeInTheDocument();
		fireEvent.click(screen.getByRole("button", { name: /try again/i }));
		expect(onRetry).toHaveBeenCalledTimes(1);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/components/VerdictBanner.test.tsx
```
Expected: FAIL — `Cannot find module './VerdictBanner'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```tsx
// apps/web/src/components/VerdictBanner.tsx
import type { VerdictState } from "../lib/session-review";

interface VerdictBannerProps {
	verdict: VerdictState;
	onRetry: () => void;
}

/**
 * "Loud, never blank": VerdictState is a closed union, so an unhandled
 * variant here is a TypeScript error, not a silently-missing render.
 */
export function VerdictBanner({ verdict, onRetry }: VerdictBannerProps) {
	switch (verdict.status) {
		case "loading":
			return (
				<p className="text-body-md text-ink-secondary" role="status">
					Putting together your review...
				</p>
			);
		case "ready":
			return <p className="text-body-md text-ink-primary">{verdict.text}</p>;
		case "failed":
			return (
				<div
					role="alert"
					className="flex items-center justify-between gap-3 rounded-md border border-danger/40 bg-danger/10 px-3 py-2"
				>
					<p className="text-body-sm text-danger">
						Couldn't put together a review this time.
					</p>
					<button
						type="button"
						onClick={onRetry}
						className="shrink-0 rounded-full border border-border-subtle px-3 py-1 text-body-sm text-ink-primary"
					>
						Try again
					</button>
				</div>
			);
	}
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/components/VerdictBanner.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/VerdictBanner.tsx apps/web/src/components/VerdictBanner.test.tsx
git commit -m "feat(web): add VerdictBanner with an explicit synthesis-failure retry state"
```

---

## Task 8: `SessionDisclosures` renders WS gaps and no-comment zones, or nothing

**Group:** B (parallel with 6, 7, 9, 10; depends on Task 1)

**Behavior being verified:** Given gaps/zones, each renders as a disclosed
line; given neither, the component renders nothing (disclosure is additive,
never implying a problem a clean session didn't have).
**Interface under test:** `SessionDisclosures({ wsGaps, noCommentZones })`

**Files:**
- Create: `apps/web/src/components/SessionDisclosures.tsx`
- Test: `apps/web/src/components/SessionDisclosures.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/SessionDisclosures.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { SessionDisclosures } from "./SessionDisclosures";

describe("SessionDisclosures", () => {
	it("renders nothing for a clean session, and discloses gaps and no-comment zones when present", () => {
		const clean = render(
			<SessionDisclosures wsGaps={[]} noCommentZones={[]} />,
		);
		expect(clean.container).toBeEmptyDOMElement();
		clean.unmount();

		render(
			<SessionDisclosures
				wsGaps={[{ startSeconds: 0, endSeconds: 240 }]}
				noCommentZones={[
					{ atSeconds: 58, reason: "transcription confidence was too low here" },
				]}
			/>,
		);
		expect(screen.getByText("4 minutes didn't reach me")).toBeInTheDocument();
		expect(
			screen.getByText(/No comment near 0:58 — transcription confidence/),
		).toBeInTheDocument();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/components/SessionDisclosures.test.tsx
```
Expected: FAIL — `Cannot find module './SessionDisclosures'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```tsx
// apps/web/src/components/SessionDisclosures.tsx
import { formatElapsed } from "../lib/mark";
import type { NoCommentZone, WsGap } from "../lib/session-review";
import { formatWsGap } from "../lib/session-review";

interface SessionDisclosuresProps {
	wsGaps: readonly WsGap[];
	noCommentZones: readonly NoCommentZone[];
}

/**
 * Renders nothing when both arrays are empty: disclosure is additive, never
 * implying a problem a clean session didn't have.
 */
export function SessionDisclosures({
	wsGaps,
	noCommentZones,
}: SessionDisclosuresProps) {
	if (wsGaps.length === 0 && noCommentZones.length === 0) return null;

	return (
		<ul
			aria-label="Session disclosures"
			className="flex flex-col gap-1 text-body-xs text-ink-tertiary"
		>
			{wsGaps.map((gap) => (
				<li key={`${gap.startSeconds}-${gap.endSeconds}`}>
					{formatWsGap(gap)}
				</li>
			))}
			{noCommentZones.map((zone) => (
				<li key={zone.atSeconds}>
					No comment near {formatElapsed(zone.atSeconds)} — {zone.reason}
				</li>
			))}
		</ul>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/components/SessionDisclosures.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/SessionDisclosures.tsx apps/web/src/components/SessionDisclosures.test.tsx
git commit -m "feat(web): disclose WS gaps and no-comment zones in session review"
```

---

## Task 9: `ScoreStand` passes `renderDetail` through to its mark layer

**Group:** B (parallel with 6, 7, 8, 10; depends on Task 4)

**Behavior being verified:** A `renderDetail` prop on `ScoreStand` reaches
`ScoreMarkLayer` and is used to expand a mark.
**Interface under test:** `ScoreStandProps.renderDetail`

**Files:**
- Modify: `apps/web/src/components/ScoreStand.tsx`
- Modify: `apps/web/src/components/ScoreStand.test.tsx`

- [ ] **Step 1: Write the failing test**

Add to `apps/web/src/components/ScoreStand.test.tsx` (add `Mark`/
`resolveAnchor` imports at the top alongside the existing ones):

```typescript
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import { scoreRenderer } from "../lib/score-renderer";
import { ScoreStand } from "./ScoreStand";
```

...and, inside `describe("ScoreStand", ...)`:

```typescript
	it("renders a custom mark detail via renderDetail when a mark is expanded", async () => {
		vi.mocked(scoreRenderer.getPage).mockImplementation(async (_pieceId, pageN) =>
			pageN === 1 ? '<svg><g id="m1"></g></svg>' : "<svg></svg>",
		);

		const mark: Mark = {
			id: "m1",
			anchor: resolveAnchor({ atSeconds: 5, bars: [1, 1], alignmentQuality: 1 }),
			taxonomy: "needs_work",
			dimension: "pedaling",
			evidence: "test evidence",
			lifecycle: "active",
		};

		render(
			<ScoreStand
				pieceId="chopin-nocturne-op9-no2"
				marks={[mark]}
				elapsedSeconds={0}
				isRecording={false}
				renderDetail={(m, onClose) => (
					<button
						type="button"
						data-testid={`custom-${m.id}`}
						onClick={onClose}
					>
						custom
					</button>
				)}
			/>,
		);

		await waitFor(() =>
			expect(
				screen.getByLabelText(/Needs work: Pedaling, bar 1/),
			).toBeInTheDocument(),
		);
		fireEvent.click(screen.getByLabelText(/Needs work: Pedaling, bar 1/));
		expect(screen.getByTestId("custom-m1")).toBeInTheDocument();
	});
```

(Uses the file's existing `TWO_PAGE_IR`/`beforeEach` mocks — `TWO_PAGE_IR`
already has `barNumber: 1, measureOn: "m1", pageN: 1`.)

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/components/ScoreStand.test.tsx
```
Expected: FAIL — `renderDetail` is not a valid prop on `ScoreStand`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/components/ScoreStand.tsx`, add the import and prop:

```typescript
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { useMetronome } from "../hooks/useMetronome";
import type { Mark } from "../lib/mark";
import { formatElapsed } from "../lib/mark";
import type { BarIR } from "../lib/score-ir";
import { scoreRenderer } from "../lib/score-renderer";
import { ScoreMarkLayer } from "./ScoreMarkLayer";
```

```typescript
interface ScoreStandProps {
	pieceId: string;
	marks: readonly Mark[];
	elapsedSeconds: number;
	isRecording: boolean;
	renderDetail?: (mark: Mark, onClose: () => void) => ReactNode;
}
```

Add `renderDetail` to the destructured function params, and forward it on
the existing `<ScoreMarkLayer ... />` call:

```tsx
					<ScoreMarkLayer
						containerRef={containerRef}
						bars={barsForCurrentPage}
						marks={marks}
						renderDetail={renderDetail}
					/>
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/components/ScoreStand.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ScoreStand.tsx apps/web/src/components/ScoreStand.test.tsx
git commit -m "feat(web): forward renderDetail from ScoreStand to its mark layer"
```

---

## Task 10: `PieceLessMode` passes `renderDetail` through to its timeline

**Group:** B (parallel with 6, 7, 8, 9; depends on Task 5)

**Behavior being verified:** A `renderDetail` prop on `PieceLessMode` reaches
`SessionTimelineStrip` and is used to expand a mark.
**Interface under test:** `PieceLessModeProps.renderDetail`

**Files:**
- Modify: `apps/web/src/components/PieceLessMode.tsx`
- Modify: `apps/web/src/components/PieceLessMode.test.tsx`

- [ ] **Step 1: Write the failing test**

Add to `apps/web/src/components/PieceLessMode.test.tsx`:

```typescript
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import { PieceLessMode } from "./PieceLessMode";

const marks: Mark[] = [
	{
		id: "m1",
		anchor: resolveAnchor({ atSeconds: 30, alignmentQuality: 0 }),
		taxonomy: "needs_work",
		dimension: "pedaling",
		evidence: "test evidence",
		lifecycle: "active",
	},
];

describe("PieceLessMode", () => {
	// ...existing test unchanged...

	it("uses renderDetail instead of MarkDetail when provided", () => {
		render(
			<PieceLessMode
				marks={marks}
				durationSeconds={90}
				elapsedSeconds={65}
				isRecording={true}
				renderDetail={(mark, onClose) => (
					<button
						type="button"
						data-testid={`custom-${mark.id}`}
						onClick={onClose}
					>
						custom
					</button>
				)}
			/>,
		);

		fireEvent.click(screen.getByLabelText(/Needs work: Pedaling, 0:30/));
		expect(screen.getByTestId("custom-m1")).toBeInTheDocument();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/components/PieceLessMode.test.tsx
```
Expected: FAIL — `renderDetail` is not a valid prop on `PieceLessMode`.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/components/PieceLessMode.tsx`:

```typescript
import type { ReactNode } from "react";
import { useMetronome } from "../hooks/useMetronome";
import type { Mark } from "../lib/mark";
import { SessionTimelineStrip } from "./SessionTimelineStrip";

interface PieceLessModeProps {
	marks: readonly Mark[];
	durationSeconds: number;
	elapsedSeconds: number;
	isRecording: boolean;
	renderDetail?: (mark: Mark, onClose: () => void) => ReactNode;
}
```

Add `renderDetail` to the destructured function params, and forward it:

```tsx
			<div className="w-full max-w-2xl">
				<SessionTimelineStrip
					durationSeconds={durationSeconds}
					marks={marks}
					renderDetail={renderDetail}
				/>
			</div>
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/components/PieceLessMode.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/PieceLessMode.tsx apps/web/src/components/PieceLessMode.test.tsx
git commit -m "feat(web): forward renderDetail from PieceLessMode to its timeline"
```

---

## Task 11: `MarkReviewDetail` — evidence + playback + "work on this"

**Group:** C (depends on Task 6)

**Behavior being verified:** Renders the mark's evidence text and a passage
playback region, and firing "Work on this" calls `onWorkOnThis` with the
mark; "Close" calls `onClose`.
**Interface under test:** `MarkReviewDetail({ mark, sessionId, onClose, onWorkOnThis })`

**Files:**
- Create: `apps/web/src/components/MarkReviewDetail.tsx`
- Test: `apps/web/src/components/MarkReviewDetail.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/web/src/components/MarkReviewDetail.test.tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";

vi.mock("./MarkPassagePlayback", () => ({
	MarkPassagePlayback: () => <div data-testid="playback-stub" />,
}));

const mark: Mark = {
	id: "m1",
	anchor: resolveAnchor({ atSeconds: 20, bars: [5, 6], alignmentQuality: 1 }),
	taxonomy: "needs_work",
	dimension: "pedaling",
	evidence: "pedal held through the bass change",
	lifecycle: "active",
};

describe("MarkReviewDetail", () => {
	it("shows evidence and the playback region, and fires onWorkOnThis with the mark", async () => {
		const { MarkReviewDetail } = await import("./MarkReviewDetail");
		const onWorkOnThis = vi.fn();
		const onClose = vi.fn();

		render(
			<MarkReviewDetail
				mark={mark}
				sessionId="s1"
				onClose={onClose}
				onWorkOnThis={onWorkOnThis}
			/>,
		);

		expect(
			screen.getByText("pedal held through the bass change"),
		).toBeInTheDocument();
		expect(screen.getByTestId("playback-stub")).toBeInTheDocument();

		fireEvent.click(screen.getByRole("button", { name: /work on this/i }));
		expect(onWorkOnThis).toHaveBeenCalledWith(mark);

		fireEvent.click(screen.getByRole("button", { name: /close/i }));
		expect(onClose).toHaveBeenCalledTimes(1);
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/components/MarkReviewDetail.test.tsx
```
Expected: FAIL — `Cannot find module './MarkReviewDetail'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```tsx
// apps/web/src/components/MarkReviewDetail.tsx
import type { Mark } from "../lib/mark";
import { anchorLabel } from "../lib/mark";
import { MarkPassagePlayback } from "./MarkPassagePlayback";

interface MarkReviewDetailProps {
	mark: Mark;
	sessionId: string;
	onClose: () => void;
	onWorkOnThis: (mark: Mark) => void;
}

/**
 * The review-only expansion: evidence + the student's own recorded audio for
 * this mark's bars (cursor-tracked) + a drill entry point. Wired into
 * ScoreMarkLayer/SessionTimelineStrip via their `renderDetail` prop, so live
 * pause-taps keep today's evidence-only MarkDetail unchanged.
 */
export function MarkReviewDetail({
	mark,
	sessionId,
	onClose,
	onWorkOnThis,
}: MarkReviewDetailProps) {
	const framing = mark.confidence === "exploratory" ? "Early read — " : "";

	return (
		<section
			className="mt-1 rounded-md border border-border-subtle bg-surface-raised p-3"
			aria-label={`Evidence, ${anchorLabel(mark.anchor)}`}
		>
			<p className="text-body-sm text-ink-secondary">
				{framing}
				{mark.evidence}
			</p>
			<MarkPassagePlayback mark={mark} sessionId={sessionId} />
			<div className="mt-3 flex items-center gap-3">
				<button
					type="button"
					onClick={() => onWorkOnThis(mark)}
					className="rounded-full border border-border-subtle px-3 py-1.5 text-body-sm text-ink-primary"
				>
					Work on this
				</button>
				<button
					type="button"
					onClick={onClose}
					className="text-label-sm text-ink-tertiary underline"
				>
					Close
				</button>
			</div>
		</section>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/components/MarkReviewDetail.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/MarkReviewDetail.tsx apps/web/src/components/MarkReviewDetail.test.tsx
git commit -m "feat(web): add MarkReviewDetail (evidence + playback + work-on-this)"
```

---

## Task 12: `SessionReview` composes verdict, canvas, disclosures, and carry-forward

**Group:** D (depends on Tasks 7, 8, 9, 10, 11, 1)

**Behavior being verified:** Given a piece and props, `SessionReview` renders
the score canvas (with `renderDetail` wired to `MarkReviewDetail`), the
verdict, and the carry-forward; given no piece, it renders the timeline
canvas instead; given no carry-forward, that line is omitted. This is one
cohesive composition contract (which canvas, what chrome), tested and
implemented together the same way `ScoreMarkLayer.test.tsx`'s single test
already covers its whole placement branch.
**Interface under test:** `SessionReview({ pieceId, sessionId, marks, ... })`

**Files:**
- Create: `apps/web/src/components/SessionReview.tsx`
- Test: `apps/web/src/components/SessionReview.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// apps/web/src/components/SessionReview.test.tsx
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import { SessionReview } from "./SessionReview";

// vi.mock calls are hoisted above these imports by vitest's transform, so
// SessionReview resolves ScoreStand/PieceLessMode/MarkPassagePlayback to
// the mocks below regardless of write order.
vi.mock("./ScoreStand", () => ({
	ScoreStand: (props: {
		pieceId: string;
		marks: readonly Mark[];
		renderDetail?: (mark: Mark, onClose: () => void) => ReactNode;
	}) => (
		<div data-testid="score-stand" data-piece-id={props.pieceId}>
			{props.renderDetail?.(props.marks[0], () => {})}
		</div>
	),
}));
vi.mock("./PieceLessMode", () => ({
	PieceLessMode: (props: {
		marks: readonly Mark[];
		renderDetail?: (mark: Mark, onClose: () => void) => ReactNode;
	}) => (
		<div data-testid="pieceless-mode">
			{props.renderDetail?.(props.marks[0], () => {})}
		</div>
	),
}));
vi.mock("./MarkPassagePlayback", () => ({
	MarkPassagePlayback: () => <div data-testid="playback-stub" />,
}));

const mark: Mark = {
	id: "m1",
	anchor: resolveAnchor({ atSeconds: 10, bars: [1, 1], alignmentQuality: 1 }),
	taxonomy: "needs_work",
	dimension: "pedaling",
	evidence: "test evidence",
	lifecycle: "active",
};

describe("SessionReview", () => {
	it("renders the score canvas, verdict, and carry-forward when there is a piece", () => {
		render(
			<SessionReview
				pieceId="chopin-nocturne-op9-no2"
				sessionId="s1"
				marks={[mark]}
				durationSeconds={120}
				verdict={{ status: "ready", text: "solid session" }}
				onRetryVerdict={() => {}}
				carryForward={{ text: "start with bars 40-44" }}
				wsGaps={[]}
				noCommentZones={[]}
				onWorkOnThis={() => {}}
			/>,
		);

		expect(screen.getByTestId("score-stand")).toHaveAttribute(
			"data-piece-id",
			"chopin-nocturne-op9-no2",
		);
		expect(screen.getByText("solid session")).toBeInTheDocument();
		expect(screen.getByText(/start with bars 40-44/)).toBeInTheDocument();
		expect(screen.getByText("test evidence")).toBeInTheDocument();
	});

	it("falls back to the pieceless timeline and omits carry-forward when there is none", () => {
		render(
			<SessionReview
				pieceId={null}
				sessionId="s1"
				marks={[mark]}
				durationSeconds={120}
				verdict={{ status: "loading" }}
				onRetryVerdict={() => {}}
				carryForward={null}
				wsGaps={[]}
				noCommentZones={[]}
				onWorkOnThis={() => {}}
			/>,
		);

		expect(screen.getByTestId("pieceless-mode")).toBeInTheDocument();
		expect(screen.queryByText(/Next:/)).not.toBeInTheDocument();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/components/SessionReview.test.tsx
```
Expected: FAIL — `Cannot find module './SessionReview'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```tsx
// apps/web/src/components/SessionReview.tsx
import type { Mark } from "../lib/mark";
import type {
	CarryForward,
	NoCommentZone,
	VerdictState,
	WsGap,
} from "../lib/session-review";
import { MarkReviewDetail } from "./MarkReviewDetail";
import { PieceLessMode } from "./PieceLessMode";
import { ScoreStand } from "./ScoreStand";
import { SessionDisclosures } from "./SessionDisclosures";
import { VerdictBanner } from "./VerdictBanner";

interface SessionReviewProps {
	pieceId: string | null;
	sessionId: string;
	marks: readonly Mark[];
	durationSeconds: number;
	verdict: VerdictState;
	onRetryVerdict: () => void;
	carryForward: CarryForward | null;
	wsGaps: readonly WsGap[];
	noCommentZones: readonly NoCommentZone[];
	onWorkOnThis: (mark: Mark) => void;
}

/**
 * The review surface (docs/apps/05-ui-system.md#3). Reuses the exact canvas
 * #158 built for live practice -- ScoreStand/PieceLessMode already take
 * `marks: readonly Mark[]` with no live/consolidated distinction, so "marks
 * consolidated" is just "pass every mark." The only thing review changes
 * about the canvas is the expansion: MarkReviewDetail replaces MarkDetail
 * via the `renderDetail` prop both canvases now accept (Tasks 9, 10).
 */
export function SessionReview({
	pieceId,
	sessionId,
	marks,
	durationSeconds,
	verdict,
	onRetryVerdict,
	carryForward,
	wsGaps,
	noCommentZones,
	onWorkOnThis,
}: SessionReviewProps) {
	const renderDetail = (mark: Mark, onClose: () => void) => (
		<MarkReviewDetail
			mark={mark}
			sessionId={sessionId}
			onClose={onClose}
			onWorkOnThis={onWorkOnThis}
		/>
	);

	return (
		<div className="flex h-full flex-col gap-3 overflow-auto px-4 py-3">
			<VerdictBanner verdict={verdict} onRetry={onRetryVerdict} />
			<div className="min-h-0 flex-1">
				{pieceId ? (
					<ScoreStand
						pieceId={pieceId}
						marks={marks}
						elapsedSeconds={durationSeconds}
						isRecording={false}
						renderDetail={renderDetail}
					/>
				) : (
					<PieceLessMode
						marks={marks}
						durationSeconds={durationSeconds}
						elapsedSeconds={durationSeconds}
						isRecording={false}
						renderDetail={renderDetail}
					/>
				)}
			</div>
			<SessionDisclosures wsGaps={wsGaps} noCommentZones={noCommentZones} />
			{carryForward && (
				<p className="border-t border-border-subtle pt-3 text-body-sm text-ink-primary">
					Next: {carryForward.text}
				</p>
			)}
		</div>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/components/SessionReview.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/SessionReview.tsx apps/web/src/components/SessionReview.test.tsx
git commit -m "feat(web): add SessionReview composing verdict, marks canvas, and carry-forward"
```

---

## Task 13: `PracticeMode` transitions in place to a review phase

**Group:** E (depends on Task 12)

**Behavior being verified:** With `phase="review"`, `PracticeMode` renders
`SessionReview` instead of the live ladder/canvas branch, and the header
shows a "Done" button that calls `onExitReview` instead of "Stop". Existing
callers that never pass `phase` are unaffected (default `"practice"`).
**Interface under test:** `PracticeModeProps.phase`, `.onExitReview`, and the
new review-data props.

**Files:**
- Modify: `apps/web/src/components/PracticeMode.tsx`
- Modify: `apps/web/src/components/PracticeMode.test.tsx`

- [ ] **Step 1: Write the failing test**

Add to the top of `apps/web/src/components/PracticeMode.test.tsx` (alongside
the existing `vi.mock("../lib/score-renderer", ...)`):

```typescript
vi.mock("./SessionReview", () => ({
	SessionReview: (props: { pieceId: string | null }) => (
		<div
			data-testid="session-review-stub"
			data-piece-id={props.pieceId ?? "none"}
		/>
	),
}));
```

...and, inside `describe("PracticeMode", ...)`:

```typescript
	it("renders SessionReview and a Done button in review phase", () => {
		const onExitReview = vi.fn();
		render(
			<PracticeMode
				userPickedPieceId={null}
				confidentGuess={null}
				marks={[]}
				elapsedSeconds={90}
				isPlaying={false}
				isRecording={false}
				onStop={() => {}}
				phase="review"
				onExitReview={onExitReview}
			/>,
		);

		expect(screen.getByTestId("session-review-stub")).toBeInTheDocument();
		expect(
			screen.queryByRole("button", { name: /stop recording/i }),
		).not.toBeInTheDocument();

		fireEvent.click(screen.getByRole("button", { name: /done/i }));
		expect(onExitReview).toHaveBeenCalledTimes(1);
	});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/components/PracticeMode.test.tsx
```
Expected: FAIL — `phase` is not a valid prop on `PracticeMode` (typecheck),
or `session-review-stub` is never found because `PracticeMode` still renders
the live-canvas branch.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/components/PracticeMode.tsx`:

```tsx
import { useState } from "react";
import { usePauseTracker } from "../hooks/usePauseTracker";
import type { Mark } from "../lib/mark";
import type { ConfidentGuess } from "../lib/piece-ladder";
import { resolvePieceLadderState } from "../lib/piece-ladder";
import type { CarryForward, NoCommentZone, VerdictState, WsGap } from "../lib/session-review";
import { ConfirmPieceChip } from "./ConfirmPieceChip";
import { PieceLessMode } from "./PieceLessMode";
import { ScoreStand } from "./ScoreStand";
import { SessionEndedBanner } from "./SessionEndedBanner";
import { SessionReview } from "./SessionReview";

interface PracticeModeProps {
	userPickedPieceId: string | null;
	confidentGuess: ConfidentGuess | null;
	marks: readonly Mark[];
	elapsedSeconds: number;
	isPlaying: boolean;
	isRecording: boolean;
	/** Ends the session for real. The only exit from this full-screen surface
	 * during the practice phase; unlike SessionEndedBanner's resume, this is
	 * terminal. */
	onStop: () => void;
	/** Defaults to "practice" so every existing caller (AppChat, the
	 * /practice-preview harness) that never passes this is unaffected. */
	phase?: "practice" | "review";
	sessionId?: string;
	verdict?: VerdictState;
	onRetryVerdict?: () => void;
	carryForward?: CarryForward | null;
	wsGaps?: readonly WsGap[];
	noCommentZones?: readonly NoCommentZone[];
	onWorkOnThis?: (mark: Mark) => void;
	/** Leaves the review surface. Distinct from onStop: by the time review is
	 * showing, recording has already ended. */
	onExitReview?: () => void;
}
```

Update the function signature and header/body branches:

```tsx
export function PracticeMode({
	userPickedPieceId,
	confidentGuess,
	marks,
	elapsedSeconds,
	isPlaying,
	isRecording,
	onStop,
	phase = "practice",
	sessionId,
	verdict,
	onRetryVerdict,
	carryForward,
	wsGaps,
	noCommentZones,
	onWorkOnThis,
	onExitReview,
}: PracticeModeProps) {
	const [dismissed, setDismissed] = useState(false);
	const pause = usePauseTracker(phase === "practice" && isPlaying);

	const ladderState = resolvePieceLadderState({
		userPicked: userPickedPieceId,
		confidentGuess,
		dismissed,
	});

	const pieceId =
		ladderState === "user-picked"
			? userPickedPieceId
			: ladderState === "confirm-chip"
				? (confidentGuess?.pieceId ?? null)
				: null;

	return (
		<div className="flex h-full flex-col">
			<div className="flex shrink-0 items-center justify-end border-b border-border-subtle px-4 py-2">
				{phase === "review" ? (
					<button
						type="button"
						onClick={onExitReview}
						aria-label="Done"
						className="rounded-full bg-accent px-4 py-2 text-body-sm text-on-accent"
					>
						Done
					</button>
				) : (
					<button
						type="button"
						onClick={onStop}
						aria-label="Stop recording"
						className="rounded-full bg-danger px-4 py-2 text-body-sm text-on-accent"
					>
						Stop
					</button>
				)}
			</div>
			<div className="flex min-h-0 flex-1 flex-col">
				{phase === "review" ? (
					<SessionReview
						pieceId={pieceId}
						sessionId={sessionId ?? ""}
						marks={marks}
						durationSeconds={elapsedSeconds}
						verdict={verdict ?? { status: "loading" }}
						onRetryVerdict={onRetryVerdict ?? (() => {})}
						carryForward={carryForward ?? null}
						wsGaps={wsGaps ?? []}
						noCommentZones={noCommentZones ?? []}
						onWorkOnThis={onWorkOnThis ?? (() => {})}
					/>
				) : pause.autoStopped ? (
					<SessionEndedBanner onResume={pause.resume} />
				) : (
					<>
						{ladderState === "confirm-chip" && confidentGuess && (
							<ConfirmPieceChip
								guess={confidentGuess}
								onDismiss={() => setDismissed(true)}
							/>
						)}
						{pieceId ? (
							<ScoreStand
								pieceId={pieceId}
								marks={marks}
								elapsedSeconds={elapsedSeconds}
								isRecording={isRecording}
							/>
						) : (
							<PieceLessMode
								marks={marks}
								durationSeconds={Math.max(elapsedSeconds, 1)}
								elapsedSeconds={elapsedSeconds}
								isRecording={isRecording}
							/>
						)}
					</>
				)}
			</div>
		</div>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/components/PracticeMode.test.tsx
```
Expected: PASS (existing practice-phase tests unaffected; new review test
passes)

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/PracticeMode.tsx apps/web/src/components/PracticeMode.test.tsx
git commit -m "feat(web): PracticeMode transitions in place to a review phase"
```

---

## Task 14: `/practice-preview` gains a review-ready fixture section

**Group:** F (sequential — runs before Task 15; depends on Task 13 and
Task 3)

**Behavior being verified:** In dev mode, `/practice-preview` renders a
`PracticeMode` instance in `phase="review"` with fixture verdict/marks/
carry-forward/disclosures, and the verdict text is visible.
**Interface under test:** `PracticePreview` (route component), new fixture
exports from `practice-preview-fixtures.ts`.

**Files:**
- Modify: `apps/web/src/lib/practice-preview-fixtures.ts`
- Modify: `apps/web/src/routes/practice-preview.tsx`
- Modify: `apps/web/src/routes/practice-preview.test.tsx`

- [ ] **Step 1: Write the failing test**

Add to `apps/web/src/routes/practice-preview.test.tsx`:

```typescript
	it("renders the review-ready fixture section with the verdict and carry-forward", async () => {
		vi.stubEnv("DEV", true);
		const { PracticePreview } = await import("./practice-preview");
		render(<PracticePreview />);

		expect(
			screen.getByTestId("practice-mode-review-preview"),
		).toBeInTheDocument();
		expect(
			screen.getByText(/bass-blur from Tuesday is gone/),
		).toBeInTheDocument();
		expect(screen.getByText(/Start with bars 40-44/)).toBeInTheDocument();
	});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/routes/practice-preview.test.tsx
```
Expected: FAIL — `practice-mode-review-preview` testid not found (section
doesn't exist yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

Add to `apps/web/src/lib/practice-preview-fixtures.ts`:

```typescript
import type {
	CarryForward,
	NoCommentZone,
	VerdictState,
	WsGap,
} from "./session-review";

export const REVIEW_VERDICT_READY: VerdictState = {
	status: "ready",
	text: "That bass-blur from Tuesday is gone.",
};

export const REVIEW_VERDICT_FAILED: VerdictState = { status: "failed" };

export const REVIEW_CARRY_FORWARD: CarryForward = {
	text: "Start with bars 40-44 — the crescendo peaked early there.",
};

export const REVIEW_WS_GAPS: readonly WsGap[] = [
	{ startSeconds: 210, endSeconds: 450 },
];

export const REVIEW_NO_COMMENT_ZONES: readonly NoCommentZone[] = [
	{ atSeconds: 58, reason: "transcription confidence was too low here" },
];
```

Modify `apps/web/src/routes/practice-preview.tsx` — add imports and a new
section inside `PracticePreview`'s returned JSX, after the existing
`practice-mode-preview` section:

```typescript
import { createFileRoute } from "@tanstack/react-router";
import { PieceLessMode } from "../components/PieceLessMode";
import { PracticeMode } from "../components/PracticeMode";
import { ScoreStand } from "../components/ScoreStand";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import type { ConfidentGuess } from "../lib/piece-ladder";
import {
	FIXTURE_MARKS,
	PIECELESS_DURATION_SECONDS,
	REVIEW_CARRY_FORWARD,
	REVIEW_NO_COMMENT_ZONES,
	REVIEW_VERDICT_READY,
	REVIEW_WS_GAPS,
} from "../lib/practice-preview-fixtures";
```

```tsx
			<div className="border-b border-border-subtle" data-testid="practice-mode-review-preview">
				<div style={{ height: 500 }}>
					<PracticeMode
						userPickedPieceId="chopin-nocturne-op9-no2"
						confidentGuess={null}
						marks={SCORE_FIXTURE_MARKS}
						elapsedSeconds={212}
						isPlaying={false}
						isRecording={false}
						onStop={() => {}}
						phase="review"
						sessionId="practice-preview-session"
						verdict={REVIEW_VERDICT_READY}
						onRetryVerdict={() => {}}
						carryForward={REVIEW_CARRY_FORWARD}
						wsGaps={REVIEW_WS_GAPS}
						noCommentZones={REVIEW_NO_COMMENT_ZONES}
						onWorkOnThis={() => {}}
						onExitReview={() => {}}
					/>
				</div>
			</div>
```

(Insert this `<div>` block as a new sibling right after the existing
`data-testid="practice-mode-preview"` section, still inside the outer
`<div className="h-dvh">`.)

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/routes/practice-preview.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/practice-preview-fixtures.ts apps/web/src/routes/practice-preview.tsx apps/web/src/routes/practice-preview.test.tsx
git commit -m "feat(web): add a review-ready fixture section to /practice-preview"
```

---

## Task 15: `/practice-preview` gains a review-failed (synthesis-failure) fixture section

**Group:** F (sequential — runs after Task 14, same files)

**Behavior being verified:** A second `/practice-preview` section shows the
explicit synthesis-failure retry state — never blank — proving the success
criterion's "the synthesis-failure path visibly renders."
**Interface under test:** `PracticePreview` (route component), using
`REVIEW_VERDICT_FAILED`.

**Files:**
- Modify: `apps/web/src/routes/practice-preview.tsx`
- Modify: `apps/web/src/routes/practice-preview.test.tsx`

- [ ] **Step 1: Write the failing test**

Add to `apps/web/src/routes/practice-preview.test.tsx`:

```typescript
	it("renders the review-failed fixture section with an explicit retry state", async () => {
		vi.stubEnv("DEV", true);
		const { PracticePreview } = await import("./practice-preview");
		render(<PracticePreview />);

		expect(
			screen.getByTestId("practice-mode-review-failed-preview"),
		).toBeInTheDocument();
		expect(
			screen.getByText(/Couldn't put together a review this time/),
		).toBeInTheDocument();
		expect(
			screen.getByRole("button", { name: /try again/i }),
		).toBeInTheDocument();
	});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/web && bun run test -- src/routes/practice-preview.test.tsx
```
Expected: FAIL — `practice-mode-review-failed-preview` testid not found.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `apps/web/src/routes/practice-preview.tsx`, add `REVIEW_VERDICT_FAILED`
to the existing import from `practice-preview-fixtures`, and append a new
section after the review-ready one added in Task 14:

```typescript
import {
	FIXTURE_MARKS,
	PIECELESS_DURATION_SECONDS,
	REVIEW_CARRY_FORWARD,
	REVIEW_NO_COMMENT_ZONES,
	REVIEW_VERDICT_FAILED,
	REVIEW_VERDICT_READY,
	REVIEW_WS_GAPS,
} from "../lib/practice-preview-fixtures";
```

```tsx
			<div data-testid="practice-mode-review-failed-preview">
				<div style={{ height: 300 }}>
					<PracticeMode
						userPickedPieceId={null}
						confidentGuess={null}
						marks={[]}
						elapsedSeconds={60}
						isPlaying={false}
						isRecording={false}
						onStop={() => {}}
						phase="review"
						sessionId="practice-preview-session"
						verdict={REVIEW_VERDICT_FAILED}
						onRetryVerdict={() => {}}
						carryForward={null}
						wsGaps={[]}
						noCommentZones={[]}
						onWorkOnThis={() => {}}
						onExitReview={() => {}}
					/>
				</div>
			</div>
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/web && bun run test -- src/routes/practice-preview.test.tsx
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/routes/practice-preview.tsx apps/web/src/routes/practice-preview.test.tsx
git commit -m "feat(web): add a review-failed fixture section to /practice-preview"
```

---

## Final verification (after all 15 tasks land)

```bash
cd apps/web
bun run test
bun run typecheck
bun run lint
bun run dev
```

Then, per the spec's Verification Architecture manual click-through:
1. Visit `/practice-preview`.
2. Confirm the review-ready section shows "That bass-blur from Tuesday is
   gone." and the carry-forward line.
3. Tap a mark glyph on that section's score canvas; confirm evidence text,
   a passage-clip region, and a "Play" button render.
4. Tap "Work on this"; confirm it doesn't throw (the harness's
   `onWorkOnThis` is a no-op stub — firing without erroring is the
   verifiable claim per the spec's "Not in scope" on real drill generation).
5. Confirm the review-failed section shows "Couldn't put together a review
   this time." and a "Try again" button — never blank.

Record the outcome and any deviations as a `STATE:` comment on issue #159,
per `CLAUDE.md`'s work-tracking convention, before `/review`.

---

## Challenge Review

### CEO Pass

**Premise Challenge.** The problem is real and specific: `docs/apps/05-ui-system.md`
and `01-product-vision.md` both name session review as a required surface and
mark it NOT STARTED, and `AppChat.tsx`'s `handleStopPracticeMode` today drops
straight to `handleExitPracticeMode()` with the review payload dumped as chat
prose. Verified directly: `PracticeMode.tsx` currently has no `phase` concept
at all (confirmed by reading the file — single-branch component, `onStop` is
the only exit). The plan's framing that "two pieces of machinery are stranded"
(`PlayPassageCard.tsx`, `ExerciseSetCard.tsx`) is accurate but their actual
path is `apps/web/src/components/cards/`, not `apps/web/src/components/` as
the plan's prose implies in a couple of places — cosmetic, not a blocker,
since no task actually imports from the wrong path.

**Real pain?** Yes — without this, #162 (backend review synthesis, blocked on
having a client contract) and #160/#161 (drill entry, ask-about-this) have
nothing to render into. This is genuinely on the critical path, not gold-plating.

**Existing coverage exploited well.** The plan's core architectural bet —
extend `ScoreMarkLayer`/`SessionTimelineStrip`/`ScoreStand`/`PieceLessMode`
with one optional `renderDetail` prop each rather than forking them — is
verified sound by reading all four files. Each currently hard-codes
`<MarkDetail mark={mark} onClose={...} />` at exactly one call site with no
other logic branching on it, so the diff really is minimal and additive, and
every default-omitting caller (there are several: `practice-preview.tsx`'s
existing three demo blocks, and presumably `AppChat.tsx`) is provably
unaffected by TypeScript's optional-prop default behavior alone.

**Scope check.** The plan matches the spec's stated scope precisely — no
drift found. The "Not in scope" list (backend synthesis, drill generation,
ask-about-this threads, live `AppChat` wiring) is honored: no task tries to
wire a live WS event into these props. 15 tasks across 6 groups touching ~13
files is on the high side of the ">8 files" complexity-smell heuristic, but
the file count is inflated by the paired-props pattern (4 near-identical
`renderDetail` plumbing tasks: 4, 5, 9, 10) rather than genuine architectural
sprawl — those four tasks are legitimately small and mechanical.

**Twelve-month alignment.**
```
CURRENT STATE                    THIS PLAN                      12-MONTH IDEAL
Stop -> chat dump,            -> SessionReview surface        -> Stop -> live
no review surface,               (fixture-driven, proven          server-fed
PlayPassageCard/Exercise-        via /practice-preview),          review with
SetCard stranded in chat          renderDetail bridges Canvas      real synthesis
(soon-removed #164)                A/B to a review expansion       (#162), real
                                                                     drill assign
                                                                     (#160/161)
```
This moves cleanly toward the ideal: it does not add tech debt that conflicts
with #162 landing later, because `VerdictState`/`CarryForward`/`WsGap`/
`NoCommentZone` are deliberately typed as the client contract this plan
commits to, not a guess at the WS payload shape (the spec's rejected
alternative — extending `PracticeWsEvent` now — is explicitly and correctly
avoided).

**Alternatives.** The spec documents two rejected alternatives with real
reasoning (duplicate canvases; always-on playback+drill in `MarkDetail`) and
one rejected alternative for the WS event shape. No `[QUESTION]` needed here —
this bar is met.

### Engineering Pass

**Architecture / data flow.**
```
PracticeMode(phase="review")
   -> SessionReview(props)
        -> VerdictBanner(verdict, onRetry)        [closed union, exhaustive switch]
        -> ScoreStand | PieceLessMode(marks, renderDetail=MarkReviewDetail)
                -> ScoreMarkLayer|SessionTimelineStrip(renderDetail)
                       -> MarkReviewDetail(mark, sessionId, onClose, onWorkOnThis)
                              -> MarkPassagePlayback(mark, sessionId)
                                     -> api.sessions.getPassage(sessionId, bars)
                                     -> scoreRenderer.getClipPlayback(pieceId, b0, b1)
                                     -> PassagePlayer(manifest, ctx).load()/.play()
                                     -> createPassageQstampSource(barTimeline, ir.bars, tSecGetter)
                                     -> ScoreCursor(pieceId, container, ir, qstampSource).start()
        -> SessionDisclosures(wsGaps, noCommentZones)
        -> carryForward && <p>Next: ...</p>
```
Verified against real code: `PassagePlayer` (`passage-player.ts`) constructor
`(manifest, ctx)`, `.load()/.play()/.onTick()/.destroy()` all match Task 6's
usage exactly. `ScoreCursor` (`score-cursor.ts`) constructor
`{pieceId, container, ir, qstampSource}` and `.start()/.stop()` match Task 6
and Task 2 exactly. `scoreRenderer.getClipPlayback` (verified in
`score-renderer.ts:189`) returns `{svg, ir, notes}` exactly as Task 6 expects
— this is the correct sibling of `getClip` (which `PlayPassageCard` uses and
which only returns `svg`), so Task 6 is not silently reusing the wrong,
narrower API. `PassageManifest` (`types.ts:120`) has `barTimeline: {bar, tSec}[]`
matching Task 2's bridge input exactly. `mark.ts`'s `resolveAnchor`,
`anchorLabel`, `formatElapsed`, and the `Mark`/`MarkAnchor`/`MarkConfidence`
shapes all match every task's usage (including the `confidence?:
"exploratory"` framing branch in `MarkReviewDetail`, Task 11).

**[RISK] (confidence: 7/10) — `sessionIdRef` is nulled by `cleanup()` at
the exact moment session data would matter for review.** Verified in
`usePracticeSession.ts`: `cleanup()` (called from the `"synthesis"` and
`"session_summary"` WS-message handlers, i.e. exactly when review data
arrives) does `sessionIdRef.current = null;`. Task 3's own stated
justification is "needed by any surface (like the future review) that must
fetch a passage scoped to this session" — but by the time a live `AppChat`
integration (future #162 work) would read `sessionId` to feed `SessionReview`,
the hook will have already cleared it in the same tick that produced the
review data. Task 3's own test never exercises this path (it stops at
`state === "recording"`, before any `synthesis`/`session_summary` message),
so the test passes without proving the property the task exists to serve.
This is not a blocker for #159 (which stops at `/practice-preview` fixtures
and never wires `AppChat`), but it is a real landmine for whoever picks up
#162's live wiring, and it would be cheap to flag now in the plan's Open
Questions or a code comment on `sessionIdRef` so it isn't rediscovered the
hard way later. Fallback: capture `sessionIdRef.current` into a separate
"last completed session id" ref/state that `cleanup()` does not clear, at
the point `#162`'s wiring actually needs it — out of scope for this plan to
build, but worth a one-line note.

**[OBS] — `ScoreStand`/`PieceLessMode` both keep rendering their live
"Metronome" toggle button when reused for review (`isRecording={false}`,
`elapsedSeconds={durationSeconds}`).** Neither component has a way to
suppress it, and the plan doesn't add one. A metronome toggle on a
post-session review screen is very likely unintended UX, but it's cosmetic,
not functionally broken, and the manual click-through step in the Final
Verification section would surface it immediately if it matters. Not
blocking; worth a one-line mention to whoever does the manual click-through.

**[OBS] — `PieceLessMode`'s `durationSeconds` and `elapsedSeconds` are both
wired to the same `durationSeconds` prop in `SessionReview`/`PracticeMode`'s
review branch** (`durationSeconds={elapsedSeconds}` when calling
`PieceLessMode`, and `SessionReview` passes its own `durationSeconds` to
both `elapsedSeconds` and `durationSeconds` params of the pieceless canvas).
Since `PieceLessMode`'s `elapsedSeconds` only drives a large "current time"
readout, showing the session's total duration as if it were "now" reads
fine for a finished session — this is very likely intentional reuse, not
a bug, but it's worth naming since it's easy to misread as a mistake during
review.

**Module depth audit.**
- `session-review.ts` — interface: 4 types + `formatWsGap`. Implementation:
  ~5 lines. Verdict: borderline SHALLOW as a standalone module, but its
  actual role is "one shared vocabulary of types + one formatting function
  used by 3 other components" (`VerdictBanner`, `SessionDisclosures`,
  `SessionReview`) — DEEP in aggregate given how many callers it decouples
  from each other, not shallow.
- `passage-cursor.ts` — interface: 1 function (`createPassageQstampSource`).
  Implementation: real branching logic (bracketing-pair search, fraction
  math, null-degradation on three distinct conditions). Verdict: DEEP —
  exactly the kind of "one new bridge that hides real complexity" the spec
  calls out as the one genuinely new algorithmic piece.
- `MarkPassagePlayback.tsx` — interface: 1 component, 4 test-only escape-hatch
  props mirroring `PlayPassageCard`'s established pattern. Implementation:
  5-state lifecycle (`no_bars/loading/ready/audio_error/error`), two effects,
  three collaborator integrations. Verdict: DEEP.
- `VerdictBanner.tsx` / `SessionDisclosures.tsx` / `MarkReviewDetail.tsx` —
  each is a small, single-purpose presentational component with 1-3 props
  and a handful of branches. Verdict: appropriately shallow-and-simple for
  what they are — not a smell, since none of them hide meaningful logic they
  shouldn't (that logic correctly lives in `passage-cursor.ts`/
  `mark.ts`/`session-review.ts` instead).
- `SessionReview.tsx` — interface: 10 props. Implementation: canvas selection
  + 4 sub-component composition. Verdict: DEEP as a composer, but it's worth
  noting 10 props is a lot for one component; most are pass-throughs
  (`wsGaps`, `noCommentZones`, `carryForward`, `verdict`, `onRetryVerdict`)
  rather than derived, so this is inherent to being a pure composition layer
  over a client contract that doesn't exist server-side yet — acceptable.

**Test philosophy / vertical slice audit.** All 15 tasks are one behavior
test -> minimal implementation -> one commit, with no task writing tests for
future tasks' implementations (verified against every task's Step
1-3 sequence) and no task bundling unrelated behaviors into one commit. Tests
exercise public component/hook interfaces and rendered output
(`getByText`/`getByRole`/`getByLabelText`), not internal state or private
methods. External boundaries (`api`, `scoreRenderer`, `PassagePlayer`,
`ScoreCursor`, `AudioContext`) are mocked at module boundaries in Tasks 6, 11,
and 12 — correct, since those are genuine I/O/DOM/Web-Audio boundaries
jsdom cannot exercise, not internal collaborators of the module under test.
Every test would fail before its implementation exists (verified by tracing
the "cannot find module" / "prop does not exist" failure mode stated in each
task's Step 2) — none are shape-only tests.

**Test coverage gaps.**
```
[+] apps/web/src/lib/passage-cursor.ts
    └── createPassageQstampSource()
        ├── [TESTED] ★★  mid-bar interpolation — Task 2 test
        ├── [TESTED] ★★  null when playback not started — Task 2 test
        ├── [TESTED] ★★  null past last known bar boundary — Task 2 test
        └── [GAP]         null when `bars` has no entry for the resolved bar
                           number (a clip IR that doesn't cover the manifest's
                           bar) — spec explicitly calls this degradation path
                           out (session-review.md line ~159) but no task tests
                           it. RISK, not BLOCKER: it's a straightforward `if
                           (!barIR) return null;` already in the reference
                           implementation, and the two tested null-paths give
                           reasonable confidence in the pattern, but it is the
                           one branch the spec singles out by name that has
                           zero test coverage.

[+] apps/web/src/components/MarkPassagePlayback.tsx
    └── MarkPassagePlayback()
        ├── [TESTED] ★★  no_bars (timestamp-only mark) — Task 6 test
        ├── [TESTED] ★★  ready (fetch + clip + Play control) — Task 6 test
        ├── [GAP]         error state (getPassage/getClipPlayback rejects) —
        │                 no test, despite the component defining this state
        │                 and the spec explicitly listing it as one of five
        │                 states this module hides
        └── [GAP]         audio_error state (PassagePlayer.load() rejects) —
                          same: defined, spec-named, untested
```
**[RISK] (confidence: 6/10) — `MarkPassagePlayback`'s `error` and
`audio_error` states are implemented but never exercised by a test**, even
though the spec's own module doc explicitly calls out "loading/ready/
audio_error/error/no_bars states" as the thing this component's tests should
cover, and `VerdictBanner`'s sibling failure-state (`failed`) *is* tested.
This is inconsistent rigor within the same plan for two components with the
same "loud, never blank" design goal. Not a BLOCKER because these aren't
data-mutation or auth paths and the existing 2-of-5 states give real signal,
but per the plan's own "Iron rule" instinct (test what the spec calls a named
state), this is worth one more `it()` block in Task 6 before merge — mock
`mockGetPassage.mockRejectedValue(...)` for `error`, and make `mockLoad`
reject for `audio_error`.

**Failure modes.** Every async operation in `MarkPassagePlayback` (fetch,
clip fetch, audio load) is wrapped in `try/catch` with a `console.error` and
an explicit failure state set — no silent failures, no bare `catch (e) {}`.
`cancelled` guards prevent state updates after unmount, and cleanup
(`player.destroy()`, `ctx.close()`) runs on every unwind path including the
mid-flight-cancel branch. `ScoreCursor`'s `tick()` (existing code, unchanged)
already wraps its per-frame work in try/catch reporting to Sentry — the new
`passage-cursor.ts` bridge composes into that existing safety net rather than
needing its own. No new database writes or transaction boundaries are
introduced by this plan (client-only). `SessionReview`'s composition has no
async operations of its own to fail.

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `ScoreMarkLayer`/`SessionTimelineStrip`/`ScoreStand`/`PieceLessMode` each have exactly one `MarkDetail` call site with no other logic branching on it | SAFE | Verified by reading all four files — each has exactly one such site. |
| `PassagePlayer`, `ScoreCursor`, `scoreRenderer.getClipPlayback`, `PassageManifest.barTimeline`, `BarIR.qstampStart/qstampEnd` have the shapes the plan's code snippets use | SAFE | Verified directly against `passage-player.ts`, `score-cursor.ts`, `score-renderer.ts`, `types.ts`, `score-ir.ts` — all match. |
| `mark.ts`'s `resolveAnchor`/`anchorLabel`/`formatElapsed`/`Mark`/`MarkConfidence` match every task's usage | SAFE | Verified directly against `mark.ts`. |
| `PlayPassageCard.tsx`'s mocking pattern is the right precedent for `MarkPassagePlayback.test.tsx` | SAFE | Verified — `PlayPassageCard.tsx` (at `apps/web/src/components/cards/`, not the path implied by the spec prose) uses the identical `_mockManifest/_mockClip/_playable` escape hatch Task 6 copies. |
| `usePracticeSession`'s exposed `sessionId` will be usable by a future review surface fed from a live session | RISKY | `cleanup()` nulls `sessionIdRef.current` exactly when `synthesis`/`session_summary` arrives — the moment review data exists. Out of scope for #159 to fix, but the assumption underlying Task 3's stated purpose does not hold as written; flag for #162. |
| `usePauseTracker(phase === "practice" && isPlaying)` correctly suppresses the auto-stop banner during review | SAFE | Hook signature verified (`usePauseTracker(isPlaying: boolean)`); gating the input on `phase === "practice"` is a sound, minimal guard. |
| `MarkGlyph`'s aria-label format (`"Needs work: Pedaling, bars 5-6"` / `"...bar 1"` / `"...0:30"`) matches what every new/modified test asserts against | SAFE | Verified `MarkGlyph.tsx` renders a computed `label` as `aria-label`; existing tests in the same files already assert this exact format, so the new tests are following an established, working pattern rather than guessing. |
| `apps/web/src/test-utils/mark-fixtures.ts` (imported by Task 4's test) exists and exports `FIXTURE_BARS`/`FIXTURE_MARKS` | SAFE | Confirmed present on disk. |
| `SCORE_FIXTURE_MARKS` referenced in Task 14's JSX snippet is in scope | SAFE | It's a pre-existing local `const` already declared in `practice-preview.tsx` (not something Task 14 needs to import) — the task's snippet correctly omits it from the new import list. |

### Summary
[BLOCKER] count: 0
[RISK]    count: 3
[QUESTION] count: 0

VERDICT: PROCEED_WITH_CAUTION — (1) `usePracticeSession`'s `sessionIdRef` is
cleared by `cleanup()` at the exact moment `synthesis`/`session_summary`
arrives, which undermines Task 3's stated rationale for exposing `sessionId`
for a future live review; not a blocker for this plan's fixture-only scope,
but should be left as a one-line comment/Open-Question so #162's wiring
doesn't silently break on it. (2) `MarkPassagePlayback`'s `error` and
`audio_error` states are implemented and spec-named but untested in Task 6 —
cheap to close with two more assertions before `/review`. (3) `ScoreStand`/
`PieceLessMode`'s metronome toggle remains visible/interactive during review
with no way to suppress it — confirm during the manual click-through whether
this is acceptable or needs a follow-up.
