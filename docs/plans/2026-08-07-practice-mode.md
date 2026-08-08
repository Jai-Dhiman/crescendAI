# Practice Mode: The Digital Music Stand — Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent
> per task). Do NOT start execution until /challenge returns
> VERDICT: PROCEED.
>
> **Every command below must be run with an explicit absolute `cd` to**
> `/Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web`
> **— a leaked `cd` to the repo root makes `bunx vitest` resolve a different
> vitest with no project config and silently pass or fail wrong.**

**Goal:** Replace the chat-first recording overlay (`ListeningMode` +
`AudioWaveformRing`) with a full-screen practice surface — a static,
manually-paged score (or a minimal pieceless timeline) that receives pause
marks silently and soft-auto-stops after a minute of silence.
**Spec:** docs/specs/2026-08-07-practice-mode-design.md
**Style:** Follow the project's coding standards (CLAUDE.md /
apps/CLAUDE.md); tabs, not spaces, matching every file read while writing
this plan.

## Task Groups

```
Group A (parallel):      Task 1, Task 2, Task 3
Group B (parallel,
  Task 4 needs Task 1,
  Task 5 needs Task 3):  Task 4, Task 5
Group C (parallel,
  independent of A/B):   Task 6, Task 7, Task 8, Task 9
Group D (depends on
  B + C):                Task 10
Group E (depends on D;
  SEQUENTIAL within the
  group — both tasks
  edit AppChat.tsx):     Task 11, then Task 12
Group F (Task 13 needs
  C's Task 8 + Task 9;
  SEQUENTIAL within the
  group — Task 14 edits
  the file Task 13
  creates):              Task 13, then Task 14
```

`[SHIPS INDEPENDENTLY]`: none. The issue's success criterion is a single
click-through across score-stand, pieceless, and auto-stop/resume, so no
task group is independently user-visible until Group E lands. Group F (the
real-browser geometry harness) is independently valuable as a regression
gate the moment it is green, even before Group E wires it into the live app.

---

### Task 1: Pause/auto-stop threshold math is pure and boundary-correct

**Group:** A (parallel with Task 2, Task 3)

**Behavior being verified:** given how long the student has been silent,
`computePauseState` reports whether a mark may show and whether auto-stop has
triggered, using `>=` at both boundaries.

**Interface under test:** `computePauseState(input: PauseStateInput):
PauseState`

**Files:**
- Create: `src/lib/pause-state.ts`
- Test: `src/lib/pause-state.test.ts`

- [x] **Step 1: Write the failing test**

```typescript
// src/lib/pause-state.test.ts
import { describe, expect, it } from "vitest";
import {
	AUTO_STOP_SILENCE_MS,
	computePauseState,
	MARK_SILENCE_MS,
} from "./pause-state";

describe("computePauseState", () => {
	it("reports no mark and no auto-stop while playing", () => {
		const state = computePauseState({
			isPlaying: true,
			silenceStartedAt: null,
			now: 100_000,
		});
		expect(state).toEqual({ silenceMs: 0, canShowMark: false, autoStopped: false });
	});

	it("allows a mark at exactly the 20s boundary, not before", () => {
		const justUnder = computePauseState({
			isPlaying: false,
			silenceStartedAt: 0,
			now: MARK_SILENCE_MS - 1,
		});
		expect(justUnder.canShowMark).toBe(false);

		const atBoundary = computePauseState({
			isPlaying: false,
			silenceStartedAt: 0,
			now: MARK_SILENCE_MS,
		});
		expect(atBoundary.canShowMark).toBe(true);
	});

	it("auto-stops at exactly the 60s boundary, not before", () => {
		const justUnder = computePauseState({
			isPlaying: false,
			silenceStartedAt: 0,
			now: AUTO_STOP_SILENCE_MS - 1,
		});
		expect(justUnder.autoStopped).toBe(false);

		const atBoundary = computePauseState({
			isPlaying: false,
			silenceStartedAt: 0,
			now: AUTO_STOP_SILENCE_MS,
		});
		expect(atBoundary.autoStopped).toBe(true);
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/lib/pause-state.test.ts
```
Expected: FAIL — `Cannot find module './pause-state'` (the file does not
exist yet).

- [x] **Step 3: Implement the minimum to make the test pass**

```typescript
// src/lib/pause-state.ts

/**
 * How long the student must be silent before a mark may show. Tunable
 * per the epic's open question ("20s is a starting value") — a single
 * named constant, not a scatter of magic numbers, is what makes that tuning
 * a one-line change later.
 */
export const MARK_SILENCE_MS = 20_000;

/** How long the student must be silent before the soft auto-stop banner shows. */
export const AUTO_STOP_SILENCE_MS = 60_000;

export interface PauseStateInput {
	readonly isPlaying: boolean;
	/** Timestamp (ms, same clock as `now`) silence began, or null while playing. */
	readonly silenceStartedAt: number | null;
	/** Current timestamp (ms), supplied by the caller so this stays a pure function. */
	readonly now: number;
}

export interface PauseState {
	readonly silenceMs: number;
	readonly canShowMark: boolean;
	readonly autoStopped: boolean;
}

/**
 * Pure boundary arithmetic over one silence interval. No timers, no DOM —
 * the caller (usePauseTracker) owns the clock and the ref; this only answers
 * "given this much silence, what should the UI show."
 */
export function computePauseState(input: PauseStateInput): PauseState {
	if (input.isPlaying || input.silenceStartedAt === null) {
		return { silenceMs: 0, canShowMark: false, autoStopped: false };
	}
	const silenceMs = Math.max(0, input.now - input.silenceStartedAt);
	return {
		silenceMs,
		canShowMark: silenceMs >= MARK_SILENCE_MS,
		autoStopped: silenceMs >= AUTO_STOP_SILENCE_MS,
	};
}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/lib/pause-state.test.ts
```
Expected: PASS (3 tests)

- [x] **Step 5: Commit**

```bash
git add src/lib/pause-state.ts src/lib/pause-state.test.ts && git commit -m "feat(practice-mode): pure pause/auto-stop threshold math"
```

---

### Task 2: Piece ladder precedence is a pure decision

**Group:** A (parallel with Task 1, Task 3)

**Behavior being verified:** given a user pick, a confident guess, and
whether the guess was dismissed, `resolvePieceLadderState` picks exactly one
of the three ladder states, with the user pick always winning and a
dismissed guess never resurfacing.

**Interface under test:** `resolvePieceLadderState(input: LadderInput):
LadderState`

**Files:**
- Create: `src/lib/piece-ladder.ts`
- Test: `src/lib/piece-ladder.test.ts`

- [x] **Step 1: Write the failing test**

```typescript
// src/lib/piece-ladder.test.ts
import { describe, expect, it } from "vitest";
import { resolvePieceLadderState } from "./piece-ladder";

const guess = {
	pieceId: "chopin-nocturne-op9-no2",
	composer: "Chopin",
	title: "Nocturne Op. 9 No. 2",
	confidence: 0.92,
};

describe("resolvePieceLadderState", () => {
	it("prefers the user's pick over a confident guess", () => {
		expect(
			resolvePieceLadderState({
				userPicked: "chopin-nocturne-op9-no2",
				confidentGuess: guess,
				dismissed: false,
			}),
		).toBe("user-picked");
	});

	it("shows the confirm chip when there is a guess and no pick", () => {
		expect(
			resolvePieceLadderState({
				userPicked: null,
				confidentGuess: guess,
				dismissed: false,
			}),
		).toBe("confirm-chip");
	});

	it("falls to pieceless once the guess is dismissed", () => {
		expect(
			resolvePieceLadderState({
				userPicked: null,
				confidentGuess: guess,
				dismissed: true,
			}),
		).toBe("pieceless");
	});

	it("is pieceless with neither a pick nor a guess", () => {
		expect(
			resolvePieceLadderState({
				userPicked: null,
				confidentGuess: null,
				dismissed: false,
			}),
		).toBe("pieceless");
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/lib/piece-ladder.test.ts
```
Expected: FAIL — `Cannot find module './piece-ladder'`

- [x] **Step 3: Implement the minimum to make the test pass**

```typescript
// src/lib/piece-ladder.ts

export interface ConfidentGuess {
	readonly pieceId: string;
	readonly composer: string;
	readonly title: string;
	readonly confidence: number;
}

export interface LadderInput {
	readonly userPicked: string | null;
	readonly confidentGuess: ConfidentGuess | null;
	readonly dismissed: boolean;
}

export type LadderState = "user-picked" | "confirm-chip" | "pieceless";

/**
 * The piece resolution ladder (docs/apps/02-pipeline.md#3): user pick beats
 * a confident guess, and a dismissed guess never resurfaces — there is no
 * fourth state to re-summon it mid-session.
 */
export function resolvePieceLadderState(input: LadderInput): LadderState {
	if (input.userPicked) return "user-picked";
	if (input.confidentGuess && !input.dismissed) return "confirm-chip";
	return "pieceless";
}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/lib/piece-ladder.test.ts
```
Expected: PASS (4 tests)

- [x] **Step 5: Commit**

```bash
git add src/lib/piece-ladder.ts src/lib/piece-ladder.test.ts && git commit -m "feat(practice-mode): pure piece-ladder precedence"
```

---

### Task 3: `PracticeWsEvent` gains a `mark` variant

**Group:** A (parallel with Task 1, Task 2)

**Behavior being verified:** a WS payload shaped `{ type: "mark", mark:
Mark }` type-checks as a `PracticeWsEvent` — this is a type-level contract,
so its "test" is `tsc`, not vitest (there is no runtime behavior in a
union-type addition to assert against).

**Interface under test:** `PracticeWsEvent` (discriminated union in
`practice-api.ts`)

**Files:**
- Modify: `src/lib/practice-api.ts`
- Test: `src/lib/practice-api.marktype.test.ts`

- [x] **Step 1: Write the failing test**

```typescript
// src/lib/practice-api.marktype.test.ts
import { describe, expect, it } from "vitest";
import { resolveAnchor } from "./mark";
import type { Mark } from "./mark";
import type { PracticeWsEvent } from "./practice-api";

describe("PracticeWsEvent mark variant", () => {
	it("accepts a { type: 'mark', mark } event and narrows it by discriminant", () => {
		const mark: Mark = {
			id: "m1",
			anchor: resolveAnchor({ atSeconds: 12, alignmentQuality: 0 }),
			taxonomy: "needs_work",
			dimension: "pedaling",
			evidence: "test evidence",
			lifecycle: "active",
		};
		const event: PracticeWsEvent = { type: "mark", mark };

		// Runtime assertion so this test is not pure compile-time theatre: the
		// discriminant narrows the union and the payload survives a JSON round
		// trip unchanged, which is what the WS transport actually does to it.
		const roundTripped: PracticeWsEvent = JSON.parse(JSON.stringify(event));
		expect(roundTripped.type).toBe("mark");
		if (roundTripped.type === "mark") {
			expect(roundTripped.mark.id).toBe("m1");
		}
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/lib/practice-api.marktype.test.ts && bunx tsc --noEmit
```
Expected: FAIL — `tsc` reports `Object literal may only specify known
properties, and 'mark' does not exist in type ... PracticeWsEvent` (the
union has no `mark` variant yet).

- [x] **Step 3: Implement the minimum to make the test pass**

In `src/lib/practice-api.ts`, add the import and the union member:

```typescript
import type { Mark } from "./mark";
```

Add near the top of the file alongside the other type imports (the file
already imports `InlineComponent` from `./types`).

Then extend the `PracticeWsEvent` union (immediately after the
`"piece_set"` variant, before `ModeChangeEvent`, so the new variant sits
next to the other single-purpose events rather than the multi-field ones):

```typescript
	| { type: "piece_set"; query: string }
	| { type: "mark"; mark: Mark }
	| ModeChangeEvent
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/lib/practice-api.marktype.test.ts && bunx tsc --noEmit
```
Expected: PASS (1 test), `tsc` exits 0.

- [x] **Step 5: Commit**

```bash
git add src/lib/practice-api.ts src/lib/practice-api.marktype.test.ts && git commit -m "feat(practice-mode): add mark WS event to PracticeWsEvent"
```

---

### Task 4: `usePauseTracker` drives `computePauseState` off a live clock

**Group:** B (depends on Task 1)

**Behavior being verified:** the hook reports `canShowMark`/`autoStopped`
consistently with elapsed fake time since `isPlaying` last went false, and
`resume()` resets the clock without needing `isPlaying` to change.

**Interface under test:** `usePauseTracker(isPlaying: boolean): {
silenceMs: number; canShowMark: boolean; autoStopped: boolean; resume: () =>
void }`

**Files:**
- Create: `src/hooks/usePauseTracker.ts`
- Test: `src/hooks/usePauseTracker.test.ts`

- [x] **Step 1: Write the failing test**

```typescript
// src/hooks/usePauseTracker.test.ts
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AUTO_STOP_SILENCE_MS, MARK_SILENCE_MS } from "../lib/pause-state";
import { usePauseTracker } from "./usePauseTracker";

describe("usePauseTracker", () => {
	beforeEach(() => {
		vi.useFakeTimers();
	});
	afterEach(() => {
		vi.useRealTimers();
	});

	it("shows no mark and does not auto-stop while playing", () => {
		const { result } = renderHook(() => usePauseTracker(true));
		act(() => {
			vi.advanceTimersByTime(AUTO_STOP_SILENCE_MS + 5000);
		});
		expect(result.current.canShowMark).toBe(false);
		expect(result.current.autoStopped).toBe(false);
	});

	it("allows a mark once silence reaches the threshold after playing stops", () => {
		const { result, rerender } = renderHook(
			({ isPlaying }) => usePauseTracker(isPlaying),
			{ initialProps: { isPlaying: true } },
		);
		rerender({ isPlaying: false });
		act(() => {
			vi.advanceTimersByTime(MARK_SILENCE_MS);
		});
		expect(result.current.canShowMark).toBe(true);
		expect(result.current.autoStopped).toBe(false);
	});

	it("resume() resets the silence clock without requiring isPlaying to change", () => {
		const { result, rerender } = renderHook(
			({ isPlaying }) => usePauseTracker(isPlaying),
			{ initialProps: { isPlaying: true } },
		);
		rerender({ isPlaying: false });
		act(() => {
			vi.advanceTimersByTime(AUTO_STOP_SILENCE_MS);
		});
		expect(result.current.autoStopped).toBe(true);

		act(() => {
			result.current.resume();
		});
		expect(result.current.autoStopped).toBe(false);
		expect(result.current.silenceMs).toBe(0);
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/hooks/usePauseTracker.test.ts
```
Expected: FAIL — `Cannot find module './usePauseTracker'`

- [x] **Step 3: Implement the minimum to make the test pass**

```typescript
// src/hooks/usePauseTracker.ts
import { useCallback, useEffect, useRef, useState } from "react";
import type { PauseState } from "../lib/pause-state";
import { computePauseState } from "../lib/pause-state";

export interface UsePauseTrackerReturn extends PauseState {
	/** Resets the silence clock in place. Does not touch isPlaying or any
	 * session/recording state — the auto-stop banner is UI-only (see spec). */
	resume: () => void;
}

/**
 * Wraps computePauseState with a live clock. silenceStartedAt is a ref, not
 * state: it changes every tick indirectly (via the 1s interval re-deriving
 * `now`), and putting the timestamp itself in state would double the
 * re-render rate for no visible benefit.
 */
export function usePauseTracker(isPlaying: boolean): UsePauseTrackerReturn {
	const silenceStartedAtRef = useRef<number | null>(isPlaying ? null : Date.now());
	const [state, setState] = useState<PauseState>(() =>
		computePauseState({
			isPlaying,
			silenceStartedAt: silenceStartedAtRef.current,
			now: Date.now(),
		}),
	);

	useEffect(() => {
		if (isPlaying) {
			silenceStartedAtRef.current = null;
		} else if (silenceStartedAtRef.current === null) {
			silenceStartedAtRef.current = Date.now();
		}
		setState(
			computePauseState({
				isPlaying,
				silenceStartedAt: silenceStartedAtRef.current,
				now: Date.now(),
			}),
		);
	}, [isPlaying]);

	useEffect(() => {
		const id = setInterval(() => {
			setState(
				computePauseState({
					isPlaying,
					silenceStartedAt: silenceStartedAtRef.current,
					now: Date.now(),
				}),
			);
		}, 1000);
		return () => clearInterval(id);
	}, [isPlaying]);

	const resume = useCallback(() => {
		silenceStartedAtRef.current = isPlaying ? null : Date.now();
		setState(
			computePauseState({
				isPlaying,
				silenceStartedAt: silenceStartedAtRef.current,
				now: Date.now(),
			}),
		);
	}, [isPlaying]);

	return { ...state, resume };
}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/hooks/usePauseTracker.test.ts
```
Expected: PASS (3 tests)

- [x] **Step 5: Commit**

```bash
git add src/hooks/usePauseTracker.ts src/hooks/usePauseTracker.test.ts && git commit -m "feat(practice-mode): usePauseTracker live-clock wrapper"
```

---

### Task 5: `usePracticeSession` accumulates marks from `mark` WS events

**Group:** B (depends on Task 3)

**Behavior being verified:** once a session is recording, a `mark` WS
message appends to `marks` in the hook's return value, in arrival order,
without touching any other state field.

**Interface under test:** `usePracticeSession(): UsePracticeSessionReturn`
(specifically the new `marks: Mark[]` field)

**Files:**
- Modify: `src/hooks/usePracticeSession.ts`
- Test: `src/hooks/usePracticeSession.marks.test.ts`

- [x] **Step 1: Write the failing test**

```typescript
// src/hooks/usePracticeSession.marks.test.ts
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resolveAnchor } from "../lib/mark";
import type { Mark } from "../lib/mark";
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
		return { fftSize: 256 };
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

describe("usePracticeSession marks", () => {
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

	it("appends a mark to state when a mark WS event arrives", async () => {
		const { result } = renderHook(() => usePracticeSession());

		await act(async () => {
			await result.current.start();
		});
		await waitFor(() => expect(result.current.state).toBe("recording"));

		const mark: Mark = {
			id: "m1",
			anchor: resolveAnchor({ atSeconds: 12, alignmentQuality: 0 }),
			taxonomy: "needs_work",
			dimension: "pedaling",
			evidence: "test evidence",
			lifecycle: "active",
		};

		act(() => {
			socket.onmessage?.({
				data: JSON.stringify({ type: "mark", mark }),
			} as MessageEvent);
		});

		expect(result.current.marks).toEqual([mark]);
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/hooks/usePracticeSession.marks.test.ts
```
Expected: FAIL — `result.current.marks` is `undefined` (`toEqual([mark])`
fails; the field does not exist on `UsePracticeSessionReturn` yet).

- [x] **Step 3: Implement the minimum to make the test pass**

In `src/hooks/usePracticeSession.ts`:

Add the import (alongside the other `../lib/mark` — there is none yet, so
add a fresh import line near the top with the other `../lib/*` imports):

```typescript
import type { Mark } from "../lib/mark";
```

Add state, next to the other `useState` declarations:

```typescript
	const [marks, setMarks] = useState<Mark[]>([]);
```

Reset it in `start()`, alongside the other per-session resets
(`setObservations([])`, `setLatestScores(null)`, etc.):

```typescript
			setMarks([]);
```

Handle the new event in the `handleWsMessage` switch, as a new `case`
(placed next to `case "piece_set":` since both are simple single-field
appends with no other side effects):

```typescript
				case "mark": {
					setMarks((prev) => [...prev, data.mark]);
					break;
				}
```

Add to the returned object, alongside `observations`:

```typescript
		marks,
```

And to the `UsePracticeSessionReturn` interface, alongside `observations:
ObservationEvent[];`:

```typescript
	marks: Mark[];
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/hooks/usePracticeSession.marks.test.ts && bunx tsc --noEmit
```
Expected: PASS (1 test), `tsc` exits 0.

- [x] **Step 5: Commit**

```bash
git add src/hooks/usePracticeSession.ts src/hooks/usePracticeSession.marks.test.ts && git commit -m "feat(practice-mode): usePracticeSession accumulates marks from WS"
```

---

### Task 6: `ConfirmPieceChip` names the guess and dismisses once

**Group:** C (parallel with Task 7, Task 8, Task 9)

**Behavior being verified:** the chip renders the guess's title and calls
`onDismiss` exactly once when its dismiss control is activated.

**Interface under test:** `<ConfirmPieceChip guess={ConfidentGuess}
onDismiss={() => void} />`

**Files:**
- Create: `src/components/ConfirmPieceChip.tsx`
- Test: `src/components/ConfirmPieceChip.test.tsx`

- [x] **Step 1: Write the failing test**

```typescript
// src/components/ConfirmPieceChip.test.tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ConfirmPieceChip } from "./ConfirmPieceChip";

const guess = {
	pieceId: "chopin-nocturne-op9-no2",
	composer: "Chopin",
	title: "Nocturne Op. 9 No. 2",
	confidence: 0.92,
};

describe("ConfirmPieceChip", () => {
	it("names the guessed piece and dismisses exactly once", () => {
		const onDismiss = vi.fn();
		render(<ConfirmPieceChip guess={guess} onDismiss={onDismiss} />);

		expect(screen.getByText(/Nocturne Op\. 9 No\. 2/)).toBeInTheDocument();

		fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
		expect(onDismiss).toHaveBeenCalledTimes(1);
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/ConfirmPieceChip.test.tsx
```
Expected: FAIL — `Cannot find module './ConfirmPieceChip'`

- [x] **Step 3: Implement the minimum to make the test pass**

```tsx
// src/components/ConfirmPieceChip.tsx
import type { ConfidentGuess } from "../lib/piece-ladder";

interface ConfirmPieceChipProps {
	guess: ConfidentGuess;
	onDismiss: () => void;
}

/**
 * Step 2 of the piece ladder: a confident but unpicked guess, shown as a
 * dismissible banner over whichever practice surface is active. Dismissal is
 * one-way — resolvePieceLadderState never re-shows a dismissed guess.
 */
export function ConfirmPieceChip({ guess, onDismiss }: ConfirmPieceChipProps) {
	return (
		<div className="flex items-center justify-between gap-3 rounded-lg border border-border-subtle bg-surface-raised px-4 py-2">
			<p className="text-body-sm text-ink-primary">
				Looks like <span className="font-medium">{guess.title}</span> — is
				that right?
			</p>
			<button
				type="button"
				onClick={onDismiss}
				className="text-label-sm text-ink-tertiary underline"
				aria-label="Dismiss piece guess"
			>
				Dismiss
			</button>
		</div>
	);
}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/ConfirmPieceChip.test.tsx
```
Expected: PASS (1 test)

- [x] **Step 5: Commit**

```bash
git add src/components/ConfirmPieceChip.tsx src/components/ConfirmPieceChip.test.tsx && git commit -m "feat(practice-mode): ConfirmPieceChip"
```

---

### Task 7: `SessionEndedBanner` offers a one-tap resume

**Group:** C (parallel with Task 6, Task 8, Task 9)

**Behavior being verified:** the banner shows the soft-stop copy and calls
`onResume` exactly once when tapped.

**Interface under test:** `<SessionEndedBanner onResume={() => void} />`

**Files:**
- Create: `src/components/SessionEndedBanner.tsx`
- Test: `src/components/SessionEndedBanner.test.tsx`

- [x] **Step 1: Write the failing test**

```typescript
// src/components/SessionEndedBanner.test.tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SessionEndedBanner } from "./SessionEndedBanner";

describe("SessionEndedBanner", () => {
	it("shows the soft-stop state and resumes exactly once", () => {
		const onResume = vi.fn();
		render(<SessionEndedBanner onResume={onResume} />);

		expect(screen.getByText(/Session ended/i)).toBeInTheDocument();

		fireEvent.click(screen.getByRole("button", { name: /keep playing/i }));
		expect(onResume).toHaveBeenCalledTimes(1);
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/SessionEndedBanner.test.tsx
```
Expected: FAIL — `Cannot find module './SessionEndedBanner'`

- [x] **Step 3: Implement the minimum to make the test pass**

```tsx
// src/components/SessionEndedBanner.tsx
interface SessionEndedBannerProps {
	onResume: () => void;
}

/**
 * The soft auto-stop state at 60s of silence. This is presentation only:
 * nothing about the recording session, WebSocket, or mic changes underneath
 * it (see spec, "Why the auto-stop is UI-only") — onResume only dismisses
 * this banner and resets the silence clock.
 */
export function SessionEndedBanner({ onResume }: SessionEndedBannerProps) {
	return (
		<div className="flex flex-col items-center justify-center gap-4 rounded-lg border border-border-subtle bg-surface-raised px-6 py-8 text-center">
			<p className="text-body-md text-ink-primary">
				Session ended — keep playing?
			</p>
			<button
				type="button"
				onClick={onResume}
				className="rounded-full bg-accent px-5 py-2 text-body-sm text-on-accent"
			>
				Keep playing
			</button>
		</div>
	);
}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/SessionEndedBanner.test.tsx
```
Expected: PASS (1 test)

- [x] **Step 5: Commit**

```bash
git add src/components/SessionEndedBanner.tsx src/components/SessionEndedBanner.test.tsx && git commit -m "feat(practice-mode): SessionEndedBanner"
```

---

### Task 8: `PieceLessMode` composes the elapsed timer and the timeline strip

**Group:** C (parallel with Task 6, Task 7, Task 9)

**Behavior being verified:** given marks and an elapsed time, the component
renders the elapsed time as `m:ss` and passes the marks straight through to
`SessionTimelineStrip`.

**Interface under test:** `<PieceLessMode marks={readonly Mark[]}
durationSeconds={number} elapsedSeconds={number} isRecording={boolean} />`

**Files:**
- Create: `src/components/PieceLessMode.tsx`
- Test: `src/components/PieceLessMode.test.tsx`

- [x] **Step 1: Write the failing test**

```typescript
// src/components/PieceLessMode.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { resolveAnchor } from "../lib/mark";
import type { Mark } from "../lib/mark";
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
	it("shows elapsed time as m:ss and renders the timeline strip with the given marks", () => {
		render(
			<PieceLessMode
				marks={marks}
				durationSeconds={90}
				elapsedSeconds={65}
				isRecording={true}
			/>,
		);

		expect(screen.getByText("1:05")).toBeInTheDocument();
		expect(screen.getByTestId("session-timeline")).toBeInTheDocument();
		expect(
			screen.getByRole("button", { name: /Needs work: Pedaling/i }),
		).toBeInTheDocument();
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/PieceLessMode.test.tsx
```
Expected: FAIL — `Cannot find module './PieceLessMode'`

- [x] **Step 3: Implement the minimum to make the test pass**

```tsx
// src/components/PieceLessMode.tsx
import type { Mark } from "../lib/mark";
import { useMetronome } from "../hooks/useMetronome";
import { SessionTimelineStrip } from "./SessionTimelineStrip";

interface PieceLessModeProps {
	marks: readonly Mark[];
	durationSeconds: number;
	elapsedSeconds: number;
	isRecording: boolean;
}

function formatElapsed(totalSeconds: number): string {
	const minutes = Math.floor(totalSeconds / 60);
	const seconds = Math.floor(totalSeconds % 60);
	return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

/**
 * The permanent pieceless surface (docs/apps/05-ui-system.md#2): a calm,
 * near-empty screen. No score to hide behind means this component has no
 * logic of its own beyond formatting elapsed time — everything else is
 * SessionTimelineStrip, which is the complete canvas by design.
 */
export function PieceLessMode({
	marks,
	durationSeconds,
	elapsedSeconds,
	isRecording,
}: PieceLessModeProps) {
	const metronome = useMetronome();

	return (
		<div className="flex h-full flex-col items-center justify-between px-6 py-12">
			<div className="flex flex-1 flex-col items-center justify-center gap-2">
				{isRecording && (
					<span className="h-2 w-2 rounded-full bg-danger" aria-hidden="true" />
				)}
				<span className="text-display-md tabular-nums text-ink-primary">
					{formatElapsed(elapsedSeconds)}
				</span>
				<button
					type="button"
					onClick={metronome.toggle}
					className="text-label-sm text-ink-tertiary underline"
				>
					{metronome.isPlaying ? `Metronome ${metronome.bpm}` : "Metronome"}
				</button>
			</div>
			<div className="w-full max-w-2xl">
				<SessionTimelineStrip durationSeconds={durationSeconds} marks={marks} />
			</div>
		</div>
	);
}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/PieceLessMode.test.tsx
```
Expected: PASS (1 test)

- [x] **Step 5: Commit**

```bash
git add src/components/PieceLessMode.tsx src/components/PieceLessMode.test.tsx && git commit -m "feat(practice-mode): PieceLessMode"
```

---

### Task 9: `ScoreStand` pages a real score manually, with no live cursor

**Group:** C (parallel with Task 6, Task 7, Task 8)

**Behavior being verified:** given a `pieceId`, the stand loads the score,
shows page 1, advances/retreats on Next/Prev, and clamps at both ends —
without ever calling `ScoreCursor` (there is no live-following import at
all). Positional/overlap correctness is NOT asserted here (jsdom has no
layout engine per the spec's hard-won lesson); that is Task 14's job.

**Interface under test:** `<ScoreStand pieceId={string} marks={readonly
Mark[]} elapsedSeconds={number} isRecording={boolean} />`

**Files:**
- Modify: `src/lib/mark.ts` (export `formatElapsed`, reused instead of
  duplicated — see Task 8 note below)
- Create: `src/components/ScoreStand.tsx`
- Test: `src/components/ScoreStand.test.tsx`

Note on `formatElapsed`: Task 8 already defines a local copy inside
`PieceLessMode.tsx` because it has no other shared dependency on `mark.ts`.
This task's `ScoreStand` needs the identical formatting for its own elapsed
readout. Rather than adding a third copy, export the one that already exists
in `mark.ts` (`formatElapsed`, currently private) and use it here. Task 8's
local copy is intentionally left as-is — reconciling it is two extra lines
and not required by either task's test, and CLAUDE.md's "touch only lines
required by the task" favors leaving Task 8's file alone once it is merged.

- [x] **Step 1: Write the failing test**

```typescript
// src/components/ScoreStand.test.tsx
import { render, screen, waitFor } from "@testing-library/react";
import { fireEvent } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { scoreRenderer } from "../lib/score-renderer";
import { ScoreStand } from "./ScoreStand";

vi.mock("../lib/score-renderer", () => ({
	scoreRenderer: {
		load: vi.fn(),
		getPage: vi.fn(),
	},
}));

const TWO_PAGE_IR = {
	pieceId: "chopin-nocturne-op9-no2",
	verovioVersion: "test",
	pageWidth: 1000,
	pages: [
		{ pageN: 1, viewBox: "0 0 100 100", width: 100, height: 100, systemBboxes: [] },
		{ pageN: 2, viewBox: "0 0 100 100", width: 100, height: 100, systemBboxes: [] },
	],
	bars: [
		{
			barNumber: 1,
			measureOn: "m1",
			pageN: 1,
			bbox: { x: 0, y: 0, w: 0, h: 0 },
			noteIds: [],
			qstampStart: 0,
			qstampEnd: 4,
		},
		{
			barNumber: 9,
			measureOn: "m9",
			pageN: 2,
			bbox: { x: 0, y: 0, w: 0, h: 0 },
			noteIds: [],
			qstampStart: 32,
			qstampEnd: 36,
		},
	],
	notes: {},
};

describe("ScoreStand", () => {
	beforeEach(() => {
		vi.mocked(scoreRenderer.load).mockResolvedValue({
			ir: TWO_PAGE_IR,
			pageSvgs: ["<svg data-page='1'></svg>", "<svg data-page='2'></svg>"],
		});
		vi.mocked(scoreRenderer.getPage).mockImplementation(
			async (_pieceId, pageN) => `<svg data-page="${pageN}"></svg>`,
		);
	});

	it("loads page 1 first, then advances and retreats, clamped at both ends", async () => {
		render(
			<ScoreStand
				pieceId="chopin-nocturne-op9-no2"
				marks={[]}
				elapsedSeconds={0}
				isRecording={true}
			/>,
		);

		await waitFor(() =>
			expect(screen.getByTestId("score-stand-page")).toHaveAttribute(
				"data-current-page",
				"1",
			),
		);

		const prevButton = screen.getByRole("button", { name: /previous page/i });
		expect(prevButton).toBeDisabled();

		fireEvent.click(screen.getByRole("button", { name: /next page/i }));
		await waitFor(() =>
			expect(screen.getByTestId("score-stand-page")).toHaveAttribute(
				"data-current-page",
				"2",
			),
		);

		const nextButton = screen.getByRole("button", { name: /next page/i });
		expect(nextButton).toBeDisabled();

		fireEvent.click(screen.getByRole("button", { name: /previous page/i }));
		await waitFor(() =>
			expect(screen.getByTestId("score-stand-page")).toHaveAttribute(
				"data-current-page",
				"1",
			),
		);
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/ScoreStand.test.tsx
```
Expected: FAIL — `Cannot find module './ScoreStand'`

- [x] **Step 3: Implement the minimum to make the test pass**

First, in `src/lib/mark.ts`, change `function formatElapsed` to `export
function formatElapsed` (one-word change, no other lines in that file move).

Note: the SVG is injected imperatively via a ref, not through React's
`dangerouslySetInnerHTML` prop — matching the established pattern at
`src/scorehost/score-host.ts:382` and the (now-deleted) `RealScoreSection`
in `marks-preview.tsx`, which keeps the SVG in a sibling node React never
re-reconciles, so `ScoreMarkLayer`'s own measurement effect isn't racing a
React re-render of the same subtree.

```tsx
// src/components/ScoreStand.tsx
import { useEffect, useMemo, useRef, useState } from "react";
import { useMetronome } from "../hooks/useMetronome";
import { formatElapsed } from "../lib/mark";
import type { Mark } from "../lib/mark";
import type { BarIR } from "../lib/score-ir";
import { scoreRenderer } from "../lib/score-renderer";
import { ScoreMarkLayer } from "./ScoreMarkLayer";

interface ScoreStandProps {
	pieceId: string;
	marks: readonly Mark[];
	elapsedSeconds: number;
	isRecording: boolean;
}

/**
 * The digital music stand (docs/apps/05-ui-system.md#2): a static, manually
 * paged score. Deliberately does not import ScoreCursor — that class exists
 * to drive a moving highlight from a live qstamp source, which is exactly
 * the "live following" this surface forbids. Page turns are the only way
 * the rendered page changes.
 */
export function ScoreStand({
	pieceId,
	marks,
	elapsedSeconds,
	isRecording,
}: ScoreStandProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const svgHostRef = useRef<HTMLDivElement>(null);
	const [pageCount, setPageCount] = useState(0);
	const [currentPage, setCurrentPage] = useState(1);
	const [pageSvg, setPageSvg] = useState<string | null>(null);
	// Full BarIR, not the narrower BarLocator: pageN is what lets this
	// component scope ScoreMarkLayer (Canvas A, lossy by design) to only the
	// bars actually on screen -- a bar on another page has no rect to place a
	// mark against here regardless of what mark-placement.ts does with it.
	const [allBars, setAllBars] = useState<readonly BarIR[]>([]);
	const [error, setError] = useState<string | null>(null);
	const metronome = useMetronome();

	useEffect(() => {
		let cancelled = false;
		async function load() {
			const result = await scoreRenderer.load(pieceId);
			if (cancelled) return;
			if (result === "failed") {
				setError("Score failed to load");
				return;
			}
			setPageCount(result.ir.pages.length);
			setAllBars(result.ir.bars);
		}
		load();
		return () => {
			cancelled = true;
		};
	}, [pieceId]);

	const barsForCurrentPage = useMemo(
		() =>
			allBars
				.filter((b) => b.pageN === currentPage)
				.map((b) => ({ barNumber: b.barNumber, measureOn: b.measureOn })),
		[allBars, currentPage],
	);

	useEffect(() => {
		let cancelled = false;
		async function loadPage() {
			const svg = await scoreRenderer.getPage(pieceId, currentPage);
			if (cancelled) return;
			// Injected imperatively into a dedicated child node, matching
			// src/scorehost/score-host.ts:382: this keeps the SVG in a sibling of
			// ScoreMarkLayer so React never owns or re-reconciles Verovio's DOM,
			// and ScoreMarkLayer's own ResizeObserver-driven measurement effect
			// isn't racing a React commit of the same subtree.
			if (svgHostRef.current) svgHostRef.current.innerHTML = svg;
			setPageSvg(svg);
		}
		if (pageCount > 0) loadPage();
		return () => {
			cancelled = true;
		};
	}, [pieceId, currentPage, pageCount]);

	if (error) {
		return <p className="text-danger">{error}</p>;
	}

	return (
		<div className="flex h-full flex-col">
			<div className="flex shrink-0 items-center justify-between border-b border-border-subtle px-4 py-2">
				<div className="flex items-center gap-2">
					{isRecording && (
						<span className="h-2 w-2 rounded-full bg-danger" aria-hidden="true" />
					)}
					<span className="text-body-sm tabular-nums text-ink-secondary">
						{formatElapsed(elapsedSeconds)}
					</span>
				</div>
				<button
					type="button"
					onClick={metronome.toggle}
					className="text-label-sm text-ink-tertiary underline"
				>
					{metronome.isPlaying ? `Metronome ${metronome.bpm}` : "Metronome"}
				</button>
			</div>

			<div
				ref={containerRef}
				data-testid="score-stand-page"
				data-current-page={currentPage}
				className="score-container relative flex-1 overflow-auto"
			>
				<div ref={svgHostRef} />
				{pageSvg && (
					// ScoreMarkLayer renders absolute inset-0, so it must be a child of
					// this relative container, not a sibling after it -- a sibling would
					// anchor against the next positioned ancestor up the tree instead
					// (the flex column above), placing every mark at the wrong origin.
					<ScoreMarkLayer
						containerRef={containerRef}
						bars={barsForCurrentPage}
						marks={marks}
					/>
				)}
			</div>

			<div className="flex shrink-0 items-center justify-center gap-4 border-t border-border-subtle px-4 py-2">
				<button
					type="button"
					onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
					disabled={currentPage <= 1}
					className="rounded-full px-3 py-1 text-body-sm text-ink-secondary disabled:opacity-40"
					aria-label="Previous page"
				>
					Prev
				</button>
				<span className="text-body-xs text-ink-tertiary tabular-nums">
					{currentPage} / {pageCount || 1}
				</span>
				<button
					type="button"
					onClick={() => setCurrentPage((p) => Math.min(pageCount, p + 1))}
					disabled={currentPage >= pageCount}
					className="rounded-full px-3 py-1 text-body-sm text-ink-secondary disabled:opacity-40"
					aria-label="Next page"
				>
					Next
				</button>
			</div>
		</div>
	);
}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/ScoreStand.test.tsx
```
Expected: PASS (1 test)

- [x] **Step 5: Commit**

```bash
git add src/lib/mark.ts src/components/ScoreStand.tsx src/components/ScoreStand.test.tsx && git commit -m "feat(practice-mode): ScoreStand, static paginated score"
```

---

### Task 10: `PracticeMode` orchestrates the ladder, the pause tracker, and auto-stop

**Group:** D (depends on Task 4, Task 5, Task 6, Task 7, Task 8, Task 9)

**Behavior being verified:** for each combination of piece-ladder inputs and
silence duration, exactly the right sub-surface is showing: `ScoreStand` for
a known piece, `PieceLessMode` otherwise, `ConfirmPieceChip` layered on top
when a guess is pending, and `SessionEndedBanner` replacing everything once
auto-stopped, with resume bringing the prior surface back. A persistent
"Stop recording" control is present regardless of which sub-surface is
showing (including the auto-stopped banner) and calls `onStop` exactly once
when activated — this is the only way to leave the full-screen surface and
end the session for real; `SessionEndedBanner`'s resume is a separate,
non-terminal action (see spec, "Why the auto-stop is UI-only"). The Stop
control lives in a dedicated `shrink-0` header row above a `flex-1` content
region, not an absolute overlay — this jsdom test cannot see geometry, so
the actual non-overlap of Stop against ScoreStand's Metronome toggle and
ConfirmPieceChip's Dismiss button is verified in a real browser by Task 14,
not here.

**Interface under test:** `<PracticeMode userPickedPieceId={string | null}
confidentGuess={ConfidentGuess | null} marks={readonly Mark[]}
elapsedSeconds={number} isPlaying={boolean} isRecording={boolean}
onStop={() => void} />`

**Files:**
- Create: `src/components/PracticeMode.tsx`
- Test: `src/components/PracticeMode.test.tsx`

- [x] **Step 1: Write the failing test**

```typescript
// src/components/PracticeMode.test.tsx
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AUTO_STOP_SILENCE_MS } from "../lib/pause-state";
import { scoreRenderer } from "../lib/score-renderer";
import { PracticeMode } from "./PracticeMode";

vi.mock("../lib/score-renderer", () => ({
	scoreRenderer: {
		load: vi.fn().mockResolvedValue({
			ir: {
				pieceId: "p1",
				verovioVersion: "test",
				pageWidth: 1000,
				pages: [{ pageN: 1, viewBox: "0 0 1 1", width: 1, height: 1, systemBboxes: [] }],
				bars: [],
				notes: {},
			},
			pageSvgs: ["<svg></svg>"],
		}),
		getPage: vi.fn().mockResolvedValue("<svg></svg>"),
	},
}));

const guess = {
	pieceId: "chopin-nocturne-op9-no2",
	composer: "Chopin",
	title: "Nocturne Op. 9 No. 2",
	confidence: 0.92,
};

describe("PracticeMode", () => {
	beforeEach(() => {
		vi.useFakeTimers();
	});
	afterEach(() => {
		vi.useRealTimers();
	});

	it("shows PieceLessMode with no pick and no guess", () => {
		render(
			<PracticeMode
				userPickedPieceId={null}
				confidentGuess={null}
				marks={[]}
				elapsedSeconds={0}
				isPlaying={true}
				isRecording={true}
				onStop={vi.fn()}
			/>,
		);
		expect(screen.getByTestId("session-timeline")).toBeInTheDocument();
		expect(screen.queryByTestId("score-stand-page")).not.toBeInTheDocument();
	});

	it("shows ScoreStand plus a dismissible confirm chip for a confident guess", async () => {
		render(
			<PracticeMode
				userPickedPieceId={null}
				confidentGuess={guess}
				marks={[]}
				elapsedSeconds={0}
				isPlaying={true}
				isRecording={true}
				onStop={vi.fn()}
			/>,
		);
		expect(screen.getByText(/Nocturne Op\. 9 No\. 2/)).toBeInTheDocument();

		fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
		await vi.waitFor(() =>
			expect(screen.queryByText(/Nocturne Op\. 9 No\. 2/)).not.toBeInTheDocument(),
		);
	});

	it("shows the session-ended banner after 60s of silence, and resume dismisses it", () => {
		render(
			<PracticeMode
				userPickedPieceId={null}
				confidentGuess={null}
				marks={[]}
				elapsedSeconds={0}
				isPlaying={false}
				isRecording={true}
				onStop={vi.fn()}
			/>,
		);

		act(() => {
			vi.advanceTimersByTime(AUTO_STOP_SILENCE_MS);
		});
		expect(screen.getByText(/Session ended/i)).toBeInTheDocument();

		fireEvent.click(screen.getByRole("button", { name: /keep playing/i }));
		expect(screen.queryByText(/Session ended/i)).not.toBeInTheDocument();
	});

	it("calls onStop exactly once when the stop control is activated, even after auto-stop", () => {
		const onStop = vi.fn();
		render(
			<PracticeMode
				userPickedPieceId={null}
				confidentGuess={null}
				marks={[]}
				elapsedSeconds={0}
				isPlaying={false}
				isRecording={true}
				onStop={onStop}
			/>,
		);

		// Present before auto-stop...
		fireEvent.click(screen.getByRole("button", { name: /stop recording/i }));
		expect(onStop).toHaveBeenCalledTimes(1);

		// ...and still present once the session-ended banner has replaced the
		// rest of the surface -- stopping for real must not require resuming
		// first.
		act(() => {
			vi.advanceTimersByTime(AUTO_STOP_SILENCE_MS);
		});
		expect(screen.getByText(/Session ended/i)).toBeInTheDocument();
		fireEvent.click(screen.getByRole("button", { name: /stop recording/i }));
		expect(onStop).toHaveBeenCalledTimes(2);
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/PracticeMode.test.tsx
```
Expected: FAIL — `Cannot find module './PracticeMode'`

- [x] **Step 3: Implement the minimum to make the test pass**

```tsx
// src/components/PracticeMode.tsx
import { useState } from "react";
import { usePauseTracker } from "../hooks/usePauseTracker";
import type { Mark } from "../lib/mark";
import type { ConfidentGuess } from "../lib/piece-ladder";
import { resolvePieceLadderState } from "../lib/piece-ladder";
import { ConfirmPieceChip } from "./ConfirmPieceChip";
import { PieceLessMode } from "./PieceLessMode";
import { ScoreStand } from "./ScoreStand";
import { SessionEndedBanner } from "./SessionEndedBanner";

interface PracticeModeProps {
	userPickedPieceId: string | null;
	confidentGuess: ConfidentGuess | null;
	marks: readonly Mark[];
	elapsedSeconds: number;
	isPlaying: boolean;
	isRecording: boolean;
	/** Ends the session for real. The only exit from this full-screen surface;
	 * unlike SessionEndedBanner's resume, this is terminal. */
	onStop: () => void;
}

/**
 * The orchestrator: the one component that knows all four practice
 * sub-surfaces exist. Everything it delegates to (ScoreStand, PieceLessMode,
 * ConfirmPieceChip, SessionEndedBanner) takes plain props and touches
 * neither the WS nor the session hook directly -- AppChat is the only place
 * that wires usePracticeSession's live state into these props.
 *
 * The stop control is rendered here, not inside ScoreStand/PieceLessMode/
 * SessionEndedBanner, so it is guaranteed present across every ladder state
 * and across the auto-stopped banner -- a student must always have a way out
 * of the full-screen surface, and duplicating a stop button into every leaf
 * component would risk one of them (or a future fifth surface) forgetting
 * it.
 *
 * It lives in its own `shrink-0` header row, stacked in normal document flow
 * above a `flex-1` content region -- not an absolute overlay pinned to a
 * corner. Two of the three sub-surfaces put their own primary control in
 * that same top-right corner (ScoreStand's Metronome toggle, ConfirmPieceChip's
 * Dismiss button), so an absolute/z-indexed Stop button would sit in the same
 * box as one of them and could cover -- and steal clicks from -- whichever is
 * underneath (loop-3 challenge, blocker 5; the same "covered mark is
 * unclickable" failure class as #157). Reserving Stop its own row makes the
 * separation a layout guarantee instead of a stacking-order one: every
 * sub-surface's own header renders strictly below it, never behind it.
 */
export function PracticeMode({
	userPickedPieceId,
	confidentGuess,
	marks,
	elapsedSeconds,
	isPlaying,
	isRecording,
	onStop,
}: PracticeModeProps) {
	const [dismissed, setDismissed] = useState(false);
	const pause = usePauseTracker(isPlaying);

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
				<button
					type="button"
					onClick={onStop}
					aria-label="Stop recording"
					className="rounded-full bg-danger px-4 py-2 text-body-sm text-on-accent"
				>
					Stop
				</button>
			</div>
			<div className="flex min-h-0 flex-1 flex-col">
				{pause.autoStopped ? (
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

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/PracticeMode.test.tsx
```
Expected: PASS (4 tests)

- [x] **Step 5: Commit**

```bash
git add src/components/PracticeMode.tsx src/components/PracticeMode.test.tsx && git commit -m "feat(practice-mode): PracticeMode orchestrator"
```

---

### Task 11: `GREETINGS` dies

**Group:** E (depends on Task 10 only insofar as it shares a file with Task
12 — run this one first within the group; it touches a disjoint region of
`AppChat.tsx` and has its own commit)

**Behavior being verified:** the empty chat landing state no longer renders
a greeting headline, and the `GREETINGS` array is gone from the module.

**Interface under test:** `AppChat` render output (empty-conversation state)

**Files:**
- Modify: `src/components/AppChat.tsx`
- Test: `src/components/AppChat.greetings.test.tsx`

- [x] **Step 1: Write the failing test**

```typescript
// src/components/AppChat.greetings.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import AppChat from "./AppChat";

// AppChat pulls in auth/conversation queries; the existing test-setup mocks
// (matchMedia, ResizeObserver, IntersectionObserver) cover its render-time
// needs. This test only asserts on text content, not on network state.
vi.mock("../hooks/useAuth", () => ({
	authQueryOptions: { queryKey: ["auth"], queryFn: () => null },
	useAuth: () => ({ data: null, isLoading: false }),
}));

describe("AppChat empty state", () => {
	it("renders no GREETINGS headline", () => {
		render(<AppChat />);
		// None of the retired lines should appear anywhere in the document.
		expect(screen.queryByText("Let's make some music.")).not.toBeInTheDocument();
		expect(screen.queryByText("Your piano misses you.")).not.toBeInTheDocument();
	});
});
```

If `AppChat` requires additional provider wrapping to render in isolation
(e.g. a `QueryClientProvider` or router context) that this plan did not
anticipate, the implementing agent must add exactly the wrapping AppChat's
own existing conventions use elsewhere in the codebase (check for a shared
test-render helper before inventing one) — do not weaken the assertion to
compensate for a render failure.

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/AppChat.greetings.test.tsx
```
Expected: FAIL — the test finds `"Let's make some music."` (or whichever
`GREETINGS` entry the random pick lands on) still in the document, or the
run is flaky across executions because `GREETINGS` is chosen at random.
Either failure mode confirms the array is still live.

- [x] **Step 3: Implement the minimum to make the test pass**

In `src/components/AppChat.tsx`, delete the `GREETINGS` array (the block
starting `const GREETINGS = [` through its closing `];`), delete the
`greeting` `useMemo` block, and delete the `<h1 ...>{greeting}</h1>` element
from the empty-state JSX, leaving the icon and `ChatInput` as the only
children of that empty-state container.

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/AppChat.greetings.test.tsx
```
Expected: PASS (1 test), and re-running it several times in a row (`bunx
vitest run src/components/AppChat.greetings.test.tsx --repeat 5` or manual
repeats) never flips — proving the randomness is gone, not just unlucky.

- [x] **Step 5: Commit**

```bash
git add src/components/AppChat.tsx src/components/AppChat.greetings.test.tsx && git commit -m "feat(practice-mode): remove GREETINGS from AppChat empty state"
```

---

### Task 12: `AppChat` mounts `PracticeMode` instead of `ListeningMode`; dead files removed

**Group:** E (depends on Task 10; sequenced after Task 11 since both touch
`AppChat.tsx`)

**Behavior being verified:** starting a recording session mounts
`PracticeMode` (not `ListeningMode`), fed from `usePracticeSession`'s live
`marks`, `elapsedSeconds`, `isPlaying`, and `state`; `ListeningMode.tsx` and
`AudioWaveformRing.tsx` are deleted and nothing else in the app imports
them. `PracticeMode`'s `onStop` is wired to `practice.stop` plus the same
exit cleanup `ListeningMode`'s `onExit` used to perform (dismissing the
full-screen surface, clearing `recordButtonRect`, and navigating to a
newly-created conversation) — clicking "Stop recording" must actually end
the session, not just be present.

**Interface under test:** `AppChat` render output while `practice.state ===
"recording"`

**Files:**
- Modify: `src/components/AppChat.tsx`
- Delete: `src/components/ListeningMode.tsx`
- Delete: `src/components/AudioWaveformRing.tsx`
- Test: `src/components/AppChat.practicemode.test.tsx`

- [x] **Step 1: Write the failing test**

```typescript
// src/components/AppChat.practicemode.test.tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import AppChat from "./AppChat";

vi.mock("../hooks/useAuth", () => ({
	authQueryOptions: { queryKey: ["auth"], queryFn: () => null },
	useAuth: () => ({ data: null, isLoading: false }),
}));

// Module-scoped so the test can assert on calls after AppChat renders --
// vi.mock's factory below closes over this same reference.
const stop = vi.fn();

// Force usePracticeSession into a recording state so AppChat's branch that
// mounts the full-screen practice surface is reachable without a real mic,
// AudioContext, or WebSocket -- the same seam Task 5 and Task 9 already
// established for those layers individually.
vi.mock("../hooks/usePracticeSession", () => ({
	usePracticeSession: () => ({
		state: "recording",
		elapsedSeconds: 12,
		observations: [],
		latestScores: null,
		summary: null,
		error: null,
		chunksProcessed: 0,
		chunkStates: [],
		wsStatus: "connected",
		isOnline: true,
		isPlaying: true,
		energy: 0,
		analyserNode: null,
		practiceMode: null,
		marks: [],
		start: vi.fn(),
		stop,
		setPiece: vi.fn(),
		observationMessages: [],
		conversationId: null,
		activeLoop: null,
	}),
}));

describe("AppChat practice mode", () => {
	it("mounts PracticeMode's pieceless surface while recording, not the waveform ring", () => {
		render(<AppChat />);
		expect(screen.getByTestId("session-timeline")).toBeInTheDocument();
		expect(screen.queryByLabelText(/toggle metronome/i)).not.toBeInTheDocument();
	});

	it("stopping recording calls practice.stop and exits the practice surface", () => {
		render(<AppChat />);
		expect(screen.getByTestId("session-timeline")).toBeInTheDocument();

		fireEvent.click(screen.getByRole("button", { name: /stop recording/i }));

		expect(stop).toHaveBeenCalledTimes(1);
		// The surface itself unmounts once onStop's exit cleanup runs --
		// otherwise a student who taps Stop would still be staring at the
		// full-screen surface with no confirmation anything happened.
		expect(screen.queryByTestId("session-timeline")).not.toBeInTheDocument();
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/AppChat.practicemode.test.tsx
```
Expected: FAIL — `session-timeline` testid is absent because `AppChat`
still mounts `ListeningMode` (which renders `AudioWaveformRing`, not the
timeline strip) whenever `showListeningMode` is true.

- [x] **Step 3: Implement the minimum to make the test pass**

In `src/components/AppChat.tsx`:

1. Remove the imports `import { ListeningMode } from "./ListeningMode";` and
   the `AudioWaveformRing` import if present directly in this file (it is
   only imported transitively via `ListeningMode` per the earlier research —
   remove only what this file itself imports).
2. Add `import { PracticeMode } from "./PracticeMode";`.
3. Replace the `{showListeningMode && (<ListeningMode ... />)}` block (the
   one at line ~1063-1080 rendering `ListeningMode` with
   `onExit={handleExitListeningMode}` etc.) with:

```tsx
				{showListeningMode && (
					<div className="fixed inset-0 z-50 bg-surface-page">
						<PracticeMode
							userPickedPieceId={null}
							confidentGuess={null}
							marks={practice.marks}
							elapsedSeconds={practice.elapsedSeconds}
							isPlaying={practice.isPlaying}
							isRecording={practice.state === "recording"}
							onStop={handleStopPracticeMode}
						/>
					</div>
				)}
```

   `userPickedPieceId` and `confidentGuess` are wired as `null` for this
   task: `AppChat` has no existing state for a user-picked piece or a
   `piece_identified`-derived confident guess (the WS event exists —
   `case "piece_identified":` in `usePracticeSession.ts` currently only
   `console.log`s it and is not surfaced to `AppChat` at all). Wiring those
   two inputs for real is out of this task's test (which only asserts the
   pieceless branch renders) and is exactly the kind of follow-up this plan
   defers rather than silently guessing at — do not invent a piece-picker UI
   here.
4. Add a small wrapper next to `handleExitListeningMode` (same place in the
   file) that reproduces what `ListeningMode`'s old "Stop recording" button
   used to do — call `practice.stop()`, then run the existing exit cleanup:

```tsx
	function handleStopPracticeMode() {
		practice.stop();
		handleExitListeningMode();
	}
```

   `handleExitListeningMode` itself is unchanged — it already does the right
   thing (`setShowListeningMode(false)`, clear `recordButtonRect`, navigate
   to a newly-created conversation) and does not need `ListeningMode` to
   exist to keep doing it.
5. Delete `src/components/ListeningMode.tsx` and
   `src/components/AudioWaveformRing.tsx`.
6. Remove any other `ListeningMode`-only plumbing (e.g. `pieceContext`/
   `sessionNotes` state) only if `tsc` reports it unused after the
   deletion — leave anything still referenced elsewhere alone.
   `handleExitListeningMode` stays; it is now called from
   `handleStopPracticeMode`.

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/AppChat.practicemode.test.tsx && bunx tsc --noEmit
```
Expected: PASS (2 tests), `tsc` exits 0 (no leftover unused imports/dead
refs to `ListeningMode` or `AudioWaveformRing` anywhere).

- [x] **Step 5: Commit**

```bash
git add -A src/components/AppChat.tsx src/components/AppChat.practicemode.test.tsx src/components/ListeningMode.tsx src/components/AudioWaveformRing.tsx && git commit -m "feat(practice-mode): mount PracticeMode in AppChat; delete ListeningMode/AudioWaveformRing"
```

---

### Task 13: Delete the `/marks-preview` route and its fixture leak; add the `practice-preview` harness

**Group:** F (depends on Task 8, Task 9 — needs real `ScoreStand` and
`PieceLessMode` to exist)

**Behavior being verified:** `/marks-preview` no longer exists;
`practice-preview.tsx` renders `null` outside dev mode and, in dev mode,
mounts the real `ScoreStand` and `PieceLessMode` components with fixture
marks (ported from the deleted `mark-fixtures.ts`) — proving the fixture
data and the production components are wired together correctly, before
Task 14 checks their real-browser geometry.

**Interface under test:** the `practice-preview.tsx` route component's
render output, gated on `import.meta.env.DEV`

**Files:**
- Delete: `src/routes/marks-preview.tsx`
- Delete: `src/routes/marks-preview.test.tsx`
- Delete: `src/test-utils/mark-fixtures.ts`
- Create: `src/routes/practice-preview.tsx`
- Test: `src/routes/practice-preview.test.tsx`

- [x] **Step 1: Write the failing test**

```typescript
// src/routes/practice-preview.test.tsx
import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { scoreRenderer } from "../lib/score-renderer";

vi.mock("../lib/score-renderer", () => ({
	scoreRenderer: {
		load: vi.fn().mockResolvedValue("failed"),
		getPage: vi.fn(),
	},
}));

describe("PracticePreview", () => {
	afterEach(() => {
		vi.unstubAllEnvs();
		vi.resetModules();
	});

	it("renders nothing outside dev mode", async () => {
		vi.stubEnv("DEV", false);
		const { PracticePreview } = await import("./practice-preview");
		const { container } = render(<PracticePreview />);
		expect(container).toBeEmptyDOMElement();
	});

	it("renders the pieceless fixture surface in dev mode", async () => {
		vi.stubEnv("DEV", true);
		const { PracticePreview } = await import("./practice-preview");
		render(<PracticePreview />);
		expect(screen.getByTestId("session-timeline")).toBeInTheDocument();
	});
});
```

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/routes/practice-preview.test.tsx
```
Expected: FAIL — `Cannot find module './practice-preview'`

- [x] **Step 3: Implement the minimum to make the test pass**

Delete `src/routes/marks-preview.tsx`, `src/routes/marks-preview.test.tsx`,
and `src/test-utils/mark-fixtures.ts`.

```tsx
// src/routes/practice-preview.tsx
import { createFileRoute } from "@tanstack/react-router";
import { PieceLessMode } from "../components/PieceLessMode";
import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";

export const Route = createFileRoute("/practice-preview")({
	component: PracticePreview,
});

// Ported from the deleted src/test-utils/mark-fixtures.ts (#157). Inlined
// here rather than re-created as a shared test-utils module, because that
// module's whole defect was being importable from a production route in the
// first place -- fixture data now lives only where it is gated out of
// production builds.
//
// Both constants below are exported (not just the component) because
// tests/marks.spec.ts's "a mark sits at its share of the session duration"
// test (loop-2 challenge, blocker 4) needs to compute its expected pixel
// fraction from these same numbers instead of hardcoding a second, driftable
// copy of them. Import this module directly from the spec file rather than
// re-typing the values there.
export const PIECELESS_DURATION_SECONDS = 120;

export const FIXTURE_MARKS: readonly Mark[] = [
	{
		id: "fixture-1",
		anchor: resolveAnchor({ atSeconds: 30, alignmentQuality: 0 }),
		taxonomy: "needs_work",
		dimension: "pedaling",
		evidence: "pedal held through the bass change",
		lifecycle: "active",
	},
	{
		id: "fixture-2",
		anchor: resolveAnchor({ atSeconds: 75, alignmentQuality: 0 }),
		taxonomy: "strong",
		dimension: "phrasing",
		evidence: "the rubato in this phrase was well shaped",
		lifecycle: "improving",
	},
	{
		// 85% of PIECELESS_DURATION_SECONDS (102s of 120s). #157's deleted
		// fixture set had a mark at this fraction specifically because a mark
		// anchored there once ran 36px past the strip's right edge at every
		// viewport (see tests/marks.spec.ts's "every timeline mark stays inside
		// the strip" tests) -- a bug a fixture set clustered in the middle of
		// the timeline cannot exercise. Kept here so that regression guard
		// still has real near-the-edge input to check, per the loop-2 challenge
		// re-review's coverage risk.
		id: "fixture-3",
		anchor: resolveAnchor({ atSeconds: 102, alignmentQuality: 0 }),
		taxonomy: "missed_opportunity",
		dimension: "dynamics",
		evidence: "the closing diminuendo flattened out early",
		lifecycle: "active",
	},
];

/**
 * Dev-only real-browser harness for #158's successor to #157's
 * marks-preview. Renders null in a production build: import.meta.env.DEV
 * is statically replaced with `false` by Vite in that build, and Rollup's
 * dead-code elimination drops this entire branch -- including the fixture
 * import above, which has no other consumer -- rather than merely hiding it
 * behind a runtime check. See docs/specs/2026-08-07-practice-mode-design.md,
 * "The replacement harness does not repeat #157's bundle leak."
 *
 * playwright.marks.config.ts's webServer runs `vite dev`, not a production
 * build+preview, specifically so this route is still reachable when
 * tests/marks.spec.ts exercises it.
 */
export function PracticePreview() {
	if (!import.meta.env.DEV) return null;

	return (
		<div className="h-dvh">
			<PieceLessMode
				marks={FIXTURE_MARKS}
				durationSeconds={PIECELESS_DURATION_SECONDS}
				elapsedSeconds={90}
				isRecording={false}
			/>
		</div>
	);
}
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/routes/practice-preview.test.tsx
```
Expected: PASS (2 tests)

- [x] **Step 5: Commit**

```bash
git add -A src/routes/practice-preview.tsx src/routes/practice-preview.test.tsx src/routes/marks-preview.tsx src/routes/marks-preview.test.tsx src/test-utils/mark-fixtures.ts && git commit -m "feat(practice-mode): replace marks-preview with dev-gated practice-preview harness"
```

**Note for the build agent:** `routeTree.gen.ts` is TanStack Router's
generated file. Regenerate it as part of this task's commit
(`bunx tsr generate` or whatever script `package.json` defines for route
generation — check for a `routes` or `generate` script before assuming the
command) so `/marks-preview` is actually gone from the route tree and
`/practice-preview` is actually registered. If no such script exists, running
`bun run dev` once and stopping it regenerates the file as a side effect;
commit the resulting diff.

---

### Task 14: Real-browser geometry harness — marks render on the real score, contained and non-overlapping

**Group:** F (depends on Task 13)

**Behavior being verified:** in a real Chromium page (not jsdom), a mark
placed on a real Verovio-rendered score sits inside its score container
(`documentElement.scrollWidth === clientWidth`, no horizontal page scroll),
and no two timeline marks overlap and become unclickable — the same two
properties #157's `tests/marks.spec.ts` proved, now against the production
`ScoreStand`/`PieceLessMode` components instead of bespoke test markup.
Additionally (loop-3 challenge, blocker 5), this task mounts `PracticeMode`
itself — not just its `ScoreStand`/`PieceLessMode` leaves — with a confident
guess pending, and asserts that no two of its interactive controls overlap:
the orchestrator's own "Stop recording" button, `ScoreStand`'s Metronome
toggle, and `ConfirmPieceChip`'s Dismiss button all compete for the top of
the screen in that state, and jsdom's zero-size layout can't see whether
they collide. Separately, this task also relocates the two color-contrast checks that
`tests/a11y.spec.ts` used to run against `/marks-preview`: that route is
gone (Task 13), and `/practice-preview`'s successor route is deliberately
`import.meta.env.DEV`-gated, which `playwright.a11y.config.ts`'s
`webServer` (a production `bun run build && vite preview`) would evaluate
to `false` — visiting `/practice-preview` under that config would render an
empty page and axe would report zero violations against nothing, not proof
of anything. `tests/marks.spec.ts` already runs against a `vite dev` server
(this task switches it to one, see Step 1) where the route renders for
real, so mark-glyph contrast coverage moves here instead of being silently
dropped.

**Interface under test:** the rendered DOM of `/practice-preview` in a real
browser

**Files:**
- Modify: `apps/web/tests/marks.spec.ts`
- Modify: `apps/web/playwright.marks.config.ts`
- Modify: `apps/web/tests/a11y.spec.ts`

- [x] **Step 1: Write the failing test (repoint the existing spec)**

In `apps/web/playwright.marks.config.ts`, change the `webServer.command`
from `"bun run build && bunx vite preview --port 4173 --strictPort"` to
`"bunx vite dev --port 4173 --strictPort"` (dev server keeps
`import.meta.env.DEV` true, which `practice-preview.tsx` requires to render
anything).

In `apps/web/tests/marks.spec.ts`, replace every `page.goto("/marks-preview")`
with `page.goto("/practice-preview")`, and update the file's header comment
to name the new route instead of the old one. Leave the collision-detection
and containment assertions themselves untouched — they are testing a
DOM-structural property (`button[aria-expanded]` geometry), not anything
specific to the deleted fixtures. Two other assertions in the same file are
NOT purely structural and do need edits (loop-2 challenge, blockers 3 and 4
— both confirmed by reading the file and its references directly, not
assumed):

- **Blocker 3 fix.** The "a mark sits over its real measure on a real
  Verovio engraving" test (lines 133-195) locates its target via
  `page.locator("[data-testid='real-score']")` (line 149). That testid was
  only ever defined on the deleted `marks-preview.tsx`'s `RealScoreSection`
  wrapper; `ScoreStand` (Task 9) tags its equivalent container
  `data-testid="score-stand-page"` instead. Change this one locator line to
  `page.locator("[data-testid='score-stand-page']")`. Do not add a
  `real-score` testid to `ScoreStand` — that component and its own test
  (`ScoreStand.test.tsx`, Task 9) already commit to `score-stand-page`, and
  changing it now would edit an already-landed task's contract for no
  reason. Re-verified against `ScoreStand`'s actual DOM shape (Task 9's
  code): the container carrying `data-testid="score-stand-page"` is the
  same relatively-positioned element that hosts both the injected Verovio
  SVG (`svgHostRef`, containing `g.measure` elements once rendered) and
  `ScoreMarkLayer`'s absolutely-positioned glyph buttons as siblings inside
  it — so the test's subsequent `realScore.locator("g.measure")` and
  `realScore.locator("button[aria-expanded]")` lookups resolve exactly as
  they did against the deleted route's `real-score` wrapper. No other line
  in this test needs to change for this fix.

- **Blocker 4 fix.** The "a mark sits at its share of the session duration"
  test (lines 101-123) hardcodes `(64 / 360) * offset.stripWidth` with the
  comment "Fixture m1 is at 64s of 360s" — values from the deleted
  `mark-fixtures.ts`, not from Task 13's replacement `FIXTURE_MARKS`
  (`fixture-1`, the pedaling mark, is at 30s of a 120s
  `PIECELESS_DURATION_SECONDS`). Replace the hardcoded fraction: import
  `FIXTURE_MARKS` and `PIECELESS_DURATION_SECONDS` from
  `../src/routes/practice-preview` at the top of `marks.spec.ts`, find the
  mark whose `dimension` is `"pedaling"` in `FIXTURE_MARKS`, and assert
  against `(pedalingMark.anchor.atSeconds / PIECELESS_DURATION_SECONDS) *
  offset.stripWidth` instead of the two hardcoded literals. This makes the
  test derive its expectation from the same fixture data the route renders,
  so a future change to `FIXTURE_MARKS` cannot silently decouple the
  assertion from what it is asserting about, which is the exact defect
  being fixed here. (Both constants are plain data — a `readonly` array and
  a `number` — with no browser-only side effects at module-import time, so
  importing `practice-preview.tsx` from the Node-side Playwright spec is
  safe; confirm at build time that no import in that module's chain executes
  a browser API eagerly at module scope before relying on this.)

Then add two new tests to the same file,
porting the two `/marks-preview` cases out of `tests/a11y.spec.ts`:

```typescript
// Added to tests/marks.spec.ts
import AxeBuilder from "@axe-core/playwright";

// #157 added these two color-contrast cases to tests/a11y.spec.ts against
// /marks-preview; #158 deletes that route. practice-preview.tsx is
// import.meta.env.DEV-gated and playwright.a11y.config.ts serves a
// production build (DEV === false there), so the cases cannot move with
// the route name alone -- they have to run under a config where DEV is
// true, which is exactly what this file's webServer now is. This is the
// only place mark contrast is verified -- never assert it from vitest.
for (const theme of ["light", "dark"] as const) {
	test(`practice-preview has no color-contrast violations (${theme})`, async ({
		page,
	}) => {
		await page.goto("/practice-preview");
		await page.evaluate((t) => {
			document.documentElement.dataset.theme = t;
		}, theme);

		const results = await new AxeBuilder({ page })
			.withRules(["color-contrast"])
			.exclude("[data-axe-exempt]")
			.analyze();

		for (const v of results.violations) {
			for (const node of v.nodes) {
				console.log(`[${theme} /practice-preview] ${v.id} :: ${node.target.join(" ")}`);
			}
		}
		expect(results.violations).toEqual([]);
	});
}
```

In `apps/web/tests/a11y.spec.ts`, remove the two
`{ theme: ..., path: "/marks-preview" }` entries from `THEME_CASES` (back
down to the original two: light `/privacy`, dark `/signin`), and update the
file's header comment to say mark-glyph contrast coverage now lives in
`tests/marks.spec.ts` (run under `vite dev`, where `/practice-preview`
actually renders) instead of naming the deleted route.

- [x] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx playwright test --config playwright.marks.config.ts
```
Expected: FAIL — `/practice-preview` currently only mounts `PieceLessMode`
(Task 13's minimal implementation), so the score-overlay assertions (which
expect a real engraving with bar-anchored marks, matching #157's original
two-canvas coverage) find no `score-container`, no `[data-testid='score-stand-page']`,
and no bar-anchored `data-measure-on` buttons. The two new color-contrast
tests may pass or fail independently of that — they only need the
pieceless surface, which already renders — but run them anyway as part of
the same red baseline. Also confirm `bun run test:a11y` is red at this
point in the branch's history (it has been since Task 13 deleted
`/marks-preview`); this task's `a11y.spec.ts` edit is what turns it green
again.

- [x] **Step 3: Implement the minimum to make the test pass**

Extend `PracticePreview` in `src/routes/practice-preview.tsx` to also mount
`ScoreStand` with a real piece and at least one bar-anchored fixture mark,
alongside the existing `PieceLessMode` section, matching what
`marks-preview.tsx` covered before deletion (both the synthetic bars case
and the real-engraving case). Use the same real piece id already present
elsewhere in this codebase (`chopin-nocturne-op9-no2`, per the deleted
`RealScoreSection`'s comment: "a third the size" of the Ballade, chosen so
the measureOn chain survives a real engraving without a long load):

```tsx
	return (
		<div className="h-dvh">
			<div className="h-1/3 border-b border-border-subtle">
				<ScoreStand
					pieceId="chopin-nocturne-op9-no2"
					marks={SCORE_FIXTURE_MARKS}
					elapsedSeconds={30}
					isRecording={false}
				/>
			</div>
			<div className="h-1/3 border-b border-border-subtle">
				<PieceLessMode
					marks={FIXTURE_MARKS}
					durationSeconds={PIECELESS_DURATION_SECONDS}
					elapsedSeconds={90}
					isRecording={false}
				/>
			</div>
			<div className="h-1/3" data-testid="practice-mode-preview">
				<PracticeMode
					userPickedPieceId={null}
					confidentGuess={CONFIRM_CHIP_GUESS}
					marks={SCORE_FIXTURE_MARKS}
					elapsedSeconds={30}
					isPlaying={true}
					isRecording={true}
					onStop={() => {}}
				/>
			</div>
		</div>
	);
```

Add the corresponding imports (`import { ScoreStand } from
"../components/ScoreStand";`, `import { PracticeMode } from
"../components/PracticeMode";`, and `import type { ConfidentGuess } from
"../lib/piece-ladder";`), a `SCORE_FIXTURE_MARKS` constant, and a
`CONFIRM_CHIP_GUESS: ConfidentGuess` constant
(`{ pieceId: "chopin-nocturne-op9-no2", composer: "Chopin", title:
"Nocturne Op. 9 No. 2", confidence: 0.92 }`, the same shape already used in
`PracticeMode.test.tsx`'s `guess` fixture).

The third section exists specifically so the real-browser geometry check
below (blocker 5) mounts `PracticeMode` itself, with a confident guess
pending, rather than only its leaf components — with `userPickedPieceId:
null` and a non-null `confidentGuess`, `resolvePieceLadderState` resolves to
`"confirm-chip"`, so `ConfirmPieceChip` renders above `ScoreStand` inside
this section, putting all three of the orchestrator's top-of-screen controls
(Stop, Metronome toggle, Dismiss) on screen at once — exactly the
combination #157-class overlap bugs hide in and unit tests (which run in
jsdom, with zero-size layout) cannot catch.

**This mark must be bar-anchored, not timestamp-only.** The retained "a mark
sits over its real measure on a real Verovio engraving" test (blocker 3
above) asserts a `data-measure-on` attribute on the glyph and matches it
against a real `g.measure` element — but `mark-placement.ts`'s `placeMarks`
only ever places marks whose `anchor.type === "bars"`; a `"timestamp"`
anchor is always routed to `unplaced` and never produces a
`data-measure-on` glyph inside `ScoreMarkLayer` at all (confirmed by reading
`placeMarks` in `src/lib/mark-placement.ts`). A purely time-anchored fixture
mark would leave `ScoreStand` rendering zero placed glyphs, and the
Verovio-engraving test would then fail at
`await expect(glyph).toBeVisible()` even after the blocker-3 locator fix
above — so time-anchoring here is not a safe simplification, it silently
reintroduces the same failure through a different assertion in the same
test.

`RealScoreSection` (the deleted route) worked around this by loading the
real IR locally and anchoring to `bars[0].barNumber` once bars were known —
`ScoreStand` doesn't expose its loaded bars to a parent, so that exact
pattern isn't available here without widening Task 9's interface, which is
out of scope for this fix. Use `bars: [1, 1]` instead: `score-ir.ts` assigns
`barNumber: idx + 1` when building `ScoreIR.bars` (confirmed by reading
`src/lib/score-ir.ts:243`), so bar 1 is guaranteed to exist, to be the first
bar of the piece, and to be on page 1 for any score with at least one
measure — no dependency on Verovio's rendered measure-numbering or on
`ScoreStand` having finished loading before the mark is constructed.

```typescript
const SCORE_FIXTURE_MARKS: readonly Mark[] = [
	{
		id: "score-fixture-1",
		anchor: resolveAnchor({
			atSeconds: 20,
			bars: [1, 1],
			alignmentQuality: 1,
		}),
		taxonomy: "needs_work",
		dimension: "pedaling",
		evidence: "pedal held through the bass change",
		lifecycle: "active",
	},
];
```

Also add one more test to `apps/web/tests/marks.spec.ts` — the blocker-5
fix — mirroring the existing "no timeline mark covers another" test's
vertical-range collision math (lines 15-49) against the three named
top-of-screen controls instead of the timeline's mark glyphs:

```typescript
// Added to tests/marks.spec.ts
test("PracticeMode's Stop control never overlaps a sub-surface's own top control", async ({
	page,
}) => {
	await page.goto("/practice-preview");

	// Scoped to the practice-mode-preview section specifically: it is the one
	// place a confident guess is pending on a known piece, so all three
	// controls that compete for the top of the screen are on screen at once
	// (loop-3 challenge, blocker 5) -- the orchestrator's own Stop button,
	// ScoreStand's Metronome toggle (rendered below it, same corner), and
	// ConfirmPieceChip's Dismiss button (rendered above ScoreStand, same
	// corner again).
	const scope = page.locator("[data-testid='practice-mode-preview']");
	const boxes = await scope.evaluate((root) => {
		const find = (label: RegExp) =>
			[...root.querySelectorAll("button")].find((b) =>
				label.test(b.getAttribute("aria-label") ?? b.textContent ?? ""),
			);
		const named = [
			["stop", find(/stop recording/i)],
			["dismiss", find(/dismiss/i)],
			["metronome", find(/metronome/i)],
		] as const;
		return named
			.filter((entry): entry is [string, Element] => entry[1] != null)
			.map(([name, el]) => {
				const r = el.getBoundingClientRect();
				// Same collision math as the timeline-mark test above: vertical
				// RANGES, not `top` equality -- Stop, Metronome, and Dismiss are
				// not the same height, so an equality check would miss exactly
				// the overlap this test exists to catch.
				return { name, l: r.left, r: r.right, t: r.top, b: r.bottom };
			});
	});

	// All three controls must actually be present and found by name -- a
	// missing control here would silently pass the collision loop below with
	// nothing to check.
	expect(boxes.map((b) => b.name).sort()).toEqual(["dismiss", "metronome", "stop"]);

	const collisions: string[] = [];
	for (let i = 0; i < boxes.length; i++) {
		for (let j = i + 1; j < boxes.length; j++) {
			const a = boxes[i];
			const b = boxes[j];
			if (a.t < b.b && b.t < a.b && a.l < b.r && b.l < a.r) {
				collisions.push(`${a.name} <-> ${b.name}`);
			}
		}
	}
	expect(collisions).toEqual([]);
});
```

- [x] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx playwright test --config playwright.marks.config.ts && bun run test:a11y
```
Expected: PASS — every test in `tests/marks.spec.ts` (including the two
ported color-contrast cases and the new `PracticeMode` control-collision
test), and `test:a11y` green again (2/2: light `/privacy`, dark `/signin`).

- [x] **Step 5: Commit**

```bash
git add apps/web/tests/marks.spec.ts apps/web/playwright.marks.config.ts apps/web/src/routes/practice-preview.tsx apps/web/tests/a11y.spec.ts && git commit -m "test(practice-mode): port real-browser mark geometry harness to practice-preview; move mark-contrast coverage off the deleted marks-preview route; assert PracticeMode's Stop control never overlaps a sub-surface control"
```

---

## Final Verification (run after all tasks land)

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bun run test && bunx tsc --noEmit && bun run lint && bun run test:a11y && bun run test:marks
```

Expected: `test` green (existing 246 tests plus every new test above);
`tsc` exit 0; `lint` 0 errors, warnings/infos at or below the accepted
baseline (107/23) — a new file introducing a new warning is a regression,
not "close enough"; `test:a11y` 2/2 (`/privacy`, `/signin` — mark-glyph
contrast coverage now lives in `test:marks`, per Task 14); `test:marks`
green against the ported harness, including its two color-contrast cases and
the `PracticeMode` control-collision test (loop-3 challenge, blocker 5).

Then the manual click-through from the issue's success criterion, performed
by a human (this is human-lit per `apps/CLAUDE.md` — "manual click-through
verdicts are human-lit"):
1. With a piece — no piece-picker exists yet: `usePracticeSession`'s
   `piece_identified` WS handler only `console.log`s free-text `composer`/
   `title` and never yields a catalog `pieceId`, and Task 12 hardcodes
   `userPickedPieceId={null}` and `confidentGuess={null}` in `AppChat` with
   no wiring path anywhere in this plan (see spec, "Not in scope"). The
   `user-picked` and `confirm-chip` ladder rungs are therefore unreachable
   from a live Record session until #160 lands the piece-picker. Perform
   this check against Task 14's dev-only `/practice-preview` harness
   instead (`bun run dev`, navigate to `/practice-preview`): score stand
   shows with its fixture mark on the right bar, Prev/Next page turns work,
   and the `PracticeMode` section (confident guess pending) shows
   `ConfirmPieceChip` above the score stand with the Stop/Metronome/Dismiss
   controls not overlapping.
2. Record pieceless — timeline strip accrues injected marks the same way.
3. Silence past `AUTO_STOP_SILENCE_MS` — hard to do at 60s in real time
   without a config surface; verify by temporarily lowering
   `AUTO_STOP_SILENCE_MS` in a local, uncommitted edit for the click-through
   only, or by confirming `usePauseTracker`'s already-green fake-timer tests
   are a faithful stand-in and documenting that the constant was not
   independently re-verified in real time. Do not commit a temporary
   threshold change.

---

## Challenge Review

### CEO Pass

**Premise Challenge.** The problem is real: `ListeningMode` +
`AudioWaveformRing` are chat-era artifacts that violate the epic's approved
"no live following, silent while playing" design, and `/marks-preview`
genuinely leaks `mark-fixtures.ts` into the production bundle today
(`src/routes/marks-preview.tsx` is a real `createFileRoute`, not
dev-gated — confirmed by reading the file). Replacing both in one pass,
sharing the `ScoreStand`/`PieceLessMode` build across the live app and the
geometry harness, is the direct path and matches the spec's stated
"Not in scope" boundaries. No dramatically simpler framing is visible.

**Existing coverage.** The plan correctly reuses `scoreRenderer`,
`ScoreMarkLayer`, `SessionTimelineStrip`, `useMetronome`, and
`resolveAnchor`/`Mark` from #157/#163 rather than re-inventing them. Good
match to CLAUDE.md's "extend existing patterns" rule.

**Scope check.** 14 tasks, ~9 new files plus edits to 3 existing ones — over
the plan's own 8-file complexity-smell threshold, but the fan-out is mostly
one-behavior-per-file (pure functions, single-purpose components), which is
the deep-module norm this codebase already uses (`mark.ts`, `mark-placement.ts`,
`timeline-lanes.ts` are precedents). Not flagged as bloat on its own.

**12-Month Alignment.**
```
CURRENT STATE                    THIS PLAN                        12-MONTH IDEAL
ListeningMode+                   PracticeMode orchestrator +       Score-first surface with
AudioWaveformRing,               ScoreStand/PieceLessMode,          real piece-ladder wiring,
GREETINGS random headline,       marks wired to a WS event          live mark pipeline, and
marks-preview bundle leak        no server yet emits, dev-gated     session review (#159/#162)
                                  practice-preview harness           built on this surface
```
Moves toward the ideal. The one drift risk: this plan deletes the app's only
recording-stop UI without replacing it (see BLOCKER below), which is a step
*backward* on basic usability even while the visual redesign moves forward.

**Alternatives.** The spec documents its one real alternative (`ScorePanel`/
`ScoreCursor` reuse) and rejects it for a concrete, verified reason
(`ScoreCursor` drives a live-following cursor, which the design forbids).
No `[QUESTION]` needed here — this satisfies the bar.

### Engineering Pass

**Architecture / data flow.**
```
AppChat (usePracticeSession: marks, elapsedSeconds, isPlaying, state)
   -> PracticeMode (resolvePieceLadderState, usePauseTracker)
        -> ScoreStand (scoreRenderer.load/getPage, ScoreMarkLayer)  [known piece]
        -> PieceLessMode (SessionTimelineStrip)                     [pieceless]
        -> ConfirmPieceChip (dismiss-only, layered)
        -> SessionEndedBanner (resume-only, replaces everything)
```
Verified against actual code: `PracticeWsEvent`'s existing union shape
(`src/lib/practice-api.ts:54-112`), `usePracticeSession`'s `handleWsMessage`
switch (`src/hooks/usePracticeSession.ts:211-346`, no `default` case, so
adding `case "mark"` is safe and non-breaking), `ScoreRenderer.load`'s
`{ir, pageSvgs} | "failed"` return shape and `getPage` signature
(`src/lib/score-renderer.ts:102-162`), and `BarIR`'s `pageN`/`measureOn`
fields (`src/lib/score-ir.ts`) all match what the plan's code assumes.

**[BLOCKER] (confidence: 9/10)** — Task 12 deletes the app's only
recording-stop control and does not replace it anywhere in the new
component tree. Today, `AppChat.tsx:1063-1083` mounts `ListeningMode` with
`onStop={practice.stop}` and `onExit={handleExitListeningMode}`;
`ListeningMode.tsx` renders an explicit "Stop recording" button
(`aria-label="Stop recording"`, `ListeningMode.tsx:315-323`) that calls
`onStop` then animates out via `onExit`. `practice.stop` is referenced
exactly once in `AppChat.tsx` (line 1070) — the block Task 12 replaces.
The plan's replacement JSX mounts `<PracticeMode userPickedPieceId={null}
confidentGuess={null} marks={...} elapsedSeconds={...} isPlaying={...}
isRecording={...} />` with no `onStop`/`onExit` prop, and none of
`PracticeMode`, `ScoreStand`, `PieceLessMode`, `ConfirmPieceChip`, or
`SessionEndedBanner` (Tasks 6-10) declare such a prop in their interfaces.
`SessionEndedBanner` (Task 7) only exposes `onResume`, not a "stop for
real" action, and per the spec ("Why the auto-stop is UI-only") resuming
never stops the session either. After this plan lands, a student who taps
record has no UI path to stop recording or leave the full-screen surface —
not even after the 60s auto-stop banner, which is deliberately non-stopping.
This is a basic usability regression the plan's own tests never catch,
because no task's test asserts a stop/exit affordance exists. Before
execution: add an explicit stop/exit control to `PracticeMode` (or one of
its sub-surfaces) wired to `practice.stop`, and add a task-level test that
clicking it actually ends the session — do not ship a full-screen surface
with no way out.

**[BLOCKER] (confidence: 9/10)** — Task 13 deletes `/marks-preview` but
never updates `apps/web/tests/a11y.spec.ts`, which hardcodes
`{ theme: "light", path: "/marks-preview" }` and
`{ theme: "dark", path: "/marks-preview" }` (`tests/a11y.spec.ts:18-19`) as
two of its four color-contrast cases, with a comment explaining these are
"the only place mark contrast is actually verified — never assert it from
vitest." Neither the plan's File Changes table nor Tasks 13/14 touch this
file. After Task 13, `bun run test:a11y` navigates to a route that no
longer exists, so the gate this plan is required to keep green (per the
binding constraints and the plan's own Final Verification section) breaks.
This compounds with a second defect even if someone naively repoints the
path to `/practice-preview`: `playwright.a11y.config.ts`'s `webServer` runs
`bun run build && vite preview` — a real production build, where
`import.meta.env.DEV` is `false` — so `PracticePreview` renders `null`
there by design (per the plan's own Task 13 code and the spec's "does not
repeat #157's bundle leak" section). An a11y run against `/practice-preview`
under that config would silently check an empty page and report zero
violations, which is exactly the "green test, not a working feature"
failure mode the working context calls out for #157's overflow bug getting
past four gates. Before execution: add a task step that either (a) updates
`tests/a11y.spec.ts` to a route/config that actually renders `ScoreStand`/
`PieceLessMode` under axe (e.g. mounting `practice-preview` via the `vite
dev`-backed marks config, or adding a production-reachable a11y fixture
route), or (b) explicitly documents why mark-glyph contrast coverage is
being dropped and gets that accepted as a deliberate scope cut — not
silently lost as a side effect of Task 13's file deletions.

**Module Depth Audit.**
| Module | Interface | Implementation | Verdict |
|---|---|---|---|
| `pause-state.ts` | 1 fn, 2 consts | ~15 LOC boundary arithmetic | DEEP |
| `usePauseTracker.ts` | 1 hook, 4-field return | ref + 2 effects + interval | DEEP |
| `piece-ladder.ts` | 1 fn, 2 types | 4-line precedence check | DEEP (thin but non-trivial precedence rule, matches spec's own framing) |
| `ScoreStand.tsx` | 1 component, 4 props | ~140 LOC: load/page effects, clamping, mark-layer wiring | DEEP |
| `PieceLessMode.tsx` | 1 component, 4 props | ~35 LOC, mostly composition | SHALLOW by design — spec says so explicitly ("intentionally the shallowest module... it has no logic of its own to hide"); acceptable, not a smell here |
| `ConfirmPieceChip.tsx` / `SessionEndedBanner.tsx` | 1 component each, 2 props | ~15-25 LOC presentational | SHALLOW but each replaces real duplicated markup that would otherwise live inline in `PracticeMode` — acceptable |
| `PracticeMode.tsx` | 1 component, 6 props | ladder + pause-tracker orchestration, ~50 LOC | DEEP |

No blocking shallow-module findings; `PieceLessMode`/`ConfirmPieceChip`/
`SessionEndedBanner` are shallow but the spec names this as an intentional
tradeoff, and CLAUDE.md's "simplicity first" favors small dedicated files
here over cramming three unrelated concerns into `PracticeMode.tsx`.

**Code quality / edge cases.**
- `formatElapsed` is duplicated between `mark.ts` (exported per Task 9) and
  a second private copy inline in `PieceLessMode.tsx` (Task 8). The plan
  names this explicitly and defers reconciling it, citing "touch only lines
  required by the task." Acceptable per CLAUDE.md, but it is a real,
  named DRY violation the plan chose not to fix — RISK, not blocker.
- **[RISK] (confidence: 6/10)** — `ScoreStand`'s page-turn effect
  (`scoreRenderer.getPage(pieceId, currentPage)`, Task 9 Step 3) re-fetches
  every page from the Verovio worker on every Prev/Next click, even though
  `scoreRenderer.load()` already returned the full `pageSvgs: string[]`
  array for every page in one call. `marks-preview.tsx`'s
  `RealScoreSection` only ever loaded page 1, so this redundant-refetch
  pattern was never exercised at more than one page before. Watch for
  janky page turns on a slow connection; fallback is trivial (index into
  the already-fetched `pageSvgs` from `load()` instead of re-calling
  `getPage`), but as written this is an unnecessary worker round-trip per
  page turn that the plan's own test won't catch (the test mocks
  `scoreRenderer.getPage` to resolve instantly).
- No unhandled catch-alls or swallowed exceptions found in the new code —
  `scoreRenderer.load` already converts failures to a `"failed"` sentinel
  handled explicitly by `ScoreStand`.

**Test philosophy / vertical slice / coverage.** Every task is one test,
written first and shown failing for a concrete reason, then one minimal
implementation, then one commit — no horizontal slicing found. Tests
exercise public component/hook interfaces (render + fireEvent/waitFor,
hook return values), not internals; no test mocks an *internal* collaborator
of the unit under test (the WS/AudioContext/MediaRecorder fakes in Task 5
and Task 12 are external-boundary mocks, which is the accepted pattern
per the constraints). Per the working-context's jsdom-geometry rule: Tasks
9, 8, and 10 explicitly avoid positional assertions in jsdom and defer
geometry to Task 14's real-browser harness — this plan does **not** commit
the #157 regression of asserting layout in jsdom.

**[RISK] (confidence: 5/10)** — Task 13's routeTree.gen.ts regeneration
step is a manual, easy-to-forget instruction ("run `bun run dev` once and
stopping it... commit the resulting diff") rather than a verified script
(`package.json` has no `routes`/`generate` script, confirmed). If skipped,
`/marks-preview` stays registered and `/practice-preview` never becomes
reachable, silently breaking Task 14's harness without any test catching it
until `bun run test:marks` 404s. Fallback: the build agent should run
`bunx tsr generate` (or the dev-server trick) and diff `routeTree.gen.ts`
before committing Task 13, and Task 14's own red-test run (`bunx playwright
test`) will catch a missed regeneration if it happens.

**Failure modes.** The mark-WS-plumbing path fails closed correctly: no
server emits `mark` today, so `marks` stays `[]` and the screen shows
nothing — the spec calls this out explicitly as the intended "silence, not
guessing" behavior, matching the epic's failure-mode principle. Auto-stop
is UI-only by design and reversible with one tap, matching the approved
design. The known accepted scope boundary (WS messages injected manually
via devtools for the click-through, no live pipeline) is verified honestly:
the plan's own Verification Architecture section states this outright
rather than dressing it up as an automated pipeline test — no dishonesty
found there.

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| `PracticeWsEvent`'s switch has no `default` case, so adding `mark` is non-breaking | SAFE | Verified by reading `usePracticeSession.ts:211-346` |
| `scoreRenderer.load` never throws, always resolves `"failed"` on error | SAFE | Verified in `score-renderer.ts:117-128` |
| No other file imports `ListeningMode`/`AudioWaveformRing` besides `AppChat.tsx` | SAFE | Verified via repo-wide grep |
| `practice.stop` has a replacement UI path after Task 12 | RISKY | Verified false — no replacement exists (see BLOCKER above) |
| `tests/a11y.spec.ts` doesn't need updating because Task 13/14 only mention `marks.spec.ts` | RISKY | Verified false — `a11y.spec.ts` hardcodes `/marks-preview` twice (see BLOCKER above) |
| Vite/Rollup DCE actually drops the `import.meta.env.DEV` branch and its fixture import in production builds | SAFE | Standard, well-documented Vite/Rollup behavior; spec's reasoning is technically sound |
| `bun run dev` regenerates `routeTree.gen.ts` as a side effect | VALIDATE | Plausible for TanStack Router's Vite plugin but not directly confirmed by reading `vite.config.ts`'s plugin list in this review |

### Summary
[BLOCKER] count: 2
[RISK]    count: 3
[QUESTION] count: 0

VERDICT: NEEDS_REWORK — (1) no replacement for the deleted recording-stop control, (2) `tests/a11y.spec.ts` still points at the deleted `/marks-preview` route and, even if repointed, would silently pass against a blank page under the a11y config's production-build webServer.

---

## Challenge Review (re-review, attempt 2, commit 44dc6198)

### Verification of the two prior blockers

**Blocker 1 (missing stop control) — RESOLVED.** Verified by reading the
edited tasks against current code: `AppChat.tsx:1070` currently passes
`onStop={practice.stop}` to `ListeningMode`, and `ListeningMode.tsx:321`
renders a real `aria-label="Stop recording"` button — confirming the plan's
premise about what's being removed. Task 10's `PracticeMode` now declares a
required `onStop: () => void` prop and renders a persistent
`aria-label="Stop recording"` button (plan lines ~1670-1677) above whichever
sub-surface is showing, including `SessionEndedBanner`, with a task-level
test (`PracticeMode.test.tsx`, "calls onStop exactly once ... even after
auto-stop") asserting it fires both before and after the 60s banner
replaces the rest of the surface. Task 12 wires `onStop={handleStopPracticeMode}`,
a new function that calls `practice.stop()` then the existing
`handleExitListeningMode()`, with its own test
(`AppChat.practicemode.test.tsx`, "stopping recording calls practice.stop
and exits the practice surface") asserting the mock `stop` fires and the
surface unmounts. This holds — there is now a stop/exit affordance present
in every ladder state and the auto-stopped state, and it is exercised by
tests at both the component and integration level.

**Blocker 2 (a11y route hardcoding `/marks-preview`) — RESOLVED for its
originally-scoped complaint, but the underlying test file it moves
content into has independent, unaddressed defects (see new BLOCKERs
below).** Verified against the live files: `tests/a11y.spec.ts:18-19`
today still hardcodes `{ theme: "light"/"dark", path: "/marks-preview" }`,
and `playwright.a11y.config.ts:11` runs `bun run build && vite preview` — a
real production build where `import.meta.env.DEV` is `false`, confirming
both halves of the original defect. Task 14 now removes those two cases
from `a11y.spec.ts` (back to the original two: `/privacy`, `/signin`) and
ports the same two color-contrast checks into `tests/marks.spec.ts`, which
Task 14 also repoints to a `vite dev` webServer (`playwright.marks.config.ts`
Step 1) so `import.meta.env.DEV` stays `true` and `/practice-preview`
actually renders. This resolves the specific complaint: `test:a11y` no
longer points at a deleted route, and the relocated cases don't silently
pass against a blank page. It is a genuine, verified fix of blocker 2 as
stated. However, re-reading `tests/marks.spec.ts` in full (not just the two
lines Task 14 adds) surfaces two new, concrete breaks in that same file —
see below.

### New findings from a fresh full read of `tests/marks.spec.ts`

[BLOCKER] (confidence: 9/10) — `tests/marks.spec.ts`'s "a mark sits over
its real measure on a real Verovio engraving" test (lines 133-195) locates
its target via `page.locator("[data-testid='real-score']")` (line 149).
That testid is defined only on `src/routes/marks-preview.tsx:111`, inside
the now-deleted `RealScoreSection` wrapper — confirmed by
`grep -rn "real-score" src/ tests/` returning exactly those two hits
(the definition and this one locator). Task 9's `ScoreStand` component
(the thing Task 14 mounts in its place) tags its container
`data-testid="score-stand-page"`, not `real-score`, and neither Task 13's
nor Task 14's text adds a `real-score` testid anywhere in the new
component tree or updates this locator. Task 14's instruction to "leave the
collision-detection and containment assertions themselves untouched"
explicitly preserves this exact line. After Task 14 repoints
`page.goto("/marks-preview")` to `/practice-preview"`, `realScore` resolves
to zero elements, so `await expect(measures.first()).toBeVisible({timeout:
90000})` (line 152) times out — a hard, ~90s-to-discover failure of
`test:marks`, one of the plan's own required regression gates (binding
constraints list `bun run test:marks` explicitly). Before execution: add a
step to Task 13 or Task 14 that either tags `ScoreStand`'s real-score wrapper
`data-testid="real-score"` (matching the deleted route's contract) or
updates this locator in `marks.spec.ts` to `[data-testid='score-stand-page']`
— and re-verify the rest of that test still resolves (`g.measure` lookup,
`data-measure-on` glyph attribute) against `ScoreStand`'s actual DOM shape.

[BLOCKER] (confidence: 9/10) — `tests/marks.spec.ts`'s "a mark sits at its
share of the session duration" test (lines 101-123) hardcodes an expected
position of `(64 / 360) * offset.stripWidth` (line 122), with the comment
"Fixture m1 is at 64s of 360s." Those numbers come from the deleted
`src/test-utils/mark-fixtures.ts`, confirmed by reading it directly:
`FIXTURE_DURATION_SECONDS = 360` and mark `m1`'s anchor is
`resolveAnchor({ atSeconds: 64, bars: [5, 6], alignmentQuality: 0.95 })`,
dimension `pedaling`. Task 13's replacement `FIXTURE_MARKS` in
`practice-preview.tsx` does not reproduce these values — its pedaling mark
(`fixture-1`) is anchored at `atSeconds: 30`, and Task 14's `PieceLessMode`
mount keeps `durationSeconds={120}` (from Task 13's original snippet,
untouched by Task 14's edit). That is a fraction of `30/120 = 0.25`, not
`64/360 ≈ 0.178` — a ~7.2% absolute difference in `offset.left` as a
fraction of strip width, well outside `toBeCloseTo`'s default 2-decimal
tolerance. Task 14's text says to leave this assertion untouched because it
is "testing a DOM-structural property... not anything specific to the
deleted fixtures," but this particular assertion IS specific to the deleted
fixture's numeric values, and Task 13 already replaced those values with
different ones. This is a real, non-flaky test failure once the route is
repointed, not a hypothetical. Before execution: update this assertion's
hardcoded fraction (or find the pedaling glyph and compute the expected
fraction from whatever constants `practice-preview.tsx` actually defines,
so the test derives its expectation rather than hardcoding stale numbers)
as part of Task 13 or Task 14.

[RISK] (confidence: 5/10) — Task 13's `FIXTURE_MARKS` carries over only 2 of
the deleted `mark-fixtures.ts`'s 6 marks (`m1`-`m6`), dropping the specific
case documented in `marks.spec.ts`'s own comment: "A mark anchored at 85% of
the session ran 36px past the strip's right edge at every viewport" (the
width-overflow regression guard, lines 63-98). With only two marks at 25%
and 62.5% of a 120s duration, neither is near the 85%-of-duration edge case
that originally caught this bug. The three `every timeline mark stays
inside the strip` tests will likely still pass (nothing in the new fixture
set is obviously positioned to escape), but they exercise materially weaker
input than before — a real edge-overflow regression could reappear
undetected. Not a blocker because the test won't fail, but the coverage
Task 14 claims to preserve ("matching what marks-preview.tsx covered before
deletion") is narrower than what it replaces. Fallback: add a mark near 85%
of `durationSeconds` to the practice-preview fixture set specifically to
keep this regression guard live.

### Updated Summary

[BLOCKER] count: 2 (both new; both prior blockers verified resolved)
[RISK]    count: 4 (3 carried over from the first pass, unaddressed but
still non-blocking: `formatElapsed` duplication, `ScoreStand`'s per-page-turn
`getPage` refetch, manual `routeTree.gen.ts` regeneration; plus 1 new:
weakened edge-overflow fixture coverage)
[QUESTION] count: 0

VERDICT: NEEDS_REWORK — both prior blockers are genuinely fixed, but a full
read of `tests/marks.spec.ts` (not just the lines Task 14 edits) finds two
concrete, high-confidence breaks in the same required gate (`test:marks`):
(1) the "real Verovio engraving" test locates `[data-testid='real-score']`,
a testid that exists only on the route being deleted and is never
reproduced on `ScoreStand`; (2) the "mark sits at its share of session
duration" test hardcodes a `64/360` fraction from the deleted fixture data
that Task 13's replacement fixtures (`30/120`) no longer match. Both are
mechanical, narrowly-scoped fixes (add/rename one testid; update one
hardcoded fraction or compute it from the new fixture constants) but must
land before `test:marks` — a gate this plan is bound not to regress — will
actually go green.

---

## Challenge Review (re-review, attempt 3, final)

### Verification of the four claimed resolutions

All four were checked against the actual current worktree files, not taken
on trust from the plan's own summary text.

1. **Missing stop control — RESOLVED, holds.** Read Task 10's `PracticeMode`
   code directly: it declares a required `onStop: () => void` prop and
   renders a persistent `aria-label="Stop recording"` button
   (`absolute right-4 top-4 z-20`) above whichever sub-surface is showing,
   including `SessionEndedBanner`. Task 12's `handleStopPracticeMode` (calls
   `practice.stop()` then `handleExitListeningMode()`) is wired to it, and
   `AppChat.tsx`'s current code (`grep`-verified: line 1070 `onStop={practice.stop}`,
   `ListeningMode.tsx:321 aria-label="Stop recording"`) confirms the plan's
   premise about what is being replaced. `PracticeMode.test.tsx`'s fourth
   test exercises both pre- and post-auto-stop clicks. Holds.

2. **`tests/a11y.spec.ts` hardcoding `/marks-preview` — RESOLVED, holds.**
   Read the live `tests/a11y.spec.ts` (still has the two `/marks-preview`
   entries pre-implementation, confirming the plan's stated baseline) and
   Task 14's Step 1 instructions: it removes those two `THEME_CASES` entries
   and ports the two color-contrast checks into `tests/marks.spec.ts`, which
   Task 14 also repoints to a `vite dev` `webServer` (confirmed against the
   live `playwright.marks.config.ts`, currently `bun run build && vite
   preview`, matching what Task 14 Step 1 says to change). Holds.

3. **`[data-testid='real-score']` locator — RESOLVED, holds.** Read the live
   `tests/marks.spec.ts:149` (`page.locator("[data-testid='real-score']")`,
   confirming the defect exists pre-fix) and Task 9's `ScoreStand` code,
   which tags its container `data-testid="score-stand-page"` as a child of
   the same `relative` element that hosts both `svgHostRef` (the injected
   Verovio SVG) and `ScoreMarkLayer`. Also read `MarkGlyph.tsx` directly and
   confirmed it renders `aria-expanded`, `aria-label`, and `data-measure-on`
   on its `<button>` — the exact attributes the retained assertions in
   `marks.spec.ts` (`button[aria-expanded]`, `.getAttribute("data-measure-on")`)
   depend on. The locator swap in Task 14 resolves against real DOM shape,
   not an assumption. Holds.

4. **Hardcoded `64/360` fraction — RESOLVED, holds.** Read Task 13's actual
   `practice-preview.tsx` code: `PIECELESS_DURATION_SECONDS = 120` and
   `FIXTURE_MARKS` are exported, with `fixture-1` (pedaling) at
   `atSeconds: 30` and a third mark (`fixture-3`) at `atSeconds: 102`
   (85% of 120s) explicitly restoring the near-edge overflow case. Task 14's
   instructions replace the hardcoded fraction with
   `(pedalingMark.anchor.atSeconds / PIECELESS_DURATION_SECONDS) *
   offset.stripWidth`, derived from the imported constants rather than a
   second hand-typed copy. Also verified `mark-placement.ts`'s `placeMarks`
   directly: it unconditionally routes any `anchor.type !== "bars"` mark to
   `unplaced` (line 55-58), and `resolveAnchor` (`mark.ts:43-51`) only
   produces a `"bars"` anchor when `alignmentQuality >= ALIGNMENT_MIN`
   (`0.8`) — confirming Task 14's `SCORE_FIXTURE_MARKS` (`bars: [1, 1]`,
   `alignmentQuality: 1`) is the correct, and necessary, way to get a
   glyph to render at all, and that `score-ir.ts:243`'s `barNumber: idx + 1`
   genuinely guarantees bar 1 exists (verified by reading that line
   directly). Holds.

### New finding from a fresh full read: the persistent Stop control physically collides with existing corner controls

[BLOCKER] (confidence: 7/10) — `PracticeMode`'s "Stop recording" button is
absolute-positioned at `right-4 top-4 z-20` against the component's own
`relative` root, pinned to the top-right corner of the full-screen surface
regardless of which sub-surface is mounted beneath it. Two of the three
non-banner sub-surfaces already place their own primary control in that same
corner, in normal document flow:

- `ScoreStand`'s header row (`flex ... justify-between ... px-4 py-2`,
  first child of the flex column, i.e. flush with the container's top edge)
  right-aligns its Metronome toggle button. With `px-4`/`py-2` padding and
  `text-label-sm` text, that button's box occupies roughly the same
  `~8-28px` vertical band and the same `right-4` horizontal edge as the
  absolutely-positioned Stop button's `~16-52px` band — the two ranges
  overlap, and both are flush against the same right edge.
- `ConfirmPieceChip`'s row (`flex items-center justify-between ... px-4
  py-2`) right-aligns its "Dismiss" button, rendered as the sibling
  immediately before `ScoreStand` in `PracticeMode`'s JSX. When a confident
  guess is pending on a known piece, this banner sits at the very top of the
  flex column — the same corner the Stop button occupies — so Stop can
  overlay Dismiss just as it can overlay the Metronome toggle.

Neither collision is caught by any test in the plan: jsdom reports zero
width/height for every element (the working context's own stated reason
`ScoreStand.test.tsx` and `PieceLessMode.test.tsx` avoid positional
assertions), and Task 14's real-browser harness (`practice-preview.tsx`)
mounts `ScoreStand` and `PieceLessMode` directly — never `PracticeMode`
itself — so the one control this plan is most insistent on making
"guaranteed present" (see spec, "Why the auto-stop is UI-only") is never
geometrically verified against the controls it is stacked on top of. This is
the same class of defect the working context calls out for #157's overflow
bug getting past four prior gates: a real, plausible, on-screen overlap that
only a real-browser layout check can see, and none exists for this
component. If the collision is real, a student could find the Stop control
(the only way to end a session) sitting on top of, or blocked by, the
Metronome toggle or the piece-guess Dismiss button, with an unpredictable
click target depending on DOM order and exact pixel overlap. Before
execution: either give the Stop button reserved space in a shared header
(so it cannot occupy the same box as `ScoreStand`'s Metronome toggle or
`ConfirmPieceChip`'s Dismiss button), or move one of the colliding controls
out of the top-right corner, and add a real-browser assertion — mounting
`PracticeMode` itself, not just its leaf components, in the geometry
harness — that no two interactive controls overlap, mirroring the existing
timeline-mark collision check already in `tests/marks.spec.ts`.

### Updated Summary

[BLOCKER] count: 1 (new; all four historical blockers/resolutions confirmed
holding by direct file reads, not the plan's own prose)
[RISK] count: 4 (carried over, unaddressed but still genuinely non-blocking:
`formatElapsed` duplication, `ScoreStand`'s per-page-turn `getPage` refetch,
manual `routeTree.gen.ts` regeneration, and the Task 14 Playwright-spec
import of `practice-preview.tsx` for its fixture constants — the plan itself
flags this needs confirming no browser-only code executes eagerly at module
scope, which is untested but mechanically easy to fix by moving the two
constants to a plain `.ts` module if it breaks)
[QUESTION] count: 0

VERDICT: NEEDS_REWORK — the four blockers from the first two passes are
genuinely resolved (verified against live code, not summary text), but a
fresh full read surfaces one new, concrete usability/geometry defect: the
newly-added persistent "Stop recording" control is absolute-positioned
directly on top of the same top-right corner `ScoreStand`'s Metronome toggle
and `ConfirmPieceChip`'s Dismiss button already occupy, and no test —
jsdom or the real-browser harness — mounts `PracticeMode` as a whole to
catch it.

---

## Challenge Review (re-review, attempt 4, final)

No code exists yet in the worktree for any of Tasks 1-14 (`find` for
`PracticeMode`/`ScoreStand`/`ConfirmPieceChip`/`practice-preview` under
`apps/web/src` returns nothing; `apps/web/src/routes/marks-preview.tsx` and
`apps/web/tests/marks.spec.ts` in their pre-#158 form are still the live
files). This review is therefore of the plan text only, verified for
internal consistency and against the actual pre-implementation files it
proposes to touch — not of built code, since none exists yet.

### Verification: does blocker 5's fix hold?

Yes. Read Task 10's current `PracticeMode` code (lines ~1659-1727) and Task
9's current `ScoreStand` code (lines ~1358-1423) directly, not the plan's own
summary prose:

- `PracticeMode` no longer has any `absolute`/`z-20` anywhere. Its return is
  `<div className="flex h-full flex-col">` with the Stop button in a
  `shrink-0` header row (`flex items-center justify-end border-b ... px-4
  py-2`) that precedes a sibling `flex min-h-0 flex-1 flex-col` content
  region holding `ConfirmPieceChip` (when applicable) and then
  `ScoreStand`/`PieceLessMode`/`SessionEndedBanner`. Grepping the whole plan
  file for `absolute`/`z-20`/`z-10` confirms the only remaining hits are (a)
  `ScoreMarkLayer`'s own intentional `absolute inset-0` overlay for mark
  glyphs (unrelated — a different layer, scoped inside `ScoreStand`'s
  `relative` container, not competing for the top-right corner) and (b)
  prose inside the three historical `## Challenge Review` sections
  describing the *old*, now-replaced code. No live task code still uses
  absolute positioning for a top-of-screen control.
- `ScoreStand`'s own header (Metronome toggle) is likewise a normal
  `shrink-0` flex row, not absolutely positioned, and is a sibling *below*
  `PracticeMode`'s Stop row and any `ConfirmPieceChip` in the same flex
  column — so in a real browser these three controls stack vertically by
  document flow, not by z-index, and cannot occupy the same box.
- `ConfirmPieceChip` (Task 6, lines 843-875) is a plain `flex` div with no
  `absolute`/`fixed` class.
- Task 14 now mounts `PracticeMode` itself in `practice-preview.tsx`'s third
  section (`data-testid="practice-mode-preview"`, `confidentGuess:
  CONFIRM_CHIP_GUESS`, `userPickedPieceId: null`), which per
  `resolvePieceLadderState` resolves to `"confirm-chip"` — the one state
  that puts all three competing controls (Stop, Dismiss, Metronome) on
  screen together — and adds a real-browser collision test
  (`tests/marks.spec.ts`, "PracticeMode's Stop control never overlaps a
  sub-surface's own top control") using the same vertical-range collision
  math (not `top` equality) as the file's existing timeline-mark overlap
  check. This is a genuine real-browser geometry assertion, not jsdom
  theatre, and it targets exactly the three controls the prior review named.

Blocker 5 is resolved in the plan text, structurally (normal flex flow
instead of stacking order) rather than just numerically (nudging pixel
offsets), which is the more durable fix.

### New finding from a fresh full read: the click-through's "record with a
picked piece" step cannot be performed against the live app as written

[BLOCKER] (confidence: 7/10) — Task 12 wires `AppChat`'s new `PracticeMode`
mount with `userPickedPieceId={null}` and `confidentGuess={null}` hardcoded,
and says so explicitly: "Wiring those two inputs for real is out of this
task's test... do not invent a piece-picker UI here." No other task in this
plan sets either prop to anything else. Confirmed by reading the actual
pre-implementation `AppChat.tsx` directly: the only piece-related state it
holds is `pieceContext` (`{ piece: string; section?: string }`, line 153),
a free-text LLM extraction (`extractPieceContext`, lines 302-329) used only
to label the old `ListeningMode`'s piece-name editor — it is not a catalog
`pieceId` and cannot satisfy `userPickedPieceId: string | null` or
`ConfidentGuess`'s `{ pieceId, composer, title, confidence }` shape without
new mapping work this plan does not do. There is also no piece-picker UI
anywhere in `AppChat` (the spec's own "Not in scope" list confirms: "The
epic-level 'Home' surface (repertoire cards, add-piece flow)... is not
redesigned"), and `usePracticeSession.ts`'s `case "piece_identified":`
handler (line 300) still only `console.log`s — it was never wired to any
hook-return field a component could consume, before or after this plan.

The practical effect: after this plan lands, tapping "Record" in the actual
running app can **only** ever reach `PracticeMode`'s pieceless branch. Both
rungs of the piece ladder above "pieceless" — `user-picked` and
`confirm-chip` — are live, tested, and correctly implemented in isolation
(Tasks 2, 6, 9, 10), but structurally unreachable from a real recording
session, because nothing in this plan or the existing codebase ever sets
`userPickedPieceId` or `confidentGuess` to a non-null value in `AppChat`.

This collides with the plan's own Final Verification section, which
instructs the click-through as: "1. Record with a picked piece — score
stand shows, Prev/Next page turns work..." with no caveat that this must be
performed against `/practice-preview` (Task 14's dev-only harness, which
*does* fully wire a confident guess) rather than the live app's Record
button. A human following that instruction literally — tap Record, expect
the score stand — will get the pieceless timeline every time, and (per the
working context) `apps/CLAUDE.md` treats the manual click-through verdict as
human-lit and load-bearing for this issue's success criterion; an
unresolved ambiguity in the one document governing that verdict is exactly
the kind of gap that produces a false "it doesn't work" (or a false-positive
pass via the wrong route) at the one gate in this plan that no automated
test protects.

This is not a new engineering problem — the spec's own "Solution" section
already describes `piece_identified`-driven score display as the intended
production behavior, and wiring it for real is legitimately out of scope
for this issue (no piece-catalog resolution work is scheduled here) — but
the plan currently ships that gap *silently*. Before execution: add one
sentence to the Final Verification section's step 1 stating explicitly that
the score-stand-with-a-piece check is performed via `/practice-preview`
(not the live Record flow), and that real `userPickedPieceId`/
`confidentGuess` wiring in `AppChat` is deferred to a tracked follow-up —
matching how the plan already handles the analogous "no backend emits
`mark` events yet" gap for step 1's mark-injection instruction. This is a
documentation fix to the plan, not a design or code change.

### Updated Summary

[BLOCKER] count: 1 (new — the click-through instruction gap above; all five
historical blockers, including loop-3's absolute-positioned Stop control,
confirmed resolved by direct reads of the plan's current task code)
[RISK] count: 4 (unchanged, carried from attempt 3: `formatElapsed`
duplication, `ScoreStand`'s per-page-turn `getPage` refetch, manual
`routeTree.gen.ts` regeneration, and the Task 14 Playwright-spec import of
`practice-preview.tsx` for its fixture constants)
[QUESTION] count: 0

Blocker 5 (the absolute-positioned Stop control from attempt 3) is
confirmed resolved: Task 10 now renders Stop in its own `shrink-0` header
row in normal flex flow, no component in the collision set uses `absolute`
positioning, and Task 14 adds a real-browser collision test that mounts
`PracticeMode` itself with all three competing controls on screen.

VERDICT: NEEDS_REWORK — one blocker: the Final Verification section's
manual click-through step 1 ("Record with a picked piece — score stand
shows...") is not achievable against the live app as written, because Task
12 hardcodes `userPickedPieceId`/`confidentGuess` to `null` with no wiring
path anywhere in this plan or the existing codebase; the fix is a one-line
clarification pointing that check at `/practice-preview` instead, not a
design or code change.
