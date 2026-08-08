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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/lib/pause-state.test.ts
```
Expected: FAIL — `Cannot find module './pause-state'` (the file does not
exist yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

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

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/lib/pause-state.test.ts
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/lib/piece-ladder.test.ts
```
Expected: FAIL — `Cannot find module './piece-ladder'`

- [ ] **Step 3: Implement the minimum to make the test pass**

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

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/lib/piece-ladder.test.ts
```
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/lib/practice-api.marktype.test.ts && bunx tsc --noEmit
```
Expected: FAIL — `tsc` reports `Object literal may only specify known
properties, and 'mark' does not exist in type ... PracticeWsEvent` (the
union has no `mark` variant yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

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

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/lib/practice-api.marktype.test.ts && bunx tsc --noEmit
```
Expected: PASS (1 test), `tsc` exits 0.

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/hooks/usePauseTracker.test.ts
```
Expected: FAIL — `Cannot find module './usePauseTracker'`

- [ ] **Step 3: Implement the minimum to make the test pass**

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

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/hooks/usePauseTracker.test.ts
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/hooks/usePracticeSession.marks.test.ts
```
Expected: FAIL — `result.current.marks` is `undefined` (`toEqual([mark])`
fails; the field does not exist on `UsePracticeSessionReturn` yet).

- [ ] **Step 3: Implement the minimum to make the test pass**

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

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/hooks/usePracticeSession.marks.test.ts && bunx tsc --noEmit
```
Expected: PASS (1 test), `tsc` exits 0.

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/ConfirmPieceChip.test.tsx
```
Expected: FAIL — `Cannot find module './ConfirmPieceChip'`

- [ ] **Step 3: Implement the minimum to make the test pass**

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

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/ConfirmPieceChip.test.tsx
```
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/SessionEndedBanner.test.tsx
```
Expected: FAIL — `Cannot find module './SessionEndedBanner'`

- [ ] **Step 3: Implement the minimum to make the test pass**

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

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/SessionEndedBanner.test.tsx
```
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/PieceLessMode.test.tsx
```
Expected: FAIL — `Cannot find module './PieceLessMode'`

- [ ] **Step 3: Implement the minimum to make the test pass**

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

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/PieceLessMode.test.tsx
```
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/ScoreStand.test.tsx
```
Expected: FAIL — `Cannot find module './ScoreStand'`

- [ ] **Step 3: Implement the minimum to make the test pass**

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

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/ScoreStand.test.tsx
```
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

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
auto-stopped, with resume bringing the prior surface back.

**Interface under test:** `<PracticeMode userPickedPieceId={string | null}
confidentGuess={ConfidentGuess | null} marks={readonly Mark[]}
elapsedSeconds={number} isPlaying={boolean} isRecording={boolean} />`

**Files:**
- Create: `src/components/PracticeMode.tsx`
- Test: `src/components/PracticeMode.test.tsx`

- [ ] **Step 1: Write the failing test**

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
			/>,
		);

		act(() => {
			vi.advanceTimersByTime(AUTO_STOP_SILENCE_MS);
		});
		expect(screen.getByText(/Session ended/i)).toBeInTheDocument();

		fireEvent.click(screen.getByRole("button", { name: /keep playing/i }));
		expect(screen.queryByText(/Session ended/i)).not.toBeInTheDocument();
	});
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/PracticeMode.test.tsx
```
Expected: FAIL — `Cannot find module './PracticeMode'`

- [ ] **Step 3: Implement the minimum to make the test pass**

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
}

/**
 * The orchestrator: the one component that knows all four practice
 * sub-surfaces exist. Everything it delegates to (ScoreStand, PieceLessMode,
 * ConfirmPieceChip, SessionEndedBanner) takes plain props and touches
 * neither the WS nor the session hook directly -- AppChat is the only place
 * that wires usePracticeSession's live state into these props.
 */
export function PracticeMode({
	userPickedPieceId,
	confidentGuess,
	marks,
	elapsedSeconds,
	isPlaying,
	isRecording,
}: PracticeModeProps) {
	const [dismissed, setDismissed] = useState(false);
	const pause = usePauseTracker(isPlaying);

	const ladderState = resolvePieceLadderState({
		userPicked: userPickedPieceId,
		confidentGuess,
		dismissed,
	});

	if (pause.autoStopped) {
		return <SessionEndedBanner onResume={pause.resume} />;
	}

	const pieceId =
		ladderState === "user-picked"
			? userPickedPieceId
			: ladderState === "confirm-chip"
				? (confidentGuess?.pieceId ?? null)
				: null;

	return (
		<div className="flex h-full flex-col">
			{ladderState === "confirm-chip" && confidentGuess && (
				<ConfirmPieceChip guess={confidentGuess} onDismiss={() => setDismissed(true)} />
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
		</div>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/PracticeMode.test.tsx
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/AppChat.greetings.test.tsx
```
Expected: FAIL — the test finds `"Let's make some music."` (or whichever
`GREETINGS` entry the random pick lands on) still in the document, or the
run is flaky across executions because `GREETINGS` is chosen at random.
Either failure mode confirms the array is still live.

- [ ] **Step 3: Implement the minimum to make the test pass**

In `src/components/AppChat.tsx`, delete the `GREETINGS` array (the block
starting `const GREETINGS = [` through its closing `];`), delete the
`greeting` `useMemo` block, and delete the `<h1 ...>{greeting}</h1>` element
from the empty-state JSX, leaving the icon and `ChatInput` as the only
children of that empty-state container.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/AppChat.greetings.test.tsx
```
Expected: PASS (1 test), and re-running it several times in a row (`bunx
vitest run src/components/AppChat.greetings.test.tsx --repeat 5` or manual
repeats) never flips — proving the randomness is gone, not just unlucky.

- [ ] **Step 5: Commit**

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
them.

**Interface under test:** `AppChat` render output while `practice.state ===
"recording"`

**Files:**
- Modify: `src/components/AppChat.tsx`
- Delete: `src/components/ListeningMode.tsx`
- Delete: `src/components/AudioWaveformRing.tsx`
- Test: `src/components/AppChat.practicemode.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
// src/components/AppChat.practicemode.test.tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import AppChat from "./AppChat";

vi.mock("../hooks/useAuth", () => ({
	authQueryOptions: { queryKey: ["auth"], queryFn: () => null },
	useAuth: () => ({ data: null, isLoading: false }),
}));

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
		stop: vi.fn(),
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
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/AppChat.practicemode.test.tsx
```
Expected: FAIL — `session-timeline` testid is absent because `AppChat`
still mounts `ListeningMode` (which renders `AudioWaveformRing`, not the
timeline strip) whenever `showListeningMode` is true.

- [ ] **Step 3: Implement the minimum to make the test pass**

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
4. Delete `src/components/ListeningMode.tsx` and
   `src/components/AudioWaveformRing.tsx`.
5. Remove `handleExitListeningMode` and any other `ListeningMode`-only
   plumbing (e.g. `pieceContext`/`sessionNotes` state) only if `tsc` reports
   them unused after the deletion — leave anything still referenced
   elsewhere alone.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/components/AppChat.practicemode.test.tsx && bunx tsc --noEmit
```
Expected: PASS (1 test), `tsc` exits 0 (no leftover unused imports/dead
refs to `ListeningMode` or `AudioWaveformRing` anywhere).

- [ ] **Step 5: Commit**

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

- [ ] **Step 1: Write the failing test**

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

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/routes/practice-preview.test.tsx
```
Expected: FAIL — `Cannot find module './practice-preview'`

- [ ] **Step 3: Implement the minimum to make the test pass**

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
const FIXTURE_MARKS: readonly Mark[] = [
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
				durationSeconds={120}
				elapsedSeconds={90}
				isRecording={false}
			/>
		</div>
	);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx vitest run src/routes/practice-preview.test.tsx
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

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

**Interface under test:** the rendered DOM of `/practice-preview` in a real
browser

**Files:**
- Modify: `apps/web/tests/marks.spec.ts`
- Modify: `apps/web/playwright.marks.config.ts`

- [ ] **Step 1: Write the failing test (repoint the existing spec)**

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
specific to the deleted fixtures.

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx playwright test --config playwright.marks.config.ts
```
Expected: FAIL — `/practice-preview` currently only mounts `PieceLessMode`
(Task 13's minimal implementation), so the score-overlay assertions (which
expect a real engraving with bar-anchored marks, matching #157's original
two-canvas coverage) find no `score-container` or bar-anchored
`data-measure-on` buttons.

- [ ] **Step 3: Implement the minimum to make the test pass**

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
			<div className="h-1/2 border-b border-border-subtle">
				<ScoreStand
					pieceId="chopin-nocturne-op9-no2"
					marks={SCORE_FIXTURE_MARKS}
					elapsedSeconds={30}
					isRecording={false}
				/>
			</div>
			<div className="h-1/2">
				<PieceLessMode
					marks={FIXTURE_MARKS}
					durationSeconds={120}
					elapsedSeconds={90}
					isRecording={false}
				/>
			</div>
		</div>
	);
```

Add the corresponding import (`import { ScoreStand } from
"../components/ScoreStand";`) and a `SCORE_FIXTURE_MARKS` constant built the
same way `RealScoreSection` built its single mark before deletion: it cannot
hardcode a bar number, because `ScoreStand`'s own load effect is what
resolves real bar numbers from the engraving. Anchor a mark to elapsed time
alone (`resolveAnchor({ atSeconds: 20, alignmentQuality: 0 })`, a
timestamp-type anchor) rather than trying to guess a bar number from outside
the component — a bar-anchored assertion belongs to `ScoreStand.test.tsx`'s
own future coverage (per Task 9's noted follow-up), not to this harness,
whose job is proving containment and non-overlap, both of which a
timestamp-anchored mark on the timeline strip already exercises.

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bunx playwright test --config playwright.marks.config.ts
```
Expected: PASS (both tests in `tests/marks.spec.ts`)

- [ ] **Step 5: Commit**

```bash
git add apps/web/tests/marks.spec.ts apps/web/playwright.marks.config.ts apps/web/src/routes/practice-preview.tsx && git commit -m "test(practice-mode): port real-browser mark geometry harness to practice-preview"
```

---

## Final Verification (run after all tasks land)

```bash
cd /Users/jdhiman/Documents/crescendai/.worktrees/issue-158-practice-mode/apps/web && bun run test && bunx tsc --noEmit && bun run lint && bun run test:a11y && bun run test:marks
```

Expected: `test` green (existing 246 tests plus every new test above);
`tsc` exit 0; `lint` 0 errors, warnings/infos at or below the accepted
baseline (107/23) — a new file introducing a new warning is a regression,
not "close enough"; `test:a11y` 4/4; `test:marks` green against the ported
harness.

Then the manual click-through from the issue's success criterion, performed
by a human (this is human-lit per `apps/CLAUDE.md` — "manual click-through
verdicts are human-lit"):
1. Record with a picked piece — score stand shows, Prev/Next page turns
   work, and (since no backend emits real `mark` events yet — see spec,
   "Not in scope") a manually-dispatched `mark` WS message via the browser
   devtools console lands on the right bar and simultaneously appears on the
   timeline if opened.
2. Record pieceless — timeline strip accrues injected marks the same way.
3. Silence past `AUTO_STOP_SILENCE_MS` — hard to do at 60s in real time
   without a config surface; verify by temporarily lowering
   `AUTO_STOP_SILENCE_MS` in a local, uncommitted edit for the click-through
   only, or by confirming `usePauseTracker`'s already-green fake-timer tests
   are a faithful stand-in and documenting that the constant was not
   independently re-verified in real time. Do not commit a temporary
   threshold change.
