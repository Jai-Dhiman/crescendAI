# Web Real Pipeline Integration -- Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the CrescendAI web practice companion from mock mode to the real inference and observation pipeline.

**Architecture:** Remove the `MOCK_MODE` flag in `usePracticeSession.ts` and let the existing real code paths run. Add a client-side observation throttle (3-minute window), chunk upload tracking with retry, and robust error handling (reconnection, offline, auth expiry). ScorePanel gated to dev-only. The session summary (pushed via WebSocket on stop) IS the automatic "how was that?" -- no explicit ask button needed. Follow-up questions ("tell me more about my pedaling") go through the existing `api.chat.send()`.

**Tech Stack:** TanStack Start (React 19), TypeScript, Zustand, Vitest, Tailwind CSS v4, @sentry/react

**Spec:** `docs/superpowers/specs/2026-03-14-web-real-pipeline-design.md`

**Scope:** `apps/web/` only. Do NOT touch `apps/api/`, `apps/ios/`, or `model/`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/lib/observation-throttle.ts` | Create | Pure TS class: 3-min delivery window, queue size 1, no React deps |
| `src/lib/observation-throttle.test.ts` | Create | Unit tests for throttle logic |
| `src/hooks/usePracticeSession.ts` | Modify | Remove mock mode, add chunk tracking, wsStatus, offline handling, throttle integration |
| `src/components/AppChat.tsx` | Modify | Remove mock refs, gate ScorePanel |
| `src/components/ListeningMode.tsx` | Modify | Add wsStatus reconnection indicator |
| `src/stores/score-panel.ts` | Modify | Gate `open()` behind `import.meta.env.DEV` |
| `vite.config.ts` | Modify | Add vitest `test` config block |

**Not modified:** `src/lib/practice-api.ts` (no `ask()` needed -- follow-ups go through existing `api.chat.send()`), `src/components/ChatInput.tsx` (no "How was that?" button).

---

## Chunk 1: ObservationThrottle + Test Infrastructure

### Task 1: Set up vitest configuration

**Files:**
- Modify: `apps/web/vite.config.ts`

- [ ] **Step 1: Add test config to vite.config.ts**

Add a `test` block to the existing vite config. The web app uses jsdom (already in devDependencies) for DOM simulation.

```typescript
// Add to the defineConfig object, after `plugins`:
// In vite.config.ts, the config should include:
const config = defineConfig({
	build: { sourcemap: true },
	test: {
		environment: "jsdom",
		include: ["src/**/*.test.ts"],
	},
	plugins: [
		// ... existing plugins unchanged
	],
});
```

- [ ] **Step 2: Verify vitest runs with no tests**

Run: `cd apps/web && bun run test`
Expected: vitest runs, reports 0 tests found, exits cleanly.

- [ ] **Step 3: Commit**

```bash
git add apps/web/vite.config.ts
git commit -m "configure vitest with jsdom for web app tests"
```

---

### Task 2: Implement ObservationThrottle

**Files:**
- Create: `apps/web/src/lib/observation-throttle.ts`
- Create: `apps/web/src/lib/observation-throttle.test.ts`

**Context:** This is a pure TypeScript class with no React dependencies. It gates when observations are delivered to the UI, enforcing a 3-minute window between deliveries and requiring a minimum number of chunks before the first observation. Queue size is 1 (new observations replace older queued ones). The hook's existing 1-second timer calls `tick()` -- no internal timers.

- [ ] **Step 1: Write the failing tests**

```typescript
// apps/web/src/lib/observation-throttle.test.ts
import { describe, expect, it } from "vitest";
import { ObservationThrottle } from "./observation-throttle";
import type { ObservationEvent } from "./practice-api";

function makeObs(text: string): ObservationEvent {
	return { text, dimension: "dynamics", framing: "correction" };
}

describe("ObservationThrottle", () => {
	it("blocks observations before minChunks reached", () => {
		const throttle = new ObservationThrottle({ minChunksBeforeFirst: 4 });
		// Only 2 chunks processed
		throttle.onChunkProcessed();
		throttle.onChunkProcessed();
		const result = throttle.enqueue(makeObs("too early"));
		expect(result).toBeNull();
	});

	it("delivers first observation after minChunks reached", () => {
		const throttle = new ObservationThrottle({ minChunksBeforeFirst: 4 });
		for (let i = 0; i < 4; i++) throttle.onChunkProcessed();
		const result = throttle.enqueue(makeObs("ready"));
		expect(result).not.toBeNull();
		expect(result?.text).toBe("ready");
	});

	it("blocks second observation within throttle window", () => {
		const throttle = new ObservationThrottle({
			minChunksBeforeFirst: 1,
			windowMs: 180_000,
		});
		throttle.onChunkProcessed();
		// First one delivers
		const first = throttle.enqueue(makeObs("first"));
		expect(first).not.toBeNull();
		// Second one is blocked (within window)
		const second = throttle.enqueue(makeObs("second"));
		expect(second).toBeNull();
	});

	it("releases queued observation via tick() after window expires", () => {
		const throttle = new ObservationThrottle({
			minChunksBeforeFirst: 1,
			windowMs: 100, // Short window for testing
		});
		throttle.onChunkProcessed();

		throttle.enqueue(makeObs("first"));
		throttle.enqueue(makeObs("queued"));

		// tick() before window expires returns null
		const tooEarly = throttle.tick();
		expect(tooEarly).toBeNull();

		// Wait for window to expire
		return new Promise<void>((resolve) => {
			setTimeout(() => {
				const released = throttle.tick();
				expect(released).not.toBeNull();
				expect(released?.text).toBe("queued");
				resolve();
			}, 150);
		});
	});

	it("replaces queued observation with newer one", () => {
		const throttle = new ObservationThrottle({
			minChunksBeforeFirst: 1,
			windowMs: 180_000,
		});
		throttle.onChunkProcessed();

		throttle.enqueue(makeObs("first"));
		throttle.enqueue(makeObs("old queued"));
		throttle.enqueue(makeObs("new queued"));

		const drained = throttle.drain();
		expect(drained).toHaveLength(1);
		expect(drained[0].text).toBe("new queued");
	});

	it("onChunkProcessed releases queued observation when minChunks newly met", () => {
		const throttle = new ObservationThrottle({
			minChunksBeforeFirst: 3,
			windowMs: 180_000,
		});
		// Enqueue before minChunks met
		throttle.onChunkProcessed();
		throttle.enqueue(makeObs("waiting"));

		// Still not enough chunks
		const r2 = throttle.onChunkProcessed();
		expect(r2).toBeNull();

		// Third chunk meets minimum
		const r3 = throttle.onChunkProcessed();
		expect(r3).not.toBeNull();
		expect(r3?.text).toBe("waiting");
	});

	it("drain() returns queued observation and empties queue", () => {
		const throttle = new ObservationThrottle({
			minChunksBeforeFirst: 1,
			windowMs: 180_000,
		});
		throttle.onChunkProcessed();
		throttle.enqueue(makeObs("first"));
		throttle.enqueue(makeObs("queued"));

		const drained = throttle.drain();
		expect(drained).toHaveLength(1);
		expect(drained[0].text).toBe("queued");

		// Drain again returns empty
		expect(throttle.drain()).toHaveLength(0);
	});

	it("reset() clears all state", () => {
		const throttle = new ObservationThrottle({ minChunksBeforeFirst: 1 });
		throttle.onChunkProcessed();
		throttle.enqueue(makeObs("obs"));
		throttle.reset();

		// After reset, minChunks not met again
		const result = throttle.enqueue(makeObs("after reset"));
		expect(result).toBeNull();
	});
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/web && bun run test`
Expected: FAIL -- `observation-throttle` module not found.

- [ ] **Step 3: Implement ObservationThrottle**

```typescript
// apps/web/src/lib/observation-throttle.ts
import type { ObservationEvent } from "./practice-api";

interface ObservationThrottleOptions {
	windowMs?: number;
	minChunksBeforeFirst?: number;
}

export class ObservationThrottle {
	private readonly windowMs: number;
	private readonly minChunksBeforeFirst: number;
	private lastDeliveredAt = 0;
	private queued: ObservationEvent | null = null;
	private chunksReceived = 0;

	constructor(options?: ObservationThrottleOptions) {
		this.windowMs = options?.windowMs ?? 180_000;
		this.minChunksBeforeFirst = options?.minChunksBeforeFirst ?? 4;
	}

	enqueue(obs: ObservationEvent): ObservationEvent | null {
		if (this.canDeliver()) {
			this.lastDeliveredAt = Date.now();
			return obs;
		}
		// Queue it (replace any existing queued observation)
		this.queued = obs;
		return null;
	}

	onChunkProcessed(): ObservationEvent | null {
		this.chunksReceived++;
		return this.tryRelease();
	}

	tick(): ObservationEvent | null {
		return this.tryRelease();
	}

	drain(): ObservationEvent[] {
		if (this.queued) {
			const obs = this.queued;
			this.queued = null;
			return [obs];
		}
		return [];
	}

	reset(): void {
		this.lastDeliveredAt = 0;
		this.queued = null;
		this.chunksReceived = 0;
	}

	private canDeliver(): boolean {
		if (this.chunksReceived < this.minChunksBeforeFirst) return false;
		if (this.lastDeliveredAt === 0) return true;
		return Date.now() - this.lastDeliveredAt >= this.windowMs;
	}

	private tryRelease(): ObservationEvent | null {
		if (this.queued && this.canDeliver()) {
			const obs = this.queued;
			this.queued = null;
			this.lastDeliveredAt = Date.now();
			return obs;
		}
		return null;
	}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/web && bun run test`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/lib/observation-throttle.ts apps/web/src/lib/observation-throttle.test.ts
git commit -m "add ObservationThrottle with tests

Pure TypeScript class that gates observation delivery to a 3-minute
window with configurable minChunks before first delivery. Queue
size 1, driven externally by the hook's 1-second timer."
```

---

## Chunk 2: practice-api.ts + ScorePanel Gate + Mock Removal

### Task 3: Gate ScorePanel behind import.meta.env.DEV

**Files:**
- Modify: `apps/web/src/stores/score-panel.ts`

**Context:** The ScorePanel depends on `MockSessionData` and real score alignment isn't available yet. Gate `open()` so it's a no-op in production.

- [ ] **Step 1: Gate the open() method**

In `apps/web/src/stores/score-panel.ts`, modify the `open` method inside `create<ScorePanelState>`:

Replace:
```typescript
	open: (data) =>
		set({ isOpen: true, sessionData: data, activeAnnotationIndex: null }),
```

With:
```typescript
	open: (data) => {
		if (!import.meta.env.DEV) return;
		set({ isOpen: true, sessionData: data, activeAnnotationIndex: null });
	},
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit`
Expected: No type errors.

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/stores/score-panel.ts
git commit -m "gate ScorePanel open() behind import.meta.env.DEV

Real score alignment (bar-range mapping) is not yet implemented.
ScorePanel stays functional in dev for testing with mock data."
```

---

### Task 4: Remove MOCK_MODE and rewrite usePracticeSession

**Files:**
- Modify: `apps/web/src/hooks/usePracticeSession.ts`

**Context:** This is the largest change. Remove the mock flag, add chunk upload tracking with retry, WebSocket status with exponential backoff, network offline handling, and observation throttle integration. The existing real code paths are mostly intact -- we're removing the mock gates around them and adding the missing pieces.

- [ ] **Step 1: Remove mock imports and MOCK_MODE flag**

In `apps/web/src/hooks/usePracticeSession.ts`:

Replace lines 1-11:
```typescript
import { useCallback, useEffect, useRef, useState } from "react";
import { generateMockSession, type MockSessionData } from "../lib/mock-session";
import type {
	DimScores,
	ObservationEvent,
	PracticeWsEvent,
} from "../lib/practice-api";
import { practiceApi } from "../lib/practice-api";
import { Sentry } from "../lib/sentry";

const MOCK_MODE = true; // Set to false when real inference is available
```

With:
```typescript
import { useCallback, useEffect, useRef, useState } from "react";
import { ObservationThrottle } from "../lib/observation-throttle";
import type {
	DimScores,
	ObservationEvent,
	PracticeWsEvent,
} from "../lib/practice-api";
import { practiceApi } from "../lib/practice-api";
import { Sentry } from "../lib/sentry";
```

- [ ] **Step 2: Update constants and types**

Replace lines 21-22:
```typescript
const MAX_RECONNECTS = 3;
const RECONNECT_DELAY_MS = 2000;
```

With:
```typescript
const MAX_RECONNECTS = 5;
const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 30_000;

export type WsStatus = "connected" | "reconnecting" | "disconnected";
type ChunkStatus = "uploading" | "complete" | "failed";
export interface ChunkState {
	index: number;
	status: ChunkStatus;
}
```

- [ ] **Step 3: Update the return interface**

Replace the `UsePracticeSessionReturn` interface:
```typescript
export interface UsePracticeSessionReturn {
	state: PracticeState;
	elapsedSeconds: number;
	observations: ObservationEvent[];
	latestScores: DimScores | null;
	summary: string | null;
	error: string | null;
	analyserNode: AnalyserNode | null;
	chunksProcessed: number;
	mockSessionData: MockSessionData | null;
	start: () => Promise<void>;
	stop: () => void;
}
```

With:
```typescript
export interface UsePracticeSessionReturn {
	state: PracticeState;
	elapsedSeconds: number;
	observations: ObservationEvent[];
	latestScores: DimScores | null;
	summary: string | null;
	error: string | null;
	analyserNode: AnalyserNode | null;
	chunksProcessed: number;
	chunkStates: ChunkState[];
	wsStatus: WsStatus;
	isOnline: boolean;
	start: () => Promise<void>;
	stop: () => void;
}
```

- [ ] **Step 4: Replace state declarations inside usePracticeSession**

Replace lines 39-48 (state declarations):
```typescript
	const [state, setState] = useState<PracticeState>("idle");
	const [elapsedSeconds, setElapsedSeconds] = useState(0);
	const [observations, setObservations] = useState<ObservationEvent[]>([]);
	const [latestScores, setLatestScores] = useState<DimScores | null>(null);
	const [summary, setSummary] = useState<string | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
	const [chunksProcessed, setChunksProcessed] = useState(0);
	const [mockSessionData, setMockSessionData] =
		useState<MockSessionData | null>(null);
```

With:
```typescript
	const [state, setState] = useState<PracticeState>("idle");
	const [elapsedSeconds, setElapsedSeconds] = useState(0);
	const [observations, setObservations] = useState<ObservationEvent[]>([]);
	const [latestScores, setLatestScores] = useState<DimScores | null>(null);
	const [summary, setSummary] = useState<string | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
	const [chunksProcessed, setChunksProcessed] = useState(0);
	const [chunkStates, setChunkStates] = useState<ChunkState[]>([]);
	const [wsStatus, setWsStatus] = useState<WsStatus>("disconnected");
	const [isOnline, setIsOnline] = useState(
		typeof navigator !== "undefined" ? navigator.onLine : true,
	);
```

- [ ] **Step 5: Add new refs**

After the existing refs block (after line 58 `stateRef`), add:

```typescript
	const throttleRef = useRef(new ObservationThrottle());
	const isOnlineRef = useRef(isOnline);
	const offlineQueueRef = useRef<Array<{ index: number; blob: Blob }>>([]);
```

And add a sync effect for isOnlineRef right after the stateRef sync:
```typescript
	useEffect(() => {
		isOnlineRef.current = isOnline;
	}, [isOnline]);
```

- [ ] **Step 6: Add online/offline event listener**

After the isOnlineRef sync effect, add:

```typescript
	// Network online/offline detection
	useEffect(() => {
		function handleOnline() {
			setIsOnline(true);
		}
		function handleOffline() {
			setIsOnline(false);
		}
		window.addEventListener("online", handleOnline);
		window.addEventListener("offline", handleOffline);
		return () => {
			window.removeEventListener("online", handleOnline);
			window.removeEventListener("offline", handleOffline);
		};
	}, []);
```

- [ ] **Step 7: Add uploadChunkWithRetry helper and chunk state updater**

Add before the `cleanup` callback:

```typescript
	const updateChunkState = useCallback(
		(index: number, status: ChunkStatus) => {
			setChunkStates((prev) => {
				const existing = prev.findIndex((c) => c.index === index);
				if (existing >= 0) {
					const updated = [...prev];
					updated[existing] = { index, status };
					return updated;
				}
				return [...prev, { index, status }];
			});
		},
		[],
	);
```

- [ ] **Step 8: Update cleanup to reset new state**

In the `cleanup` callback, add resets for the new refs:

After `reconnectAttemptsRef.current = 0;`, add:
```typescript
		throttleRef.current.reset();
		offlineQueueRef.current = [];
```

- [ ] **Step 9: Update handleWsMessage for throttle + summary building**

Replace the `handleWsMessage` callback entirely:

```typescript
	const handleWsMessage = useCallback((event: MessageEvent) => {
		const data: PracticeWsEvent = JSON.parse(event.data);
		switch (data.type) {
			case "chunk_processed": {
				setLatestScores(data.scores);
				setChunksProcessed((prev) => prev + 1);
				// Check if throttle can release a queued observation
				const released = throttleRef.current.onChunkProcessed();
				if (released) {
					setObservations((prev) => [...prev, released]);
				}
				break;
			}
			case "observation": {
				const obs: ObservationEvent = {
					text: data.text,
					dimension: data.dimension,
					framing: data.framing,
				};
				const immediate = throttleRef.current.enqueue(obs);
				if (immediate) {
					setObservations((prev) => [...prev, immediate]);
				}
				break;
			}
			case "session_summary": {
				// Drain any undelivered queued observations
				const drained = throttleRef.current.drain();
				const allObs = [...data.observations];
				for (const obs of drained) {
					if (!allObs.some((o) => o.text === obs.text)) {
						allObs.push(obs);
					}
				}
				setObservations(allObs);

				// Build summary string
				const obsLines = allObs
					.map((o) => `- ${o.text}`)
					.join("\n");
				const chunksCount = chunkIndexRef.current;
				const builtSummary = obsLines
					? `I listened to ${chunksCount} sections of your playing.\n\nDuring the session, I noticed:\n${obsLines}\n\nWant to hear more about any of these?`
					: `I listened to ${chunksCount} sections of your playing.\n\nI didn't notice anything specific to flag this time. Want to talk about how it felt?`;
				setSummary(builtSummary);
				setState("idle");
				cleanup();
				break;
			}
			case "error":
				setError(data.message);
				break;
		}
	}, [cleanup]);
```

- [ ] **Step 10: Update connectWebSocket for exponential backoff + wsStatus**

Replace the `connectWebSocket` callback:

```typescript
	const connectWebSocket = useCallback(
		(sessionId: string): Promise<WebSocket> => {
			return new Promise((resolve, reject) => {
				const ws = practiceApi.connectWebSocket(sessionId);
				wsRef.current = ws;

				ws.onmessage = handleWsMessage;

				ws.onopen = () => {
					reconnectAttemptsRef.current = 0;
					setWsStatus("connected");
					resolve(ws);
				};

				ws.onerror = () => {
					Sentry.captureException(new Error("WebSocket failed to connect"), {
						extra: { sessionId },
					});
					reject(new Error("WebSocket failed to connect"));
				};

				ws.onclose = () => {
					if (
						stateRef.current === "recording" &&
						reconnectAttemptsRef.current < MAX_RECONNECTS
					) {
						setWsStatus("reconnecting");
						const attempt = reconnectAttemptsRef.current++;
						const delay = Math.min(
							RECONNECT_BASE_DELAY_MS * 2 ** attempt,
							RECONNECT_MAX_DELAY_MS,
						);
						setTimeout(() => {
							if (stateRef.current === "recording" && sessionIdRef.current) {
								connectWebSocket(sessionIdRef.current).catch(() => {
									Sentry.captureMessage("WebSocket reconnection failed", {
										level: "error",
										extra: { attempts: reconnectAttemptsRef.current },
									});
									setWsStatus("disconnected");
									setError("Connection lost. Please try again.");
									setState("error");
									cleanup();
								});
							}
						}, delay);
					} else if (stateRef.current === "recording") {
						setWsStatus("disconnected");
						setError("Connection lost. Please try again.");
						setState("error");
						cleanup();
					}
				};
			});
		},
		[handleWsMessage, cleanup],
	);
```

- [ ] **Step 11: Remove mock branches from start()**

Remove the `MOCK_MODE` block in `start()` (the `if (MOCK_MODE) { ... return; }` block at lines 197-206). The code should flow directly from AnalyserNode setup to "3. Start session on server".

Also reset new state in the start() setup:
After `setChunksProcessed(0);` add:
```typescript
		setChunkStates([]);
		setWsStatus("disconnected");
```

And remove:
```typescript
		setMockSessionData(null);
```

- [ ] **Step 12: Add uploadWithRetry as a local function inside start()**

Add this right before the `recorder.ondataavailable` assignment (after `chunkIndexRef.current = 0;`):

```typescript
		async function uploadWithRetry(
			sid: string,
			idx: number,
			blob: Blob,
		): Promise<void> {
			updateChunkState(idx, "uploading");
			for (let attempt = 0; attempt < 2; attempt++) {
				try {
					const { r2Key } = await practiceApi.uploadChunk(sid, idx, blob);
					updateChunkState(idx, "complete");
					const ws = wsRef.current;
					if (ws?.readyState === WebSocket.OPEN) {
						ws.send(
							JSON.stringify({ type: "chunk_ready", index: idx, r2Key }),
						);
					}
					return;
				} catch (e) {
					// Auth expiry: surface immediately, do not retry
					if (e instanceof Error && e.message.includes("401")) {
						updateChunkState(idx, "failed");
						setError("Session expired. Please sign in again.");
						setState("error");
						cleanup();
						return;
					}
					if (attempt === 0) {
						await new Promise((r) => setTimeout(r, 2000));
					} else {
						updateChunkState(idx, "failed");
						Sentry.captureException(e, {
							extra: { chunkIndex: idx, sessionId: sid },
						});
					}
				}
			}
		}
```

Note: `updateChunkState` is the callback defined in Step 7. Since `start()` is defined inside the hook, it has closure access to `updateChunkState`.

- [ ] **Step 13: Replace ondataavailable with retry + offline support**

Replace the `recorder.ondataavailable` handler in `start()`:

```typescript
		recorder.ondataavailable = async (event) => {
			if (event.data.size === 0) return;
			const idx = chunkIndexRef.current++;

			if (!isOnlineRef.current) {
				offlineQueueRef.current.push({ index: idx, blob: event.data });
				updateChunkState(idx, "uploading");
				return;
			}

			await uploadWithRetry(sessionId, idx, event.data);
		};
```

- [ ] **Step 14: Add throttle tick to the elapsed timer**

In the elapsed timer `setInterval` inside `start()`, add throttle tick:

Replace:
```typescript
		timerRef.current = setInterval(() => {
			setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
		}, 1000);
```

With:
```typescript
		timerRef.current = setInterval(() => {
			setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
			const released = throttleRef.current.tick();
			if (released) {
				setObservations((prev) => [...prev, released]);
			}
		}, 1000);
```

- [ ] **Step 15: Add offline queue flush effect**

After the online/offline event listener effect, add:

```typescript
	// Flush offline queue when back online
	useEffect(() => {
		if (!isOnline || !sessionIdRef.current) return;
		const queue = offlineQueueRef.current;
		if (queue.length === 0) return;

		const sessionId = sessionIdRef.current;
		offlineQueueRef.current = [];

		(async () => {
			for (const { index, blob } of queue) {
				try {
					const { r2Key } = await practiceApi.uploadChunk(
						sessionId,
						index,
						blob,
					);
					updateChunkState(index, "complete");
					const ws = wsRef.current;
					if (ws?.readyState === WebSocket.OPEN) {
						ws.send(
							JSON.stringify({ type: "chunk_ready", index, r2Key }),
						);
					}
				} catch (e) {
					updateChunkState(index, "failed");
					Sentry.captureException(e, {
						extra: { chunkIndex: index, sessionId },
					});
				}
			}
		})();
	}, [isOnline, updateChunkState]);
```

- [ ] **Step 16: Remove mock branches from stop() and fix dependency array**

Remove the entire `if (MOCK_MODE) { ... return; }` block from `stop()` (lines 303-324).

Update the dependency array of `stop`:

Replace:
```typescript
	}, [state, cleanup, elapsedSeconds]);
```

With:
```typescript
	}, [state, cleanup]);
```

- [ ] **Step 17: Update return object**

Replace the return block:

```typescript
	return {
		state,
		elapsedSeconds,
		observations,
		latestScores,
		summary,
		error,
		analyserNode,
		chunksProcessed,
		mockSessionData,
		start,
		stop,
	};
```

With:
```typescript
	return {
		state,
		elapsedSeconds,
		observations,
		latestScores,
		summary,
		error,
		analyserNode,
		chunksProcessed,
		chunkStates,
		wsStatus,
		isOnline,
		start,
		stop,
	};
```

- [ ] **Step 18: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit`
Expected: Errors in `AppChat.tsx` (references to removed `mockSessionData`) and `ListeningMode.tsx` (new `wsStatus` prop) -- these are expected and will be fixed in Tasks 5-6.

- [ ] **Step 19: Commit**

```bash
git add apps/web/src/hooks/usePracticeSession.ts
git commit -m "remove mock mode, add real pipeline integration to usePracticeSession

- Remove MOCK_MODE flag and all mock branches
- Add chunk upload tracking with retry (2s delay, 1 retry)
- Add WebSocket status with exponential backoff (1s-30s, 5 attempts)
- Add network offline detection and chunk queuing
- Integrate ObservationThrottle for 3-minute delivery window
- Build session summary from real WebSocket data"
```

---

## Chunk 3: UI Components (ListeningMode, AppChat)

Note: `RecordingBar.tsx` exists but is never imported or rendered anywhere in the codebase. `ListeningMode` is the active recording UI. RecordingBar and ChatInput are left unchanged.

No "How was that?" button is needed. The session summary (pushed via WebSocket when recording stops) IS the automatic feedback. Follow-up questions ("tell me more about my pedaling") go through the existing `api.chat.send()` in the chat input.

### Task 5: Update ListeningMode for wsStatus

**Files:**
- Modify: `apps/web/src/components/ListeningMode.tsx`

**Context:** Add `wsStatus` prop for reconnection indicator during recording.

- [ ] **Step 1: Add wsStatus to props**

In `apps/web/src/components/ListeningMode.tsx`, update the interface:

```typescript
import type { WsStatus } from "../hooks/usePracticeSession";

interface ListeningModeProps {
	state: PracticeState;
	observations: ObservationEvent[];
	analyserNode: AnalyserNode | null;
	latestScores: DimScores | null;
	error: string | null;
	wsStatus: WsStatus;
	onStop: () => void;
	originRect: DOMRect | null;
	onExit: () => void;
	pieceContext?: { piece: string; section?: string } | null;
	sessionNotes?: string;
	onNotesChange?: (notes: string) => void;
}
```

Add `wsStatus` to the destructured props.

- [ ] **Step 2: Add reconnection indicator**

In the center section, above the observation toasts div, add:

```tsx
						{wsStatus === "reconnecting" && isRecording && (
							<div className="absolute top-4 left-4 flex items-center gap-2 text-amber-400 z-10">
								<CircleNotch size={14} className="animate-spin" />
								<span className="text-body-xs">Reconnecting...</span>
							</div>
						)}
```

Import `CircleNotch` at the top:
```typescript
import {
	CircleNotch,
	Metronome as MetronomeIcon,
	Minus,
	Plus,
	Stop,
} from "@phosphor-icons/react";
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit`
Expected: Errors in AppChat (caller for new `wsStatus` prop) -- expected, fixed next.

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/components/ListeningMode.tsx
git commit -m "add WebSocket reconnection indicator to ListeningMode

Shows 'Reconnecting...' with spinner when wsStatus is reconnecting
during recording."
```

---

### Task 6: Update AppChat to wire everything together

**Files:**
- Modify: `apps/web/src/components/AppChat.tsx`

**Context:** Remove `mockSessionData` references, pass `wsStatus` to ListeningMode, gate ScorePanel auto-open.

- [ ] **Step 1: Remove mock-related code**

In `apps/web/src/components/AppChat.tsx`:

Remove the `useEffect` that auto-opens ScorePanel from `practice.mockSessionData`. This also eliminates the last reference to `practice.mockSessionData` (which no longer exists on the hook return type after Task 4).

```typescript
	// Auto-open score panel when mock session data arrives
	useEffect(() => {
		if (practice.mockSessionData && !showListeningMode) {
			scorePanel.open(practice.mockSessionData);
		}
	}, [practice.mockSessionData, showListeningMode, scorePanel.open]);
```

- [ ] **Step 2: Pass wsStatus to ListeningMode**

Update the `<ListeningMode>` JSX to include `wsStatus`:

```tsx
				{showListeningMode && (
					<ListeningMode
						state={practice.state}
						observations={practice.observations}
						analyserNode={practice.analyserNode}
						latestScores={practice.latestScores}
						error={practice.error}
						wsStatus={practice.wsStatus}
						onStop={practice.stop}
						originRect={recordButtonRect}
						onExit={handleExitListeningMode}
						sessionNotes={sessionNotes}
						onNotesChange={setSessionNotes}
						pieceContext={pieceContext}
					/>
				)}
```

- [ ] **Step 3: Verify TypeScript compiles cleanly**

Run: `cd apps/web && bunx tsc --noEmit`
Expected: No type errors. All removed `mockSessionData` references are gone, `wsStatus` is wired.

- [ ] **Step 4: Run all tests**

Run: `cd apps/web && bun run test`
Expected: All ObservationThrottle tests pass.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/AppChat.tsx
git commit -m "wire real pipeline integration in AppChat

- Remove mockSessionData references and ScorePanel auto-open
- Pass wsStatus to ListeningMode"
```

---

### Task 7: Verify build and lint

**Files:** None (verification only)

- [ ] **Step 1: Run lint**

Run: `cd apps/web && bun run lint`
Expected: No errors (warnings acceptable).

- [ ] **Step 2: Run build**

Run: `cd apps/web && bun run build`
Expected: Build succeeds.

- [ ] **Step 3: Run tests**

Run: `cd apps/web && bun run test`
Expected: All tests pass.

- [ ] **Step 4: Commit any lint fixes if needed**

```bash
git add -u apps/web/src/
git commit -m "fix lint issues from pipeline integration"
```

(Skip this step if no lint fixes are needed.)
