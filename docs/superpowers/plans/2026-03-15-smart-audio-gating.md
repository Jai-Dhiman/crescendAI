# Smart Audio Gating Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect piano playing via audio energy analysis, gate chunk uploads to avoid processing silence, and replace the wave visualization with organic ripples.

**Architecture:** A `useAudioActivity` hook computes spectral energy from the existing `AnalyserNode` and exposes `isPlaying` / `energy`. `usePracticeSession` calls it internally and switches from automatic 15s timeslice chunking to manual `requestData()` gated by `isPlaying`. A new `ResonanceRipples` canvas component replaces `FlowingWaves`. All dev logging goes through a `createLogger` utility gated by `import.meta.env.DEV`.

**Tech Stack:** React hooks, Web Audio API (`AnalyserNode`), Canvas 2D, MediaRecorder API (`requestData()`), TypeScript

**Spec:** `docs/superpowers/specs/2026-03-15-smart-audio-gating-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `apps/web/src/lib/logger.ts` | CREATE | Dev-only tagged console logger |
| `apps/web/src/hooks/useAudioActivity.ts` | CREATE | Spectral energy computation + debounced play detection |
| `apps/web/src/hooks/usePracticeSession.ts` | MODIFY | Integrate `useAudioActivity`, switch to manual chunking, expose `isPlaying`/`energy` |
| `apps/web/src/components/ResonanceRipples.tsx` | CREATE | Organic ripple visualization driven by `energy` + `isPlaying` |
| `apps/web/src/components/ListeningMode.tsx` | MODIFY | Swap `FlowingWaves` for `ResonanceRipples`, pass new props |
| `apps/web/src/components/AppChat.tsx` | MODIFY | Pass `isPlaying`/`energy` from `practice` to child components |
| `apps/web/src/components/FlowingWaves.tsx` | DELETE | Replaced by `ResonanceRipples` |
| `apps/web/src/components/RecordingBar.tsx` | DELETE | Dead code -- never imported, superseded by `ListeningMode` |

---

## Chunk 1: Logger + Audio Activity Detection

### Task 1: Dev Logger Utility

**Files:**
- Create: `apps/web/src/lib/logger.ts`

- [ ] **Step 1: Create the logger utility**

```ts
// apps/web/src/lib/logger.ts

interface Logger {
	log: (...args: unknown[]) => void;
	warn: (...args: unknown[]) => void;
	error: (...args: unknown[]) => void;
}

export function createLogger(tag: string): Logger {
	const prefix = `[${tag}]`;

	if (!import.meta.env.DEV) {
		const noop = () => {};
		return { log: noop, warn: noop, error: noop };
	}

	return {
		log: (...args: unknown[]) => console.log(prefix, ...args),
		warn: (...args: unknown[]) => console.warn(prefix, ...args),
		error: (...args: unknown[]) => console.error(prefix, ...args),
	};
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd apps/web && bun run build 2>&1 | head -20`
Expected: No errors related to `logger.ts`

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/lib/logger.ts
git commit -m "add dev-only tagged console logger utility"
```

---

### Task 2: Audio Activity Detection Hook

**Files:**
- Create: `apps/web/src/hooks/useAudioActivity.ts`

- [ ] **Step 1: Create the hook**

```ts
// apps/web/src/hooks/useAudioActivity.ts

import { useEffect, useRef, useState } from "react";
import { createLogger } from "../lib/logger";

const log = createLogger("AudioActivity");

/** Spectral energy threshold (0-1). Piano audio should comfortably exceed this. */
export const ENERGY_THRESHOLD = 0.04;

/** Frames above threshold before onset triggers (~150ms at 60fps) */
const ONSET_FRAMES = 4;

/** Milliseconds below threshold before offset triggers */
const OFFSET_MS = 2000;

export interface AudioActivityState {
	isPlaying: boolean;
	energy: number;
}

export function useAudioActivity(
	analyserNodeRef: React.RefObject<AnalyserNode | null>,
): AudioActivityState {
	const [isPlaying, setIsPlaying] = useState(false);
	const [energy, setEnergy] = useState(0);

	const isPlayingRef = useRef(false);
	const onsetCountRef = useRef(0);
	const offsetStartRef = useRef<number | null>(null);
	const rafRef = useRef<number>(0);
	const lastLogTimeRef = useRef(0);

	useEffect(() => {
		const dataArrayRef: { current: Uint8Array | null } = { current: null };

		function tick(timestamp: number) {
			rafRef.current = requestAnimationFrame(tick);

			const analyser = analyserNodeRef.current;
			if (!analyser) return;

			// Lazily allocate data array
			if (!dataArrayRef.current || dataArrayRef.current.length !== analyser.frequencyBinCount) {
				dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);
			}

			// Compute spectral energy
			analyser.getByteFrequencyData(dataArrayRef.current);
			let sum = 0;
			for (let i = 0; i < dataArrayRef.current.length; i++) {
				sum += dataArrayRef.current[i];
			}
			const rawEnergy = sum / (dataArrayRef.current.length * 255);

			setEnergy(rawEnergy);

			const aboveThreshold = rawEnergy > ENERGY_THRESHOLD;

			if (!isPlayingRef.current) {
				// Currently silent -- check for onset
				if (aboveThreshold) {
					onsetCountRef.current++;
					if (onsetCountRef.current >= ONSET_FRAMES) {
						isPlayingRef.current = true;
						setIsPlaying(true);
						onsetCountRef.current = 0;
						offsetStartRef.current = null;
						log.log(`PLAYING detected (sustained ${ONSET_FRAMES} frames)`);
					} else {
						log.log(
							`Energy: ${rawEnergy.toFixed(3)} (above threshold ${ENERGY_THRESHOLD}) -- onset debounce ${onsetCountRef.current}/${ONSET_FRAMES}`,
						);
					}
				} else {
					onsetCountRef.current = 0;
					// Throttle idle energy logs to 1/sec
					if (timestamp - lastLogTimeRef.current > 1000) {
						log.log(
							`Energy: ${rawEnergy.toFixed(3)} (below threshold ${ENERGY_THRESHOLD}) -- idle`,
						);
						lastLogTimeRef.current = timestamp;
					}
				}
			} else {
				// Currently playing -- check for offset
				if (!aboveThreshold) {
					if (offsetStartRef.current === null) {
						offsetStartRef.current = timestamp;
					}
					const silenceDuration = timestamp - offsetStartRef.current;
					if (silenceDuration >= OFFSET_MS) {
						isPlayingRef.current = false;
						setIsPlaying(false);
						offsetStartRef.current = null;
						log.log(
							`SILENCE detected (sustained ${(silenceDuration / 1000).toFixed(1)}s)`,
						);
					}
				} else {
					// Reset offset timer -- still playing
					offsetStartRef.current = null;
				}
			}
		}

		rafRef.current = requestAnimationFrame(tick);

		return () => {
			cancelAnimationFrame(rafRef.current);
		};
	}, [analyserNodeRef]);

	return { isPlaying, energy };
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd apps/web && bun run build 2>&1 | head -20`
Expected: No errors related to `useAudioActivity.ts`

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/hooks/useAudioActivity.ts
git commit -m "add useAudioActivity hook with spectral energy and debounced play detection"
```

---

## Chunk 2: Smart Chunking in usePracticeSession

### Task 3: Integrate Audio Activity + Manual Chunking

**Files:**
- Modify: `apps/web/src/hooks/usePracticeSession.ts`

This is the largest change. We need to:
1. Call `useAudioActivity` internally using the analyser ref
2. Switch from `recorder.start(15000)` to `recorder.start()` (continuous)
3. Add a chunk gate state machine (WAITING/BUFFERING) driven by `isPlaying`
4. Update `stop()` to handle zero-chunk flush
5. Expose `isPlaying` and `energy` in the return type

- [ ] **Step 1: Add imports, refs, and state for chunk gating**

At the top of `usePracticeSession.ts`, add import for `useAudioActivity` and `createLogger`. Add a `ChunkGateState` type and new refs.

Add after line 9 (`import { Sentry } from "../lib/sentry";`):
```ts
import { useAudioActivity } from "./useAudioActivity";
import { createLogger } from "../lib/logger";

const chunkLog = createLogger("ChunkGate");

type ChunkGateState = "waiting" | "buffering";
```

- [ ] **Step 2: Add analyser ref and call useAudioActivity**

Currently `analyserNode` is stored via `useState` (line 54). We need a ref so `useAudioActivity` can read it. Keep the state for backward compat (it's in the return type), but add a ref that `useAudioActivity` consumes.

Add after line 73 (`const offlineQueueRef = ...`):
```ts
const analyserRef = useRef<AnalyserNode | null>(null);
const chunkGateRef = useRef<ChunkGateState>("waiting");
const chunkTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
```

Add after the `isOnlineRef` sync effect (after line 82):
```ts
// Audio activity detection (spectral energy + debounced play/silence)
const { isPlaying, energy } = useAudioActivity(analyserRef);
const isPlayingRef = useRef(false);

useEffect(() => {
	isPlayingRef.current = isPlaying;
}, [isPlaying]);
```

- [ ] **Step 3: Add chunk gate effect driven by isPlaying**

This is the core state machine. Add after the online/offline effect (after line 98):

```ts
// Chunk gating: start/stop uploading based on piano activity
useEffect(() => {
	// Only gate when actively recording
	if (state !== "recording") return;
	const recorder = mediaRecorderRef.current;
	const sessionId = sessionIdRef.current;
	if (!recorder || !sessionId) return;

	if (isPlaying && chunkGateRef.current === "waiting") {
		// Transition: WAITING -> BUFFERING
		chunkGateRef.current = "buffering";
		chunkLog.log("State: WAITING -> BUFFERING");

		// Start 15s interval timer. First requestData() includes accumulated
		// silence -- that's fine per spec (no discard mechanism needed).
		chunkLog.log("Chunk timer started (15s)");
		chunkTimerRef.current = setInterval(() => {
			if (recorder.state === "recording") {
				recorder.requestData();
			}
		}, 15000);
	} else if (!isPlaying && chunkGateRef.current === "buffering") {
		// Transition: BUFFERING -> WAITING
		chunkLog.log("State: BUFFERING -> WAITING (silence offset)");

		// Clear the chunk timer
		if (chunkTimerRef.current) {
			clearInterval(chunkTimerRef.current);
			chunkTimerRef.current = null;
		}

		// Flush partial chunk
		if (recorder.state === "recording") {
			recorder.requestData();
			chunkLog.log("Partial chunk flushed on silence offset");
		}

		chunkGateRef.current = "waiting";
	}
}, [isPlaying, state]);
```

- [ ] **Step 4: Update start() -- switch to continuous MediaRecorder**

In the `start` callback, change `recorder.start(15000)` to `recorder.start()` (line 374). Also set the analyser ref when creating it. Also reset the chunk gate state.

Change the analyser setup section (around line 283-286) -- add the ref assignment after `setAnalyserNode(analyser)`:
```ts
analyserRef.current = analyser;
```

Change `recorder.start(15000);` (line 374) to:
```ts
recorder.start(); // Continuous mode -- chunks cut manually via requestData()
chunkGateRef.current = "waiting";
chunkLog.log("MediaRecorder started in continuous mode (gated by audio activity)");
```

- [ ] **Step 5: Add logging to ondataavailable**

Update the `ondataavailable` handler to log chunk details. Add after `const idx = chunkIndexRef.current++;` (line 363):
```ts
chunkLog.log(
	`Chunk #${idx} cut: ${(event.data.size / 1024).toFixed(1)}KB -- uploading to R2`,
);
```

And in the `uploadWithRetry` function, after the successful upload (after `updateChunkState(idx, "complete")`), add:
```ts
chunkLog.log(`Upload complete: chunk #${idx} -> r2Key=${r2Key}`);
```

And after the WS send:
```ts
chunkLog.log(`WS sent: chunk_ready #${idx}`);
```

- [ ] **Step 6: Update stop() for zero-chunk flush**

Replace the zero-chunk guard in `stop()` (lines 424-428) with:
```ts
// Minimum blob size to consider as real audio (~1s of Opus audio)
const MIN_FLUSH_BLOB_SIZE = 10_000; // ~10KB

if (chunkIndexRef.current === 0) {
	// No chunks sent -- flush buffer as last resort
	const recorder = mediaRecorderRef.current;
	if (recorder?.state === "recording") {
		// Temporarily override ondataavailable to check blob size
		const origHandler = recorder.ondataavailable;
		recorder.ondataavailable = async (event) => {
			recorder.ondataavailable = origHandler; // restore
			if (event.data.size < MIN_FLUSH_BLOB_SIZE) {
				// Too small -- likely just silence
				chunkLog.log(
					`Flush blob too small: ${(event.data.size / 1024).toFixed(1)}KB < ${(MIN_FLUSH_BLOB_SIZE / 1024).toFixed(0)}KB threshold`,
				);
				setError(
					"I couldn't hear any playing. Make sure your microphone is picking up the piano.",
				);
				setState("idle");
				cleanup();
			} else {
				// Blob has content -- upload and proceed normally
				chunkLog.log(
					`Flush blob accepted: ${(event.data.size / 1024).toFixed(1)}KB -- uploading`,
				);
				if (origHandler) {
					origHandler.call(recorder, event);
				}
				setState("summarizing");
				if (wsRef.current?.readyState === WebSocket.OPEN) {
					wsRef.current.send(JSON.stringify({ type: "end_session" }));
				}
			}
		};
		recorder.requestData();
	} else {
		setError(
			"I couldn't hear any playing. Make sure your microphone is picking up the piano.",
		);
		setState("idle");
		cleanup();
	}
	return;
}
```

- [ ] **Step 7: Update cleanup() to clear chunk gate state**

In the `cleanup` callback, add after `offlineQueueRef.current = [];` (line 132):
```ts
if (chunkTimerRef.current) {
	clearInterval(chunkTimerRef.current);
	chunkTimerRef.current = null;
}
chunkGateRef.current = "waiting";
analyserRef.current = null;
```

- [ ] **Step 8: Update return type and return statement**

Add to the `UsePracticeSessionReturn` interface (after line 41: `isOnline: boolean;`):
```ts
isPlaying: boolean;
energy: number;
```

Add to the return object (after `isOnline,`):
```ts
isPlaying,
energy,
```

Remove `analyserNode` from the return type and return object -- it's no longer needed externally since `ResonanceRipples` will consume `energy` instead. But wait -- we should keep it for now and remove it in the integration task (Task 5) to avoid breaking the build mid-task.

Actually, keep `analyserNode` in the return for now. We'll remove it in Task 5 when we swap the components.

- [ ] **Step 9: Verify it compiles**

Run: `cd apps/web && bun run build 2>&1 | head -20`
Expected: No errors (warnings about unused `isPlaying`/`energy` are fine at this stage)

- [ ] **Step 10: Commit**

```bash
git add apps/web/src/hooks/usePracticeSession.ts
git commit -m "integrate audio activity detection and manual chunk gating into practice session"
```

---

## Chunk 3: Resonance Ripples Visualization

### Task 4: Create ResonanceRipples Component

**Files:**
- Create: `apps/web/src/components/ResonanceRipples.tsx`

- [ ] **Step 1: Create the component**

```tsx
// apps/web/src/components/ResonanceRipples.tsx

import { useEffect, useRef } from "react";
import { createLogger } from "../lib/logger";

const log = createLogger("Ripples");

interface ResonanceRipplesProps {
	energy: number;
	isPlaying: boolean;
	active: boolean;
}

// Sage green RGB
const SAGE_R = 122;
const SAGE_G = 154;
const SAGE_B = 130;

// Idle ripple interval (ms)
const IDLE_RIPPLE_INTERVAL = 3000;
// Min interval between active ripples at peak energy (ms)
const MIN_RIPPLE_INTERVAL = 400;
// Max interval between active ripples at low energy (ms)
const MAX_RIPPLE_INTERVAL = 1500;
// Ripple expansion speed (px/sec)
const EXPANSION_SPEED = 80;
// Max ripple radius as fraction of canvas min dimension
const MAX_RADIUS_FRACTION = 0.45;
// Wobble noise amplitude as fraction of radius
const WOBBLE_AMOUNT = 0.04;
// Idle throttle: skip frames when delta < this (ms) for ~15fps
const IDLE_FRAME_MIN_MS = 66;

interface Ripple {
	birthTime: number;
	maxRadius: number;
	wobblePhase1: number;
	wobblePhase2: number;
	baseOpacity: number;
}

export function ResonanceRipples({
	energy,
	isPlaying,
	active,
}: ResonanceRipplesProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const rafRef = useRef<number>(0);
	const ripplesRef = useRef<Ripple[]>([]);
	const lastRippleTimeRef = useRef(0);
	const lastFrameTimeRef = useRef(0);
	const frameCountRef = useRef(0);

	// Store props in refs so the rAF loop reads fresh values without
	// restarting the effect on every energy change (~60 times/sec).
	const energyRef = useRef(energy);
	const isPlayingRef = useRef(isPlaying);
	energyRef.current = energy;
	isPlayingRef.current = isPlaying;

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;

		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		function draw(timestamp: number) {
			rafRef.current = requestAnimationFrame(draw);

			const dt = lastFrameTimeRef.current
				? timestamp - lastFrameTimeRef.current
				: 16;

			const currentIsPlaying = isPlayingRef.current;
			const currentEnergy = energyRef.current;

			// Throttle to ~15fps when idle
			if (!currentIsPlaying && dt < IDLE_FRAME_MIN_MS) return;

			lastFrameTimeRef.current = timestamp;

			// Resize canvas for DPR
			const rect = canvas!.getBoundingClientRect();
			const dpr = window.devicePixelRatio || 1;
			const w = rect.width * dpr;
			const h = rect.height * dpr;

			if (canvas!.width !== w || canvas!.height !== h) {
				canvas!.width = w;
				canvas!.height = h;
			}

			ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
			ctx!.clearRect(0, 0, rect.width, rect.height);

			const centerX = rect.width / 2;
			const centerY = rect.height / 2;
			const maxRadius =
				Math.min(rect.width, rect.height) * MAX_RADIUS_FRACTION;

			// Spawn new ripples
			const timeSinceLastRipple = timestamp - lastRippleTimeRef.current;

			if (currentIsPlaying) {
				// Active: ripple frequency scales with energy
				const interval =
					MAX_RIPPLE_INTERVAL -
					(MAX_RIPPLE_INTERVAL - MIN_RIPPLE_INTERVAL) *
						Math.min(currentEnergy * 5, 1); // energy ~0.04-0.2 maps to 0-1
				if (timeSinceLastRipple > interval) {
					ripplesRef.current.push({
						birthTime: timestamp,
						maxRadius: maxRadius * (0.6 + currentEnergy * 4 * 0.4), // bigger at higher energy
						wobblePhase1: Math.random() * Math.PI * 2,
						wobblePhase2: Math.random() * Math.PI * 2,
						baseOpacity: Math.min(0.15 + currentEnergy * 3, 0.55), // brighter at higher energy
					});
					lastRippleTimeRef.current = timestamp;
				}
			} else {
				// Idle: one faint ripple every ~3s
				if (timeSinceLastRipple > IDLE_RIPPLE_INTERVAL) {
					ripplesRef.current.push({
						birthTime: timestamp,
						maxRadius: maxRadius * 0.5,
						wobblePhase1: Math.random() * Math.PI * 2,
						wobblePhase2: Math.random() * Math.PI * 2,
						baseOpacity: 0.1,
					});
					lastRippleTimeRef.current = timestamp;
				}
			}

			// Draw and cull ripples
			const alive: Ripple[] = [];

			for (const ripple of ripplesRef.current) {
				const age = (timestamp - ripple.birthTime) / 1000; // seconds
				const radius = age * EXPANSION_SPEED;

				if (radius > ripple.maxRadius) continue; // expired

				alive.push(ripple);

				// Opacity fades linearly with radius
				const progress = radius / ripple.maxRadius;
				const opacity = ripple.baseOpacity * (1 - progress);

				if (opacity < 0.005) continue; // invisible

				// Draw organic ring
				ctx!.beginPath();
				ctx!.strokeStyle = `rgba(${SAGE_R}, ${SAGE_G}, ${SAGE_B}, ${opacity})`;
				ctx!.lineWidth = 1.8;

				const steps = 64;
				const wobbleNoise = radius * WOBBLE_AMOUNT;

				for (let i = 0; i <= steps; i++) {
					const theta = (i / steps) * Math.PI * 2;
					const wobble =
						wobbleNoise *
						(Math.sin(theta * 3 + ripple.wobblePhase1) * 0.6 +
							Math.sin(theta * 5 + ripple.wobblePhase2) * 0.4);
					const r = radius + wobble;
					const x = centerX + Math.cos(theta) * r;
					const y = centerY + Math.sin(theta) * r;

					if (i === 0) {
						ctx!.moveTo(x, y);
					} else {
						ctx!.lineTo(x, y);
					}
				}

				ctx!.closePath();
				ctx!.stroke();
			}

			ripplesRef.current = alive;

			// Dev logging (throttled)
			frameCountRef.current++;
			if (frameCountRef.current % 60 === 0) {
				log.log(
					`Active ripples: ${alive.length}, energy: ${currentEnergy.toFixed(3)}, isPlaying: ${currentIsPlaying}`,
				);
			}
		}

		if (active) {
			rafRef.current = requestAnimationFrame(draw);
		}

		return () => {
			cancelAnimationFrame(rafRef.current);
			lastFrameTimeRef.current = 0;
		};
	}, [active]); // Only depends on active -- energy/isPlaying read from refs

	return (
		<canvas
			ref={canvasRef}
			className="block w-full h-full"
			style={{ width: "100%", height: "100%" }}
		/>
	);
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd apps/web && bun run build 2>&1 | head -20`
Expected: No errors (component not yet imported anywhere)

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/ResonanceRipples.tsx
git commit -m "add ResonanceRipples organic ripple visualization component"
```

---

## Chunk 4: Integration + Cleanup

### Task 5: Swap Components and Wire Props

**Files:**
- Modify: `apps/web/src/components/ListeningMode.tsx`
- Modify: `apps/web/src/components/AppChat.tsx`
- Delete: `apps/web/src/components/RecordingBar.tsx` (dead code)
- Delete: `apps/web/src/components/FlowingWaves.tsx`

- [ ] **Step 1: Update ListeningMode props and import**

In `apps/web/src/components/ListeningMode.tsx`:

Replace the import (line 13):
```ts
import { FlowingWaves } from "./FlowingWaves";
```
with:
```ts
import { ResonanceRipples } from "./ResonanceRipples";
```

In the `ListeningModeProps` interface, replace:
```ts
analyserNode: AnalyserNode | null;
```
with:
```ts
energy: number;
isPlaying: boolean;
```

Update the destructured props in the function signature -- replace `analyserNode,` with `energy, isPlaying,`.

Replace the FlowingWaves usage (line 289):
```tsx
<FlowingWaves analyserNode={analyserNode} active={isRecording} />
```
with:
```tsx
<ResonanceRipples energy={energy} isPlaying={isPlaying} active={isRecording} />
```

- [ ] **Step 2: Delete RecordingBar (dead code)**

`RecordingBar` is exported but never imported by any other component -- it was superseded by `ListeningMode`. Delete it alongside `FlowingWaves`:

```bash
git rm apps/web/src/components/RecordingBar.tsx
```

- [ ] **Step 3: Update AppChat to pass new props**

In `apps/web/src/components/AppChat.tsx`, update the `ListeningMode` usage (around line 723-736).

Replace:
```tsx
analyserNode={practice.analyserNode}
```
with:
```tsx
energy={practice.energy}
isPlaying={practice.isPlaying}
```

- [ ] **Step 4: Remove analyserNode from usePracticeSession return type**

In `apps/web/src/hooks/usePracticeSession.ts`:

Remove `analyserNode: AnalyserNode | null;` from `UsePracticeSessionReturn` interface.

Remove `analyserNode,` from the return object.

Remove `const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);` (line 54). Keep `setAnalyserNode` only if it's used to set state for other consumers -- but since we're removing it from the return, we can remove the state entirely. The analyser is stored in `analyserRef` for `useAudioActivity`. However, we still need `setAnalyserNode` to be called somewhere -- actually we don't. Just remove the state and keep the ref.

In `start()`, replace `setAnalyserNode(analyser);` (line 286) with just having the ref assignment (which was added in Task 3 Step 2): `analyserRef.current = analyser;`

In `cleanup()`, add `analyserRef.current = null;` if not already there (was added in Task 3 Step 7).

- [ ] **Step 5: Delete FlowingWaves**

(RecordingBar was already deleted in Step 2.)

```bash
git rm apps/web/src/components/FlowingWaves.tsx
```

- [ ] **Step 6: Verify full build**

Run: `cd apps/web && bun run build 2>&1 | tail -20`
Expected: Build succeeds with no errors. No references to `FlowingWaves` remain.

Verify no remaining references:
Run: `cd apps/web && grep -r "FlowingWaves" src/`
Expected: No output

- [ ] **Step 7: Commit**

```bash
git add -A apps/web/src/
git commit -m "swap FlowingWaves for ResonanceRipples and wire audio activity props through component tree"
```

---

### Task 6: Manual QA Verification

- [ ] **Step 1: Start dev server**

Run: `cd apps/web && bun run dev`

- [ ] **Step 2: Test idle state**

Open the app, start a recording session (click record). Do NOT play any audio.

Verify in browser console:
- `[AudioActivity] Energy: 0.00x (below threshold 0.04) -- idle` logs appearing ~1/sec
- `[ChunkGate] MediaRecorder started in continuous mode` appears
- NO `[ChunkGate] Chunk #N` logs (no chunks uploaded during silence)

Verify visually:
- Faint ripple appears every ~3 seconds from center
- Ripple edges are slightly wobbly (organic, not perfect circles)
- Ripple fades as it expands

- [ ] **Step 3: Test playing state**

Play audio (piano, or any audio source near the mic).

Verify in browser console:
- `[AudioActivity] PLAYING detected` appears
- `[ChunkGate] State: WAITING -> BUFFERING` appears
- `[ChunkGate] Chunk #0 cut` appears after 15s
- Chunk uploads proceed normally

Verify visually:
- Ripples become more frequent and brighter
- Ripple frequency responds to volume

- [ ] **Step 4: Test silence offset**

Stop playing audio. Wait 2+ seconds.

Verify in browser console:
- `[AudioActivity] SILENCE detected (sustained 2.0s)` appears
- `[ChunkGate] State: BUFFERING -> WAITING` appears
- `[ChunkGate] Partial chunk flushed on silence offset` appears
- Chunk uploads stop

Verify visually:
- Ripples return to idle (faint, infrequent)

- [ ] **Step 5: Test stop with zero chunks**

Start a new session. Do NOT play audio. Hit stop within 15 seconds.

Verify: Error message "I couldn't hear any playing. Make sure your microphone is picking up the piano." appears (not the old "Play for at least 15 seconds" message).

- [ ] **Step 6: Commit any fixes if needed**

```bash
git add -A apps/web/src/
git commit -m "fix: [describe any QA fixes]"
```
