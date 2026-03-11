# Listening Mode Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the recording top-bar with a full-screen listening mode that radially expands from the record button, providing a focused practice environment with metronome, notepad, dimension scores, and piece context.

**Architecture:** New `ListeningMode` component rendered as a portal overlay, orchestrated by `AppChat`. The existing `usePracticeSession` hook is unchanged -- `ListeningMode` consumes its output. A new `useMetronome` hook handles audio scheduling via a separate `AudioContext`. `ChatInput` forwards a ref to the record button for transition origin.

**Tech Stack:** React 19, Tailwind CSS v4, Web Audio API, CSS `clip-path` animations, Phosphor Icons, createPortal

**Spec:** `docs/superpowers/specs/2026-03-10-listening-mode-design.md`

---

## Chunk 1: Radial Transition + Shell

### File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `apps/web/src/components/ListeningMode.tsx` | Full-screen overlay shell with radial transition |
| Modify | `apps/web/src/components/ChatInput.tsx` | Forward ref to record button |
| Modify | `apps/web/src/components/AppChat.tsx` | Orchestrate transition, replace RecordingBar with ListeningMode |
| Modify | `apps/web/src/styles/app.css` | Add radial expand/collapse + content fade animations |

---

### Task 1: Add radial expand/collapse CSS animations

**Files:**
- Modify: `apps/web/src/styles/app.css:256` (append after existing animations)

- [ ] **Step 1: Add keyframes and animation classes for the radial transition**

Append to the end of `apps/web/src/styles/app.css`:

```css
/* Listening mode transitions */
@keyframes listening-content-in {
	from { opacity: 0; }
	to { opacity: 1; }
}

.animate-listening-content-in {
	animation: listening-content-in 300ms ease-out both;
}

.listening-overlay {
	position: fixed;
	inset: 0;
	z-index: 50;
	background-color: var(--color-espresso);
}

/* Radial clip-path transition (GPU-accelerated, no layout thrashing) */
.listening-overlay[data-transition="expanding"] {
	clip-path: circle(150vmax at var(--origin-x, 50%) var(--origin-y, 50%));
	transition: clip-path 600ms cubic-bezier(0.4, 0, 0.2, 1);
}

.listening-overlay[data-transition="collapsed"] {
	clip-path: circle(0% at var(--origin-x, 50%) var(--origin-y, 50%));
	transition: clip-path 600ms cubic-bezier(0.4, 0, 0.2, 1);
}

.listening-overlay[data-transition="open"] {
	clip-path: none;
}

/* Dim scores pulse on update */
@keyframes score-pulse {
	0% { color: var(--color-accent-lighter); }
	100% { color: var(--color-accent); }
}

.animate-score-pulse {
	animation: score-pulse 600ms ease-out;
}
```

- [ ] **Step 2: Verify CSS compiles**

Run: `cd apps/web && bun run build 2>&1 | head -20`
Expected: No CSS errors

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/styles/app.css
git commit -m "feat: add listening mode CSS animations (radial clip-path, score pulse)"
```

---

### Task 2: Forward record button ref from ChatInput

**Files:**
- Modify: `apps/web/src/components/ChatInput.tsx:1-91`

- [ ] **Step 1: Add `recordButtonRef` prop to ChatInput**

In `ChatInput.tsx`, update the interface and component to accept and attach a ref:

Change the interface (line 4-10):
```typescript
interface ChatInputProps {
	onSend: (message: string) => void;
	onRecord?: () => void;
	disabled: boolean;
	placeholder?: string;
	centered?: boolean;
	recordButtonRef?: React.RefObject<HTMLButtonElement | null>;
}
```

Update the destructuring (line 12-18) to include `recordButtonRef`:
```typescript
export function ChatInput({
	onSend,
	onRecord,
	disabled,
	placeholder,
	centered,
	recordButtonRef,
}: ChatInputProps) {
```

Add `ref={recordButtonRef}` to the record button element (line 79-87):
```tsx
{!hasText && (
	<button
		ref={recordButtonRef}
		type="button"
		onClick={onRecord}
		className="shrink-0 w-16 h-16 flex items-center justify-center rounded-full bg-accent text-on-accent hover:brightness-110 transition animate-pop-in"
		aria-label="Record audio"
	>
		<Waveform size={24} />
	</button>
)}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit 2>&1 | head -20`
Expected: No type errors (existing usages of ChatInput don't pass recordButtonRef, which is optional)

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/ChatInput.tsx
git commit -m "feat: add recordButtonRef prop to ChatInput"
```

---

### Task 3: Create ListeningMode shell component

**Files:**
- Create: `apps/web/src/components/ListeningMode.tsx`

- [ ] **Step 1: Create ListeningMode with radial transition and basic layout**

Create `apps/web/src/components/ListeningMode.tsx`:

```tsx
import { Stop } from "@phosphor-icons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { PracticeState } from "../hooks/usePracticeSession";
import type { DimScores, ObservationEvent } from "../lib/practice-api";
import { FlowingWaves } from "./FlowingWaves";
import { ObservationToast } from "./ObservationToast";

interface ListeningModeProps {
	state: PracticeState;
	observations: ObservationEvent[];
	analyserNode: AnalyserNode | null;
	latestScores: DimScores | null;
	error: string | null;
	onStop: () => void;
	originRect: DOMRect | null;
	onExit: () => void;
}

type TransitionPhase = "collapsed" | "expanding" | "open" | "collapsing";

export function ListeningMode({
	state,
	observations,
	analyserNode,
	latestScores,
	error,
	onStop,
	originRect,
	onExit,
}: ListeningModeProps) {
	const [phase, setPhase] = useState<TransitionPhase>("collapsed");
	const [contentVisible, setContentVisible] = useState(false);
	const overlayRef = useRef<HTMLDivElement>(null);
	const [dismissedIds, setDismissedIds] = useState<Set<number>>(new Set());
	const [notes, setNotes] = useState("");
	const [showNotepad, setShowNotepad] = useState(false);

	// Compute origin point for clip-path (center of record button, or fallback)
	const originX = originRect
		? ((originRect.left + originRect.width / 2) / window.innerWidth) * 100
		: 50;
	const originY = originRect
		? ((originRect.top + originRect.height / 2) / window.innerHeight) * 100
		: 90;

	// Open transition sequence
	useEffect(() => {
		// Start collapsed, then expand on next frame
		requestAnimationFrame(() => {
			setPhase("expanding");
		});

		const expandTimer = setTimeout(() => {
			setPhase("open");
			setContentVisible(true);
		}, 650); // slightly after 600ms transition

		return () => clearTimeout(expandTimer);
	}, []);

	// Exit on error or WebSocket disconnect
	useEffect(() => {
		if (state === "error") {
			handleClose();
		}
	}, [state, handleClose]);

	// Close transition sequence
	const handleClose = useCallback(() => {
		setContentVisible(false);
		setPhase("collapsing");

		setTimeout(() => {
			setPhase("collapsed");
			setTimeout(() => {
				onStop();
				onExit();
			}, 50);
		}, 600);
	}, [onStop, onExit]);

	// Stop recording then animate out
	const handleStop = useCallback(() => {
		onStop();
		setContentVisible(false);
		setPhase("collapsing");

		setTimeout(() => {
			setPhase("collapsed");
			setTimeout(onExit, 50);
		}, 600);
	}, [onStop, onExit]);

	const handleDismiss = useCallback((idx: number) => {
		setDismissedIds((prev) => new Set(prev).add(idx));
	}, []);

	const visibleObservations = observations
		.map((obs, idx) => ({ ...obs, idx }))
		.filter(({ idx }) => !dismissedIds.has(idx))
		.slice(-3);

	const isRecording = state === "recording";

	// Screen wake lock
	useEffect(() => {
		let wakeLock: WakeLockSentinel | null = null;

		async function requestWakeLock() {
			try {
				if ("wakeLock" in navigator) {
					wakeLock = await navigator.wakeLock.request("screen");
				}
			} catch {
				// Progressive enhancement -- fail silently
			}
		}

		requestWakeLock();
		return () => {
			wakeLock?.release();
		};
	}, []);

	const transitionAttr =
		phase === "collapsing" ? "collapsed" : phase === "expanding" ? "expanding" : phase;

	return createPortal(
		<div
			ref={overlayRef}
			className="listening-overlay"
			data-transition={transitionAttr}
			style={{
				"--origin-x": `${originX}%`,
				"--origin-y": `${originY}%`,
			} as React.CSSProperties}
		>
			{contentVisible && (
				<div className="h-dvh flex flex-col animate-listening-content-in">
					{/* Top bar: piece info */}
					<div className="shrink-0 flex items-center justify-between px-6 py-4 border-b border-border">
						<div className="flex items-center gap-3">
							{/* Metronome placeholder -- Task 6 */}
							<div className="w-10 h-10 rounded-lg bg-surface flex items-center justify-center text-text-secondary text-body-sm">
								M
							</div>
						</div>
						<div className="text-right">
							<span className="text-label-sm text-text-tertiary uppercase tracking-wider">
								Now practicing
							</span>
							<div className="text-body-sm text-cream">
								Unknown piece
							</div>
						</div>
					</div>

					{/* Center: waveform + scores */}
					<div className="flex-1 flex flex-col items-center justify-center px-6 relative">
						{/* Observation toasts */}
						<div className="absolute top-4 right-4 flex flex-col gap-3 z-10">
							{visibleObservations.map(({ idx, text, dimension }) => (
								<ObservationToast
									key={idx}
									text={text}
									dimension={dimension}
									onDismiss={() => handleDismiss(idx)}
								/>
							))}
						</div>

						{/* Waveform */}
						<div className="w-full max-w-3xl h-32 sm:h-40 md:h-48">
							<FlowingWaves analyserNode={analyserNode} active={isRecording} />
						</div>

						{/* Dimension scores */}
						<DimensionScores scores={latestScores} />
					</div>

					{/* Bottom bar: notes + stop */}
					<div className="shrink-0 flex items-center justify-between px-6 py-4 border-t border-border">
						<div>{/* Spacer */}</div>
						<div className="flex items-center gap-4">
							{/* Notepad toggle */}
							<button
								type="button"
								onClick={() => setShowNotepad(!showNotepad)}
								className="w-10 h-10 rounded-lg bg-surface flex items-center justify-center text-text-secondary hover:text-cream transition"
								aria-label="Toggle notepad"
							>
								<span className="text-body-sm">N</span>
							</button>
							{/* Stop button */}
							<button
								type="button"
								onClick={handleStop}
								disabled={!isRecording}
								className="w-14 h-14 flex items-center justify-center rounded-full bg-red-600 hover:bg-red-500 text-on-accent transition-colors disabled:opacity-50"
								aria-label="Stop recording"
							>
								<Stop size={22} weight="fill" />
							</button>
						</div>
					</div>

					{/* Notepad drawer */}
					{showNotepad && (
						<NotepadDrawer
							notes={notes}
							onChange={setNotes}
							onClose={() => setShowNotepad(false)}
						/>
					)}
				</div>
			)}
		</div>,
		document.body,
	);
}

// --- Sub-components ---

function DimensionScores({ scores }: { scores: DimScores | null }) {
	const prevScoresRef = useRef<DimScores | null>(null);
	const [pulsing, setPulsing] = useState<Set<string>>(new Set());

	useEffect(() => {
		if (!scores || !prevScoresRef.current) {
			prevScoresRef.current = scores;
			return;
		}

		const changed = new Set<string>();
		const prev = prevScoresRef.current;
		for (const key of Object.keys(scores) as (keyof DimScores)[]) {
			if (scores[key] !== prev[key]) {
				changed.add(key);
			}
		}

		if (changed.size > 0) {
			setPulsing(changed);
			const timer = setTimeout(() => setPulsing(new Set()), 600);
			prevScoresRef.current = scores;
			return () => clearTimeout(timer);
		}

		prevScoresRef.current = scores;
	}, [scores]);

	const dims: { key: keyof DimScores; label: string }[] = [
		{ key: "dynamics", label: "DYN" },
		{ key: "timing", label: "TIM" },
		{ key: "pedaling", label: "PED" },
		{ key: "articulation", label: "ART" },
		{ key: "phrasing", label: "PHR" },
		{ key: "interpretation", label: "INT" },
	];

	return (
		<div className="flex flex-wrap justify-center gap-x-6 gap-y-3 mt-6">
			{dims.map(({ key, label }) => (
				<div key={key} className="text-center">
					<div
						className={`text-body-md font-semibold text-accent tabular-nums ${
							pulsing.has(key) ? "animate-score-pulse" : ""
						}`}
					>
						{scores ? scores[key].toFixed(1) : "--"}
					</div>
					<div className="text-body-xs text-text-tertiary uppercase tracking-wider">
						{label}
					</div>
				</div>
			))}
		</div>
	);
}

function NotepadDrawer({
	notes,
	onChange,
	onClose,
}: {
	notes: string;
	onChange: (v: string) => void;
	onClose: () => void;
}) {
	const textareaRef = useRef<HTMLTextAreaElement>(null);

	useEffect(() => {
		textareaRef.current?.focus();
	}, []);

	return (
		<>
			{/* Backdrop */}
			<button
				type="button"
				className="fixed inset-0 z-40 bg-black/30"
				onClick={onClose}
				aria-label="Close notepad"
			/>
			{/* Drawer */}
			<div className="fixed bottom-0 left-0 right-0 z-50 bg-espresso border-t border-border rounded-t-2xl max-h-[40vh] md:max-h-[40vh] flex flex-col animate-overlay-in">
				<div className="flex items-center justify-between px-5 py-3 border-b border-border">
					<span className="text-label-sm text-text-tertiary uppercase tracking-wider">
						Notes
					</span>
					<button
						type="button"
						onClick={onClose}
						className="text-body-sm text-accent hover:text-accent-lighter transition"
					>
						Done
					</button>
				</div>
				<div className="flex-1 overflow-y-auto p-4">
					<textarea
						ref={textareaRef}
						value={notes}
						onChange={(e) => onChange(e.target.value)}
						placeholder="Jot down thoughts while you play..."
						className="w-full h-full min-h-[120px] bg-transparent text-body-sm text-cream placeholder:text-text-tertiary outline-none resize-none"
					/>
				</div>
			</div>
		</>
	);
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit 2>&1 | head -20`
Expected: No type errors

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/ListeningMode.tsx
git commit -m "feat: create ListeningMode component with radial transition, scores, notepad, observations"
```

---

### Task 4: Wire ListeningMode into AppChat, replace RecordingBar

**Files:**
- Modify: `apps/web/src/components/AppChat.tsx`

- [ ] **Step 1: Add ListeningMode state and ref wiring to AppChat**

In `AppChat.tsx`, make these changes:

1. Replace the RecordingBar import (line 30):
```typescript
import { ListeningMode } from "./ListeningMode";
```

2. Add a ref for the record button and state for listening mode. After line 107 (`const addToast = ...`), add:
```typescript
const recordButtonRef = useRef<HTMLButtonElement>(null);
const [showListeningMode, setShowListeningMode] = useState(false);
const [recordButtonRect, setRecordButtonRect] = useState<DOMRect | null>(null);
```

3. Update `handleRecord` (lines 173-175) to capture button rect and show listening mode:
```typescript
function handleRecord() {
	const rect = recordButtonRef.current?.getBoundingClientRect() ?? null;
	setRecordButtonRect(rect);
	setShowListeningMode(true);
	practice.start();
}
```

4. Add a handler to exit listening mode. After `handleRecord`:
```typescript
function handleExitListeningMode() {
	setShowListeningMode(false);
	setRecordButtonRect(null);
}
```

5. Update the practice summary effect (lines 177-188) to also inject notes:
```typescript
useEffect(() => {
	if (practice.summary) {
		const summaryMsg: RichMessage = {
			id: `practice-${Date.now()}`,
			role: "assistant",
			content: practice.summary,
			created_at: new Date().toISOString(),
		};
		setMessages((prev) => [...prev, summaryMsg]);
	}
}, [practice.summary]);
```

6. Replace the RecordingBar rendering block (lines 601-611) with:
```tsx
{showListeningMode && (
	<ListeningMode
		state={practice.state}
		observations={practice.observations}
		analyserNode={practice.analyserNode}
		latestScores={practice.latestScores}
		error={practice.error}
		onStop={practice.stop}
		originRect={recordButtonRect}
		onExit={handleExitListeningMode}
	/>
)}
```

7. Pass `recordButtonRef` to both ChatInput instances (lines 624-630 and 635-641):
```tsx
<ChatInput
	onSend={handleSend}
	onRecord={handleRecord}
	disabled={isStreaming || practice.state === "recording"}
	placeholder="What are you practicing today?"
	centered={true}
	recordButtonRef={recordButtonRef}
/>
```

And the second one:
```tsx
<ChatInput
	onSend={handleSend}
	onRecord={handleRecord}
	disabled={isStreaming || practice.state === "recording"}
	placeholder="Message your teacher..."
	centered={false}
	recordButtonRef={recordButtonRef}
/>
```

8. Remove the unused `RecordingBar` import (line 30, now replaced).

- [ ] **Step 2: Remove RecordingBar import from AppChat**

The `RecordingBar` import was on line 30. It should now be replaced with `ListeningMode`. Also remove the `RecordingBar` from imports at the top. The `RecordingBar.tsx` file itself stays (not deleted yet) in case there are other consumers, but AppChat no longer uses it.

- [ ] **Step 3: Verify TypeScript compiles and app builds**

Run: `cd apps/web && bunx tsc --noEmit 2>&1 | head -20`
Expected: No type errors

Run: `cd apps/web && bun run build 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/components/AppChat.tsx
git commit -m "feat: wire ListeningMode into AppChat, replace RecordingBar usage"
```

---

## Chunk 2: Metronome

### File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `apps/web/src/hooks/useMetronome.ts` | Web Audio API metronome with tap tempo, time sig, accent |
| Modify | `apps/web/src/components/ListeningMode.tsx` | Integrate metronome UI panel |

---

### Task 5: Create useMetronome hook

**Files:**
- Create: `apps/web/src/hooks/useMetronome.ts`

- [ ] **Step 1: Create the useMetronome hook**

Create `apps/web/src/hooks/useMetronome.ts`:

```typescript
import { useCallback, useEffect, useRef, useState } from "react";

interface MetronomeState {
	isPlaying: boolean;
	bpm: number;
	timeSignature: "4/4" | "3/4" | "6/8";
	accentFirstBeat: boolean;
	currentBeat: number;
}

export interface UseMetronomeReturn extends MetronomeState {
	start: () => void;
	stop: () => void;
	toggle: () => void;
	setBpm: (bpm: number) => void;
	adjustBpm: (delta: number) => void;
	setTimeSignature: (ts: "4/4" | "3/4" | "6/8") => void;
	setAccentFirstBeat: (accent: boolean) => void;
	tapTempo: () => void;
}

const MIN_BPM = 20;
const MAX_BPM = 300;
const TAP_WINDOW_MS = 2000;
const MIN_TAPS = 2;

export function useMetronome(): UseMetronomeReturn {
	const [isPlaying, setIsPlaying] = useState(false);
	const [bpm, setBpmState] = useState(120);
	const [timeSignature, setTimeSignature] = useState<"4/4" | "3/4" | "6/8">("4/4");
	const [accentFirstBeat, setAccentFirstBeat] = useState(true);
	const [currentBeat, setCurrentBeat] = useState(0);

	const audioCtxRef = useRef<AudioContext | null>(null);
	const nextBeatTimeRef = useRef(0);
	const schedulerRef = useRef<ReturnType<typeof setInterval> | null>(null);
	const beatCountRef = useRef(0);
	const tapTimesRef = useRef<number[]>([]);

	// Refs for values accessed in the scheduler interval (avoids stale closures)
	const bpmRef = useRef(bpm);
	const accentRef = useRef(accentFirstBeat);
	const beatsRef = useRef(4);

	const beatsPerMeasure = timeSignature === "6/8" ? 6 : timeSignature === "3/4" ? 3 : 4;

	// Keep refs in sync
	useEffect(() => { bpmRef.current = bpm; }, [bpm]);
	useEffect(() => { accentRef.current = accentFirstBeat; }, [accentFirstBeat]);
	useEffect(() => { beatsRef.current = beatsPerMeasure; }, [beatsPerMeasure]);

	function getOrCreateAudioCtx(): AudioContext {
		if (!audioCtxRef.current || audioCtxRef.current.state === "closed") {
			audioCtxRef.current = new AudioContext();
		}
		return audioCtxRef.current;
	}

	function playClick(time: number, accent: boolean) {
		const ctx = getOrCreateAudioCtx();
		const osc = ctx.createOscillator();
		const gain = ctx.createGain();

		osc.connect(gain);
		gain.connect(ctx.destination);

		// Higher pitch + louder for accent
		osc.frequency.value = accent ? 1000 : 800;
		osc.type = "sine";

		gain.gain.setValueAtTime(accent ? 0.6 : 0.3, time);
		gain.gain.exponentialRampToValueAtTime(0.001, time + 0.05);

		osc.start(time);
		osc.stop(time + 0.05);
	}

	function scheduleBeats() {
		const ctx = getOrCreateAudioCtx();
		const secondsPerBeat = 60 / bpmRef.current;
		const lookahead = 0.1; // schedule 100ms ahead

		while (nextBeatTimeRef.current < ctx.currentTime + lookahead) {
			const beatInMeasure = beatCountRef.current % beatsRef.current;
			const isAccent = accentRef.current && beatInMeasure === 0;

			playClick(nextBeatTimeRef.current, isAccent);
			setCurrentBeat(beatInMeasure);

			beatCountRef.current++;
			nextBeatTimeRef.current += secondsPerBeat;
		}
	}

	const start = useCallback(() => {
		const ctx = getOrCreateAudioCtx();
		if (ctx.state === "suspended") {
			ctx.resume();
		}

		beatCountRef.current = 0;
		nextBeatTimeRef.current = ctx.currentTime;
		setCurrentBeat(0);

		// Schedule at 25ms intervals for tight timing
		schedulerRef.current = setInterval(scheduleBeats, 25);
		setIsPlaying(true);
	}, []);

	const stop = useCallback(() => {
		if (schedulerRef.current) {
			clearInterval(schedulerRef.current);
			schedulerRef.current = null;
		}
		setIsPlaying(false);
		setCurrentBeat(0);
	}, []);

	const toggle = useCallback(() => {
		if (isPlaying) stop();
		else start();
	}, [isPlaying, start, stop]);

	// Restart scheduler when bpm/timeSignature/accent changes while playing
	useEffect(() => {
		if (!isPlaying) return;
		// Clear old scheduler and restart
		if (schedulerRef.current) {
			clearInterval(schedulerRef.current);
		}
		const ctx = getOrCreateAudioCtx();
		nextBeatTimeRef.current = ctx.currentTime;
		schedulerRef.current = setInterval(scheduleBeats, 25);
	}, [bpm, beatsPerMeasure, accentFirstBeat, isPlaying]);

	const setBpm = useCallback((value: number) => {
		setBpmState(Math.max(MIN_BPM, Math.min(MAX_BPM, Math.round(value))));
	}, []);

	const adjustBpm = useCallback((delta: number) => {
		setBpmState((prev) => Math.max(MIN_BPM, Math.min(MAX_BPM, prev + delta)));
	}, []);

	const tapTempo = useCallback(() => {
		const now = performance.now();
		const taps = tapTimesRef.current;

		// Remove stale taps
		while (taps.length > 0 && now - taps[0] > TAP_WINDOW_MS) {
			taps.shift();
		}

		taps.push(now);

		if (taps.length >= MIN_TAPS) {
			const intervals: number[] = [];
			for (let i = 1; i < taps.length; i++) {
				intervals.push(taps[i] - taps[i - 1]);
			}
			const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
			const newBpm = Math.round(60000 / avgInterval);
			setBpm(newBpm);
		}
	}, [setBpm]);

	// Cleanup on unmount
	useEffect(() => {
		return () => {
			if (schedulerRef.current) clearInterval(schedulerRef.current);
			if (audioCtxRef.current?.state !== "closed") {
				audioCtxRef.current?.close();
			}
		};
	}, []);

	return {
		isPlaying,
		bpm,
		timeSignature,
		accentFirstBeat,
		currentBeat,
		start,
		stop,
		toggle,
		setBpm,
		adjustBpm,
		setTimeSignature,
		setAccentFirstBeat,
		tapTempo,
	};
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit 2>&1 | head -20`
Expected: No type errors

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/hooks/useMetronome.ts
git commit -m "feat: create useMetronome hook with Web Audio scheduling, tap tempo, time signatures"
```

---

### Task 6: Integrate metronome UI into ListeningMode

**Files:**
- Modify: `apps/web/src/components/ListeningMode.tsx`

- [ ] **Step 1: Add metronome panel to ListeningMode**

In `ListeningMode.tsx`, make these changes:

1. Add import at the top:
```typescript
import { Metronome as MetronomeIcon, Minus, Plus } from "@phosphor-icons/react";
import { useMetronome } from "../hooks/useMetronome";
```

2. Inside the `ListeningMode` component, add:
```typescript
const metronome = useMetronome();
const [showMetronome, setShowMetronome] = useState(false);
```

3. Replace the metronome placeholder in the top bar (the `<div>` with "M") with:
```tsx
<div className="relative">
	<button
		type="button"
		onClick={() => setShowMetronome(!showMetronome)}
		className={`flex items-center gap-2 px-3 py-2 rounded-lg transition ${
			metronome.isPlaying
				? "bg-accent/20 text-accent"
				: "bg-surface text-text-secondary hover:text-cream"
		}`}
		aria-label="Toggle metronome"
	>
		<MetronomeIcon
			size={20}
			weight="fill"
			className={metronome.isPlaying ? "animate-pulse" : ""}
		/>
		{metronome.isPlaying && (
			<span className="text-body-sm tabular-nums">{metronome.bpm}</span>
		)}
	</button>

	{showMetronome && (
		<MetronomePanel
			metronome={metronome}
			onClose={() => setShowMetronome(false)}
		/>
	)}
</div>
```

4. Add the MetronomePanel sub-component at the bottom of the file:
```tsx
function MetronomePanel({
	metronome,
	onClose,
}: {
	metronome: import("../hooks/useMetronome").UseMetronomeReturn;
	onClose: () => void;
}) {
	return (
		<>
			<button
				type="button"
				className="fixed inset-0 z-40"
				onClick={onClose}
				aria-label="Close metronome"
			/>
			<div className="absolute top-full left-0 mt-2 z-50 bg-surface border border-border rounded-xl p-4 min-w-[220px] shadow-card animate-overlay-in">
				{/* BPM display + controls */}
				<div className="flex items-center justify-center gap-3 mb-4">
					<button
						type="button"
						onClick={() => metronome.adjustBpm(-1)}
						className="w-8 h-8 rounded-full bg-surface-2 flex items-center justify-center text-text-secondary hover:text-cream transition"
						aria-label="Decrease BPM"
					>
						<Minus size={14} />
					</button>
					<div className="text-center">
						<span className="text-display-sm text-cream tabular-nums">
							{metronome.bpm}
						</span>
						<span className="block text-body-xs text-text-tertiary">BPM</span>
					</div>
					<button
						type="button"
						onClick={() => metronome.adjustBpm(1)}
						className="w-8 h-8 rounded-full bg-surface-2 flex items-center justify-center text-text-secondary hover:text-cream transition"
						aria-label="Increase BPM"
					>
						<Plus size={14} />
					</button>
				</div>

				{/* Tap tempo */}
				<button
					type="button"
					onClick={metronome.tapTempo}
					className="w-full py-2 rounded-lg bg-surface-2 text-body-sm text-text-secondary hover:text-cream transition mb-3"
				>
					Tap Tempo
				</button>

				{/* Time signature */}
				<div className="flex gap-2 mb-3">
					{(["4/4", "3/4", "6/8"] as const).map((ts) => (
						<button
							key={ts}
							type="button"
							onClick={() => metronome.setTimeSignature(ts)}
							className={`flex-1 py-1.5 rounded-lg text-body-sm transition ${
								metronome.timeSignature === ts
									? "bg-accent text-on-accent"
									: "bg-surface-2 text-text-secondary hover:text-cream"
							}`}
						>
							{ts}
						</button>
					))}
				</div>

				{/* Accent + On/Off */}
				<div className="flex items-center justify-between">
					<label className="flex items-center gap-2 text-body-sm text-text-secondary cursor-pointer">
						<input
							type="checkbox"
							checked={metronome.accentFirstBeat}
							onChange={(e) => metronome.setAccentFirstBeat(e.target.checked)}
							className="accent-accent"
						/>
						Accent beat 1
					</label>
					<button
						type="button"
						onClick={metronome.toggle}
						className={`px-3 py-1.5 rounded-lg text-body-sm transition ${
							metronome.isPlaying
								? "bg-red-600 text-on-accent hover:bg-red-500"
								: "bg-accent text-on-accent hover:brightness-110"
						}`}
					>
						{metronome.isPlaying ? "Stop" : "Start"}
					</button>
				</div>
			</div>
		</>
	);
}
```

- [ ] **Step 2: Verify the Metronome icon exists in phosphor-icons**

Run: `cd apps/web && grep -r "Metronome" node_modules/@phosphor-icons/react/dist/index.d.ts | head -5`

If `Metronome` doesn't exist, use `Timer` or `Metronome` from phosphor -- check and adjust. A fallback would be to use a text "M" label.

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit 2>&1 | head -20`
Expected: No type errors

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/components/ListeningMode.tsx
git commit -m "feat: integrate metronome UI panel into ListeningMode"
```

---

## Chunk 3: Piece Context + Error Handling + Cleanup

### File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `apps/web/src/components/ListeningMode.tsx` | Add piece editor, error exit behavior |
| Modify | `apps/web/src/components/AppChat.tsx` | Pass piece context from conversation, handle error exit |

---

### Task 7: Add piece/section editor to ListeningMode

**Files:**
- Modify: `apps/web/src/components/ListeningMode.tsx`

- [ ] **Step 1: Add piece context props and inline editor**

1. Update `ListeningModeProps` to include piece context:
```typescript
interface ListeningModeProps {
	state: PracticeState;
	observations: ObservationEvent[];
	analyserNode: AnalyserNode | null;
	latestScores: DimScores | null;
	error: string | null;
	onStop: () => void;
	originRect: DOMRect | null;
	onExit: () => void;
	pieceContext?: { piece: string; section?: string } | null;
	sessionNotes?: string;
	onNotesChange?: (notes: string) => void;
}
```

2. Inside the component, add state for piece editing:
```typescript
const [pieceName, setPieceName] = useState(pieceContext?.piece ?? "Unknown piece");
const [sectionName, setSectionName] = useState(pieceContext?.section ?? "");
const [isEditingPiece, setIsEditingPiece] = useState(false);
```

3. Also add a `useEffect` to update when pieceContext arrives asynchronously:
```typescript
useEffect(() => {
	if (pieceContext) {
		setPieceName(pieceContext.piece);
		if (pieceContext.section) setSectionName(pieceContext.section);
	}
}, [pieceContext]);
```

4. Replace the static piece info in the top bar with:
```tsx
<div className="text-right">
	{isEditingPiece ? (
		<div className="flex flex-col gap-1 items-end">
			<input
				type="text"
				value={pieceName}
				onChange={(e) => setPieceName(e.target.value)}
				placeholder="Piece name"
				className="bg-surface border border-border rounded-lg px-3 py-1 text-body-sm text-cream outline-none w-56"
				autoFocus
			/>
			<input
				type="text"
				value={sectionName}
				onChange={(e) => setSectionName(e.target.value)}
				placeholder="Section (e.g., bars 1-16)"
				className="bg-surface border border-border rounded-lg px-3 py-1 text-body-xs text-cream outline-none w-56"
			/>
			<button
				type="button"
				onClick={() => setIsEditingPiece(false)}
				className="text-body-xs text-accent hover:text-accent-lighter transition mt-1"
			>
				Done
			</button>
		</div>
	) : (
		<button
			type="button"
			onClick={() => setIsEditingPiece(true)}
			className="text-right group"
		>
			<span className="text-label-sm text-text-tertiary uppercase tracking-wider block">
				Now practicing
			</span>
			<span className="text-body-sm text-cream group-hover:text-accent transition">
				{pieceName}
			</span>
			{sectionName && (
				<span className="text-body-xs text-accent ml-2">{sectionName}</span>
			)}
		</button>
	)}
</div>
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit 2>&1 | head -20`
Expected: No type errors

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/ListeningMode.tsx
git commit -m "feat: add piece/section inline editor to ListeningMode"
```

---

### Task 8: Handle error exit and notes injection in AppChat

**Files:**
- Modify: `apps/web/src/components/AppChat.tsx`

- [ ] **Step 1: Track session notes in AppChat and inject into summary**

1. Add state for session notes after the `recordButtonRect` state:
```typescript
const [sessionNotes, setSessionNotes] = useState("");
```

2. Update the practice summary effect to include notes:
```typescript
useEffect(() => {
	if (practice.summary) {
		let content = practice.summary;
		if (sessionNotes.trim()) {
			content += `\n\n**Your notes:**\n${sessionNotes.trim()}`;
		}
		const summaryMsg: RichMessage = {
			id: `practice-${Date.now()}`,
			role: "assistant",
			content,
			created_at: new Date().toISOString(),
		};
		setMessages((prev) => [...prev, summaryMsg]);
		setSessionNotes("");
	}
}, [practice.summary]);
```

3. Update the `handleExitListeningMode` to also show errors:
```typescript
function handleExitListeningMode() {
	setShowListeningMode(false);
	setRecordButtonRect(null);
	if (practice.error) {
		addToast({ type: "error", message: practice.error });
	}
}
```

4. Pass `sessionNotes` and `onNotesChange` to `ListeningMode`:
```tsx
{showListeningMode && (
	<ListeningMode
		state={practice.state}
		observations={practice.observations}
		analyserNode={practice.analyserNode}
		latestScores={practice.latestScores}
		error={practice.error}
		onStop={practice.stop}
		originRect={recordButtonRect}
		onExit={handleExitListeningMode}
		sessionNotes={sessionNotes}
		onNotesChange={setSessionNotes}
	/>
)}
```

5. In `ListeningMode.tsx`, wire the notes to use the props instead of local state. Change:
```typescript
const [notes, setNotes] = useState("");
```
to:
```typescript
const notes = props.sessionNotes ?? "";
const setNotes = props.onNotesChange ?? (() => {});
```

(Where `props` refers to the destructured props -- adjust variable names accordingly. The notepad drawer already uses `notes` and `onChange` which should now point to these.)

- [ ] **Step 2: Verify TypeScript compiles and app builds**

Run: `cd apps/web && bunx tsc --noEmit 2>&1 | head -20`
Expected: No type errors

Run: `cd apps/web && bun run build 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/AppChat.tsx apps/web/src/components/ListeningMode.tsx
git commit -m "feat: inject session notes into summary, handle error exit with toast"
```

---

### Task 9: Extract piece context from conversation messages

**Files:**
- Modify: `apps/web/src/components/AppChat.tsx`

- [ ] **Step 1: Add piece context extraction on record start**

In `AppChat.tsx`, add piece context state and extraction logic:

1. Add state for piece context after `sessionNotes` state:
```typescript
const [pieceContext, setPieceContext] = useState<{ piece: string; section?: string } | null>(null);
```

2. Create an async function to extract piece context from messages:
```typescript
async function extractPieceContext(msgs: RichMessage[]) {
	if (msgs.length === 0) return;
	try {
		const conversationText = msgs
			.slice(-10) // Last 10 messages for context
			.map((m) => `${m.role}: ${m.content}`)
			.join("\n");

		const res = await fetch(`${import.meta.env.PROD ? "https://api.crescend.ai" : "http://localhost:8787"}/api/extract-goals`, {
			method: "POST",
			credentials: "include",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				message: `Extract the piece name, composer, and section/bars being discussed from this conversation. Return JSON: {"piece": "Composer - Title", "section": "bars X-Y"} or null if no piece is mentioned.\n\n${conversationText}`,
			}),
		});

		if (res.ok) {
			const data = await res.json();
			if (data.piece) {
				setPieceContext(data);
			}
		}
	} catch {
		// Non-critical -- fail silently, user can edit manually
	}
}
```

3. Update `handleRecord` to trigger extraction in parallel:
```typescript
function handleRecord() {
	const rect = recordButtonRef.current?.getBoundingClientRect() ?? null;
	setRecordButtonRect(rect);
	setPieceContext(null); // Reset
	setShowListeningMode(true);
	practice.start();
	// Extract piece context in parallel (non-blocking)
	extractPieceContext(messages);
}
```

4. Pass `pieceContext` to ListeningMode (already in Task 8's updated props).

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit 2>&1 | head -20`
Expected: No type errors

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/AppChat.tsx
git commit -m "feat: extract piece context from conversation for listening mode"
```

---

### Task 10: Add summarizing loading indicator in chat

**Files:**
- Modify: `apps/web/src/components/AppChat.tsx`

- [ ] **Step 1: Show a loading message while summarizing**

After the listening mode exits, if `practice.state === "summarizing"`, show a placeholder message in chat.

1. Add a `useEffect` that appends a loading message when entering summarizing state:
```typescript
useEffect(() => {
	if (practice.state === "summarizing" && !showListeningMode) {
		setMessages((prev) => {
			// Don't add duplicate
			if (prev.some((m) => m.id === "summarizing-placeholder")) return prev;
			return [
				...prev,
				{
					id: "summarizing-placeholder",
					role: "assistant" as const,
					content: "Reviewing your practice session...",
					created_at: new Date().toISOString(),
					streaming: true,
				},
			];
		});
	}
}, [practice.state, showListeningMode]);
```

2. In the existing practice summary effect, remove the placeholder when the real summary arrives:
```typescript
useEffect(() => {
	if (practice.summary) {
		let content = practice.summary;
		if (sessionNotes.trim()) {
			content += `\n\n**Your notes:**\n${sessionNotes.trim()}`;
		}
		const summaryMsg: RichMessage = {
			id: `practice-${Date.now()}`,
			role: "assistant",
			content,
			created_at: new Date().toISOString(),
		};
		setMessages((prev) => [
			...prev.filter((m) => m.id !== "summarizing-placeholder"),
			summaryMsg,
		]);
		setSessionNotes("");
	}
}, [practice.summary]);
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd apps/web && bunx tsc --noEmit 2>&1 | head -20`
Expected: No type errors

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/AppChat.tsx
git commit -m "feat: show loading indicator in chat while practice session summarizes"
```

---

### Task 11: Mobile responsive adjustments

**Files:**
- Modify: `apps/web/src/components/ListeningMode.tsx`

- [ ] **Step 1: Add mobile-specific responsive classes**

The layout already uses Tailwind responsive utilities. Verify and add:

1. Dimension scores: wrap to 2 rows on narrow screens (already using `flex-wrap`, which handles this naturally with `gap-x-6 gap-y-3`).

2. Metronome panel on mobile: render as a bottom sheet. In the `MetronomePanel`, replace the absolute positioning with:
```tsx
<div className="absolute top-full left-0 mt-2 z-50 bg-surface border border-border rounded-xl p-4 min-w-[220px] shadow-card animate-overlay-in md:block hidden">
	{/* ...existing panel content... */}
</div>
{/* Mobile bottom sheet */}
<div className="fixed bottom-0 left-0 right-0 z-50 bg-surface border-t border-border rounded-t-2xl p-4 shadow-card animate-overlay-in md:hidden">
	{/* ...same panel content... */}
</div>
```

3. Notepad drawer: Use `max-h-[60vh]` on mobile. Update the drawer div:
```tsx
<div className="fixed bottom-0 left-0 right-0 z-50 bg-espresso border-t border-border rounded-t-2xl max-h-[60vh] sm:max-h-[40vh] flex flex-col animate-overlay-in">
```

4. Stop button: already `w-14 h-14` (56px) which meets 48px minimum.

5. Add `visualViewport` handling to the `NotepadDrawer` so it stays above the virtual keyboard on mobile:
```typescript
const [bottomOffset, setBottomOffset] = useState(0);

useEffect(() => {
	const vv = window.visualViewport;
	if (!vv) return;

	function handleResize() {
		if (!vv) return;
		const offset = window.innerHeight - vv.height - vv.offsetTop;
		setBottomOffset(Math.max(0, offset));
	}

	vv.addEventListener("resize", handleResize);
	vv.addEventListener("scroll", handleResize);
	return () => {
		vv.removeEventListener("resize", handleResize);
		vv.removeEventListener("scroll", handleResize);
	};
}, []);
```

Then apply `style={{ bottom: bottomOffset }}` to the drawer `<div>`.

- [ ] **Step 2: Verify build succeeds**

Run: `cd apps/web && bun run build 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/ListeningMode.tsx
git commit -m "feat: add mobile responsive adjustments for metronome panel and notepad drawer"
```

---

### Task 12: Final cleanup and verification

**Files:**
- Review: all modified files

- [ ] **Step 1: Run lint**

Run: `cd apps/web && bunx biome check src/ 2>&1 | tail -20`
Fix any issues reported.

- [ ] **Step 2: Run full build**

Run: `cd apps/web && bun run build 2>&1`
Expected: Clean build with no errors or warnings

- [ ] **Step 3: Verify RecordingBar is no longer imported anywhere**

Run: `grep -r "RecordingBar" apps/web/src/ --include="*.tsx" --include="*.ts"`
Expected: Only `RecordingBar.tsx` itself shows up (the file still exists but is unused).

- [ ] **Step 4: Final commit if any lint fixes were needed**

```bash
git add -A apps/web/src/
git commit -m "chore: lint fixes for listening mode implementation"
```
