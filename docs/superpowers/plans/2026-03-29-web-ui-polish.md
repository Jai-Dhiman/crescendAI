# Web UI Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Five targeted UI improvements to the web practice companion: remove observation toasts, remove "Pianist" subtitle, refine listening mode animations, add chat search, and replace ripple visualization with audio-reactive waveform ring.

**Architecture:** All changes are client-side only (React components, CSS, hooks). No backend changes. Each task is independent and produces a working commit. The largest change (Task 5) adds a new canvas component and exposes the AnalyserNode from the audio pipeline via useState.

**Tech Stack:** React 19, Tailwind CSS v4, Web Audio API (AnalyserNode), Canvas 2D, TypeScript

**Spec:** `docs/superpowers/specs/2026-03-29-web-ui-polish-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `apps/web/src/components/AppChat.tsx` | Modify | Remove "Pianist" span, add search UI |
| `apps/web/src/components/ListeningMode.tsx` | Modify | Remove toast rendering, animation refinements, swap waveform component |
| `apps/web/src/components/ObservationToast.tsx` | Delete | No longer used |
| `apps/web/src/components/ResonanceRipples.tsx` | Delete | Replaced by AudioWaveformRing |
| `apps/web/src/components/AudioWaveformRing.tsx` | Create | Canvas-based circular waveform |
| `apps/web/src/styles/app.css` | Modify | Animation timing, edge ring styles |
| `apps/web/src/hooks/usePracticeSession.ts` | Modify | Expose analyserNode as state |

---

### Task 1: Remove "Pianist" Subtitle

**Files:**
- Modify: `apps/web/src/components/AppChat.tsx:718`

- [ ] **Step 1: Remove the "Pianist" span**

In `apps/web/src/components/AppChat.tsx`, delete line 718:

```tsx
// DELETE this line:
<span className="text-body-xs text-text-tertiary">Pianist</span>
```

The `<div className="flex flex-col items-start min-w-0">` block (lines 714-719) should become:

```tsx
<div className="flex flex-col items-start min-w-0">
	<span className="text-body-sm text-cream truncate">
		{user?.displayName ?? user?.email ?? "User"}
	</span>
</div>
```

- [ ] **Step 2: Verify the dev server renders correctly**

Run: `just web` (if not already running)
Check: sidebar profile section shows username only, no subtitle.

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/AppChat.tsx
git commit -m "fix: remove hardcoded 'Pianist' subtitle from sidebar profile"
```

---

### Task 2: Remove Observation Toasts from Listening Mode

**Files:**
- Modify: `apps/web/src/components/ListeningMode.tsx`
- Delete: `apps/web/src/components/ObservationToast.tsx`

- [ ] **Step 1: Remove toast-related imports and state from ListeningMode.tsx**

In `apps/web/src/components/ListeningMode.tsx`:

Remove the import on line 14:
```tsx
// DELETE:
import { ObservationToast } from "./ObservationToast";
```

In the component function, remove the dismissed observations state (lines 49-50):
```tsx
// DELETE these two lines:
const [dismissedObs, setDismissedObs] = useState<Set<string>>(new Set());
const activeObs = observations.filter((o) => !dismissedObs.has(o.id));
```

Rename the `observations` destructured prop to `_observations` to signal intentionally unused:
```tsx
// In the function signature (line 46), change:
observations,
// to:
observations: _observations,
```

- [ ] **Step 2: Remove the observation toast JSX block**

Remove the entire toast rendering block (lines 273-285):
```tsx
// DELETE this entire block:
{/* Observation toasts */}
{activeObs.length > 0 && (
	<div className="absolute top-4 right-4 flex flex-col gap-2 z-10">
		{activeObs.slice(-3).map((obs) => (
			<ObservationToast
				key={obs.id}
				text={obs.text}
				dimension={obs.dimension}
				onDismiss={() => setDismissedObs((prev) => new Set(prev).add(obs.id))}
			/>
		))}
	</div>
)}
```

- [ ] **Step 3: Delete ObservationToast.tsx**

```bash
rm apps/web/src/components/ObservationToast.tsx
```

- [ ] **Step 4: Verify dev server compiles without errors**

Run: `just web`
Check: no TypeScript errors, listening mode opens without toasts.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/components/ListeningMode.tsx
git add apps/web/src/components/ObservationToast.tsx
git commit -m "fix: remove observation toasts from listening mode

Toasts were distracting during play. Observations are accumulated
by the session brain and synthesized on exit."
```

---

### Task 3: Listening Mode Animation Refinement

**Files:**
- Modify: `apps/web/src/styles/app.css:289-302`
- Modify: `apps/web/src/components/ListeningMode.tsx`

- [ ] **Step 1: Update CSS animation timing and fix open state clip-path**

In `apps/web/src/styles/app.css`, replace the listening overlay transition rules (lines 289-302):

```css
/* Radial clip-path transition (GPU-accelerated, no layout thrashing) */
.listening-overlay[data-transition="expanding"] {
	clip-path: circle(150vmax at var(--origin-x, 50%) var(--origin-y, 50%));
	transition: clip-path 750ms cubic-bezier(0.16, 1, 0.3, 1);
}

.listening-overlay[data-transition="collapsed"] {
	clip-path: circle(0% at var(--origin-x, 50%) var(--origin-y, 50%));
	transition: clip-path 500ms cubic-bezier(0.16, 1, 0.3, 1);
}

.listening-overlay[data-transition="open"] {
	clip-path: circle(150vmax at var(--origin-x, 50%) var(--origin-y, 50%));
}
```

- [ ] **Step 2: Add edge ring sibling styles**

In `apps/web/src/styles/app.css`, after the listening overlay rules, add:

```css
/* Edge ring: sibling div outside clip-path overlay */
.listening-edge-ring {
	position: fixed;
	z-index: 49;
	pointer-events: none;
	border-radius: 50%;
	border: 1.5px solid rgba(122, 154, 130, 0.6);
	box-shadow: 0 0 12px rgba(122, 154, 130, 0.3);
	transform: translate(-50%, -50%);
	opacity: 0;
}

.listening-edge-ring[data-transition="expanding"] {
	width: 300vmax;
	height: 300vmax;
	opacity: 0;
	transition:
		width 750ms cubic-bezier(0.16, 1, 0.3, 1),
		height 750ms cubic-bezier(0.16, 1, 0.3, 1),
		opacity 400ms ease-out;
}

.listening-edge-ring[data-transition="collapsed"] {
	width: 0;
	height: 0;
	opacity: 0;
	transition:
		width 500ms cubic-bezier(0.16, 1, 0.3, 1),
		height 500ms cubic-bezier(0.16, 1, 0.3, 1),
		opacity 300ms ease-in;
}

.listening-edge-ring[data-transition="active"] {
	opacity: 1;
	transition: opacity 200ms ease-out;
}
```

- [ ] **Step 3: Add content fade-out CSS**

In `apps/web/src/styles/app.css`, after the listening-content-in animation:

```css
.listening-content-fading {
	opacity: 0;
	transition: opacity 150ms ease-in;
}
```

- [ ] **Step 4: Update ListeningMode.tsx state machine and add edge ring**

In `apps/web/src/components/ListeningMode.tsx`:

Add a content visibility type and refs for setTimeout cleanup. Replace the relevant state and handlers:

```tsx
type ContentVisibility = "hidden" | "visible" | "fading";
```

Replace line 51:
```tsx
// REPLACE:
const [contentVisible, setContentVisible] = useState(false);
// WITH:
const [contentVis, setContentVis] = useState<ContentVisibility>("hidden");
```

Add timer refs after the existing refs:
```tsx
const expandTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
const collapseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
const fadeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
const exitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
```

Add an edge ring phase state:
```tsx
const [ringPhase, setRingPhase] = useState<"collapsed" | "active" | "expanding">("collapsed");
```

Replace the open transition sequence (lines 80-93):
```tsx
useMountEffect(() => {
	requestAnimationFrame(() => {
		setPhase("expanding");
		setRingPhase("active");
	});

	// Ring fades out as expansion completes
	const ringTimer = setTimeout(() => setRingPhase("expanding"), 400);

	expandTimerRef.current = setTimeout(() => {
		setPhase("open");
		setContentVis("visible");
		setRingPhase("collapsed");
	}, 800); // slightly after 750ms transition

	return () => {
		clearTimeout(ringTimer);
		if (expandTimerRef.current) clearTimeout(expandTimerRef.current);
	};
});
```

Replace handleClose (lines 96-107):
```tsx
const handleClose = useCallback(() => {
	setContentVis("fading");
	setRingPhase("active");

	fadeTimerRef.current = setTimeout(() => {
		setContentVis("hidden");
		setPhase("collapsing");

		collapseTimerRef.current = setTimeout(() => {
			setPhase("collapsed");
			setRingPhase("collapsed");
			exitTimerRef.current = setTimeout(() => {
				onStop();
				onExit();
			}, 50);
		}, 550); // slightly after 500ms transition
	}, 100); // 100ms fade overlap with shrink start
}, [onStop, onExit]);
```

Replace handleStop (lines 117-126):
```tsx
const handleStop = useCallback(() => {
	onStop();
	setContentVis("fading");
	setRingPhase("active");

	fadeTimerRef.current = setTimeout(() => {
		setContentVis("hidden");
		setPhase("collapsing");

		collapseTimerRef.current = setTimeout(() => {
			setPhase("collapsed");
			setRingPhase("collapsed");
			exitTimerRef.current = setTimeout(onExit, 50);
		}, 550);
	}, 100);
}, [onStop, onExit]);
```

Add cleanup effect:
```tsx
useEffect(() => {
	return () => {
		if (expandTimerRef.current) clearTimeout(expandTimerRef.current);
		if (collapseTimerRef.current) clearTimeout(collapseTimerRef.current);
		if (fadeTimerRef.current) clearTimeout(fadeTimerRef.current);
		if (exitTimerRef.current) clearTimeout(exitTimerRef.current);
	};
}, []);
```

Update the content rendering (line 175):
```tsx
// REPLACE:
{contentVisible && (
	<div className="h-dvh flex flex-col animate-listening-content-in">
// WITH:
{contentVis !== "hidden" && (
	<div className={`h-dvh flex flex-col ${
		contentVis === "fading" ? "listening-content-fading" : "animate-listening-content-in"
	}`}>
```

Add the edge ring sibling in the portal return, after the overlay div but still inside `createPortal`:
```tsx
return createPortal(
	<>
		<div
			ref={overlayRef}
			className="listening-overlay"
			data-transition={transitionAttr}
			style={{
				"--origin-x": `${originX}%`,
				"--origin-y": `${originY}%`,
			} as React.CSSProperties}
		>
			{/* ... existing content ... */}
		</div>
		<div
			className="listening-edge-ring"
			data-transition={ringPhase}
			style={{
				left: `${originX}%`,
				top: `${originY}%`,
			}}
		/>
	</>,
	document.body,
);
```

Note: `createPortal` now wraps a Fragment (`<>...</>`) instead of a single div.

- [ ] **Step 5: Verify animations in dev server**

Run: `just web`
Check:
- Opening listening mode: circle expands from record button with sage-green edge glow that fades as it fills the screen
- Content fades in after expansion
- Closing: content fades, circle shrinks with edge glow, cleanup is clean
- No console warnings about setState on unmounted component

- [ ] **Step 6: Commit**

```bash
git add apps/web/src/styles/app.css apps/web/src/components/ListeningMode.tsx
git commit -m "feat: refine listening mode open/close animation

- Slower open (750ms), snappier close (500ms) with spring easing
- Edge ring sibling div with sage-green glow (outside clip-path)
- Content fade-out overlaps with clip shrink for fluid motion
- Fix clip-path: none jump on close (explicit 150vmax on open state)
- Store all setTimeout IDs in refs for cleanup on unmount"
```

---

### Task 4: Chat Search

**Files:**
- Modify: `apps/web/src/components/AppChat.tsx`

- [ ] **Step 1: Add search state**

In `apps/web/src/components/AppChat.tsx`, after the existing state declarations (around line 119), add:

```tsx
const [searchOpen, setSearchOpen] = useState(false);
const [searchQuery, setSearchQuery] = useState("");
const searchInputRef = useRef<HTMLInputElement>(null);
```

- [ ] **Step 2: Add Cmd+K keyboard listener**

After the search state, add:

```tsx
useEffect(() => {
	const handler = (e: KeyboardEvent) => {
		if ((e.metaKey || e.ctrlKey) && e.key === "k") {
			e.preventDefault();
			if (!sidebarOpen) setSidebarOpen(true);
			setSearchOpen(true);
			requestAnimationFrame(() => {
				searchInputRef.current?.focus();
			});
		}
	};
	document.addEventListener("keydown", handler);
	return () => document.removeEventListener("keydown", handler);
}, [sidebarOpen, setSidebarOpen]);
```

- [ ] **Step 3: Add filtered conversations memo**

After the keyboard listener, add:

```tsx
const filteredConversations = useMemo(() => {
	if (!searchOpen || !searchQuery.trim()) return null;
	const q = searchQuery.toLowerCase();
	return conversations.filter((c) =>
		(c.title ?? "New conversation").toLowerCase().includes(q),
	);
}, [searchOpen, searchQuery, conversations]);
```

- [ ] **Step 4: Replace search button with search input when active**

Replace the search SidebarButton block (lines 623-630):

```tsx
<div className="w-full">
	{searchOpen && sidebarOpen ? (
		<div className="flex items-center gap-1 px-2 py-1">
			<MagnifyingGlass size={16} className="shrink-0 text-text-tertiary" />
			<input
				ref={searchInputRef}
				type="text"
				value={searchQuery}
				onChange={(e) => setSearchQuery(e.target.value)}
				onKeyDown={(e) => {
					if (e.key === "Escape") {
						setSearchOpen(false);
						setSearchQuery("");
					}
				}}
				placeholder="Search conversations..."
				className="flex-1 bg-transparent text-body-sm text-cream placeholder:text-text-tertiary outline-none min-w-0"
				// biome-ignore lint/a11y/noAutofocus: intentional UX for search activation
				autoFocus
			/>
			<button
				type="button"
				onClick={() => {
					setSearchOpen(false);
					setSearchQuery("");
				}}
				className="shrink-0 w-6 h-6 flex items-center justify-center text-text-tertiary hover:text-cream transition"
				aria-label="Close search"
			>
				<X size={14} />
			</button>
		</div>
	) : (
		<SidebarButton
			icon={<MagnifyingGlass size={20} />}
			label="Search"
			expanded={sidebarOpen}
			onClick={() => {
				if (!sidebarOpen) setSidebarOpen(true);
				setSearchOpen(true);
				requestAnimationFrame(() => {
					searchInputRef.current?.focus();
				});
			}}
		/>
	)}
</div>
```

- [ ] **Step 5: Update conversation list to use filtered results**

Replace the conversation list rendering (lines 642-677) with:

```tsx
{isConversationsPending ? (
	<ConversationSkeleton />
) : (
	<>
		{(filteredConversations ?? conversations.slice(0, 8)).map((conv) => (
			<div
				role="button"
				tabIndex={0}
				key={conv.id}
				className={`group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 cursor-pointer text-body-sm transition min-h-[36px] text-left ${
					conv.id === activeConversationId
						? "bg-surface text-cream"
						: "text-text-secondary hover:text-cream hover:bg-surface"
				}`}
				onClick={() => {
					loadConversation(conv.id);
					setSearchOpen(false);
					setSearchQuery("");
				}}
				onKeyDown={(e) => {
					if (e.key === "Enter" || e.key === " ") {
						e.preventDefault();
						loadConversation(conv.id);
						setSearchOpen(false);
						setSearchQuery("");
					}
				}}
			>
				<ChatCircle size={14} className="shrink-0" />
				<span className="flex-1 truncate">
					{conv.title ?? "New conversation"}
				</span>
				{!searchOpen && (
					<button
						type="button"
						onClick={(e) => {
							e.stopPropagation();
							handleDeleteConversation(conv.id);
						}}
						className="opacity-0 group-hover:opacity-100 shrink-0 w-7 h-7 flex items-center justify-center text-text-tertiary hover:text-cream transition"
						aria-label="Delete conversation"
					>
						<Trash size={14} />
					</button>
				)}
			</div>
		))}
		{filteredConversations !== null && filteredConversations.length === 0 && (
			<div className="px-3 py-6 text-center">
				<span className="text-body-xs text-text-tertiary">
					No conversations matching &lsquo;{searchQuery}&rsquo;
				</span>
			</div>
		)}
		{!searchOpen && conversations.length > 8 && (
			<button
				type="button"
				className="w-full mt-1 px-3 py-2 text-body-xs text-text-tertiary hover:text-cream transition text-left"
				onClick={() => navigate({ to: "/app/chats" })}
			>
				See All Chats
			</button>
		)}
	</>
)}
```

- [ ] **Step 6: Verify search works**

Run: `just web`
Check:
- Click search icon: input appears inline, focused
- Type a query: conversation list filters (against ALL conversations, not just 8)
- Zero results: shows "No conversations matching 'X'"
- Escape: closes search, restores normal list
- Cmd+K: opens search (test in Safari/Firefox; Chrome may intercept)
- Click a result: navigates to conversation, search closes

- [ ] **Step 7: Commit**

```bash
git add apps/web/src/components/AppChat.tsx
git commit -m "feat: add sidebar chat search with Cmd+K shortcut

Client-side conversation title filtering. Filters full conversation
list (not just visible 8). Empty state for zero results. Escape to
dismiss. Auto-expands collapsed sidebar on Cmd+K."
```

---

### Task 5: Audio-Reactive Waveform Ring

**Files:**
- Modify: `apps/web/src/hooks/usePracticeSession.ts`
- Create: `apps/web/src/components/AudioWaveformRing.tsx`
- Modify: `apps/web/src/components/ListeningMode.tsx`
- Delete: `apps/web/src/components/ResonanceRipples.tsx`

#### Sub-task 5a: Expose AnalyserNode from usePracticeSession

- [ ] **Step 1: Add analyserNode state to the hook**

In `apps/web/src/hooks/usePracticeSession.ts`:

Add to `UsePracticeSessionReturn` interface (after line 52 `energy: number;`):
```tsx
analyserNode: AnalyserNode | null;
```

Add state after `analyserRef` (line 98):
```tsx
const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
```

After the analyser is created (line 367 `analyserRef.current = analyser;`), add:
```tsx
setAnalyserNode(analyser);
```

In the cleanup where `analyserRef.current = null` is set (there are two locations -- search for `analyserRef.current = null`), add alongside each:
```tsx
setAnalyserNode(null);
```

Add `analyserNode` to the return object (after `energy,` on line 642):
```tsx
analyserNode,
```

- [ ] **Step 2: Verify hook compiles**

Run: `just web`
Check: no TypeScript errors.

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/hooks/usePracticeSession.ts
git commit -m "feat: expose analyserNode as state from usePracticeSession"
```

#### Sub-task 5b: Create AudioWaveformRing component

- [ ] **Step 4: Create the component file**

Create `apps/web/src/components/AudioWaveformRing.tsx`:

```tsx
import { useEffect, useRef } from "react";

interface AudioWaveformRingProps {
	analyserNode: AnalyserNode | null;
	isPlaying: boolean;
	active: boolean;
}

// Sage green
const SAGE_R = 122;
const SAGE_G = 154;
const SAGE_B = 130;

// Number of points around the circle
const NUM_POINTS = 128;
// Breathing animation period (ms)
const BREATH_PERIOD = 4000;
// Lerp coefficient (frame-rate-independent base)
const LERP_BASE = 0.15;
// Rotation speed (deg/frame at 60fps)
const ROTATION_SPEED = 0.5;
// Idle frame throttle (ms) for ~15fps
const IDLE_FRAME_MIN_MS = 66;
// Crossfade duration from idle to active (ms)
const CROSSFADE_MS = 400;
// Max displacement as fraction of ring radius
const MAX_DISPLACEMENT = 0.35;
// Base ring stroke width
const STROKE_WIDTH = 1.5;

/**
 * Build a log-scale mapping table: for each of NUM_POINTS positions around
 * the circle, which frequency bin index to read. Concentrates resolution in
 * low frequencies where piano fundamentals live.
 */
function buildLogBinMap(binCount: number): number[] {
	const map: number[] = [];
	for (let i = 0; i < NUM_POINTS; i++) {
		const t = i / NUM_POINTS;
		// Quadratic mapping concentrates low frequencies
		const binIndex = Math.floor(Math.pow(t, 2) * (binCount - 1));
		map.push(Math.min(binIndex, binCount - 1));
	}
	return map;
}

export function AudioWaveformRing({
	analyserNode,
	isPlaying,
	active,
}: AudioWaveformRingProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const rafRef = useRef<number>(0);
	const lastFrameRef = useRef(0);
	const rotationRef = useRef(0);
	const displacementsRef = useRef<Float32Array | null>(null);
	const dataArrayRef = useRef<Uint8Array | null>(null);
	const logBinMapRef = useRef<number[] | null>(null);
	const crossfadeRef = useRef(0); // 0 = full breathing, 1 = full frequency
	const isPlayingRef = useRef(isPlaying);
	const sizeRef = useRef({ w: 0, h: 0 });

	// Keep isPlaying ref in sync
	useEffect(() => {
		isPlayingRef.current = isPlaying;
	}, [isPlaying]);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;

		// Initialize displacements
		if (!displacementsRef.current) {
			displacementsRef.current = new Float32Array(NUM_POINTS);
		}

		// Set up analyser data structures
		if (analyserNode && !logBinMapRef.current) {
			const binCount = analyserNode.frequencyBinCount;
			logBinMapRef.current = buildLogBinMap(binCount);
			dataArrayRef.current = new Uint8Array(binCount);
		}

		// ResizeObserver for canvas sizing
		const observer = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const { width, height } = entry.contentRect;
				const dpr = window.devicePixelRatio || 1;
				canvas.width = width * dpr;
				canvas.height = height * dpr;
				sizeRef.current = { w: width, h: height };
			}
		});
		observer.observe(canvas);

		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		function draw(timestamp: number) {
			const dt = lastFrameRef.current ? timestamp - lastFrameRef.current : 16.67;
			lastFrameRef.current = timestamp;

			// Throttle in idle mode
			if (!active && dt < IDLE_FRAME_MIN_MS) {
				rafRef.current = requestAnimationFrame(draw);
				return;
			}

			// Self-throttle on slow devices
			if (dt > 33 && active) {
				// Skip this frame to catch up
				rafRef.current = requestAnimationFrame(draw);
				lastFrameRef.current = timestamp;
				return;
			}

			const { w, h } = sizeRef.current;
			if (w === 0 || h === 0) {
				rafRef.current = requestAnimationFrame(draw);
				return;
			}

			const dpr = window.devicePixelRatio || 1;
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
			ctx.clearRect(0, 0, w, h);

			const cx = w / 2;
			const cy = h / 2;
			const radius = Math.min(w, h) * 0.38;
			const displacements = displacementsRef.current!;

			// Update crossfade (0 = breathing, 1 = frequency)
			const crossfadeTarget = isPlayingRef.current && analyserNode ? 1 : 0;
			const crossfadeAlpha = 1 - Math.pow(1 - 0.005, dt);
			crossfadeRef.current += (crossfadeTarget - crossfadeRef.current) * crossfadeAlpha;

			// Frame-rate-independent lerp alpha
			const lerpAlpha = 1 - Math.pow(1 - LERP_BASE, dt / 16.67);

			// Calculate target displacements
			let totalEnergy = 0;

			// Read frequency data once per frame
			if (analyserNode && dataArrayRef.current) {
				analyserNode.getByteFrequencyData(dataArrayRef.current);
			}

			for (let i = 0; i < NUM_POINTS; i++) {
				let freqTarget = 0;

				// Frequency-reactive displacement
				if (dataArrayRef.current && logBinMapRef.current) {
					const binIdx = logBinMapRef.current[i];
					const value = dataArrayRef.current[binIdx] / 255;
					freqTarget = value * MAX_DISPLACEMENT * radius;
					totalEnergy += value;
				}

				// Breathing displacement (sinusoidal)
				const breathPhase = (timestamp % BREATH_PERIOD) / BREATH_PERIOD;
				const breathTarget = Math.sin(breathPhase * Math.PI * 2) * radius * 0.03;

				// Blend breathing and frequency based on crossfade
				const cf = crossfadeRef.current;
				const target = freqTarget * cf + breathTarget * (1 - cf);

				// Lerp toward target
				displacements[i] += (target - displacements[i]) * lerpAlpha;
			}

			// Compute opacity from energy (0.6 to 1.0)
			const avgEnergy = analyserNode ? totalEnergy / NUM_POINTS : 0;
			const opacity = 0.6 + avgEnergy * 0.4;

			// Update rotation (frame-rate-normalized)
			rotationRef.current += ROTATION_SPEED * (dt / 16.67);
			const rotRad = (rotationRef.current * Math.PI) / 180;

			// Draw the ring
			ctx.beginPath();
			for (let i = 0; i <= NUM_POINTS; i++) {
				const idx = i % NUM_POINTS;
				const angle = (idx / NUM_POINTS) * Math.PI * 2 - Math.PI / 2 + rotRad;
				const r = radius + displacements[idx];
				const x = cx + Math.cos(angle) * r;
				const y = cy + Math.sin(angle) * r;

				if (i === 0) {
					ctx.moveTo(x, y);
				} else {
					// Smooth curve through points using quadratic bezier
					const prevIdx = (i - 1) % NUM_POINTS;
					const prevAngle = (prevIdx / NUM_POINTS) * Math.PI * 2 - Math.PI / 2 + rotRad;
					const prevR = radius + displacements[prevIdx];
					const prevX = cx + Math.cos(prevAngle) * prevR;
					const prevY = cy + Math.sin(prevAngle) * prevR;
					const cpX = (prevX + x) / 2;
					const cpY = (prevY + y) / 2;
					ctx.quadraticCurveTo(prevX, prevY, cpX, cpY);
				}
			}
			ctx.closePath();

			ctx.strokeStyle = `rgba(${SAGE_R}, ${SAGE_G}, ${SAGE_B}, ${opacity})`;
			ctx.lineWidth = STROKE_WIDTH;
			ctx.stroke();

			rafRef.current = requestAnimationFrame(draw);
		}

		rafRef.current = requestAnimationFrame(draw);

		return () => {
			cancelAnimationFrame(rafRef.current);
			observer.disconnect();
		};
	}, [analyserNode, active]);

	return (
		<canvas
			ref={canvasRef}
			className="w-full h-full"
			aria-hidden="true"
		/>
	);
}
```

- [ ] **Step 5: Verify the component compiles**

Run: `just web`
Check: no TypeScript errors (component not yet wired in).

- [ ] **Step 6: Commit**

```bash
git add apps/web/src/components/AudioWaveformRing.tsx
git commit -m "feat: add AudioWaveformRing canvas component

Circular waveform with log-scale frequency bin mapping for piano,
frame-rate-independent lerp, idle breathing animation, crossfade
transition from idle to active, and self-throttle for low-end devices."
```

#### Sub-task 5c: Wire into ListeningMode and remove ResonanceRipples

- [ ] **Step 7: Update ListeningMode to use AudioWaveformRing**

In `apps/web/src/components/ListeningMode.tsx`:

Replace the import (line 15):
```tsx
// REPLACE:
import { ResonanceRipples } from "./ResonanceRipples";
// WITH:
import { AudioWaveformRing } from "./AudioWaveformRing";
```

Add `analyserNode` to the props interface:
```tsx
interface ListeningModeProps {
	state: PracticeState;
	energy: number;
	isPlaying: boolean;
	error: string | null;
	wsStatus: WsStatus;
	onStop: () => void;
	originRect: DOMRect | null;
	onExit: () => void;
	pieceContext?: { piece: string; section?: string } | null;
	sessionNotes?: string;
	onNotesChange?: (notes: string) => void;
	observations: Array<{ text: string; dimension: string; id: string }>;
	analyserNode: AnalyserNode | null;
}
```

Add `analyserNode` to the destructured props in the function signature.

Replace the waveform container and component (the div at line 269-271):
```tsx
{/* Waveform */}
<div className="w-full max-w-md aspect-square">
	<AudioWaveformRing
		analyserNode={analyserNode}
		isPlaying={isPlaying}
		active={isRecording}
	/>
</div>
```

Note: changed from `max-w-3xl h-32 sm:h-40 md:h-48` (wide rectangle) to `max-w-md aspect-square` (square container for the circular waveform).

- [ ] **Step 8: Update the parent that renders ListeningMode**

Find where `<ListeningMode>` is rendered (in `AppChat.tsx` or the practice session wiring). Pass the new `analyserNode` prop:

```tsx
<ListeningMode
	{/* ...existing props... */}
	analyserNode={practice.analyserNode}
/>
```

- [ ] **Step 9: Delete ResonanceRipples.tsx**

```bash
rm apps/web/src/components/ResonanceRipples.tsx
```

- [ ] **Step 10: Verify the waveform renders**

Run: `just web`
Check:
- Open listening mode: see a breathing sage-green circle (idle state)
- Start playing piano (or any audio): ring deforms reactively to the sound
- Stop playing: ring settles back to a circle smoothly (no snap)
- Low frequencies produce visible deformation (log-scale mapping working)
- No dropped frames on desktop

- [ ] **Step 11: Commit**

```bash
git add apps/web/src/components/ListeningMode.tsx apps/web/src/components/ResonanceRipples.tsx apps/web/src/components/AudioWaveformRing.tsx
git commit -m "feat: replace ripple waves with audio-reactive waveform ring

Swap ResonanceRipples for AudioWaveformRing in ListeningMode.
Square container for circular visualization. Pass analyserNode from
usePracticeSession. Delete ResonanceRipples.tsx."
```

---

### Task 6: Final Verification

- [ ] **Step 1: Run the full app and verify all 5 changes together**

Run: `just web`

Checklist:
- [ ] Sidebar: no "Pianist" subtitle under username
- [ ] Sidebar: search icon opens inline search input
- [ ] Sidebar: Cmd+K focuses search (test Safari/Firefox)
- [ ] Sidebar: search filters conversations, shows empty state for no results
- [ ] Listening mode: opens with smooth animation + edge glow ring
- [ ] Listening mode: shows breathing waveform circle at idle
- [ ] Listening mode: waveform responds to audio when playing
- [ ] Listening mode: no observation toasts during play
- [ ] Listening mode: closes with fade + shrink, edge glow on close
- [ ] No console errors or warnings

- [ ] **Step 2: Run type check**

```bash
cd apps/web && bunx tsc --noEmit
```

Expected: no errors.
