# Artifact Container System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified `<Artifact>` container with three visual states (collapsed, inline, expanded) that wraps rich content in chat messages, shipping with exercise_set as the only artifact type for beta.

**Architecture:** A Zustand store (`useArtifactStore`) tracks display state and exercise completion per artifact. The `<Artifact>` component reads from the store and renders one of three views: `CollapsedPreview`, `InlineCard` (existing), or `ArtifactOverlay` (React portal). IntersectionObserver auto-collapses artifacts when they scroll out of the chat viewport. The overlay is a fixed backdrop+panel that renders once at the app level.

**Tech Stack:** React 19, Zustand 5, Vitest + @testing-library/react, Tailwind CSS v4 (design tokens in `app.css`)

**Spec:** `docs/superpowers/specs/2026-03-19-artifact-container-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/stores/artifact.ts` | Create | Zustand store: display state per artifact, exercise completion state, actions |
| `src/stores/artifact.test.ts` | Create | Store unit tests: state transitions, guards, exercise state sync |
| `src/contexts/artifact-scroll.ts` | Create | ArtifactScrollContext + useArtifactScrollContext hook (shared by Artifact + ArtifactOverlay) |
| `src/lib/exercise-utils.ts` | Create | Shared `handsLabel()` utility used by ExerciseSetCard + ExerciseSetExpanded |
| `src/components/Artifact.tsx` | Create | State machine wrapper, IntersectionObserver, renders collapsed/inline/expanded |
| `src/components/ArtifactOverlay.tsx` | Create | React portal overlay with backdrop, Escape handler, scroll lock |
| `src/components/cards/CollapsedPreview.tsx` | Create | Mini card (~56px): title, subtitle, badge, expand chevron |
| `src/components/cards/ExerciseSetExpanded.tsx` | Create | Workspace view: all exercises expanded, assign+complete flow, progress |
| `src/components/cards/ExerciseSetCard.tsx` | Modify | Add `onExpand` prop, expand icon in header, read exercise state from store (with local fallback) |
| `src/components/InlineCard.tsx` | Modify | Accept and forward `onExpand` prop |
| `src/components/ChatMessages.tsx` | Modify | Swap `InlineCard` for `Artifact`, wrap in ArtifactScrollContext provider |
| `src/components/AppChat.tsx` | Modify | Add `<ArtifactOverlay />` as sibling to ChatMessages |
| `src/styles/app.css` | Modify | Add artifact-collapse, backdrop, and panel-out animation keyframes |

All paths are relative to `apps/web/`.

---

### Task 1: Zustand Store

**Files:**
- Create: `src/stores/artifact.ts`
- Create: `src/stores/artifact.test.ts`

- [ ] **Step 1: Write the store tests**

```typescript
// src/stores/artifact.test.ts
import { afterEach, describe, expect, it } from "vitest";
import { useArtifactStore, getExpandedArtifact } from "./artifact";
import type { InlineComponent } from "../lib/types";

const mockComponent: InlineComponent = {
  type: "exercise_set",
  config: {
    source_passage: "bars 20-24",
    target_skill: "dynamic control",
    exercises: [
      {
        title: "Three-level dynamics",
        instruction: "Play at pp, mf, ff.",
        focus_dimension: "dynamics",
        hands: "both",
        exercise_id: "ex-001",
      },
    ],
  },
};

function getState() {
  return useArtifactStore.getState();
}

describe("useArtifactStore", () => {
  afterEach(() => {
    // Reset store between tests
    const state = getState();
    for (const id of Object.keys(state.states)) {
      state.unregister(id);
    }
  });

  it("register sets initial state to inline", () => {
    getState().register("a1", mockComponent);
    expect(getState().states.a1.state).toBe("inline");
    expect(getState().states.a1.component).toBe(mockComponent);
  });

  it("collapse transitions from inline to collapsed", () => {
    getState().register("a1", mockComponent);
    getState().collapse("a1");
    expect(getState().states.a1.state).toBe("collapsed");
  });

  it("collapse is no-op from expanded", () => {
    getState().register("a1", mockComponent);
    getState().expand("a1");
    getState().collapse("a1");
    expect(getState().states.a1.state).toBe("expanded");
  });

  it("collapse is no-op if already collapsed", () => {
    getState().register("a1", mockComponent);
    getState().collapse("a1");
    getState().collapse("a1");
    expect(getState().states.a1.state).toBe("collapsed");
  });

  it("expand transitions from any state", () => {
    getState().register("a1", mockComponent);
    getState().collapse("a1");
    getState().expand("a1");
    expect(getState().states.a1.state).toBe("expanded");
  });

  it("expand sets previous expanded artifact back to inline", () => {
    getState().register("a1", mockComponent);
    getState().register("a2", mockComponent);
    getState().expand("a1");
    getState().expand("a2");
    expect(getState().states.a1.state).toBe("inline");
    expect(getState().states.a2.state).toBe("expanded");
  });

  it("restore transitions from collapsed to inline", () => {
    getState().register("a1", mockComponent);
    getState().collapse("a1");
    getState().restore("a1");
    expect(getState().states.a1.state).toBe("inline");
  });

  it("restore is no-op from inline", () => {
    getState().register("a1", mockComponent);
    getState().restore("a1");
    expect(getState().states.a1.state).toBe("inline");
  });

  it("closeOverlay transitions from expanded to inline", () => {
    getState().register("a1", mockComponent);
    getState().expand("a1");
    getState().closeOverlay("a1");
    expect(getState().states.a1.state).toBe("inline");
  });

  it("closeOverlay is no-op if not expanded", () => {
    getState().register("a1", mockComponent);
    getState().closeOverlay("a1");
    expect(getState().states.a1.state).toBe("inline");
  });

  it("unregister removes entry", () => {
    getState().register("a1", mockComponent);
    getState().unregister("a1");
    expect(getState().states.a1).toBeUndefined();
  });

  it("getExpandedArtifact returns null when nothing expanded", () => {
    getState().register("a1", mockComponent);
    const result = getExpandedArtifact(getState());
    expect(result).toBeNull();
  });

  it("getExpandedArtifact returns expanded artifact", () => {
    getState().register("a1", mockComponent);
    getState().expand("a1");
    const result = getExpandedArtifact(getState());
    expect(result).not.toBeNull();
    expect(result?.id).toBe("a1");
    expect(result?.entry.component).toBe(mockComponent);
  });

  it("setExerciseStatus tracks exercise state", () => {
    getState().register("a1", mockComponent);
    getState().setExerciseStatus("a1", "ex-001", "assigned", "se-123");
    const entry = getState().states.a1;
    expect(entry.exerciseStates?.["ex-001"]).toEqual({
      status: "assigned",
      studentExerciseId: "se-123",
    });
  });

  it("setExerciseStatus updates existing exercise state", () => {
    getState().register("a1", mockComponent);
    getState().setExerciseStatus("a1", "ex-001", "assigned", "se-123");
    getState().setExerciseStatus("a1", "ex-001", "completed", "se-123");
    expect(getState().states.a1.exerciseStates?.["ex-001"]?.status).toBe(
      "completed",
    );
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/web && bun run test -- src/stores/artifact.test.ts`
Expected: FAIL -- module `./artifact` not found

- [ ] **Step 3: Implement the store**

```typescript
// src/stores/artifact.ts
import { create } from "zustand";
import type { InlineComponent } from "../lib/types";

export type ExerciseStatus =
  | "idle"
  | "loading"
  | "assigned"
  | "completing"
  | "completed"
  | "error";

export interface ExerciseState {
  status: ExerciseStatus;
  studentExerciseId?: string;
}

export interface ArtifactEntry {
  state: "collapsed" | "inline" | "expanded";
  component: InlineComponent;
  exerciseStates?: Record<string, ExerciseState>;
}

interface ArtifactStore {
  states: Record<string, ArtifactEntry>;

  register(id: string, component: InlineComponent): void;
  collapse(id: string): void;
  expand(id: string): void;
  restore(id: string): void;
  closeOverlay(id: string): void;
  unregister(id: string): void;

  setExerciseStatus(
    artifactId: string,
    exerciseId: string,
    status: ExerciseStatus,
    studentExerciseId?: string,
  ): void;
}

/** Atomic selector: returns the expanded artifact's id + entry, or null. */
export function getExpandedArtifact(
  state: Pick<ArtifactStore, "states">,
): { id: string; entry: ArtifactEntry } | null {
  const entries = Object.entries(state.states);
  const expanded = entries.find(([_, e]) => e.state === "expanded");
  return expanded ? { id: expanded[0], entry: expanded[1] } : null;
}

export const useArtifactStore = create<ArtifactStore>((set) => ({
  states: {},

  register(id, component) {
    set((s) => ({
      states: {
        ...s.states,
        [id]: { state: "inline", component },
      },
    }));
  },

  collapse(id) {
    set((s) => {
      const entry = s.states[id];
      if (!entry || entry.state !== "inline") return s;
      return {
        states: {
          ...s.states,
          [id]: { ...entry, state: "collapsed" },
        },
      };
    });
  },

  expand(id) {
    set((s) => {
      const entry = s.states[id];
      if (!entry) return s;
      const updated = { ...s.states };
      // Close any currently expanded artifact
      for (const [key, val] of Object.entries(updated)) {
        if (val.state === "expanded") {
          updated[key] = { ...val, state: "inline" };
        }
      }
      updated[id] = { ...entry, state: "expanded" };
      return { states: updated };
    });
  },

  restore(id) {
    set((s) => {
      const entry = s.states[id];
      if (!entry || entry.state !== "collapsed") return s;
      return {
        states: {
          ...s.states,
          [id]: { ...entry, state: "inline" },
        },
      };
    });
  },

  closeOverlay(id) {
    set((s) => {
      const entry = s.states[id];
      if (!entry || entry.state !== "expanded") return s;
      return {
        states: {
          ...s.states,
          [id]: { ...entry, state: "inline" },
        },
      };
    });
  },

  unregister(id) {
    set((s) => {
      const { [id]: _, ...rest } = s.states;
      return { states: rest };
    });
  },

  setExerciseStatus(artifactId, exerciseId, status, studentExerciseId) {
    set((s) => {
      const entry = s.states[artifactId];
      if (!entry) return s;
      return {
        states: {
          ...s.states,
          [artifactId]: {
            ...entry,
            exerciseStates: {
              ...entry.exerciseStates,
              [exerciseId]: { status, studentExerciseId },
            },
          },
        },
      };
    });
  },
}));
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/web && bun run test -- src/stores/artifact.test.ts`
Expected: All 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/stores/artifact.ts apps/web/src/stores/artifact.test.ts
git commit -m "feat(web): add artifact Zustand store with display and exercise state"
```

---

### Task 1b: Shared Utilities (Scroll Context + handsLabel)

**Files:**
- Create: `src/contexts/artifact-scroll.ts`
- Create: `src/lib/exercise-utils.ts`

- [ ] **Step 1: Create the ArtifactScrollContext**

This is extracted to its own file so that both `Artifact.tsx` and `ArtifactOverlay.tsx` can import it without depending on `ChatMessages.tsx` (which would create a circular dependency and break the build chain for parallel tasks).

```typescript
// src/contexts/artifact-scroll.ts
import { createContext, useContext } from "react";

export const ArtifactScrollContext = createContext<React.RefObject<HTMLDivElement | null> | null>(null);

export function useArtifactScrollContext() {
	return useContext(ArtifactScrollContext);
}
```

- [ ] **Step 2: Create the shared handsLabel utility**

```typescript
// src/lib/exercise-utils.ts
export function handsLabel(hands: "left" | "right" | "both"): string {
	if (hands === "left") return "LH";
	if (hands === "right") return "RH";
	return "Both";
}
```

- [ ] **Step 3: Verify the app builds**

Run: `cd apps/web && bun run build`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/contexts/artifact-scroll.ts apps/web/src/lib/exercise-utils.ts
git commit -m "feat(web): add ArtifactScrollContext and shared exercise utils"
```

---

### Task 2: CSS Animations

**Files:**
- Modify: `src/styles/app.css`

- [ ] **Step 1: Add artifact animation keyframes**

Add after the existing `.animate-panel-slide-in` block (line 355 of `app.css`):

```css
/* Artifact container transitions */
@keyframes artifact-collapse {
	from {
		opacity: 1;
	}
	to {
		opacity: 0;
	}
}

@keyframes artifact-expand-inline {
	from {
		opacity: 0;
	}
	to {
		opacity: 1;
	}
}

@keyframes backdrop-in {
	from {
		opacity: 0;
	}
	to {
		opacity: 1;
	}
}

@keyframes backdrop-out {
	from {
		opacity: 1;
	}
	to {
		opacity: 0;
	}
}

.animate-artifact-collapse {
	animation: artifact-collapse 200ms ease-out both;
}

.animate-artifact-expand-inline {
	animation: artifact-expand-inline 300ms ease-out both;
}

.animate-backdrop-in {
	animation: backdrop-in 200ms ease-out both;
}

.animate-backdrop-out {
	animation: backdrop-out 200ms ease-in both;
}

@keyframes panel-out {
	from {
		opacity: 1;
		transform: translateY(0) scale(1);
	}
	to {
		opacity: 0;
		transform: translateY(12px) scale(0.98);
	}
}

.animate-panel-out {
	animation: panel-out 150ms ease-in both;
}
```

- [ ] **Step 2: Verify the app still builds**

Run: `cd apps/web && bun run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/styles/app.css
git commit -m "feat(web): add artifact transition animations"
```

---

### Task 3: CollapsedPreview Component

**Files:**
- Create: `src/components/cards/CollapsedPreview.tsx`

- [ ] **Step 1: Create the collapsed mini card component**

```tsx
// src/components/cards/CollapsedPreview.tsx
import { CaretRight } from "@phosphor-icons/react";

interface CollapsedPreviewProps {
	title: string;
	subtitle: string;
	badge: string;
	onRestore: () => void;
	onExpand: () => void;
}

export function CollapsedPreview({
	title,
	subtitle,
	badge,
	onRestore,
	onExpand,
}: CollapsedPreviewProps) {
	return (
		<div
			role="button"
			tabIndex={0}
			onClick={onRestore}
			onKeyDown={(e) => {
				if (e.key === "Enter" || e.key === " ") {
					e.preventDefault();
					onRestore();
				}
			}}
			className="bg-surface-card border border-border rounded-xl px-3 py-2 mt-3 flex items-center gap-3 cursor-pointer hover:bg-surface transition group"
		>
			{/* Accent bar */}
			<div className="w-1 self-stretch rounded-full bg-accent shrink-0" />

			{/* Content */}
			<div className="flex-1 min-w-0">
				<div className="flex items-center gap-2">
					<span className="text-body-sm font-medium text-cream truncate">
						{title}
					</span>
					<span className="text-body-xs text-text-tertiary shrink-0">
						{badge}
					</span>
				</div>
				<p className="text-body-xs text-text-secondary truncate">
					{subtitle}
				</p>
			</div>

			{/* Expand chevron */}
			<button
				type="button"
				onClick={(e) => {
					e.stopPropagation();
					onExpand();
				}}
				className="shrink-0 w-7 h-7 flex items-center justify-center rounded-md text-text-tertiary hover:text-cream hover:bg-surface-2 transition"
				aria-label="Expand artifact"
			>
				<CaretRight size={14} weight="bold" />
			</button>
		</div>
	);
}
```

- [ ] **Step 2: Verify the app builds**

Run: `cd apps/web && bun run build`
Expected: Build succeeds (component not yet wired)

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/cards/CollapsedPreview.tsx
git commit -m "feat(web): add CollapsedPreview mini card component"
```

---

### Task 4: Modify ExerciseSetCard (Add Expand Button + Store Integration)

**Files:**
- Modify: `src/components/cards/ExerciseSetCard.tsx`

- [ ] **Step 1: Add onExpand prop and expand icon to ExerciseSetCard header**

Modify `ExerciseSetCard` to accept `onExpand` and `artifactId` props. Add expand icon to card header. Refactor `ExerciseItem` to read assign state from the artifact store instead of local `useState`.

```tsx
// src/components/cards/ExerciseSetCard.tsx
import { useState } from "react";
import { ArrowsOut } from "@phosphor-icons/react";
import { api } from "../../lib/api";
import type { ExerciseSetConfig } from "../../lib/types";
import { useArtifactStore } from "../../stores/artifact";
import { handsLabel } from "../../lib/exercise-utils";

interface ExerciseSetCardProps {
	config: ExerciseSetConfig;
	onExpand?: () => void;
	artifactId?: string;
}

type LocalAssignState = "idle" | "loading" | "assigned" | "error";

interface ExerciseItemProps {
	exercise: ExerciseSetConfig["exercises"][number];
	isExpanded: boolean;
	onToggle: () => void;
	artifactId?: string;
}

function ExerciseItem({
	exercise,
	isExpanded,
	onToggle,
	artifactId,
}: ExerciseItemProps) {
	// Store-backed state when artifactId is provided (inside Artifact wrapper)
	const exerciseState = useArtifactStore((s) => {
		if (!artifactId || !exercise.exercise_id) return undefined;
		return s.states[artifactId]?.exerciseStates?.[exercise.exercise_id];
	});
	const setExerciseStatus = useArtifactStore((s) => s.setExerciseStatus);

	// Local fallback for standalone usage (without Artifact wrapper)
	const [localState, setLocalState] = useState<LocalAssignState>("idle");
	const useStore = Boolean(artifactId);

	const status = useStore ? (exerciseState?.status ?? "idle") : localState;

	async function handleAssign() {
		if (!exercise.exercise_id) return;
		if (useStore && artifactId) {
			setExerciseStatus(artifactId, exercise.exercise_id, "loading");
		} else {
			setLocalState("loading");
		}
		try {
			const result = await api.exercises.assign({
				exercise_id: exercise.exercise_id,
			});
			if (useStore && artifactId) {
				setExerciseStatus(
					artifactId,
					exercise.exercise_id,
					"assigned",
					result.id,
				);
			} else {
				setLocalState("assigned");
			}
		} catch {
			if (useStore && artifactId) {
				setExerciseStatus(artifactId, exercise.exercise_id, "error");
			} else {
				setLocalState("error");
			}
		}
	}

	return (
		<div className="border border-border rounded-lg overflow-hidden">
			<button
				type="button"
				onClick={onToggle}
				className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-surface transition"
			>
				<span className="text-body-sm text-cream font-medium">
					{exercise.title}
				</span>
				<div className="flex items-center gap-1.5 ml-2 shrink-0">
					{exercise.hands && (
						<span className="text-body-xs text-text-tertiary bg-surface px-1.5 py-0.5 rounded">
							{handsLabel(exercise.hands)}
						</span>
					)}
					<span className="text-body-xs text-text-tertiary">
						{exercise.focus_dimension}
					</span>
				</div>
			</button>
			{isExpanded && (
				<div className="px-3 pb-3 pt-1 border-t border-border">
					<p className="text-body-sm text-text-secondary mb-3">
						{exercise.instruction}
					</p>
					{exercise.exercise_id && (
						<button
							type="button"
							onClick={handleAssign}
							disabled={
								status === "loading" ||
								status === "assigned" ||
								status === "completed"
							}
							className={`text-body-xs px-3 py-1.5 rounded-lg border transition ${
								status === "assigned" || status === "completed"
									? "border-accent text-accent cursor-default"
									: status === "error"
										? "border-red-500 text-red-400 hover:bg-red-500/10"
										: "border-border text-text-secondary hover:text-cream hover:border-accent hover:bg-surface disabled:opacity-50"
							}`}
						>
							{status === "loading"
								? "Assigning..."
								: status === "assigned"
									? "Added to practice"
									: status === "completed"
										? "Completed"
										: status === "error"
											? "Try again"
											: "Try this"}
						</button>
					)}
				</div>
			)}
		</div>
	);
}

export function ExerciseSetCard({
	config,
	onExpand,
	artifactId,
}: ExerciseSetCardProps) {
	const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

	return (
		<div className="bg-surface-card border border-border rounded-xl p-4 mt-3">
			<div className="flex items-start justify-between mb-1">
				<h4 className="text-body-sm font-medium text-accent">
					{config.target_skill}
				</h4>
				{onExpand && (
					<button
						type="button"
						onClick={onExpand}
						className="shrink-0 w-7 h-7 flex items-center justify-center rounded-md text-text-tertiary hover:text-cream hover:bg-surface transition"
						aria-label="Expand exercises"
					>
						<ArrowsOut size={14} weight="bold" />
					</button>
				)}
			</div>
			<p className="text-body-xs text-text-secondary mb-3">
				{config.source_passage}
			</p>
			<div className="space-y-2">
				{config.exercises.map((exercise, i) => (
					<ExerciseItem
						key={exercise.title}
						exercise={exercise}
						isExpanded={expandedIndex === i}
						onToggle={() =>
							setExpandedIndex(expandedIndex === i ? null : i)
						}
						artifactId={artifactId}
					/>
				))}
			</div>
		</div>
	);
}
```

- [ ] **Step 2: Verify the app builds**

Run: `cd apps/web && bun run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/cards/ExerciseSetCard.tsx
git commit -m "feat(web): add expand button and store-backed exercise state to ExerciseSetCard"
```

---

### Task 5: Modify InlineCard (Forward onExpand + artifactId)

**Files:**
- Modify: `src/components/InlineCard.tsx`

- [ ] **Step 1: Update InlineCard to accept and forward props**

```tsx
// src/components/InlineCard.tsx
import type { InlineComponent } from "../lib/types";
import { ExerciseSetCard } from "./cards/ExerciseSetCard";
import { PlaceholderCard } from "./cards/PlaceholderCard";

interface InlineCardProps {
	component: InlineComponent;
	onExpand?: () => void;
	artifactId?: string;
}

export function InlineCard({ component, onExpand, artifactId }: InlineCardProps) {
	switch (component.type) {
		case "exercise_set":
			return (
				<ExerciseSetCard
					config={component.config}
					onExpand={onExpand}
					artifactId={artifactId}
				/>
			);
		default:
			return <PlaceholderCard type={component.type} />;
	}
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/components/InlineCard.tsx
git commit -m "feat(web): forward onExpand and artifactId through InlineCard"
```

---

### Task 6: ExerciseSetExpanded Component

**Files:**
- Create: `src/components/cards/ExerciseSetExpanded.tsx`

- [ ] **Step 1: Create the expanded workspace view**

```tsx
// src/components/cards/ExerciseSetExpanded.tsx
import { api } from "../../lib/api";
import type { ExerciseSetConfig } from "../../lib/types";
import {
	useArtifactStore,
	type ExerciseStatus,
} from "../../stores/artifact";
import { handsLabel } from "../../lib/exercise-utils";

interface ExerciseSetExpandedProps {
	config: ExerciseSetConfig;
	artifactId: string;
}

interface ExpandedExerciseItemProps {
	exercise: ExerciseSetConfig["exercises"][number];
	artifactId: string;
}

function ExpandedExerciseItem({
	exercise,
	artifactId,
}: ExpandedExerciseItemProps) {
	const exerciseState = useArtifactStore((s) => {
		if (!exercise.exercise_id) return undefined;
		return s.states[artifactId]?.exerciseStates?.[exercise.exercise_id];
	});
	const setExerciseStatus = useArtifactStore((s) => s.setExerciseStatus);

	const status: ExerciseStatus = exerciseState?.status ?? "idle";
	const studentExerciseId = exerciseState?.studentExerciseId;

	async function handleStart() {
		if (!exercise.exercise_id) return;
		setExerciseStatus(artifactId, exercise.exercise_id, "loading");
		try {
			const result = await api.exercises.assign({
				exercise_id: exercise.exercise_id,
			});
			setExerciseStatus(
				artifactId,
				exercise.exercise_id,
				"assigned",
				result.id,
			);
		} catch {
			setExerciseStatus(artifactId, exercise.exercise_id, "error");
		}
	}

	async function handleComplete() {
		if (!exercise.exercise_id || !studentExerciseId) return;
		setExerciseStatus(
			artifactId,
			exercise.exercise_id,
			"completing",
			studentExerciseId,
		);
		try {
			await api.exercises.complete({
				student_exercise_id: studentExerciseId,
			});
			setExerciseStatus(
				artifactId,
				exercise.exercise_id,
				"completed",
				studentExerciseId,
			);
		} catch {
			setExerciseStatus(
				artifactId,
				exercise.exercise_id,
				"error",
				studentExerciseId,
			);
		}
	}

	function buttonProps(): {
		label: string;
		onClick: () => void;
		disabled: boolean;
		className: string;
	} {
		const base =
			"text-body-sm px-4 py-2 rounded-lg border transition font-medium";

		switch (status) {
			case "idle":
				return {
					label: exercise.exercise_id ? "Start" : "Not yet saved",
					onClick: handleStart,
					disabled: !exercise.exercise_id,
					className: `${base} ${
						exercise.exercise_id
							? "border-accent text-accent hover:bg-accent/10"
							: "border-border text-text-tertiary cursor-not-allowed"
					}`,
				};
			case "loading":
				return {
					label: "Starting...",
					onClick: () => {},
					disabled: true,
					className: `${base} border-border text-text-tertiary opacity-50`,
				};
			case "assigned":
				return {
					label: "Complete",
					onClick: handleComplete,
					disabled: false,
					className: `${base} border-accent text-accent hover:bg-accent/10`,
				};
			case "completing":
				return {
					label: "Completing...",
					onClick: () => {},
					disabled: true,
					className: `${base} border-border text-text-tertiary opacity-50`,
				};
			case "completed":
				return {
					label: "Completed",
					onClick: () => {},
					disabled: true,
					className: `${base} border-accent text-accent cursor-default`,
				};
			case "error":
				return {
					label: "Try again",
					onClick: studentExerciseId ? handleComplete : handleStart,
					disabled: false,
					className: `${base} border-red-500 text-red-400 hover:bg-red-500/10`,
				};
		}
	}

	const btn = buttonProps();

	return (
		<div className="border border-border rounded-xl p-4">
			<div className="flex items-start justify-between mb-2">
				<h5 className="text-body-md font-medium text-cream">
					{exercise.title}
				</h5>
				<div className="flex items-center gap-1.5 ml-2 shrink-0">
					{exercise.hands && (
						<span className="text-body-xs text-text-tertiary bg-surface px-1.5 py-0.5 rounded">
							{handsLabel(exercise.hands)}
						</span>
					)}
					<span className="text-body-xs text-text-tertiary">
						{exercise.focus_dimension}
					</span>
				</div>
			</div>
			<p className="text-body-sm text-text-secondary mb-4 leading-relaxed">
				{exercise.instruction}
			</p>
			<button
				type="button"
				onClick={btn.onClick}
				disabled={btn.disabled}
				className={btn.className}
			>
				{btn.label}
			</button>
		</div>
	);
}

export function ExerciseSetExpanded({
	config,
	artifactId,
}: ExerciseSetExpandedProps) {
	const exerciseStates = useArtifactStore(
		(s) => s.states[artifactId]?.exerciseStates,
	);

	const exercisesWithIds = config.exercises.filter((e) => e.exercise_id);
	const completedCount = exercisesWithIds.filter(
		(e) =>
			e.exercise_id &&
			exerciseStates?.[e.exercise_id]?.status === "completed",
	).length;
	const totalTrackable = exercisesWithIds.length;

	return (
		<div>
			{/* Header */}
			<div className="mb-6">
				<h3 className="text-body-lg font-medium text-cream mb-1">
					{config.target_skill}
				</h3>
				<p className="text-body-sm text-text-secondary">
					{config.source_passage}
				</p>
			</div>

			{/* Exercises */}
			<div className="space-y-3">
				{config.exercises.map((exercise) => (
					<ExpandedExerciseItem
						key={exercise.title}
						exercise={exercise}
						artifactId={artifactId}
					/>
				))}
			</div>

			{/* Progress */}
			{totalTrackable > 0 && (
				<div className="mt-6 pt-4 border-t border-border">
					<div className="flex items-center justify-between text-body-sm">
						<span className="text-text-secondary">Progress</span>
						<span className="text-cream font-medium">
							{completedCount} of {totalTrackable} completed
						</span>
					</div>
					<div className="mt-2 h-1.5 bg-surface rounded-full overflow-hidden">
						<div
							className="h-full bg-accent rounded-full transition-all duration-300"
							style={{
								width: `${(completedCount / totalTrackable) * 100}%`,
							}}
						/>
					</div>
				</div>
			)}
		</div>
	);
}
```

- [ ] **Step 2: Verify the app builds**

Run: `cd apps/web && bun run build`
Expected: Build succeeds

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/cards/ExerciseSetExpanded.tsx
git commit -m "feat(web): add ExerciseSetExpanded workspace view with completion tracking"
```

---

### Task 7: ArtifactOverlay Component

**Files:**
- Create: `src/components/ArtifactOverlay.tsx`

- [ ] **Step 1: Create the overlay portal component**

```tsx
// src/components/ArtifactOverlay.tsx
import { useCallback, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { X } from "@phosphor-icons/react";
import { useArtifactStore, getExpandedArtifact } from "../stores/artifact";
import { ExerciseSetExpanded } from "./cards/ExerciseSetExpanded";
import { useArtifactScrollContext } from "../contexts/artifact-scroll";

function ArtifactOverlayContent() {
	// Atomic selector: both expandedId and entry come from a single subscription
	const expanded = useArtifactStore((s) => getExpandedArtifact(s));
	const expandedId = expanded?.id ?? null;
	const entry = expanded?.entry ?? undefined;
	const closeOverlay = useArtifactStore((s) => s.closeOverlay);
	const scrollContainerRef = useArtifactScrollContext();
	const isClosingRef = useRef(false);
	const overlayRef = useRef<HTMLDivElement>(null);

	const handleClose = useCallback(() => {
		if (!expandedId || isClosingRef.current) return;
		isClosingRef.current = true;

		// Animate out
		const overlay = overlayRef.current;
		if (overlay) {
			const backdrop = overlay.querySelector("[data-backdrop]");
			const panel = overlay.querySelector("[data-panel]");
			backdrop?.classList.remove("animate-backdrop-in");
			backdrop?.classList.add("animate-backdrop-out");
			panel?.classList.remove("animate-overlay-in");
			panel?.classList.add("animate-panel-out");
		}

		setTimeout(() => {
			closeOverlay(expandedId);
			isClosingRef.current = false;
		}, 200);
	}, [expandedId, closeOverlay]);

	// Escape key handler
	useEffect(() => {
		if (!expandedId) return;
		function onKeyDown(e: KeyboardEvent) {
			if (e.key === "Escape") handleClose();
		}
		document.addEventListener("keydown", onKeyDown);
		return () => document.removeEventListener("keydown", onKeyDown);
	}, [expandedId, handleClose]);

	// Scroll lock on chat container
	useEffect(() => {
		if (!expandedId) return;
		const el = scrollContainerRef?.current;
		if (!el) return;
		const prev = el.style.overflow;
		el.style.overflow = "hidden";
		return () => {
			el.style.overflow = prev;
		};
	}, [expandedId, scrollContainerRef]);

	if (!expandedId || !entry) return null;

	function renderExpanded() {
		if (!entry || !expandedId) return null;
		switch (entry.component.type) {
			case "exercise_set":
				return (
					<ExerciseSetExpanded
						config={entry.component.config}
						artifactId={expandedId}
					/>
				);
			default:
				return (
					<p className="text-body-sm text-text-tertiary italic">
						{entry.component.type.replace(/_/g, " ")} (expanded view
						coming soon)
					</p>
				);
		}
	}

	return (
		<div ref={overlayRef} className="fixed inset-0 z-50">
			{/* Backdrop */}
			<div
				data-backdrop
				className="absolute inset-0 bg-black/60 animate-backdrop-in"
				onClick={handleClose}
				onKeyDown={(e) => {
					if (e.key === "Enter" || e.key === " ") handleClose();
				}}
				role="button"
				tabIndex={-1}
				aria-label="Close overlay"
			/>

			{/* Panel */}
			<div
				data-panel
				className="relative max-w-2xl mx-auto mt-16 max-h-[80vh] overflow-y-auto bg-surface-card border border-border rounded-2xl p-6 shadow-card animate-overlay-in"
			>
				{/* Close button */}
				<button
					type="button"
					onClick={handleClose}
					className="absolute top-4 right-4 w-8 h-8 flex items-center justify-center rounded-lg text-text-tertiary hover:text-cream hover:bg-surface transition"
					aria-label="Close"
				>
					<X size={18} weight="bold" />
				</button>

				{renderExpanded()}
			</div>
		</div>
	);
}

export function ArtifactOverlay() {
	const hasExpanded = useArtifactStore(
		(s) => getExpandedArtifact(s) !== null,
	);
	if (!hasExpanded) return null;
	return createPortal(<ArtifactOverlayContent />, document.body);
}
```

- [ ] **Step 2: Verify the app builds**

Run: `cd apps/web && bun run build`
Expected: Build succeeds (imports resolve to `contexts/artifact-scroll.ts` from Task 1b)

- [ ] **Step 3: Commit**

```bash
git add apps/web/src/components/ArtifactOverlay.tsx
git commit -m "feat(web): add ArtifactOverlay portal component with backdrop and scroll lock"
```

---

### Task 8: Artifact Wrapper Component

**Files:**
- Create: `src/components/Artifact.tsx`

- [ ] **Step 1: Create the Artifact state machine wrapper**

```tsx
// src/components/Artifact.tsx
import { useEffect, useRef } from "react";
import type { InlineComponent, ExerciseSetConfig } from "../lib/types";
import { useArtifactStore } from "../stores/artifact";
import { InlineCard } from "./InlineCard";
import { CollapsedPreview } from "./cards/CollapsedPreview";
import { useArtifactScrollContext } from "../contexts/artifact-scroll";

interface ArtifactProps {
	artifactId: string;
	component: InlineComponent;
}

/** Map any InlineComponent to generic CollapsedPreview props. */
function getCollapsedProps(component: InlineComponent): {
	title: string;
	subtitle: string;
	badge: string;
} {
	switch (component.type) {
		case "exercise_set": {
			const config = component.config as ExerciseSetConfig;
			return {
				title: config.target_skill,
				subtitle: config.source_passage,
				badge: `${config.exercises.length} exercise${config.exercises.length !== 1 ? "s" : ""}`,
			};
		}
		default:
			return {
				title: component.type.replace(/_/g, " "),
				subtitle: "",
				badge: "",
			};
	}
}

export function Artifact({ artifactId, component }: ArtifactProps) {
	const entry = useArtifactStore((s) => s.states[artifactId]);
	const register = useArtifactStore((s) => s.register);
	const unregister = useArtifactStore((s) => s.unregister);
	const collapse = useArtifactStore((s) => s.collapse);
	const expand = useArtifactStore((s) => s.expand);
	const restore = useArtifactStore((s) => s.restore);

	const elementRef = useRef<HTMLDivElement>(null);
	const collapseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const mountedRef = useRef(false);
	const scrollContainerRef = useArtifactScrollContext();

	// Register on mount, unregister on unmount
	useEffect(() => {
		register(artifactId, component);
		// Delay enabling the observer so the initial intersection callback
		// (which fires immediately with isIntersecting=false if off-screen)
		// does not collapse the artifact before the user has seen it.
		const timer = setTimeout(() => {
			mountedRef.current = true;
		}, 1000);
		return () => {
			clearTimeout(timer);
			mountedRef.current = false;
			unregister(artifactId);
		};
	}, [artifactId, component, register, unregister]);

	// IntersectionObserver for auto-collapse
	useEffect(() => {
		const el = elementRef.current;
		const root = scrollContainerRef?.current;
		if (!el || !root) return;

		const observer = new IntersectionObserver(
			([entry]) => {
				// Skip callbacks until mount grace period expires
				if (!mountedRef.current) return;
				if (entry.isIntersecting) {
					// Came back into view -- cancel pending collapse
					if (collapseTimerRef.current) {
						clearTimeout(collapseTimerRef.current);
						collapseTimerRef.current = null;
					}
				} else {
					// Scrolled out -- schedule collapse after 200ms
					collapseTimerRef.current = setTimeout(() => {
						collapse(artifactId);
						collapseTimerRef.current = null;
					}, 200);
				}
			},
			{ root, threshold: [0] },
		);

		observer.observe(el);
		return () => {
			observer.disconnect();
			if (collapseTimerRef.current) {
				clearTimeout(collapseTimerRef.current);
			}
		};
	}, [artifactId, collapse, scrollContainerRef]);

	const displayState = entry?.state ?? "inline";

	return (
		<div ref={elementRef}>
			{displayState === "collapsed" && (
				<div className="animate-artifact-expand-inline">
					<CollapsedPreview
						{...getCollapsedProps(component)}
						onRestore={() => restore(artifactId)}
						onExpand={() => expand(artifactId)}
					/>
				</div>
			)}
			{displayState === "inline" && (
				<div className="animate-fade-in">
					<InlineCard
						component={component}
						onExpand={() => expand(artifactId)}
						artifactId={artifactId}
					/>
				</div>
			)}
			{/* Expanded state: inline card stays visible in chat (dimmed behind overlay).
			    The overlay itself renders via ArtifactOverlay portal. */}
			{displayState === "expanded" && (
				<InlineCard
					component={component}
					onExpand={() => {}}
					artifactId={artifactId}
				/>
			)}
		</div>
	);
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/components/Artifact.tsx
git commit -m "feat(web): add Artifact wrapper with state machine and IntersectionObserver"
```

---

### Task 9: Wire Into ChatMessages (ArtifactScrollContext + Swap InlineCard for Artifact)

**Files:**
- Modify: `src/components/ChatMessages.tsx`

- [ ] **Step 1: Add ArtifactScrollContext and replace InlineCard with Artifact**

Replace the contents of `ChatMessages.tsx` with:

```tsx
// src/components/ChatMessages.tsx
import {
	memo,
	useCallback,
	useEffect,
	useRef,
	useState,
} from "react";
import type { RichMessage } from "../lib/types";
import { Artifact } from "./Artifact";
import { MessageContent } from "./MessageContent";
import { ArtifactScrollContext } from "../contexts/artifact-scroll";

interface ChatMessagesProps {
	messages: RichMessage[];
	children?: React.ReactNode;
	onTryExercises?: (dimension: string) => Promise<void>;
}

export function ChatMessages({
	messages,
	children,
	onTryExercises,
}: ChatMessagesProps) {
	const scrollContainerRef = useRef<HTMLDivElement>(null);
	const isNearBottomRef = useRef(true);
	const prevMessageCountRef = useRef(0);

	const scrollToBottom = useCallback(
		(behavior: ScrollBehavior = "instant") => {
			const container = scrollContainerRef.current;
			if (!container) return;
			if (behavior === "smooth") {
				container.scrollTo({
					top: container.scrollHeight,
					behavior: "smooth",
				});
			} else {
				container.scrollTop = container.scrollHeight;
			}
		},
		[],
	);

	// Track whether user is near the bottom
	useEffect(() => {
		const container = scrollContainerRef.current;
		if (!container) return;

		function handleScroll() {
			if (!container) return;
			const threshold = 150;
			const distanceFromBottom =
				container.scrollHeight -
				container.scrollTop -
				container.clientHeight;
			isNearBottomRef.current = distanceFromBottom <= threshold;
		}

		container.addEventListener("scroll", handleScroll, { passive: true });
		return () => container.removeEventListener("scroll", handleScroll);
	}, []);

	// Auto-scroll on content changes
	useEffect(() => {
		if (!isNearBottomRef.current) return;

		const isNewMessage = messages.length > prevMessageCountRef.current;
		prevMessageCountRef.current = messages.length;

		const lastMsg = messages[messages.length - 1];
		const behavior = lastMsg?.streaming
			? "instant"
			: isNewMessage
				? "smooth"
				: "instant";
		scrollToBottom(behavior);
	}, [messages, scrollToBottom]);

	// Scroll on mount
	useEffect(() => {
		scrollToBottom("instant");
	}, [scrollToBottom]);

	if (messages.length === 0) {
		return null;
	}

	return (
		<ArtifactScrollContext.Provider value={scrollContainerRef}>
			<div
				ref={scrollContainerRef}
				className="flex-1 overflow-y-auto px-6 pt-8 flex flex-col"
				style={{ scrollBehavior: "auto" }}
			>
				<div className="flex-1 max-w-3xl mx-auto space-y-6 w-full">
					{messages.map((msg) => (
						<MessageBubble
							key={msg.id}
							message={msg}
							onTryExercises={onTryExercises}
						/>
					))}
				</div>
				{children}
			</div>
		</ArtifactScrollContext.Provider>
	);
}

const MessageBubble = memo(function MessageBubble({
	message,
	onTryExercises,
}: {
	message: RichMessage;
	onTryExercises?: (dimension: string) => Promise<void>;
}) {
	const [tryState, setTryState] = useState<"idle" | "loading" | "error">(
		"idle",
	);

	if (message.role === "user") {
		return (
			<div className="flex justify-end">
				<div className="bg-surface border border-border rounded-2xl px-5 py-3 max-w-[80%]">
					<p className="text-body-md text-cream whitespace-pre-wrap">
						{message.content}
					</p>
				</div>
			</div>
		);
	}

	async function handleTryExercises() {
		if (!message.dimension || !onTryExercises) return;
		setTryState("loading");
		try {
			await onTryExercises(message.dimension);
			setTryState("idle");
		} catch {
			setTryState("error");
		}
	}

	const showTryButton =
		message.dimension && !message.streaming && onTryExercises;

	return (
		<div className="flex justify-start animate-fade-in">
			<div className="max-w-[80%]">
				<MessageContent content={message.content} />
				{!message.streaming &&
					message.components?.map((component, i) => (
						<Artifact
							key={`${message.id}-artifact-${i}`}
							artifactId={`${message.id}-artifact-${i}`}
							component={component}
						/>
					))}
				{showTryButton && (
					<button
						type="button"
						onClick={handleTryExercises}
						disabled={tryState === "loading"}
						className={`mt-2 text-body-xs px-3 py-1.5 rounded-lg border transition ${
							tryState === "error"
								? "border-red-500 text-red-400 hover:bg-red-500/10"
								: "border-border text-text-tertiary hover:text-cream hover:border-accent hover:bg-surface disabled:opacity-50"
						}`}
					>
						{tryState === "loading"
							? "Loading exercises..."
							: tryState === "error"
								? "Try again"
								: "Try exercises for this"}
					</button>
				)}
			</div>
		</div>
	);
});
```

Key changes from the original:
- Import `Artifact` instead of `InlineCard`
- Import `ArtifactScrollContext` from `contexts/artifact-scroll` (extracted to its own file)
- Wrap scroll container in `ArtifactScrollContext.Provider`
- Replace `<InlineCard>` with `<Artifact>` in `MessageBubble`
- Guard artifact rendering on `!message.streaming`
- Use `${message.id}-artifact-${i}` as stable artifact ID

- [ ] **Step 2: Verify the app builds**

Run: `cd apps/web && bun run build`
Expected: Build succeeds

- [ ] **Step 3: Run existing tests**

Run: `cd apps/web && bun run test`
Expected: All tests pass (store tests + existing tests)

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/components/ChatMessages.tsx
git commit -m "feat(web): wire Artifact into ChatMessages with scroll context"
```

---

### Task 10: Wire ArtifactOverlay Into AppChat

**Files:**
- Modify: `src/components/AppChat.tsx`

- [ ] **Step 1: Add ArtifactOverlay import and render**

At the top of `AppChat.tsx`, add the import:

```typescript
import { ArtifactOverlay } from "./ArtifactOverlay";
```

Then inside the render, add `<ArtifactOverlay />` as a sibling right after the main content div closes (after line 788 `</div>`, before `<ScorePanel />`):

```tsx
			</div>

			{/* Artifact expanded overlay */}
			<ArtifactOverlay />

			{/* Score panel (artifacts-style right sidebar) */}
			<ScorePanel />
```

- [ ] **Step 2: Verify the app builds**

Run: `cd apps/web && bun run build`
Expected: Build succeeds

- [ ] **Step 3: Run all tests**

Run: `cd apps/web && bun run test`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/components/AppChat.tsx
git commit -m "feat(web): mount ArtifactOverlay in AppChat"
```

---

### Task 11: Manual Smoke Test

This is a manual verification task. No code changes.

- [ ] **Step 1: Start the dev server**

Run: `cd apps/web && bun run dev`

- [ ] **Step 2: Verify inline card renders**

Navigate to the app, start a conversation, and trigger an observation with exercises (click "Try exercises for this" on an observation with a dimension). Verify:
- The exercise card renders inline in the chat with an expand icon (ArrowsOut) in the top-right header
- Exercises accordion still works (click to expand/collapse individual exercises)

- [ ] **Step 3: Verify expanded overlay**

Click the expand icon on the exercise card. Verify:
- Overlay appears with backdrop (dimmed background)
- All exercises are shown expanded (no accordion)
- Each exercise has a "Start" button (or "Not yet saved" if no exercise_id)
- Close button (X) works
- Escape key closes the overlay
- Clicking the backdrop closes the overlay
- Progress bar shows at the bottom

- [ ] **Step 4: Verify collapsed state**

Scroll the chat so the exercise card is fully out of view (send several messages to push it up). Then scroll back. Verify:
- The card has collapsed to a mini card (~56px) showing target_skill, exercise count, and source_passage
- Clicking the mini card body restores it to inline
- Clicking the chevron on the mini card opens the expanded overlay

- [ ] **Step 5: Verify exercise state sync**

Assign an exercise in the inline card ("Try this"), then expand. Verify the expanded view shows "Complete" (not "Start") for that exercise. Complete it in the expanded view, close the overlay, and verify the inline card shows "Completed".

---

## Dependency Order

```
Task 1 (store) ──────────────┐
Task 1b (scroll ctx + utils) ┤
Task 2 (CSS) ────────────────┤
Task 3 (CollapsedPreview) ───┤
                             ▼
Task 4 (ExerciseSetCard) ────┤
Task 5 (InlineCard) ────────┤
Task 6 (ExerciseSetExpanded) │
                             ▼
Task 7 (ArtifactOverlay) ───┤  (imports from contexts/artifact-scroll, not ChatMessages)
Task 8 (Artifact wrapper) ──┤  (imports from contexts/artifact-scroll, not ChatMessages)
                             ▼
Task 9 (ChatMessages wiring) ──── all components compile independently
                             │
Task 10 (AppChat wiring) ───┘
                             │
Task 11 (Smoke test) ───────┘
```

Tasks 1, 1b, 2, 3 can run in parallel. Tasks 4, 5, 6 can run in parallel (after 1 + 1b). Tasks 7, 8 can run in parallel (all imports resolve). Tasks 9, 10 are sequential wiring. Task 11 requires everything. Every commit compiles.
