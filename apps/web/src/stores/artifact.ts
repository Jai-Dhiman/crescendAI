import { create } from "zustand";
import type { InlineComponent } from "../lib/types";

export type ExerciseStatus = "idle" | "loading" | "assigned" | "completing" | "completed" | "error";

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

// Pure selector function -- use in components for atomic reads
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
		set((state) => ({
			states: {
				...state.states,
				[id]: { state: "inline", component },
			},
		}));
	},

	collapse(id) {
		set((state) => {
			const entry = state.states[id];
			if (!entry || entry.state !== "inline") {
				return state;
			}
			return {
				states: {
					...state.states,
					[id]: { ...entry, state: "collapsed" },
				},
			};
		});
	},

	expand(id) {
		set((state) => {
			const updated: Record<string, ArtifactEntry> = {};
			for (const [key, entry] of Object.entries(state.states)) {
				if (key === id) {
					updated[key] = { ...entry, state: "expanded" };
				} else if (entry.state === "expanded") {
					updated[key] = { ...entry, state: "inline" };
				} else {
					updated[key] = entry;
				}
			}
			return { states: updated };
		});
	},

	restore(id) {
		set((state) => {
			const entry = state.states[id];
			if (!entry || entry.state !== "collapsed") {
				return state;
			}
			return {
				states: {
					...state.states,
					[id]: { ...entry, state: "inline" },
				},
			};
		});
	},

	closeOverlay(id) {
		set((state) => {
			const entry = state.states[id];
			if (!entry || entry.state !== "expanded") {
				return state;
			}
			return {
				states: {
					...state.states,
					[id]: { ...entry, state: "inline" },
				},
			};
		});
	},

	unregister(id) {
		set((state) => {
			const { [id]: _, ...remaining } = state.states;
			return { states: remaining };
		});
	},

	setExerciseStatus(artifactId, exerciseId, status, studentExerciseId) {
		set((state) => {
			const entry = state.states[artifactId];
			if (!entry) {
				return state;
			}
			const exerciseState: ExerciseState = studentExerciseId
				? { status, studentExerciseId }
				: { status };
			return {
				states: {
					...state.states,
					[artifactId]: {
						...entry,
						exerciseStates: {
							...entry.exerciseStates,
							[exerciseId]: exerciseState,
						},
					},
				},
			};
		});
	},
}));
