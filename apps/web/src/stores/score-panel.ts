import { create } from "zustand";
import type { MockSessionData } from "../lib/mock-session";
import type { ScoreHighlightConfig } from "../lib/types";

interface ScorePanelState {
	isOpen: boolean;
	sessionData: MockSessionData | null;
	highlightData: ScoreHighlightConfig | null;
	activeAnnotationIndex: number | null;
	panelWidth: number;
	open: (data: MockSessionData) => void;
	openHighlight: (data: ScoreHighlightConfig) => void;
	close: () => void;
	toggle: () => void;
	setActiveAnnotation: (index: number | null) => void;
	setPanelWidth: (width: number) => void;
	clear: () => void;
}

const PANEL_WIDTH_KEY = "crescend-score-panel-width";
const DEFAULT_PANEL_WIDTH = 480;

function loadPersistedWidth(): number {
	try {
		const stored = localStorage.getItem(PANEL_WIDTH_KEY);
		if (stored) {
			const parsed = Number(stored);
			if (parsed >= 320 && parsed <= window.innerWidth * 0.6) return parsed;
		}
	} catch {
		// SSR or localStorage unavailable
	}
	return DEFAULT_PANEL_WIDTH;
}

export const useScorePanelStore = create<ScorePanelState>((set) => ({
	isOpen: false,
	sessionData: null,
	highlightData: null,
	activeAnnotationIndex: null,
	panelWidth: loadPersistedWidth(),
	open: (data) => {
		// DEV gate removed -- panel works in all environments
		set({
			isOpen: true,
			sessionData: data,
			highlightData: null,
			activeAnnotationIndex: null,
		});
	},
	openHighlight: (data) => {
		set({
			isOpen: true,
			highlightData: data,
			sessionData: null,
			activeAnnotationIndex: null,
		});
	},
	close: () => set({ isOpen: false }),
	toggle: () => set((s) => ({ isOpen: !s.isOpen })),
	setActiveAnnotation: (index) => set({ activeAnnotationIndex: index }),
	setPanelWidth: (width) => {
		try {
			localStorage.setItem(PANEL_WIDTH_KEY, String(width));
		} catch {
			// SSR or localStorage unavailable
		}
		set({ panelWidth: width });
	},
	clear: () =>
		set({
			isOpen: false,
			sessionData: null,
			highlightData: null,
			activeAnnotationIndex: null,
		}),
}));
