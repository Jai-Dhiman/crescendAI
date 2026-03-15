import { create } from "zustand";
import type { MockSessionData } from "../lib/mock-session";

interface ScorePanelState {
	isOpen: boolean;
	sessionData: MockSessionData | null;
	activeAnnotationIndex: number | null;
	panelWidth: number;
	open: (data: MockSessionData) => void;
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
	activeAnnotationIndex: null,
	panelWidth: loadPersistedWidth(),
	open: (data) => {
		if (!import.meta.env.DEV) return;
		set({ isOpen: true, sessionData: data, activeAnnotationIndex: null });
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
		set({ isOpen: false, sessionData: null, activeAnnotationIndex: null }),
}));
