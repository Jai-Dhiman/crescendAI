import { create } from "zustand";
import { resolveTheme } from "../lib/theme-resolve";

interface ThemeState {
	theme: "light" | "dark";
	toggleTheme: () => void;
}

function readStorage(): string | null {
	if (typeof window === "undefined") return null;
	return localStorage.getItem("crescend-theme");
}

function now(): Date | null {
	return typeof window === "undefined" ? null : new Date();
}

const initial = resolveTheme({ stored: readStorage(), now: now() });

export const useThemeStore = create<ThemeState>((set) => ({
	theme: initial,
	toggleTheme: () =>
		set((s) => {
			const next = s.theme === "dark" ? "light" : "dark";
			localStorage.setItem("crescend-theme", next);
			return { theme: next };
		}),
}));
