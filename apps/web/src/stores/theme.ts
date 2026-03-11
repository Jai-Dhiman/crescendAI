import { create } from "zustand";

interface ThemeState {
	theme: "light" | "dark";
	toggleTheme: () => void;
}

function getSystemTheme(): "light" | "dark" {
	if (typeof window === "undefined") return "dark";
	return window.matchMedia("(prefers-color-scheme: dark)").matches
		? "dark"
		: "light";
}

function readStorage(): "light" | "dark" {
	if (typeof window === "undefined") return "dark";
	const stored = localStorage.getItem("crescend-theme");
	if (stored === "light" || stored === "dark") return stored;
	return getSystemTheme();
}

const initial = readStorage();

export const useThemeStore = create<ThemeState>((set) => ({
	theme: initial,
	toggleTheme: () =>
		set((s) => {
			const next = s.theme === "dark" ? "light" : "dark";
			localStorage.setItem("crescend-theme", next);
			return { theme: next };
		}),
}));
