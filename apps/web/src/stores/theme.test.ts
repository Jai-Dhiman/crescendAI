import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

describe("useThemeStore initial theme", () => {
	beforeEach(() => {
		vi.resetModules();
		localStorage.clear();
	});

	afterEach(() => {
		vi.useRealTimers();
	});

	it("initializes from resolveTheme's time-of-day rule, not prefers-color-scheme", async () => {
		vi.useFakeTimers();
		vi.setSystemTime(new Date(2026, 0, 1, 12, 0)); // noon -> light
		window.matchMedia = vi.fn().mockImplementation(() => ({
			matches: true, // system says dark; should be ignored
		})) as unknown as typeof window.matchMedia;

		const { useThemeStore } = await import("./theme");
		expect(useThemeStore.getState().theme).toBe("light");
	});

	it("honors a stored manual override", async () => {
		localStorage.setItem("crescend-theme", "dark");
		const { useThemeStore } = await import("./theme");
		expect(useThemeStore.getState().theme).toBe("dark");
	});
});
