import { describe, expect, it } from "vitest";
import { resolveTheme } from "./theme-resolve";

describe("resolveTheme", () => {
	it("honors a stored manual override over the clock", () => {
		const noon = new Date(2026, 0, 1, 12, 0);
		expect(resolveTheme({ stored: "dark", now: noon })).toBe("dark");
		const midnight = new Date(2026, 0, 1, 23, 0);
		expect(resolveTheme({ stored: "light", now: midnight })).toBe("light");
	});

	it("ignores an invalid stored value and falls through to the clock", () => {
		const noon = new Date(2026, 0, 1, 12, 0);
		expect(resolveTheme({ stored: "sepia", now: noon })).toBe("light");
	});

	it("is dark from 19:00 up to (not including) 07:00, device-local", () => {
		expect(
			resolveTheme({ stored: null, now: new Date(2026, 0, 1, 19, 0) }),
		).toBe("dark");
		expect(
			resolveTheme({ stored: null, now: new Date(2026, 0, 1, 23, 59) }),
		).toBe("dark");
		expect(
			resolveTheme({ stored: null, now: new Date(2026, 0, 1, 6, 59) }),
		).toBe("dark");
		expect(
			resolveTheme({ stored: null, now: new Date(2026, 0, 1, 18, 59) }),
		).toBe("light");
		expect(
			resolveTheme({ stored: null, now: new Date(2026, 0, 1, 7, 0) }),
		).toBe("light");
	});

	it("falls back to light when no clock is available", () => {
		expect(resolveTheme({ stored: null, now: null })).toBe("light");
	});
});
