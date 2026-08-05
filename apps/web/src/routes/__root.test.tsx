import { describe, expect, it } from "vitest";
import { DAWN_HOUR, DUSK_HOUR } from "../lib/theme-resolve";
import { resolveDocumentTheme, THEME_FLASH_SCRIPT } from "./__root";

describe("THEME_FLASH_SCRIPT", () => {
	it("uses the same dusk/dawn hour constants as resolveTheme, so the two can't drift apart", () => {
		const match = THEME_FLASH_SCRIPT.match(/h>=(\d+)\|\|h<(\d+)/);
		expect(match).not.toBeNull();
		const [, dusk, dawn] = match as RegExpMatchArray;
		expect(Number(dusk)).toBe(DUSK_HOUR);
		expect(Number(dawn)).toBe(DAWN_HOUR);
	});
});

describe("resolveDocumentTheme", () => {
	it("is always dark on the always-dark marketing routes", () => {
		expect(resolveDocumentTheme({ pathname: "/", storeTheme: "light" })).toBe(
			"dark",
		);
		expect(
			resolveDocumentTheme({ pathname: "/signin", storeTheme: "light" }),
		).toBe("dark");
	});

	it("follows the store's theme on app routes", () => {
		expect(
			resolveDocumentTheme({ pathname: "/app", storeTheme: "light" }),
		).toBe("light");
		expect(resolveDocumentTheme({ pathname: "/app", storeTheme: "dark" })).toBe(
			"dark",
		);
	});
});
