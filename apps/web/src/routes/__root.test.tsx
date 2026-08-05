import { describe, expect, it } from "vitest";
import { resolveDocumentTheme } from "./__root";

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
