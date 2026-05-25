// Regression test: dead SvgClip helpers are deleted and the sandbox
// route module loads without dangling imports.
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("SvgClip deletion", () => {
	it("SvgClip.tsx and SvgClipBBox.tsx files are deleted from the components dir", () => {
		const root = resolve(__dirname, "..", "..");
		expect(existsSync(`${root}/src/components/SvgClip.tsx`)).toBe(false);
		expect(existsSync(`${root}/src/components/SvgClipBBox.tsx`)).toBe(false);
	});

	it("app.sandbox route module loads without dangling imports", async () => {
		await expect(import("../routes/app.sandbox")).resolves.toBeDefined();
	});
});
