// Regression test: dead SvgClip helpers are deleted.
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("SvgClip deletion", () => {
	it("SvgClip.tsx and SvgClipBBox.tsx files are deleted from the components dir", () => {
		const root = resolve(__dirname, "..", "..");
		expect(existsSync(`${root}/src/components/SvgClip.tsx`)).toBe(false);
		expect(existsSync(`${root}/src/components/SvgClipBBox.tsx`)).toBe(false);
	});
});
