import { describe, expect, it } from "vitest";
import { contrastRatio } from "./contrast";

describe("contrastRatio", () => {
	it("returns 21:1 for black on white", () => {
		expect(contrastRatio("#000000", "#ffffff")).toBeCloseTo(21, 1);
	});

	it("returns 1:1 for identical colors", () => {
		expect(contrastRatio("#7a9a82", "#7a9a82")).toBeCloseTo(1, 5);
	});

	it("is symmetric", () => {
		const a = contrastRatio("#2a2622", "#fdfaf4");
		const b = contrastRatio("#fdfaf4", "#2a2622");
		expect(a).toBeCloseTo(b, 5);
	});
});
