import { describe, expect, it } from "vitest";
import { DIMENSION_COLOR_VAR } from "./dimension-colors";

describe("DIMENSION_COLOR_VAR", () => {
	it("has exactly the six score dimensions", () => {
		expect(Object.keys(DIMENSION_COLOR_VAR).sort()).toEqual(
			[
				"articulation",
				"dynamics",
				"interpretation",
				"pedaling",
				"phrasing",
				"timing",
			].sort(),
		);
	});

	it("resolves each dimension to its own CSS variable reference", () => {
		expect(DIMENSION_COLOR_VAR.dynamics).toBe("var(--dim-dynamics)");
		expect(DIMENSION_COLOR_VAR.timing).toBe("var(--dim-timing)");
		expect(DIMENSION_COLOR_VAR.pedaling).toBe("var(--dim-pedaling)");
		expect(DIMENSION_COLOR_VAR.articulation).toBe("var(--dim-articulation)");
		expect(DIMENSION_COLOR_VAR.phrasing).toBe("var(--dim-phrasing)");
		expect(DIMENSION_COLOR_VAR.interpretation).toBe(
			"var(--dim-interpretation)",
		);
	});
});
