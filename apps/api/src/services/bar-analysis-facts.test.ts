import { describe, expect, it } from "vitest";
import { buildBarAnalysisFacts } from "./bar-analysis-facts";
import type { ChunkAnalysis } from "./wasm-bridge";

const baselines = {
	dynamics: 0.5,
	timing: 0.5,
	pedaling: 0.5,
	articulation: 0.5,
	phrasing: 0.5,
	interpretation: 0.5,
} as const;

describe("buildBarAnalysisFacts", () => {
	it("returns null when analysis.dimensions is empty", () => {
		const analysis: ChunkAnalysis = {
			tier: 3,
			bar_range: null,
			dimensions: [],
		};
		const scores: [number, number, number, number, number, number] = [
			0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		];
		const result = buildBarAnalysisFacts(
			analysis,
			scores,
			baselines,
			"timing",
		);
		expect(result).toBeNull();
	});

	it("selected is the matching dimension; correlated excludes it", () => {
		const analysis: ChunkAnalysis = {
			tier: 1,
			bar_range: "4-7",
			dimensions: [
				{ dimension: "dynamics", analysis: "dyn-text" },
				{ dimension: "timing", analysis: "tim-text" },
				{ dimension: "pedaling", analysis: "ped-text" },
				{ dimension: "articulation", analysis: "art-text" },
				{ dimension: "phrasing", analysis: "phr-text" },
				{ dimension: "interpretation", analysis: "int-text" },
			],
		};
		// All scores equal baseline → no correlated entries clear the 0.15 threshold.
		const scores: [number, number, number, number, number, number] = [
			0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
		];
		const result = buildBarAnalysisFacts(analysis, scores, baselines, "timing");
		expect(result).not.toBeNull();
		expect(result?.selected.dimension).toBe("timing");
		expect(result?.selected.analysis).toBe("tim-text");
		expect(result?.correlated.map((d) => d.dimension)).not.toContain("timing");
		expect(result?.correlated).toHaveLength(0);
		expect(result?.tier).toBe(1);
		expect(result?.bar_range).toBe("4-7");
	});
});
