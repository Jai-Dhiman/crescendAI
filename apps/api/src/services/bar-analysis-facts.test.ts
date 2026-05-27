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
});
