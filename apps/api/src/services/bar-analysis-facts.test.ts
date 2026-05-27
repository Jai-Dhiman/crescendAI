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

	it("correlated is capped at 2 and sorted by absolute deviation descending", () => {
		const analysis: ChunkAnalysis = {
			tier: 1,
			bar_range: "12-14",
			dimensions: [
				{ dimension: "dynamics", analysis: "dyn" },
				{ dimension: "timing", analysis: "tim" },
				{ dimension: "pedaling", analysis: "ped" },
				{ dimension: "articulation", analysis: "art" },
				{ dimension: "phrasing", analysis: "phr" },
				{ dimension: "interpretation", analysis: "int" },
			],
		};
		// deviations from baseline 0.5: dyn=+0.30, tim=selected, ped=-0.20, art=+0.40, phr=+0.05, int=-0.25
		// Non-selected ≥0.15: dyn(0.30), ped(0.20), art(0.40), int(0.25). Top 2: art(0.40), dyn(0.30).
		const scores: [number, number, number, number, number, number] = [
			0.80, 0.50, 0.30, 0.90, 0.55, 0.25,
		];
		const result = buildBarAnalysisFacts(analysis, scores, baselines, "timing");
		expect(result?.correlated.map((d) => d.dimension)).toEqual([
			"articulation",
			"dynamics",
		]);
	});

	it("dimensions below the 0.15 threshold are excluded from correlated", () => {
		const analysis: ChunkAnalysis = {
			tier: 1,
			bar_range: "1-3",
			dimensions: [
				{ dimension: "dynamics", analysis: "dyn" },
				{ dimension: "timing", analysis: "tim" },
			],
		};
		// dyn deviation 0.14 < 0.15 → excluded.
		const scores: [number, number, number, number, number, number] = [
			0.64, 0.50, 0.50, 0.50, 0.50, 0.50,
		];
		const result = buildBarAnalysisFacts(analysis, scores, baselines, "timing");
		expect(result?.correlated).toHaveLength(0);
	});
});
