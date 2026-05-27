import { describe, expect, it } from "vitest";
import { SessionAccumulator, type AccumulatedMoment } from "./accumulator";
import type { BarAnalysisFacts } from "./bar-analysis-facts";

describe("AccumulatedMoment.llmAnalysis", () => {
	it("round-trips a BarAnalysisFacts object through toJSON/fromJSON", () => {
		const facts: BarAnalysisFacts = {
			tier: 1,
			bar_range: "4-7",
			selected: { dimension: "timing", analysis: "rushing 45ms" },
			correlated: [{ dimension: "articulation", analysis: "clipped" }],
		};
		const moment: AccumulatedMoment = {
			chunkIndex: 0,
			dimension: "timing",
			score: 0.3,
			baseline: 0.5,
			deviation: -0.2,
			isPositive: false,
			reasoning: "below baseline",
			barRange: [4, 7],
			analysisTier: 1,
			timestampMs: 1,
			llmAnalysis: facts,
		};
		const acc = new SessionAccumulator();
		acc.accumulateMoment(moment);
		const restored = SessionAccumulator.fromJSON(JSON.parse(JSON.stringify(acc.toJSON())));
		expect(restored.teachingMoments[0]?.llmAnalysis).toEqual(facts);
	});
});
