import { describe, expect, it } from "vitest";
import {
	buildProgressSummary,
	ProgressSummarySchema,
} from "./session-progress";

// A raw score distinctive enough to catch if it ever leaked into modelSummary.
const DISTINCT_A = 0.7314159;
const DISTINCT_B = 0.2718281;

describe("buildProgressSummary — artifact shape", () => {
	it("returns a schema-valid ProgressSummary", () => {
		const artifact = buildProgressSummary("session_detail", null);
		expect(() => ProgressSummarySchema.parse(artifact)).not.toThrow();
		expect(typeof artifact.modelSummary).toBe("string");
		expect(artifact.queryType).toBe("session_detail");
	});

	it("keeps raw rows in chartData while modelSummary stays prose", () => {
		const data = [
			{
				id: "sess-new",
				startedAt: new Date("2026-07-04T00:00:00Z"),
				avgDynamics: 0.52,
				avgTiming: DISTINCT_A,
				avgPedaling: DISTINCT_B,
				avgArticulation: 0.6,
				avgPhrasing: 0.58,
				avgInterpretation: 0.55,
			},
		];
		const artifact = buildProgressSummary("recent_sessions", data);
		// Client keeps the numbers…
		expect(JSON.stringify(artifact.chartData)).toContain(String(DISTINCT_A));
		// …the model does not.
		expect(artifact.modelSummary).not.toContain(String(DISTINCT_A));
		expect(artifact.modelSummary).not.toContain(String(DISTINCT_B));
	});
});

describe("buildProgressSummary — dimension_history", () => {
	const data = [
		{
			dimension: "dynamics",
			dimensionScore: 0.42,
			observationText: "Dynamics were flat.",
			framing: "Earlier: dynamics were flat.",
			createdAt: new Date("2026-06-01T00:00:00Z"),
		},
		{
			dimension: "dynamics",
			dimensionScore: DISTINCT_A,
			observationText: "More shape now.",
			framing: "Your dynamic shaping is opening up nicely.",
			createdAt: new Date("2026-07-01T00:00:00Z"),
		},
	];

	it("names the dimension, reports an improving trend, surfaces the latest framing", () => {
		const { modelSummary } = buildProgressSummary("dimension_history", data);
		expect(modelSummary).toContain("dynamics");
		expect(modelSummary.toLowerCase()).toContain("improv");
		expect(modelSummary).toContain(
			"Your dynamic shaping is opening up nicely.",
		);
		expect(modelSummary).not.toContain(String(DISTINCT_A));
		expect(modelSummary).not.toContain("0.42");
	});
});

describe("buildProgressSummary — recent_sessions", () => {
	it("ranks strongest and weakest when there is a real spread, without numbers", () => {
		const data = [
			{
				startedAt: new Date("2026-07-01T00:00:00Z"),
				avgDynamics: DISTINCT_A,
				avgTiming: 0.91,
				avgPedaling: DISTINCT_B,
				avgArticulation: 0.5,
				avgPhrasing: 0.6,
				avgInterpretation: 0.55,
			},
			{
				startedAt: new Date("2026-06-20T00:00:00Z"),
				avgDynamics: 0.3,
				avgTiming: 0.7,
				avgPedaling: 0.2,
				avgArticulation: 0.5,
				avgPhrasing: 0.6,
				avgInterpretation: 0.55,
			},
		];
		const { modelSummary } = buildProgressSummary("recent_sessions", data);
		expect(modelSummary).toContain("timing"); // strongest (0.91)
		expect(modelSummary).toContain("pedaling"); // weakest (~0.27)
		expect(modelSummary).not.toContain("0.91");
		expect(modelSummary).not.toContain(String(DISTINCT_A));
	});
});

describe("buildProgressSummary — session_detail", () => {
	it("identifies strongest and weakest as prose, no raw scores", () => {
		const data = {
			avgDynamics: DISTINCT_A, // strongest
			avgTiming: 0.6,
			avgPedaling: DISTINCT_B, // weakest
			avgArticulation: 0.5,
			avgPhrasing: 0.55,
			avgInterpretation: 0.58,
		};
		const { modelSummary } = buildProgressSummary("session_detail", data);
		expect(modelSummary).toContain("dynamics");
		expect(modelSummary).toContain("pedaling");
		expect(modelSummary).not.toContain(String(DISTINCT_A));
		expect(modelSummary).not.toContain(String(DISTINCT_B));
	});

	it("collapses to an even-work phrasing when scores are within noise (no false gap)", () => {
		const data = {
			avgDynamics: 0.5,
			avgTiming: 0.51,
			avgPedaling: 0.49,
			avgArticulation: 0.5,
			avgPhrasing: 0.5,
			avgInterpretation: 0.5,
		};
		const { modelSummary } = buildProgressSummary("session_detail", data);
		expect(modelSummary.toLowerCase()).toContain("even across dimensions");
		expect(modelSummary).not.toContain("room to grow");
	});
});

describe("buildProgressSummary — degenerate input", () => {
	it("handles null session_detail gracefully", () => {
		const { modelSummary } = buildProgressSummary("session_detail", null);
		expect(modelSummary.toLowerCase()).toContain("no session data");
	});

	it("handles empty dimension_history gracefully", () => {
		const { modelSummary } = buildProgressSummary("dimension_history", []);
		expect(modelSummary.length).toBeGreaterThan(0);
	});
});
