import { describe, expect, it } from "vitest";
import { proposeMarks } from "./mark-gate";

describe("proposeMarks — cold start", () => {
	it("fires nothing when a dimension has fewer than 3 session samples", () => {
		const result = proposeMarks({
			atSeconds: 30,
			scores: {
				dynamics: 0.1,
				timing: 0.5,
				pedaling: 0.5,
				articulation: 0.5,
				phrasing: 0.5,
				interpretation: 0.5,
			},
			baseline: null,
			sessionSamples: { dynamics: [0.9, 0.1] },
			emittedDimensions: new Set(),
			firePersistence: 3,
		});
		expect(result.candidates).toEqual([]);
	});
});

const BASELINE_DIM = {
	lifecycle: "active" as const,
	longMean: 0.6,
	longSd: 0.05,
	shortMean: 0.6,
	noiseFloor: 0.02,
	noiseFloorMeasured: true,
	halfWidth: 0.05,
	consecutiveOutOfBand: 0,
	consecutiveInBand: 0,
	promoted: false,
	evidenceWeeks: [],
	initialized: true,
	updateCount: 5,
	confidence: "provisional" as const,
};

function baselineWith(
	dim: string,
	overrides: Partial<typeof BASELINE_DIM> = {},
) {
	const absent = {
		...BASELINE_DIM,
		lifecycle: "absent" as const,
		initialized: false,
	};
	const dims = {
		dynamics: absent,
		timing: absent,
		pedaling: absent,
		articulation: absent,
		phrasing: absent,
		interpretation: absent,
	};
	dims[dim as keyof typeof dims] = { ...BASELINE_DIM, ...overrides };
	return { lastSessionTimestamp: null, dimensions: dims };
}

describe("proposeMarks — persisted baseline", () => {
	it("fires needs_work after 3 consecutive below-band samples", () => {
		const result = proposeMarks({
			atSeconds: 45,
			scores: {
				dynamics: 0.5,
				timing: 0.2,
				pedaling: 0.5,
				articulation: 0.5,
				phrasing: 0.5,
				interpretation: 0.5,
			},
			baseline: baselineWith("timing"),
			sessionSamples: { timing: [0.6, 0.2, 0.2, 0.2] },
			emittedDimensions: new Set(),
			firePersistence: 3,
		});
		expect(result.candidates).toEqual([
			{ dimension: "timing", taxonomy: "needs_work", atSeconds: 45 },
		]);
	});

	it("fires strong when the latest sample is above the band", () => {
		const result = proposeMarks({
			atSeconds: 45,
			scores: {
				dynamics: 0.5,
				timing: 0.5,
				pedaling: 0.5,
				articulation: 0.5,
				phrasing: 0.5,
				interpretation: 0.5,
			},
			baseline: baselineWith("dynamics"),
			sessionSamples: { dynamics: [0.6, 0.95, 0.95, 0.95] },
			emittedDimensions: new Set(),
			firePersistence: 3,
		});
		expect(result.candidates).toEqual([
			{ dimension: "dynamics", taxonomy: "strong", atSeconds: 45 },
		]);
	});

	it("never re-fires a dimension already in emittedDimensions", () => {
		const result = proposeMarks({
			atSeconds: 45,
			scores: {
				dynamics: 0.5,
				timing: 0.2,
				pedaling: 0.5,
				articulation: 0.5,
				phrasing: 0.5,
				interpretation: 0.5,
			},
			baseline: baselineWith("timing"),
			sessionSamples: { timing: [0.2, 0.2, 0.2, 0.2] },
			emittedDimensions: new Set(["timing"]),
			firePersistence: 3,
		});
		expect(result.candidates).toEqual([]);
	});

	it("yields at most one candidate when two dimensions qualify in the same call", () => {
		const dims = {
			...baselineWith("dynamics").dimensions,
			timing: { ...BASELINE_DIM },
		};
		const result = proposeMarks({
			atSeconds: 45,
			scores: {
				dynamics: 0.1,
				timing: 0.2,
				pedaling: 0.5,
				articulation: 0.5,
				phrasing: 0.5,
				interpretation: 0.5,
			},
			baseline: { lastSessionTimestamp: null, dimensions: dims },
			sessionSamples: {
				dynamics: [0.6, 0.1, 0.1, 0.1],
				timing: [0.6, 0.2, 0.2, 0.2],
			},
			emittedDimensions: new Set(),
			firePersistence: 3,
		});
		expect(result.candidates).toHaveLength(1);
		expect(result.candidates[0]?.dimension).toBe("dynamics"); // DIMS_6 order
	});
});
