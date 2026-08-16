import { describe, expect, it } from "vitest";
import { extractMpmFeatures } from "./mpm-features";

describe("extractMpmFeatures", () => {
	it("returns one zeroed feature set for fewer than 2 notes", () => {
		const result = extractMpmFeatures({
			windowIndex: 3,
			perfNotes: [{ pitch: 60, onset: 0, offset: 0.5, velocity: 80 }],
			perfPedal: [],
		});
		expect(result).toHaveLength(1);
		expect(result[0]).toEqual({
			windowIndex: 3,
			localTempoBpm: 0,
			tempoDeviationPct: 0,
			velocityMean: 0,
			velocityRange: 0,
			velocitySlope: 0,
			articulationRatio: 0,
			pedalOverlapFraction: 0,
		});
	});
});
