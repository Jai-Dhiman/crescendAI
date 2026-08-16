import { describe, expect, it } from "vitest";
import { evidenceFor, extractMpmFeatures } from "./mpm-features";

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

describe("extractMpmFeatures — full computation", () => {
	it("computes local tempo and a positive tempoDeviationPct when the second half speeds up", () => {
		// IOIs: 1.0, 1.0 (first half, slow) then 0.5, 0.5 (second half, fast).
		const perfNotes = [
			{ pitch: 60, onset: 0, offset: 0.4, velocity: 60 },
			{ pitch: 60, onset: 1.0, offset: 1.4, velocity: 60 },
			{ pitch: 60, onset: 2.0, offset: 2.4, velocity: 60 },
			{ pitch: 60, onset: 2.5, offset: 2.9, velocity: 60 },
			{ pitch: 60, onset: 3.0, offset: 3.4, velocity: 60 },
		];
		const [set] = extractMpmFeatures({
			windowIndex: 0,
			perfNotes,
			perfPedal: [],
		});
		expect(set?.localTempoBpm).toBeGreaterThan(0);
		expect(set?.tempoDeviationPct).toBeGreaterThan(0); // sped up
	});

	it("computes velocityMean, velocityRange, and pedalOverlapFraction", () => {
		const perfNotes = [
			{ pitch: 60, onset: 0, offset: 0.4, velocity: 40 },
			{ pitch: 62, onset: 0.5, offset: 0.9, velocity: 100 },
		];
		const perfPedal = [
			{ time: 0.1, value: 100 },
			{ time: 0.6, value: 0 },
		];
		const [set] = extractMpmFeatures({ windowIndex: 0, perfNotes, perfPedal });
		expect(set?.velocityMean).toBe(70);
		expect(set?.velocityRange).toBe(60);
		expect(set?.pedalOverlapFraction).toBe(0.5);
	});

	it("evidenceFor(timing) matches the direction and magnitude of tempoDeviationPct", () => {
		const set = {
			windowIndex: 0,
			localTempoBpm: 120,
			tempoDeviationPct: 9.4,
			velocityMean: 70,
			velocityRange: 20,
			velocitySlope: 3,
			articulationRatio: 0.5,
			pedalOverlapFraction: 0.5,
		};
		expect(evidenceFor(set, "timing")).toBe(
			"local tempo up 9%, uneven note lengths",
		);
	});

	it("evidenceFor covers every dimension without throwing", () => {
		const set = {
			windowIndex: 0,
			localTempoBpm: 120,
			tempoDeviationPct: -5,
			velocityMean: 70,
			velocityRange: 20,
			velocitySlope: -3,
			articulationRatio: 0.5,
			pedalOverlapFraction: 0.5,
		};
		for (const dim of [
			"dynamics",
			"timing",
			"pedaling",
			"articulation",
			"phrasing",
			"interpretation",
		] as const) {
			expect(typeof evidenceFor(set, dim)).toBe("string");
			expect(evidenceFor(set, dim).length).toBeGreaterThan(0);
		}
	});
});
