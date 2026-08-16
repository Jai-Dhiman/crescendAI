import { describe, expect, it } from "vitest";
import { scoreChunk } from "./scoring";

describe("scoreChunk", () => {
	it("returns neutral 0.5 for every dimension when there are no notes", () => {
		const result = scoreChunk([], []);
		expect(result).toEqual({
			dynamics: 0.5,
			timing: 0.5,
			pedaling: 0.5,
			articulation: 0.5,
			phrasing: 0.5,
			interpretation: 0.5,
		});
	});

	it("is deterministic across repeated calls with the same input", () => {
		const notes = [
			{ pitch: 60, onset: 0, offset: 0.4, velocity: 70 },
			{ pitch: 62, onset: 0.5, offset: 0.9, velocity: 80 },
			{ pitch: 64, onset: 1.0, offset: 1.4, velocity: 60 },
		];
		const pedal = [{ time: 0.2, value: 100 }];
		const first = scoreChunk(notes, pedal);
		const second = scoreChunk(notes, pedal);
		expect(second).toEqual(first);
	});

	it("keeps every dimension within [0, 1]", () => {
		const notes = [
			{ pitch: 60, onset: 0, offset: 0.05, velocity: 127 },
			{ pitch: 62, onset: 0.01, offset: 5, velocity: 1 },
		];
		const result = scoreChunk(notes, []);
		for (const value of Object.values(result)) {
			expect(value).toBeGreaterThanOrEqual(0);
			expect(value).toBeLessThanOrEqual(1);
		}
	});

	it("higher mean velocity produces a higher dynamics score", () => {
		const soft = [
			{ pitch: 60, onset: 0, offset: 0.4, velocity: 20 },
			{ pitch: 62, onset: 0.5, offset: 0.9, velocity: 25 },
		];
		const loud = [
			{ pitch: 60, onset: 0, offset: 0.4, velocity: 110 },
			{ pitch: 62, onset: 0.5, offset: 0.9, velocity: 115 },
		];
		expect(scoreChunk(loud, []).dynamics).toBeGreaterThan(
			scoreChunk(soft, []).dynamics,
		);
	});
});
