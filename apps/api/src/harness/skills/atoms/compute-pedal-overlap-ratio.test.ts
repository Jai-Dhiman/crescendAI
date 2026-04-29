import { test, expect } from "vitest";
import { computePedalOverlapRatio } from "./compute-pedal-overlap-ratio";

test("computePedalOverlapRatio: two notes, pedal [200,800]ms → ratio 0.6", async () => {
	// note0: onset=0ms, dur=1000ms → pedal active from 200-800 = 600ms of 1000ms
	// note1: onset=500ms, dur=500ms → pedal active from 500-800 = 300ms of 500ms
	// total pedaled = 600 + 300 = 900ms; total duration = 1000 + 500 = 1500ms
	// ratio = 900 / 1500 = 0.6
	const notes = [
		{ onset_ms: 0, duration_ms: 1000 },
		{ onset_ms: 500, duration_ms: 500 },
	];
	const pedal_cc = [
		{ time_ms: 200, value: 127 }, // pedal down
		{ time_ms: 800, value: 0 }, // pedal up
	];
	const result = await computePedalOverlapRatio.invoke({ notes, pedal_cc });
	expect(result).toBeCloseTo(0.6, 5);
});
