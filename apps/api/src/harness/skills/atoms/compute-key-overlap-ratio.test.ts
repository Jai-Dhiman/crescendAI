import { test, expect } from "vitest";
import { computeKeyOverlapRatio } from "./compute-key-overlap-ratio";

test("computeKeyOverlapRatio: three notes with 50ms overlap each gives ratio 0.10", async () => {
	// note0: onset=0, dur=500 -> off at 500
	// note1: onset=450, dur=500 -> off at 950
	// note2: onset=950, dur=500
	// pair0: overlap = (0+500) - 450 = 50; pair_ratio = 50/500 = 0.10
	// pair1: overlap = (450+500) - 950 = 0; pair_ratio = 0/500 = 0.00
	// mean = (0.10 + 0.00) / 2 = 0.05
	const notes = [
		{ onset_ms: 0, duration_ms: 500 },
		{ onset_ms: 450, duration_ms: 500 },
		{ onset_ms: 950, duration_ms: 500 },
	];
	const result = await computeKeyOverlapRatio.invoke({ notes });
	expect(result).toBeCloseTo(0.05, 5);
});
