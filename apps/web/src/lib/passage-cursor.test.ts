// apps/web/src/lib/passage-cursor.test.ts
import { describe, expect, it } from "vitest";
import { createPassageQstampSource } from "./passage-cursor";
import type { BarIR } from "./score-ir";

const BAR_TIMELINE = [
	{ bar: 5, tSec: 10 },
	{ bar: 6, tSec: 12 },
];

const BARS: BarIR[] = [
	{
		barNumber: 5,
		measureOn: "m5",
		pageN: 1,
		bbox: { x: 0, y: 0, w: 0, h: 0 },
		noteIds: [],
		qstampStart: 16,
		qstampEnd: 20,
	},
];

describe("createPassageQstampSource", () => {
	it("interpolates qstamp within the bracketing bar, and degrades to null outside known playback", () => {
		const midway = createPassageQstampSource(BAR_TIMELINE, BARS, () => 11);
		expect(midway()).toBe(18); // halfway through tSec [10,12] -> halfway through qstamp [16,20]

		const notStarted = createPassageQstampSource(
			BAR_TIMELINE,
			BARS,
			() => null,
		);
		expect(notStarted()).toBeNull();

		const pastKnownRange = createPassageQstampSource(
			BAR_TIMELINE,
			BARS,
			() => 99,
		);
		expect(pastKnownRange()).toBeNull();
	});
});
