import { describe, expect, it } from "vitest";
import type { Mark } from "./mark";
import { resolveAnchor } from "./mark";
import type { BarLocator, MeasureRect } from "./mark-placement";
import { GLYPH_OFFSET_PX, placeMarks } from "./mark-placement";

function markAtBars(id: string, bars: readonly [number, number]): Mark {
	return {
		id,
		anchor: resolveAnchor({ atSeconds: 30, bars, alignmentQuality: 1 }),
		taxonomy: "needs_work",
		dimension: "timing",
		evidence: "e",
		lifecycle: "active",
	};
}

describe("placeMarks", () => {
	it("resolves a bar through its measureOn id, not its array position", () => {
		// Bar 7 sits at array index 0 and bar 3 at index 1. An index-based
		// implementation would place bar 7 at bar 3's rect, or miss entirely.
		const bars: BarLocator[] = [
			{ barNumber: 7, measureOn: "m-seven" },
			{ barNumber: 3, measureOn: "m-three" },
		];
		const rects = new Map<string, MeasureRect>([
			["m-seven", { top: 200, left: 400, width: 50, height: 60 }],
			["m-three", { top: 100, left: 20, width: 50, height: 60 }],
		]);

		const { placed, unplaced } = placeMarks(bars, rects, [
			markAtBars("a", [7, 7]),
		]);

		expect(unplaced).toHaveLength(0);
		expect(placed).toHaveLength(1);
		expect(placed[0].left).toBe(400);
		expect(placed[0].top).toBe(200 - GLYPH_OFFSET_PX);
	});
});
