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

	it("reports a timestamp-anchored mark as unplaced rather than dropping it", () => {
		const timestampMark: Mark = {
			id: "stamp",
			anchor: resolveAnchor({
				atSeconds: 97,
				bars: [5, 6],
				alignmentQuality: 0.1,
			}),
			taxonomy: "needs_work",
			dimension: "timing",
			evidence: "e",
			lifecycle: "active",
		};
		const bars: BarLocator[] = [{ barNumber: 5, measureOn: "m-five" }];
		const rects = new Map<string, MeasureRect>([
			["m-five", { top: 100, left: 20, width: 50, height: 60 }],
		]);

		const { placed, unplaced } = placeMarks(bars, rects, [timestampMark]);

		// Bar 5 is right there with a rect — but resolveAnchor threw the bars
		// away, so there is nothing to place against and nothing to guess from.
		// The mark must still surface, or it vanishes from the product entirely.
		expect(placed).toHaveLength(0);
		expect(unplaced.map((m) => m.id)).toEqual(["stamp"]);
	});

	it("reports a bar that is not on the rendered page as unplaced", () => {
		const bars: BarLocator[] = [
			{ barNumber: 5, measureOn: "m-five" },
			{ barNumber: 88, measureOn: "m-eighty-eight" },
		];
		// Bar 88's element is not in the DOM — it is on another page.
		const rects = new Map<string, MeasureRect>([
			["m-five", { top: 100, left: 20, width: 50, height: 60 }],
		]);

		const { placed, unplaced } = placeMarks(bars, rects, [
			markAtBars("on-page", [5, 6]),
			markAtBars("off-page", [88, 89]),
			markAtBars("unknown-bar", [999, 999]),
		]);

		expect(placed.map((p) => p.mark.id)).toEqual(["on-page"]);
		expect(unplaced.map((m) => m.id)).toEqual(["off-page", "unknown-bar"]);
	});
});
