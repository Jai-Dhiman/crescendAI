import { describe, expect, it } from "vitest";
import type { LaneItem } from "./timeline-lanes";
import { assignLanes, clampToStrip } from "./timeline-lanes";

describe("clampToStrip", () => {
	it("pulls a mark that would overrun the right edge back inside", () => {
		// Measured from the real preview: a mark at 84.7% of the session sat at
		// left 603 with a 153px glyph on a 720px strip, ending 36px outside.
		expect(clampToStrip(603, 153, 720)).toBe(720 - 153);
	});

	it("leaves a mark that already fits exactly where time put it", () => {
		expect(clampToStrip(100, 150, 720)).toBe(100);
	});

	it("never pushes a mark off the left edge to make it fit", () => {
		// A glyph wider than its strip cannot fit; flush-left is the only
		// position where any of it is readable.
		expect(clampToStrip(50, 900, 720)).toBe(0);
	});

	it("leaves positions untouched before the strip has been measured", () => {
		expect(clampToStrip(603, 153, 0)).toBe(603);
	});
});

describe("assignLanes", () => {
	it("keeps marks that would overlap out of each other's lane", () => {
		// Measured from the real preview at 1024px: these two genuinely collided
		// by 84px, which made the covered mark unclickable in a real browser.
		const items: LaneItem[] = [
			{ id: "pedaling", left: 368, width: 150 },
			{ id: "timing", left: 434, width: 115 },
		];

		const lanes = assignLanes(items);

		expect(lanes.get("pedaling")).toBe(0);
		expect(lanes.get("timing")).not.toBe(0);
	});

	it("reuses a lane once the horizontal gap is clear", () => {
		const items: LaneItem[] = [
			{ id: "a", left: 0, width: 100 },
			{ id: "b", left: 50, width: 100 },
			{ id: "c", left: 400, width: 100 },
		];

		const lanes = assignLanes(items);

		expect(lanes.get("a")).toBe(0);
		expect(lanes.get("b")).toBe(1);
		// `c` clears `a` entirely, so it belongs back on the first lane rather
		// than stacking forever.
		expect(lanes.get("c")).toBe(0);
	});

	it("orders by position, not by input order", () => {
		const items: LaneItem[] = [
			{ id: "late", left: 400, width: 100 },
			{ id: "early", left: 0, width: 100 },
		];

		const lanes = assignLanes(items);

		expect(lanes.get("early")).toBe(0);
		expect(lanes.get("late")).toBe(0);
	});
});
