import { describe, expect, it } from "vitest";
import { ALIGNMENT_MIN, anchorLabel, resolveAnchor } from "./mark";

describe("resolveAnchor", () => {
	it("discards bars when alignment quality is below the threshold", () => {
		const anchor = resolveAnchor({
			atSeconds: 97,
			bars: [21, 22],
			alignmentQuality: ALIGNMENT_MIN - 0.01,
		});

		expect(anchor.type).toBe("timestamp");
		expect(anchor).not.toHaveProperty("bars");
		expect(anchor.atSeconds).toBe(97);
	});

	it("keeps bars at exactly the threshold and still carries elapsed time", () => {
		const anchor = resolveAnchor({
			atSeconds: 64,
			bars: [5, 6],
			alignmentQuality: ALIGNMENT_MIN,
		});

		expect(anchor.type).toBe("bars");
		if (anchor.type !== "bars") throw new Error("unreachable");
		expect(anchor.bars).toEqual([5, 6]);
		expect(anchor.atSeconds).toBe(64);
	});

	it("returns a timestamp anchor when no bars are supplied at all", () => {
		const anchor = resolveAnchor({ atSeconds: 12, alignmentQuality: 1 });

		expect(anchor.type).toBe("timestamp");
		expect(anchor.atSeconds).toBe(12);
	});
});

describe("anchorLabel", () => {
	it("names a range, a single bar, and a timestamp in the student's words", () => {
		const range = resolveAnchor({
			atSeconds: 64,
			bars: [5, 6],
			alignmentQuality: 1,
		});
		const single = resolveAnchor({
			atSeconds: 151,
			bars: [12, 12],
			alignmentQuality: 1,
		});
		const stamp = resolveAnchor({ atSeconds: 97, alignmentQuality: 1 });

		expect(anchorLabel(range)).toBe("bars 5-6");
		expect(anchorLabel(single)).toBe("bar 12");
		expect(anchorLabel(stamp)).toBe("1:37");
	});

	it("zero-pads seconds under ten", () => {
		const stamp = resolveAnchor({ atSeconds: 305, alignmentQuality: 0 });
		expect(anchorLabel(stamp)).toBe("5:05");
	});
});
