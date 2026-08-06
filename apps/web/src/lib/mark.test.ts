import { describe, expect, it } from "vitest";
import { ALIGNMENT_MIN, resolveAnchor } from "./mark";

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
});
