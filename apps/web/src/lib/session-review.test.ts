import { describe, expect, it } from "vitest";
import { formatWsGap } from "./session-review";

describe("formatWsGap", () => {
	it("discloses a multi-minute gap in plural and a one-minute gap in singular", () => {
		expect(formatWsGap({ startSeconds: 0, endSeconds: 240 })).toBe(
			"4 minutes didn't reach me",
		);
		expect(formatWsGap({ startSeconds: 100, endSeconds: 160 })).toBe(
			"1 minute didn't reach me",
		);
	});

	it("never renders a zero-minute gap for a sub-minute gap", () => {
		expect(formatWsGap({ startSeconds: 0, endSeconds: 20 })).toBe(
			"under a minute didn't reach me",
		);
	});
});
