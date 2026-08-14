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
});
