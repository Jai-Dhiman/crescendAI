import { describe, expect, it } from "vitest";
import type { InlineComponent } from "../lib/types";
import { getCollapsedProps } from "./Artifact";

describe("getCollapsedProps", () => {
	it("returns title and subtitle for play_passage", () => {
		const component: InlineComponent = {
			type: "play_passage",
			config: {
				sessionId: "00000000-0000-0000-0000-0000000000aa",
				bars: [5, 8],
				dimension: "timing",
				annotation: "you rushed here",
			},
		};
		const result = getCollapsedProps(component);
		expect(result.title).toBe("Play Passage");
		expect(result.subtitle).toBe("bars 5-8, timing");
		expect(result.badge).toBe("");
	});
});
