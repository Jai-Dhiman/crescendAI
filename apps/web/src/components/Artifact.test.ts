import { describe, expect, it } from "vitest";
import type { InlineComponent } from "../lib/types";

describe("getCollapsedProps", () => {
	it("returns dimension and bar range for score_highlight", async () => {
		const { getCollapsedProps } = await import("./Artifact");

		const component: InlineComponent = {
			type: "score_highlight",
			config: {
				pieceId: "123e4567-e89b-12d3-a456-426614174000",
				highlights: [
					{ bars: [4, 8] as [number, number], dimension: "dynamics", annotation: "crescendo" },
					{ bars: [12, 16] as [number, number], dimension: "pedaling" },
				],
			},
		};

		const props = getCollapsedProps(component);
		expect(props.title).toBe("Score Highlight");
		expect(props.subtitle).toContain("bars 4-8");
		expect(props.badge).toBe("2 regions");
	});

	it("returns singular badge for single region", async () => {
		const { getCollapsedProps } = await import("./Artifact");

		const component: InlineComponent = {
			type: "score_highlight",
			config: {
				pieceId: "123e4567-e89b-12d3-a456-426614174000",
				highlights: [
					{ bars: [1, 4] as [number, number], dimension: "timing" },
				],
			},
		};

		const props = getCollapsedProps(component);
		expect(props.badge).toBe("1 region");
	});
});
