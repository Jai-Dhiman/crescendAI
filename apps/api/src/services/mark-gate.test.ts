import { describe, expect, it } from "vitest";
import { proposeMarks } from "./mark-gate";

describe("proposeMarks — cold start", () => {
	it("fires nothing when a dimension has fewer than 3 session samples", () => {
		const result = proposeMarks({
			atSeconds: 30,
			scores: {
				dynamics: 0.1,
				timing: 0.5,
				pedaling: 0.5,
				articulation: 0.5,
				phrasing: 0.5,
				interpretation: 0.5,
			},
			baseline: null,
			sessionSamples: { dynamics: [0.9, 0.1] },
			emittedDimensions: new Set(),
			firePersistence: 3,
		});
		expect(result.candidates).toEqual([]);
	});
});
