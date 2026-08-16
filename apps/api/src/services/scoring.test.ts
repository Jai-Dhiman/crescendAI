import { describe, expect, it } from "vitest";
import { scoreChunk } from "./scoring";

describe("scoreChunk", () => {
	it("returns neutral 0.5 for every dimension when there are no notes", () => {
		const result = scoreChunk([], []);
		expect(result).toEqual({
			dynamics: 0.5,
			timing: 0.5,
			pedaling: 0.5,
			articulation: 0.5,
			phrasing: 0.5,
			interpretation: 0.5,
		});
	});
});
