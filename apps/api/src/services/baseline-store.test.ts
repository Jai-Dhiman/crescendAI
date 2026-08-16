import { describe, expect, it } from "vitest";
import { loadBaselineState } from "./baseline-store";

describe("loadBaselineState", () => {
	it("returns null when the student has no row", async () => {
		const fakeDb = {
			query: { studentProfiles: { findFirst: async () => undefined } },
		};
		const result = await loadBaselineState(fakeDb as never, "stu-1");
		expect(result).toBeNull();
	});

	it("returns null when the row exists but baseline_state is null", async () => {
		const fakeDb = {
			query: {
				studentProfiles: {
					findFirst: async () => ({ studentId: "stu-1", baselineState: null }),
				},
			},
		};
		const result = await loadBaselineState(fakeDb as never, "stu-1");
		expect(result).toBeNull();
	});
});
