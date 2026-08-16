import { describe, expect, it } from "vitest";
import { loadBaselineState, saveBaselineState } from "./baseline-store";
import { initialBaselineState } from "./student-baseline";

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

	it("throws ValidationError when the stored shape is corrupt", async () => {
		const fakeDb = {
			query: {
				studentProfiles: {
					findFirst: async () => ({
						studentId: "stu-1",
						baselineState: { not: "a baseline state" },
					}),
				},
			},
		};
		await expect(loadBaselineState(fakeDb as never, "stu-1")).rejects.toThrow(
			/corrupt baseline_state/,
		);
	});
});

describe("saveBaselineState", () => {
	it("inserts studentId and baselineState via onConflictDoUpdate", async () => {
		const state = initialBaselineState();
		const calls: { values?: unknown; conflictSet?: unknown } = {};
		const fakeDb = {
			insert: () => ({
				values: (v: unknown) => {
					calls.values = v;
					return {
						onConflictDoUpdate: (opts: { set: unknown }) => {
							calls.conflictSet = opts.set;
							return Promise.resolve();
						},
					};
				},
			}),
		};
		await saveBaselineState(fakeDb as never, "stu-1", state);
		expect(calls.values).toMatchObject({
			studentId: "stu-1",
			baselineState: state,
		});
		expect(calls.conflictSet).toMatchObject({ baselineState: state });
	});
});
