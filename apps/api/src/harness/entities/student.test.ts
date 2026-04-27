import { describe, expect, test } from "vitest";
import { StudentSchema, resolveStudent } from "./student";

describe("StudentSchema", () => {
	test("parses a valid Student row", () => {
		const result = StudentSchema.safeParse({
			studentId: "apple:user:abc123",
			inferredLevel: "intermediate",
			baselineDynamics: 0.6,
			baselineTiming: 0.55,
			baselinePedaling: 0.5,
			baselineArticulation: 0.62,
			baselinePhrasing: 0.58,
			baselineInterpretation: 0.6,
			baselineSessionCount: 12,
			explicitGoals: null,
			createdAt: "2026-04-01T00:00:00.000Z",
			updatedAt: "2026-04-25T00:00:00.000Z",
		});
		expect(result.success).toBe(true);
	});

	test("fails parse when studentId is missing", () => {
		const result = StudentSchema.safeParse({
			inferredLevel: "intermediate",
			baselineSessionCount: 0,
			createdAt: "2026-04-01T00:00:00.000Z",
			updatedAt: "2026-04-25T00:00:00.000Z",
		});
		expect(result.success).toBe(false);
	});
});

describe("resolveStudent", () => {
	test("canonicalizes appleUserId to studentId", () => {
		expect(resolveStudent({ appleUserId: "apple:user:abc123" })).toEqual({
			studentId: "apple:user:abc123",
		});
	});
});
