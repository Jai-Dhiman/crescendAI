import { describe, expect, test } from "vitest";
import { factSchema } from "./fact";

const validFact = {
	id: "11111111-2222-3333-4444-555555555555",
	studentId: "apple:user:abc",
	factText: "Student recurrently over-pedals in slow movements.",
	assertionType: "recurring_issue",
	dimension: "pedaling",
	validAt: "2026-04-01T00:00:00.000Z",
	invalidAt: null,
	entityMentions: [{ kind: "student", studentId: "apple:user:abc" }],
	evidence: [
		{
			kind: "observation",
			observation_id: "22222222-3333-4444-5555-666666666666",
		},
	],
	trend: "stable",
	confidence: "high",
	sourceType: "synthesized",
	createdAt: "2026-04-15T00:00:00.000Z",
	expiredAt: null,
};

describe("factSchema", () => {
	test("parses a valid Fact", () => {
		const result = factSchema.safeParse(validFact);
		expect(result.success).toBe(true);
	});

	test("fails when entityMentions is empty", () => {
		const result = factSchema.safeParse({ ...validFact, entityMentions: [] });
		expect(result.success).toBe(false);
	});

	test("fails when evidence is empty", () => {
		const result = factSchema.safeParse({ ...validFact, evidence: [] });
		expect(result.success).toBe(false);
	});

	test("fails when invalidAt is before validAt", () => {
		const result = factSchema.safeParse({
			...validFact,
			invalidAt: "2026-03-01T00:00:00.000Z",
		});
		expect(result.success).toBe(false);
	});

	test("fails when expiredAt is before createdAt", () => {
		const result = factSchema.safeParse({
			...validFact,
			expiredAt: "2026-03-01T00:00:00.000Z",
		});
		expect(result.success).toBe(false);
	});
});
