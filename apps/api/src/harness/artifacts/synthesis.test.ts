import { describe, test, expect } from "vitest";
import { SynthesisArtifactSchema, type SynthesisArtifact } from "./synthesis";

const baseValid: SynthesisArtifact = {
	session_id: "sess:abc123",
	synthesis_scope: "session",
	strengths: [
		{
			dimension: "phrasing",
			one_liner: "Clean shape across the second theme.",
		},
	],
	focus_areas: [
		{
			dimension: "pedaling",
			one_liner: "Over-pedaling in slow passages.",
			severity: "moderate",
		},
	],
	proposed_exercises: ["ex:abc123"],
	dominant_dimension: "pedaling",
	recurring_pattern: null,
	next_session_focus: "Work on pedal-release timing in the slow movement.",
	diagnosis_refs: ["diag:abc123"],
	headline:
		"You played with real shape in the second theme today. The thing pulling the picture out of focus is the pedal in the slow passages — let's spend tomorrow on releasing it cleanly between phrases. " +
		"Your hands know what they want to do; the foot just needs to catch up. " +
		"Keep that musical instinct going into the next session and we will lock this down together.",
};

describe("SynthesisArtifactSchema", () => {
	test("accepts a fully valid session-scope artifact", () => {
		expect(() => SynthesisArtifactSchema.parse(baseValid)).not.toThrow();
	});

	test("rejects when synthesis_scope=weekly and recurring_pattern is null", () => {
		const invalid = {
			...baseValid,
			synthesis_scope: "weekly" as const,
			recurring_pattern: null,
		};
		const result = SynthesisArtifactSchema.safeParse(invalid);
		expect(result.success).toBe(false);
		expect(
			result.error?.issues.some((i) => i.message.includes("recurring_pattern")),
		).toBe(true);
	});

	test("accepts when synthesis_scope=weekly and recurring_pattern is populated", () => {
		const valid = {
			...baseValid,
			synthesis_scope: "weekly" as const,
			recurring_pattern: "Third session in a row over-pedaling slow movements.",
		};
		expect(() => SynthesisArtifactSchema.parse(valid)).not.toThrow();
	});

	test("rejects when synthesis_scope=piece_onboarding and any focus_area severity is not minor", () => {
		const invalid = {
			...baseValid,
			synthesis_scope: "piece_onboarding" as const,
			focus_areas: [
				{
					dimension: "pedaling" as const,
					one_liner: "x",
					severity: "moderate" as const,
				},
			],
		};
		const result = SynthesisArtifactSchema.safeParse(invalid);
		expect(result.success).toBe(false);
		expect(
			result.error?.issues.some((i) => i.message.includes("piece_onboarding")),
		).toBe(true);
	});

	test("accepts when synthesis_scope=piece_onboarding and all focus_areas are minor", () => {
		const valid = {
			...baseValid,
			synthesis_scope: "piece_onboarding" as const,
			focus_areas: [
				{
					dimension: "pedaling" as const,
					one_liner: "x",
					severity: "minor" as const,
				},
			],
		};
		expect(() => SynthesisArtifactSchema.parse(valid)).not.toThrow();
	});

	test("rejects when strengths exceeds 2 items", () => {
		const invalid = {
			...baseValid,
			strengths: [
				{ dimension: "phrasing" as const, one_liner: "a" },
				{ dimension: "timing" as const, one_liner: "b" },
				{ dimension: "pedaling" as const, one_liner: "c" },
			],
		};
		expect(SynthesisArtifactSchema.safeParse(invalid).success).toBe(false);
	});

	test("rejects when focus_areas exceeds 3 items", () => {
		const invalid = {
			...baseValid,
			focus_areas: [
				{
					dimension: "pedaling" as const,
					one_liner: "a",
					severity: "moderate" as const,
				},
				{
					dimension: "timing" as const,
					one_liner: "b",
					severity: "moderate" as const,
				},
				{
					dimension: "phrasing" as const,
					one_liner: "c",
					severity: "moderate" as const,
				},
				{
					dimension: "dynamics" as const,
					one_liner: "d",
					severity: "moderate" as const,
				},
			],
		};
		expect(SynthesisArtifactSchema.safeParse(invalid).success).toBe(false);
	});

	test("rejects when proposed_exercises exceeds 3 items", () => {
		const invalid = { ...baseValid, proposed_exercises: ["a", "b", "c", "d"] };
		expect(SynthesisArtifactSchema.safeParse(invalid).success).toBe(false);
	});

	test("rejects when headline is shorter than 300 chars", () => {
		const invalid = { ...baseValid, headline: "x".repeat(299) };
		expect(SynthesisArtifactSchema.safeParse(invalid).success).toBe(false);
	});

	test("rejects when headline exceeds 500 chars", () => {
		const invalid = { ...baseValid, headline: "x".repeat(501) };
		expect(SynthesisArtifactSchema.safeParse(invalid).success).toBe(false);
	});

	test("rejects when next_session_focus exceeds 200 chars", () => {
		const invalid = { ...baseValid, next_session_focus: "x".repeat(201) };
		expect(SynthesisArtifactSchema.safeParse(invalid).success).toBe(false);
	});

	test("accepts next_session_focus as null", () => {
		const valid = { ...baseValid, next_session_focus: null };
		expect(() => SynthesisArtifactSchema.parse(valid)).not.toThrow();
	});

	test("accepts empty strengths array", () => {
		const valid = { ...baseValid, strengths: [] };
		expect(() => SynthesisArtifactSchema.parse(valid)).not.toThrow();
	});
});
