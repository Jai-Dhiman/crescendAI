import { describe, test, expect } from "vitest";
import { ExerciseArtifactSchema, type ExerciseArtifact } from "./exercise";

const baseValid: ExerciseArtifact = {
	diagnosis_ref: "diag:abc123",
	diagnosis_summary: "Over-pedaled in slow passage at bars 12-16.",
	target_dimension: "pedaling",
	exercise_type: "pedal_isolation",
	exercise_subtype: null,
	bar_range: [12, 16],
	instruction:
		"Play bars 12-16 with no pedal at all. Listen for sustain in the line itself.",
	success_criterion: "Three consecutive clean repetitions with no pedal.",
	estimated_minutes: 5,
	action_binding: { tool: "mute_pedal", args: { bars: [12, 16] } },
};

describe("ExerciseArtifactSchema", () => {
	test("accepts a fully valid baseline artifact", () => {
		expect(() => ExerciseArtifactSchema.parse(baseValid)).not.toThrow();
	});

	test("rejects when instruction exceeds 400 chars", () => {
		const invalid = { ...baseValid, instruction: "x".repeat(401) };
		expect(ExerciseArtifactSchema.safeParse(invalid).success).toBe(false);
	});

	test("rejects when success_criterion exceeds 200 chars", () => {
		const invalid = { ...baseValid, success_criterion: "x".repeat(201) };
		expect(ExerciseArtifactSchema.safeParse(invalid).success).toBe(false);
	});

	test("rejects when estimated_minutes is outside 1-15", () => {
		expect(
			ExerciseArtifactSchema.safeParse({ ...baseValid, estimated_minutes: 0 })
				.success,
		).toBe(false);
		expect(
			ExerciseArtifactSchema.safeParse({ ...baseValid, estimated_minutes: 16 })
				.success,
		).toBe(false);
	});

	test("rejects when action_binding is null for pedal_isolation", () => {
		const invalid = { ...baseValid, action_binding: null };
		const result = ExerciseArtifactSchema.safeParse(invalid);
		expect(result.success).toBe(false);
		expect(
			result.error?.issues.some((i) => i.message.includes("action_binding")),
		).toBe(true);
	});

	test("rejects when action_binding is null for segment_loop", () => {
		const invalid = {
			...baseValid,
			exercise_type: "segment_loop" as const,
			action_binding: null,
		};
		expect(ExerciseArtifactSchema.safeParse(invalid).success).toBe(false);
	});

	test("rejects when action_binding is null for isolated_hands", () => {
		const invalid = {
			...baseValid,
			exercise_type: "isolated_hands" as const,
			action_binding: null,
		};
		expect(ExerciseArtifactSchema.safeParse(invalid).success).toBe(false);
	});

	test("accepts action_binding null for slow_practice (verbal-only type)", () => {
		const valid = {
			...baseValid,
			exercise_type: "slow_practice" as const,
			action_binding: null,
		};
		expect(() => ExerciseArtifactSchema.parse(valid)).not.toThrow();
	});

	test("rejects when bar_range start > end", () => {
		const invalid = { ...baseValid, bar_range: [16, 12] as [number, number] };
		expect(ExerciseArtifactSchema.safeParse(invalid).success).toBe(false);
	});

	test("accepts a non-null exercise_subtype", () => {
		const valid = { ...baseValid, exercise_subtype: "half-pedal-only" };
		expect(() => ExerciseArtifactSchema.parse(valid)).not.toThrow();
	});

	test("rejects when bar_range is null (exercises always require a specific bar range)", () => {
		const invalid = { ...baseValid, bar_range: null };
		expect(ExerciseArtifactSchema.safeParse(invalid).success).toBe(false);
	});
});
