import { describe, expect, test } from "vitest";
import { ExerciseSchema, exerciseDedupKey } from "./exercise";

describe("ExerciseSchema", () => {
	test("parses a valid Exercise row", () => {
		const result = ExerciseSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			title: "Slow-hands voicing on bar 12-16",
			description: "Right-hand melody isolation",
			instructions: "Play right hand only at half tempo, exaggerate top voice",
			difficulty: "intermediate",
			category: "voicing",
			repertoireTags: null,
			notationContent: null,
			notationFormat: null,
			midiContent: null,
			source: "molecule:voicing-diagnosis",
			variantsJson: null,
			createdAt: "2026-04-01T00:00:00.000Z",
		});
		expect(result.success).toBe(true);
	});

	test("fails parse when title is empty", () => {
		const result = ExerciseSchema.safeParse({
			id: "11111111-2222-3333-4444-555555555555",
			title: "",
			description: "desc",
			instructions: "ins",
			difficulty: "intermediate",
			category: "voicing",
			source: "molecule:voicing-diagnosis",
			createdAt: "2026-04-01T00:00:00.000Z",
		});
		expect(result.success).toBe(false);
	});
});

describe("exerciseDedupKey", () => {
	test("returns same key for whitespace and case variants", () => {
		const a = exerciseDedupKey({
			title: "Slow-Hands Voicing",
			source: "Molecule:Voicing-Diagnosis",
		});
		const b = exerciseDedupKey({
			title: "  slow-hands voicing  ",
			source: "molecule:voicing-diagnosis",
		});
		expect(a).toBe(b);
	});

	test("returns different keys for different titles", () => {
		const a = exerciseDedupKey({ title: "A", source: "x" });
		const b = exerciseDedupKey({ title: "B", source: "x" });
		expect(a).not.toBe(b);
	});
});
