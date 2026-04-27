import { describe, expect, test } from "vitest";
import { entityRefSchema, entityRefSchemas } from "./index";

describe("entityRefSchema", () => {
	test.each([
		["student", { kind: "student", studentId: "apple:user:abc" }],
		["piece", { kind: "piece", pieceId: "chopin.etudes_op_25.1" }],
		[
			"movement",
			{ kind: "movement", pieceId: "chopin.etudes_op_25.1", movementIndex: 0 },
		],
		[
			"bar",
			{
				kind: "bar",
				pieceId: "chopin.etudes_op_25.1",
				movementIndex: 0,
				barNumber: 47,
			},
		],
		[
			"session",
			{ kind: "session", sessionId: "11111111-2222-3333-4444-555555555555" },
		],
		[
			"exercise",
			{ kind: "exercise", exerciseId: "11111111-2222-3333-4444-555555555555" },
		],
	])("parses a valid EntityRef of kind %s", (_label, ref) => {
		const result = entityRefSchema.safeParse(ref);
		expect(result.success).toBe(true);
	});

	test("fails parse for an unknown kind", () => {
		const result = entityRefSchema.safeParse({ kind: "alien", x: 1 });
		expect(result.success).toBe(false);
	});
});

describe("entityRefSchemas registry", () => {
	test("has exactly the six expected keys", () => {
		expect(Object.keys(entityRefSchemas).sort()).toEqual([
			"bar",
			"exercise",
			"movement",
			"piece",
			"session",
			"student",
		]);
	});
});
