import { describe, expect, test } from "vitest";
import { contentSchemas, evidenceRefSchema } from "./index";

describe("evidenceRefSchema", () => {
	test.each([
		[
			"signal",
			{
				kind: "signal",
				chunk_id: "chunk-abc",
				schema_name: "MuQQuality",
				row_id: "row-1",
			},
		],
		[
			"observation",
			{
				kind: "observation",
				observation_id: "11111111-2222-3333-4444-555555555555",
			},
		],
		[
			"artifact",
			{
				kind: "artifact",
				artifact_id: "11111111-2222-3333-4444-555555555555",
				schema_name: "DiagnosisArtifact",
			},
		],
	])("parses a valid EvidenceRef of kind %s", (_label, ref) => {
		const result = evidenceRefSchema.safeParse(ref);
		expect(result.success).toBe(true);
	});

	test("fails parse for an unknown kind", () => {
		const result = evidenceRefSchema.safeParse({ kind: "alien", x: 1 });
		expect(result.success).toBe(false);
	});
});

describe("contentSchemas registry", () => {
	test("has exactly the six expected keys", () => {
		expect(Object.keys(contentSchemas).sort()).toEqual([
			"AMTTranscription",
			"ArtifactRow",
			"MuQQuality",
			"Observation",
			"ScoreAlignment",
			"StopMoment",
		]);
	});
});
