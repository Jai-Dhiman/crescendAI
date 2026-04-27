import { describe, expect, test } from "vitest";
import { ArtifactRowSchema } from "./artifact";

describe("ArtifactRowSchema", () => {
	test("parses a valid ArtifactRow with opaque payload", () => {
		const result = ArtifactRowSchema.safeParse({
			artifact_id: "11111111-2222-3333-4444-555555555555",
			schema_name: "DiagnosisArtifact",
			schema_version: 1,
			producer: "molecule:voicing-diagnosis",
			created_at: "2026-04-25T10:05:00.000Z",
			payload: { anything: "v5 owns this" },
		});
		expect(result.success).toBe(true);
	});

	test("fails parse when schema_name is missing", () => {
		const result = ArtifactRowSchema.safeParse({
			artifact_id: "11111111-2222-3333-4444-555555555555",
			schema_version: 1,
			producer: "molecule:voicing-diagnosis",
			created_at: "2026-04-25T10:05:00.000Z",
			payload: {},
		});
		expect(result.success).toBe(false);
	});
});
