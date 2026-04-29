import { describe, test, expect } from "vitest";
import { ARTIFACT_NAMES, artifactSchemas } from "./index";

describe("artifacts barrel", () => {
	test("ARTIFACT_NAMES contains exactly the three known names", () => {
		expect([...ARTIFACT_NAMES].sort()).toEqual([
			"DiagnosisArtifact",
			"ExerciseArtifact",
			"SynthesisArtifact",
		]);
	});

	test("artifactSchemas has a schema for every name in ARTIFACT_NAMES", () => {
		for (const name of ARTIFACT_NAMES) {
			expect(artifactSchemas[name]).toBeDefined();
			expect(typeof artifactSchemas[name].safeParse).toBe("function");
		}
	});

	test("artifactSchemas has no extra keys beyond ARTIFACT_NAMES", () => {
		expect(Object.keys(artifactSchemas).sort()).toEqual(
			[...ARTIFACT_NAMES].sort(),
		);
	});
});
