import { test, expect } from "vitest";
import { prioritizeDiagnoses } from "./prioritize-diagnoses";
import type { DiagnosisArtifact } from "../../artifacts/diagnosis";

const makeD = (overrides: Partial<DiagnosisArtifact>): DiagnosisArtifact => ({
	primary_dimension: "dynamics",
	dimensions: ["dynamics"],
	severity: "minor",
	scope: "session",
	bar_range: null,
	evidence_refs: ["ref:1"],
	one_sentence_finding: "test",
	confidence: "medium",
	finding_type: "issue",
	...overrides,
});

test("prioritizeDiagnoses: significant/medium/articulation > moderate/high/timing > minor/high/pedaling-strength", async () => {
	const diagA = makeD({
		primary_dimension: "timing",
		dimensions: ["timing"],
		severity: "moderate",
		confidence: "high",
		finding_type: "issue",
	});
	const diagB = makeD({
		primary_dimension: "articulation",
		dimensions: ["articulation"],
		severity: "significant",
		confidence: "medium",
		finding_type: "issue",
	});
	const diagC = makeD({
		primary_dimension: "pedaling",
		dimensions: ["pedaling"],
		severity: "minor",
		confidence: "high",
		finding_type: "strength",
	});

	const result = (await prioritizeDiagnoses.invoke({
		diagnoses: [diagA, diagB, diagC],
	})) as DiagnosisArtifact[];
	expect(result).toHaveLength(3);
	expect(result[0]).toEqual(diagB); // significant first
	expect(result[1]).toEqual(diagA); // moderate second
	expect(result[2]).toEqual(diagC); // strength always last
});
