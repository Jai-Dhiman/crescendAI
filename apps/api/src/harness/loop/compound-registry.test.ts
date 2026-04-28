import { describe, expect, it } from "vitest";
import { getCompoundBinding } from "./compound-registry";
import { SynthesisArtifactSchema } from "../artifacts/synthesis";

describe("compound-registry", () => {
	it("returns a binding for OnSessionEnd with all 15 atoms registered", () => {
		const binding = getCompoundBinding("OnSessionEnd");
		expect(binding).toBeDefined();
		expect(binding?.compoundName).toBe("session-synthesis");
		expect(binding?.artifactSchema).toBe(SynthesisArtifactSchema);
		expect(binding?.artifactToolName).toBe("write_synthesis_artifact");
		expect(binding?.tools).toHaveLength(15);
		const names = new Set(binding?.tools.map((t) => t.name));
		expect(names.size).toBe(15);
	});

	it("returns undefined for OnChatMessage in V6 (declared, unbound)", () => {
		const binding = getCompoundBinding("OnChatMessage");
		expect(binding).toBeUndefined();
	});

	it("returns undefined for OnStop, OnPieceDetected, OnBarRegression, OnWeeklyReview in V6", () => {
		expect(getCompoundBinding("OnStop")).toBeUndefined();
		expect(getCompoundBinding("OnPieceDetected")).toBeUndefined();
		expect(getCompoundBinding("OnBarRegression")).toBeUndefined();
		expect(getCompoundBinding("OnWeeklyReview")).toBeUndefined();
	});
});
