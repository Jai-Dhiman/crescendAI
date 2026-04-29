import { describe, expect, it } from "vitest";
import { getCompoundBinding } from "./compound-registry";
import { SynthesisArtifactSchema } from "../artifacts/synthesis";
import { ALL_ATOMS } from "../skills/atoms";

describe("compound-registry", () => {
	it("returns a binding for OnSessionEnd pointing at session-synthesis", () => {
		const binding = getCompoundBinding("OnSessionEnd");
		expect(binding).toBeDefined();
		expect(binding?.compoundName).toBe("session-synthesis");
		expect(binding?.artifactSchema).toBe(SynthesisArtifactSchema);
		expect(binding?.artifactToolName).toBe("write_synthesis_artifact");
		expect(binding?.tools).toHaveLength(ALL_ATOMS.length);
		const names = binding!.tools.map((t) => t.name);
		expect(new Set(names).size).toBe(names.length);
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
