import { describe, expect, it } from "vitest";
import { getCompoundBinding } from "./compound-registry";
import { SynthesisArtifactSchema } from "../artifacts/synthesis";
import { ALL_MOLECULES } from "../skills/molecules";

describe("compound-registry", () => {
	it("returns a binding for OnSessionEnd pointing at session-synthesis", () => {
		const binding = getCompoundBinding("OnSessionEnd");
		expect(binding).toBeDefined();
		expect(binding?.compoundName).toBe("session-synthesis");
		expect(binding?.artifactSchema).toBe(SynthesisArtifactSchema);
		expect(binding?.artifactToolName).toBe("write_synthesis_artifact");
		expect(binding?.tools).toHaveLength(ALL_MOLECULES.length);
		const names = binding!.tools.map((t) => t.name);
		expect(new Set(names).size).toBe(names.length);
	});

	it("returns a streaming binding for OnChatMessage with 6 tools", () => {
		const binding = getCompoundBinding("OnChatMessage");
		expect(binding).toBeDefined();
		expect(binding?.compoundName).toBe("chat-response");
		expect(binding?.mode).toBe("streaming");
		expect(binding?.phases).toBe(1);
		expect(binding?.tools).toHaveLength(6);
		const names = binding!.tools.map((t) => t.name);
		expect(new Set(names).size).toBe(names.length);
		expect(names).toContain("create_exercise");
		expect(names).toContain("search_catalog");
	});

	it("returns undefined for OnStop, OnPieceDetected, OnBarRegression, OnWeeklyReview in V6", () => {
		expect(getCompoundBinding("OnStop")).toBeUndefined();
		expect(getCompoundBinding("OnPieceDetected")).toBeUndefined();
		expect(getCompoundBinding("OnBarRegression")).toBeUndefined();
		expect(getCompoundBinding("OnWeeklyReview")).toBeUndefined();
	});
});
