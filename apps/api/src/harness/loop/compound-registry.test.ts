import { describe, expect, it, test } from "vitest";
import { getCompoundBinding } from "./compound-registry";
import { SynthesisArtifactSchema } from "../artifacts/synthesis";
import { ALL_MOLECULES } from "../skills/molecules";
import { TOOL_REGISTRY } from "../../services/tool-processor";

describe("compound-registry", () => {
	it("returns a binding for OnSessionEnd pointing at session-synthesis", () => {
		const binding = getCompoundBinding("OnSessionEnd");
		expect(binding).toBeDefined();
		expect(binding?.compoundName).toBe("session-synthesis");
		expect(binding?.artifactSchema).toBe(SynthesisArtifactSchema);
		expect(binding?.artifactToolName).toBe("write_synthesis_artifact");
		expect(binding?.tools).toHaveLength(ALL_MOLECULES.length + 2);
		const names = binding!.tools.map((t) => t.name);
		expect(new Set(names).size).toBe(names.length);
		expect(names).toContain("assign_segment_loop");
	});

	it("returns a streaming binding for OnChatMessage with prescribe_exercise tool", () => {
		const binding = getCompoundBinding("OnChatMessage");
		expect(binding).toBeDefined();
		expect(binding?.compoundName).toBe("chat-response");
		expect(binding?.mode).toBe("streaming");
		expect(binding?.phases).toBe(1);
		expect(binding!.tools.length).toBeGreaterThanOrEqual(Object.values(TOOL_REGISTRY).length + 1);
		const names = binding!.tools.map((t) => t.name);
		expect(new Set(names).size).toBe(names.length);
		expect(names).toContain("prescribe_exercise");
		expect(names).not.toContain("create_exercise");
		expect(names).toContain("search_catalog");
		expect(names).toContain("assign_segment_loop");
	});

	it("returns undefined for OnStop, OnPieceDetected, OnBarRegression, OnWeeklyReview in V6", () => {
		expect(getCompoundBinding("OnStop")).toBeUndefined();
		expect(getCompoundBinding("OnPieceDetected")).toBeUndefined();
		expect(getCompoundBinding("OnBarRegression")).toBeUndefined();
		expect(getCompoundBinding("OnWeeklyReview")).toBeUndefined();
	});
});

test('SESSION_SYNTHESIS_PROCEDURE contains "bar_range, scope, and evidence_refs" instruction', () => {
	const binding = getCompoundBinding('OnSessionEnd')!
	expect(binding.procedurePrompt).toContain('bar_range, scope, and evidence_refs')
})

test('OnSessionEnd procedurePrompt does not contain old "signal data from the digest" instruction', () => {
	const binding = getCompoundBinding('OnSessionEnd')!
	expect(binding.procedurePrompt).not.toContain('signal data from the digest')
})

test('OnSessionEnd tool list includes extract-bar-range-signals', () => {
	const binding = getCompoundBinding('OnSessionEnd')!
	expect(binding.tools.some(t => t.name === 'extract-bar-range-signals')).toBe(true)
})

test('OnSessionEnd tool list does NOT include articulation-clarity-check', () => {
	const binding = getCompoundBinding('OnSessionEnd')!
	expect(binding.tools.some(t => t.name === 'articulation-clarity-check')).toBe(false)
})
