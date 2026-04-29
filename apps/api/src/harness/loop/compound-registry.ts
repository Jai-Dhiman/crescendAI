import { SynthesisArtifactSchema } from "../artifacts/synthesis";
import type { CompoundBinding, HookKind } from "./types";
import { ALL_ATOMS } from "../skills/atoms";

const SESSION_SYNTHESIS_PROCEDURE = `You are running the session-synthesis compound.
Phase 1 (this call): analyze the session digest and dispatch any registered diagnosis tools across plausible bar ranges. When you have enough material, end your turn without calling tools.
Phase 2 (next call): you will be prompted again with the collected diagnoses and asked to write a SynthesisArtifact. Do not attempt to write the artifact in this phase.`;

const REGISTRY: Map<HookKind, CompoundBinding> = new Map([
	[
		"OnSessionEnd" as const,
		{
			compoundName: "session-synthesis",
			procedurePrompt: SESSION_SYNTHESIS_PROCEDURE,
			tools: [...ALL_ATOMS],
			mode: "buffered" as const,
			phases: 2 as const,
			artifactSchema: SynthesisArtifactSchema,
			artifactToolName: "write_synthesis_artifact",
		},
	],
]);

export function getCompoundBinding(
	hook: HookKind,
): CompoundBinding | undefined {
	return REGISTRY.get(hook);
}
