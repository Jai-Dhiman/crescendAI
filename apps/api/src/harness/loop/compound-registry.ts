import { SynthesisArtifactSchema } from "../artifacts/synthesis";
import type { CompoundBinding, HookKind } from "./types";
import { ALL_MOLECULES } from "../skills/molecules";
import { TOOL_REGISTRY } from "../../services/tool-processor";
import { ASSIGN_SEGMENT_LOOP_TOOL } from "../atoms/assign-segment-loop";

const SESSION_SYNTHESIS_PROCEDURE = `You are running the session-synthesis compound.
Phase 1 (this call): analyze the session digest and dispatch any registered diagnosis molecules across plausible bar ranges. Each molecule chains the necessary atoms deterministically — you only need to pick the right molecule and supply the bar range and signal data from the digest. When you have enough diagnoses, end your turn without calling tools.
Phase 2 (next call): you will be prompted again with the collected diagnoses and asked to write a SynthesisArtifact. Do not attempt to write the artifact in this phase.`;

const REGISTRY: Map<HookKind, CompoundBinding> = new Map([
	[
		"OnSessionEnd" as const,
		{
			compoundName: "session-synthesis",
			procedurePrompt: SESSION_SYNTHESIS_PROCEDURE,
			tools: [...ALL_MOLECULES],
			mode: "buffered" as const,
			phases: 2 as const,
			artifactSchema: SynthesisArtifactSchema,
			artifactToolName: "write_synthesis_artifact",
		},
	],
	[
		"OnChatMessage" as const,
		{
			compoundName: "chat-response",
			procedurePrompt: "",
			tools: [
				...Object.values(TOOL_REGISTRY).map((t) => ({
					name: t.name,
					description: t.description,
					input_schema: t.anthropicSchema.input_schema as Record<string, unknown>,
					invoke: async (_input: unknown): Promise<unknown> => ({}),
				})),
				ASSIGN_SEGMENT_LOOP_TOOL,
			],
			mode: "streaming" as const,
			phases: 1 as const,
		},
	],
]);

export function getCompoundBinding(
	hook: HookKind,
): CompoundBinding | undefined {
	return REGISTRY.get(hook);
}
