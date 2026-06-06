import { zodToJsonSchema } from "zod-to-json-schema";
import { InferenceError } from "../../lib/errors";
import { FIRST_SESSION_GUARDRAIL } from "../../services/prompts";
import { withRetries } from "./middleware";
import { routeModel } from "./route-model";
import type { HookEvent, Phase2Binding, PhaseContext } from "./types";

interface AnthropicMessageResponse {
	content: Array<
		| { type: "text"; text: string }
		| { type: "tool_use"; id: string; name: string; input: unknown }
	>;
	stop_reason: string;
}

// NOTE: callAnthropicMessage is intentionally duplicated from phase1.ts for Plan 1.
// Extraction to a shared anthropic-client.ts is planned for a future task.
async function callAnthropicMessage(
	env: PhaseContext["env"],
	body: unknown,
): Promise<AnthropicMessageResponse> {
	const client = routeModel("phase2_voice");
	const url = `${env[client.gatewayUrlVar]}/anthropic/v1/messages`;
	const res = await fetch(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"x-api-key": env.ANTHROPIC_API_KEY,
			"anthropic-version": "2023-06-01",
		},
		body: JSON.stringify(body),
	});
	if (!res.ok) {
		throw new InferenceError(
			`phase2 anthropic call failed: ${res.status} ${await res.text()}`,
		);
	}
	return (await res.json()) as AnthropicMessageResponse;
}

export function buildPhase2Prompt(
	digest: Record<string, unknown>,
	diagnoses: unknown[],
	guardrail: string,
): string {
	const reflectionInstruction =
		"Headline instructions: write a light reflection in 2-4 sentences about what happened " +
		"in this session, ending in exactly one directional question about the dominant_dimension " +
		"(e.g. 'Want a drill targeting that?'). The headline must be 300-500 characters total. " +
		"Do not list all dimensions; focus on the one area that matters most.\n\n";

	const exerciseInstruction =
		"Exercise instructions: if proposed_exercises is non-empty, proposed_exercises[0] must " +
		"target the dominant_dimension so that the pre-staged exercise aligns with the question.\n\n";

	return (
		`Session digest:\n${JSON.stringify(digest, null, 2)}\n\n` +
		`Collected diagnoses (${diagnoses.length}):\n${JSON.stringify(diagnoses, null, 2)}\n\n` +
		guardrail +
		reflectionInstruction +
		exerciseInstruction +
		`Write the SynthesisArtifact now using the write_synthesis_artifact tool.`
	);
}

export async function* runPhase2(
	ctx: PhaseContext,
	binding: Phase2Binding,
	diagnoses: unknown[],
): AsyncGenerator<HookEvent<unknown>> {
	yield { type: "phase2_started" };

	const writeTool = {
		name: binding.artifactToolName,
		description:
			"Write the final compound artifact. Call this exactly once with the structured fields.",
		input_schema: artifactInputSchema(binding.artifactSchema),
	};

	const guardrail =
		ctx.digest.reference_mode === "within_session"
			? `${FIRST_SESSION_GUARDRAIL}\n\n`
			: "";

	const userPrompt = buildPhase2Prompt(ctx.digest, diagnoses, guardrail);

	const client = routeModel("phase2_voice");
	const response = await withRetries(() =>
		callAnthropicMessage(ctx.env, {
			model: client.model,
			max_tokens: 2048,
			messages: [{ role: "user", content: userPrompt }],
			tools: [writeTool],
			tool_choice: { type: "tool", name: binding.artifactToolName },
		}),
	);

	const toolUse = response.content.find(
		(b): b is { type: "tool_use"; id: string; name: string; input: unknown } =>
			b.type === "tool_use" && b.name === binding.artifactToolName,
	);

	if (!toolUse) {
		yield {
			type: "phase_error",
			phase: 2,
			error: "no tool_use returned despite forced tool_choice",
		};
		return;
	}

	const parsed = binding.artifactSchema.safeParse(toolUse.input);
	if (!parsed.success) {
		yield {
			type: "validation_error",
			raw: toolUse.input,
			zodError: parsed.error.message,
		};
		return;
	}

	yield { type: "artifact", value: parsed.data };
}

function artifactInputSchema(
	schema: Phase2Binding["artifactSchema"],
): Record<string, unknown> {
	return zodToJsonSchema(schema, { target: "openApi3" }) as Record<
		string,
		unknown
	>;
}
