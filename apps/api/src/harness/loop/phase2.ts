import { zodToJsonSchema } from "zod-to-json-schema";
import { InferenceError } from "../../lib/errors";
import { withRetries } from "./middleware";
import { routeModel } from "./route-model";
import type { Phase2Binding, HookEvent, PhaseContext } from "./types";

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

	const userPrompt =
		`Session digest:\n${JSON.stringify(ctx.digest, null, 2)}\n\n` +
		`Collected diagnoses (${diagnoses.length}):\n${JSON.stringify(diagnoses, null, 2)}\n\n` +
		`Write the SynthesisArtifact now using the ${binding.artifactToolName} tool.`;

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
