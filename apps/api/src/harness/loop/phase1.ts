import { InferenceError } from "../../lib/errors";
import { withRetries, wrapToolCall } from "./middleware";
import { routeModel } from "./route-model";
import type {
	CompoundBinding,
	PhaseContext,
	Phase1Event,
	ToolDefinition,
} from "./types";

interface AnthropicMessageResponse {
	content: Array<
		| { type: "text"; text: string }
		| { type: "tool_use"; id: string; name: string; input: unknown }
	>;
	stop_reason: string;
}

interface UserMsg {
	role: "user";
	content:
		| string
		| Array<{
				type: "tool_result";
				tool_use_id: string;
				content: string;
				is_error?: boolean;
		  }>;
}

interface AssistantMsg {
	role: "assistant";
	content: AnthropicMessageResponse["content"];
}

type Msg = UserMsg | AssistantMsg;

function buildPhase1Tools(tools: ToolDefinition[]): unknown[] {
	return tools.map((t) => ({
		name: t.name,
		description: t.description,
		input_schema: t.input_schema,
	}));
}

// NOTE: callAnthropicMessage is intentionally duplicated in phase2.ts for Plan 1.
// Extraction to a shared anthropic-client.ts is planned for a future task.
async function callAnthropicMessage(
	env: PhaseContext["env"],
	body: unknown,
): Promise<AnthropicMessageResponse> {
	const client = routeModel("phase1_analysis");
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
			`phase1 anthropic call failed: ${res.status} ${await res.text()}`,
		);
	}
	return (await res.json()) as AnthropicMessageResponse;
}

export async function* runPhase1(
	ctx: PhaseContext,
	binding: CompoundBinding,
): AsyncGenerator<Phase1Event> {
	const messages: Msg[] = [
		{
			role: "user",
			content:
				`Session digest:\n${JSON.stringify(ctx.digest, null, 2)}\n\n` +
				binding.procedurePrompt,
		},
	];
	const toolMap = new Map(binding.tools.map((t) => [t.name, t]));
	const client = routeModel("phase1_analysis");
	let toolCallCount = 0;
	let turnCount = 0;

	while (turnCount < ctx.turnCap) {
		turnCount++;
		const response = await withRetries(() =>
			callAnthropicMessage(ctx.env, {
				model: client.model,
				max_tokens: 2048,
				messages,
				tools: buildPhase1Tools(binding.tools),
				tool_choice: { type: "auto" },
			}),
		);

		const toolUses = response.content.filter(
			(b): b is { type: "tool_use"; id: string; name: string; input: unknown } =>
				b.type === "tool_use",
		);

		if (toolUses.length === 0) {
			yield { type: "phase1_done", toolCallCount, turnCount };
			return;
		}

		messages.push({ role: "assistant", content: response.content });
		const toolResults: Array<{
			type: "tool_result";
			tool_use_id: string;
			content: string;
			is_error?: boolean;
		}> = [];

		for (const tu of toolUses) {
			toolCallCount++;
			yield { type: "phase1_tool_call", id: tu.id, tool: tu.name, input: tu.input };
			const def = toolMap.get(tu.name);
			if (!def) {
				const error = `unknown tool: ${tu.name}`;
				yield { type: "phase1_tool_result", id: tu.id, tool: tu.name, ok: false, error };
				toolResults.push({
					type: "tool_result",
					tool_use_id: tu.id,
					content: error,
					is_error: true,
				});
				continue;
			}
			try {
				const output = await wrapToolCall(() => def.invoke(tu.input));
				yield { type: "phase1_tool_result", id: tu.id, tool: tu.name, ok: true, output };
				toolResults.push({
					type: "tool_result",
					tool_use_id: tu.id,
					content: JSON.stringify(output),
				});
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				yield { type: "phase1_tool_result", id: tu.id, tool: tu.name, ok: false, error: message };
				toolResults.push({
					type: "tool_result",
					tool_use_id: tu.id,
					content: message,
					is_error: true,
				});
			}
		}

		messages.push({ role: "user", content: toolResults });
	}

	yield { type: "phase1_done", toolCallCount, turnCount };
}
