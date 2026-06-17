import { withRetries, wrapToolCall } from "./middleware";
import { callModel } from "./gateway-client";
import { routeModel } from "./route-model";
import type {
	CompoundBinding,
	PhaseContext,
	Phase1Event,
	ToolDefinition,
} from "./types";

function buildPhase1Tools(tools: ToolDefinition[]): unknown[] {
	return tools.map((t) => ({
		name: t.name,
		description: t.description,
		input_schema: t.input_schema,
	}));
}

export async function* runPhase1(
	ctx: PhaseContext,
	binding: CompoundBinding,
): AsyncGenerator<Phase1Event> {
	const messages: Array<{
		role: "user" | "assistant";
		content:
			| string
			| Array<
					| { type: "text"; text: string }
					| { type: "tool_use"; id: string; name: string; input: unknown }
					| { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean }
			  >;
	}> = [
		{
			role: "user",
			content:
				`Session digest:\n${JSON.stringify(ctx.digest, null, 2)}\n\n` +
				binding.procedurePrompt,
		},
	];
	const toolMap = new Map(binding.tools.map((t) => [t.name, t]));
	const client = routeModel("phase1_analysis", ctx.env);
	let toolCallCount = 0;
	let turnCount = 0;

	while (turnCount < ctx.turnCap) {
		turnCount++;
		const response = await withRetries(() =>
			callModel(ctx.env, client, {
				model: client.model,
				max_tokens: 2048,
				messages,
				tools: buildPhase1Tools(binding.tools) as Array<{
					name: string;
					description: string;
					input_schema: Record<string, unknown>;
				}>,
				tool_choice: { type: "auto" },
			}),
		);

		const toolUses = response.content.filter(
			(
				b,
			): b is { type: "tool_use"; id: string; name: string; input: unknown } =>
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
			yield {
				type: "phase1_tool_call",
				id: tu.id,
				tool: tu.name,
				input: tu.input,
			};
			const def = toolMap.get(tu.name);
			if (!def) {
				const error = `unknown tool: ${tu.name}`;
				yield {
					type: "phase1_tool_result",
					id: tu.id,
					tool: tu.name,
					ok: false,
					error,
				};
				toolResults.push({
					type: "tool_result",
					tool_use_id: tu.id,
					content: error,
					is_error: true,
				});
				continue;
			}
			try {
				const output = await wrapToolCall(tu.name, ctx, () =>
					def.invoke(tu.input, ctx),
				);
				yield {
					type: "phase1_tool_result",
					id: tu.id,
					tool: tu.name,
					ok: true,
					output,
				};
				toolResults.push({
					type: "tool_result",
					tool_use_id: tu.id,
					content: JSON.stringify(output),
				});
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				yield {
					type: "phase1_tool_result",
					id: tu.id,
					tool: tu.name,
					ok: false,
					error: message,
				};
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
