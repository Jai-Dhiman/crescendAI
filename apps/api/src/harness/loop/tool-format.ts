// apps/api/src/harness/loop/tool-format.ts
import { InferenceError } from "../../lib/errors";

export interface AnthropicToolDef {
	name: string;
	description: string;
	input_schema: Record<string, unknown>;
}

export interface AnthropicToolChoiceAuto {
	type: "auto";
}

export interface AnthropicToolChoiceTool {
	type: "tool";
	name: string;
}

export type AnthropicToolChoice =
	| AnthropicToolChoiceAuto
	| AnthropicToolChoiceTool;

export interface AnthropicToolUseBlock {
	type: "tool_use";
	id: string;
	name: string;
	input: unknown;
}

export interface AnthropicTextBlock {
	type: "text";
	text: string;
}

export interface AnthropicToolResultBlock {
	type: "tool_result";
	tool_use_id: string;
	content: string;
	is_error?: boolean;
}

export type AnthropicContentBlock =
	| AnthropicToolUseBlock
	| AnthropicTextBlock
	| AnthropicToolResultBlock;

export interface AnthropicMessage {
	role: "user" | "assistant";
	content: string | AnthropicContentBlock[];
}

export interface AnthropicChatRequest {
	model: string;
	max_tokens: number;
	messages: AnthropicMessage[];
	tools?: AnthropicToolDef[];
	tool_choice?: AnthropicToolChoice;
}

export interface AnthropicMessageResponse {
	content: Array<AnthropicTextBlock | AnthropicToolUseBlock>;
	stop_reason: string;
}

// ---------------------------------------------------------------------------
// toOpenAIChatRequest
// ---------------------------------------------------------------------------

interface OpenAIToolDef {
	type: "function";
	function: {
		name: string;
		description: string;
		parameters: Record<string, unknown>;
	};
}

type OpenAIToolChoice =
	| "auto"
	| "none"
	| { type: "function"; function: { name: string } };

interface OpenAIToolCallFunction {
	name: string;
	arguments: string;
}

interface OpenAIToolCall {
	id: string;
	type: "function";
	function: OpenAIToolCallFunction;
}

type OpenAIMessage =
	| { role: "user"; content: string }
	| { role: "assistant"; content: string | null; tool_calls?: OpenAIToolCall[] }
	| { role: "tool"; tool_call_id: string; content: string };

interface OpenAIChatRequest {
	model: string;
	max_tokens: number;
	messages: OpenAIMessage[];
	tools: OpenAIToolDef[];
	tool_choice: OpenAIToolChoice;
}

export function toOpenAIChatRequest(
	req: AnthropicChatRequest,
): OpenAIChatRequest {
	const tools: OpenAIToolDef[] = (req.tools ?? []).map((t) => ({
		type: "function",
		function: {
			name: t.name,
			description: t.description,
			parameters: t.input_schema,
		},
	}));

	let tool_choice: OpenAIToolChoice = "auto";
	if (req.tool_choice) {
		if (req.tool_choice.type === "auto") {
			tool_choice = "auto";
		} else if (req.tool_choice.type === "tool") {
			tool_choice = {
				type: "function",
				function: { name: req.tool_choice.name },
			};
		}
	}

	const messages: OpenAIMessage[] = [];
	for (const msg of req.messages) {
		if (typeof msg.content === "string") {
			messages.push({
				role: msg.role as "user" | "assistant",
				content: msg.content,
			});
		} else if (Array.isArray(msg.content)) {
			if (msg.role === "assistant") {
				const toolCalls: OpenAIToolCall[] = [];
				for (const block of msg.content) {
					if (block.type === "tool_use") {
						toolCalls.push({
							id: block.id,
							type: "function",
							function: {
								name: block.name,
								arguments: JSON.stringify(block.input),
							},
						});
					}
				}
				messages.push({
					role: "assistant",
					content: null,
					tool_calls: toolCalls,
				});
			} else if (msg.role === "user") {
				for (const block of msg.content) {
					if (block.type === "tool_result") {
						messages.push({
							role: "tool",
							tool_call_id: block.tool_use_id,
							content: block.content,
						});
					}
				}
			}
		}
	}

	return {
		model: req.model,
		max_tokens: req.max_tokens,
		messages,
		tools,
		tool_choice,
	};
}

// ---------------------------------------------------------------------------
// toAnthropicResponse
// ---------------------------------------------------------------------------

interface OpenAIChatResponse {
	choices: Array<{
		message: {
			role: string;
			content: string | null;
			tool_calls?: Array<{
				id: string;
				type: string;
				function: { name: string; arguments: string };
			}> | null;
		};
		finish_reason: string;
	}>;
}

export function toAnthropicResponse(
	res: OpenAIChatResponse,
): AnthropicMessageResponse {
	const choice = res.choices[0];
	if (!choice) {
		return { content: [], stop_reason: "end_turn" };
	}

	const { message } = choice;
	const content: Array<AnthropicTextBlock | AnthropicToolUseBlock> = [];

	if (message.tool_calls && message.tool_calls.length > 0) {
		for (const tc of message.tool_calls) {
			let input: unknown;
			try {
				input = JSON.parse(tc.function.arguments);
			} catch {
				throw new InferenceError(
					`tool-call ${tc.id} (${tc.function.name}) returned invalid JSON arguments: ${tc.function.arguments}`,
				);
			}
			content.push({
				type: "tool_use",
				id: tc.id,
				name: tc.function.name,
				input,
			});
		}
		return { content, stop_reason: "tool_use" };
	}

	if (message.content) {
		content.push({ type: "text", text: message.content });
	}

	return { content, stop_reason: "end_turn" };
}
