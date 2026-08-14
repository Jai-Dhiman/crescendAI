import type { AnthropicChatRequest } from "../harness/loop/tool-format";
import { toOpenAIChatRequest } from "../harness/loop/tool-format";
import {
	gatewayId,
	unifiedChatCompletionsHeaders,
	unifiedChatCompletionsUrl,
} from "../lib/ai-gateway";
import { InferenceError } from "../lib/errors";
import type { Bindings } from "../lib/types";

// ---------------------------------------------------------------------------
// Content block types for multi-turn tool_use conversations
// ---------------------------------------------------------------------------

export type AnthropicContentBlock =
	| { type: "text"; text: string }
	| { type: "tool_use"; id: string; name: string; input: unknown }
	| {
			type: "tool_result";
			tool_use_id: string;
			content: string;
			is_error?: boolean;
	  };

interface LlmMessage {
	role: "system" | "user" | "assistant";
	content: string | AnthropicContentBlock[];
}

export interface AnthropicSystemBlock {
	type: "text";
	text: string;
	cache_control?: { type: "ephemeral" };
}

interface AnthropicRequest {
	model: string;
	max_tokens: number;
	system?: string | AnthropicSystemBlock[];
	messages: LlmMessage[];
	stream?: boolean;
	tools?: unknown[];
	tool_choice?: unknown;
}

interface AnthropicResponse {
	content: Array<{
		type: string;
		text?: string;
		id?: string;
		name?: string;
		input?: unknown;
	}>;
	stop_reason: string;
	usage: { input_tokens: number; output_tokens: number };
}

export async function callAnthropic(
	env: Bindings,
	request: AnthropicRequest,
): Promise<AnthropicResponse> {
	const url = `${env.AI_GATEWAY_ENDPOINT}/anthropic/v1/messages`;
	const res = await fetch(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"cf-aig-authorization": `Bearer ${env.AI_GATEWAY_TOKEN}`,
			"anthropic-version": "2023-06-01",
			"anthropic-beta": "prompt-caching-2024-07-31",
		},
		body: JSON.stringify({ ...request, stream: false }),
	});

	if (!res.ok) {
		const text = await res.text();
		throw new InferenceError(`Anthropic request failed: ${res.status} ${text}`);
	}

	return res.json() as Promise<AnthropicResponse>;
}

export async function callAnthropicStream(
	env: Bindings,
	request: AnthropicRequest,
): Promise<ReadableStream> {
	const url = `${env.AI_GATEWAY_ENDPOINT}/anthropic/v1/messages`;
	const res = await fetch(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"cf-aig-authorization": `Bearer ${env.AI_GATEWAY_TOKEN}`,
			"anthropic-version": "2023-06-01",
			"anthropic-beta": "prompt-caching-2024-07-31",
		},
		body: JSON.stringify({ ...request, stream: true }),
	});

	if (!res.ok) {
		const text = await res.text();
		throw new InferenceError(
			`Anthropic stream request failed: ${res.status} ${text}`,
		);
	}

	if (!res.body) {
		throw new InferenceError("Anthropic stream response has no body");
	}

	return res.body;
}

export async function callWorkersAIStream(
	env: Bindings,
	body: AnthropicChatRequest,
): Promise<ReadableStream> {
	const oaiBody = toOpenAIChatRequest(body);
	const res = await fetch(unifiedChatCompletionsUrl(env), {
		method: "POST",
		headers: unifiedChatCompletionsHeaders(env),
		body: JSON.stringify({ ...oaiBody, stream: true }),
	});

	if (!res.ok) {
		const text = await res.text();
		throw new InferenceError(
			`Workers AI stream request failed: ${res.status} ${text}`,
		);
	}

	if (!res.body) {
		throw new InferenceError("Workers AI stream response has no body");
	}

	return res.body;
}

export async function callWorkersAI(
	env: Bindings,
	model: string,
	messages: Array<{ role: string; content: string }>,
	maxTokens: number = 100,
	chatTemplateKwargs?: { enable_thinking?: boolean; clear_thinking?: boolean },
): Promise<string> {
	const inputs: Record<string, unknown> = {
		messages,
		max_tokens: maxTokens,
	};
	if (chatTemplateKwargs) {
		inputs.chat_template_kwargs = chatTemplateKwargs;
	}
	// `model` is chosen at runtime and is not always present in the generated
	// AiModels catalog, so the binding's model-keyed generics cannot be satisfied
	// statically. The response is validated below instead.
	const data = (await env.AI.run(model as keyof AiModels, inputs as never, {
		gateway: { id: gatewayId(env) },
	})) as { response?: string };
	if (!data?.response) {
		throw new InferenceError(
			`Workers AI returned no content. model=${model} body=${JSON.stringify(data).slice(0, 500)}`,
		);
	}
	return data.response;
}
