import { InferenceError } from "../lib/errors";
import type { Bindings } from "../lib/types";

// ---------------------------------------------------------------------------
// Content block types for multi-turn tool_use conversations
// ---------------------------------------------------------------------------

export type AnthropicContentBlock =
	| { type: "text"; text: string }
	| { type: "tool_use"; id: string; name: string; input: unknown }
	| { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean };

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
	const url = `${env.AI_GATEWAY_TEACHER}/anthropic/v1/messages`;
	const res = await fetch(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"x-api-key": env.ANTHROPIC_API_KEY,
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
	const url = `${env.AI_GATEWAY_TEACHER}/anthropic/v1/messages`;
	const res = await fetch(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"x-api-key": env.ANTHROPIC_API_KEY,
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

export async function callWorkersAI(
	env: Bindings,
	model: string,
	messages: Array<{ role: string; content: string }>,
	maxTokens: number = 100,
	chatTemplateKwargs?: { enable_thinking?: boolean; clear_thinking?: boolean },
): Promise<string> {
	const url = `${env.AI_GATEWAY_BACKGROUND}/workers-ai/v1/chat/completions`;
	const body: Record<string, unknown> = {
		model,
		messages,
		max_tokens: maxTokens,
	};
	if (chatTemplateKwargs) {
		body.chat_template_kwargs = chatTemplateKwargs;
	}
	const res = await fetch(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			Authorization: `Bearer ${env.CLOUDFLARE_API_TOKEN}`,
		},
		body: JSON.stringify(body),
	});
	if (!res.ok) {
		throw new InferenceError(`Workers AI error: ${res.status}`);
	}
	const data = (await res.json()) as {
		choices: Array<{ message: { content: string } }>;
	};
	if (!data.choices?.[0]?.message?.content) {
		throw new InferenceError(
			`Workers AI returned no content. model=${model} body=${JSON.stringify(data).slice(0, 500)}`,
		);
	}
	return data.choices[0].message.content;
}
