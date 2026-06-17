// apps/api/src/harness/loop/gateway-client.ts
import { InferenceError } from "../../lib/errors";
import type { Bindings } from "../../lib/types";
import type {
	AnthropicChatRequest,
	AnthropicMessageResponse,
} from "./tool-format";
import { toAnthropicResponse, toOpenAIChatRequest } from "./tool-format";

export interface ModelClient {
	provider: "anthropic" | "workers-ai";
	model: string;
}

export async function callModel(
	env: Bindings,
	client: ModelClient,
	body: AnthropicChatRequest,
): Promise<AnthropicMessageResponse> {
	if (client.provider === "anthropic") {
		const url = `${env.AI_GATEWAY_ENDPOINT}/anthropic/v1/messages`;
		const res = await fetch(url, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				"cf-aig-authorization": `Bearer ${env.AI_GATEWAY_TOKEN}`,
				"anthropic-version": "2023-06-01",
			},
			body: JSON.stringify(body),
		});
		if (!res.ok) {
			throw new InferenceError(
				`callModel anthropic failed: ${res.status} ${await res.text()}`,
			);
		}
		return (await res.json()) as AnthropicMessageResponse;
	}

	// workers-ai: translate request, call, translate response
	const url = `${env.AI_GATEWAY_ENDPOINT}/workers-ai/v1/chat/completions`;
	const oaiBody = toOpenAIChatRequest(body);
	const res = await fetch(url, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"cf-aig-authorization": `Bearer ${env.AI_GATEWAY_TOKEN}`,
			Authorization: `Bearer ${env.CLOUDFLARE_API_TOKEN}`,
		},
		body: JSON.stringify(oaiBody),
	});
	if (!res.ok) {
		throw new InferenceError(
			`callModel workers-ai failed: ${res.status} ${await res.text()}`,
		);
	}
	const oaiRes = await res.json();
	return toAnthropicResponse(
		oaiRes as Parameters<typeof toAnthropicResponse>[0],
	);
}
