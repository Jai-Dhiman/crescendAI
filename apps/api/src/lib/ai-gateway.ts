// apps/api/src/lib/ai-gateway.ts
//
// Unified Workers AI / AI Gateway control plane (GA 2026-08-07).
//
// Workers AI traffic that needs the OpenAI chat-completions wire format (tool
// calling and SSE streaming) goes to the unified REST API on api.cloudflare.com
// instead of the pre-unification gateway.ai.cloudflare.com/{account}/{gateway}
// URL. Auth is the Cloudflare API token; the gateway is selected with
// `cf-aig-gateway-id`, which Workers AI requests always require.
//
// Simple non-streaming calls use `env.AI.run(..., { gateway: { id } })` instead
// and need none of this.
//
// Anthropic is NOT routed here: BYOK provider keys are not supported through
// the unified binding or the unified REST API, so those calls keep using the
// provider-native gateway endpoint.
import { ConfigError } from "./errors";
import type { Bindings } from "./types";

export function unifiedChatCompletionsUrl(env: Bindings): string {
	if (!env.CLOUDFLARE_ACCOUNT_ID) {
		throw new ConfigError(
			"CLOUDFLARE_ACCOUNT_ID is required to reach the unified AI REST API",
		);
	}
	return `https://api.cloudflare.com/client/v4/accounts/${env.CLOUDFLARE_ACCOUNT_ID}/ai/v1/chat/completions`;
}

export function unifiedChatCompletionsHeaders(
	env: Bindings,
): Record<string, string> {
	if (!env.CLOUDFLARE_API_TOKEN) {
		throw new ConfigError(
			"CLOUDFLARE_API_TOKEN is required to reach the unified AI REST API",
		);
	}
	return {
		"Content-Type": "application/json",
		Authorization: `Bearer ${env.CLOUDFLARE_API_TOKEN}`,
		"cf-aig-gateway-id": gatewayId(env),
	};
}

export function gatewayId(env: Bindings): string {
	if (!env.AI_GATEWAY_ID) {
		throw new ConfigError("AI_GATEWAY_ID is not configured");
	}
	return env.AI_GATEWAY_ID;
}
