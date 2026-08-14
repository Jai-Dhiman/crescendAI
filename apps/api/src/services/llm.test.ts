import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ConfigError, InferenceError } from "../lib/errors";
import type { Bindings } from "../lib/types";
import { callWorkersAI, callWorkersAIStream } from "./llm";

afterEach(() => {
	vi.restoreAllMocks();
});

function envWithAiRun(run: ReturnType<typeof vi.fn>): Bindings {
	return {
		AI: { run } as unknown as Ai,
		AI_GATEWAY_ID: "crescendai",
	} as unknown as Bindings;
}

describe("callWorkersAI", () => {
	it("runs the model on the AI binding routed through the configured gateway", async () => {
		const run = vi.fn().mockResolvedValueOnce({ response: "Test title" });

		const result = await callWorkersAI(
			envWithAiRun(run),
			"@cf/google/gemma-4-26b-a4b-it",
			[{ role: "user", content: "Generate a title" }],
			30,
			{ enable_thinking: false },
		);

		expect(run).toHaveBeenCalledOnce();
		expect(run).toHaveBeenCalledWith(
			"@cf/google/gemma-4-26b-a4b-it",
			{
				messages: [{ role: "user", content: "Generate a title" }],
				max_tokens: 30,
				chat_template_kwargs: { enable_thinking: false },
			},
			{ gateway: { id: "crescendai" } },
		);
		expect(result).toBe("Test title");
	});

	it("throws InferenceError when the binding returns no content", async () => {
		const run = vi.fn().mockResolvedValueOnce({ usage: {} });

		await expect(
			callWorkersAI(envWithAiRun(run), "@cf/test-model", [
				{ role: "user", content: "hi" },
			]),
		).rejects.toThrow(InferenceError);
	});

	it("throws ConfigError when no gateway is configured", async () => {
		const run = vi.fn();
		const mockEnv = { AI: { run } as unknown as Ai } as unknown as Bindings;

		await expect(
			callWorkersAI(mockEnv, "@cf/test-model", [
				{ role: "user", content: "hi" },
			]),
		).rejects.toThrow(ConfigError);
		expect(run).not.toHaveBeenCalled();
	});
});

const streamMockEnv: Pick<
	Bindings,
	"CLOUDFLARE_ACCOUNT_ID" | "AI_GATEWAY_ID" | "CLOUDFLARE_API_TOKEN"
> = {
	CLOUDFLARE_ACCOUNT_ID: "acct-123",
	AI_GATEWAY_ID: "crescendai",
	CLOUDFLARE_API_TOKEN: "cf-token",
};

const stubBody: import("../harness/loop/tool-format").AnthropicChatRequest = {
	model: "@cf/zai-org/glm-4.7-flash",
	max_tokens: 2048,
	messages: [{ role: "user", content: "Hello" }],
	tools: [],
	tool_choice: { type: "auto" },
};

describe("callWorkersAIStream", () => {
	beforeEach(() => {
		vi.stubGlobal("fetch", vi.fn());
	});

	it("POSTs to the unified AI REST endpoint with stream:true and gateway routing, returns res.body", async () => {
		const fakeBody = new ReadableStream();
		vi.mocked(fetch).mockResolvedValue(new Response(fakeBody, { status: 200 }));

		const result = await callWorkersAIStream(
			streamMockEnv as Bindings,
			stubBody,
		);

		expect(fetch).toHaveBeenCalledOnce();
		const [url, init] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit];
		expect(url).toBe(
			"https://api.cloudflare.com/client/v4/accounts/acct-123/ai/v1/chat/completions",
		);
		const headers = init.headers as Record<string, string>;
		expect(headers["cf-aig-gateway-id"]).toBe("crescendai");
		expect(headers["Authorization"]).toBe("Bearer cf-token");
		const sentBody = JSON.parse(init.body as string) as Record<string, unknown>;
		expect(sentBody.stream).toBe(true);
		expect(result).toBe(fakeBody);
	});

	it("throws InferenceError when upstream returns non-OK status", async () => {
		vi.mocked(fetch).mockResolvedValue(
			new Response("upstream error", { status: 503 }),
		);

		await expect(
			callWorkersAIStream(streamMockEnv as Bindings, stubBody),
		).rejects.toBeInstanceOf(InferenceError);
	});

	it("throws InferenceError when response body is null", async () => {
		vi.mocked(fetch).mockResolvedValue({
			ok: true,
			status: 200,
			body: null,
			text: async () => "",
		} as unknown as Response);

		await expect(
			callWorkersAIStream(streamMockEnv as Bindings, stubBody),
		).rejects.toBeInstanceOf(InferenceError);
	});
});
