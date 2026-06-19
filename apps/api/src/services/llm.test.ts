import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { callWorkersAI, callWorkersAIStream } from "./llm";
import { InferenceError } from "../lib/errors";
import type { Bindings } from "../lib/types";

afterEach(() => {
	vi.restoreAllMocks();
});

describe("callWorkersAI", () => {
	it("sends Authorization Bearer + cf-aig-authorization headers to AI_GATEWAY_ENDPOINT/workers-ai path", async () => {
		const mockFetch = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
			new Response(
				JSON.stringify({
					choices: [{ message: { content: "Test title" } }],
				}),
				{ status: 200, headers: { "Content-Type": "application/json" } },
			),
		);

		const mockEnv = {
			AI_GATEWAY_ENDPOINT: "https://gateway.example.com",
			AI_GATEWAY_TOKEN: "gw-token-abc",
			CLOUDFLARE_API_TOKEN: "test-cf-token-abc123",
		} as unknown as Bindings;

		const result = await callWorkersAI(
			mockEnv,
			"@cf/google/gemma-4-26b-a4b-it",
			[{ role: "user", content: "Generate a title" }],
			30,
		);

		expect(mockFetch).toHaveBeenCalledOnce();
		const [url, init] = mockFetch.mock.calls[0] as [string, RequestInit];
		expect(url).toBe(
			"https://gateway.example.com/workers-ai/v1/chat/completions",
		);

		const headers = init.headers as Record<string, string>;
		expect(headers["Authorization"]).toBe("Bearer test-cf-token-abc123");
		expect(headers["cf-aig-authorization"]).toBe("Bearer gw-token-abc");
		expect(result).toBe("Test title");
	});

	it("throws InferenceError on non-ok response", async () => {
		vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
			new Response("Unauthorized", { status: 401 }),
		);

		const mockEnv = {
			AI_GATEWAY_ENDPOINT: "https://gateway.example.com",
			AI_GATEWAY_TOKEN: "bad-gw-token",
			CLOUDFLARE_API_TOKEN: "bad-token",
		} as unknown as Bindings;

		await expect(
			callWorkersAI(mockEnv, "@cf/test-model", [
				{ role: "user", content: "hi" },
			]),
		).rejects.toThrow("Workers AI error: 401");
	});
});

const streamMockEnv: Pick<Bindings, "AI_GATEWAY_ENDPOINT" | "AI_GATEWAY_TOKEN" | "CLOUDFLARE_API_TOKEN"> = {
	AI_GATEWAY_ENDPOINT: "https://gateway.example.com",
	AI_GATEWAY_TOKEN: "gw-token",
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

	it("POSTs to workers-ai completions endpoint with stream:true and both auth headers, returns res.body", async () => {
		const fakeBody = new ReadableStream();
		vi.mocked(fetch).mockResolvedValue(
			new Response(fakeBody, { status: 200 }),
		);

		const result = await callWorkersAIStream(streamMockEnv as Bindings, stubBody);

		expect(fetch).toHaveBeenCalledOnce();
		const [url, init] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit];
		expect(url).toBe("https://gateway.example.com/workers-ai/v1/chat/completions");
		const headers = init.headers as Record<string, string>;
		expect(headers["cf-aig-authorization"]).toBe("Bearer gw-token");
		expect(headers["Authorization"]).toBe("Bearer cf-token");
		const sentBody = JSON.parse(init.body as string) as Record<string, unknown>;
		expect(sentBody.stream).toBe(true);
		expect(result).toBe(fakeBody);
	});

	it("throws InferenceError when upstream returns non-OK status", async () => {
		vi.mocked(fetch).mockResolvedValue(
			new Response("upstream error", { status: 503 }),
		);

		await expect(callWorkersAIStream(streamMockEnv as Bindings, stubBody)).rejects.toBeInstanceOf(InferenceError);
	});

	it("throws InferenceError when response body is null", async () => {
		vi.mocked(fetch).mockResolvedValue(
			{ ok: true, status: 200, body: null, text: async () => "" } as unknown as Response,
		);

		await expect(callWorkersAIStream(streamMockEnv as Bindings, stubBody)).rejects.toBeInstanceOf(InferenceError);
	});
});
