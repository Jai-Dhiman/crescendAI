import { afterEach, describe, expect, it, vi } from "vitest";
import { callWorkersAI } from "./llm";
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
