// apps/api/src/harness/loop/gateway-client.test.ts
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Bindings } from "../../lib/types";
import { callModel } from "./gateway-client";
import { InferenceError } from "../../lib/errors";

const BASE_ENV = {
	AI_GATEWAY_ENDPOINT: "https://gw.example.com",
	AI_GATEWAY_TOKEN: "gw-token-abc",
	CLOUDFLARE_API_TOKEN: "cf-token-xyz",
	TEACHER_PROVIDER: "anthropic",
} as unknown as Bindings;

const WORKERS_AI_ENV = {
	...BASE_ENV,
	TEACHER_PROVIDER: "workers-ai",
} as unknown as Bindings;

const ANTHROPIC_RESPONSE = {
	content: [{ type: "text", text: "Hello from Anthropic" }],
	stop_reason: "end_turn",
	usage: { input_tokens: 10, output_tokens: 5 },
};

const WORKERS_AI_OPENAI_RESPONSE = {
	choices: [
		{
			message: {
				role: "assistant",
				content: null,
				tool_calls: [
					{
						id: "call_1",
						type: "function",
						function: {
							name: "write_synthesis_artifact",
							arguments: '{"headline":"Good session","focus_areas":[]}',
						},
					},
				],
			},
			finish_reason: "tool_calls",
		},
	],
};

describe("callModel — anthropic provider", () => {
	const fetchSpy = vi.fn();

	beforeEach(() => {
		fetchSpy.mockReset();
		vi.stubGlobal("fetch", fetchSpy);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("POSTs to /anthropic/v1/messages with cf-aig-authorization and anthropic-version headers", async () => {
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify(ANTHROPIC_RESPONSE), { status: 200 }),
		);

		const client = {
			provider: "anthropic" as const,
			model: "claude-sonnet-4-20250514",
		};
		const body = {
			model: client.model,
			max_tokens: 2048,
			messages: [{ role: "user" as const, content: "test" }],
			tools: [],
			tool_choice: { type: "auto" as const },
		};

		const result = await callModel(BASE_ENV, client, body);

		expect(fetchSpy).toHaveBeenCalledOnce();
		const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
		expect(url).toBe("https://gw.example.com/anthropic/v1/messages");
		const headers = init.headers as Record<string, string>;
		expect(headers["cf-aig-authorization"]).toBe("Bearer gw-token-abc");
		expect(headers["anthropic-version"]).toBe("2023-06-01");
		expect(headers["Content-Type"]).toBe("application/json");
		expect(result.content).toHaveLength(1);
		expect(result.stop_reason).toBe("end_turn");
	});

	it("throws InferenceError on non-2xx from anthropic path", async () => {
		fetchSpy.mockResolvedValueOnce(
			new Response("credit balance too low", { status: 400 }),
		);

		const client = {
			provider: "anthropic" as const,
			model: "claude-sonnet-4-20250514",
		};
		const body = {
			model: client.model,
			max_tokens: 2048,
			messages: [{ role: "user" as const, content: "test" }],
			tools: [],
			tool_choice: { type: "auto" as const },
		};

		await expect(callModel(BASE_ENV, client, body)).rejects.toThrow(
			InferenceError,
		);
	});
});

describe("callModel — workers-ai provider", () => {
	const fetchSpy = vi.fn();

	beforeEach(() => {
		fetchSpy.mockReset();
		vi.stubGlobal("fetch", fetchSpy);
	});

	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("POSTs to /workers-ai/v1/chat/completions with cf-aig-authorization + Authorization Bearer headers", async () => {
		fetchSpy.mockResolvedValueOnce(
			new Response(JSON.stringify(WORKERS_AI_OPENAI_RESPONSE), { status: 200 }),
		);

		const client = {
			provider: "workers-ai" as const,
			model: "@cf/qwen/qwen3-30b-a3b-fp8",
		};
		const body = {
			model: client.model,
			max_tokens: 2048,
			messages: [{ role: "user" as const, content: "test" }],
			tools: [
				{
					name: "write_synthesis_artifact",
					description: "Write artifact",
					input_schema: { type: "object" },
				},
			],
			tool_choice: { type: "tool" as const, name: "write_synthesis_artifact" },
		};

		const result = await callModel(WORKERS_AI_ENV, client, body);

		expect(fetchSpy).toHaveBeenCalledOnce();
		const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
		expect(url).toBe("https://gw.example.com/workers-ai/v1/chat/completions");
		const headers = init.headers as Record<string, string>;
		expect(headers["cf-aig-authorization"]).toBe("Bearer gw-token-abc");
		expect(headers["Authorization"]).toBe("Bearer cf-token-xyz");
		expect(headers["Content-Type"]).toBe("application/json");
		// Response is translated back to Anthropic shape
		expect(result.stop_reason).toBe("tool_use");
		expect(result.content).toHaveLength(1);
		const block = result.content[0] as {
			type: string;
			name: string;
			input: unknown;
		};
		expect(block.type).toBe("tool_use");
		expect(block.name).toBe("write_synthesis_artifact");
		expect(block.input).toEqual({ headline: "Good session", focus_areas: [] });
	});

	it("throws InferenceError on non-2xx from workers-ai path", async () => {
		fetchSpy.mockResolvedValueOnce(
			new Response("Unauthorized", { status: 401 }),
		);

		const client = {
			provider: "workers-ai" as const,
			model: "@cf/qwen/qwen3-30b-a3b-fp8",
		};
		const body = {
			model: client.model,
			max_tokens: 2048,
			messages: [{ role: "user" as const, content: "test" }],
			tools: [],
			tool_choice: { type: "auto" as const },
		};

		await expect(callModel(WORKERS_AI_ENV, client, body)).rejects.toThrow(
			InferenceError,
		);
	});
});
