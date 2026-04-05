import { Hono } from "hono";
import type { Mock } from "vitest";
import { describe, expect, it, vi } from "vitest";

// Mock chat service
vi.mock("../services/chat", () => ({
	prepareChatContext: vi.fn().mockResolvedValue({
		conversationId: "00000000-0000-0000-0000-000000000001",
		isNewConversation: true,
		messages: [{ role: "user", content: "Hello" }],
		dynamicContext: "",
	}),
	saveAssistantMessage: vi.fn().mockResolvedValue(undefined),
}));

// Mock teacher service with an async generator that yields delta + done
vi.mock("../services/teacher", () => ({
	chat: vi.fn().mockImplementation(async function* () {
		yield { type: "delta", text: "Hi " };
		yield { type: "delta", text: "there!" };
		yield { type: "done", fullText: "Hi there!", allComponents: [] };
	}),
}));

// Now import the route (after mocks are set up)
const { chatRoutes } = await import("./chat");

function createTestApp() {
	return new Hono()
		.use("*", async (c, next) => {
			c.set("studentId" as never, "test-student-id");
			c.set("db" as never, {});
			// Stub executionCtx.waitUntil for the waitUntil call in the route
			Object.defineProperty(c, "executionCtx", {
				get: () => ({ waitUntil: () => {} }),
			});
			await next();
		})
		.route("/api/chat", chatRoutes);
}

/** Parse raw SSE text into an array of {event, data, id} objects */
function parseSSE(raw: string): Array<{ event?: string; data?: string; id?: string }> {
	const events: Array<{ event?: string; data?: string; id?: string }> = [];
	// SSE events are separated by blank lines
	const blocks = raw.split(/\n\n+/).filter((b) => b.trim());
	for (const block of blocks) {
		const entry: { event?: string; data?: string; id?: string } = {};
		for (const line of block.split("\n")) {
			if (line.startsWith("event: ")) entry.event = line.slice(7);
			else if (line.startsWith("data: ")) entry.data = line.slice(6);
			else if (line.startsWith("id: ")) entry.id = line.slice(4);
		}
		if (entry.data !== undefined || entry.event !== undefined) {
			events.push(entry);
		}
	}
	return events;
}

const testApp = createTestApp();

describe("POST /api/chat", () => {
	it("returns 401 without auth", async () => {
		const noAuthApp = new Hono().route("/api/chat", chatRoutes);
		const res = await noAuthApp.request("/api/chat", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ message: "Hello" }),
		});
		expect(res.status).toBe(401);
	});

	it("returns 400 for empty message", async () => {
		const res = await testApp.request("/api/chat", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ message: "" }),
		});
		expect(res.status).toBe(400);
	});

	it("returns 400 for missing message", async () => {
		const res = await testApp.request("/api/chat", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({}),
		});
		expect(res.status).toBe(400);
	});

	it("accepts null conversationId for new chat", async () => {
		const res = await testApp.request("/api/chat", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ conversationId: null, message: "Hello" }),
		});
		// Should NOT be 400 — null is a valid way to express "no conversation" in JSON.
		expect(res.status).not.toBe(400);
	});

	it("SSE data payloads all contain a type field matching ChatStreamEvent", async () => {
		const res = await testApp.request("/api/chat", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ message: "Hello" }),
		});

		expect(res.status).toBe(200);
		expect(res.headers.get("content-type")).toContain("text/event-stream");

		const raw = await res.text();
		const sseEvents = parseSSE(raw);

		// Should have: start, delta, delta, done (at minimum)
		expect(sseEvents.length).toBeGreaterThanOrEqual(4);

		// Every SSE data payload must be valid JSON with a `type` field
		const validTypes = new Set(["start", "delta", "tool_result", "done", "error"]);
		const parsedEvents: Array<{ type: string; [key: string]: unknown }> = [];

		for (const sse of sseEvents) {
			expect(sse.data).toBeDefined();
			const parsed = JSON.parse(sse.data!);
			expect(parsed).toHaveProperty("type");
			expect(validTypes).toContain(parsed.type);
			// SSE event field should match the type in the data payload
			expect(sse.event).toBe(parsed.type);
			parsedEvents.push(parsed);
		}

		// Verify event sequence: start -> deltas -> done
		expect(parsedEvents[0].type).toBe("start");
		expect(parsedEvents[0]).toHaveProperty("conversationId");
		expect(parsedEvents[parsedEvents.length - 1].type).toBe("done");

		// Verify delta events carry text
		const deltas = parsedEvents.filter((e) => e.type === "delta");
		expect(deltas.length).toBe(2);
		expect(deltas[0].text).toBe("Hi ");
		expect(deltas[1].text).toBe("there!");
	});

	it("SSE error event includes type field when teacher service throws", async () => {
		const teacherService = await import("../services/teacher");
		(teacherService.chat as Mock).mockImplementationOnce(async function* () {
			throw new Error("LLM unavailable");
		});

		const res = await testApp.request("/api/chat", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ message: "Hello" }),
		});

		expect(res.status).toBe(200);
		const raw = await res.text();
		const sseEvents = parseSSE(raw);

		const errorEvents = sseEvents.filter((e) => e.event === "error");
		expect(errorEvents.length).toBe(1);

		const parsed = JSON.parse(errorEvents[0].data!);
		expect(parsed.type).toBe("error");
		expect(parsed.message).toBeDefined();
	});

	it("SSE tool_result event includes type field", async () => {
		const teacherService = await import("../services/teacher");
		(teacherService.chat as Mock).mockImplementationOnce(async function* () {
			yield { type: "delta", text: "Try this: " };
			yield {
				type: "tool_result",
				name: "create_exercise",
				componentsJson: [{ type: "exercise", data: { id: "ex-1" } }],
			};
			yield {
				type: "done",
				fullText: "Try this: ",
				allComponents: [{ type: "exercise", data: { id: "ex-1" } }],
			};
		});

		const res = await testApp.request("/api/chat", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ message: "Give me an exercise" }),
		});

		expect(res.status).toBe(200);
		const raw = await res.text();
		const sseEvents = parseSSE(raw);

		const toolEvents = sseEvents.filter((e) => e.event === "tool_result");
		expect(toolEvents.length).toBe(1);

		const parsed = JSON.parse(toolEvents[0].data!);
		expect(parsed.type).toBe("tool_result");
		expect(parsed.name).toBe("create_exercise");
		expect(parsed.componentsJson).toBeDefined();
	});
});
