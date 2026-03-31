import { describe, it, expect } from "vitest";
import { Hono } from "hono";
import { chatRoutes } from "./chat";

const testApp = new Hono().route("/api/chat", chatRoutes);

describe("POST /api/chat", () => {
	it("returns 401 without auth", async () => {
		const res = await testApp.request("/api/chat", {
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
});
