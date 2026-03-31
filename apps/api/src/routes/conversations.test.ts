import { describe, it, expect } from "vitest";
import { Hono } from "hono";
import { conversationsRoutes } from "./conversations";

const testApp = new Hono().route("/api/conversations", conversationsRoutes);

describe("conversations routes", () => {
	it("GET /api/conversations returns 401 without auth", async () => {
		const res = await testApp.request("/api/conversations");
		expect(res.status).toBe(401);
	});

	it("GET /api/conversations/:id returns 401 without auth", async () => {
		const res = await testApp.request(
			"/api/conversations/00000000-0000-0000-0000-000000000001",
		);
		expect(res.status).toBe(401);
	});

	it("DELETE /api/conversations/:id returns 401 without auth", async () => {
		const res = await testApp.request(
			"/api/conversations/00000000-0000-0000-0000-000000000001",
			{ method: "DELETE" },
		);
		expect(res.status).toBe(401);
	});
});
