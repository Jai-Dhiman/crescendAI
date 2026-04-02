import { Hono } from "hono";
import { describe, expect, it } from "vitest";
import { syncRoutes } from "./sync";

const testApp = new Hono().route("/api/sync", syncRoutes);

describe("POST /api/sync", () => {
	it("returns 401 without auth", async () => {
		const res = await testApp.request("/api/sync", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ student: {}, newSessions: [] }),
		});
		expect(res.status).toBe(401);
	});

	it("returns 400 for invalid body", async () => {
		const res = await testApp.request("/api/sync", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ invalid: true }),
		});
		expect(res.status).toBe(400);
	});
});
