import { Hono } from "hono";
import { describe, expect, it } from "vitest";
import { waitlistRoutes } from "./waitlist";

const testApp = new Hono().route("/api/waitlist", waitlistRoutes);

describe("POST /api/waitlist", () => {
	it("returns 400 for missing email", async () => {
		const res = await testApp.request("/api/waitlist", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({}),
		});
		expect(res.status).toBe(400);
	});

	it("returns 400 for invalid email", async () => {
		const res = await testApp.request("/api/waitlist", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ email: "not-an-email" }),
		});
		expect(res.status).toBe(400);
	});
});
