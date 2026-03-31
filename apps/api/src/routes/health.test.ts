import { describe, it, expect } from "vitest";
import { Hono } from "hono";
import { healthRoutes } from "./health";

// Test health routes directly without the Sentry wrapper.
// Sentry.withSentry proxies app.fetch and requires a real Worker env (not
// available when using app.request() in unit tests). Testing the route module
// directly avoids the env-injection gap.
const testApp = new Hono()
	.route("/health", healthRoutes)
	.notFound((c) => c.json({ error: "Not found" }, 404));

describe("GET /health", () => {
	it("returns ok status", async () => {
		const res = await testApp.request("/health");
		expect(res.status).toBe(200);

		const body = await res.json();
		expect(body).toEqual({
			status: "ok",
			version: "2.0.0",
			stack: "hono",
		});
	});

	it("returns 404 for unknown routes", async () => {
		const res = await testApp.request("/nonexistent");
		expect(res.status).toBe(404);

		const body = await res.json();
		expect(body).toEqual({ error: "Not found" });
	});
});
