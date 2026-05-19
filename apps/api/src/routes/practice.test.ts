import { Hono } from "hono";
import { describe, expect, it } from "vitest";
import { errorHandler } from "../middleware/error-handler";
import { practiceRoutes } from "./practice";

const testApp = new Hono().route("/api/practice", practiceRoutes);

// Authenticated test app: injects a studentId and stubs the DB.
function makeAuthApp(dbStub: Record<string, unknown>) {
	const app = new Hono();
	app.use("*", async (c, next) => {
		c.set("studentId", "student-a");
		c.set("db", dbStub);
		await next();
	});
	app.route("/api/practice", practiceRoutes);
	app.onError(errorHandler);
	return app;
}

describe("practice routes", () => {
	it("POST /api/practice/start returns 401 without auth", async () => {
		const res = await testApp.request("/api/practice/start", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({}),
		});
		expect(res.status).toBe(401);
	});

	it("POST /api/practice/chunk returns 401 without auth", async () => {
		const res = await testApp.request(
			"/api/practice/chunk?sessionId=00000000-0000-0000-0000-000000000001&chunkIndex=0",
			{
				method: "POST",
				body: new ArrayBuffer(100),
			},
		);
		expect(res.status).toBe(401);
	});

	it("GET /api/practice/ws/:id returns 426 without upgrade header", async () => {
		const res = await testApp.request("/api/practice/ws/test-session");
		// Without Upgrade header, should get 426 (or 401 if auth check first)
		expect([401, 426]).toContain(res.status);
	});

	it("GET /api/practice/needs-synthesis returns 401 without auth", async () => {
		const res = await testApp.request(
			"/api/practice/needs-synthesis?conversationId=00000000-0000-0000-0000-000000000001",
		);
		expect(res.status).toBe(401);
	});

	it("POST /api/practice/synthesize returns 401 without auth", async () => {
		const res = await testApp.request("/api/practice/synthesize", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				sessionId: "00000000-0000-0000-0000-000000000001",
			}),
		});
		expect(res.status).toBe(401);
	});

	it("GET /api/practice/chunk returns 401 without auth", async () => {
		const res = await testApp.request(
			"/api/practice/chunk?sessionId=00000000-0000-0000-0000-000000000001&chunkIndex=0",
		);
		expect(res.status).toBe(401);
	});

	it("GET /api/practice/chunk returns 404 when session belongs to a different student", async () => {
		// DB stub: findFirst returns undefined (session exists but studentId doesn't match)
		const dbStub = {
			query: {
				sessions: {
					findFirst: async () => undefined,
				},
			},
		};
		const app = makeAuthApp(dbStub);
		const res = await app.request(
			"/api/practice/chunk?sessionId=00000000-0000-0000-0000-000000000001&chunkIndex=0",
		);
		expect(res.status).toBe(404);
	});
});
