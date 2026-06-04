import { Hono } from "hono";
import { describe, expect, it } from "vitest";
import type { Bindings, Variables } from "../lib/types";
import { errorHandler } from "../middleware/error-handler";
import { practiceRoutes, resolveSessionStudentId } from "./practice";

const testApp = new Hono().route("/api/practice", practiceRoutes);

// Authenticated test app: injects a studentId and stubs the DB.
function makeAuthApp(dbStub: Record<string, unknown>) {
	const app = new Hono<{ Bindings: Bindings; Variables: Variables }>();
	app.use("*", async (c, next) => {
		c.set("studentId", "student-a");
		c.set("db", dbStub);
		await next();
	});
	app.route("/api/practice", practiceRoutes);
	app.onError(errorHandler);
	return app;
}

describe("resolveSessionStudentId", () => {
	// Helper: all-conditions-true (should grant override)
	const grantArgs = {
		isEvalQuery: true,
		evalStudentId: "eval-r42",
		authStudentId: "auth-user-id",
		overrideAllowed: true,
		secretOk: true,
	} as const;

	// --- GRANTED cases ---

	it("grants override when all conditions met (flag=true, secret=ok, evalStudentId present)", () => {
		const result = resolveSessionStudentId(grantArgs);
		expect(result).toBe("eval-r42");
	});

	// --- DENIED cases (fail-closed) ---

	it("denies override when overrideAllowed=false (flag absent/not-true in env)", () => {
		const result = resolveSessionStudentId({
			...grantArgs,
			overrideAllowed: false,
		});
		expect(result).toBe("auth-user-id");
	});

	it("denies override when secretOk=false (wrong or missing x-eval-secret header)", () => {
		const result = resolveSessionStudentId({
			...grantArgs,
			secretOk: false,
		});
		expect(result).toBe("auth-user-id");
	});

	it("denies override when EVAL_SHARED_SECRET is unset (secretOk=false)", () => {
		// Simulates empty/unset EVAL_SHARED_SECRET binding: secretOk computed false at call site
		const result = resolveSessionStudentId({
			...grantArgs,
			secretOk: false,
		});
		expect(result).toBe("auth-user-id");
	});

	it("denies override when evalStudentId is empty even with flag+secret", () => {
		const result = resolveSessionStudentId({
			...grantArgs,
			evalStudentId: "",
		});
		expect(result).toBe("auth-user-id");
	});

	it("denies override when isEvalQuery=false even with flag+secret", () => {
		const result = resolveSessionStudentId({
			...grantArgs,
			isEvalQuery: false,
		});
		expect(result).toBe("auth-user-id");
	});

	it("denies IDOR: flag+secret both false cannot impersonate victim studentId", () => {
		const result = resolveSessionStudentId({
			isEvalQuery: true,
			evalStudentId: "victim-id",
			authStudentId: "real-auth-id",
			overrideAllowed: false,
			secretOk: false,
		});
		expect(result).not.toBe("victim-id");
		expect(result).toBe("real-auth-id");
	});

	it("denies IDOR: overrideAllowed=true but wrong secret cannot impersonate victim", () => {
		const result = resolveSessionStudentId({
			isEvalQuery: true,
			evalStudentId: "victim-id",
			authStudentId: "real-auth-id",
			overrideAllowed: true,
			secretOk: false,
		});
		expect(result).not.toBe("victim-id");
		expect(result).toBe("real-auth-id");
	});

	it("returns authStudentId (null) when no auth and override denied", () => {
		const result = resolveSessionStudentId({
			isEvalQuery: true,
			evalStudentId: "eval-r42",
			authStudentId: null,
			overrideAllowed: false,
			secretOk: true,
		});
		expect(result).toBeNull();
	});
});

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
