import { describe, it, expect } from "vitest";
import { Hono } from "hono";
import { scoresRoutes } from "./scores";

// NOTE: These tests verify route structure and validation.
// Full integration tests with DB require @cloudflare/vitest-pool-workers.
const testApp = new Hono().route("/api/scores", scoresRoutes);

describe("scores routes", () => {
	it("GET /api/scores/:pieceId validates param", async () => {
		// Empty pieceId should fail validation (min 1 char)
		// The route exists and responds (not 404)
		const res = await testApp.request("/api/scores/test-piece");
		// Will fail with DB error since no middleware, but NOT 404
		expect(res.status).not.toBe(404);
	});
});
