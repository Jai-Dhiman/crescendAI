import { Hono } from "hono";
import { describe, expect, it } from "vitest";
import { exercisesRoutes } from "./exercises";

const testApp = new Hono().route("/api/exercises", exercisesRoutes);

describe("exercises routes", () => {
	it("GET /api/exercises returns 401 without auth", async () => {
		const res = await testApp.request("/api/exercises");
		expect(res.status).toBe(401);
	});

	it("POST /api/exercises/assign returns 401 without auth", async () => {
		const res = await testApp.request("/api/exercises/assign", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				exerciseId: "00000000-0000-0000-0000-000000000001",
			}),
		});
		expect(res.status).toBe(401);
	});

	it("POST /api/exercises/complete returns 401 without auth", async () => {
		const res = await testApp.request("/api/exercises/complete", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				studentExerciseId: "00000000-0000-0000-0000-000000000001",
			}),
		});
		expect(res.status).toBe(401);
	});
});
