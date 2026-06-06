import { Hono } from "hono";
import { describe, expect, it, vi } from "vitest";
import { errorHandler } from "../middleware/error-handler";
import * as exercisesService from "../services/exercises";
import { exercisesRoutes } from "./exercises";

const SESSION_ID = "00000000-0000-0000-0000-000000000010";
const EXERCISE_ID = "00000000-0000-0000-0000-000000000001";

const noAuthApp = new Hono()
	.onError(errorHandler)
	.route("/api/exercises", exercisesRoutes);

function makeAuthApp(studentId: string) {
	return new Hono()
		.use("*", async (c, next) => {
			c.set("studentId", studentId);
			c.set("db", {} as never);
			await next();
		})
		.onError(errorHandler)
		.route("/api/exercises", exercisesRoutes);
}

describe("exercises routes — auth gating", () => {
	it("GET /api/exercises returns 401 without auth", async () => {
		const res = await noAuthApp.request("/api/exercises");
		expect(res.status).toBe(401);
	});

	it("POST /api/exercises/assign returns 401 without auth", async () => {
		const res = await noAuthApp.request("/api/exercises/assign", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ exerciseId: EXERCISE_ID }),
		});
		expect(res.status).toBe(401);
	});

	it("POST /api/exercises/complete returns 401 without auth", async () => {
		const res = await noAuthApp.request("/api/exercises/complete", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ studentExerciseId: EXERCISE_ID }),
		});
		expect(res.status).toBe(401);
	});

	it("POST /api/exercises/assign-pending returns 401 without auth", async () => {
		const res = await noAuthApp.request("/api/exercises/assign-pending", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ sessionId: SESSION_ID, exerciseId: EXERCISE_ID }),
		});
		expect(res.status).toBe(401);
	});
});

describe("POST /api/exercises/assign-pending — authenticated", () => {
	it("returns 200 with ExerciseSetPayload on valid owned pending row", async () => {
		const payload = {
			sourcePassage: "Running passage bars 3-6",
			targetSkill: "pedaling",
			exercises: [
				{
					title: "Pedal Separation Drill",
					instruction: "Play with clean pedal lifts.",
					focusDimension: "pedaling",
					exerciseId: EXERCISE_ID,
				},
			],
		};
		vi.spyOn(exercisesService, "assignPendingExercise").mockResolvedValueOnce(
			payload,
		);

		const res = await makeAuthApp("student-abc").request(
			"/api/exercises/assign-pending",
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					sessionId: SESSION_ID,
					exerciseId: EXERCISE_ID,
				}),
			},
		);

		expect(res.status).toBe(200);
		expect(await res.json()).toMatchObject(payload);
	});

	it("returns 404 when assignPendingExercise throws NotFoundError (IDOR: foreign exerciseId)", async () => {
		const { NotFoundError } = await import("../lib/errors");
		vi.spyOn(exercisesService, "assignPendingExercise").mockRejectedValueOnce(
			new NotFoundError(
				"pending exercise",
				"00000000-0000-0000-0000-000000000099",
			),
		);

		const res = await makeAuthApp("student-abc").request(
			"/api/exercises/assign-pending",
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					sessionId: SESSION_ID,
					exerciseId: "00000000-0000-0000-0000-000000000099",
				}),
			},
		);

		expect(res.status).toBe(404);
		expect(await res.json()).toHaveProperty("error");
	});
});
