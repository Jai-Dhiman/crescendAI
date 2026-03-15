import { describe, test, expect, beforeAll } from "bun:test";

const BASE = "http://localhost:8787";
let cookie = "";

beforeAll(async () => {
	// Authenticate via debug endpoint (dev-only, returns JWT cookie)
	const res = await fetch(`${BASE}/api/auth/debug`, {
		method: "POST",
		credentials: "include",
	});
	expect(res.ok).toBe(true);
	const setCookie = res.headers.get("set-cookie");
	expect(setCookie).toBeTruthy();
	// Extract just the cookie name=value pair
	cookie = setCookie!.split(";")[0];
});

function authedFetch(path: string, options: RequestInit = {}) {
	return fetch(`${BASE}${path}`, {
		...options,
		headers: {
			"Content-Type": "application/json",
			Cookie: cookie,
			...options.headers,
		},
	});
}

describe("GET /api/exercises", () => {
	test("returns exercises without filters", async () => {
		const res = await authedFetch("/api/exercises");
		expect(res.ok).toBe(true);
		const data = (await res.json()) as { exercises: unknown[] };
		expect(data.exercises).toBeDefined();
		expect(data.exercises.length).toBeGreaterThan(0);
		expect(data.exercises.length).toBeLessThanOrEqual(3);
	});

	test("filters by dimension", async () => {
		const res = await authedFetch("/api/exercises?dimension=dynamics");
		expect(res.ok).toBe(true);
		const data = (await res.json()) as {
			exercises: Array<{ dimensions: string[] }>;
		};
		for (const exercise of data.exercises) {
			expect(exercise.dimensions).toContain("dynamics");
		}
	});

	test("filters by level", async () => {
		const res = await authedFetch("/api/exercises?level=beginner");
		expect(res.ok).toBe(true);
		const data = (await res.json()) as {
			exercises: Array<{ difficulty: string }>;
		};
		for (const exercise of data.exercises) {
			expect(exercise.difficulty).toBe("beginner");
		}
	});

	test("combined dimension and level filter", async () => {
		const res = await authedFetch(
			"/api/exercises?dimension=dynamics&level=intermediate",
		);
		expect(res.ok).toBe(true);
		const data = (await res.json()) as {
			exercises: Array<{ difficulty: string; dimensions: string[] }>;
		};
		for (const exercise of data.exercises) {
			expect(exercise.difficulty).toBe("intermediate");
			expect(exercise.dimensions).toContain("dynamics");
		}
	});

	test("requires auth", async () => {
		const res = await fetch(`${BASE}/api/exercises`);
		expect(res.status).toBe(401);
	});
});

describe("POST /api/exercises/assign", () => {
	test("assigns an exercise", async () => {
		const res = await authedFetch("/api/exercises/assign", {
			method: "POST",
			body: JSON.stringify({ exercise_id: "ex-dyn-001" }),
		});
		expect(res.status).toBe(201);
		const data = (await res.json()) as {
			id: string;
			exercise_id: string;
			completed: boolean;
			times_assigned: number;
		};
		expect(data.id).toMatch(/^se-/);
		expect(data.exercise_id).toBe("ex-dyn-001");
		expect(data.completed).toBe(false);
		expect(data.times_assigned).toBe(1);
	});

	test("excludes assigned exercise from GET results", async () => {
		// ex-dyn-001 was assigned above, so it should not appear in results
		const res = await authedFetch("/api/exercises?dimension=dynamics");
		expect(res.ok).toBe(true);
		const data = (await res.json()) as {
			exercises: Array<{ id: string }>;
		};
		const ids = data.exercises.map((e) => e.id);
		expect(ids).not.toContain("ex-dyn-001");
	});

	test("increments times_assigned on re-assignment", async () => {
		const res = await authedFetch("/api/exercises/assign", {
			method: "POST",
			body: JSON.stringify({
				exercise_id: "ex-dyn-001",
				session_id: "sess-2",
			}),
		});
		expect(res.status).toBe(201);
		const data = (await res.json()) as { times_assigned: number };
		expect(data.times_assigned).toBe(2);
	});
});

describe("POST /api/exercises/complete", () => {
	let studentExerciseId: string;

	test("assign exercise for completion test", async () => {
		const res = await authedFetch("/api/exercises/assign", {
			method: "POST",
			body: JSON.stringify({
				exercise_id: "ex-tim-001",
				session_id: "sess-complete",
			}),
		});
		expect(res.status).toBe(201);
		const data = (await res.json()) as { id: string };
		studentExerciseId = data.id;
	});

	test("completes an exercise", async () => {
		const res = await authedFetch("/api/exercises/complete", {
			method: "POST",
			body: JSON.stringify({
				student_exercise_id: studentExerciseId,
				response: "positive",
				notes: "felt good",
			}),
		});
		expect(res.ok).toBe(true);
		const data = (await res.json()) as {
			completed: boolean;
			response: string;
		};
		expect(data.completed).toBe(true);
		expect(data.response).toBe("positive");
	});

	test("returns 404 for non-existent record", async () => {
		const res = await authedFetch("/api/exercises/complete", {
			method: "POST",
			body: JSON.stringify({
				student_exercise_id: "se-nonexistent",
			}),
		});
		expect(res.status).toBe(404);
	});
});
