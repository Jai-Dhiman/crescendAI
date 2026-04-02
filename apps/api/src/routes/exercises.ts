import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import {
	assignExercise,
	completeExercise,
	listExercises,
} from "../services/exercises";

const assignSchema = z.object({
	exerciseId: z.string().uuid(),
	sessionId: z.string().uuid().optional(),
});

const completeSchema = z.object({
	studentExerciseId: z.string().uuid(),
	response: z.string().optional(),
	dimensionBeforeJson: z.unknown().optional(),
	dimensionAfterJson: z.unknown().optional(),
	notes: z.string().optional(),
});

const listQuerySchema = z.object({
	dimension: z.string().optional(),
	level: z.string().optional(),
	repertoire: z.string().optional(),
});

const exercisesRoutes = new Hono<{ Bindings: Bindings; Variables: Variables }>()
	.get("/", validate("query", listQuerySchema), async (c) => {
		requireAuth(c.var.studentId);
		const { dimension, level, repertoire } = c.req.valid("query");
		const result = await listExercises(
			{ db: c.var.db, env: c.env },
			{ studentId: c.var.studentId, dimension, level, repertoire },
		);
		return c.json(result);
	})
	.post("/assign", validate("json", assignSchema), async (c) => {
		requireAuth(c.var.studentId);
		const body = c.req.valid("json");
		const result = await assignExercise(
			{ db: c.var.db, env: c.env },
			{ studentId: c.var.studentId, ...body },
		);
		return c.json(result, 201);
	})
	.post("/complete", validate("json", completeSchema), async (c) => {
		requireAuth(c.var.studentId);
		const body = c.req.valid("json");
		const result = await completeExercise(
			{ db: c.var.db, env: c.env },
			{ studentId: c.var.studentId, ...body },
		);
		return c.json(result);
	});

export { exercisesRoutes };
