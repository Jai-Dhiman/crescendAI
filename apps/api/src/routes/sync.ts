import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import * as syncService from "../services/sync";

const studentDeltaSchema = z.object({
	inferredLevel: z.string().optional(),
	baselineDynamics: z.number().optional(),
	baselineTiming: z.number().optional(),
	baselinePedaling: z.number().optional(),
	baselineArticulation: z.number().optional(),
	baselinePhrasing: z.number().optional(),
	baselineInterpretation: z.number().optional(),
	baselineSessionCount: z.number().optional(),
});

const sessionDeltaSchema = z.object({
	id: z.string().uuid(),
	startedAt: z.string(),
	endedAt: z.string().optional(),
	avgDynamics: z.number().optional(),
	avgTiming: z.number().optional(),
	avgPedaling: z.number().optional(),
	avgArticulation: z.number().optional(),
	avgPhrasing: z.number().optional(),
	avgInterpretation: z.number().optional(),
	observationsJson: z.unknown().optional(),
	chunksSummaryJson: z.unknown().optional(),
});

const syncSchema = z.object({
	student: studentDeltaSchema,
	newSessions: z.array(sessionDeltaSchema),
	lastSyncTimestamp: z.string().optional(),
});

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>().post(
	"/",
	validate("json", syncSchema),
	async (c) => {
		const studentId = c.var.studentId;
		requireAuth(studentId);
		const body = c.req.valid("json");
		const result = await syncService.handleSync(
			{ db: c.var.db, env: c.env },
			studentId,
			body,
		);
		return c.json(result, 200);
	},
);

export { app as syncRoutes };
