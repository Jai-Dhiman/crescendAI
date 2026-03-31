import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import * as goalsService from "../services/goals";

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>().post(
  "/",
  validate("json", z.object({ message: z.string().min(1).max(5000) })),
  async (c) => {
    requireAuth(c.var.studentId);
    const { message } = c.req.valid("json");
    const ctx = { db: c.var.db, env: c.env };
    const goals = await goalsService.extractGoals(ctx, c.var.studentId, message);
    return c.json({ goals });
  },
);

export { app as goalsRoutes };
