import { Hono } from "hono";
import type { Bindings, Variables } from "../lib/types";
import { requireAuth } from "../middleware/auth-session";
import { NotFoundError, ValidationError } from "../lib/errors";
import * as segmentLoopsService from "../services/segment-loops";

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>();

app.post("/:id/accept", async (c) => {
  requireAuth(c.var.studentId);
  const studentId = c.var.studentId!;
  const id = c.req.param("id");
  try {
    const artifact = await segmentLoopsService.acceptSegmentLoop(c.var.db, id, studentId);
    return c.json(artifact);
  } catch (err) {
    if (err instanceof NotFoundError) return c.json({ error: err.message }, 404);
    if (err instanceof ValidationError) return c.json({ error: err.message, code: "invalid_state" }, 409);
    throw err;
  }
});

app.post("/:id/decline", async (c) => {
  requireAuth(c.var.studentId);
  const studentId = c.var.studentId!;
  const id = c.req.param("id");
  try {
    const artifact = await segmentLoopsService.declineSegmentLoop(c.var.db, id, studentId);
    return c.json(artifact);
  } catch (err) {
    if (err instanceof NotFoundError) return c.json({ error: err.message }, 404);
    if (err instanceof ValidationError) return c.json({ error: err.message, code: "invalid_state" }, 409);
    throw err;
  }
});

app.post("/:id/dismiss", async (c) => {
  requireAuth(c.var.studentId);
  const studentId = c.var.studentId!;
  const id = c.req.param("id");
  try {
    const artifact = await segmentLoopsService.dismissSegmentLoop(c.var.db, id, studentId);
    return c.json(artifact);
  } catch (err) {
    if (err instanceof NotFoundError) return c.json({ error: err.message }, 404);
    if (err instanceof ValidationError) return c.json({ error: err.message, code: "invalid_state" }, 409);
    throw err;
  }
});

export { app as segmentLoopsRoutes };
