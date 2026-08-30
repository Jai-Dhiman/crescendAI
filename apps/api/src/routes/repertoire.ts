import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import * as repertoireService from "../services/repertoire";

const pieceIdParamSchema = z.object({ pieceId: z.string().min(1) });

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>()
	.get("/", async (c) => {
		requireAuth(c.var.studentId);
		const entries = await repertoireService.listRepertoire(
			{ db: c.var.db, env: c.env },
			c.var.studentId,
		);
		return c.json({ entries });
	})
	.get("/:pieceId", validate("param", pieceIdParamSchema), async (c) => {
		requireAuth(c.var.studentId);
		const { pieceId } = c.req.valid("param");
		const history = await repertoireService.listPieceHistory(
			{ db: c.var.db, env: c.env },
			c.var.studentId,
			pieceId,
		);
		return c.json(history);
	});

export { app as repertoireRoutes };
