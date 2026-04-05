import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import {
	deleteConversation,
	getConversation,
	listConversations,
} from "../services/conversations";

const uuidParamSchema = z.object({
	id: z.string().uuid(),
});

const conversationsApp = new Hono<{
	Bindings: Bindings;
	Variables: Variables;
}>()
	.get("/", async (c) => {
		requireAuth(c.var.studentId);
		const result = await listConversations(
			{ db: c.var.db, env: c.env },
			c.var.studentId,
		);
		return c.json({ conversations: result });
	})
	.get("/:id", validate("param", uuidParamSchema), async (c) => {
		requireAuth(c.var.studentId);
		const { id } = c.req.valid("param");
		const result = await getConversation(
			{ db: c.var.db, env: c.env },
			id,
			c.var.studentId,
		);
		return c.json(result);
	})
	.delete("/:id", validate("param", uuidParamSchema), async (c) => {
		requireAuth(c.var.studentId);
		const { id } = c.req.valid("param");
		await deleteConversation({ db: c.var.db, env: c.env }, id, c.var.studentId);
		return c.json({ success: true });
	});

export { conversationsApp as conversationsRoutes };
