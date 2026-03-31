import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import * as waitlistService from "../services/waitlist";

const waitlistSchema = z.object({
	email: z.string().email(),
	context: z.string().optional(),
	source: z.string().optional(),
	website: z.string().optional(), // honeypot
});

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>().post(
	"/",
	validate("json", waitlistSchema),
	async (c) => {
		const body = c.req.valid("json");
		if (body.website) {
			return c.json({ success: true });
		}
		const result = await waitlistService.addToWaitlist(
			{ db: c.var.db, env: c.env },
			body,
		);
		return c.json(result, 201);
	},
);

export { app as waitlistRoutes };
