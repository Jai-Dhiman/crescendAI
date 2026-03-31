import { Hono } from "hono";
import type { Bindings, Variables } from "../lib/types";
import { createAuth } from "../lib/auth";
import { createDb } from "../db/client";

const auth = new Hono<{ Bindings: Bindings; Variables: Variables }>();

auth.all("/*", async (c) => {
	const db = createDb(c.env.HYPERDRIVE);
	const authInstance = createAuth(db, c.env);
	return authInstance.handler(c.req.raw);
});

export { auth as authRoutes };
