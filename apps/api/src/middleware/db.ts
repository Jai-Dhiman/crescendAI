import { createMiddleware } from "hono/factory";
import { createDb } from "../db/client";
import type { Bindings, Variables } from "../lib/types";

export const dbMiddleware = createMiddleware<{
	Bindings: Bindings;
	Variables: Variables;
}>(async (c, next) => {
	const db = createDb(c.env.HYPERDRIVE);
	c.set("db", db);
	await next();
});
