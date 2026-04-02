import { createMiddleware } from "hono/factory";
import { HTTPException } from "hono/http-exception";
import { createDb } from "../db/client";
import { createAuth } from "../lib/auth";
import type { Bindings, Variables } from "../lib/types";

export const authSessionMiddleware = createMiddleware<{
	Bindings: Bindings;
	Variables: Variables;
}>(async (c, next) => {
	const db = createDb(c.env.HYPERDRIVE);
	const auth = createAuth(db, c.env);
	const session = await auth.api.getSession({ headers: c.req.raw.headers });
	c.set("studentId", session?.user?.id ?? null);
	await next();
});

export function requireAuth(
	studentId: string | null,
): asserts studentId is string {
	if (!studentId) {
		throw new HTTPException(401, { message: "Authentication required" });
	}
}
