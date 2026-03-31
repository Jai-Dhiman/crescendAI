import { createMiddleware } from "hono/factory";
import type { Bindings, Variables } from "../lib/types";

export const structuredLogger = createMiddleware<{
	Bindings: Bindings;
	Variables: Variables;
}>(async (c, next) => {
	const start = Date.now();
	await next();
	const duration = Date.now() - start;

	console.log(
		JSON.stringify({
			level: "info",
			method: c.req.method,
			path: c.req.path,
			status: c.res.status,
			duration_ms: duration,
		}),
	);
});
