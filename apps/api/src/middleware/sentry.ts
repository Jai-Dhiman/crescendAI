import * as Sentry from "@sentry/cloudflare";
import { createMiddleware } from "hono/factory";
import type { Bindings, Variables } from "../lib/types";

export const sentryMiddleware = createMiddleware<{
	Bindings: Bindings;
	Variables: Variables;
}>(async (c, next) => {
	Sentry.setContext("request", {
		method: c.req.method,
		url: c.req.url,
		path: c.req.path,
	});
	try {
		await next();
	} catch (err) {
		Sentry.captureException(err);
		throw err;
	}
});
