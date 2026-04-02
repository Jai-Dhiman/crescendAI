import { zValidator } from "@hono/zod-validator";
import type { z } from "zod";

/**
 * Typed Zod validator that returns JSON error responses.
 * Wraps @hono/zod-validator with a hook that formats validation errors as JSON.
 */
export function validate<T extends z.ZodSchema>(
	target: "json" | "query" | "param",
	schema: T,
) {
	return zValidator(target, schema, (result, c) => {
		if (!result.success) {
			console.log(JSON.stringify({
				level: "warn",
				message: "Zod validation failed",
				target,
				issues: result.error.issues,
				path: c.req.path,
				method: c.req.method,
			}));
			return c.json(
				{ error: "Validation failed", issues: result.error.issues },
				400,
			);
		}
	});
}
