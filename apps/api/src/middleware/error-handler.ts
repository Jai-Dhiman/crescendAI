import type { ErrorHandler } from "hono";
import { HTTPException } from "hono/http-exception";
import {
	AuthenticationError,
	ConflictError,
	ForbiddenError,
	NotFoundError,
	ValidationError,
} from "../lib/errors";

export const errorHandler: ErrorHandler = (err, c) => {
	if (err instanceof HTTPException) {
		return err.getResponse();
	}

	if (err instanceof NotFoundError) {
		return c.json({ error: err.message }, 404);
	}

	if (err instanceof AuthenticationError) {
		return c.json({ error: err.message }, 401);
	}

	if (err instanceof ValidationError) {
		return c.json({ error: err.message }, 400);
	}

	if (err instanceof ConflictError) {
		return c.json({ error: err.message }, 409);
	}

	if (err instanceof ForbiddenError) {
		return c.json({ error: err.message }, 403);
	}

	console.error(
		JSON.stringify({
			level: "error",
			message: err.message,
			stack: err.stack,
			name: err.name,
		}),
	);

	return c.json({ error: "Internal server error" }, 500);
};
