import { Hono } from "hono";
import { createDb } from "../db/client";
import { createAuth } from "../lib/auth";
import type { Bindings, Variables } from "../lib/types";

const auth = new Hono<{ Bindings: Bindings; Variables: Variables }>();

auth.post("/debug", async (c) => {
	if (c.env.ENVIRONMENT === "production") {
		return c.json({ error: "Not available in production" }, 403);
	}

	const db = createDb(c.env.HYPERDRIVE);
	const authInstance = createAuth(db, c.env);

	const debugEmail = "debug@crescend.ai";
	const debugPassword = "debug-local-only";

	// Try sign in first, fall back to sign up if user doesn't exist
	let response = await authInstance.api.signInEmail({
		body: { email: debugEmail, password: debugPassword },
		asResponse: true,
	});

	if (!response.ok) {
		response = await authInstance.api.signUpEmail({
			body: {
				email: debugEmail,
				password: debugPassword,
				name: "Debug User",
			},
			asResponse: true,
		});
	}

	if (!response.ok) {
		const text = await response.text();
		throw new Error(`Debug login failed: ${text}`);
	}

	const data = (await response.clone().json()) as {
		user?: { id: string; email: string; name: string };
	};

	// Build our response with better-auth's Set-Cookie headers
	const result = {
		studentId: data.user?.id ?? "",
		email: data.user?.email ?? debugEmail,
		displayName: data.user?.name ?? "Debug User",
	};

	const res = c.json(result);
	const setCookie = response.headers.get("set-cookie");
	if (setCookie) {
		res.headers.set("set-cookie", setCookie);
	}
	return res;
});

auth.all("/*", async (c) => {
	const db = createDb(c.env.HYPERDRIVE);
	const authInstance = createAuth(db, c.env);
	return authInstance.handler(c.req.raw);
});

export { auth as authRoutes };
