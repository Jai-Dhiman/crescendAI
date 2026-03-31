import { Hono } from "hono";
import { cors } from "hono/cors";
import * as Sentry from "@sentry/cloudflare";
import type { Bindings, Variables } from "./lib/types";
import { dbMiddleware } from "./middleware/db";
import { authSessionMiddleware } from "./middleware/auth-session";
import { structuredLogger } from "./middleware/logger";
import { sentryMiddleware } from "./middleware/sentry";
import { errorHandler } from "./middleware/error-handler";
import { authRoutes } from "./routes/auth";
import { healthRoutes } from "./routes/health";
import { waitlistRoutes } from "./routes/waitlist";
import { scoresRoutes } from "./routes/scores";
import { exercisesRoutes } from "./routes/exercises";
import { conversationsRoutes } from "./routes/conversations";
import { syncRoutes } from "./routes/sync";
import { chatRoutes } from "./routes/chat";

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>();

app.onError(errorHandler);

app.use("*", async (c, next) => {
	const origin = c.env.ALLOWED_ORIGIN || "http://localhost:3000";
	return cors({
		origin,
		allowMethods: ["GET", "POST", "OPTIONS", "DELETE"],
		allowHeaders: ["Content-Type", "Authorization", "Cookie"],
		credentials: true,
	})(c, next);
});

app.use("*", structuredLogger);
app.use("*", sentryMiddleware);
app.use("/api/*", dbMiddleware);
app.use("/api/*", authSessionMiddleware);

const routes = app
	.route("/health", healthRoutes)
	.route("/api/auth", authRoutes)
	.route("/api/waitlist", waitlistRoutes)
	.route("/api/scores", scoresRoutes)
	.route("/api/exercises", exercisesRoutes)
	.route("/api/conversations", conversationsRoutes)
	.route("/api/sync", syncRoutes)
	.route("/api/chat", chatRoutes);

app.notFound((c) => c.json({ error: "Not found" }, 404));

export type AppType = typeof routes;
export { app };

export default Sentry.withSentry(
	(env) => ({
		dsn: env.SENTRY_DSN,
		tracesSampleRate: 1.0,
	}),
	app,
);
