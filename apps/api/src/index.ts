import * as Sentry from "@sentry/cloudflare";
import { Hono } from "hono";
import { cors } from "hono/cors";
import type { Bindings, Variables } from "./lib/types";
import { authSessionMiddleware } from "./middleware/auth-session";
import { dbMiddleware } from "./middleware/db";
import { errorHandler } from "./middleware/error-handler";
import { structuredLogger } from "./middleware/logger";
import { sentryMiddleware } from "./middleware/sentry";
import { authRoutes } from "./routes/auth";
import { chatRoutes } from "./routes/chat";
import { conversationsRoutes } from "./routes/conversations";
import { exercisesRoutes } from "./routes/exercises";
import { goalsRoutes } from "./routes/goals";
import { healthRoutes } from "./routes/health";
import { practiceRoutes } from "./routes/practice";
import { scoresRoutes } from "./routes/scores";
import { syncRoutes } from "./routes/sync";
import { waitlistRoutes } from "./routes/waitlist";

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
	.route("/api/chat", chatRoutes)
	.route("/api/practice", practiceRoutes)
	.route("/api/extract-goals", goalsRoutes);

app.notFound((c) => c.json({ error: "Not found" }, 404));

export type AppType = typeof routes;
export { app };
import { SessionBrain as _SessionBrain } from "./do/session-brain";
export const SessionBrain = Sentry.instrumentDurableObjectWithSentry(
	(env: Bindings) => ({
		dsn: env.SENTRY_DSN,
		tracesSampleRate: 1.0,
	}),
	_SessionBrain,
);

export default Sentry.withSentry(
	(env: Bindings) => ({
		dsn: env.SENTRY_DSN,
		tracesSampleRate: 1.0,
	}),
	app,
);
