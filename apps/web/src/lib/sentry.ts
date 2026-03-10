import * as Sentry from "@sentry/react";

Sentry.init({
	dsn: import.meta.env.VITE_SENTRY_DSN,
	environment: import.meta.env.PROD ? "production" : "development",
	tracesSampleRate: 0.1,
});

export { Sentry };
