import type { PostgresJsDatabase } from "drizzle-orm/postgres-js";
import type * as schema from "../db/schema/index";

export type Db = PostgresJsDatabase<typeof schema>;

export interface Bindings {
	HYPERDRIVE: Hyperdrive;
	CHUNKS: R2Bucket;
	SCORES: R2Bucket;
	ENVIRONMENT: string;
	ALLOWED_ORIGIN: string;
	APPLE_BUNDLE_ID: string;
	APPLE_WEB_SERVICES_ID: string;
	GOOGLE_CLIENT_ID: string;
	BETTER_AUTH_URL: string;
	AUTH_SECRET: string;
	APPLE_CLIENT_SECRET: string;
	GOOGLE_CLIENT_SECRET: string;
	SENTRY_DSN: string;
	// Unified authenticated AI Gateway: base URL (…/<account>/<gateway>) with the
	// provider path appended per call (/anthropic/v1/messages, /workers-ai/...).
	AI_GATEWAY_ENDPOINT: string;
	// Gateway-auth token for the authenticated gateway (cf-aig-authorization).
	// Provider keys are injected by the gateway's BYOK vault, not sent here.
	AI_GATEWAY_TOKEN: string;
	// Retained for rollback during the BYOK migration; not sent once the
	// Anthropic key is vaulted. Safe to delete after the gateway is confirmed.
	ANTHROPIC_API_KEY: string;
	CLOUDFLARE_API_TOKEN: string;
	MUQ_ENDPOINT: string;
	AMT_ENDPOINT: string;
	SESSION_BRAIN: DurableObjectNamespace;
	HARNESS_V6_ENABLED: string;
	ALLOW_EVAL_STUDENT_OVERRIDE: string;
	EVAL_SHARED_SECRET: string;
}

export interface Variables {
	db: Db;
	studentId: string | null;
}

export interface ServiceContext {
	db: Db;
	env: Bindings;
}
