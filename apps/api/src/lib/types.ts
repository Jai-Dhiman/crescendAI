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
	HF_INFERENCE_ENDPOINT: string;
	ANTHROPIC_API_KEY: string;
	AI_GATEWAY_TEACHER: string;
	AI_GATEWAY_BACKGROUND: string;
	CLOUDFLARE_API_TOKEN: string;
	MUQ_ENDPOINT: string;
	AMT_ENDPOINT: string;
	SESSION_BRAIN: DurableObjectNamespace;
	HARNESS_V6_ENABLED: string;
	HARNESS_V6_CHAT_ENABLED: string;
}

export interface Variables {
	db: Db;
	studentId: string | null;
}

export interface ServiceContext {
	db: Db;
	env: Bindings;
}
