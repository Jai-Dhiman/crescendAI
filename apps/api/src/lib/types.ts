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
	AI: Ai;
	// Anthropic is reached through the provider-native gateway URL: BYOK keys are
	// not usable through the unified AI binding, so this plumbing stays.
	AI_GATEWAY_ENDPOINT: string;
	AI_GATEWAY_TOKEN: string;
	// Unified Workers AI / AI Gateway control plane (2026-08-07).
	CLOUDFLARE_ACCOUNT_ID: string;
	AI_GATEWAY_ID: string;
	TEACHER_PROVIDER?: string;
	TEACHER_MODEL?: string;
	CLOUDFLARE_API_TOKEN: string;
	MUQ_ENDPOINT: string;
	AMT_ENDPOINT: string;
	SESSION_BRAIN: DurableObjectNamespace;
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
