import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import * as schema from "../db/schema/auth";
import { studentProfiles } from "../db/schema/students";
import type { Bindings, Db } from "./types";

export function createAuth(db: Db, env: Bindings) {
	const isProd = env.ENVIRONMENT === "production";

	return betterAuth({
		database: drizzleAdapter(db, {
			provider: "pg",
			schema,
		}),
		baseURL: env.BETTER_AUTH_URL,
		secret: env.AUTH_SECRET,
		trustedOrigins: [env.ALLOWED_ORIGIN],
		emailAndPassword: {
			enabled: !isProd,
		},
		socialProviders: {
			apple: {
				clientId: env.APPLE_WEB_SERVICES_ID,
				clientSecret: env.APPLE_CLIENT_SECRET,
			},
			google: {
				clientId: env.GOOGLE_CLIENT_ID,
				clientSecret: env.GOOGLE_CLIENT_SECRET,
			},
		},
		databaseHooks: {
			user: {
				create: {
					after: async (user) => {
						await db
							.insert(studentProfiles)
							.values({ studentId: user.id })
							.onConflictDoNothing();
					},
				},
			},
		},
		session: {
			expiresIn: 60 * 60 * 24 * 30,
			cookieCache: {
				enabled: false,
			},
		},
		advanced: {
			database: {
				generateId: () => crypto.randomUUID(),
			},
			crossSubDomainCookies: isProd
				? { enabled: true, domain: ".crescend.ai" }
				: { enabled: false },
		},
	});
}

export type Auth = ReturnType<typeof createAuth>;
