import { betterAuth } from "better-auth";
import { drizzleAdapter } from "better-auth/adapters/drizzle";
import type { Db, Bindings } from "./types";

export function createAuth(db: Db, env: Bindings) {
	return betterAuth({
		database: drizzleAdapter(db, {
			provider: "pg",
		}),
		baseURL: env.BETTER_AUTH_URL,
		secret: env.AUTH_SECRET,
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
		session: {
			expiresIn: 60 * 60 * 24 * 30,
			cookieCache: {
				enabled: false,
			},
		},
		advanced: {
			crossSubDomainCookies: {
				enabled: true,
				domain: ".crescend.ai",
			},
		},
	});
}

export type Auth = ReturnType<typeof createAuth>;
