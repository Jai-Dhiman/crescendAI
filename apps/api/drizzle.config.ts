import { defineConfig } from "drizzle-kit";

// The postgres npm package (used by drizzle-kit) passes URL query params as
// Postgres startup parameters. Supabase connection strings include sslrootcert
// which Postgres rejects as unknown — strip it before passing to the driver.
const rawUrl = process.env.DATABASE_URL ?? "";
const migrateUrl = rawUrl.replace(/[?&]sslrootcert=[^&]*/g, "").replace(/\?&/, "?");

export default defineConfig({
	dialect: "postgresql",
	schema: "./src/db/schema/index.ts",
	out: "./src/db/migrations",
	dbCredentials: {
		url: migrateUrl,
	},
});
