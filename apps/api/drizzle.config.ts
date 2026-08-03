import { defineConfig } from "drizzle-kit";

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
