import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "./schema/index";

export function createDb(hyperdrive: Hyperdrive) {
	// max: 5 is Cloudflare's recommended cap for Workers with Hyperdrive (docs: hyperdrive/use-postgres-js).
	// idle_timeout: 10 releases idle connections after 10 seconds, preventing pool exhaustion
	// in wrangler dev where connections go directly to Postgres (no Hyperdrive proxy pooling).
	// In production, Hyperdrive manages the real connection pool; these limits apply to the
	// per-Worker client and are safe to set conservatively.
	const sql = postgres(hyperdrive.connectionString, {
		max: 5,
		idle_timeout: 10,
		fetch_types: false,
	});
	return drizzle(sql, { schema });
}
