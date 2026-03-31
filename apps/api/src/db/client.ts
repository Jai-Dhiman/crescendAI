import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "./schema/index";

export function createDb(hyperdrive: Hyperdrive) {
	const sql = postgres(hyperdrive.connectionString);
	return drizzle(sql, { schema });
}
