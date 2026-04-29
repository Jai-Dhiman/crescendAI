import { and, eq, isNull } from "drizzle-orm";
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "../db/schema/index";
import { parseTitleFields } from "../services/catalog-parse";

const DATABASE_URL = process.env.DATABASE_URL;
if (!DATABASE_URL) {
	throw new Error("DATABASE_URL environment variable is required");
}

const sql = postgres(DATABASE_URL);
const db = drizzle(sql, { schema });
const { pieces } = schema;

try {
	// Only fetch pieces that haven't been backfilled yet (idempotent)
	const rows = await db
		.select({ pieceId: pieces.pieceId, title: pieces.title })
		.from(pieces)
		.where(
			and(
				isNull(pieces.opusNumber),
				isNull(pieces.pieceNumber),
				isNull(pieces.catalogueType),
			),
		);

	console.log(
		JSON.stringify({ message: "backfill starting", total: rows.length }),
	);

	let updated = 0;
	let skipped = 0;

	for (const row of rows) {
		const fields = parseTitleFields(row.title);

		if (
			fields.opusNumber === null &&
			fields.pieceNumber === null &&
			fields.catalogueType === null
		) {
			skipped++;
			continue;
		}

		await db
			.update(pieces)
			.set({
				opusNumber: fields.opusNumber,
				pieceNumber: fields.pieceNumber,
				catalogueType: fields.catalogueType,
			})
			.where(eq(pieces.pieceId, row.pieceId));

		updated++;
		console.log(
			JSON.stringify({
				pieceId: row.pieceId,
				title: row.title,
				opusNumber: fields.opusNumber,
				pieceNumber: fields.pieceNumber,
				catalogueType: fields.catalogueType,
			}),
		);
	}

	console.log(
		JSON.stringify({ message: "backfill complete", updated, skipped }),
	);
} finally {
	await sql.end();
}
