import { eq, asc } from "drizzle-orm";
import type { ServiceContext, Bindings } from "../lib/types";
import { pieces } from "../db/schema/catalog";
import { NotFoundError } from "../lib/errors";

export async function listPieces(ctx: ServiceContext, composer?: string) {
	const rows = await ctx.db
		.select()
		.from(pieces)
		.where(composer !== undefined ? eq(pieces.composer, composer) : undefined)
		.orderBy(asc(pieces.composer), asc(pieces.title));

	return rows;
}

export async function getPiece(ctx: ServiceContext, pieceId: string) {
	const row = await ctx.db
		.select()
		.from(pieces)
		.where(eq(pieces.pieceId, pieceId))
		.limit(1);

	if (row.length === 0) {
		throw new NotFoundError("piece", pieceId);
	}

	return row[0];
}

export async function getPieceData(env: Bindings, pieceId: string) {
	const object = await env.SCORES.get(`scores/v1/${pieceId}.json`);

	if (object === null) {
		throw new NotFoundError("piece data", pieceId);
	}

	return object;
}
