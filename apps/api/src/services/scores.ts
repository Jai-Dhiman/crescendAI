import { asc, eq } from "drizzle-orm";
import { pieces } from "../db/schema/catalog";
import { NotFoundError } from "../lib/errors";
import type { Bindings, ServiceContext } from "../lib/types";

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

// Render assets live under scores/v1/ in two formats: PD-clean Verovio MEI
// (.mei, preferred) and legacy ASAP MusicXML (.mxl). Verovio renders both, so
// we prefer the clean MEI and fall back to the MusicXML zip.
const SCORE_FORMATS = [
	{ ext: "mei", contentType: "application/mei+xml" },
	{ ext: "mxl", contentType: "application/vnd.recordare.musicxml+zip" },
] as const;

export async function getPieceData(env: Bindings, pieceId: string) {
	for (const { ext, contentType } of SCORE_FORMATS) {
		const object = await env.SCORES.get(`scores/v1/${pieceId}.${ext}`);
		if (object !== null) {
			return { object, contentType };
		}
	}

	throw new NotFoundError("piece data", pieceId);
}
