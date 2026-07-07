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

// Render assets live under scores/v1/ in three tiers, served in preference order:
// PD-clean Verovio MEI (.mei) and legacy ASAP MusicXML (.mxl) are both Verovio-
// rendered (interactive: per-bar highlight + clip playback); LilyPond-engraved
// Mutopia .svg is a pre-rendered, display-only fallback (CC-BY-SA, LOCAL-ONLY --
// never seeded to prod R2). The score-worker detects the SVG and skips Verovio.
const SCORE_FORMATS = [
	{ ext: "mei", contentType: "application/mei+xml" },
	{ ext: "mxl", contentType: "application/vnd.recordare.musicxml+zip" },
	{ ext: "svg", contentType: "image/svg+xml" },
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
