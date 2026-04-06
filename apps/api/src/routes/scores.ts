import { Hono } from "hono";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { getPiece, getPieceData, listPieces } from "../services/scores";

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>()
	.get("/", async (c) => {
		const composer = c.req.query("composer");
		const result = await listPieces({ db: c.var.db, env: c.env }, composer);
		return c.json({ pieces: result });
	})
	.get(
		"/:pieceId",
		validate("param", z.object({ pieceId: z.string().min(1) })),
		async (c) => {
			const { pieceId } = c.req.valid("param");
			const piece = await getPiece({ db: c.var.db, env: c.env }, pieceId);
			return c.json(piece);
		},
	)
	.get(
		"/:pieceId/data",
		validate("param", z.object({ pieceId: z.string().min(1) })),
		async (c) => {
			const { pieceId } = c.req.valid("param");
			const object = await getPieceData(c.env, pieceId);
			return new Response(object.body, {
				headers: {
					"Content-Type": "application/vnd.recordare.musicxml",
					"Cache-Control": "public, max-age=31536000, immutable",
				},
			});
		},
	);

export { app as scoresRoutes };
