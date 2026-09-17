import { Hono } from "hono";
import { describe, expect, it, vi } from "vitest";
import type { Bindings, Db, Variables } from "../lib/types";
import { errorHandler } from "../middleware/error-handler";
import * as repertoireService from "../services/repertoire";
import { repertoireRoutes } from "./repertoire";

const noAuthApp = new Hono()
	.onError(errorHandler)
	.route("/api/repertoire", repertoireRoutes);

function makeAuthApp(studentId: string) {
	return new Hono<{ Bindings: Bindings; Variables: Variables }>()
		.use("*", async (c, next) => {
			c.set("studentId", studentId);
			c.set("db", {} as unknown as Db);
			await next();
		})
		.onError(errorHandler)
		.route("/api/repertoire", repertoireRoutes);
}

describe("repertoire routes", () => {
	it("GET /api/repertoire returns 401 without auth", async () => {
		const res = await noAuthApp.request("/api/repertoire");
		expect(res.status).toBe(401);
	});

	it("GET /api/repertoire/:pieceId returns 401 without auth", async () => {
		const res = await noAuthApp.request("/api/repertoire/p1");
		expect(res.status).toBe(401);
	});

	it("GET /api/repertoire returns the student's entries", async () => {
		vi.spyOn(repertoireService, "listRepertoire").mockResolvedValueOnce([
			{
				pieceId: "p1",
				composer: "Chopin",
				title: "Nocturne Op. 9 No. 2",
				sessionCount: 3,
				lastPracticedAt: "2026-08-25T18:00:00.000Z",
				openMarkCount: null,
			},
		]);

		const res = await makeAuthApp("student-a").request("/api/repertoire");
		expect(res.status).toBe(200);
		expect(await res.json()).toEqual({
			entries: [
				{
					pieceId: "p1",
					composer: "Chopin",
					title: "Nocturne Op. 9 No. 2",
					sessionCount: 3,
					lastPracticedAt: "2026-08-25T18:00:00.000Z",
					openMarkCount: null,
				},
			],
		});
	});

	it("GET /api/repertoire/:pieceId maps NotFoundError to 404", async () => {
		const { NotFoundError } = await import("../lib/errors");
		vi.spyOn(repertoireService, "listPieceHistory").mockRejectedValueOnce(
			new NotFoundError("piece", "ghost"),
		);

		const res = await makeAuthApp("student-a").request("/api/repertoire/ghost");
		expect(res.status).toBe(404);
	});
});
