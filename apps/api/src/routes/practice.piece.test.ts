import { getTableName } from "drizzle-orm";
import type { PgTable } from "drizzle-orm/pg-core";
import { Hono } from "hono";
import { describe, expect, it } from "vitest";
import type { Bindings, Db, Variables } from "../lib/types";
import { errorHandler } from "../middleware/error-handler";
import { practiceRoutes } from "./practice";

/** Records what was inserted into `sessions` so the test can assert on it. */
function makeDb(
	captured: { values?: Record<string, unknown> },
	piece: unknown,
) {
	return {
		query: {
			conversations: { findFirst: async () => undefined },
			pieces: { findFirst: async () => piece },
		},
		insert: (table: PgTable) => ({
			values: (v: Record<string, unknown>) => {
				if (getTableName(table) === "sessions") captured.values = v;
				return {
					returning: async () => [
						{ id: "00000000-0000-0000-0000-0000000000aa" },
					],
				};
			},
		}),
	} as unknown as Db;
}

function makeApp(db: Db) {
	return new Hono<{ Bindings: Bindings; Variables: Variables }>()
		.use("*", async (c, next) => {
			c.set("studentId", "student-a");
			c.set("db", db);
			await next();
		})
		.onError(errorHandler)
		.route("/api/practice", practiceRoutes);
}

describe("POST /api/practice/start -- piece identity", () => {
	it("stores a catalog pieceId on the session", async () => {
		const captured: { values?: Record<string, unknown> } = {};
		const db = makeDb(captured, { pieceId: "chopin-nocturne-op9-no2" });

		const res = await makeApp(db).request("/api/practice/start", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ pieceId: "chopin-nocturne-op9-no2" }),
		});

		expect(res.status).toBe(201);
		expect(captured.values?.pieceId).toBe("chopin-nocturne-op9-no2");
	});

	it("rejects a pieceId the catalog does not have, rather than storing it", async () => {
		const captured: { values?: Record<string, unknown> } = {};
		const db = makeDb(captured, undefined);

		const res = await makeApp(db).request("/api/practice/start", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ pieceId: "ghost" }),
		});

		expect(res.status).toBe(404);
		expect(captured.values).toBeUndefined();
	});

	it("still starts a pieceless session when no pieceId is sent", async () => {
		const captured: { values?: Record<string, unknown> } = {};
		const db = makeDb(captured, undefined);

		const res = await makeApp(db).request("/api/practice/start", {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({}),
		});

		expect(res.status).toBe(201);
		expect(captured.values?.pieceId).toBeUndefined();
	});
});
