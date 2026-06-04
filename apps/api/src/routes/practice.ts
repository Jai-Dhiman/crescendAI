import { and, eq } from "drizzle-orm";
import { Hono } from "hono";
import { z } from "zod";
import { conversations, messages } from "../db/schema/conversations";
import { sessions } from "../db/schema/sessions";
import { ForbiddenError, NotFoundError } from "../lib/errors";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";

const startBodySchema = z.object({
	conversationId: z.string().uuid().optional(),
});

const chunkQuerySchema = z.object({
	sessionId: z.string().uuid(),
	chunkIndex: z.coerce.number().int().min(0),
});

const needsSynthesisQuerySchema = z.object({
	conversationId: z.string().uuid(),
});

const synthesizeBodySchema = z.object({
	sessionId: z.string().uuid(),
});

/**
 * Resolves the effective studentId for a WebSocket practice session.
 *
 * SECURITY: fail-closed design — the override is granted ONLY when ALL three
 * conditions hold simultaneously:
 *   1. overrideAllowed === true  (ALLOW_EVAL_STUDENT_OVERRIDE === "true" in env)
 *   2. secretOk === true         (x-eval-secret header matches EVAL_SHARED_SECRET
 *                                 AND EVAL_SHARED_SECRET is non-empty)
 *   3. evalStudentId is non-empty
 *
 * Both flags are computed from c.env / request headers at the call site so this
 * function remains pure and unit-testable without a live Hono context.
 * If ALLOW_EVAL_STUDENT_OVERRIDE is absent/not "true", or the secret is wrong
 * or unset, the override is unconditionally denied regardless of query params.
 */
export function resolveSessionStudentId({
	isEvalQuery,
	evalStudentId,
	authStudentId,
	overrideAllowed,
	secretOk,
}: {
	isEvalQuery: boolean;
	evalStudentId: string;
	authStudentId: string | null;
	overrideAllowed: boolean;
	secretOk: boolean;
}): string | null {
	if (isEvalQuery && overrideAllowed && secretOk && evalStudentId) {
		return evalStudentId;
	}
	return authStudentId;
}

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>()
	.post("/start", validate("json", startBodySchema), async (c) => {
		requireAuth(c.var.studentId);
		const studentId = c.var.studentId;
		const body = c.req.valid("json");
		const db = c.var.db;

		let conversationId: string;

		if (body.conversationId) {
			// Verify ownership before attaching session to existing conversation
			const existing = await db.query.conversations.findFirst({
				where: (conv, { eq: e, and: a }) =>
					a(e(conv.id, body.conversationId!), e(conv.studentId, studentId)),
			});
			if (!existing)
				throw new NotFoundError("conversation", body.conversationId);
			conversationId = body.conversationId;
		} else {
			const [conv] = await db
				.insert(conversations)
				.values({ studentId })
				.returning({ id: conversations.id });
			conversationId = conv.id;
		}

		const [session] = await db
			.insert(sessions)
			.values({ studentId, conversationId })
			.returning({ id: sessions.id });

		await db.insert(messages).values({
			conversationId,
			role: "system",
			content: "session_start",
			messageType: "session_start",
			sessionId: session.id,
		});

		return c.json({ sessionId: session.id, conversationId }, 201);
	})
	.post("/chunk", validate("query", chunkQuerySchema), async (c) => {
		requireAuth(c.var.studentId);
		const { sessionId, chunkIndex } = c.req.valid("query");

		// Verify ownership
		const session = await c.var.db.query.sessions.findFirst({
			where: (s, { eq: e, and: a }) =>
				a(e(s.id, sessionId), e(s.studentId, c.var.studentId!)),
		});
		if (!session) throw new NotFoundError("session", sessionId);

		const body = await c.req.arrayBuffer();
		const r2Key = `sessions/${sessionId}/chunks/${chunkIndex}.webm`;
		await c.env.CHUNKS.put(r2Key, body);

		return c.json({ r2Key, sessionId, chunkIndex });
	})
	.get("/chunk", validate("query", chunkQuerySchema), async (c) => {
		requireAuth(c.var.studentId);
		const { sessionId, chunkIndex } = c.req.valid("query");

		const session = await c.var.db.query.sessions.findFirst({
			where: (s, { eq: e, and: a }) =>
				a(e(s.id, sessionId), e(s.studentId, c.var.studentId!)),
		});
		if (!session) throw new NotFoundError("session", sessionId);

		const r2Key = `sessions/${sessionId}/chunks/${chunkIndex}.webm`;
		const obj = await c.env.CHUNKS.get(r2Key);
		if (!obj) throw new NotFoundError("chunk", `${sessionId}/${chunkIndex}`);

		return new Response(obj.body, {
			status: 200,
			headers: {
				"Content-Type": "audio/webm",
				"Cache-Control": "private, max-age=0, no-store",
			},
		});
	})
	.get("/ws/:sessionId", async (c) => {
		if (c.req.header("Upgrade") !== "websocket") {
			return c.text("Expected WebSocket upgrade", 426);
		}

		requireAuth(c.var.studentId);

		const sessionId = c.req.param("sessionId");
		const id = c.env.SESSION_BRAIN.idFromName(sessionId);
		const stub = c.env.SESSION_BRAIN.get(id);

		const url = new URL(c.req.url);

		// Eval sessions may supply a per-recording identity so each recording
		// starts with empty sessionHistory/pastDiagnoses (cold-start guarantee).
		// SECURITY: fail-closed flag+secret design. The override requires BOTH:
		//   (a) ALLOW_EVAL_STUDENT_OVERRIDE === "true" (absent in prod configs), AND
		//   (b) x-eval-secret header matching EVAL_SHARED_SECRET (must be non-empty).
		// A missing/typo'd binding, absent header, or any mismatch denies the override.
		// Production workers never set ALLOW_EVAL_STUDENT_OVERRIDE, so the override
		// is structurally unreachable in prod regardless of query params.
		const evalSecret = c.env.EVAL_SHARED_SECRET;
		const overrideAllowed = c.env.ALLOW_EVAL_STUDENT_OVERRIDE === "true";
		const secretOk =
			typeof evalSecret === "string" &&
			evalSecret.length > 0 &&
			c.req.header("x-eval-secret") === evalSecret;
		const effectiveStudentId = resolveSessionStudentId({
			isEvalQuery: c.req.query("eval") === "true",
			evalStudentId: c.req.query("evalStudentId") ?? "",
			authStudentId: c.var.studentId,
			overrideAllowed,
			secretOk,
		});

		url.searchParams.set("studentId", effectiveStudentId ?? "");
		const conversationId = c.req.query("conversationId");
		if (conversationId) {
			url.searchParams.set("conversationId", conversationId);
		}

		return stub.fetch(new Request(url.toString(), c.req.raw));
	})
	.get(
		"/needs-synthesis",
		validate("query", needsSynthesisQuerySchema),
		async (c) => {
			requireAuth(c.var.studentId);
			const studentId = c.var.studentId;
			const { conversationId } = c.req.valid("query");
			const db = c.var.db;

			const rows = await db
				.select({ id: sessions.id })
				.from(sessions)
				.where(
					and(
						eq(sessions.conversationId, conversationId),
						eq(sessions.studentId, studentId),
						eq(sessions.needsSynthesis, true),
					),
				);

			const sessionIds = rows.map((r) => r.id);
			return c.json({ sessionIds });
		},
	)
	.post("/synthesize", validate("json", synthesizeBodySchema), async (c) => {
		requireAuth(c.var.studentId);
		const studentId = c.var.studentId;
		const { sessionId } = c.req.valid("json");
		const db = c.var.db;

		const session = await db.query.sessions.findFirst({
			where: (s, { eq }) => eq(s.id, sessionId),
		});

		if (!session) {
			throw new NotFoundError("session", sessionId);
		}

		if (session.studentId !== studentId) {
			throw new ForbiddenError();
		}

		// Route through the DO to ensure synthesisCompleted flag is checked atomically
		// and the alarm/deferred paths don't race
		const doId = c.env.SESSION_BRAIN.idFromName(sessionId);
		const stub = c.env.SESSION_BRAIN.get(doId);
		await stub.fetch(new Request("https://do/synthesize", { method: "POST" }));

		return c.json({ status: "completed", sessionId });
	});

export { app as practiceRoutes };
