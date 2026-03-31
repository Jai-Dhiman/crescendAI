import { Hono } from "hono";
import { z } from "zod";
import { eq, and } from "drizzle-orm";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import { NotFoundError, ForbiddenError } from "../lib/errors";
import { sessions } from "../db/schema/sessions";
import { conversations, messages } from "../db/schema/conversations";
import { SessionAccumulator } from "../services/accumulator";
import {
	buildSynthesisPrompt,
	callSynthesisLlm,
	persistSynthesisMessage,
	clearNeedsSynthesis,
	loadBaselinesFromDb,
	type SynthesisContext,
} from "../services/synthesis";

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
			if (!existing) throw new NotFoundError("conversation", body.conversationId);
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
	.get("/ws/:sessionId", async (c) => {
		if (c.req.header("Upgrade") !== "websocket") {
			return c.text("Expected WebSocket upgrade", 426);
		}

		requireAuth(c.var.studentId);

		const sessionId = c.req.param("sessionId");
		const id = c.env.SESSION_BRAIN.idFromName(sessionId);
		const stub = c.env.SESSION_BRAIN.get(id);

		const url = new URL(c.req.url);
		url.searchParams.set("studentId", c.var.studentId);
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

		const acc = SessionAccumulator.fromJSON(session.accumulatorJson);

		const baselines = await loadBaselinesFromDb(db, studentId);

		const sessionDurationMs = session.endedAt
			? session.endedAt.getTime() - session.startedAt.getTime()
			: 0;

		const context: SynthesisContext = {
			sessionId,
			studentId,
			conversationId: session.conversationId,
			baselines,
			pieceContext: null,
			studentMemory: null,
			totalChunks: acc.timeline.length,
			sessionDurationMs,
		};

		const promptContext = buildSynthesisPrompt(acc, context);
		const result = await callSynthesisLlm(c.env, promptContext);

		if (session.conversationId) {
			await persistSynthesisMessage(
				db,
				session.conversationId,
				result.text,
				sessionId,
			);
		}

		await clearNeedsSynthesis(db, sessionId);

		return c.json({
			status: "completed",
			sessionId,
			isFallback: result.isFallback,
		});
	});

export { app as practiceRoutes };
