import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import * as chatService from "../services/chat";

const chatSchema = z.object({
	conversationId: z.string().uuid().optional(),
	message: z.string().min(1).max(10000),
});

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>().post(
	"/",
	validate("json", chatSchema),
	async (c) => {
		requireAuth(c.var.studentId);
		const body = c.req.valid("json");
		const ctx = { db: c.var.db, env: c.env };

		const { conversationId, isNewConversation, stream } =
			await chatService.handleChatStream(ctx, c.var.studentId, body);

		c.header("Content-Encoding", "Identity");

		return streamSSE(c, async (sseStream) => {
			await sseStream.writeSSE({
				data: JSON.stringify({ conversationId }),
				event: "start",
				id: "0",
			});

			const reader = stream.getReader();
			const decoder = new TextDecoder();
			let fullContent = "";
			let id = 1;

			try {
				while (true) {
					const { done, value } = await reader.read();
					if (done) break;
					const chunk = decoder.decode(value, { stream: true });
					fullContent += chunk;
					await sseStream.writeSSE({
						data: chunk,
						event: "delta",
						id: String(id++),
					});
				}
			} finally {
				reader.releaseLock();
			}

			await sseStream.writeSSE({
				data: "[DONE]",
				event: "done",
				id: String(id),
			});

			c.executionCtx.waitUntil(
				chatService.saveAssistantMessage(
					c.var.db,
					c.env,
					conversationId,
					fullContent,
					isNewConversation,
					body.message,
				),
			);
		});
	},
);

export { app as chatRoutes };
