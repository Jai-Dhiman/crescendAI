import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import { z } from "zod";
import type { Bindings, Variables } from "../lib/types";
import { validate } from "../lib/validate";
import { requireAuth } from "../middleware/auth-session";
import * as chatService from "../services/chat";
import * as teacherService from "../services/teacher";
import type { InlineComponent } from "../services/tool-processor";

const chatSchema = z.object({
	conversationId: z.string().uuid().optional(),
	message: z.string().min(1).max(10000),
});

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>().post(
	"/",
	validate("json", chatSchema),
	async (c) => {
		requireAuth(c.var.studentId);
		const studentId = c.var.studentId;
		const body = c.req.valid("json");
		const ctx = { db: c.var.db, env: c.env };

		const { conversationId, isNewConversation, messages, dynamicContext } =
			await chatService.prepareChatContext(ctx, studentId, body);

		c.header("Content-Encoding", "Identity");

		return streamSSE(c, async (sseStream) => {
			let id = 0;

			await sseStream.writeSSE({
				data: JSON.stringify({ conversationId }),
				event: "start",
				id: String(id++),
			});

			let fullText = "";
			let allComponents: InlineComponent[] = [];

			try {
				for await (const event of teacherService.chat(ctx, studentId, messages, dynamicContext)) {
					if (event.type === "delta") {
						await sseStream.writeSSE({
							data: event.text,
							event: "delta",
							id: String(id++),
						});
					} else if (event.type === "tool_result") {
						await sseStream.writeSSE({
							data: JSON.stringify({ name: event.name, componentsJson: JSON.stringify(event.componentsJson) }),
							event: "tool_result",
							id: String(id++),
						});
					} else if (event.type === "done") {
						fullText = event.fullText;
						allComponents = event.allComponents;
					}
				}
			} catch (err) {
				await sseStream.writeSSE({
					data: JSON.stringify({ message: err instanceof Error ? err.message : String(err) }),
					event: "error",
					id: String(id++),
				});
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
					fullText,
					isNewConversation,
					body.message,
					allComponents.length > 0 ? allComponents : null,
				),
			);
		});
	},
);

export { app as chatRoutes };
