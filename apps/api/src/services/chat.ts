import { and, asc, eq } from "drizzle-orm";
import { conversations, messages } from "../db/schema/conversations";
import { studentProfiles } from "../db/schema/students";
import { NotFoundError } from "../lib/errors";
import type { Bindings, Db, ServiceContext } from "../lib/types";
import { callWorkersAI } from "./llm";
import { buildMemoryContext } from "./memory";
import { buildChatFraming, buildTitlePrompt } from "./prompts";
import type { InlineComponent } from "./tool-processor";

interface ChatInput {
	conversationId?: string | null;
	message: string;
}

export interface ChatContext {
	conversationId: string;
	isNewConversation: boolean;
	messages: Array<{ role: "user" | "assistant"; content: string }>;
	dynamicContext: string;
}

export async function prepareChatContext(
	ctx: ServiceContext,
	studentId: string,
	input: ChatInput,
): Promise<ChatContext> {
	let conversationId: string;
	let isNewConversation: boolean;

	if (input.conversationId) {
		const existing = await ctx.db
			.select({ id: conversations.id })
			.from(conversations)
			.where(
				and(
					eq(conversations.id, input.conversationId),
					eq(conversations.studentId, studentId),
				),
			)
			.limit(1);

		if (existing.length === 0) {
			throw new NotFoundError("conversation", input.conversationId);
		}

		conversationId = input.conversationId;
		isNewConversation = false;
	} else {
		const [newConversation] = await ctx.db
			.insert(conversations)
			.values({ studentId })
			.returning({ id: conversations.id });

		conversationId = newConversation.id;
		isNewConversation = true;
	}

	await ctx.db.insert(messages).values({
		conversationId,
		role: "user",
		content: input.message,
	});

	const studentRows = await ctx.db
		.select({
			inferredLevel: studentProfiles.inferredLevel,
			explicitGoals: studentProfiles.explicitGoals,
		})
		.from(studentProfiles)
		.where(eq(studentProfiles.studentId, studentId))
		.limit(1);

	const student = studentRows[0] ?? null;

	const memoryContext = await buildMemoryContext(ctx, studentId);

	const recentMessages = await ctx.db
		.select({ role: messages.role, content: messages.content })
		.from(messages)
		.where(eq(messages.conversationId, conversationId))
		.orderBy(asc(messages.createdAt))
		.limit(20);

	const dynamicContext = buildChatFraming(
		student?.inferredLevel ?? "",
		student?.explicitGoals ?? "",
		memoryContext,
	);

	const filteredMessages = recentMessages
		.filter(
			(m): m is { role: "user" | "assistant"; content: string } =>
				m.role === "user" || m.role === "assistant",
		)
		.map((m) => ({ role: m.role, content: m.content }));

	return {
		conversationId,
		isNewConversation,
		messages: filteredMessages,
		dynamicContext,
	};
}

export async function saveAssistantMessage(
	db: Db,
	env: Bindings,
	conversationId: string,
	content: string,
	isNewConversation: boolean,
	firstUserMessage: string,
	componentsJson?: InlineComponent[] | null,
): Promise<void> {
	await db.insert(messages).values({
		conversationId,
		role: "assistant",
		content,
		componentsJson: componentsJson ?? null,
	});

	await db
		.update(conversations)
		.set({ updatedAt: new Date() })
		.where(eq(conversations.id, conversationId));

	if (isNewConversation && env.AI_GATEWAY_BACKGROUND) {
		try {
			const titlePrompt = buildTitlePrompt(firstUserMessage);
			const title = await callWorkersAI(
				env,
				"@cf/qwen/qwen3-30b-a3b-fp8",
				[{ role: "user", content: titlePrompt }],
				30,
			);

			await db
				.update(conversations)
				.set({ title: title.trim() })
				.where(eq(conversations.id, conversationId));
		} catch (err) {
			console.warn(
				JSON.stringify({
					level: "warn",
					message: "Title generation failed",
					conversationId,
					error: err instanceof Error ? err.message : String(err),
				}),
			);
		}
	}
}
