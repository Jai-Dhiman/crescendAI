import { eq, and, asc } from "drizzle-orm";
import type { Db, ServiceContext, Bindings } from "../lib/types";
import { NotFoundError } from "../lib/errors";
import { conversations, messages } from "../db/schema/conversations";
import { studentProfiles } from "../db/schema/students";
import { callAnthropicStream, callGroq, type AnthropicSystemBlock } from "./llm";
import { buildMemoryContext } from "./memory";
import { CHAT_SYSTEM, buildChatUserContext, buildTitlePrompt } from "./prompts";

interface ChatInput {
	conversationId?: string;
	message: string;
}

interface ChatStreamResult {
	conversationId: string;
	isNewConversation: boolean;
	stream: ReadableStream;
}

export async function handleChatStream(
	ctx: ServiceContext,
	studentId: string,
	input: ChatInput,
): Promise<ChatStreamResult> {
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
			baselineDynamics: studentProfiles.baselineDynamics,
			baselineTiming: studentProfiles.baselineTiming,
			baselinePedaling: studentProfiles.baselinePedaling,
			baselineArticulation: studentProfiles.baselineArticulation,
			baselinePhrasing: studentProfiles.baselinePhrasing,
			baselineInterpretation: studentProfiles.baselineInterpretation,
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

	const userContext = student
		? buildChatUserContext({
				inferredLevel: student.inferredLevel,
				explicitGoals: student.explicitGoals,
				baselines: {
					dynamics: student.baselineDynamics,
					timing: student.baselineTiming,
					pedaling: student.baselinePedaling,
					articulation: student.baselineArticulation,
					phrasing: student.baselinePhrasing,
					interpretation: student.baselineInterpretation,
				},
			})
		: "";

	const dynamicContext = [userContext, memoryContext]
		.filter((p) => p.trim().length > 0)
		.join("\n\n");

	const systemBlocks: AnthropicSystemBlock[] = [
		{ type: "text", text: CHAT_SYSTEM, cache_control: { type: "ephemeral" } },
		...(dynamicContext
			? [{ type: "text" as const, text: dynamicContext }]
			: []),
	];

	const anthropicMessages = recentMessages
		.filter((m): m is { role: "user" | "assistant"; content: string } =>
			m.role === "user" || m.role === "assistant",
		)
		.map((m) => ({ role: m.role, content: m.content }));

	const stream = await callAnthropicStream(ctx.env, {
		model: "claude-sonnet-4-20250514",
		max_tokens: 2048,
		system: systemBlocks,
		messages: anthropicMessages,
	});

	return { conversationId, isNewConversation, stream };
}

export async function saveAssistantMessage(
	db: Db,
	env: Bindings,
	conversationId: string,
	content: string,
	isNewConversation: boolean,
	firstUserMessage: string,
): Promise<void> {
	await db.insert(messages).values({
		conversationId,
		role: "assistant",
		content,
	});

	await db
		.update(conversations)
		.set({ updatedAt: new Date() })
		.where(eq(conversations.id, conversationId));

	if (isNewConversation) {
		try {
			const titlePrompt = buildTitlePrompt(firstUserMessage);
			const title = await callGroq(
				env,
				"llama-3.3-70b-versatile",
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
