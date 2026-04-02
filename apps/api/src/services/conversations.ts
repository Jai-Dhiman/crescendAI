import { and, asc, desc, eq } from "drizzle-orm";
import { conversations, messages } from "../db/schema/conversations";
import { NotFoundError } from "../lib/errors";
import type { ServiceContext } from "../lib/types";

export async function listConversations(
	ctx: ServiceContext,
	studentId: string,
) {
	return ctx.db
		.select({
			id: conversations.id,
			title: conversations.title,
			updatedAt: conversations.updatedAt,
		})
		.from(conversations)
		.where(eq(conversations.studentId, studentId))
		.orderBy(desc(conversations.updatedAt));
}

export async function getConversation(
	ctx: ServiceContext,
	conversationId: string,
	studentId: string,
) {
	const conv = await ctx.db.query.conversations.findFirst({
		where: (c, { eq, and }) =>
			and(eq(c.id, conversationId), eq(c.studentId, studentId)),
	});

	if (!conv) {
		throw new NotFoundError("conversation", conversationId);
	}

	const msgs = await ctx.db
		.select()
		.from(messages)
		.where(eq(messages.conversationId, conversationId))
		.orderBy(asc(messages.createdAt));

	return { ...conv, messages: msgs };
}

export async function deleteConversation(
	ctx: ServiceContext,
	conversationId: string,
	studentId: string,
) {
	const conv = await ctx.db.query.conversations.findFirst({
		where: (c, { eq, and }) =>
			and(eq(c.id, conversationId), eq(c.studentId, studentId)),
	});

	if (!conv) {
		throw new NotFoundError("conversation", conversationId);
	}

	await ctx.db
		.delete(conversations)
		.where(
			and(
				eq(conversations.id, conversationId),
				eq(conversations.studentId, studentId),
			),
		);
}
