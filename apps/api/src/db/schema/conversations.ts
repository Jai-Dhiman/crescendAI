import {
	index,
	jsonb,
	pgTable,
	text,
	timestamp,
	uuid,
} from "drizzle-orm/pg-core";

export const conversations = pgTable(
	"conversations",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: text("student_id").notNull(),
		title: text("title"),
		createdAt: timestamp("created_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
		updatedAt: timestamp("updated_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
	},
	(t) => [index("idx_conversations_student").on(t.studentId, t.updatedAt)],
);

export const messages = pgTable(
	"messages",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		conversationId: uuid("conversation_id")
			.notNull()
			.references(() => conversations.id, { onDelete: "cascade" }),
		role: text("role").notNull(),
		content: text("content").notNull(),
		createdAt: timestamp("created_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
		messageType: text("message_type").notNull().default("chat"),
		dimension: text("dimension"),
		framing: text("framing"),
		componentsJson: jsonb("components_json"),
		sessionId: uuid("session_id"),
		observationId: uuid("observation_id"),
	},
	(t) => [index("idx_messages_conversation").on(t.conversationId, t.createdAt)],
);
