import {
	boolean,
	index,
	integer,
	pgTable,
	real,
	text,
	timestamp,
	uuid,
} from "drizzle-orm/pg-core";

export const observations = pgTable(
	"observations",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: text("student_id").notNull(),
		sessionId: uuid("session_id").notNull(),
		chunkIndex: integer("chunk_index"),
		dimension: text("dimension").notNull(),
		observationText: text("observation_text").notNull(),
		elaborationText: text("elaboration_text"),
		reasoningTrace: text("reasoning_trace"),
		framing: text("framing"),
		dimensionScore: real("dimension_score"),
		studentBaseline: real("student_baseline"),
		pieceContext: text("piece_context"),
		learningArc: text("learning_arc"),
		isFallback: boolean("is_fallback").notNull().default(false),
		createdAt: timestamp("created_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
		messageId: text("message_id"),
		conversationId: text("conversation_id"),
	},
	(t) => [
		index("idx_observations_student").on(t.studentId, t.createdAt),
		index("idx_observations_session").on(t.sessionId),
	],
);

export const teachingApproaches = pgTable(
	"teaching_approaches",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: text("student_id").notNull(),
		observationId: uuid("observation_id").notNull(),
		dimension: text("dimension").notNull(),
		framing: text("framing").notNull(),
		approachSummary: text("approach_summary").notNull(),
		engaged: boolean("engaged").notNull().default(false),
		createdAt: timestamp("created_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
	},
	(t) => [
		index("idx_teaching_approaches_student").on(t.studentId),
		index("idx_teaching_approaches_observation").on(t.observationId),
	],
);
