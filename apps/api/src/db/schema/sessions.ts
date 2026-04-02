import {
	boolean,
	index,
	jsonb,
	pgTable,
	real,
	text,
	timestamp,
	uuid,
} from "drizzle-orm/pg-core";

export const sessions = pgTable(
	"sessions",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: text("student_id").notNull(),
		startedAt: timestamp("started_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
		endedAt: timestamp("ended_at", { withTimezone: true }),
		avgDynamics: real("avg_dynamics"),
		avgTiming: real("avg_timing"),
		avgPedaling: real("avg_pedaling"),
		avgArticulation: real("avg_articulation"),
		avgPhrasing: real("avg_phrasing"),
		avgInterpretation: real("avg_interpretation"),
		observationsJson: jsonb("observations_json"),
		chunksSummaryJson: jsonb("chunks_summary_json"),
		conversationId: text("conversation_id"),
		accumulatorJson: jsonb("accumulator_json"),
		needsSynthesis: boolean("needs_synthesis").notNull().default(false),
	},
	(t) => [
		index("idx_sessions_student").on(t.studentId, t.startedAt),
		index("idx_sessions_conversation").on(t.conversationId),
	],
);

export const studentCheckIns = pgTable(
	"student_check_ins",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: text("student_id").notNull(),
		sessionId: uuid("session_id").references(() => sessions.id, {
			onDelete: "set null",
		}),
		question: text("question").notNull(),
		answer: text("answer"),
		createdAt: timestamp("created_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
	},
	(t) => [index("idx_checkins_student").on(t.studentId)],
);
