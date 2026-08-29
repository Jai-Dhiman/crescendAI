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
import { pieces } from "./catalog";

export const sessions = pgTable(
	"sessions",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: text("student_id").notNull(),
		startedAt: timestamp("started_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
		endedAt: timestamp("ended_at", { withTimezone: true }),
		// Nullable: a pieceless session is a permanent first-class state, not a
		// missing value. `set null` rather than `cascade` because the catalog is
		// codegen-seeded and a piece leaving it must never delete practice
		// history.
		pieceId: text("piece_id").references(() => pieces.pieceId, {
			onDelete: "set null",
		}),
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
		// The repertoire query's access path: every session for one student,
		// grouped by piece.
		index("idx_sessions_student_piece").on(t.studentId, t.pieceId),
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
