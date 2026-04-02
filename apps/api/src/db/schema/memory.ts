import {
	index,
	integer,
	jsonb,
	pgTable,
	text,
	timestamp,
	uuid,
} from "drizzle-orm/pg-core";

export const synthesizedFacts = pgTable(
	"synthesized_facts",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		studentId: text("student_id").notNull(),
		factText: text("fact_text").notNull(),
		factType: text("fact_type").notNull(),
		dimension: text("dimension"),
		pieceContext: text("piece_context"),
		validAt: timestamp("valid_at", { withTimezone: true }).notNull(),
		invalidAt: timestamp("invalid_at", { withTimezone: true }),
		trend: text("trend"),
		confidence: text("confidence").notNull(),
		evidence: text("evidence").notNull(),
		sourceType: text("source_type").notNull(),
		createdAt: timestamp("created_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
		expiredAt: timestamp("expired_at", { withTimezone: true }),
		entities: jsonb("entities"),
	},
	(t) => [
		index("idx_synthesized_facts_student").on(t.studentId),
		index("idx_synthesized_facts_active").on(
			t.studentId,
			t.invalidAt,
			t.expiredAt,
		),
		index("idx_sf_student_dimension").on(t.studentId, t.dimension),
		index("idx_sf_student_source").on(t.studentId, t.sourceType),
	],
);

export const studentMemoryMeta = pgTable("student_memory_meta", {
	studentId: text("student_id").primaryKey(),
	lastSynthesisAt: timestamp("last_synthesis_at", { withTimezone: true }),
	totalObservations: integer("total_observations").notNull().default(0),
	totalFacts: integer("total_facts").notNull().default(0),
});
