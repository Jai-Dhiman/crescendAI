import {
	pgTable,
	uuid,
	text,
	real,
	integer,
	timestamp,
	primaryKey,
	index,
	uniqueIndex,
} from "drizzle-orm/pg-core";

export const students = pgTable("students", {
	studentId: uuid("student_id").defaultRandom().primaryKey(),
	email: text("email"),
	displayName: text("display_name"),
	inferredLevel: text("inferred_level"),
	baselineDynamics: real("baseline_dynamics"),
	baselineTiming: real("baseline_timing"),
	baselinePedaling: real("baseline_pedaling"),
	baselineArticulation: real("baseline_articulation"),
	baselinePhrasing: real("baseline_phrasing"),
	baselineInterpretation: real("baseline_interpretation"),
	baselineSessionCount: integer("baseline_session_count").notNull().default(0),
	explicitGoals: text("explicit_goals"),
	createdAt: timestamp("created_at", { withTimezone: true })
		.notNull()
		.defaultNow(),
	updatedAt: timestamp("updated_at", { withTimezone: true })
		.notNull()
		.defaultNow(),
});

export const authIdentities = pgTable(
	"auth_identities",
	{
		provider: text("provider").notNull(),
		providerUserId: text("provider_user_id").notNull(),
		studentId: uuid("student_id")
			.notNull()
			.references(() => students.studentId, { onDelete: "cascade" }),
		createdAt: timestamp("created_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
	},
	(t) => [
		primaryKey({ columns: [t.provider, t.providerUserId] }),
		uniqueIndex("idx_auth_identities_provider_user").on(
			t.provider,
			t.providerUserId,
		),
		index("idx_auth_identities_student").on(t.studentId),
	],
);
