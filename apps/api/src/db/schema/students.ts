import { integer, pgTable, real, text, timestamp } from "drizzle-orm/pg-core";

export const studentProfiles = pgTable("student_profiles", {
	studentId: text("student_id").primaryKey(),
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
