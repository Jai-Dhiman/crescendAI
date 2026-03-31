import {
  pgTable,
  uuid,
  text,
  boolean,
  integer,
  timestamp,
  jsonb,
  primaryKey,
  index,
  uniqueIndex,
} from "drizzle-orm/pg-core";

export const exercises = pgTable(
  "exercises",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    title: text("title").notNull(),
    description: text("description").notNull(),
    instructions: text("instructions").notNull(),
    difficulty: text("difficulty").notNull(),
    category: text("category").notNull(),
    repertoireTags: jsonb("repertoire_tags"),
    notationContent: text("notation_content"),
    notationFormat: text("notation_format"),
    midiContent: text("midi_content"),
    source: text("source").notNull(),
    variantsJson: jsonb("variants_json"),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [index("idx_exercises_difficulty").on(t.difficulty)],
);

export const exerciseDimensions = pgTable(
  "exercise_dimensions",
  {
    exerciseId: uuid("exercise_id")
      .notNull()
      .references(() => exercises.id, { onDelete: "cascade" }),
    dimension: text("dimension").notNull(),
  },
  (t) => [
    primaryKey({ columns: [t.exerciseId, t.dimension] }),
    index("idx_exercise_dimensions_dim").on(t.dimension),
  ],
);

export const studentExercises = pgTable(
  "student_exercises",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    studentId: uuid("student_id").notNull(),
    exerciseId: uuid("exercise_id").notNull(),
    sessionId: uuid("session_id"),
    assignedAt: timestamp("assigned_at", { withTimezone: true }).notNull().defaultNow(),
    completed: boolean("completed").notNull().default(false),
    response: text("response"),
    dimensionBeforeJson: jsonb("dimension_before_json"),
    dimensionAfterJson: jsonb("dimension_after_json"),
    notes: text("notes"),
    timesAssigned: integer("times_assigned").notNull().default(1),
  },
  (t) => [
    uniqueIndex("idx_student_exercises_unique").on(t.studentId, t.exerciseId, t.sessionId),
    index("idx_student_exercises").on(t.studentId, t.exerciseId),
  ],
);
