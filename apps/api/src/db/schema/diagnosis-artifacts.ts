import { index, integer, jsonb, pgTable, text, timestamp, uuid } from 'drizzle-orm/pg-core'

export const diagnosisArtifacts = pgTable(
  'diagnosis_artifacts',
  {
    id: uuid('id').defaultRandom().primaryKey(),
    sessionId: uuid('session_id').notNull(),
    studentId: text('student_id').notNull(),
    pieceId: text('piece_id'),
    barRangeStart: integer('bar_range_start'),
    barRangeEnd: integer('bar_range_end'),
    primaryDimension: text('primary_dimension').notNull(),
    artifactJson: jsonb('artifact_json').notNull(),
    createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [
    index('idx_diagnosis_session').on(t.sessionId),
    index('idx_diagnosis_student').on(t.studentId, t.createdAt),
  ],
)
