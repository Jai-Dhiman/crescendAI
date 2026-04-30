import { integer, pgTable, text, timestamp } from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";

export const SEGMENT_LOOP_STATUSES = [
  "pending",
  "active",
  "completed",
  "dismissed",
  "superseded",
] as const;

export type SegmentLoopStatus = (typeof SEGMENT_LOOP_STATUSES)[number];

export const SEGMENT_LOOP_TRIGGERS = ["chat", "synthesis"] as const;
export type SegmentLoopTrigger = (typeof SEGMENT_LOOP_TRIGGERS)[number];

export const segmentLoops = pgTable("segment_loops", {
  id: text("id")
    .primaryKey()
    .default(sql`gen_random_uuid()`),
  studentId: text("student_id").notNull(),
  pieceId: text("piece_id").notNull(),
  conversationId: text("conversation_id"),
  barsStart: integer("bars_start").notNull(),
  barsEnd: integer("bars_end").notNull(),
  dimension: text("dimension"),
  requiredCorrect: integer("required_correct").notNull().default(5),
  attemptsCompleted: integer("attempts_completed").notNull().default(0),
  status: text("status").notNull().default("pending"),
  trigger: text("trigger").notNull(),
  createdAt: timestamp("created_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
});
