ALTER TABLE "pending_exercises" DROP COLUMN "exercise_id";
--> statement-breakpoint
ALTER TABLE "pending_exercises" ADD COLUMN "title" text;
--> statement-breakpoint
ALTER TABLE "pending_exercises" ADD COLUMN "instruction" text;
--> statement-breakpoint
ALTER TABLE "pending_exercises" ADD COLUMN "routing_json" jsonb;
--> statement-breakpoint
ALTER TABLE "pending_exercises" ADD COLUMN "piece_id" text;
--> statement-breakpoint
DROP INDEX IF EXISTS "idx_pending_exercises_unique";
--> statement-breakpoint
CREATE UNIQUE INDEX "idx_pending_exercises_unique" ON "pending_exercises" USING btree ("student_id","session_id","id");
