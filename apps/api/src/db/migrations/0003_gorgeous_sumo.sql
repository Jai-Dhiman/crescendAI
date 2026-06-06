CREATE TABLE "pending_exercises" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"student_id" text NOT NULL,
	"session_id" uuid NOT NULL,
	"exercise_id" uuid NOT NULL,
	"focus_dimension" text NOT NULL,
	"preview_title" text NOT NULL,
	"consumed" boolean DEFAULT false NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX "idx_pending_exercises_unique" ON "pending_exercises" USING btree ("student_id","session_id","exercise_id");--> statement-breakpoint
CREATE INDEX "idx_pending_exercises_lookup" ON "pending_exercises" USING btree ("student_id","consumed");