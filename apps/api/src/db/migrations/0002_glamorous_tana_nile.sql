CREATE TABLE "diagnosis_artifacts" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"session_id" uuid NOT NULL,
	"student_id" text NOT NULL,
	"piece_id" text,
	"bar_range_start" integer,
	"bar_range_end" integer,
	"primary_dimension" text NOT NULL,
	"artifact_json" jsonb NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE INDEX "idx_diagnosis_artifacts_session" ON "diagnosis_artifacts" USING btree ("session_id");--> statement-breakpoint
CREATE INDEX "idx_diagnosis_artifacts_student" ON "diagnosis_artifacts" USING btree ("student_id","created_at");