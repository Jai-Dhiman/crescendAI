CREATE TABLE "account" (
	"id" text PRIMARY KEY NOT NULL,
	"account_id" text NOT NULL,
	"provider_id" text NOT NULL,
	"user_id" text NOT NULL,
	"access_token" text,
	"refresh_token" text,
	"id_token" text,
	"access_token_expires_at" timestamp,
	"refresh_token_expires_at" timestamp,
	"scope" text,
	"password" text,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp NOT NULL
);
--> statement-breakpoint
CREATE TABLE "session" (
	"id" text PRIMARY KEY NOT NULL,
	"expires_at" timestamp NOT NULL,
	"token" text NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp NOT NULL,
	"ip_address" text,
	"user_agent" text,
	"user_id" text NOT NULL,
	CONSTRAINT "session_token_unique" UNIQUE("token")
);
--> statement-breakpoint
CREATE TABLE "user" (
	"id" text PRIMARY KEY NOT NULL,
	"name" text NOT NULL,
	"email" text NOT NULL,
	"email_verified" boolean DEFAULT false NOT NULL,
	"image" text,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "user_email_unique" UNIQUE("email")
);
--> statement-breakpoint
CREATE TABLE "verification" (
	"id" text PRIMARY KEY NOT NULL,
	"identifier" text NOT NULL,
	"value" text NOT NULL,
	"expires_at" timestamp NOT NULL,
	"created_at" timestamp DEFAULT now() NOT NULL,
	"updated_at" timestamp DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "student_profiles" (
	"student_id" text PRIMARY KEY NOT NULL,
	"inferred_level" text,
	"baseline_dynamics" real,
	"baseline_timing" real,
	"baseline_pedaling" real,
	"baseline_articulation" real,
	"baseline_phrasing" real,
	"baseline_interpretation" real,
	"baseline_session_count" integer DEFAULT 0 NOT NULL,
	"explicit_goals" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "sessions" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"student_id" text NOT NULL,
	"started_at" timestamp with time zone DEFAULT now() NOT NULL,
	"ended_at" timestamp with time zone,
	"avg_dynamics" real,
	"avg_timing" real,
	"avg_pedaling" real,
	"avg_articulation" real,
	"avg_phrasing" real,
	"avg_interpretation" real,
	"observations_json" jsonb,
	"chunks_summary_json" jsonb,
	"conversation_id" text,
	"accumulator_json" jsonb,
	"needs_synthesis" boolean DEFAULT false NOT NULL
);
--> statement-breakpoint
CREATE TABLE "student_check_ins" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"student_id" text NOT NULL,
	"session_id" uuid,
	"question" text NOT NULL,
	"answer" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "observations" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"student_id" text NOT NULL,
	"session_id" uuid NOT NULL,
	"chunk_index" integer,
	"dimension" text NOT NULL,
	"observation_text" text NOT NULL,
	"elaboration_text" text,
	"reasoning_trace" text,
	"framing" text,
	"dimension_score" real,
	"student_baseline" real,
	"piece_context" text,
	"learning_arc" text,
	"is_fallback" boolean DEFAULT false NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"message_id" text,
	"conversation_id" text
);
--> statement-breakpoint
CREATE TABLE "teaching_approaches" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"student_id" text NOT NULL,
	"observation_id" uuid NOT NULL,
	"dimension" text NOT NULL,
	"framing" text NOT NULL,
	"approach_summary" text NOT NULL,
	"engaged" boolean DEFAULT false NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "conversations" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"student_id" text NOT NULL,
	"title" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "messages" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"conversation_id" uuid NOT NULL,
	"role" text NOT NULL,
	"content" text NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"message_type" text DEFAULT 'chat' NOT NULL,
	"dimension" text,
	"framing" text,
	"components_json" jsonb,
	"session_id" uuid,
	"observation_id" uuid
);
--> statement-breakpoint
CREATE TABLE "exercise_dimensions" (
	"exercise_id" uuid NOT NULL,
	"dimension" text NOT NULL,
	CONSTRAINT "exercise_dimensions_exercise_id_dimension_pk" PRIMARY KEY("exercise_id","dimension")
);
--> statement-breakpoint
CREATE TABLE "exercises" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"title" text NOT NULL,
	"description" text NOT NULL,
	"instructions" text NOT NULL,
	"difficulty" text NOT NULL,
	"category" text NOT NULL,
	"repertoire_tags" jsonb,
	"notation_content" text,
	"notation_format" text,
	"midi_content" text,
	"source" text NOT NULL,
	"variants_json" jsonb,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "student_exercises" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"student_id" text NOT NULL,
	"exercise_id" uuid NOT NULL,
	"session_id" uuid,
	"assigned_at" timestamp with time zone DEFAULT now() NOT NULL,
	"completed" boolean DEFAULT false NOT NULL,
	"response" text,
	"dimension_before_json" jsonb,
	"dimension_after_json" jsonb,
	"notes" text,
	"times_assigned" integer DEFAULT 1 NOT NULL
);
--> statement-breakpoint
CREATE TABLE "student_memory_meta" (
	"student_id" text PRIMARY KEY NOT NULL,
	"last_synthesis_at" timestamp with time zone,
	"total_observations" integer DEFAULT 0 NOT NULL,
	"total_facts" integer DEFAULT 0 NOT NULL
);
--> statement-breakpoint
CREATE TABLE "synthesized_facts" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"student_id" text NOT NULL,
	"fact_text" text NOT NULL,
	"fact_type" text NOT NULL,
	"dimension" text,
	"piece_context" text,
	"valid_at" timestamp with time zone NOT NULL,
	"invalid_at" timestamp with time zone,
	"trend" text,
	"confidence" text NOT NULL,
	"evidence" text NOT NULL,
	"source_type" text NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"expired_at" timestamp with time zone,
	"entities" jsonb
);
--> statement-breakpoint
CREATE TABLE "piece_requests" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"query" text NOT NULL,
	"student_id" text NOT NULL,
	"matched_piece_id" text,
	"match_confidence" real,
	"match_method" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "pieces" (
	"piece_id" text PRIMARY KEY NOT NULL,
	"composer" text NOT NULL,
	"title" text NOT NULL,
	"key_signature" text,
	"time_signature" text,
	"tempo_bpm" integer,
	"bar_count" integer NOT NULL,
	"duration_seconds" real,
	"note_count" integer NOT NULL,
	"pitch_range_low" integer,
	"pitch_range_high" integer,
	"has_time_sig_changes" boolean DEFAULT false NOT NULL,
	"has_tempo_changes" boolean DEFAULT false NOT NULL,
	"source" text DEFAULT 'asap' NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "waitlist" (
	"email" text PRIMARY KEY NOT NULL,
	"context" text,
	"source" text DEFAULT 'web' NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
ALTER TABLE "account" ADD CONSTRAINT "account_user_id_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."user"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "session" ADD CONSTRAINT "session_user_id_user_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."user"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "student_check_ins" ADD CONSTRAINT "student_check_ins_session_id_sessions_id_fk" FOREIGN KEY ("session_id") REFERENCES "public"."sessions"("id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "messages" ADD CONSTRAINT "messages_conversation_id_conversations_id_fk" FOREIGN KEY ("conversation_id") REFERENCES "public"."conversations"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "exercise_dimensions" ADD CONSTRAINT "exercise_dimensions_exercise_id_exercises_id_fk" FOREIGN KEY ("exercise_id") REFERENCES "public"."exercises"("id") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "account_userId_idx" ON "account" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "session_userId_idx" ON "session" USING btree ("user_id");--> statement-breakpoint
CREATE INDEX "verification_identifier_idx" ON "verification" USING btree ("identifier");--> statement-breakpoint
CREATE INDEX "idx_sessions_student" ON "sessions" USING btree ("student_id","started_at");--> statement-breakpoint
CREATE INDEX "idx_sessions_conversation" ON "sessions" USING btree ("conversation_id");--> statement-breakpoint
CREATE INDEX "idx_checkins_student" ON "student_check_ins" USING btree ("student_id");--> statement-breakpoint
CREATE INDEX "idx_observations_student" ON "observations" USING btree ("student_id","created_at");--> statement-breakpoint
CREATE INDEX "idx_observations_session" ON "observations" USING btree ("session_id");--> statement-breakpoint
CREATE INDEX "idx_teaching_approaches_student" ON "teaching_approaches" USING btree ("student_id");--> statement-breakpoint
CREATE INDEX "idx_teaching_approaches_observation" ON "teaching_approaches" USING btree ("observation_id");--> statement-breakpoint
CREATE INDEX "idx_conversations_student" ON "conversations" USING btree ("student_id","updated_at");--> statement-breakpoint
CREATE INDEX "idx_messages_conversation" ON "messages" USING btree ("conversation_id","created_at");--> statement-breakpoint
CREATE INDEX "idx_exercise_dimensions_dim" ON "exercise_dimensions" USING btree ("dimension");--> statement-breakpoint
CREATE INDEX "idx_exercises_difficulty" ON "exercises" USING btree ("difficulty");--> statement-breakpoint
CREATE UNIQUE INDEX "idx_student_exercises_unique" ON "student_exercises" USING btree ("student_id","exercise_id","session_id");--> statement-breakpoint
CREATE INDEX "idx_student_exercises" ON "student_exercises" USING btree ("student_id","exercise_id");--> statement-breakpoint
CREATE INDEX "idx_synthesized_facts_student" ON "synthesized_facts" USING btree ("student_id");--> statement-breakpoint
CREATE INDEX "idx_synthesized_facts_active" ON "synthesized_facts" USING btree ("student_id","invalid_at","expired_at");--> statement-breakpoint
CREATE INDEX "idx_sf_student_dimension" ON "synthesized_facts" USING btree ("student_id","dimension");--> statement-breakpoint
CREATE INDEX "idx_sf_student_source" ON "synthesized_facts" USING btree ("student_id","source_type");--> statement-breakpoint
CREATE INDEX "idx_piece_requests_unmatched" ON "piece_requests" USING btree ("matched_piece_id");--> statement-breakpoint
CREATE INDEX "idx_pieces_composer" ON "pieces" USING btree ("composer");--> statement-breakpoint
CREATE INDEX "idx_waitlist_created" ON "waitlist" USING btree ("created_at");