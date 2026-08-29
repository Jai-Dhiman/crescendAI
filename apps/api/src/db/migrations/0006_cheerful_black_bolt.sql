ALTER TABLE "sessions" ADD COLUMN "piece_id" text;--> statement-breakpoint
ALTER TABLE "sessions" ADD CONSTRAINT "sessions_piece_id_pieces_piece_id_fk" FOREIGN KEY ("piece_id") REFERENCES "public"."pieces"("piece_id") ON DELETE set null ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "idx_sessions_student_piece" ON "sessions" USING btree ("student_id","piece_id");