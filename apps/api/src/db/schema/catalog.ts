import {
	boolean,
	index,
	integer,
	pgTable,
	real,
	text,
	timestamp,
	uuid,
} from "drizzle-orm/pg-core";

export const pieces = pgTable(
	"pieces",
	{
		pieceId: text("piece_id").primaryKey(),
		composer: text("composer").notNull(),
		title: text("title").notNull(),
		keySignature: text("key_signature"),
		timeSignature: text("time_signature"),
		tempoBpm: integer("tempo_bpm"),
		barCount: integer("bar_count").notNull(),
		durationSeconds: real("duration_seconds"),
		noteCount: integer("note_count").notNull(),
		pitchRangeLow: integer("pitch_range_low"),
		pitchRangeHigh: integer("pitch_range_high"),
		hasTimeSigChanges: boolean("has_time_sig_changes").notNull().default(false),
		hasTempoChanges: boolean("has_tempo_changes").notNull().default(false),
		source: text("source").notNull().default("asap"),
		createdAt: timestamp("created_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
	},
	(t) => [index("idx_pieces_composer").on(t.composer)],
);

export const pieceRequests = pgTable(
	"piece_requests",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		query: text("query").notNull(),
		studentId: text("student_id").notNull(),
		matchedPieceId: text("matched_piece_id"),
		matchConfidence: real("match_confidence"),
		matchMethod: text("match_method"),
		createdAt: timestamp("created_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
	},
	(t) => [index("idx_piece_requests_unmatched").on(t.matchedPieceId)],
);

export const waitlist = pgTable(
	"waitlist",
	{
		email: text("email").primaryKey(),
		context: text("context"),
		source: text("source").notNull().default("web"),
		createdAt: timestamp("created_at", { withTimezone: true })
			.notNull()
			.defaultNow(),
	},
	(t) => [index("idx_waitlist_created").on(t.createdAt)],
);
