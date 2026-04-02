/**
 * One-time D1 -> Postgres migration script.
 *
 * Usage:
 *   DATABASE_URL="postgres://..." D1_DATABASE_ID="crescendai-db" \
 *     bun run scripts/migrate-d1.ts
 *
 * Safe to re-run: all inserts use .onConflictDoNothing().
 */

import { execFileSync } from "node:child_process";
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import * as schema from "../src/db/schema/index";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const DATABASE_URL = process.env.DATABASE_URL;
const D1_DB = process.env.D1_DATABASE_ID;

if (!DATABASE_URL) {
	console.error("Missing required env var: DATABASE_URL");
	process.exit(1);
}

if (!D1_DB) {
	console.error("Missing required env var: D1_DATABASE_ID");
	process.exit(1);
}

// ---------------------------------------------------------------------------
// DB connection
// ---------------------------------------------------------------------------

const sql = postgres(DATABASE_URL);
const db = drizzle(sql, { schema });

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

function toDate(val: string | null | undefined): Date | null {
	if (!val) return null;
	return new Date(val);
}

function toBool(val: number | null | undefined): boolean {
	return val === 1;
}

function parseJson<T>(val: string | null | undefined): T | null {
	if (!val) return null;
	return JSON.parse(val) as T;
}

// ---------------------------------------------------------------------------
// D1 query helper
// ---------------------------------------------------------------------------

function queryD1(query: string): unknown[] {
	const result = execFileSync(
		"wrangler",
		["d1", "execute", D1_DB as string, "--json", "--command", query],
		{ encoding: "utf-8", maxBuffer: 50 * 1024 * 1024 },
	);
	const parsed = JSON.parse(result) as Array<{ results?: unknown[] }>;
	return parsed[0]?.results ?? [];
}

// ---------------------------------------------------------------------------
// Migration helper
// ---------------------------------------------------------------------------

async function migrateTable(
	name: string,
	query: string,
	insertFn: (rows: unknown[]) => Promise<void>,
): Promise<void> {
	console.log(`Migrating ${name}...`);
	const rows = queryD1(query);
	console.log(`  Found ${rows.length} rows`);
	if (rows.length > 0) {
		await insertFn(rows);
		console.log(`  Inserted ${rows.length} rows`);
	}
}

// ---------------------------------------------------------------------------
// Table migrations (FK-safe order)
// ---------------------------------------------------------------------------

async function migrateStudents(): Promise<void> {
	await migrateTable("students", "SELECT * FROM students", async (rows) => {
		for (const row of rows) {
			const r = row as Record<string, unknown>;
			await db
				.insert(schema.students)
				.values({
					studentId: r.student_id as string,
					email: (r.email as string | null) ?? null,
					displayName: (r.display_name as string | null) ?? null,
					inferredLevel: (r.inferred_level as string | null) ?? null,
					baselineDynamics: (r.baseline_dynamics as number | null) ?? null,
					baselineTiming: (r.baseline_timing as number | null) ?? null,
					baselinePedaling: (r.baseline_pedaling as number | null) ?? null,
					baselineArticulation:
						(r.baseline_articulation as number | null) ?? null,
					baselinePhrasing: (r.baseline_phrasing as number | null) ?? null,
					baselineInterpretation:
						(r.baseline_interpretation as number | null) ?? null,
					baselineSessionCount:
						(r.baseline_session_count as number | null) ?? 0,
					explicitGoals: (r.explicit_goals as string | null) ?? null,
					createdAt: toDate(r.created_at as string | null) ?? new Date(),
					updatedAt: toDate(r.updated_at as string | null) ?? new Date(),
				})
				.onConflictDoNothing();
		}
	});
}

async function migrateAuthIdentities(): Promise<void> {
	await migrateTable(
		"auth_identities",
		"SELECT * FROM auth_identities",
		async (rows) => {
			for (const row of rows) {
				const r = row as Record<string, unknown>;
				await db
					.insert(schema.authIdentities)
					.values({
						provider: r.provider as string,
						providerUserId: r.provider_user_id as string,
						studentId: r.student_id as string,
						createdAt: toDate(r.created_at as string | null) ?? new Date(),
					})
					.onConflictDoNothing();
			}
		},
	);
}

async function migrateSessions(): Promise<void> {
	await migrateTable("sessions", "SELECT * FROM sessions", async (rows) => {
		for (const row of rows) {
			const r = row as Record<string, unknown>;
			await db
				.insert(schema.sessions)
				.values({
					id: r.id as string,
					studentId: r.student_id as string,
					startedAt: toDate(r.started_at as string | null) ?? new Date(),
					endedAt: toDate(r.ended_at as string | null),
					avgDynamics: (r.avg_dynamics as number | null) ?? null,
					avgTiming: (r.avg_timing as number | null) ?? null,
					avgPedaling: (r.avg_pedaling as number | null) ?? null,
					avgArticulation: (r.avg_articulation as number | null) ?? null,
					avgPhrasing: (r.avg_phrasing as number | null) ?? null,
					avgInterpretation: (r.avg_interpretation as number | null) ?? null,
					observationsJson: parseJson(r.observations_json as string | null),
					chunksSummaryJson: parseJson(r.chunks_summary_json as string | null),
					conversationId: (r.conversation_id as string | null) ?? null,
					accumulatorJson: parseJson(r.accumulator_json as string | null),
					needsSynthesis: toBool(r.needs_synthesis as number | null),
				})
				.onConflictDoNothing();
		}
	});
}

async function migrateStudentCheckIns(): Promise<void> {
	await migrateTable(
		"student_check_ins",
		"SELECT * FROM student_check_ins",
		async (rows) => {
			for (const row of rows) {
				const r = row as Record<string, unknown>;
				await db
					.insert(schema.studentCheckIns)
					.values({
						id: r.id as string,
						studentId: r.student_id as string,
						sessionId: (r.session_id as string | null) ?? null,
						question: r.question as string,
						answer: (r.answer as string | null) ?? null,
						createdAt: toDate(r.created_at as string | null) ?? new Date(),
					})
					.onConflictDoNothing();
			}
		},
	);
}

async function migrateConversations(): Promise<void> {
	await migrateTable(
		"conversations",
		"SELECT * FROM conversations",
		async (rows) => {
			for (const row of rows) {
				const r = row as Record<string, unknown>;
				await db
					.insert(schema.conversations)
					.values({
						id: r.id as string,
						studentId: r.student_id as string,
						title: (r.title as string | null) ?? null,
						createdAt: toDate(r.created_at as string | null) ?? new Date(),
						updatedAt: toDate(r.updated_at as string | null) ?? new Date(),
					})
					.onConflictDoNothing();
			}
		},
	);
}

async function migrateMessages(): Promise<void> {
	await migrateTable("messages", "SELECT * FROM messages", async (rows) => {
		for (const row of rows) {
			const r = row as Record<string, unknown>;
			await db
				.insert(schema.messages)
				.values({
					id: r.id as string,
					conversationId: r.conversation_id as string,
					role: r.role as string,
					content: r.content as string,
					createdAt: toDate(r.created_at as string | null) ?? new Date(),
					messageType: (r.message_type as string | null) ?? "chat",
					dimension: (r.dimension as string | null) ?? null,
					framing: (r.framing as string | null) ?? null,
					componentsJson: parseJson(r.components_json as string | null),
					sessionId: (r.session_id as string | null) ?? null,
					observationId: (r.observation_id as string | null) ?? null,
				})
				.onConflictDoNothing();
		}
	});
}

async function migrateObservations(): Promise<void> {
	await migrateTable(
		"observations",
		"SELECT * FROM observations",
		async (rows) => {
			for (const row of rows) {
				const r = row as Record<string, unknown>;
				await db
					.insert(schema.observations)
					.values({
						id: r.id as string,
						studentId: r.student_id as string,
						sessionId: r.session_id as string,
						chunkIndex: (r.chunk_index as number | null) ?? null,
						dimension: r.dimension as string,
						observationText: r.observation_text as string,
						elaborationText: (r.elaboration_text as string | null) ?? null,
						reasoningTrace: (r.reasoning_trace as string | null) ?? null,
						framing: (r.framing as string | null) ?? null,
						dimensionScore: (r.dimension_score as number | null) ?? null,
						studentBaseline: (r.student_baseline as number | null) ?? null,
						pieceContext: (r.piece_context as string | null) ?? null,
						learningArc: (r.learning_arc as string | null) ?? null,
						isFallback: toBool(r.is_fallback as number | null),
						createdAt: toDate(r.created_at as string | null) ?? new Date(),
						messageId: (r.message_id as string | null) ?? null,
						conversationId: (r.conversation_id as string | null) ?? null,
					})
					.onConflictDoNothing();
			}
		},
	);
}

async function migrateTeachingApproaches(): Promise<void> {
	await migrateTable(
		"teaching_approaches",
		"SELECT * FROM teaching_approaches",
		async (rows) => {
			for (const row of rows) {
				const r = row as Record<string, unknown>;
				await db
					.insert(schema.teachingApproaches)
					.values({
						id: r.id as string,
						studentId: r.student_id as string,
						observationId: r.observation_id as string,
						dimension: r.dimension as string,
						framing: r.framing as string,
						approachSummary: r.approach_summary as string,
						engaged: toBool(r.engaged as number | null),
						createdAt: toDate(r.created_at as string | null) ?? new Date(),
					})
					.onConflictDoNothing();
			}
		},
	);
}

async function migrateSynthesizedFacts(): Promise<void> {
	await migrateTable(
		"synthesized_facts",
		"SELECT * FROM synthesized_facts",
		async (rows) => {
			for (const row of rows) {
				const r = row as Record<string, unknown>;
				await db
					.insert(schema.synthesizedFacts)
					.values({
						id: r.id as string,
						studentId: r.student_id as string,
						factText: r.fact_text as string,
						factType: r.fact_type as string,
						dimension: (r.dimension as string | null) ?? null,
						pieceContext: (r.piece_context as string | null) ?? null,
						validAt: toDate(r.valid_at as string | null)!,
						invalidAt: toDate(r.invalid_at as string | null),
						trend: (r.trend as string | null) ?? null,
						confidence: r.confidence as string,
						evidence: r.evidence as string,
						sourceType: r.source_type as string,
						createdAt: toDate(r.created_at as string | null) ?? new Date(),
						expiredAt: toDate(r.expired_at as string | null),
						entities: parseJson(r.entities as string | null),
					})
					.onConflictDoNothing();
			}
		},
	);
}

async function migrateStudentMemoryMeta(): Promise<void> {
	await migrateTable(
		"student_memory_meta",
		"SELECT * FROM student_memory_meta",
		async (rows) => {
			for (const row of rows) {
				const r = row as Record<string, unknown>;
				await db
					.insert(schema.studentMemoryMeta)
					.values({
						studentId: r.student_id as string,
						lastSynthesisAt: toDate(r.last_synthesis_at as string | null),
						totalObservations: (r.total_observations as number | null) ?? 0,
						totalFacts: (r.total_facts as number | null) ?? 0,
					})
					.onConflictDoNothing();
			}
		},
	);
}

async function migrateExercises(): Promise<void> {
	await migrateTable("exercises", "SELECT * FROM exercises", async (rows) => {
		for (const row of rows) {
			const r = row as Record<string, unknown>;
			await db
				.insert(schema.exercises)
				.values({
					id: r.id as string,
					title: r.title as string,
					description: r.description as string,
					instructions: r.instructions as string,
					difficulty: r.difficulty as string,
					category: r.category as string,
					repertoireTags: parseJson(r.repertoire_tags as string | null),
					notationContent: (r.notation_content as string | null) ?? null,
					notationFormat: (r.notation_format as string | null) ?? null,
					midiContent: (r.midi_content as string | null) ?? null,
					source: r.source as string,
					variantsJson: parseJson(r.variants_json as string | null),
					createdAt: toDate(r.created_at as string | null) ?? new Date(),
				})
				.onConflictDoNothing();
		}
	});
}

async function migrateExerciseDimensions(): Promise<void> {
	await migrateTable(
		"exercise_dimensions",
		"SELECT * FROM exercise_dimensions",
		async (rows) => {
			for (const row of rows) {
				const r = row as Record<string, unknown>;
				await db
					.insert(schema.exerciseDimensions)
					.values({
						exerciseId: r.exercise_id as string,
						dimension: r.dimension as string,
					})
					.onConflictDoNothing();
			}
		},
	);
}

async function migrateStudentExercises(): Promise<void> {
	await migrateTable(
		"student_exercises",
		"SELECT * FROM student_exercises",
		async (rows) => {
			for (const row of rows) {
				const r = row as Record<string, unknown>;
				await db
					.insert(schema.studentExercises)
					.values({
						id: r.id as string,
						studentId: r.student_id as string,
						exerciseId: r.exercise_id as string,
						sessionId: (r.session_id as string | null) ?? null,
						assignedAt: toDate(r.assigned_at as string | null) ?? new Date(),
						completed: toBool(r.completed as number | null),
						response: (r.response as string | null) ?? null,
						dimensionBeforeJson: parseJson(
							r.dimension_before_json as string | null,
						),
						dimensionAfterJson: parseJson(
							r.dimension_after_json as string | null,
						),
						notes: (r.notes as string | null) ?? null,
						timesAssigned: (r.times_assigned as number | null) ?? 1,
					})
					.onConflictDoNothing();
			}
		},
	);
}

async function migratePieces(): Promise<void> {
	await migrateTable("pieces", "SELECT * FROM pieces", async (rows) => {
		for (const row of rows) {
			const r = row as Record<string, unknown>;
			await db
				.insert(schema.pieces)
				.values({
					pieceId: r.piece_id as string,
					composer: r.composer as string,
					title: r.title as string,
					keySignature: (r.key_signature as string | null) ?? null,
					timeSignature: (r.time_signature as string | null) ?? null,
					tempoBpm: (r.tempo_bpm as number | null) ?? null,
					barCount: r.bar_count as number,
					durationSeconds: (r.duration_seconds as number | null) ?? null,
					noteCount: r.note_count as number,
					pitchRangeLow: (r.pitch_range_low as number | null) ?? null,
					pitchRangeHigh: (r.pitch_range_high as number | null) ?? null,
					hasTimeSigChanges: toBool(r.has_time_sig_changes as number | null),
					hasTempoChanges: toBool(r.has_tempo_changes as number | null),
					source: (r.source as string | null) ?? "asap",
					createdAt: toDate(r.created_at as string | null) ?? new Date(),
				})
				.onConflictDoNothing();
		}
	});
}

async function migratePieceRequests(): Promise<void> {
	await migrateTable(
		"piece_requests",
		"SELECT * FROM piece_requests",
		async (rows) => {
			for (const row of rows) {
				const r = row as Record<string, unknown>;
				await db
					.insert(schema.pieceRequests)
					.values({
						id: r.id as string,
						query: r.query as string,
						studentId: r.student_id as string,
						matchedPieceId: (r.matched_piece_id as string | null) ?? null,
						matchConfidence: (r.match_confidence as number | null) ?? null,
						matchMethod: (r.match_method as string | null) ?? null,
						createdAt: toDate(r.created_at as string | null) ?? new Date(),
					})
					.onConflictDoNothing();
			}
		},
	);
}

async function migrateWaitlist(): Promise<void> {
	await migrateTable("waitlist", "SELECT * FROM waitlist", async (rows) => {
		for (const row of rows) {
			const r = row as Record<string, unknown>;
			await db
				.insert(schema.waitlist)
				.values({
					email: r.email as string,
					context: (r.context as string | null) ?? null,
					source: (r.source as string | null) ?? "web",
					createdAt: toDate(r.created_at as string | null) ?? new Date(),
				})
				.onConflictDoNothing();
		}
	});
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
	console.log("Starting D1 -> Postgres migration...\n");

	// FK-safe order
	await migrateStudents();
	await migrateAuthIdentities();
	await migrateSessions();
	await migrateStudentCheckIns();
	await migrateConversations();
	await migrateMessages();
	await migrateObservations();
	await migrateTeachingApproaches();
	await migrateSynthesizedFacts();
	await migrateStudentMemoryMeta();
	await migrateExercises();
	await migrateExerciseDimensions();
	await migrateStudentExercises();
	await migratePieces();
	await migratePieceRequests();
	await migrateWaitlist();

	console.log("\nMigration complete!");
	await sql.end();
}

main().catch((err) => {
	console.error("Migration failed:", err);
	process.exit(1);
});
