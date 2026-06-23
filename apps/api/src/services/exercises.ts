import { and, eq, sql } from "drizzle-orm";
import {
	exerciseDimensions,
	exercises,
	pendingExercises,
	studentExercises,
} from "../db/schema/exercises";
import { ExerciseRoutingDecisionSchema } from "../harness/artifacts/exercise-routing";
import { InferenceError, NotFoundError } from "../lib/errors";
import type { ServiceContext } from "../lib/types";
import { buildCorpusDrillClip } from "./corpus-drill";

export type ExerciseSetPayload = {
	sourcePassage: string;
	targetSkill: string;
	scoreClip?: { pieceId: string; bars: [number, number]; tempoFactor?: number; transpose?: number };
	exercises: Array<{
		title: string;
		instruction: string;
		focusDimension: string;
		hands?: "left" | "right" | "both";
		exerciseId?: string;
	}>;
};

export async function listExercises(
	ctx: ServiceContext,
	filters: {
		studentId: string;
		dimension?: string;
		level?: string;
		repertoire?: string;
	},
) {
	let query = ctx.db
		.selectDistinct({
			id: exercises.id,
			title: exercises.title,
			description: exercises.description,
			instructions: exercises.instructions,
			difficulty: exercises.difficulty,
			category: exercises.category,
			repertoireTags: exercises.repertoireTags,
			notationContent: exercises.notationContent,
			notationFormat: exercises.notationFormat,
			midiContent: exercises.midiContent,
			source: exercises.source,
			variantsJson: exercises.variantsJson,
			createdAt: exercises.createdAt,
		})
		.from(exercises)
		.innerJoin(
			exerciseDimensions,
			eq(exerciseDimensions.exerciseId, exercises.id),
		)
		.$dynamic();

	if (filters.dimension) {
		query = query.where(eq(exerciseDimensions.dimension, filters.dimension));
	}

	if (filters.level) {
		query = query.where(eq(exercises.difficulty, filters.level));
	}

	const rows = await query.orderBy(exercises.title).limit(3);

	if (rows.length === 0) {
		return [];
	}

	const exerciseIds = rows.map((r) => r.id);

	const dimensionRows = await ctx.db
		.select({
			exerciseId: exerciseDimensions.exerciseId,
			dimension: exerciseDimensions.dimension,
		})
		.from(exerciseDimensions)
		.where(sql`${exerciseDimensions.exerciseId} = ANY(${exerciseIds})`);

	const dimsByExercise = new Map<string, string[]>();
	for (const row of dimensionRows) {
		const existing = dimsByExercise.get(row.exerciseId) ?? [];
		existing.push(row.dimension);
		dimsByExercise.set(row.exerciseId, existing);
	}

	return rows.map((r) => ({
		...r,
		dimensions: dimsByExercise.get(r.id) ?? [],
	}));
}

export async function assignExercise(
	ctx: ServiceContext,
	data: { studentId: string; exerciseId: string; sessionId?: string },
) {
	const exercise = await ctx.db.query.exercises.findFirst({
		where: (e, { eq }) => eq(e.id, data.exerciseId),
	});

	if (!exercise) {
		throw new NotFoundError("exercise", data.exerciseId);
	}

	const [row] = await ctx.db
		.insert(studentExercises)
		.values({
			studentId: data.studentId,
			exerciseId: data.exerciseId,
			sessionId: data.sessionId ?? null,
		})
		.onConflictDoUpdate({
			target: [
				studentExercises.studentId,
				studentExercises.exerciseId,
				studentExercises.sessionId,
			],
			set: {
				timesAssigned: sql`${studentExercises.timesAssigned} + 1`,
			},
		})
		.returning();

	return row;
}

export async function completeExercise(
	ctx: ServiceContext,
	data: {
		studentExerciseId: string;
		studentId: string;
		response?: string;
		dimensionBeforeJson?: unknown;
		dimensionAfterJson?: unknown;
		notes?: string;
	},
) {
	const existing = await ctx.db.query.studentExercises.findFirst({
		where: (se, { and, eq }) =>
			and(eq(se.id, data.studentExerciseId), eq(se.studentId, data.studentId)),
	});

	if (!existing) {
		throw new NotFoundError("studentExercise", data.studentExerciseId);
	}

	const [row] = await ctx.db
		.update(studentExercises)
		.set({
			completed: true,
			...(data.response !== undefined && { response: data.response }),
			...(data.dimensionBeforeJson !== undefined && {
				dimensionBeforeJson: data.dimensionBeforeJson,
			}),
			...(data.dimensionAfterJson !== undefined && {
				dimensionAfterJson: data.dimensionAfterJson,
			}),
			...(data.notes !== undefined && { notes: data.notes }),
		})
		.where(eq(studentExercises.id, data.studentExerciseId))
		.returning();

	return row;
}

export async function assignPendingExercise(
	ctx: ServiceContext,
	args: {
		studentId: string;
		sessionId: string;
		exerciseId: string; // = pending_exercises.id (the pending row PK)
	},
): Promise<ExerciseSetPayload> {
	const [pendingRow] = await ctx.db
		.select()
		.from(pendingExercises)
		.where(
			and(
				eq(pendingExercises.studentId, args.studentId),
				eq(pendingExercises.id, args.exerciseId),
				eq(pendingExercises.consumed, false),
			),
		);

	if (!pendingRow) {
		throw new NotFoundError("pending exercise", args.exerciseId);
	}

	if (!pendingRow.routingJson) {
		throw new InferenceError(
			`pending exercise ${args.exerciseId} has no routing_json`,
		);
	}

	const routingResult = ExerciseRoutingDecisionSchema.safeParse(
		pendingRow.routingJson,
	);
	if (!routingResult.success) {
		throw new InferenceError(
			`pending exercise ${args.exerciseId} routing_json invalid: ${routingResult.error.message}`,
		);
	}
	const routing = routingResult.data;

	// TODO(S3): wire studentExercises tracking for routed exercises when the catalog
	// has a real exercise row to reference.
	// Do NOT call assignExercise here — it would throw NotFoundError because
	// the pending row id is not in the exercises catalog table.

	await ctx.db
		.update(pendingExercises)
		.set({ consumed: true })
		.where(eq(pendingExercises.id, pendingRow.id));

	const title = pendingRow.title ?? pendingRow.previewTitle;
	const instruction =
		pendingRow.instruction ??
		`${pendingRow.focusDimension} exercise — bars ${routing.bar_range[0]}-${routing.bar_range[1]}`;

	if (routing.kind === "own_passage_loop") {
		const pieceId = pendingRow.pieceId ?? null;
		const scoreClip =
			pieceId !== null
				? { pieceId, bars: routing.bar_range as [number, number], tempoFactor: routing.tempo_factor }
				: undefined;

		if (!scoreClip) {
			console.log(
				JSON.stringify({
					level: "warn",
					message:
						"assignPendingExercise: own_passage_loop has no piece_id; rendering text-only",
					exerciseId: args.exerciseId,
				}),
			);
		}

		return {
			sourcePassage: `bars ${routing.bar_range[0]}-${routing.bar_range[1]}`,
			targetSkill: pendingRow.focusDimension,
			scoreClip,
			exercises: [
				{
					title,
					instruction,
					focusDimension: pendingRow.focusDimension,
					exerciseId: pendingRow.id,
				},
			],
		};
	}

	// corpus_drill — render a matched primitive clip (transposed into the
	// student's key when resolvable). pieceId is the STUDENT passage piece used
	// only for key resolution; the clip itself is the primitive.
	return buildCorpusDrillClip(ctx, routing, pendingRow.pieceId ?? null);
}
