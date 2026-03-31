import { eq, sql } from "drizzle-orm";
import type { ServiceContext } from "../lib/types";
import { NotFoundError } from "../lib/errors";
import {
	exercises,
	exerciseDimensions,
	studentExercises,
} from "../db/schema/exercises";

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
		.where(
			sql`${exerciseDimensions.exerciseId} = ANY(${exerciseIds})`,
		);

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
