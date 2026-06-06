import {
	exerciseDimensions,
	exercises,
	pendingExercises,
} from "../db/schema/exercises";
import { InferenceError } from "../lib/errors";
import type { Db } from "../lib/types";
import type { InlineComponent } from "./tool-processor";

export type PendingExercise = {
	exerciseId: string;
	focusDimension: string;
	previewTitle: string;
};

export async function stageDominantExercise(
	db: Db,
	args: {
		studentId: string;
		sessionId: string;
		dominantDimension: string;
		proposedExercise: string;
		pieceMetadata: { title?: string; composer?: string } | null;
	},
): Promise<PendingExercise> {
	const trimmed = args.proposedExercise.trim();
	const previewTitle =
		trimmed.length > 0
			? trimmed.slice(0, 60)
			: `${args.dominantDimension} focus drill`;

	const [inserted] = await db
		.insert(exercises)
		.values({
			title: previewTitle,
			description: "Staged from session synthesis",
			instructions: args.proposedExercise,
			difficulty: "intermediate",
			category: "generated",
			source: "teacher_llm",
		})
		.returning({ id: exercises.id });

	if (!inserted) {
		throw new InferenceError("Failed to insert staged exercise");
	}

	await db.insert(exerciseDimensions).values({
		exerciseId: inserted.id,
		dimension: args.dominantDimension,
	});

	await db.insert(pendingExercises).values({
		studentId: args.studentId,
		sessionId: args.sessionId,
		exerciseId: inserted.id,
		focusDimension: args.dominantDimension,
		previewTitle,
		consumed: false,
	});

	return {
		exerciseId: inserted.id,
		focusDimension: args.dominantDimension,
		previewTitle,
	};
}

export function buildPendingExerciseComponent(
	staged: PendingExercise,
): InlineComponent {
	return {
		type: "pending_exercise",
		config: {
			exerciseId: staged.exerciseId,
			focusDimension: staged.focusDimension,
			previewTitle: staged.previewTitle,
		},
	};
}
