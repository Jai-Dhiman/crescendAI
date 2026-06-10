import { pendingExercises } from "../db/schema/exercises";
import { InferenceError } from "../lib/errors";
import type { Db } from "../lib/types";
import type { ExerciseRoutingDecision } from "../harness/artifacts/exercise-routing";
import type { InlineComponent } from "./tool-processor";

export type PendingExercise = {
	exerciseId: string; // = pending_exercises.id (the row PK)
	focusDimension: string;
	previewTitle: string;
};

export async function stageDominantExercise(
	db: Db,
	args: {
		studentId: string;
		sessionId: string;
		dominantDimension: string;
		routing: ExerciseRoutingDecision;
		pieceCtx: { pieceId?: string } | null;
	},
): Promise<PendingExercise> {
	const barLabel = `bars ${args.routing.bar_range[0]}-${args.routing.bar_range[1]}`;
	const title =
		args.routing.kind === "own_passage_loop"
			? `Own passage loop: ${args.dominantDimension} (${barLabel})`
			: `${args.dominantDimension} drill (${barLabel})`;
	const previewTitle = title.slice(0, 60);

	const instruction =
		args.routing.kind === "own_passage_loop"
			? `Loop ${barLabel} from your recording at ${Math.round(args.routing.tempo_factor * 100)}% tempo, focusing on ${args.dominantDimension}.`
			: `${args.dominantDimension} drill — ${barLabel} at ${Math.round(args.routing.tempo_factor * 100)}% tempo.`;

	const pieceId = args.pieceCtx?.pieceId ?? null;

	const [inserted] = await db
		.insert(pendingExercises)
		.values({
			studentId: args.studentId,
			sessionId: args.sessionId,
			focusDimension: args.dominantDimension,
			previewTitle,
			title,
			instruction,
			routingJson: args.routing as unknown as Record<string, unknown>,
			pieceId,
			consumed: false,
		})
		.returning({ id: pendingExercises.id });

	if (!inserted) {
		throw new InferenceError("Failed to insert pending exercise");
	}

	return {
		exerciseId: inserted.id, // pending row PK
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
