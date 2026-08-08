export interface ConfidentGuess {
	readonly pieceId: string;
	readonly composer: string;
	readonly title: string;
	readonly confidence: number;
}

export interface LadderInput {
	readonly userPicked: string | null;
	readonly confidentGuess: ConfidentGuess | null;
	readonly dismissed: boolean;
}

export type LadderState = "user-picked" | "confirm-chip" | "pieceless";

/**
 * The piece resolution ladder (docs/apps/02-pipeline.md#3): user pick beats
 * a confident guess, and a dismissed guess never resurfaces — there is no
 * fourth state to re-summon it mid-session.
 */
export function resolvePieceLadderState(input: LadderInput): LadderState {
	if (input.userPicked) return "user-picked";
	if (input.confidentGuess && !input.dismissed) return "confirm-chip";
	return "pieceless";
}
