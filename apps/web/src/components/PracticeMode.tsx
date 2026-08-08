import { useState } from "react";
import { usePauseTracker } from "../hooks/usePauseTracker";
import type { Mark } from "../lib/mark";
import type { ConfidentGuess } from "../lib/piece-ladder";
import { resolvePieceLadderState } from "../lib/piece-ladder";
import { ConfirmPieceChip } from "./ConfirmPieceChip";
import { PieceLessMode } from "./PieceLessMode";
import { ScoreStand } from "./ScoreStand";
import { SessionEndedBanner } from "./SessionEndedBanner";

interface PracticeModeProps {
	userPickedPieceId: string | null;
	confidentGuess: ConfidentGuess | null;
	marks: readonly Mark[];
	elapsedSeconds: number;
	isPlaying: boolean;
	isRecording: boolean;
	/** Ends the session for real. The only exit from this full-screen surface;
	 * unlike SessionEndedBanner's resume, this is terminal. */
	onStop: () => void;
}

/**
 * The orchestrator: the one component that knows all four practice
 * sub-surfaces exist. Everything it delegates to (ScoreStand, PieceLessMode,
 * ConfirmPieceChip, SessionEndedBanner) takes plain props and touches
 * neither the WS nor the session hook directly -- AppChat is the only place
 * that wires usePracticeSession's live state into these props.
 *
 * The stop control is rendered here, not inside ScoreStand/PieceLessMode/
 * SessionEndedBanner, so it is guaranteed present across every ladder state
 * and across the auto-stopped banner. It lives in its own `shrink-0` header
 * row, stacked in normal document flow above a `flex-1` content region --
 * not an absolute overlay pinned to a corner. Two of the three sub-surfaces
 * put their own primary control in that same top-right corner (ScoreStand's
 * Metronome toggle, ConfirmPieceChip's Dismiss button), so an absolute/
 * z-indexed Stop button would sit in the same box as one of them and could
 * cover -- and steal clicks from -- whichever is underneath. Reserving Stop
 * its own row makes the separation a layout guarantee instead of a
 * stacking-order one: every sub-surface's own header renders strictly below
 * it, never behind it.
 */
export function PracticeMode({
	userPickedPieceId,
	confidentGuess,
	marks,
	elapsedSeconds,
	isPlaying,
	isRecording,
	onStop,
}: PracticeModeProps) {
	const [dismissed, setDismissed] = useState(false);
	const pause = usePauseTracker(isPlaying);

	const ladderState = resolvePieceLadderState({
		userPicked: userPickedPieceId,
		confidentGuess,
		dismissed,
	});

	const pieceId =
		ladderState === "user-picked"
			? userPickedPieceId
			: ladderState === "confirm-chip"
				? (confidentGuess?.pieceId ?? null)
				: null;

	return (
		<div className="flex h-full flex-col">
			<div className="flex shrink-0 items-center justify-end border-b border-border-subtle px-4 py-2">
				<button
					type="button"
					onClick={onStop}
					aria-label="Stop recording"
					className="rounded-full bg-danger px-4 py-2 text-body-sm text-on-accent"
				>
					Stop
				</button>
			</div>
			<div className="flex min-h-0 flex-1 flex-col">
				{pause.autoStopped ? (
					<SessionEndedBanner onResume={pause.resume} />
				) : (
					<>
						{ladderState === "confirm-chip" && confidentGuess && (
							<ConfirmPieceChip
								guess={confidentGuess}
								onDismiss={() => setDismissed(true)}
							/>
						)}
						{pieceId ? (
							<ScoreStand
								pieceId={pieceId}
								marks={marks}
								elapsedSeconds={elapsedSeconds}
								isRecording={isRecording}
							/>
						) : (
							<PieceLessMode
								marks={marks}
								durationSeconds={Math.max(elapsedSeconds, 1)}
								elapsedSeconds={elapsedSeconds}
								isRecording={isRecording}
							/>
						)}
					</>
				)}
			</div>
		</div>
	);
}
