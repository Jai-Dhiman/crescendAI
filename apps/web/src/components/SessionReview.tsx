import type { Mark } from "../lib/mark";
import type {
	CarryForward,
	NoCommentZone,
	VerdictState,
	WsGap,
} from "../lib/session-review";
import { MarkReviewDetail } from "./MarkReviewDetail";
import { PieceLessMode } from "./PieceLessMode";
import { ScoreStand } from "./ScoreStand";
import { SessionDisclosures } from "./SessionDisclosures";
import { VerdictBanner } from "./VerdictBanner";

interface SessionReviewProps {
	pieceId: string | null;
	sessionId: string;
	marks: readonly Mark[];
	durationSeconds: number;
	verdict: VerdictState;
	onRetryVerdict: () => void;
	carryForward: CarryForward | null;
	wsGaps: readonly WsGap[];
	noCommentZones: readonly NoCommentZone[];
	onWorkOnThis: (mark: Mark) => void;
}

/**
 * The review surface (docs/apps/05-ui-system.md#3). Reuses the exact canvas
 * #158 built for live practice -- ScoreStand/PieceLessMode already take
 * `marks: readonly Mark[]` with no live/consolidated distinction, so "marks
 * consolidated" is just "pass every mark." The only thing review changes
 * about the canvas is the expansion: MarkReviewDetail replaces MarkDetail
 * via the `renderDetail` prop both canvases now accept.
 */
export function SessionReview({
	pieceId,
	sessionId,
	marks,
	durationSeconds,
	verdict,
	onRetryVerdict,
	carryForward,
	wsGaps,
	noCommentZones,
	onWorkOnThis,
}: SessionReviewProps) {
	const renderDetail = (mark: Mark, onClose: () => void) => (
		<MarkReviewDetail
			mark={mark}
			sessionId={sessionId}
			onClose={onClose}
			onWorkOnThis={onWorkOnThis}
		/>
	);

	return (
		<div className="flex h-full flex-col gap-3 overflow-auto px-4 py-3">
			<VerdictBanner verdict={verdict} onRetry={onRetryVerdict} />
			<div className="min-h-0 flex-1">
				{pieceId ? (
					<ScoreStand
						pieceId={pieceId}
						marks={marks}
						elapsedSeconds={durationSeconds}
						isRecording={false}
						renderDetail={renderDetail}
					/>
				) : (
					<PieceLessMode
						marks={marks}
						durationSeconds={durationSeconds}
						elapsedSeconds={durationSeconds}
						isRecording={false}
						renderDetail={renderDetail}
					/>
				)}
			</div>
			<SessionDisclosures wsGaps={wsGaps} noCommentZones={noCommentZones} />
			{carryForward && (
				<p className="border-t border-border-subtle pt-3 text-body-sm text-ink-primary">
					Next: {carryForward.text}
				</p>
			)}
		</div>
	);
}
