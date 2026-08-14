import type { Mark } from "../lib/mark";
import { anchorLabel } from "../lib/mark";
import { MarkPassagePlayback } from "./MarkPassagePlayback";

interface MarkReviewDetailProps {
	mark: Mark;
	sessionId: string;
	onClose: () => void;
	onWorkOnThis: (mark: Mark) => void;
}

/**
 * The review-only expansion: evidence + the student's own recorded audio for
 * this mark's bars (cursor-tracked) + a drill entry point. Wired into
 * ScoreMarkLayer/SessionTimelineStrip via their `renderDetail` prop, so live
 * pause-taps keep today's evidence-only MarkDetail unchanged.
 */
export function MarkReviewDetail({
	mark,
	sessionId,
	onClose,
	onWorkOnThis,
}: MarkReviewDetailProps) {
	const framing = mark.confidence === "exploratory" ? "Early read — " : "";

	return (
		<section
			className="mt-1 rounded-md border border-border-subtle bg-surface-raised p-3"
			aria-label={`Evidence, ${anchorLabel(mark.anchor)}`}
		>
			<p className="text-body-sm text-ink-secondary">
				{framing}
				{mark.evidence}
			</p>
			<MarkPassagePlayback mark={mark} sessionId={sessionId} />
			<div className="mt-3 flex items-center gap-3">
				<button
					type="button"
					onClick={() => onWorkOnThis(mark)}
					className="rounded-full border border-border-subtle px-3 py-1.5 text-body-sm text-ink-primary"
				>
					Work on this
				</button>
				<button
					type="button"
					onClick={onClose}
					className="text-label-sm text-ink-tertiary underline"
				>
					Close
				</button>
			</div>
		</section>
	);
}
