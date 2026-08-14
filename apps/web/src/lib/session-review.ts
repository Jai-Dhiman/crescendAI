/** Closed union so "loud, never blank" is a compile-time property: an
 * unhandled variant is a type error in VerdictBanner's switch, not a
 * silently-missing render. */
export type VerdictState =
	| { status: "loading" }
	| { status: "ready"; text: string }
	| { status: "failed" };

/** A single object, not an array -- "exactly one carry-forward" is enforced
 * by the shape, the same way mark.ts's branded MarkAnchor enforces "no wrong
 * bar numbers" structurally rather than with a runtime guard. */
export interface CarryForward {
	readonly text: string;
}

export interface WsGap {
	readonly startSeconds: number;
	readonly endSeconds: number;
}

export interface NoCommentZone {
	readonly atSeconds: number;
	readonly reason: string;
}

/** The one place a WS gap becomes words. */
export function formatWsGap(gap: WsGap): string {
	const seconds = gap.endSeconds - gap.startSeconds;
	if (seconds < 60) {
		return "under a minute didn't reach me";
	}
	const minutes = Math.round(seconds / 60);
	const unit = minutes === 1 ? "minute" : "minutes";
	return `${minutes} ${unit} didn't reach me`;
}
