import { formatElapsed } from "../lib/mark";
import type { NoCommentZone, WsGap } from "../lib/session-review";
import { formatWsGap } from "../lib/session-review";

interface SessionDisclosuresProps {
	wsGaps: readonly WsGap[];
	noCommentZones: readonly NoCommentZone[];
}

/**
 * Renders nothing when both arrays are empty: disclosure is additive, never
 * implying a problem a clean session didn't have.
 */
export function SessionDisclosures({
	wsGaps,
	noCommentZones,
}: SessionDisclosuresProps) {
	if (wsGaps.length === 0 && noCommentZones.length === 0) return null;

	return (
		<ul
			aria-label="Session disclosures"
			className="flex flex-col gap-1 text-body-xs text-ink-tertiary"
		>
			{wsGaps.map((gap) => (
				<li key={`${gap.startSeconds}-${gap.endSeconds}`}>
					{formatWsGap(gap)}
				</li>
			))}
			{noCommentZones.map((zone) => (
				<li key={zone.atSeconds}>
					No comment near {formatElapsed(zone.atSeconds)} — {zone.reason}
				</li>
			))}
		</ul>
	);
}
