import { useState } from "react";
import type { Mark } from "../lib/mark";
import { MarkDetail } from "./MarkDetail";
import { MarkGlyph } from "./MarkGlyph";

interface SessionTimelineStripProps {
	durationSeconds: number;
	marks: readonly Mark[];
}

/**
 * Canvas B: the complete view.
 *
 * Every mark appears here, including bar-anchored marks the score canvas could
 * not place. Elapsed time is the one coordinate every anchor carries, which is
 * what makes this canvas total and Canvas A the lossy one.
 */
export function SessionTimelineStrip({
	durationSeconds,
	marks,
}: SessionTimelineStripProps) {
	const [expandedId, setExpandedId] = useState<string | null>(null);
	const span = durationSeconds > 0 ? durationSeconds : 1;

	return (
		<div className="relative h-24 w-full border-t border-border-subtle">
			{marks.map((mark) => (
				<div
					key={mark.id}
					className="absolute top-0"
					style={{ left: `${(mark.anchor.atSeconds / span) * 100}%` }}
				>
					<MarkGlyph
						mark={mark}
						expanded={expandedId === mark.id}
						onToggle={(id) => setExpandedId((cur) => (cur === id ? null : id))}
					/>
					{expandedId === mark.id && (
						<MarkDetail mark={mark} onClose={() => setExpandedId(null)} />
					)}
				</div>
			))}
		</div>
	);
}
