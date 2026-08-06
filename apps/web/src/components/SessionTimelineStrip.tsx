import { useCallback, useLayoutEffect, useRef, useState } from "react";
import type { Mark } from "../lib/mark";
import type { LaneItem } from "../lib/timeline-lanes";
import { assignLanes } from "../lib/timeline-lanes";
import { MarkDetail } from "./MarkDetail";
import { MarkGlyph } from "./MarkGlyph";

/** Vertical pitch between lanes. Fits three lanes inside the strip's h-24. */
const LANE_HEIGHT_PX = 26;

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
	const [lanes, setLanes] = useState<ReadonlyMap<string, number>>(new Map());
	const stripRef = useRef<HTMLDivElement>(null);
	const glyphRefs = useRef(new Map<string, HTMLElement>());
	const span = durationSeconds > 0 ? durationSeconds : 1;

	const registerGlyph = useCallback((id: string, el: HTMLElement | null) => {
		if (el) glyphRefs.current.set(id, el);
		else glyphRefs.current.delete(id);
	}, []);

	// Measure widths, then lane-pack. Widths come from rendered text, so they
	// cannot be known before layout — and marks close in time otherwise cover
	// each other, which makes the covered one untappable rather than merely
	// ugly. Only `top` changes as a result, and `top` does not affect width,
	// so this settles in one pass instead of looping.
	useLayoutEffect(() => {
		const strip = stripRef.current;
		if (!strip) return;

		const measure = () => {
			const base = strip.getBoundingClientRect();
			const items: LaneItem[] = [];
			for (const mark of marks) {
				const el = glyphRefs.current.get(mark.id);
				if (!el) continue;
				const r = el.getBoundingClientRect();
				items.push({ id: mark.id, left: r.left - base.left, width: r.width });
			}
			setLanes(assignLanes(items));
		};

		measure();
		// Narrowing the strip pushes marks together, so lanes must be recomputed
		// or marks silently start overlapping again at smaller widths.
		const observer = new ResizeObserver(measure);
		observer.observe(strip);
		return () => observer.disconnect();
	}, [marks]);

	return (
		<div
			ref={stripRef}
			className="relative h-24 w-full border-t border-border-subtle"
		>
			{marks.map((mark) => (
				<div
					key={mark.id}
					ref={(el) => registerGlyph(mark.id, el)}
					className="absolute"
					style={{
						left: `${(mark.anchor.atSeconds / span) * 100}%`,
						top: (lanes.get(mark.id) ?? 0) * LANE_HEIGHT_PX,
					}}
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
