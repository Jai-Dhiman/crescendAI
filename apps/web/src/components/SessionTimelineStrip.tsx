import { useCallback, useLayoutEffect, useRef, useState } from "react";
import type { Mark } from "../lib/mark";
import type { LaneItem } from "../lib/timeline-lanes";
import { assignLanes, clampToStrip } from "../lib/timeline-lanes";
import { MarkDetail } from "./MarkDetail";
import { MarkGlyph } from "./MarkGlyph";

/**
 * Vertical pitch between lanes. Fits three lanes inside the strip's h-24.
 *
 * This is only sound while every glyph is one line tall — MarkGlyph is
 * whitespace-nowrap for exactly this reason. A wrapped glyph would be 52px and
 * silently span two lanes, which lane packing (horizontal only) cannot see.
 */
const LANE_HEIGHT_PX = 26;

/** Where a mark ended up: clamped horizontal position, and its packed lane. */
interface MarkPosition {
	readonly left: number;
	readonly lane: number;
}

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
	const [layout, setLayout] = useState<ReadonlyMap<string, MarkPosition>>(
		new Map(),
	);
	const stripRef = useRef<HTMLDivElement>(null);
	const glyphRefs = useRef(new Map<string, HTMLElement>());
	const span = durationSeconds > 0 ? durationSeconds : 1;

	const registerGlyph = useCallback((id: string, el: HTMLElement | null) => {
		if (el) glyphRefs.current.set(id, el);
		else glyphRefs.current.delete(id);
	}, []);

	// Measure widths, then derive every position from them. Widths come from
	// rendered text and cannot be known before layout; positions are then pure
	// arithmetic over (strip width, elapsed time, glyph width).
	//
	// `left` is computed here rather than read back off the rect, because once
	// clamping moves a glyph the measured left IS the clamped left, and feeding
	// that back in would drift a little further every pass. Width is safe to
	// measure: the glyph does not wrap, so it does not depend on position.
	useLayoutEffect(() => {
		const strip = stripRef.current;
		if (!strip) return;

		const measure = () => {
			const stripWidth = strip.getBoundingClientRect().width;
			const items: LaneItem[] = [];
			for (const mark of marks) {
				const el = glyphRefs.current.get(mark.id);
				if (!el) continue;
				// The button, not the wrapper: an open MarkDetail makes the wrapper
				// far wider, which would shove the mark sideways on expand.
				const measured = el.querySelector("button") ?? el;
				const width = measured.getBoundingClientRect().width;
				const atTime = (mark.anchor.atSeconds / span) * stripWidth;
				items.push({
					id: mark.id,
					left: clampToStrip(atTime, width, stripWidth),
					width,
				});
			}
			// Lanes are packed against the CLAMPED positions, so the two agree:
			// packing against pre-clamp positions would declare marks separate
			// that clamping has since pushed together at the right edge.
			const lanes = assignLanes(items);
			setLayout(
				new Map(
					items.map((i) => [
						i.id,
						{ left: i.left, lane: lanes.get(i.id) ?? 0 },
					]),
				),
			);
		};

		measure();
		// Narrowing the strip pushes marks together, so lanes must be recomputed
		// or marks silently start overlapping again at smaller widths.
		const observer = new ResizeObserver(measure);
		observer.observe(strip);
		return () => observer.disconnect();
	}, [marks, span]);

	return (
		<div
			ref={stripRef}
			data-testid="session-timeline"
			className="relative h-24 w-full border-t border-border-subtle"
		>
			{marks.map((mark) => (
				<div
					key={mark.id}
					ref={(el) => registerGlyph(mark.id, el)}
					className="absolute"
					style={{
						left: layout.get(mark.id)?.left ?? 0,
						top: (layout.get(mark.id)?.lane ?? 0) * LANE_HEIGHT_PX,
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
