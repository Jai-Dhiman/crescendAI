import { type RefObject, useEffect, useState } from "react";
import type { Mark } from "../lib/mark";
import type { BarLocator, MeasureRect } from "../lib/mark-placement";
import { placeMarks } from "../lib/mark-placement";
import { MarkDetail } from "./MarkDetail";
import { MarkGlyph } from "./MarkGlyph";

interface ScoreMarkLayerProps {
	containerRef: RefObject<HTMLElement | null>;
	bars: readonly BarLocator[];
	marks: readonly Mark[];
}

/**
 * Canvas A: the DOM adapter over a rendered Verovio SVG.
 *
 * Shallow on purpose. All substance — bar resolution, the no-fallback rule,
 * offsets — lives in mark-placement.ts, because jsdom has no layout engine and
 * anything measured here is untestable. This file only reads rects and hands
 * them over.
 */
export function ScoreMarkLayer({
	containerRef,
	bars,
	marks,
}: ScoreMarkLayerProps) {
	const [rects, setRects] = useState<ReadonlyMap<string, MeasureRect>>(
		new Map(),
	);
	const [expandedId, setExpandedId] = useState<string | null>(null);

	useEffect(() => {
		const el = containerRef.current;
		if (!el) return;

		const measure = () => {
			const base = el.getBoundingClientRect();
			const found = new Map<string, MeasureRect>();
			for (const bar of bars) {
				// Attribute selector rather than getElementById: measureOn ids are
				// Verovio-generated and need no CSS escaping this way, and the
				// lookup stays scoped to this score container.
				const node = el.querySelector(`[id="${bar.measureOn}"]`);
				if (!node) continue;
				const r = node.getBoundingClientRect();
				found.set(bar.measureOn, {
					top: r.top - base.top,
					left: r.left - base.left,
					width: r.width,
					height: r.height,
				});
			}
			setRects(found);
		};

		measure();
		// Verovio reflows on width change, so stale rects would place marks on
		// the wrong bars — the exact defect this module exists to prevent.
		const observer = new ResizeObserver(measure);
		observer.observe(el);
		return () => observer.disconnect();
	}, [containerRef, bars]);

	const { placed, unplaced } = placeMarks(bars, rects, marks);

	return (
		<div className="pointer-events-none absolute inset-0">
			{placed.map(({ mark, top, left, measureOn }) => (
				<div
					key={mark.id}
					className="pointer-events-auto absolute"
					style={{ top, left }}
				>
					<MarkGlyph
						mark={mark}
						measureOn={measureOn}
						expanded={expandedId === mark.id}
						onToggle={(id) => setExpandedId((cur) => (cur === id ? null : id))}
					/>
					{expandedId === mark.id && (
						<MarkDetail mark={mark} onClose={() => setExpandedId(null)} />
					)}
				</div>
			))}
			{unplaced.length > 0 && (
				// Carries its own surface for the same reason MarkGlyph does: this
				// sits over the Verovio engraving, and the score paper is white in
				// BOTH themes, so dark-theme ink on it fails 4.5:1. Real-browser
				// axe caught this; the token pair itself is fine on a real surface.
				<p className="pointer-events-auto absolute bottom-0 left-0 rounded bg-surface-raised px-2 py-0.5 text-label-sm text-ink-tertiary">
					{/* "not on this page" asserted a reason that is false for most of
					    these: only a bar-anchored mark on another page is off-page,
					    while a timestamp-anchored mark has no page at all. The count
					    was honest; the explanation was not. */}
					{unplaced.length === 1
						? "1 mark on the timeline only"
						: `${unplaced.length} marks on the timeline only`}
				</p>
			)}
		</div>
	);
}
