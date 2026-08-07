/** A rendered glyph's measured horizontal extent, in container pixels. */
export interface LaneItem {
	readonly id: string;
	readonly left: number;
	readonly width: number;
}

/** Clearance between neighbouring glyphs sharing a lane. */
export const LANE_GAP_PX = 8;

/**
 * Hold a glyph inside the strip that owns it.
 *
 * Elapsed time picks a position before the glyph's width is known, so a mark
 * late in the session runs off the right edge — measured at 36px past a 720px
 * strip for a mark at 84.7%, which also widened the document and made the whole
 * page scroll sideways below 768px. Collision packing cannot see this: it
 * compares marks to each other and never to their container.
 *
 * Clamping rather than scaling keeps the mapping from time to position honest
 * everywhere except the last glyph-width of the strip, where there is no
 * position that is both truthful and inside the box.
 */
export function clampToStrip(
	left: number,
	width: number,
	stripWidth: number,
): number {
	// Before the first measurement stripWidth is 0; leaving `left` alone then
	// avoids slamming every mark to 0 on the frame before layout settles.
	if (stripWidth <= 0 || width <= 0) return left;
	return Math.max(0, Math.min(left, stripWidth - width));
}

/**
 * Pack timeline marks into horizontal lanes so none covers another.
 *
 * Marks positioned purely by elapsed time collide whenever two moments fall
 * close together: measured on the fixture set at 1024px, four pairs overlapped
 * by up to 84px and the covered mark could not be clicked at all — the click
 * landed on whichever sibling sat on top. Visible but untappable is worse than
 * absent, because nothing signals that a mark is being swallowed.
 *
 * Pure, and takes measured widths rather than reading the DOM, for the same
 * reason mark-placement.ts does: jsdom has no layout engine, so a version that
 * measured internally could not be tested at all.
 */
export function assignLanes(
	items: readonly LaneItem[],
	gapPx: number = LANE_GAP_PX,
): Map<string, number> {
	// Left-to-right, so a lane's occupancy is a single right edge rather than a
	// list of intervals to search.
	const ordered = [...items].sort((a, b) => a.left - b.left);
	const laneRightEdges: number[] = [];
	const lanes = new Map<string, number>();

	for (const item of ordered) {
		let lane = laneRightEdges.findIndex((edge) => edge + gapPx <= item.left);
		if (lane === -1) {
			lane = laneRightEdges.length;
		}
		laneRightEdges[lane] = item.left + item.width;
		lanes.set(item.id, lane);
	}

	return lanes;
}
