/** A rendered glyph's measured horizontal extent, in container pixels. */
export interface LaneItem {
	readonly id: string;
	readonly left: number;
	readonly width: number;
}

/** Clearance between neighbouring glyphs sharing a lane. */
export const LANE_GAP_PX = 8;

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
