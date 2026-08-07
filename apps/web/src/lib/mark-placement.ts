import type { Mark } from "./mark";
import type { BarIR } from "./score-ir";

/**
 * Exactly the part of score-ir's BarIR that placement needs. Reusing the real
 * contract rather than restating it keeps the two from drifting.
 */
export type BarLocator = Pick<BarIR, "barNumber" | "measureOn">;

/** Container-relative geometry, measured by the caller. */
export interface MeasureRect {
	readonly top: number;
	readonly left: number;
	readonly width: number;
	readonly height: number;
}

export interface PlacedMark {
	readonly mark: Mark;
	readonly top: number;
	readonly left: number;
	/** The measure element this mark resolved to. Lets an E2E test assert the
	 * glyph overlaps the bar it names, rather than merely that it rendered. */
	readonly measureOn: string;
}

export interface Placement {
	readonly placed: readonly PlacedMark[];
	readonly unplaced: readonly Mark[];
}

/** Vertical clearance so the glyph sits above the staff rather than on it. */
export const GLYPH_OFFSET_PX = 28;

/**
 * Pure bar-to-pixel mapping. Reads no DOM — the caller measures and passes
 * rects in, which is what makes this testable at all (jsdom has no layout
 * engine, so getBoundingClientRect returns zeros there).
 *
 * There is deliberately no fallback coordinate. A mark this function cannot
 * resolve to a real rect comes back in `unplaced` for the caller to route to
 * the timeline canvas. Inventing a position is the defect this module exists
 * to eliminate.
 */
export function placeMarks(
	bars: readonly BarLocator[],
	rectsByMeasureOn: ReadonlyMap<string, MeasureRect>,
	marks: readonly Mark[],
): Placement {
	const measureOnByBar = new Map(bars.map((b) => [b.barNumber, b.measureOn]));
	const placed: PlacedMark[] = [];
	const unplaced: Mark[] = [];

	for (const mark of marks) {
		if (mark.anchor.type !== "bars") {
			unplaced.push(mark);
			continue;
		}
		const measureOn = measureOnByBar.get(mark.anchor.bars[0]);
		const rect = measureOn ? rectsByMeasureOn.get(measureOn) : undefined;
		// No fallback coordinate. Inventing a position is the defect this
		// module exists to eliminate — see ScorePanel.tsx:371.
		// `rect` is only ever set when `measureOn` was, but testing both is what
		// narrows measureOn to string for the PlacedMark below.
		if (!measureOn || !rect) {
			unplaced.push(mark);
			continue;
		}
		placed.push({
			mark,
			top: rect.top - GLYPH_OFFSET_PX,
			left: rect.left,
			measureOn,
		});
	}

	return { placed, unplaced };
}
