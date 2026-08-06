/**
 * The brand is not exported. `resolveAnchor` is therefore the only function in
 * the codebase that can produce a MarkAnchor: no object literal written
 * anywhere else is assignable to this type. That is what makes "wrong bar
 * numbers are never shown" a compile-time property rather than a runtime guard
 * a future surface can route around.
 */
declare const anchorBrand: unique symbol;

type AnchorBrand = { readonly [anchorBrand]: true };

export type MarkAnchor = AnchorBrand &
	(
		| {
				readonly type: "bars";
				readonly bars: readonly [number, number];
				readonly atSeconds: number;
		  }
		| { readonly type: "timestamp"; readonly atSeconds: number }
	);

/**
 * Uncalibrated. There is no distribution of real alignment-quality scores in
 * this repo yet, so this is a starting value chosen to be conservative, in the
 * same spirit as #165's note on DEVIANT_SAMPLE_MULTIPLE. Tune against real
 * alignment output, not intuition.
 */
export const ALIGNMENT_MIN = 0.8;

export interface AnchorCandidate {
	readonly atSeconds: number;
	readonly bars?: readonly [number, number];
	readonly alignmentQuality: number;
}

/**
 * The single degradation function. Every anchor carries atSeconds — including
 * the bars variant — because the timeline canvas must be able to place any
 * mark, and elapsed time is the one coordinate every mark always has.
 */
export function resolveAnchor(candidate: AnchorCandidate): MarkAnchor {
	const { atSeconds, bars, alignmentQuality } = candidate;
	// `>=`, not `>`: ALIGNMENT_MIN is the lowest quality still trusted for
	// bars. The boundary case is pinned by the test above.
	if (bars && alignmentQuality >= ALIGNMENT_MIN) {
		return { type: "bars", bars, atSeconds } as unknown as MarkAnchor;
	}
	return { type: "timestamp", atSeconds } as unknown as MarkAnchor;
}
