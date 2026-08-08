import type { Dimension } from "./mock-session";

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

export function formatElapsed(totalSeconds: number): string {
	const minutes = Math.floor(totalSeconds / 60);
	const seconds = Math.floor(totalSeconds % 60);
	return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

/** The one place an anchor becomes words. Both canvases call it. */
export function anchorLabel(anchor: MarkAnchor): string {
	if (anchor.type === "bars") {
		const [start, end] = anchor.bars;
		return start === end ? `bar ${start}` : `bars ${start}-${end}`;
	}
	return formatElapsed(anchor.atSeconds);
}

export type MarkTaxonomy = "needs_work" | "missed_opportunity" | "strong";

/** The three mark-worthy values of #163's Lifecycle. `absent` produces no mark. */
export type MarkLifecycle = "active" | "improving" | "resolved";

/** Mirrors #163's Lifecycle at apps/api/src/services/student-baseline.ts. */
export type BaselineLifecycle = "absent" | MarkLifecycle;

/** Display hint only. Never gates rendering, placement, or visibility. */
export type MarkConfidence = "exploratory" | "provisional" | "established";

export interface Mark {
	readonly id: string;
	readonly anchor: MarkAnchor;
	readonly taxonomy: MarkTaxonomy;
	readonly dimension: Dimension;
	readonly evidence: string;
	readonly lifecycle: MarkLifecycle;
	readonly confidence?: MarkConfidence;
}

/**
 * The single derivation of mark-worthiness. #157 deliberately has no
 * `markWorthy` field: two copies of one fact drift.
 */
export function isMarkWorthy(lifecycle: BaselineLifecycle): boolean {
	return lifecycle !== "absent";
}

export const TAXONOMY_GLYPH: Readonly<Record<MarkTaxonomy, string>> = {
	needs_work: "◉",
	missed_opportunity: "○",
	strong: "★",
};

export const TAXONOMY_LABEL: Readonly<Record<MarkTaxonomy, string>> = {
	needs_work: "Needs work",
	missed_opportunity: "Missed opportunity",
	strong: "Strong",
};

/**
 * Lifecycle -> visual strength. A lookup, never a computation: the client is
 * forbidden from deriving or transitioning lifecycle, which is server state.
 */
export const LIFECYCLE_OPACITY: Readonly<Record<MarkLifecycle, number>> = {
	active: 1,
	improving: 0.7,
	resolved: 0.4,
};
