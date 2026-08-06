import type { Mark } from "../lib/mark";
import { resolveAnchor } from "../lib/mark";
import type { BarLocator } from "../lib/mark-placement";

export const FIXTURE_DURATION_SECONDS = 360;

export const FIXTURE_MARKS: readonly Mark[] = [
	{
		id: "m1",
		anchor: resolveAnchor({
			atSeconds: 64,
			bars: [5, 6],
			alignmentQuality: 0.95,
		}),
		taxonomy: "needs_work",
		dimension: "pedaling",
		evidence:
			"pedal held through the bass change at 5.3; the blur between hands is about three times your usual",
		lifecycle: "active",
		confidence: "established",
	},
	{
		id: "m2",
		anchor: resolveAnchor({
			atSeconds: 151,
			bars: [12, 12],
			alignmentQuality: 0.91,
		}),
		taxonomy: "missed_opportunity",
		dimension: "dynamics",
		evidence:
			"the approach to 12 stayed flat; the phrase is asking for more shape",
		lifecycle: "improving",
		confidence: "provisional",
	},
	{
		id: "m3",
		anchor: resolveAnchor({
			atSeconds: 252,
			bars: [30, 32],
			alignmentQuality: 0.88,
		}),
		taxonomy: "strong",
		dimension: "phrasing",
		evidence: "the line breathes across 30 to 32 exactly as the slur asks",
		lifecycle: "resolved",
		confidence: "established",
	},
	{
		// Bars WERE supplied ([21, 22]) and resolveAnchor discarded them. This is
		// the fixture that proves a wrong bar number cannot reach the screen.
		id: "m4",
		anchor: resolveAnchor({
			atSeconds: 97,
			bars: [21, 22],
			alignmentQuality: 0.31,
		}),
		taxonomy: "needs_work",
		dimension: "timing",
		evidence: "the left hand lagged behind the right through this passage",
		lifecycle: "active",
		confidence: "exploratory",
	},
	{
		// Bar 88 resolves, but it is not on the rendered page: Canvas A must
		// disclose it rather than draw it, and Canvas B must still show it.
		id: "m5",
		anchor: resolveAnchor({
			atSeconds: 305,
			bars: [88, 89],
			alignmentQuality: 0.97,
		}),
		taxonomy: "needs_work",
		dimension: "articulation",
		evidence: "the staccato flattened into portato here",
		lifecycle: "active",
		confidence: "established",
	},
	{
		id: "m6",
		anchor: resolveAnchor({ atSeconds: 12, alignmentQuality: 1 }),
		taxonomy: "missed_opportunity",
		dimension: "interpretation",
		evidence: "the opening stated the theme without committing to a character",
		lifecycle: "improving",
		confidence: "exploratory",
	},
];

/**
 * Bar locators as score-ir.ts produces them. measureOn is the id attribute of
 * the measure <g> in the rendered Verovio SVG. Bar 88 is present here but its
 * element is absent from the rendered page in tests — that asymmetry is the
 * point.
 */
export const FIXTURE_BARS: readonly BarLocator[] = [
	{ barNumber: 5, measureOn: "measure-0000000000000005" },
	{ barNumber: 12, measureOn: "measure-0000000000000012" },
	{ barNumber: 30, measureOn: "measure-0000000000000030" },
	{ barNumber: 88, measureOn: "measure-0000000000000088" },
];
