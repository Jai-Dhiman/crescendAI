import type { ObservationEvent } from "./practice-api";

export interface MockSessionData {
	piece: string;
	section: string;
	durationSeconds: number;
	observations: ObservationEvent[];
}

export type Dimension =
	| "dynamics"
	| "timing"
	| "pedaling"
	| "articulation"
	| "phrasing"
	| "interpretation";

export const DIMENSION_LABELS: Record<Dimension, string> = {
	dynamics: "Dynamics",
	timing: "Timing",
	pedaling: "Pedaling",
	articulation: "Articulation",
	phrasing: "Phrasing",
	interpretation: "Interpretation",
};
