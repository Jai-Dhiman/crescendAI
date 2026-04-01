import type { ObservationEvent } from "./practice-api";

export interface MockSessionData {
	piece: string;
	section: string;
	durationSeconds: number;
	observations: ObservationEvent[];
}

type Dimension =
	| "dynamics"
	| "timing"
	| "pedaling"
	| "articulation"
	| "phrasing"
	| "interpretation";

export const DIMENSION_COLORS: Record<Dimension, string> = {
	dynamics: "#b0816a",
	timing: "#9a8a7a",
	pedaling: "#7a8a9a",
	articulation: "#9a7a8a",
	phrasing: "#8a9a7a",
	interpretation: "#9a917a",
};

export const DIMENSION_LABELS: Record<Dimension, string> = {
	dynamics: "Dynamics",
	timing: "Timing",
	pedaling: "Pedaling",
	articulation: "Articulation",
	phrasing: "Phrasing",
	interpretation: "Interpretation",
};
