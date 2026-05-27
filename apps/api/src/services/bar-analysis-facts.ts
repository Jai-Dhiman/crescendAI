import type { Dimension } from "../lib/dims";
import type { ChunkAnalysis, DimensionAnalysis } from "./wasm-bridge";

export interface StudentBaselines {
	dynamics: number;
	timing: number;
	pedaling: number;
	articulation: number;
	phrasing: number;
	interpretation: number;
}

export interface BarAnalysisFacts {
	tier: number;
	bar_range: string | null;
	selected: DimensionAnalysis;
	correlated: DimensionAnalysis[];
}

export function buildBarAnalysisFacts(
	analysis: ChunkAnalysis,
	scoresArray: [number, number, number, number, number, number],
	baselines: StudentBaselines,
	selectedDimension: Dimension,
): BarAnalysisFacts | null {
	if (analysis.dimensions.length === 0) {
		return null;
	}
	// Unreachable in this task; future tasks extend.
	throw new Error("buildBarAnalysisFacts: non-empty case not yet implemented");
}
