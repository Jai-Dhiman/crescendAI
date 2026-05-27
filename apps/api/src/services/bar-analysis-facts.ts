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

const DIM_ORDER: Dimension[] = [
	"dynamics",
	"timing",
	"pedaling",
	"articulation",
	"phrasing",
	"interpretation",
];

export function buildBarAnalysisFacts(
	analysis: ChunkAnalysis,
	scoresArray: [number, number, number, number, number, number],
	baselines: StudentBaselines,
	selectedDimension: Dimension,
): BarAnalysisFacts | null {
	if (analysis.dimensions.length === 0) {
		return null;
	}
	const selected = analysis.dimensions.find(
		(d) => d.dimension === selectedDimension,
	);
	if (selected === undefined) {
		return null;
	}

	const deviations = DIM_ORDER.map((dim, i) => ({
		dim,
		dev: Math.abs((scoresArray[i] ?? 0) - baselines[dim]),
	}));

	const correlated = analysis.dimensions
		.filter((d) => d.dimension !== selectedDimension)
		.map((d) => {
			const entry = deviations.find((e) => e.dim === d.dimension);
			return { d, dev: entry?.dev ?? 0 };
		})
		.filter((x) => x.dev >= 0.15)
		.sort((a, b) => b.dev - a.dev)
		.slice(0, 2)
		.map((x) => x.d);

	return {
		tier: analysis.tier,
		bar_range: analysis.bar_range,
		selected,
		correlated,
	};
}
