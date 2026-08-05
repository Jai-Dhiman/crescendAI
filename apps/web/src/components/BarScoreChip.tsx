// apps/web/src/components/BarScoreChip.tsx
import type { KeyboardEvent } from "react";
import { DIMENSION_COLOR_VAR } from "../lib/dimension-colors";
import type { BarQualityScores } from "../types/landing";

const DIMENSIONS: Array<keyof BarQualityScores> = [
	"dynamics",
	"timing",
	"pedaling",
	"articulation",
	"phrasing",
	"interpretation",
];

interface BarScoreChipProps {
	scores: BarQualityScores;
	barNumber: number;
	onClose: () => void;
}

export function BarScoreChip({
	scores,
	barNumber,
	onClose,
}: BarScoreChipProps) {
	function handleKeyDown(e: KeyboardEvent<HTMLDivElement>) {
		if (e.key === "Escape") onClose();
	}

	return (
		<div
			data-testid="bar-score-chip"
			className="bg-surface-page border border-border-subtle rounded-lg p-3 shadow-lg min-w-[180px]"
			role="dialog"
			aria-modal="true"
			aria-label={`Quality scores for bar ${barNumber}`}
			tabIndex={-1}
			onKeyDown={handleKeyDown}
		>
			<div className="flex items-center justify-between mb-2">
				<span className="text-label-sm text-ink-tertiary">Bar {barNumber}</span>
				<button
					type="button"
					onClick={onClose}
					aria-label="Close bar scores"
					className="text-ink-tertiary hover:text-ink-primary text-xs"
				>
					&#x2715;
				</button>
			</div>
			<div className="flex items-end gap-1.5 h-12">
				{DIMENSIONS.map((dim) => {
					const value = Math.max(0, Math.min(1, scores[dim]));
					return (
						<div
							key={dim}
							className="flex flex-col items-center gap-0.5 flex-1"
						>
							<div
								className="w-full rounded-sm"
								style={{
									height: `${Math.round(value * 40)}px`,
									backgroundColor: DIMENSION_COLOR_VAR[dim],
									opacity: 0.85,
								}}
								title={`${dim}: ${Math.round(value * 100)}%`}
							/>
						</div>
					);
				})}
			</div>
			<div className="flex justify-between mt-1.5">
				{DIMENSIONS.map((dim) => (
					<span key={dim} className="text-[9px] text-ink-tertiary capitalize">
						<span aria-hidden="true">{dim.slice(0, 3)}</span>
						<span className="sr-only">{dim}</span>
					</span>
				))}
			</div>
		</div>
	);
}
