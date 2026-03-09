import { useState } from "react";
import type { ExerciseSetConfig } from "../../lib/types";

interface ExerciseSetCardProps {
	config: ExerciseSetConfig;
}

export function ExerciseSetCard({ config }: ExerciseSetCardProps) {
	const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

	return (
		<div className="bg-surface-card border border-border rounded-xl p-4 mt-3">
			<h4 className="text-body-sm font-medium text-accent mb-1">
				{config.target_skill}
			</h4>
			<p className="text-body-xs text-text-secondary mb-3">
				{config.source_passage}
			</p>
			<div className="space-y-2">
				{config.exercises.map((exercise, i) => {
					const isExpanded = expandedIndex === i;
					return (
						<div
							key={exercise.title}
							className="border border-border rounded-lg overflow-hidden"
						>
							<button
								type="button"
								onClick={() => setExpandedIndex(isExpanded ? null : i)}
								className="w-full flex items-center justify-between px-3 py-2 text-left hover:bg-surface transition"
							>
								<span className="text-body-sm text-cream font-medium">
									{exercise.title}
								</span>
								<span className="text-body-xs text-text-tertiary ml-2">
									{exercise.focus_dimension}
								</span>
							</button>
							{isExpanded && (
								<div className="px-3 pb-3 pt-1 border-t border-border">
									<p className="text-body-sm text-text-secondary">
										{exercise.instruction}
									</p>
								</div>
							)}
						</div>
					);
				})}
			</div>
		</div>
	);
}
