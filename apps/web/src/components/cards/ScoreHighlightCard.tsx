import { ArrowsOut } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
import { DIMENSION_COLORS } from "../../lib/mock-session";
import { osmdManager } from "../../lib/osmd-manager";
import type { ScoreHighlightConfig } from "../../lib/types";
import { useScorePanelStore } from "../../stores/score-panel";

interface ScoreHighlightCardProps {
	config: ScoreHighlightConfig;
	onExpand?: () => void;
	artifactId?: string;
}

type RenderState = "loading" | "rendered" | "error";

export function ScoreHighlightCard({
	config,
	onExpand,
}: ScoreHighlightCardProps) {
	const [renderState, setRenderState] = useState<RenderState>("loading");
	const svgContainerRef = useRef<HTMLDivElement>(null);
	const openHighlight = useScorePanelStore((s) => s.openHighlight);

	useEffect(() => {
		let cancelled = false;

		async function loadScore() {
			try {
				await osmdManager.ensureRendered(config.pieceId);
				if (cancelled) return;

				// Clip SVG fragments for each highlight region
				if (svgContainerRef.current) {
					svgContainerRef.current.innerHTML = "";
					for (const highlight of config.highlights) {
						const svg = osmdManager.clipBars(
							config.pieceId,
							highlight.bars[0],
							highlight.bars[1],
						);
						if (svg) {
							// Apply dimension color overlay
							const color =
								DIMENSION_COLORS[
									highlight.dimension as keyof typeof DIMENSION_COLORS
								] ?? "#7a9a82";
							svg.style.border = `2px solid ${color}`;
							svg.style.borderRadius = "8px";
							svg.style.marginBottom = "8px";
							svgContainerRef.current.appendChild(svg);
						}
					}
				}

				setRenderState("rendered");
			} catch (err) {
				console.error("ScoreHighlightCard: failed to load score", err);
				if (!cancelled) setRenderState("error");
			}
		}

		loadScore();
		return () => {
			cancelled = true;
		};
	}, [config.pieceId, JSON.stringify(config.highlights)]);

	return (
		<div className="bg-surface-card border border-border rounded-xl p-4 mt-3">
			{/* Header */}
			<div className="flex items-center justify-between mb-3">
				<span className="text-body-sm font-medium text-cream">
					Score Highlight
				</span>
				{onExpand && (
					<button
						type="button"
						onClick={() => {
							openHighlight(config);
							onExpand?.();
						}}
						className="w-7 h-7 flex items-center justify-center rounded-lg text-text-secondary hover:text-cream hover:bg-surface transition"
						aria-label="Expand score highlight"
					>
						<ArrowsOut size={16} />
					</button>
				)}
			</div>

			{/* SVG snippets or fallback */}
			{renderState === "loading" && (
				<div className="flex items-center justify-center h-20 text-text-tertiary text-body-sm">
					Loading score...
				</div>
			)}

			<div ref={svgContainerRef} className="osmd-clip-container" />

			{/* Highlights legend (always shown -- text fallback for error state) */}
			<div className="flex flex-col gap-2 mt-2">
				{config.highlights.map((h) => {
					const color =
						DIMENSION_COLORS[
							h.dimension as keyof typeof DIMENSION_COLORS
						] ?? "#7a9a82";
					return (
						<div
							key={`${h.dimension}-${h.bars[0]}-${h.bars[1]}`}
							className="flex items-start gap-2"
						>
							<span
								className="w-2 h-2 rounded-full mt-1.5 shrink-0"
								style={{ backgroundColor: color }}
							/>
							<div className="min-w-0">
								<span className="text-body-xs text-text-secondary capitalize">
									{h.dimension} -- bars {h.bars[0]}-{h.bars[1]}
								</span>
								{h.annotation && (
									<p className="text-body-xs text-text-tertiary mt-0.5">
										{h.annotation}
									</p>
								)}
							</div>
						</div>
					);
				})}
			</div>

			{renderState === "error" && (
				<p className="text-body-xs text-text-tertiary mt-2 italic">
					Score preview unavailable
				</p>
			)}
		</div>
	);
}
