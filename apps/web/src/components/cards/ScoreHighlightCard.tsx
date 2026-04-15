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

function clearChildren(el: HTMLDivElement) {
	while (el.firstChild) {
		el.removeChild(el.firstChild);
	}
}

export function ScoreHighlightCard({
	config,
	onExpand,
}: ScoreHighlightCardProps) {
	const [renderState, setRenderState] = useState<RenderState>("loading");
	const [hasClips, setHasClips] = useState(false);
	const svgContainerRef = useRef<HTMLDivElement>(null);
	const openHighlight = useScorePanelStore((s) => s.openHighlight);

	// biome-ignore lint/correctness/useExhaustiveDependencies: highlights array identity is not stable; JSON.stringify produces a stable change signal
	useEffect(() => {
		let cancelled = false;

		async function loadScore() {
			try {
				await osmdManager.ensureRendered(config.pieceId);
				if (cancelled) return;

				let clipsAdded = 0;
				if (svgContainerRef.current) {
					clearChildren(svgContainerRef.current);
					for (const highlight of config.highlights) {
						const svg = osmdManager.clipBars(
							config.pieceId,
							highlight.bars[0],
							highlight.bars[1],
						);
						if (svg) {
							const color =
								DIMENSION_COLORS[
									highlight.dimension as keyof typeof DIMENSION_COLORS
								] ?? "#7a9a82";
							svg.style.borderRadius = "6px";
							svg.style.border = `1.5px solid ${color}40`;
							svgContainerRef.current.appendChild(svg);
							clipsAdded++;
						}
					}
				}

				if (!cancelled) {
					setHasClips(clipsAdded > 0);
					setRenderState("rendered");
				}
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
		<div className="bg-surface-card border border-border rounded-xl overflow-hidden mt-3">
			{/* Score clips — only shown when OSMD successfully clips bars */}
			{renderState === "loading" && (
				<div className="h-10 flex items-center justify-center">
					<div className="w-3.5 h-3.5 rounded-full border-2 border-text-tertiary/50 border-t-transparent animate-spin" />
				</div>
			)}

			{renderState !== "loading" && hasClips && (
				<div
					ref={svgContainerRef}
					className="px-3 pt-3 pb-0 flex flex-col gap-2 [&>svg]:block [&>svg]:w-full [&>svg]:bg-white [&>svg]:rounded-md"
				/>
			)}

			{/* Invisible ref container when no clips produced */}
			{renderState !== "loading" && !hasClips && (
				<div ref={svgContainerRef} className="sr-only" />
			)}

			{/* Annotations */}
			<div
				className={`p-4 flex flex-col gap-3.5 ${
					hasClips ? "border-t border-border/40" : ""
				}`}
			>
				<div className="flex items-center justify-between">
					<span className="text-body-xs text-text-tertiary">
						{config.highlights.length === 1
							? "1 annotation"
							: `${config.highlights.length} annotations`}
					</span>
					{onExpand && (
						<button
							type="button"
							onClick={() => {
								openHighlight(config);
								onExpand?.();
							}}
							className="w-6 h-6 flex items-center justify-center rounded text-text-tertiary hover:text-cream hover:bg-surface transition-colors"
							aria-label="Expand score highlight"
						>
							<ArrowsOut size={13} />
						</button>
					)}
				</div>

				{config.highlights.map((h) => {
					const color =
						DIMENSION_COLORS[h.dimension as keyof typeof DIMENSION_COLORS] ??
						"#7a9a82";
					return (
						<div
							key={`${h.dimension}-${h.bars[0]}-${h.bars[1]}`}
							className="flex items-start gap-3"
						>
							<div className="flex items-center gap-1.5 shrink-0 mt-1">
								<span
									className="w-1.5 h-1.5 rounded-full shrink-0"
									style={{ backgroundColor: color }}
								/>
								<span className="text-label-sm text-text-tertiary uppercase tracking-wide">
									{h.dimension}
								</span>
							</div>
							<div className="min-w-0">
								<span className="text-body-xs text-text-tertiary">
									bars {h.bars[0]}–{h.bars[1]}
								</span>
								{h.annotation && (
									<p className="text-body-sm text-text-primary mt-0.5 leading-snug">
										{h.annotation}
									</p>
								)}
							</div>
						</div>
					);
				})}
			</div>
		</div>
	);
}
