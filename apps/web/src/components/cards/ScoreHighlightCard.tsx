import { ArrowsOut } from "@phosphor-icons/react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { DIMENSION_COLORS } from "../../lib/mock-session";
import { scoreRenderer } from "../../lib/score-renderer";
import type { ScoreHighlightConfig } from "../../lib/types";
import { useScorePanelStore } from "../../stores/score-panel";

interface ScoreHighlightCardProps {
	config: ScoreHighlightConfig;
	onExpand?: () => void;
	artifactId?: string;
}

type RenderState = "loading" | "rendered" | "error";

interface LoadedClip {
	svg: string;
	dimension: string;
	bars: [number, number];
	annotation?: string;
}

function ClipSvg({ svg }: { svg: string }) {
	const ref = useRef<HTMLDivElement>(null);
	useLayoutEffect(() => {
		if (!ref.current) return;
		ref.current.textContent = "";
		// biome-ignore lint/security/noDomManipulation: controlled SVG from Verovio WASM, not user input
		ref.current.insertAdjacentHTML("afterbegin", svg);
		const svgEl = ref.current.querySelector("svg");
		if (svgEl) {
			svgEl.setAttribute("width", "100%");
			svgEl.removeAttribute("height");
			(svgEl as SVGElement).style.display = "block";
		}
	}, [svg]);
	return <div ref={ref} className="[&>svg]:w-full [&>svg]:block" />;
}

export function ScoreHighlightCard({
	config,
	onExpand,
}: ScoreHighlightCardProps) {
	const [renderState, setRenderState] = useState<RenderState>("loading");
	const [clips, setClips] = useState<LoadedClip[]>([]);
	const openHighlight = useScorePanelStore((s) => s.openHighlight);

	// biome-ignore lint/correctness/useExhaustiveDependencies: highlights array identity is not stable; JSON.stringify produces a stable change signal
	useEffect(() => {
		let cancelled = false;

		Promise.all(
			config.highlights.map((highlight) =>
				scoreRenderer
					.getClip(config.pieceId, highlight.bars[0], highlight.bars[1])
					.then((svg) => ({
						svg,
						dimension: highlight.dimension,
						bars: highlight.bars,
						annotation: highlight.annotation,
					})),
			),
		)
			.then((results) => {
				if (cancelled) return;
				setClips(results);
				setRenderState("rendered");
			})
			.catch((err) => {
				console.error("ScoreHighlightCard: failed to load score", err);
				if (!cancelled) setRenderState("error");
			});

		return () => {
			cancelled = true;
		};
	}, [config.pieceId, JSON.stringify(config.highlights)]);

	return (
		<div className="bg-surface-card border border-border rounded-xl overflow-hidden mt-3">
			{renderState === "loading" && (
				<div className="h-10 flex items-center justify-center">
					<div className="w-3.5 h-3.5 rounded-full border-2 border-text-tertiary/50 border-t-transparent animate-spin" />
				</div>
			)}

			{renderState === "rendered" && clips.length > 0 && (
				<div className="px-3 pt-3 pb-0 flex flex-col gap-2">
					{clips.map((clip) => {
						const color =
							DIMENSION_COLORS[
								clip.dimension as keyof typeof DIMENSION_COLORS
							] ?? "#7a9a82";
						return (
							<div
								key={`${clip.dimension}-${clip.bars[0]}-${clip.bars[1]}`}
								style={{
									position: "relative",
									borderRadius: "6px",
									border: `1.5px solid ${color}40`,
									backgroundColor: "white",
									overflow: "hidden",
								}}
							>
								<ClipSvg svg={clip.svg} />
								<div
									style={{
										position: "absolute",
										inset: 0,
										backgroundColor: `${color}22`,
										borderRadius: "5px",
										pointerEvents: "none",
									}}
								/>
							</div>
						);
					})}
				</div>
			)}

			<div
				className={`p-4 flex flex-col gap-3.5 ${
					renderState === "rendered" && clips.length > 0
						? "border-t border-border/40"
						: ""
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
