import { ArrowLeft, MusicNote, X } from "@phosphor-icons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useIsMobile } from "../hooks/useDom";
import { useMountEffect } from "../hooks/useFoundation";
import { DIMENSION_COLORS } from "../lib/mock-session";
import { osmdManager } from "../lib/osmd-manager";
import { useScorePanelStore } from "../stores/score-panel";
import { ScoreAnnotation } from "./ScoreAnnotation";

const MIN_PANEL_WIDTH = 320;
const MAX_PANEL_WIDTH_RATIO = 0.6;

interface AnnotationPosition {
	top: number;
	left: number;
}

export function ScorePanel() {
	const isOpen = useScorePanelStore((s) => s.isOpen);
	const sessionData = useScorePanelStore((s) => s.sessionData);
	const highlightData = useScorePanelStore((s) => s.highlightData);
	const activeAnnotationIndex = useScorePanelStore(
		(s) => s.activeAnnotationIndex,
	);
	const panelWidth = useScorePanelStore((s) => s.panelWidth);
	const close = useScorePanelStore((s) => s.close);
	const setActiveAnnotation = useScorePanelStore((s) => s.setActiveAnnotation);
	const setPanelWidth = useScorePanelStore((s) => s.setPanelWidth);
	const isMobile = useIsMobile();
	const isDraggingRef = useRef(false);
	const dragWidthRef = useRef(panelWidth);

	// Drag handle for resizing
	const asideRef = useRef<HTMLDivElement>(null);
	// Shared ref for OSMD instance -- ScorePanelScore sets it, drag handler reads it
	// biome-ignore lint/suspicious/noExplicitAny: OSMD has no exported type for the instance
	const osmdRef = useRef<any>(null);

	const handleDragStart = useCallback(
		(e: React.MouseEvent) => {
			if (isMobile) return;
			e.preventDefault();
			isDraggingRef.current = true;
			dragWidthRef.current = panelWidth;
			const startX = e.clientX;
			const startWidth = panelWidth;

			function onMouseMove(ev: MouseEvent) {
				if (!isDraggingRef.current) return;
				const maxWidth = window.innerWidth * MAX_PANEL_WIDTH_RATIO;
				// Dragging left (negative delta) makes panel wider
				const delta = startX - ev.clientX;
				const newWidth = Math.min(
					maxWidth,
					Math.max(MIN_PANEL_WIDTH, startWidth + delta),
				);
				dragWidthRef.current = newWidth;
				if (asideRef.current) {
					asideRef.current.style.width = `${newWidth}px`;
				}
			}

			function onMouseUp() {
				isDraggingRef.current = false;
				document.removeEventListener("mousemove", onMouseMove);
				document.removeEventListener("mouseup", onMouseUp);
				document.body.style.cursor = "";
				document.body.style.userSelect = "";
				setPanelWidth(dragWidthRef.current);
				// Re-render OSMD at new width
				if (osmdRef.current) {
					try {
						osmdRef.current.render();
					} catch {
						// OSMD re-render failed silently
					}
				}
			}

			document.addEventListener("mousemove", onMouseMove);
			document.addEventListener("mouseup", onMouseUp);
			document.body.style.cursor = "col-resize";
			document.body.style.userSelect = "none";
		},
		[isMobile, panelWidth, setPanelWidth],
	);

	const handleAnnotationClick = useCallback(
		(index: number) => {
			setActiveAnnotation(activeAnnotationIndex === index ? null : index);
		},
		[activeAnnotationIndex, setActiveAnnotation],
	);

	if (!sessionData && !highlightData) return null;

	// Derive observations from highlightData or sessionData
	const observations = highlightData
		? highlightData.highlights.map((h) => ({
				dimension: h.dimension,
				barRange: h.bars as [number, number],
				text: h.annotation ?? "",
				framing: "" as string,
			}))
		: (sessionData?.observations ?? []);

	const pieceId = highlightData?.pieceId ?? "";
	const title = highlightData ? "Score Highlight" : (sessionData?.piece ?? "");
	const section = highlightData
		? `bars ${highlightData.highlights[0]?.bars[0]}-${highlightData.highlights[highlightData.highlights.length - 1]?.bars[1]}`
		: (sessionData?.section ?? "");
	const durationSeconds = sessionData?.durationSeconds ?? 0;

	const panelContent = (
		<>
			{/* Header */}
			<div className="flex items-center gap-3 px-4 py-3 border-b border-border shrink-0">
				{isMobile && (
					<button
						type="button"
						onClick={close}
						className="w-8 h-8 flex items-center justify-center rounded-lg text-text-secondary hover:text-cream hover:bg-surface transition"
						aria-label="Close score panel"
					>
						<ArrowLeft size={18} />
					</button>
				)}
				<MusicNote size={20} className="text-accent shrink-0" />
				<div className="flex-1 min-w-0">
					<h2 className="text-body-sm font-medium text-cream truncate">
						{title}
					</h2>
					<p className="text-body-xs text-text-tertiary">
						{section}
						{durationSeconds > 0 && (
							<span className="ml-2">
								{Math.floor(durationSeconds / 60)} min
							</span>
						)}
					</p>
				</div>
				{!isMobile && (
					<button
						type="button"
						onClick={close}
						className="w-8 h-8 flex items-center justify-center rounded-lg text-text-secondary hover:text-cream hover:bg-surface transition"
						aria-label="Close score panel"
					>
						<X size={16} />
					</button>
				)}
			</div>

			{/* Dimension legend */}
			<div className="flex flex-wrap gap-2 px-4 py-2 border-b border-border shrink-0">
				{observations.map((obs, i) => {
					const color =
						DIMENSION_COLORS[obs.dimension as keyof typeof DIMENSION_COLORS] ??
						"#7a9a82";
					return (
						<button
							type="button"
							key={`${obs.dimension}-${obs.barRange?.[0] ?? i}`}
							onClick={() => handleAnnotationClick(i)}
							className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-body-xs transition cursor-pointer ${
								activeAnnotationIndex === i
									? "bg-surface-2 text-cream"
									: "text-text-secondary hover:text-cream hover:bg-surface"
							}`}
						>
							<span
								className="w-2 h-2 rounded-full"
								style={{ backgroundColor: color }}
							/>
							<span className="capitalize">{obs.dimension}</span>
							{obs.barRange && (
								<span className="text-text-tertiary">
									b.{obs.barRange[0]}-{obs.barRange[1]}
								</span>
							)}
						</button>
					);
				})}
			</div>

			{/* Score rendering -- keyed to remount cleanly when session data changes */}
			<ScorePanelScore
				key={`${pieceId}-${title}-${observations.length}`}
				pieceId={pieceId}
				sessionData={sessionData}
				observations={observations}
				activeAnnotationIndex={activeAnnotationIndex}
				osmdRef={osmdRef}
				onAnnotationClick={handleAnnotationClick}
			/>
		</>
	);

	// Mobile: full-screen overlay
	if (isMobile) {
		return (
			<div
				className={`fixed inset-0 z-50 bg-espresso flex flex-col transition-transform duration-300 ${
					isOpen ? "translate-x-0" : "translate-x-full"
				}`}
			>
				{panelContent}
			</div>
		);
	}

	// Desktop: right sidebar panel
	return (
		<aside
			ref={asideRef}
			className={`shrink-0 border-l border-border bg-espresso flex flex-col overflow-hidden relative ${
				isOpen ? "" : "!w-0"
			}`}
			style={isOpen ? { width: panelWidth } : undefined}
		>
			{isOpen && (
				<>
					{/* Drag handle */}
					<div
						onMouseDown={handleDragStart}
						className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize z-10 bg-border hover:bg-accent transition-colors"
						role="separator"
						aria-orientation="vertical"
						aria-label="Resize score panel"
					/>
					{panelContent}
				</>
			)}
		</aside>
	);
}

/**
 * Inner component that initializes OSMD and calculates annotation positions.
 * Keyed by session data so React unmounts/remounts cleanly when the session changes,
 * turning the init effect into a simple mount effect (Rule 5: reset with key).
 */
interface ScorePanelScoreProps {
	pieceId: string;
	sessionData: NonNullable<ReturnType<typeof useScorePanelStore>["sessionData"]> | null;
	observations: Array<{
		dimension: string;
		barRange?: [number, number];
		text?: string;
		framing?: string;
	}>;
	activeAnnotationIndex: number | null;
	// biome-ignore lint/suspicious/noExplicitAny: OSMD has no exported type
	osmdRef: React.MutableRefObject<any>;
	onAnnotationClick: (index: number) => void;
}

function ScorePanelScore({
	pieceId,
	sessionData,
	observations,
	activeAnnotationIndex,
	osmdRef,
	onAnnotationClick,
}: ScorePanelScoreProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const [isRendered, setIsRendered] = useState(false);
	const [isError, setIsError] = useState(false);
	const [annotationPositions, setAnnotationPositions] = useState<
		AnnotationPosition[]
	>([]);

	// Initialize OSMD on mount (component is keyed, so this runs once per session)
	useMountEffect(() => {
		let cancelled = false;

		async function initOSMD() {
			const osmdContainer = containerRef.current;
			if (!osmdContainer || cancelled) return;

			if (!pieceId) {
				// No pieceId -- sessionData path shows annotation list without a rendered score
				// Still set isRendered so the annotation-position effect can run (positions fall
				// through to the fallback distributor since osmdRef.current remains null)
				setIsRendered(true);
				return;
			}

			try {
				// Use OSMD Manager for cached rendering
				await osmdManager.ensureRendered(pieceId);
				if (cancelled) return;

				const cached = osmdManager.getOsmdInstance(pieceId);
				if (cached) {
					// Move the rendered SVG into our container
					const sourceSvg = cached.container.querySelector("svg");
					if (sourceSvg) {
						const cloned = sourceSvg.cloneNode(true) as SVGElement;
						osmdContainer.appendChild(cloned);
					}
					osmdRef.current = cached.osmd;
					setIsRendered(true);
					return;
				}
			} catch (err) {
				console.error("OSMD render failed:", err);
				if (!cancelled) setIsError(true);
			}
		}

		initOSMD();

		return () => {
			cancelled = true;
			osmdRef.current = null;
		};
	});

	// Calculate annotation positions after OSMD renders
	useEffect(() => {
		if (!isRendered || !osmdRef.current || !containerRef.current) return;

		const osmd = osmdRef.current;
		const containerRect = containerRef.current.getBoundingClientRect();
		const positions: AnnotationPosition[] = [];

		for (const obs of observations) {
			if (!obs.barRange) {
				positions.push({ top: 0, left: 0 });
				continue;
			}

			const measureIndex = obs.barRange[0] - 1; // 0-indexed
			try {
				const measureList = osmd.graphic?.measureList;
				if (
					measureList &&
					measureIndex >= 0 &&
					measureIndex < measureList.length
				) {
					const measure = measureList[measureIndex]?.[0];
					if (measure?.stave?.SVGElement) {
						const svgEl = measure.stave.SVGElement as SVGElement;
						const bbox = svgEl.getBoundingClientRect();
						positions.push({
							top: bbox.top - containerRect.top - 28,
							left: bbox.left - containerRect.left,
						});
						continue;
					}
					// Fallback: use bounding box from the measure
					if (measure?.boundingBox) {
						const absPos = measure.boundingBox.absolutePosition;
						if (absPos) {
							// OSMD units are ~10x pixels
							positions.push({
								top: absPos.y * 10 - 28,
								left: absPos.x * 10,
							});
							continue;
						}
					}
				}
			} catch {
				// Fallback positioning
			}

			// Last resort: distribute evenly
			const fallbackTop = 60 + positions.length * 80;
			positions.push({ top: fallbackTop, left: 20 });
		}

		setAnnotationPositions(positions);
	}, [isRendered, observations, osmdRef]);

	return (
		<div className="flex-1 overflow-y-auto overflow-x-hidden px-4 py-4 relative">
			{isError && (
				<div className="flex items-center justify-center h-32 text-text-tertiary text-body-sm">
					Score unavailable
				</div>
			)}
			{!isRendered && !isError && pieceId && (
				<div className="flex items-center justify-center h-32 text-text-tertiary text-body-sm">
					Loading score...
				</div>
			)}
			<div className="relative">
				<div ref={containerRef} className="osmd-container" />
				{/* Annotation markers */}
				{isRendered &&
					observations.map((obs, i) => {
						if (!obs.barRange || !annotationPositions[i]) return null;
						return (
							<ScoreAnnotation
								key={`${obs.dimension}-${obs.barRange[0]}`}
								dimension={obs.dimension}
								barRange={obs.barRange}
								index={i}
								isActive={activeAnnotationIndex === i}
								style={{
									top: annotationPositions[i].top,
									left: annotationPositions[i].left,
								}}
								onClick={onAnnotationClick}
							/>
						);
					})}
			</div>

			{/* Active observation detail */}
			{activeAnnotationIndex !== null &&
				observations[activeAnnotationIndex] && (
					<div className="sticky bottom-0 mt-4 p-3 bg-surface-2 border border-border rounded-lg animate-fade-in">
						<p className="text-body-sm text-cream">
							{observations[activeAnnotationIndex].text}
						</p>
						<p className="text-body-xs text-text-tertiary mt-1 capitalize">
							{observations[activeAnnotationIndex].dimension} --{" "}
							{observations[activeAnnotationIndex].framing}
						</p>
					</div>
				)}
		</div>
	);
}
