import { ArrowLeft, MusicNote, X } from "@phosphor-icons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useIsMobile } from "../hooks/useDom";
import { useMountEffect } from "../hooks/useFoundation";
import { DIMENSION_COLOR_VAR } from "../lib/dimension-colors";
import type { MockSessionData } from "../lib/mock-session";
import { scoreRenderer } from "../lib/score-renderer";
import { useScorePanelStore } from "../stores/score-panel";
import { ScoreAnnotation } from "./ScoreAnnotation";

const MIN_PANEL_WIDTH = 320;
const MAX_PANEL_WIDTH_RATIO = 0.6;
// Pixels moved per arrow-key press on the resize separator.
const RESIZE_STEP = 24;

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
	const [scoreRenderKey, setScoreRenderKey] = useState(0);
	// Mirrors the drag clamp so the separator can report its range. Seeded on
	// mount rather than during render, which also runs on the server.
	const [maxPanelWidth, setMaxPanelWidth] = useState(MIN_PANEL_WIDTH);

	// Drag handle for resizing
	const asideRef = useRef<HTMLDivElement>(null);

	// Keyboard equivalent of the drag: ArrowLeft widens, ArrowRight narrows,
	// matching the drag direction (dragging left makes the panel wider).
	useEffect(() => {
		function syncMax() {
			setMaxPanelWidth(window.innerWidth * MAX_PANEL_WIDTH_RATIO);
		}
		syncMax();
		window.addEventListener("resize", syncMax);
		return () => window.removeEventListener("resize", syncMax);
	}, []);

	const handleResizeKey = useCallback(
		(e: React.KeyboardEvent) => {
			if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
			e.preventDefault();
			const maxWidth = window.innerWidth * MAX_PANEL_WIDTH_RATIO;
			const delta = e.key === "ArrowLeft" ? RESIZE_STEP : -RESIZE_STEP;
			setPanelWidth(
				Math.min(maxWidth, Math.max(MIN_PANEL_WIDTH, panelWidth + delta)),
			);
			setScoreRenderKey((k) => k + 1);
		},
		[panelWidth, setPanelWidth],
	);

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
				setScoreRenderKey((k) => k + 1);
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
			<div className="flex items-center gap-3 px-4 py-3 border-b border-border-subtle shrink-0">
				{isMobile && (
					<button
						type="button"
						onClick={close}
						className="w-8 h-8 flex items-center justify-center rounded-lg text-ink-secondary hover:text-ink-primary hover:bg-surface-raised transition"
						aria-label="Close score panel"
					>
						<ArrowLeft size={18} />
					</button>
				)}
				<MusicNote size={20} className="text-accent shrink-0" />
				<div className="flex-1 min-w-0">
					<h2 className="text-body-sm font-medium text-ink-primary truncate">
						{title}
					</h2>
					<p className="text-body-xs text-ink-tertiary">
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
						className="w-8 h-8 flex items-center justify-center rounded-lg text-ink-secondary hover:text-ink-primary hover:bg-surface-raised transition"
						aria-label="Close score panel"
					>
						<X size={16} />
					</button>
				)}
			</div>

			{/* Dimension legend */}
			<div className="flex flex-wrap gap-2 px-4 py-2 border-b border-border-subtle shrink-0">
				{observations.map((obs, i) => {
					const color =
						DIMENSION_COLOR_VAR[
							obs.dimension as keyof typeof DIMENSION_COLOR_VAR
						] ?? "var(--color-accent)";
					return (
						<button
							type="button"
							key={`${obs.dimension}-${obs.barRange?.[0] ?? i}`}
							onClick={() => handleAnnotationClick(i)}
							className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-body-xs transition cursor-pointer ${
								activeAnnotationIndex === i
									? "bg-surface-sunken text-ink-primary"
									: "text-ink-secondary hover:text-ink-primary hover:bg-surface-raised"
							}`}
						>
							<span
								className="w-2 h-2 rounded-full"
								style={{ backgroundColor: color }}
							/>
							<span className="capitalize">{obs.dimension}</span>
							{obs.barRange && (
								<span className="text-ink-tertiary">
									b.{obs.barRange[0]}-{obs.barRange[1]}
								</span>
							)}
						</button>
					);
				})}
			</div>

			{/* Score rendering -- keyed to remount cleanly when session data changes */}
			<ScorePanelScore
				key={`${pieceId}-${title}-${observations.length}-${scoreRenderKey}`}
				pieceId={pieceId}
				sessionData={sessionData}
				observations={observations}
				activeAnnotationIndex={activeAnnotationIndex}
				onAnnotationClick={handleAnnotationClick}
			/>
		</>
	);

	// Mobile: full-screen overlay
	if (isMobile) {
		return (
			<div
				className={`fixed inset-0 z-50 bg-surface-page flex flex-col transition-transform duration-300 ${
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
			className={`shrink-0 border-l border-border-subtle bg-surface-page flex flex-col overflow-hidden relative ${
				isOpen ? "" : "!w-0"
			}`}
			style={isOpen ? { width: panelWidth } : undefined}
		>
			{isOpen && (
				<>
					{/* Drag handle */}
					{/* biome-ignore lint/a11y/useSemanticElements: a focusable window splitter cannot be an <hr>; it needs drag and key handlers. */}
					<div
						onMouseDown={handleDragStart}
						onKeyDown={handleResizeKey}
						tabIndex={0}
						className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize z-10 bg-border-subtle hover:bg-accent transition-colors"
						role="separator"
						aria-orientation="vertical"
						aria-label="Resize score panel"
						aria-valuenow={panelWidth}
						aria-valuemin={MIN_PANEL_WIDTH}
						aria-valuemax={maxPanelWidth}
					/>
					{panelContent}
				</>
			)}
		</aside>
	);
}

/**
 * Inner component that loads score SVG via scoreRenderer and calculates annotation positions.
 * Keyed by session data so React unmounts/remounts cleanly when the session changes,
 * turning the init effect into a simple mount effect (Rule 5: reset with key).
 */
interface ScorePanelScoreProps {
	pieceId: string;
	sessionData: MockSessionData | null;
	observations: Array<{
		dimension: string;
		barRange?: [number, number];
		text?: string;
		framing?: string;
	}>;
	activeAnnotationIndex: number | null;
	onAnnotationClick: (index: number) => void;
}

function ScorePanelScore({
	pieceId,
	observations,
	activeAnnotationIndex,
	onAnnotationClick,
}: ScorePanelScoreProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const [isRendered, setIsRendered] = useState(false);
	const [isError, setIsError] = useState(false);
	const [annotationPositions, setAnnotationPositions] = useState<
		AnnotationPosition[]
	>([]);

	// Track which pieceIds have already been loaded so repeated renders don't re-trigger load().
	// This ref lives at the module level via the singleton scoreRenderer, but we track it here
	// per-component-instance since each ScorePanel is keyed to a session.
	const loadedPieceRef = useRef<string | null>(null);

	// Load score SVG on mount (component is keyed, so this runs once per session)
	useMountEffect(() => {
		let cancelled = false;

		async function loadScore() {
			const container = containerRef.current;
			if (!container || cancelled) return;

			if (!pieceId) {
				setIsRendered(true);
				return;
			}

			try {
				// Call load() before getPage() as required by the worker protocol.
				// Memoize by pieceId so resize-triggered re-renders skip the round-trip.
				if (loadedPieceRef.current !== pieceId) {
					const loadResult = await scoreRenderer.load(pieceId);
					if (cancelled) return;
					if (loadResult === "failed") {
						setIsError(true);
						return;
					}
					loadedPieceRef.current = pieceId;
				}

				const svg = await scoreRenderer.getPage(pieceId, 1);
				if (cancelled) return;
				container.textContent = "";
				// Controlled SVG from Verovio WASM, not user input.
				container.insertAdjacentHTML("beforeend", svg);
				setIsRendered(true);
			} catch (err) {
				console.error("ScorePanel: score render failed", err);
				if (!cancelled) setIsError(true);
			}
		}

		loadScore();

		return () => {
			cancelled = true;
		};
	});

	// Calculate annotation positions after score renders
	useEffect(() => {
		if (!isRendered || !containerRef.current) return;

		const containerRect = containerRef.current.getBoundingClientRect();
		const measureEls = Array.from(
			containerRef.current.querySelectorAll<Element>(".measure"),
		);
		const positions: AnnotationPosition[] = [];

		for (const obs of observations) {
			if (!obs.barRange) {
				positions.push({ top: 0, left: 0 });
				continue;
			}
			const measureIdx = obs.barRange[0] - 1;
			const el = measureEls[measureIdx];
			if (el) {
				const rect = el.getBoundingClientRect();
				positions.push({
					top: rect.top - containerRect.top - 28,
					left: rect.left - containerRect.left,
				});
			} else {
				positions.push({ top: 60 + positions.length * 80, left: 20 });
			}
		}

		setAnnotationPositions(positions);
	}, [isRendered, observations]);

	return (
		<div className="flex-1 overflow-y-auto overflow-x-hidden px-4 py-4 relative">
			{isError && (
				<div className="flex items-center justify-center h-32 text-ink-tertiary text-body-sm">
					Score unavailable
				</div>
			)}
			{!isRendered && !isError && pieceId && (
				<div className="flex items-center justify-center h-32 text-ink-tertiary text-body-sm">
					Loading score...
				</div>
			)}
			<div className="relative">
				<div ref={containerRef} className="score-container" />
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
					<div className="sticky bottom-0 mt-4 p-3 bg-surface-sunken border border-border-subtle rounded-lg animate-fade-in">
						<p className="text-body-sm text-ink-primary">
							{observations[activeAnnotationIndex].text}
						</p>
						<p className="text-body-xs text-ink-tertiary mt-1 capitalize">
							{observations[activeAnnotationIndex].dimension} --{" "}
							{observations[activeAnnotationIndex].framing}
						</p>
					</div>
				)}
		</div>
	);
}
