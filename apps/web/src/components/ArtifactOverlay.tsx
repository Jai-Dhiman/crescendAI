import { useCallback, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { X } from "@phosphor-icons/react";
import { useArtifactStore, getExpandedArtifact } from "../stores/artifact";
import { useArtifactScrollContext } from "../contexts/artifact-scroll";
import { ExerciseSetExpanded } from "./cards/ExerciseSetExpanded";
import type { ArtifactEntry } from "../stores/artifact";

interface ArtifactOverlayContentProps {
	expandedId: string;
	entry: ArtifactEntry;
}

function ArtifactOverlayContent({ expandedId, entry }: ArtifactOverlayContentProps) {
	const closeOverlay = useArtifactStore((s) => s.closeOverlay);
	const scrollContainer = useArtifactScrollContext();
	const isClosingRef = useRef(false);
	const overlayRef = useRef<HTMLDivElement>(null);

	const handleClose = useCallback(() => {
		if (isClosingRef.current) return;
		isClosingRef.current = true;

		const backdrop = overlayRef.current?.querySelector<HTMLElement>("[data-backdrop]");
		const panel = overlayRef.current?.querySelector<HTMLElement>("[data-panel]");

		if (backdrop) {
			backdrop.classList.remove("animate-backdrop-in");
			backdrop.classList.add("animate-backdrop-out");
		}
		if (panel) {
			panel.classList.remove("animate-overlay-in");
			panel.classList.add("animate-panel-out");
		}

		setTimeout(() => {
			closeOverlay(expandedId);
		}, 200);
	}, [expandedId, closeOverlay]);

	useEffect(() => {
		function onKeyDown(e: KeyboardEvent) {
			if (e.key === "Escape") {
				handleClose();
			}
		}

		if (expandedId) {
			document.addEventListener("keydown", onKeyDown);
		}

		return () => {
			document.removeEventListener("keydown", onKeyDown);
		};
	}, [expandedId, handleClose]);

	useEffect(() => {
		const container = scrollContainer?.current;
		if (!container) return;

		const previous = container.style.overflow;
		container.style.overflow = "hidden";

		return () => {
			container.style.overflow = previous;
		};
	}, [scrollContainer]);

	function renderExpanded() {
		switch (entry.component.type) {
			case "exercise_set":
				return (
					<ExerciseSetExpanded
						config={entry.component.config}
						artifactId={expandedId}
					/>
				);
			default:
				return (
					<p className="text-body-md text-text-secondary">
						No expanded view available for this component.
					</p>
				);
		}
	}

	return (
		<div ref={overlayRef} className="fixed inset-0 z-50">
			<div
				data-backdrop
				role="button"
				tabIndex={-1}
				className="absolute inset-0 bg-black/60 animate-backdrop-in"
				onClick={handleClose}
				onKeyDown={(e) => {
					if (e.key === "Enter" || e.key === " ") {
						handleClose();
					}
				}}
			/>
			<div
				data-panel
				className="relative max-w-2xl mx-auto mt-16 max-h-[80vh] overflow-y-auto bg-surface-card border border-border rounded-2xl p-6 shadow-card animate-overlay-in"
			>
				<button
					type="button"
					className="absolute top-4 right-4 text-text-tertiary hover:text-text-primary transition-colors"
					onClick={handleClose}
					aria-label="Close"
				>
					<X size={20} />
				</button>
				{renderExpanded()}
			</div>
		</div>
	);
}

export function ArtifactOverlay() {
	const hasExpanded = useArtifactStore((s) => getExpandedArtifact(s) !== null);

	if (!hasExpanded) return null;

	return createPortal(<ArtifactOverlayInner />, document.body);
}

function ArtifactOverlayInner() {
	const expanded = useArtifactStore((s) => getExpandedArtifact(s));

	if (!expanded) return null;

	return <ArtifactOverlayContent expandedId={expanded.id} entry={expanded.entry} />;
}
