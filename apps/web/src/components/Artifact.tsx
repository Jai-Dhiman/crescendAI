import { useEffect, useRef } from "react";
import type { InlineComponent } from "../lib/types";
import { useArtifactStore } from "../stores/artifact";
import { useArtifactScrollContext } from "../contexts/artifact-scroll";
import { CollapsedPreview } from "./cards/CollapsedPreview";
import { InlineCard } from "./InlineCard";

interface ArtifactProps {
	artifactId: string;
	component: InlineComponent;
}

function getCollapsedProps(component: InlineComponent): {
	title: string;
	subtitle: string;
	badge: string;
} {
	if (component.type === "exercise_set") {
		const count = component.config.exercises.length;
		return {
			title: component.config.target_skill,
			subtitle: component.config.source_passage,
			badge: `${count} exercise${count === 1 ? "" : "s"}`,
		};
	}

	return {
		title: component.type.replace(/_/g, " "),
		subtitle: "",
		badge: "",
	};
}

export function Artifact({ artifactId, component }: ArtifactProps) {
	const entry = useArtifactStore((s) => s.states[artifactId]);
	const register = useArtifactStore((s) => s.register);
	const unregister = useArtifactStore((s) => s.unregister);
	const collapse = useArtifactStore((s) => s.collapse);
	const expand = useArtifactStore((s) => s.expand);
	const restore = useArtifactStore((s) => s.restore);
	const scrollContainerRef = useArtifactScrollContext();

	const elementRef = useRef<HTMLDivElement>(null);
	const mountedRef = useRef(false);
	const collapseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const mountTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

	// Lifecycle: register on mount, unregister on unmount
	useEffect(() => {
		register(artifactId, component);

		mountTimerRef.current = setTimeout(() => {
			mountedRef.current = true;
		}, 1000);

		return () => {
			if (mountTimerRef.current !== null) {
				clearTimeout(mountTimerRef.current);
				mountTimerRef.current = null;
			}
			mountedRef.current = false;
			unregister(artifactId);
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [artifactId]);

	// IntersectionObserver: collapse when scrolled out, restore timer when scrolled back
	useEffect(() => {
		const element = elementRef.current;
		if (!element) {
			return;
		}

		const observer = new IntersectionObserver(
			(entries) => {
				if (!mountedRef.current) {
					return;
				}

				const entry = entries[0];
				if (!entry) {
					return;
				}

				if (!entry.isIntersecting) {
					collapseTimerRef.current = setTimeout(() => {
						collapse(artifactId);
					}, 200);
				} else {
					if (collapseTimerRef.current !== null) {
						clearTimeout(collapseTimerRef.current);
						collapseTimerRef.current = null;
					}
				}
			},
			{
				root: scrollContainerRef?.current ?? null,
				threshold: [0],
			},
		);

		observer.observe(element);

		return () => {
			observer.disconnect();
			if (collapseTimerRef.current !== null) {
				clearTimeout(collapseTimerRef.current);
				collapseTimerRef.current = null;
			}
		};
	}, [artifactId, collapse, scrollContainerRef]);

	const artifactState = entry?.state ?? "inline";
	const collapsedProps = getCollapsedProps(component);

	if (artifactState === "collapsed") {
		return (
			<div ref={elementRef} className="animate-artifact-expand-inline">
				<CollapsedPreview
					title={collapsedProps.title}
					subtitle={collapsedProps.subtitle}
					badge={collapsedProps.badge}
					onRestore={() => restore(artifactId)}
					onExpand={() => expand(artifactId)}
				/>
			</div>
		);
	}

	if (artifactState === "expanded") {
		return (
			<div ref={elementRef}>
				<InlineCard
					component={component}
					onExpand={() => {}}
					artifactId={artifactId}
				/>
			</div>
		);
	}

	// inline (default)
	return (
		<div ref={elementRef} className="animate-fade-in">
			<InlineCard
				component={component}
				onExpand={() => expand(artifactId)}
				artifactId={artifactId}
			/>
		</div>
	);
}
