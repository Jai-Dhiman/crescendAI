import type { Mark } from "../lib/mark";
import { anchorLabel } from "../lib/mark";

interface MarkDetailProps {
	mark: Mark;
	onClose: () => void;
}

/**
 * The expanded state of a mark, shared by both canvases.
 *
 * `confidence` changes only the wording here. It gates nothing: it never hides
 * a mark, never changes placement, and never suppresses rendering — matching
 * #163's invariant that confidence never gates firing.
 */
export function MarkDetail({ mark, onClose }: MarkDetailProps) {
	const framing = mark.confidence === "exploratory" ? "Early read — " : "";

	return (
		// A <section>, not a <div>: aria-label is ignored on a roleless div, so
		// the panel would have no accessible name at all. Only one detail panel
		// is open at a time, so this adds at most one landmark.
		<section
			className="mt-1 rounded-md border border-border-subtle bg-surface-raised p-3"
			aria-label={`Evidence, ${anchorLabel(mark.anchor)}`}
		>
			<p className="text-body-sm text-ink-secondary">
				{framing}
				{mark.evidence}
			</p>
			<button
				type="button"
				onClick={onClose}
				className="mt-2 text-label-sm text-ink-tertiary underline"
			>
				Close
			</button>
		</section>
	);
}
