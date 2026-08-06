import type { CSSProperties } from "react";
import { DIMENSION_COLOR_VAR } from "../lib/dimension-colors";
import type { Mark } from "../lib/mark";
import {
	anchorLabel,
	LIFECYCLE_OPACITY,
	TAXONOMY_GLYPH,
	TAXONOMY_LABEL,
} from "../lib/mark";
import { DIMENSION_LABELS } from "../lib/mock-session";

interface MarkGlyphProps {
	mark: Mark;
	expanded: boolean;
	onToggle: (id: string) => void;
	style?: CSSProperties;
}

/**
 * The one visual atom both canvases render. If each canvas drew its own chip
 * the two could diverge silently, and "the same mark renders correctly on both
 * canvases" would stop being enforceable.
 *
 * The dimension tint is a decorative dot, not a background: the --dim-* values
 * are muted mid-tones that would fail a 4.5:1 text gate, and the dimension is
 * carried in text regardless.
 */
export function MarkGlyph({ mark, expanded, onToggle, style }: MarkGlyphProps) {
	const location = anchorLabel(mark.anchor);
	const dimension = DIMENSION_LABELS[mark.dimension];
	const label = `${TAXONOMY_LABEL[mark.taxonomy]}: ${dimension}, ${location}`;

	return (
		<button
			type="button"
			aria-expanded={expanded}
			aria-label={label}
			onClick={() => onToggle(mark.id)}
			className="flex items-center gap-1.5 rounded-full border border-border-subtle bg-surface-raised px-2 py-0.5 text-label-sm text-ink-primary"
			// A lookup, never a computation. Lifecycle is server state; the
			// client is forbidden from deriving or transitioning it.
			style={{ ...style, opacity: LIFECYCLE_OPACITY[mark.lifecycle] }}
		>
			<span
				aria-hidden="true"
				className="h-1.5 w-1.5 rounded-full"
				style={{ backgroundColor: DIMENSION_COLOR_VAR[mark.dimension] }}
			/>
			<span aria-hidden="true">{TAXONOMY_GLYPH[mark.taxonomy]}</span>
			<span>{dimension}</span>
			<span className="text-ink-tertiary">{location}</span>
		</button>
	);
}
