import { Check, CircleNotch, Minus } from "@phosphor-icons/react";
import type { ToolCallStatus } from "../lib/types";

const TOOL_LOADING_LABELS: Record<string, string> = {
	search_catalog: "Searching catalog...",
	create_exercise: "Creating exercises...",
	show_session_data: "Loading session data...",
	score_highlight: "Looking up score...",
	keyboard_guide: "Building keyboard guide...",
	reference_browser: "Loading reference...",
};

function getLoadingLabel(name: string): string {
	return TOOL_LOADING_LABELS[name] ?? "Working...";
}

export function ToolCallBar({ toolCall }: { toolCall: ToolCallStatus }) {
	if (toolCall.status === "pending") {
		return (
			<div className="flex items-center gap-2 text-body-xs text-text-tertiary">
				<CircleNotch size={12} className="animate-spin shrink-0" />
				<span>{getLoadingLabel(toolCall.name)}</span>
			</div>
		);
	}

	if (toolCall.status === "found") {
		return (
			<div className="flex items-center gap-2 text-body-xs text-text-secondary">
				<Check size={12} className="shrink-0 text-accent" />
				<span>{toolCall.label}</span>
			</div>
		);
	}

	if (toolCall.status === "not_found") {
		return (
			<div className="flex items-center gap-2 text-body-xs text-text-tertiary">
				<Minus size={12} className="shrink-0" />
				<span>Piece not found</span>
			</div>
		);
	}

	return (
		<div className="flex items-center gap-2 text-body-xs text-text-secondary">
			<Check size={12} className="shrink-0 text-accent" />
			<span>Done</span>
		</div>
	);
}
