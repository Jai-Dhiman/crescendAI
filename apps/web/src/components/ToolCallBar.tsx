import {
	Check,
	CircleNotch,
	Minus,
	WarningCircle,
} from "@phosphor-icons/react";
import type { ToolCallStatus } from "../lib/types";

const TOOL_LOADING_LABELS: Record<string, string> = {
	search_catalog: "Searching catalog...",
	create_exercise: "Creating exercises...",
	show_session_data: "Loading session data...",
	score_highlight: "Looking up score...",
	keyboard_guide: "Building keyboard guide...",
	reference_browser: "Loading reference...",
};

const TOOL_DISPLAY_NAMES: Record<string, string> = {
	search_catalog: "Catalog search",
	create_exercise: "Exercise creation",
	show_session_data: "Session lookup",
	score_highlight: "Score highlight",
	keyboard_guide: "Keyboard guide",
	reference_browser: "Reference browser",
};

function getLoadingLabel(name: string): string {
	return TOOL_LOADING_LABELS[name] ?? "Working...";
}

function getDisplayName(name: string): string {
	return TOOL_DISPLAY_NAMES[name] ?? name;
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

	if (toolCall.status === "error") {
		return (
			<div
				role="alert"
				className="flex items-start gap-2 rounded-lg border border-red-500/40 bg-red-500/10 px-3 py-2 text-body-xs text-red-300"
			>
				<WarningCircle size={14} className="mt-0.5 shrink-0" />
				<div className="min-w-0">
					<div className="font-medium">
						{getDisplayName(toolCall.name)} failed
					</div>
					<div className="mt-0.5 text-red-200/80 break-words">
						{toolCall.message}
					</div>
				</div>
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
