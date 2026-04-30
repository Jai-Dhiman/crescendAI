// Rich message type system for inline UI cards (Slice 10)

export type ToolCallStatus =
	| { name: string; status: "pending" }
	| { name: string; status: "found"; label: string }
	| { name: string; status: "not_found" }
	| { name: string; status: "done" }
	| { name: string; status: "error"; message: string };

export interface RichMessage {
	id: string;
	role: "user" | "assistant";
	content: string;
	createdAt: string;
	streaming?: boolean;
	components?: InlineComponent[];
	toolCalls?: ToolCallStatus[];
	dimension?: string;
	messageType?:
		| "chat"
		| "observation"
		| "session_start"
		| "session_end"
		| "summary"
		| "synthesis";
	sessionId?: string;
	framing?: string;
}

export type InlineComponent =
	| { type: "exercise_set"; config: ExerciseSetConfig }
	| { type: "score_highlight"; config: ScoreHighlightConfig }
	| { type: "keyboard_guide"; config: KeyboardGuideConfig }
	| { type: "reference_browser"; config: ReferenceBrowserConfig }
	| { type: "segment_loop"; config: SegmentLoopConfig };

export interface ExerciseSetConfig {
	sourcePassage: string;
	targetSkill: string;
	exercises: Array<{
		title: string;
		instruction: string;
		focusDimension: string;
		hands?: "left" | "right" | "both";
		exerciseId?: string;
	}>;
}

export interface ScoreHighlightConfig {
	pieceId: string;
	highlights: Array<{
		bars: [number, number];
		dimension: string;
		annotation?: string;
	}>;
}

export interface KeyboardGuideConfig {
	[key: string]: unknown;
}

export interface ReferenceBrowserConfig {
	[key: string]: unknown;
}

export interface SegmentLoopConfig {
	id: string;
	pieceId: string;
	barsStart: number;
	barsEnd: number;
	requiredCorrect: number;
	attemptsCompleted: number;
	status: "pending" | "active" | "completed" | "dismissed" | "superseded";
	dimension: string | null;
}
