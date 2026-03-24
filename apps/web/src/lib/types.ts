// Rich message type system for inline UI cards (Slice 10)

export interface RichMessage {
	id: string;
	role: "user" | "assistant";
	content: string;
	created_at: string;
	streaming?: boolean;
	components?: InlineComponent[];
	dimension?: string;
	message_type?: "chat" | "observation" | "session_start" | "session_end" | "summary" | "synthesis";
	session_id?: string;
	framing?: string;
}

export type InlineComponent =
	| { type: "exercise_set"; config: ExerciseSetConfig }
	| { type: "score_highlight"; config: ScoreHighlightConfig }
	| { type: "keyboard_guide"; config: KeyboardGuideConfig }
	| { type: "reference_browser"; config: ReferenceBrowserConfig };

export interface ExerciseSetConfig {
	source_passage: string;
	target_skill: string;
	exercises: Array<{
		title: string;
		instruction: string;
		focus_dimension: string;
		hands?: "left" | "right" | "both";
		exercise_id?: string;
	}>;
}

// Stub interfaces -- flesh out when implementing Slice 10
export interface ScoreHighlightConfig {
	[key: string]: unknown;
}

export interface KeyboardGuideConfig {
	[key: string]: unknown;
}

export interface ReferenceBrowserConfig {
	[key: string]: unknown;
}
