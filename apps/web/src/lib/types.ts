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
	| { type: "play_passage"; config: PlayPassageConfig }
	| { type: "segment_loop"; config: SegmentLoopConfig }
	| { type: "pending_exercise"; config: PendingExerciseConfig };

export interface ExerciseSetConfig {
	sourcePassage: string;
	targetSkill: string;
	scoreClip?: { pieceId: string; bars: [number, number]; tempoFactor?: number };
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

export interface PlayPassageConfig {
	pieceId: string;
	sessionId?: string;
	bars: [number, number];
	focusBars?: [number, number];
	dimension: string;
	annotation: string;
}

export interface PassageManifest {
	source: { kind: "session"; sessionId: string };
	pieceId: string;
	bars: [number, number];
	chunks: Array<{ url: string; chunkIndex: number; durationSec: number }>;
	startOffsetSec: number;
	endOffsetSec: number;
	barTimeline: Array<{ bar: number; tSec: number }>;
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

export interface PendingExerciseConfig {
	exerciseId: string;
	focusDimension: string;
	previewTitle: string;
}
