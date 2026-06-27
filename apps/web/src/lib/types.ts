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
	| { type: "session_data"; config: SessionDataConfig }
	| { type: "play_passage"; config: PlayPassageConfig }
	| { type: "segment_loop"; config: SegmentLoopConfig }
	| { type: "pending_exercise"; config: PendingExerciseConfig };

export interface ExerciseSetConfig {
	sourcePassage: string;
	targetSkill: string;
	scoreClip?: {
		pieceId: string;
		bars: [number, number];
		tempoFactor?: number;
		transpose?: number;
	};
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
	title: string;
	description: string;
	hands: "left" | "right" | "both";
	fingering?: string;
}

// Emitted by the `show_session_data` tool (apps/api tool-processor.ts).
// `data` shape depends on `queryType`; it crosses the JSON wire, so the
// renderer narrows it defensively rather than trusting the static type.
export type SessionDataQueryType =
	| "dimension_history"
	| "recent_sessions"
	| "session_detail";

export interface SessionDataObservationRow {
	id: string;
	dimension: string;
	dimensionScore: number | null;
	observationText: string | null;
	framing: string | null;
	createdAt: string;
	sessionId: string | null;
}

export interface SessionDataSessionRow {
	id: string;
	startedAt: string;
	endedAt: string | null;
	avgDynamics: number | null;
	avgTiming: number | null;
	avgPedaling: number | null;
	avgArticulation: number | null;
	avgPhrasing: number | null;
	avgInterpretation: number | null;
}

export interface SessionDataConfig {
	queryType: SessionDataQueryType;
	studentId: string;
	data: unknown;
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
