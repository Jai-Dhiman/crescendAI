// Rich message type system for inline UI cards (Slice 10)

export interface RichMessage {
	id: string;
	role: "user" | "assistant";
	content: string;
	createdAt: string;
	streaming?: boolean;
	components?: InlineComponent[];
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
	| { type: "play_passage"; config: PlayPassageConfig };

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

// Mirrors what apps/api processPlayPassage actually emits: session_id is a
// required uuid and no pieceId is ever sent. PlayPassageCard resolves the piece
// from the fetched PassageManifest, not from this config.
export interface PlayPassageConfig {
	sessionId: string;
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
