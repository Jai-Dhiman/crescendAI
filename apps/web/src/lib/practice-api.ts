import type { InlineComponent } from "./types";

const API_BASE = import.meta.env.PROD
	? "https://api.crescend.ai"
	: "http://localhost:8787";

const WS_BASE = import.meta.env.PROD
	? "wss://api.crescend.ai"
	: "ws://localhost:8787";

export interface PracticeStartResponse {
	sessionId: string;
	conversationId: string;
}

export interface ChunkUploadResponse {
	r2Key: string;
	sessionId: string;
	chunkIndex: string;
}

export interface DimScores {
	dynamics: number;
	timing: number;
	pedaling: number;
	articulation: number;
	phrasing: number;
	interpretation: number;
}

export interface ObservationEvent {
	text: string;
	dimension: string;
	framing: string;
	barRange?: [number, number];
	components?: InlineComponent[];
}

export interface SessionSummary {
	observations: ObservationEvent[];
	summary: string;
}

export type PracticeMode =
	| "warming"
	| "drilling"
	| "running"
	| "winding"
	| "regular";

// ModeChangeContext: the DO sends a string description (e.g. "warming -> drilling after 30s")
export type ModeChangeContext = string | Record<string, never>;

export interface ModeChangeEvent {
	type: "mode_change";
	mode: PracticeMode;
	chunkIndex: number;
	context: ModeChangeContext;
}

export type PracticeWsEvent =
	| { type: "connected" }
	| { type: "chunk_processed"; index: number; scores: DimScores }
	| {
			type: "observation";
			text: string;
			dimension: string;
			framing: string;
			barRange?: string;
			components?: InlineComponent[];
	  }
	| {
			type: "synthesis";
			text: string;
			isFallback: boolean;
	  }
	| {
			type: "session_summary";
			observations: ObservationEvent[];
			summary: string;
			inferenceFailures?: number;
			totalChunks?: number;
	  }
	| { type: "piece_identified"; pieceId: string; composer: string; title: string; confidence: number; method: string }
	| { type: "piece_set"; query: string }
	| ModeChangeEvent
	| { type: "error"; message: string };

export const practiceApi = {
	async start(conversationId?: string): Promise<PracticeStartResponse> {
		const res = await fetch(`${API_BASE}/api/practice/start`, {
			method: "POST",
			credentials: "include",
			headers: conversationId ? { "Content-Type": "application/json" } : undefined,
			body: conversationId ? JSON.stringify({ conversationId }) : undefined,
		});
		if (!res.ok) throw new Error(`Failed to start session: ${res.status}`);
		return res.json();
	},

	async uploadChunk(
		sessionId: string,
		chunkIndex: number,
		blob: Blob,
	): Promise<ChunkUploadResponse> {
		const res = await fetch(
			`${API_BASE}/api/practice/chunk?sessionId=${sessionId}&chunkIndex=${chunkIndex}`,
			{
				method: "POST",
				credentials: "include",
				body: blob,
			},
		);
		if (!res.ok) throw new Error(`Failed to upload chunk: ${res.status}`);
		return res.json();
	},

	connectWebSocket(sessionId: string, conversationId?: string): WebSocket {
		const url = conversationId
			? `${WS_BASE}/api/practice/ws/${sessionId}?conversationId=${conversationId}`
			: `${WS_BASE}/api/practice/ws/${sessionId}`;
		return new WebSocket(url);
	},
};
