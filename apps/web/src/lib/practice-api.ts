const API_BASE = import.meta.env.PROD
	? "https://api.crescend.ai"
	: "http://localhost:8787";

const WS_BASE = import.meta.env.PROD
	? "wss://api.crescend.ai"
	: "ws://localhost:8787";

export interface PracticeStartResponse {
	sessionId: string;
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

// ModeChangeContext is discriminated by the parent ModeChangeEvent.mode field
export type ModeChangeContext =
	| { bars?: [number, number]; repetition: number }
	| { piece?: string }
	| Record<string, never>;

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
	  }
	| {
			type: "session_summary";
			observations: ObservationEvent[];
			summary: string;
			inference_failures?: number;
			total_chunks?: number;
	  }
	| { type: "piece_set"; query: string }
	| ModeChangeEvent
	| { type: "error"; message: string };

export const practiceApi = {
	async start(): Promise<PracticeStartResponse> {
		const res = await fetch(`${API_BASE}/api/practice/start`, {
			method: "POST",
			credentials: "include",
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

	connectWebSocket(sessionId: string): WebSocket {
		return new WebSocket(`${WS_BASE}/api/practice/ws/${sessionId}`);
	},
};
