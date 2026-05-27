import { InferenceError } from "../lib/errors";
import type { Bindings } from "../lib/types";

const RETRY_DELAYS_MS = [10_000, 20_000, 40_000];

export interface MuqScores {
	dynamics: number;
	timing: number;
	pedaling: number;
	articulation: number;
	phrasing: number;
	interpretation: number;
}

export interface MuqConfidences {
	dynamics: number;
	timing: number;
	pedaling: number;
	articulation: number;
	phrasing: number;
	interpretation: number;
}

export interface MuqResult {
	scores: MuqScores;
	confidences: MuqConfidences | null;
	chromaBytes: Uint8Array | null;
	chromaFrames: number;
	chromaFrameRateHz: number;
}

export interface PerfNote {
	pitch: number;
	onset: number;
	offset: number;
	velocity: number;
}

export interface PerfPedalEvent {
	time: number;
	value: number; // >= 64 = on
}

export interface AmtResult {
	notes: PerfNote[];
	pedalEvents: PerfPedalEvent[];
}

interface MuqResponseRaw {
	predictions: Record<string, number>;
	confidences?: Record<string, number>;
	chroma_b64?: string;
	chroma_frames?: number;
	chroma_frame_rate_hz?: number;
}

interface AmtResponseRaw {
	midi_notes: PerfNote[];
	pedal_events: PerfPedalEvent[];
}

const MUQ_DIMS = [
	"dynamics",
	"timing",
	"pedaling",
	"articulation",
	"phrasing",
	"interpretation",
] as const;

async function fetchWithRetry(
	url: string,
	init: RequestInit,
	retryDelays: readonly number[],
): Promise<Response> {
	let lastErr = "";

	for (let attempt = 0; attempt <= retryDelays.length; attempt++) {
		let response: Response;

		try {
			response = await fetch(url, init);
		} catch (e) {
			lastErr = `fetch failed: ${e instanceof Error ? e.message : String(e)}`;
			if (attempt < retryDelays.length) {
				const delay = retryDelays[attempt];
				console.log(
					JSON.stringify({
						level: "warn",
						message: "inference fetch error, retrying",
						attempt: attempt + 1,
						delay_ms: delay,
						error: lastErr,
					}),
				);
				await new Promise((resolve) => setTimeout(resolve, delay));
				continue;
			}
			throw new InferenceError(lastErr);
		}

		if (response.status === 503 || response.status === 429) {
			const body = await response.text().catch(() => "");
			lastErr = `upstream returned ${response.status}: ${body}`;
			if (attempt < retryDelays.length) {
				const delay = retryDelays[attempt];
				console.log(
					JSON.stringify({
						level: "warn",
						message: "inference retryable error",
						status: response.status,
						attempt: attempt + 1,
						delay_ms: delay,
					}),
				);
				await new Promise((resolve) => setTimeout(resolve, delay));
				continue;
			}
			throw new InferenceError(lastErr);
		}

		if (attempt > 0) {
			console.log(
				JSON.stringify({
					level: "info",
					message: "inference succeeded after retries",
					retries: attempt,
				}),
			);
		}

		return response;
	}

	throw new InferenceError(lastErr);
}

export async function callMuqEndpoint(
	env: Bindings,
	audioBytes: ArrayBuffer,
): Promise<MuqResult> {
	const response = await fetchWithRetry(
		env.MUQ_ENDPOINT,
		{
			method: "POST",
			headers: { "Content-Type": "audio/webm;codecs=opus" },
			body: audioBytes,
		},
		RETRY_DELAYS_MS,
	);

	if (!response.ok) {
		const body = await response.text().catch(() => "");
		throw new InferenceError(
			`MuQ returned ${response.status}: ${body.slice(0, 200)}`,
		);
	}

	const raw = (await response.json()) as MuqResponseRaw;
	return parseMuqResponse(raw);
}

function encodeBase64(buffer: ArrayBuffer): string {
	return btoa(String.fromCharCode(...new Uint8Array(buffer)));
}

/** Parse a raw MuQ JSON response into a typed MuqResult. Exported for unit testing. */
export function parseMuqResponse(raw: MuqResponseRaw): MuqResult {
	const missingDims = MUQ_DIMS.filter(
		(dim) => typeof raw.predictions[dim] !== "number",
	);
	if (missingDims.length > 0) {
		throw new InferenceError(
			`MuQ response missing dimensions: ${missingDims.join(", ")}`,
		);
	}

	const scores: MuqScores = {
		dynamics: raw.predictions.dynamics,
		timing: raw.predictions.timing,
		pedaling: raw.predictions.pedaling,
		articulation: raw.predictions.articulation,
		phrasing: raw.predictions.phrasing,
		interpretation: raw.predictions.interpretation,
	};

	const confidences: MuqConfidences | null = raw.confidences
		? {
				dynamics: raw.confidences.dynamics ?? 1.0,
				timing: raw.confidences.timing ?? 1.0,
				pedaling: raw.confidences.pedaling ?? 1.0,
				articulation: raw.confidences.articulation ?? 1.0,
				phrasing: raw.confidences.phrasing ?? 1.0,
				interpretation: raw.confidences.interpretation ?? 1.0,
			}
		: null;

	let chromaBytes: Uint8Array | null = null;
	let chromaFrames = 0;
	let chromaFrameRateHz = Number.NaN;

	if (
		raw.chroma_b64 !== undefined &&
		raw.chroma_frames !== undefined &&
		raw.chroma_frames > 0
	) {
		const binaryStr = atob(raw.chroma_b64);
		const bytes = new Uint8Array(binaryStr.length);
		for (let i = 0; i < binaryStr.length; i++) {
			bytes[i] = binaryStr.charCodeAt(i);
		}
		chromaBytes = bytes;
		chromaFrames = raw.chroma_frames;
		chromaFrameRateHz = raw.chroma_frame_rate_hz ?? 50.0;
	}

	return { scores, confidences, chromaBytes, chromaFrames, chromaFrameRateHz };
}

export async function callAmtEndpoint(
	env: Bindings,
	chunkAudio: ArrayBuffer,
	contextAudio: ArrayBuffer | null,
): Promise<AmtResult> {
	const payload = JSON.stringify({
		chunk_audio: encodeBase64(chunkAudio),
		context_audio: contextAudio !== null ? encodeBase64(contextAudio) : null,
	});

	const response = await fetchWithRetry(
		`${env.AMT_ENDPOINT}/transcribe`,
		{
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: payload,
		},
		RETRY_DELAYS_MS,
	);

	if (!response.ok) {
		const body = await response.text().catch(() => "");
		throw new InferenceError(
			`AMT returned ${response.status}: ${body.slice(0, 200)}`,
		);
	}

	const raw = (await response.json()) as AmtResponseRaw;

	return {
		notes: raw.midi_notes,
		pedalEvents: raw.pedal_events,
	};
}
