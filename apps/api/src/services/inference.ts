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

interface AmtResponseRaw {
	midi_notes: PerfNote[];
	pedal_events: PerfPedalEvent[];
}

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

function encodeBase64(buffer: ArrayBuffer): string {
	const bytes = new Uint8Array(buffer);
	// Encode in fixed-size blocks; spreading the whole array into
	// String.fromCharCode overflows the call stack for real-sized audio (~720KB).
	const BLOCK = 0x8000;
	let binary = "";
	for (let i = 0; i < bytes.length; i += BLOCK) {
		binary += String.fromCharCode(...bytes.subarray(i, i + BLOCK));
	}
	return btoa(binary);
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

	const raw = (await response.json()) as AmtResponseRaw & {
		error?: { code?: string; message?: string };
	};

	// The AMT endpoint can return an error-shaped body with HTTP 200 (the local
	// dev server does this on internal exceptions, and a degenerate transcription
	// can omit midi_notes). Validate the shape here and surface it as an
	// InferenceError so the caller's graceful-degradation path (AMT failure ->
	// Tier 3) handles it, instead of letting an undefined `notes` reach
	// `perfNotes.length` in the DO and crash the isolate.
	if (raw.error) {
		throw new InferenceError(
			`AMT returned an error payload: ${raw.error.message ?? raw.error.code ?? "unknown"}`,
		);
	}
	if (!Array.isArray(raw.midi_notes)) {
		throw new InferenceError("AMT response missing midi_notes array");
	}

	return {
		notes: raw.midi_notes,
		pedalEvents: Array.isArray(raw.pedal_events) ? raw.pedal_events : [],
	};
}
