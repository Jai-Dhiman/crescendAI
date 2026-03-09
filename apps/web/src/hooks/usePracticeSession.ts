import { useCallback, useEffect, useRef, useState } from "react";
import type {
	DimScores,
	ObservationEvent,
	PracticeWsEvent,
} from "../lib/practice-api";
import { practiceApi } from "../lib/practice-api";

export type PracticeState =
	| "idle"
	| "requesting-mic"
	| "connecting"
	| "recording"
	| "summarizing"
	| "error";

const MAX_RECONNECTS = 3;
const RECONNECT_DELAY_MS = 2000;

export interface UsePracticeSessionReturn {
	state: PracticeState;
	elapsedSeconds: number;
	observations: ObservationEvent[];
	latestScores: DimScores | null;
	summary: string | null;
	error: string | null;
	analyserNode: AnalyserNode | null;
	chunksProcessed: number;
	start: () => Promise<void>;
	stop: () => void;
}

export function usePracticeSession(): UsePracticeSessionReturn {
	const [state, setState] = useState<PracticeState>("idle");
	const [elapsedSeconds, setElapsedSeconds] = useState(0);
	const [observations, setObservations] = useState<ObservationEvent[]>([]);
	const [latestScores, setLatestScores] = useState<DimScores | null>(null);
	const [summary, setSummary] = useState<string | null>(null);
	const [error, setError] = useState<string | null>(null);
	const [analyserNode, setAnalyserNode] = useState<AnalyserNode | null>(null);
	const [chunksProcessed, setChunksProcessed] = useState(0);

	const sessionIdRef = useRef<string | null>(null);
	const wsRef = useRef<WebSocket | null>(null);
	const mediaRecorderRef = useRef<MediaRecorder | null>(null);
	const audioContextRef = useRef<AudioContext | null>(null);
	const streamRef = useRef<MediaStream | null>(null);
	const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
	const chunkIndexRef = useRef(0);
	const reconnectAttemptsRef = useRef(0);
	const stateRef = useRef<PracticeState>("idle");

	// Keep stateRef in sync
	useEffect(() => {
		stateRef.current = state;
	}, [state]);

	const cleanup = useCallback(() => {
		if (timerRef.current) clearInterval(timerRef.current);
		if (mediaRecorderRef.current?.state === "recording")
			mediaRecorderRef.current.stop();
		if (audioContextRef.current?.state !== "closed")
			audioContextRef.current?.close();
		streamRef.current?.getTracks().forEach((t) => {
			t.stop();
		});
		wsRef.current?.close();

		timerRef.current = null;
		mediaRecorderRef.current = null;
		audioContextRef.current = null;
		streamRef.current = null;
		wsRef.current = null;
		sessionIdRef.current = null;
		chunkIndexRef.current = 0;
		reconnectAttemptsRef.current = 0;
	}, []);

	const handleWsMessage = useCallback((event: MessageEvent) => {
		const data: PracticeWsEvent = JSON.parse(event.data);
		switch (data.type) {
			case "chunk_processed":
				setLatestScores(data.scores);
				setChunksProcessed((prev) => prev + 1);
				break;
			case "observation":
				setObservations((prev) => [
					...prev,
					{
						text: data.text,
						dimension: data.dimension,
						framing: data.framing,
					},
				]);
				break;
			case "session_summary":
				setSummary(data.summary);
				setObservations(data.observations);
				setState("idle");
				break;
			case "error":
				setError(data.message);
				break;
		}
	}, []);

	const connectWebSocket = useCallback(
		(sessionId: string): Promise<WebSocket> => {
			return new Promise((resolve, reject) => {
				const ws = practiceApi.connectWebSocket(sessionId);
				wsRef.current = ws;

				ws.onmessage = handleWsMessage;

				ws.onopen = () => {
					reconnectAttemptsRef.current = 0;
					resolve(ws);
				};

				ws.onerror = () => {
					reject(new Error("WebSocket failed to connect"));
				};

				ws.onclose = () => {
					// Only attempt reconnect if still recording
					if (
						stateRef.current === "recording" &&
						reconnectAttemptsRef.current < MAX_RECONNECTS
					) {
						reconnectAttemptsRef.current++;
						setTimeout(() => {
							if (stateRef.current === "recording" && sessionIdRef.current) {
								connectWebSocket(sessionIdRef.current).catch(() => {
									setError("Connection lost. Please try again.");
									setState("error");
									cleanup();
								});
							}
						}, RECONNECT_DELAY_MS);
					}
				};
			});
		},
		[handleWsMessage, cleanup],
	);

	const start = useCallback(async () => {
		setState("requesting-mic");
		setElapsedSeconds(0);
		setObservations([]);
		setLatestScores(null);
		setSummary(null);
		setError(null);
		setChunksProcessed(0);

		// 1. Request mic
		let stream: MediaStream;
		try {
			stream = await navigator.mediaDevices.getUserMedia({ audio: true });
			streamRef.current = stream;
		} catch (_e) {
			setState("error");
			setError(
				"Microphone access denied. Please allow mic access and try again.",
			);
			return;
		}

		// 2. Set up AudioContext + AnalyserNode for waveform visualization
		const audioCtx = new AudioContext();
		audioContextRef.current = audioCtx;
		const source = audioCtx.createMediaStreamSource(stream);
		const analyser = audioCtx.createAnalyser();
		analyser.fftSize = 256;
		source.connect(analyser);
		setAnalyserNode(analyser);

		// 3. Start session on server
		setState("connecting");
		let sessionId: string;
		try {
			const { sessionId: sid } = await practiceApi.start();
			sessionId = sid;
			sessionIdRef.current = sid;
		} catch (_e) {
			cleanup();
			setState("error");
			setError("Failed to start practice session. Please try again.");
			return;
		}

		// 4. Connect WebSocket
		try {
			await connectWebSocket(sessionId);
		} catch (_e) {
			cleanup();
			setState("error");
			setError("Failed to connect. Please try again.");
			return;
		}

		// 5. Start MediaRecorder with 15s chunks
		const recorder = new MediaRecorder(stream, {
			mimeType: "audio/webm;codecs=opus",
		});
		mediaRecorderRef.current = recorder;
		chunkIndexRef.current = 0;

		recorder.ondataavailable = async (event) => {
			if (event.data.size === 0) return;
			const idx = chunkIndexRef.current++;
			try {
				const { r2Key } = await practiceApi.uploadChunk(
					sessionId,
					idx,
					event.data,
				);
				const ws = wsRef.current;
				if (ws?.readyState === WebSocket.OPEN) {
					ws.send(JSON.stringify({ type: "chunk_ready", index: idx, r2Key }));
				}
			} catch (e) {
				console.error("Chunk upload failed:", e);
			}
		};

		recorder.start(15000); // 15-second timeslice
		setState("recording");

		// 6. Start elapsed timer
		const startTime = Date.now();
		timerRef.current = setInterval(() => {
			setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
		}, 1000);
	}, [cleanup, connectWebSocket]);

	const stop = useCallback(() => {
		if (state !== "recording") return;

		// If session is too short (no chunks recorded), skip inference
		if (chunkIndexRef.current === 0) {
			setError("Play for at least 15 seconds so I can listen.");
			setState("idle");
			cleanup();
			return;
		}

		setState("summarizing");

		// Stop recording (triggers final ondataavailable)
		if (mediaRecorderRef.current?.state === "recording") {
			mediaRecorderRef.current.stop();
		}

		// Tell DO to end session
		if (wsRef.current?.readyState === WebSocket.OPEN) {
			wsRef.current.send(JSON.stringify({ type: "end_session" }));
		}

		// Timer cleanup (WS stays open until summary arrives)
		if (timerRef.current) {
			clearInterval(timerRef.current);
			timerRef.current = null;
		}
	}, [state, cleanup]);

	// Graceful stop on page unload
	useEffect(() => {
		function handleBeforeUnload() {
			if (wsRef.current?.readyState === WebSocket.OPEN) {
				wsRef.current.send(JSON.stringify({ type: "end_session" }));
				wsRef.current.close();
			}
			if (mediaRecorderRef.current?.state === "recording") {
				mediaRecorderRef.current.stop();
			}
			streamRef.current?.getTracks().forEach((t) => {
				t.stop();
			});
		}
		window.addEventListener("beforeunload", handleBeforeUnload);
		return () => window.removeEventListener("beforeunload", handleBeforeUnload);
	}, []);

	return {
		state,
		elapsedSeconds,
		observations,
		latestScores,
		summary,
		error,
		analyserNode,
		chunksProcessed,
		start,
		stop,
	};
}
