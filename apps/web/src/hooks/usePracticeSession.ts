import { useCallback, useEffect, useRef, useState } from "react";
import { ObservationThrottle } from "../lib/observation-throttle";
import type {
	DimScores,
	ObservationEvent,
	PracticeWsEvent,
} from "../lib/practice-api";
import { practiceApi } from "../lib/practice-api";
import { Sentry } from "../lib/sentry";
import { useAudioActivity } from "./useAudioActivity";
import { createLogger } from "../lib/logger";

const chunkLog = createLogger("ChunkGate");

type ChunkGateState = "waiting" | "buffering";

export type PracticeState =
	| "idle"
	| "requesting-mic"
	| "connecting"
	| "recording"
	| "summarizing"
	| "error";

const MAX_RECONNECTS = 5;
const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 30_000;

export type WsStatus = "connected" | "reconnecting" | "disconnected";
type ChunkStatus = "uploading" | "complete" | "failed";
export interface ChunkState {
	index: number;
	status: ChunkStatus;
}

export interface UsePracticeSessionReturn {
	state: PracticeState;
	elapsedSeconds: number;
	observations: ObservationEvent[];
	latestScores: DimScores | null;
	summary: string | null;
	error: string | null;
	analyserNode: AnalyserNode | null;
	chunksProcessed: number;
	chunkStates: ChunkState[];
	wsStatus: WsStatus;
	isOnline: boolean;
	isPlaying: boolean;
	energy: number;
	start: () => Promise<void>;
	stop: () => void;
	setPiece: (query: string) => void;
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
	const [chunkStates, setChunkStates] = useState<ChunkState[]>([]);
	const [wsStatus, setWsStatus] = useState<WsStatus>("disconnected");
	const [isOnline, setIsOnline] = useState(
		typeof navigator !== "undefined" ? navigator.onLine : true,
	);

	const sessionIdRef = useRef<string | null>(null);
	const wsRef = useRef<WebSocket | null>(null);
	const mediaRecorderRef = useRef<MediaRecorder | null>(null);
	const audioContextRef = useRef<AudioContext | null>(null);
	const streamRef = useRef<MediaStream | null>(null);
	const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
	const chunkIndexRef = useRef(0);
	const reconnectAttemptsRef = useRef(0);
	const stateRef = useRef<PracticeState>("idle");
	const throttleRef = useRef(new ObservationThrottle());
	const isOnlineRef = useRef(isOnline);
	const offlineQueueRef = useRef<Array<{ index: number; blob: Blob }>>([]);
	const analyserRef = useRef<AnalyserNode | null>(null);
	const chunkGateRef = useRef<ChunkGateState>("waiting");
	const chunkTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

	// Keep stateRef in sync
	useEffect(() => {
		stateRef.current = state;
	}, [state]);

	useEffect(() => {
		isOnlineRef.current = isOnline;
	}, [isOnline]);

	// Audio activity detection (spectral energy + debounced play/silence)
	const { isPlaying, energy } = useAudioActivity(analyserRef);
	const isPlayingRef = useRef(false);

	useEffect(() => {
		isPlayingRef.current = isPlaying;
	}, [isPlaying]);

	// Network online/offline detection
	useEffect(() => {
		function handleOnline() {
			setIsOnline(true);
		}
		function handleOffline() {
			setIsOnline(false);
		}
		window.addEventListener("online", handleOnline);
		window.addEventListener("offline", handleOffline);
		return () => {
			window.removeEventListener("online", handleOnline);
			window.removeEventListener("offline", handleOffline);
		};
	}, []);

	// Chunk gating: start/stop uploading based on piano activity
	useEffect(() => {
		// Only gate when actively recording
		if (state !== "recording") return;
		const recorder = mediaRecorderRef.current;
		const sessionId = sessionIdRef.current;
		if (!recorder || !sessionId) return;

		if (isPlaying && chunkGateRef.current === "waiting") {
			// Transition: WAITING -> BUFFERING
			chunkGateRef.current = "buffering";
			chunkLog.log("State: WAITING -> BUFFERING");

			// Start 15s interval timer. First requestData() includes accumulated
			// silence -- that's fine per spec (no discard mechanism needed).
			chunkLog.log("Chunk timer started (15s)");
			chunkTimerRef.current = setInterval(() => {
				if (recorder.state === "recording") {
					recorder.requestData();
				}
			}, 15000);
		} else if (!isPlaying && chunkGateRef.current === "buffering") {
			// Transition: BUFFERING -> WAITING
			chunkLog.log("State: BUFFERING -> WAITING (silence offset)");

			// Clear the chunk timer
			if (chunkTimerRef.current) {
				clearInterval(chunkTimerRef.current);
				chunkTimerRef.current = null;
			}

			// Flush partial chunk
			if (recorder.state === "recording") {
				recorder.requestData();
				chunkLog.log("Partial chunk flushed on silence offset");
			}

			chunkGateRef.current = "waiting";
		}
	}, [isPlaying, state]);

	const updateChunkState = useCallback((index: number, status: ChunkStatus) => {
		setChunkStates((prev) => {
			const existing = prev.findIndex((c) => c.index === index);
			if (existing >= 0) {
				const updated = [...prev];
				updated[existing] = { index, status };
				return updated;
			}
			return [...prev, { index, status }];
		});
	}, []);

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
		throttleRef.current.reset();
		offlineQueueRef.current = [];
		if (chunkTimerRef.current) {
			clearInterval(chunkTimerRef.current);
			chunkTimerRef.current = null;
		}
		chunkGateRef.current = "waiting";
		analyserRef.current = null;
	}, []);

	const handleWsMessage = useCallback(
		(event: MessageEvent) => {
			const data: PracticeWsEvent = JSON.parse(event.data);
			switch (data.type) {
				case "chunk_processed": {
					setLatestScores(data.scores);
					setChunksProcessed((prev) => prev + 1);
					// Check if throttle can release a queued observation
					const released = throttleRef.current.onChunkProcessed();
					if (released) {
						setObservations((prev) => [...prev, released]);
					}
					break;
				}
				case "observation": {
					const obs: ObservationEvent = {
						text: data.text,
						dimension: data.dimension,
						framing: data.framing,
					};
					const immediate = throttleRef.current.enqueue(obs);
					if (immediate) {
						setObservations((prev) => [...prev, immediate]);
					}
					break;
				}
				case "session_summary": {
					// Drain any undelivered queued observations
					const drained = throttleRef.current.drain();
					const allObs = [...data.observations];
					for (const obs of drained) {
						if (!allObs.some((o) => o.text === obs.text)) {
							allObs.push(obs);
						}
					}
					setObservations(allObs);

					// Build summary string
					const obsLines = allObs.map((o) => `- ${o.text}`).join("\n");
					const chunksCount = chunkIndexRef.current;
					const builtSummary = obsLines
						? `I listened to ${chunksCount} sections of your playing.\n\nDuring the session, I noticed:\n${obsLines}\n\nWant to hear more about any of these?`
						: `I listened to ${chunksCount} sections of your playing.\n\nI didn't notice anything specific to flag this time. Want to talk about how it felt?`;
					setSummary(builtSummary);
					setState("idle");
					cleanup();
					break;
				}
				case "piece_set":
					console.log("Piece context set:", data.query);
					break;
				case "error":
					setError(data.message);
					break;
			}
		},
		[cleanup],
	);

	const connectWebSocket = useCallback(
		(sessionId: string): Promise<WebSocket> => {
			return new Promise((resolve, reject) => {
				const ws = practiceApi.connectWebSocket(sessionId);
				wsRef.current = ws;

				ws.onmessage = handleWsMessage;

				ws.onopen = () => {
					reconnectAttemptsRef.current = 0;
					setWsStatus("connected");
					resolve(ws);
				};

				ws.onerror = () => {
					Sentry.captureException(new Error("WebSocket failed to connect"), {
						extra: { sessionId },
					});
					reject(new Error("WebSocket failed to connect"));
				};

				ws.onclose = () => {
					if (
						stateRef.current === "recording" &&
						reconnectAttemptsRef.current < MAX_RECONNECTS
					) {
						setWsStatus("reconnecting");
						const attempt = reconnectAttemptsRef.current++;
						const delay = Math.min(
							RECONNECT_BASE_DELAY_MS * 2 ** attempt,
							RECONNECT_MAX_DELAY_MS,
						);
						setTimeout(() => {
							if (stateRef.current === "recording" && sessionIdRef.current) {
								connectWebSocket(sessionIdRef.current).catch(() => {
									Sentry.captureMessage("WebSocket reconnection failed", {
										level: "error",
										extra: { attempts: reconnectAttemptsRef.current },
									});
									setWsStatus("disconnected");
									setError("Connection lost. Please try again.");
									setState("error");
									cleanup();
								});
							}
						}, delay);
					} else if (stateRef.current === "recording") {
						setWsStatus("disconnected");
						setError("Connection lost. Please try again.");
						setState("error");
						cleanup();
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
		setChunkStates([]);
		setWsStatus("disconnected");

		// 1. Request mic
		let stream: MediaStream;
		try {
			stream = await navigator.mediaDevices.getUserMedia({ audio: true });
			streamRef.current = stream;
		} catch (e) {
			Sentry.captureException(e, {
				extra: { context: "microphone-access" },
			});
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
		analyserRef.current = analyser;

		// 3. Start session on server
		setState("connecting");
		let sessionId: string;
		try {
			const { sessionId: sid } = await practiceApi.start();
			sessionId = sid;
			sessionIdRef.current = sid;
		} catch (e) {
			Sentry.captureException(e, {
				extra: { context: "session-start" },
			});
			cleanup();
			setState("error");
			setError("Failed to start practice session. Please try again.");
			return;
		}

		// 4. Connect WebSocket
		try {
			await connectWebSocket(sessionId);
		} catch (e) {
			Sentry.captureException(e, {
				extra: { context: "websocket-connect", sessionId },
			});
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

		async function uploadWithRetry(
			sid: string,
			idx: number,
			blob: Blob,
		): Promise<void> {
			updateChunkState(idx, "uploading");
			for (let attempt = 0; attempt < 2; attempt++) {
				try {
					const { r2Key } = await practiceApi.uploadChunk(sid, idx, blob);
					updateChunkState(idx, "complete");
					chunkLog.log(`Upload complete: chunk #${idx} -> r2Key=${r2Key}`);
					const ws = wsRef.current;
					if (ws?.readyState === WebSocket.OPEN) {
						ws.send(JSON.stringify({ type: "chunk_ready", index: idx, r2Key }));
						chunkLog.log(`WS sent: chunk_ready #${idx}`);
					}
					return;
				} catch (e) {
					// Auth expiry: surface immediately, do not retry
					if (e instanceof Error && e.message.includes("401")) {
						updateChunkState(idx, "failed");
						setError("Session expired. Please sign in again.");
						setState("error");
						cleanup();
						return;
					}
					if (attempt === 0) {
						await new Promise((r) => setTimeout(r, 2000));
					} else {
						updateChunkState(idx, "failed");
						Sentry.captureException(e, {
							extra: { chunkIndex: idx, sessionId: sid },
						});
					}
				}
			}
		}

		recorder.ondataavailable = async (event) => {
			if (event.data.size === 0) return;
			const idx = chunkIndexRef.current++;
			chunkLog.log(
				`Chunk #${idx} cut: ${(event.data.size / 1024).toFixed(1)}KB -- uploading to R2`,
			);

			if (!isOnlineRef.current) {
				offlineQueueRef.current.push({ index: idx, blob: event.data });
				updateChunkState(idx, "uploading");
				return;
			}

			await uploadWithRetry(sessionId, idx, event.data);
		};

		recorder.start(); // Continuous mode -- chunks cut manually via requestData()
		chunkGateRef.current = "waiting";
		chunkLog.log("MediaRecorder started in continuous mode (gated by audio activity)");
		setState("recording");

		// 6. Start elapsed timer + throttle tick
		const startTime = Date.now();
		timerRef.current = setInterval(() => {
			setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
			const released = throttleRef.current.tick();
			if (released) {
				setObservations((prev) => [...prev, released]);
			}
		}, 1000);
	}, [cleanup, connectWebSocket, updateChunkState]);

	// Flush offline queue when back online
	useEffect(() => {
		if (!isOnline || !sessionIdRef.current) return;
		const queue = offlineQueueRef.current;
		if (queue.length === 0) return;

		const sessionId = sessionIdRef.current;
		offlineQueueRef.current = [];

		(async () => {
			for (const { index, blob } of queue) {
				try {
					const { r2Key } = await practiceApi.uploadChunk(
						sessionId,
						index,
						blob,
					);
					updateChunkState(index, "complete");
					const ws = wsRef.current;
					if (ws?.readyState === WebSocket.OPEN) {
						ws.send(JSON.stringify({ type: "chunk_ready", index, r2Key }));
					}
				} catch (e) {
					updateChunkState(index, "failed");
					Sentry.captureException(e, {
						extra: { chunkIndex: index, sessionId },
					});
				}
			}
		})();
	}, [isOnline, updateChunkState]);

	const stop = useCallback(() => {
		if (state !== "recording") return;

		// If session is too short (no chunks recorded), skip inference
		// Minimum blob size to consider as real audio (~1s of Opus audio)
		const MIN_FLUSH_BLOB_SIZE = 10_000; // ~10KB

		if (chunkIndexRef.current === 0) {
			// No chunks sent -- flush buffer as last resort
			const recorder = mediaRecorderRef.current;
			if (recorder?.state === "recording") {
				// Temporarily override ondataavailable to check blob size
				const origHandler = recorder.ondataavailable;
				recorder.ondataavailable = async (event) => {
					recorder.ondataavailable = origHandler; // restore
					if (event.data.size < MIN_FLUSH_BLOB_SIZE) {
						// Too small -- likely just silence
						chunkLog.log(
							`Flush blob too small: ${(event.data.size / 1024).toFixed(1)}KB < ${(MIN_FLUSH_BLOB_SIZE / 1024).toFixed(0)}KB threshold`,
						);
						setError(
							"I couldn't hear any playing. Make sure your microphone is picking up the piano.",
						);
						setState("idle");
						cleanup();
					} else {
						// Blob has content -- upload and proceed normally
						chunkLog.log(
							`Flush blob accepted: ${(event.data.size / 1024).toFixed(1)}KB -- uploading`,
						);
						if (origHandler) {
							origHandler.call(recorder, event);
						}
						setState("summarizing");
						if (wsRef.current?.readyState === WebSocket.OPEN) {
							wsRef.current.send(JSON.stringify({ type: "end_session" }));
						}
					}
				};
				recorder.requestData();
			} else {
				setError(
					"I couldn't hear any playing. Make sure your microphone is picking up the piano.",
				);
				setState("idle");
				cleanup();
			}
			return;
		}

		setState("summarizing");

		// Release mic and timer immediately
		if (timerRef.current) {
			clearInterval(timerRef.current);
			timerRef.current = null;
		}
		if (audioContextRef.current?.state !== "closed") {
			audioContextRef.current?.close();
		}
		audioContextRef.current = null;
		streamRef.current?.getTracks().forEach((t) => {
			t.stop();
		});
		streamRef.current = null;

		// Stop recording (triggers final ondataavailable)
		if (mediaRecorderRef.current?.state === "recording") {
			mediaRecorderRef.current.stop();
		}

		// Tell DO to end session
		if (wsRef.current?.readyState === WebSocket.OPEN) {
			wsRef.current.send(JSON.stringify({ type: "end_session" }));
		}

		// WS stays open to receive session_summary
		mediaRecorderRef.current = null;
	}, [state, cleanup]);

	const setPiece = useCallback((query: string) => {
		if (wsRef.current?.readyState === WebSocket.OPEN) {
			wsRef.current.send(JSON.stringify({
				type: "set_piece",
				query,
			}));
		}
	}, []);

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
		chunkStates,
		wsStatus,
		isOnline,
		isPlaying,
		energy,
		start,
		stop,
		setPiece,
	};
}
