import { DurableObject } from "cloudflare:workers";
import { eq, sql } from "drizzle-orm";
import { createDb } from "../db/client";
import { sessions } from "../db/schema/sessions";
import { diagnosisArtifacts } from "../db/schema/diagnosis-artifacts";
import { messages } from "../db/schema/conversations";
import { persistDiagnosisArtifacts } from "../services/synthesis";
import type { SessionHistoryRecord, PastDiagnosisRecord } from "../services/teacher";
import type { Dimension } from "../lib/dims";
import { DIMS_6 } from "../lib/dims";
import type { Bindings, ServiceContext } from "../lib/types";
import { SessionAccumulator } from "../services/accumulator";
import { applyConfidenceGate } from "../services/confidence_gate";
import { callAmtEndpoint, callMuqEndpoint } from "../services/inference";
import { type ChunkSignal, ModeDetector } from "../services/practice-mode";
import {
	clearNeedsSynthesis,
	loadBaselinesFromDb,
	persistAccumulatedMoments,
	persistSynthesisMessage,
} from "../services/synthesis";
import type { InlineComponent } from "../services/tool-processor";
import { toLoopComponent } from "../services/segment-loops";
import {
	type SynthesisInput,
	synthesize as teacherSynthesize,
	synthesizeV6,
} from "../services/teacher";
import type { SynthesisArtifact } from "../harness/artifacts/synthesis";
import type {
	FollowerState,
	NgramIndex,
	NoteAlignment,
	PerfNote,
	PerfPedalEvent,
	RerankFeatures,
	ScoreContext,
	ScoredChunk,
	StudentBaselines,
} from "../services/wasm-bridge";
// WASM bridge — imported at top level (bridge handles missing pkg gracefully)
import * as wasm from "../services/wasm-bridge";
import {
	createInitialState,
	type SessionState,
	sessionStateSchema,
	wsIncomingMessageSchema,
} from "./session-brain.schema";

/**
 * Pure mapping from a validated SynthesisArtifact to the DO's WebSocket payload shape.
 * Exported for unit testing. `components` is always [] in V6; Plan 4 wires real exercise components.
 */
export function buildV6WsPayload(
	artifact: SynthesisArtifact,
	loopComponents?: InlineComponent[],
): {
	type: "synthesis";
	text: string;
	components: InlineComponent[];
	isFallback: false;
} {
	return {
		type: "synthesis",
		text: artifact.headline,
		components: loopComponents ?? [],
		isFallback: false,
	};
}

// Per-instance non-persisted state (lost on hibernation/eviction — intentional)
// Used only for AMT overlap context: previous chunk audio bytes
const previousChunkAudio = new WeakMap<SessionBrain, ArrayBuffer | null>();

// Minimum notes before attempting piece identification
const MIN_NOTES_FOR_IDENTIFICATION = 30;

// How often to attempt teaching moment selection (every N chunks)
const TEACHING_MOMENT_INTERVAL = 2;

export interface EnrichedChunk {
	chunkIndex: number;
	muq_scores: number[];
	midi_notes: {
		pitch: number;
		onset_ms: number;
		duration_ms: number;
		velocity: number;
	}[];
	pedal_cc: { time_ms: number; value: number }[];
	alignment: {
		perf_index: number;
		score_index: number;
		expected_onset_ms: number;
		bar: number;
	}[];
	bar_coverage: [number, number] | null;
}

export function toEnrichedChunk(
	chunkIndex: number,
	muqScores: number[],
	perfNotes: PerfNote[],
	perfPedal: PerfPedalEvent[],
	alignments: NoteAlignment[],
	barCoverage: [number, number] | null,
): EnrichedChunk {
	const midi_notes = perfNotes.map((n) => ({
		pitch: n.pitch,
		onset_ms: Math.round(n.onset * 1000),
		duration_ms: Math.round((n.offset - n.onset) * 1000),
		velocity: n.velocity,
	}));

	const pedal_cc = perfPedal.map((p) => ({
		time_ms: Math.round(p.time * 1000),
		value: p.value,
	}));

	const alignment = alignments.map((a, i) => ({
		perf_index: i,
		score_index: i,
		expected_onset_ms: Math.round(a.perf_onset * 1000 - a.onset_deviation_ms),
		bar: a.score_bar,
	}));

	return {
		chunkIndex,
		muq_scores: muqScores,
		midi_notes,
		pedal_cc,
		alignment,
		bar_coverage: barCoverage,
	};
}

export class SessionBrain extends DurableObject<Bindings> {
	private readonly ALARM_DURATION_MS = 30 * 60 * 1000; // 30 minutes

	// ─── WebSocket Lifecycle ───────────────────────────────────────────────────

	async fetch(request: Request): Promise<Response> {
		// Parse identity from path + query
		const url = new URL(request.url);

		// Deferred synthesis endpoint: POST /synthesize
		if (url.pathname === "/synthesize" && request.method === "POST") {
			await this.runSynthesisAndPersist();
			return new Response(JSON.stringify({ ok: true }), {
				headers: { "Content-Type": "application/json" },
			});
		}

		const pathParts = url.pathname.split("/");
		const sessionId = pathParts[pathParts.length - 1] ?? "";
		const studentId = url.searchParams.get("studentId") ?? "";
		const conversationId = url.searchParams.get("conversationId") ?? null;

		if (!sessionId || !studentId) {
			return new Response("Missing sessionId or studentId", { status: 400 });
		}

		// Persist identity on first connect (idempotent: only write if not present)
		const existing = await this.ctx.storage.get<SessionState>("state");
		if (!existing) {
			const initialState = createInitialState(
				sessionId,
				studentId,
				conversationId,
			);
			await this.ctx.storage.put("state", initialState);

			// Create session row in DB (marks session as open, sets needsSynthesis=true)
			try {
				const db = createDb(this.env.HYPERDRIVE);
				await db
					.insert(sessions)
					.values({
						id: sessionId,
						studentId,
						conversationId: conversationId ?? undefined,
						needsSynthesis: true,
					})
					.onConflictDoNothing();
			} catch (err) {
				const error = err as Error;
				console.error(
					JSON.stringify({
						level: "error",
						message: "failed to create session row",
						sessionId,
						error: error.message,
					}),
				);
				// Not fatal: proceed; synthesis deferred recovery will handle it
			}
		}

		// Close any existing WebSockets (reconnection handling — hibernation restores via webSocketMessage)
		const existingSockets = this.ctx.getWebSockets();
		for (const sock of existingSockets) {
			try {
				sock.close(1001, "reconnect");
			} catch {
				// already closed
			}
		}

		// Upgrade to WebSocket (hibernation API)
		const pair = new WebSocketPair();
		const [client, server] = Object.values(pair) as [WebSocket, WebSocket];
		this.ctx.acceptWebSocket(server);
		server.serializeAttachment({ sessionId, connectedAt: Date.now() });

		// Schedule 30-minute alarm (only if none pending)
		const existingAlarm = await this.ctx.storage.getAlarm();
		if (!existingAlarm) {
			await this.ctx.storage.setAlarm(Date.now() + this.ALARM_DURATION_MS);
		}

		// Welcome message
		this.sendWs(server, { type: "connected", sessionId });

		return new Response(null, { status: 101, webSocket: client });
	}

	async webSocketMessage(
		ws: WebSocket,
		message: string | ArrayBuffer,
	): Promise<void> {
		if (typeof message !== "string") return; // ignore binary frames

		let parsed: ReturnType<typeof wsIncomingMessageSchema.safeParse>;
		try {
			parsed = wsIncomingMessageSchema.safeParse(JSON.parse(message));
		} catch {
			return; // invalid JSON
		}

		if (!parsed.success) {
			console.log(
				JSON.stringify({
					level: "warn",
					message: "invalid WS message",
					issues: parsed.error.issues,
				}),
			);
			return;
		}

		const msg = parsed.data;

		switch (msg.type) {
			case "chunk_ready":
				await this.handleChunkReady(ws, msg.index, msg.r2Key);
				break;
			case "end_session":
				await this.handleEndSession();
				break;
			case "set_piece":
				await this.handleSetPiece(ws, msg.query);
				break;
		}
	}

	async webSocketClose(
		ws: WebSocket,
		code: number,
		reason: string,
	): Promise<void> {
		// Mark session as ending and schedule immediate alarm for synthesis
		try {
			const state = await this.readState();
			if (!state.sessionEnding) {
				state.sessionEnding = true;
				await this.writeState(state);
			}
		} catch {
			// Storage may not be initialized yet
		}

		// Schedule immediate synthesis (1ms alarm)
		await this.ctx.storage.setAlarm(Date.now() + 1);

		try {
			ws.close(code, reason);
		} catch {
			// already closed
		}
	}

	async alarm(): Promise<void> {
		// Convergence point for ALL exit paths (end_session, WS close, 30-min timeout)
		let state: SessionState;
		try {
			state = await this.readState();
		} catch {
			// No state — nothing to do
			return;
		}

		if (!state.synthesisCompleted) {
			await this.runSynthesisAndPersist();
		}

		if (!state.finalized) {
			await this.finalizeSession();
		}
	}

	// ─── Per-Chunk Pipeline ────────────────────────────────────────────────────

	private async handleChunkReady(
		ws: WebSocket,
		index: number,
		r2Key: string,
	): Promise<void> {
		// 1. Read state, snapshot version
		// Note: chunksInFlight is tracked separately from the version counter
		// to avoid false positives when concurrent chunks both increment it.
		// The version counter only tracks semantically significant mutations
		// (accumulator, modeDetector, piece identification, session ending).
		const state = await this.readState();
		const stateVersion = state.version;

		// Increment chunks in flight (does NOT bump version)
		state.chunksInFlight++;
		await this.ctx.storage.put("state", state);

		// 2. Fetch audio from R2
		let audioBytes: ArrayBuffer;
		try {
			const r2Object = await this.env.CHUNKS.get(r2Key);
			if (!r2Object) {
				console.error(
					JSON.stringify({
						level: "error",
						message: "R2 object not found",
						r2Key,
					}),
				);
				await this.decrementChunksInFlight();
				this.sendWs(ws, {
					type: "chunk_processed",
					index,
					scores: {
						dynamics: 0,
						timing: 0,
						pedaling: 0,
						articulation: 0,
						phrasing: 0,
						interpretation: 0,
					},
				});
				return;
			}
			audioBytes = await r2Object.arrayBuffer();
		} catch (err) {
			const error = err as Error;
			console.error(
				JSON.stringify({
					level: "error",
					message: "R2 fetch failed",
					r2Key,
					error: error.message,
				}),
			);
			await this.decrementChunksInFlight();
			this.sendWs(ws, {
				type: "chunk_processed",
				index,
				scores: {
					dynamics: 0,
					timing: 0,
					pedaling: 0,
					articulation: 0,
					phrasing: 0,
					interpretation: 0,
				},
			});
			return;
		}

		// Capture previous chunk audio for AMT context (non-persisted; null on first chunk or after eviction)
		const contextAudio = previousChunkAudio.get(this) ?? null;
		previousChunkAudio.set(this, audioBytes);

		// 3. Parallel inference — this is the major await; state may change during it
		const [muqResult, amtResult] = await Promise.allSettled([
			callMuqEndpoint(this.env, audioBytes),
			callAmtEndpoint(this.env, audioBytes, contextAudio),
		]);

		// 4. Re-read state, check version hasn't changed
		const currentState = await this.readState();
		if (currentState.version !== stateVersion) {
			// Another event mutated state during our await — bail to avoid clobbering
			console.log(
				JSON.stringify({
					level: "warn",
					message:
						"state version changed during inference, bailing chunk pipeline",
					index,
					expectedVersion: stateVersion,
					gotVersion: currentState.version,
				}),
			);
			await this.decrementChunksInFlight();
			return;
		}

		// Decrement chunksInFlight (will be re-written with final state)
		currentState.chunksInFlight = Math.max(0, currentState.chunksInFlight - 1);

		// 5. Process MuQ result — send chunk_processed immediately
		if (muqResult.status === "rejected") {
			console.error(
				JSON.stringify({
					level: "error",
					message: "MuQ inference failed",
					index,
					error:
						muqResult.reason instanceof Error
							? muqResult.reason.message
							: String(muqResult.reason),
				}),
			);
			currentState.inferenceFailures++;
			currentState.version++;
			await this.writeState(currentState);
			this.sendWs(ws, {
				type: "chunk_processed",
				index,
				scores: {
					dynamics: 0,
					timing: 0,
					pedaling: 0,
					articulation: 0,
					phrasing: 0,
					interpretation: 0,
				},
			});
			return;
		}

		const { scores: muqScores, confidences: muqConfidences } = muqResult.value;
		const scoresArray: [number, number, number, number, number, number] = [
			muqScores.dynamics,
			muqScores.timing,
			muqScores.pedaling,
			muqScores.articulation,
			muqScores.phrasing,
			muqScores.interpretation,
		];

		this.sendWs(ws, {
			type: "chunk_processed",
			index,
			scores: {
				dynamics: muqScores.dynamics,
				timing: muqScores.timing,
				pedaling: muqScores.pedaling,
				articulation: muqScores.articulation,
				phrasing: muqScores.phrasing,
				interpretation: muqScores.interpretation,
			},
		});

		currentState.scoredChunks.push({ chunkIndex: index, scores: scoresArray });

		// 6. Process AMT result (graceful degradation: AMT failure -> Tier 3)
		let perfNotes: PerfNote[] = [];
		let perfPedal: PerfPedalEvent[] = [];

		if (amtResult.status === "rejected") {
			console.error(
				JSON.stringify({
					level: "error",
					message: "AMT inference failed",
					index,
					error:
						amtResult.reason instanceof Error
							? amtResult.reason.message
							: String(amtResult.reason),
				}),
			);
			currentState.inferenceFailures++;
		} else {
			perfNotes = amtResult.value.notes;
			perfPedal = amtResult.value.pedalEvents;
		}

		// 6b. Bar analysis: align chunk + analyze (WASM; skip gracefully if not built)
		let chunkAnalysisTier = 3;
		let chunkBarRange: [number, number] | null = null;
		let barMapAlignments: import("../services/wasm-bridge").NoteAlignment[] = [];

		if (perfNotes.length > 0) {
			try {
				const scoreCtx = currentState.pieceIdentification
					? await this.loadScoreContext(
							currentState.pieceIdentification.pieceId,
						)
					: null;

				if (scoreCtx !== null) {
					// Try Tier 1/2 via WASM
					const followerState: FollowerState = {
						last_known_bar: currentState.followerState.lastKnownBar,
					};

					const alignResult = wasm.alignChunk(
						index,
						perfNotes,
						scoreCtx.score.bars,
						followerState,
					);

					currentState.followerState.lastKnownBar =
						alignResult.state.last_known_bar;

					if (alignResult.bar_map !== null) {
						barMapAlignments = alignResult.bar_map.alignments;
						const analysis = wasm.analyzeTier1(
							alignResult.bar_map,
							perfNotes,
							perfPedal,
							scoresArray,
							scoreCtx,
						);
						chunkAnalysisTier = analysis.tier;
						const barStr = analysis.bar_range;
						if (barStr !== null) {
							const parts = barStr.split("-").map(Number);
							if (
								parts.length === 2 &&
								parts[0] !== undefined &&
								parts[1] !== undefined
							) {
								chunkBarRange = [parts[0], parts[1]];
							}
						}
					} else {
						// Bar map failed, try Tier 2
						const analysis = wasm.analyzeTier2(
							perfNotes,
							perfPedal,
							scoresArray,
						);
						chunkAnalysisTier = analysis.tier;
						const barStr = analysis.bar_range;
						if (barStr !== null) {
							const parts = barStr.split("-").map(Number);
							if (
								parts.length === 2 &&
								parts[0] !== undefined &&
								parts[1] !== undefined
							) {
								chunkBarRange = [parts[0], parts[1]];
							}
						}
					}
				} else {
					// No score context: Tier 2 (MIDI only)
					const analysis = wasm.analyzeTier2(perfNotes, perfPedal, scoresArray);
					chunkAnalysisTier = analysis.tier;
					const barStr = analysis.bar_range;
					if (barStr !== null) {
						const parts = barStr.split("-").map(Number);
						if (
							parts.length === 2 &&
							parts[0] !== undefined &&
							parts[1] !== undefined
						) {
							chunkBarRange = [parts[0], parts[1]];
						}
					}
				}
			} catch (err) {
				const error = err as Error;
				console.log(
					JSON.stringify({
						level: "warn",
						message: "WASM bar analysis skipped",
						index,
						error: error.message,
					}),
				);
				// Tier 3: scores only — chunkAnalysisTier stays 3
			}
		}

		// Store enriched chunk in sibling DO storage key (non-fatal if it fails)
		try {
			const enriched = toEnrichedChunk(
				index,
				scoresArray as unknown as number[],
				perfNotes,
				perfPedal,
				barMapAlignments,
				chunkBarRange,
			);
			await this.ctx.storage.put(`chunk_enriched:${index}`, enriched);
		} catch (err) {
			const error = err as Error;
			console.log(
				JSON.stringify({
					level: "warn",
					message: "enriched chunk write failed",
					index,
					error: error.message,
				}),
			);
		}

		// 7. Try piece identification (if not locked and notes available)
		if (!currentState.pieceLocked && perfNotes.length > 0) {
			currentState.identificationNoteCount += perfNotes.length;

			if (
				currentState.identificationNoteCount >= MIN_NOTES_FOR_IDENTIFICATION
			) {
				try {
					const identified = await this.tryIdentifyPiece(
						perfNotes,
						currentState.identificationNoteCount,
					);

					if (identified !== null) {
						currentState.pieceLocked = true;
						currentState.pieceIdentification = {
							pieceId: identified.pieceId,
							confidence: identified.confidence,
							method: identified.method,
						};

						this.sendWs(ws, {
							type: "piece_identified",
							pieceId: identified.pieceId,
							composer: identified.composer,
							title: identified.title,
							confidence: identified.confidence,
							method: identified.method,
						});
					}
				} catch (err) {
					const error = err as Error;
					console.log(
						JSON.stringify({
							level: "warn",
							message: "piece identification skipped",
							error: error.message,
						}),
					);
				}
			}
		}

		// 8. Load baselines (one-time DB query)
		if (!currentState.baselinesLoaded) {
			try {
				const db = createDb(this.env.HYPERDRIVE);
				const loaded = await loadBaselinesFromDb(db, currentState.studentId);
				currentState.baselines =
					loaded !== null
						? (Object.fromEntries(
								DIMS_6.map((d) => [
									d,
									(loaded as Record<string, number>)[d] ?? 0,
								]),
							) as Record<string, number>)
						: null;
				currentState.baselinesLoaded = true;
			} catch (err) {
				const error = err as Error;
				console.error(
					JSON.stringify({
						level: "error",
						message: "baselines load failed",
						error: error.message,
					}),
				);
				// Not fatal: proceed without baselines
			}
		}

		// 9. Update ModeDetector and emit mode_change events
		const acc = SessionAccumulator.fromJSON(currentState.accumulator);
		const modeDetector =
			currentState.modeDetector !== null
				? ModeDetector.fromJSON(currentState.modeDetector)
				: new ModeDetector();

		const hasPieceMatch = currentState.pieceLocked;
		const pitchBigrams = new Set<string>();
		// Build pitch bigrams from perf notes
		for (let i = 0; i < perfNotes.length - 1; i++) {
			const n1 = perfNotes[i];
			const n2 = perfNotes[i + 1];
			if (n1 !== undefined && n2 !== undefined) {
				pitchBigrams.add(`${n1.pitch},${n2.pitch}`);
			}
		}

		const chunkSignal: ChunkSignal = {
			chunkIndex: index,
			timestampMs: Date.now(),
			barRange: chunkBarRange,
			pitchBigrams,
			hasPieceMatch,
			barsProgressing: false, // ModeDetector computes this internally
			scores: scoresArray,
		};

		const modeTransitions = modeDetector.update(chunkSignal);

		for (const transition of modeTransitions) {
			this.sendWs(ws, {
				type: "mode_change",
				mode: transition.to,
				chunkIndex: transition.chunkIndex,
				context: `${transition.from} -> ${transition.to} after ${Math.round(transition.dwellMs / 1000)}s`,
			});
		}

		// Accumulate mode transitions
		for (const transition of modeTransitions) {
			acc.accumulateModeTransition({
				from: transition.from,
				to: transition.to,
				chunkIndex: transition.chunkIndex,
				timestampMs: transition.timestampMs,
				dwellMs: transition.dwellMs,
			});

			// If exiting drilling, record drilling passage
			if (transition.from === "drilling") {
				const dp = modeDetector.takeDrillingPassage();
				if (dp !== null) {
					acc.accumulateDrillingRecord({
						barRange: dp.barRange,
						repetitionCount: dp.repetitionCount,
						firstScores: dp.firstScores,
						finalScores: scoresArray,
						startedAtChunk: dp.startedAtChunk,
						endedAtChunk: index,
					});
				}
			}
		}

		// Accumulate timeline event
		acc.accumulateTimelineEvent({
			chunkIndex: index,
			timestampMs: Date.now(),
			hasAudio: perfNotes.length > 0,
		});

		// 10. Every TEACHING_MOMENT_INTERVAL chunks: select teaching moment
		const chunkCount = currentState.scoredChunks.length;
		const shouldAttemptMoment =
			chunkCount >= 2 && chunkCount % TEACHING_MOMENT_INTERVAL === 0;

		if (shouldAttemptMoment && currentState.baselines !== null) {
			try {
				const baselines: StudentBaselines = {
					dynamics:
						(currentState.baselines as Record<string, number>)["dynamics"] ?? 0,
					timing:
						(currentState.baselines as Record<string, number>)["timing"] ?? 0,
					pedaling:
						(currentState.baselines as Record<string, number>)["pedaling"] ?? 0,
					articulation:
						(currentState.baselines as Record<string, number>)[
							"articulation"
						] ?? 0,
					phrasing:
						(currentState.baselines as Record<string, number>)["phrasing"] ?? 0,
					interpretation:
						(currentState.baselines as Record<string, number>)[
							"interpretation"
						] ?? 0,
				};

				const wasmChunks: ScoredChunk[] = currentState.scoredChunks.map(
					(c) => ({
						chunk_index: c.chunkIndex,
						scores: c.scores as [
							number,
							number,
							number,
							number,
							number,
							number,
						],
					}),
				);

				const recentObs = acc.teachingMoments
					.slice(-3)
					.map((m) => ({ dimension: m.dimension }));

				const moment = wasm.selectTeachingMoment(
					wasmChunks,
					baselines,
					recentObs,
				);

				if (moment !== null) {
					const momentDim = moment.dimension as Dimension;

					// Suppress high-uncertainty dims: if the gaussian head provided
					// confidences and this dim's sigma exceeds threshold, skip it.
					const gated =
						muqConfidences !== null
							? applyConfidenceGate(muqScores, muqConfidences)
							: null;
					const dimSuppressed =
						gated !== null && gated.suppressed.includes(momentDim);

					const accMoment = {
						chunkIndex: moment.chunk_index,
						dimension: momentDim,
						score: moment.score,
						baseline: moment.baseline,
						deviation: moment.deviation,
						isPositive: moment.is_positive,
						reasoning: moment.reasoning,
						barRange: chunkBarRange,
						analysisTier: chunkAnalysisTier,
						timestampMs: Date.now(),
						llmAnalysis: null,
					};

					if (!dimSuppressed) {
						acc.accumulateMoment(accMoment);

						// Send lightweight observation to client
						const obsText = moment.is_positive
							? `Nice work on your ${moment.dimension}.`
							: `I'm noticing something in your ${moment.dimension} -- let's talk after.`;

						const framing = moment.is_positive ? "recognition" : "correction";

						this.sendWs(ws, {
							type: "observation",
							text: obsText,
							dimension: moment.dimension,
							framing,
						});
					}
				}
			} catch (err) {
				const error = err as Error;
				console.log(
					JSON.stringify({
						level: "warn",
						message: "teaching moment selection skipped",
						error: error.message,
					}),
				);
			}
		}

		// Persist updated state
		currentState.accumulator = acc.toJSON();
		currentState.modeDetector = modeDetector.toJSON();
		currentState.version++;
		await this.writeState(currentState);

		// Reset alarm (extend session window on activity)
		await this.ctx.storage.setAlarm(Date.now() + this.ALARM_DURATION_MS);
	}

	// ─── Session End ──────────────────────────────────────────────────────────

	private async handleEndSession(): Promise<void> {
		const state = await this.readState();
		state.sessionEnding = true;
		state.version++;
		await this.writeState(state);

		// If no chunks currently processing, trigger synthesis immediately
		if (state.chunksInFlight === 0) {
			await this.ctx.storage.setAlarm(Date.now() + 1);
		}
		// If chunks are in flight, they will reset the alarm when they complete;
		// the 30-minute failsafe catches anything else.
	}

	private async handleSetPiece(ws: WebSocket, query: string): Promise<void> {
		const state = await this.readState();
		state.pieceLocked = true;
		state.pieceIdentification = null; // will be resolved on next chunk
		state.followerState = { lastKnownBar: null };
		state.version++;
		await this.writeState(state);

		this.sendWs(ws, { type: "piece_set", query });
	}

	// ─── Synthesis & Finalization ────────────────────────────────────────────

	private async runSynthesisAndPersist(): Promise<void> {
		// Claim synthesis slot atomically before the LLM call.
		// This prevents duplicate synthesis if two alarm firings race.
		const state = await this.readState();
		if (state.synthesisCompleted) return; // idempotent
		state.synthesisCompleted = true; // tentative claim
		state.version++;
		await this.writeState(state);

		const acc = SessionAccumulator.fromJSON(state.accumulator);
		const sessionDurationMs =
			acc.timeline.length > 0
				? (acc.timeline[acc.timeline.length - 1]?.timestampMs ?? 0) -
					(acc.timeline[0]?.timestampMs ?? 0)
				: 0;

		const pieceCtx =
			state.pieceIdentification !== null
				? await this.loadPieceMetadata(state.pieceIdentification.pieceId)
				: null;

		// Build practice pattern JSON for teacher input
		const practicePattern = JSON.stringify(
			this.buildPracticePattern(acc, sessionDurationMs),
		);

		// Build top moments for teacher input
		const topMoments = acc.topMoments().map((m) => {
			const obj: Record<string, unknown> = {
				dimension: m.dimension,
				deviation: Math.round(m.deviation * 1000) / 1000,
				is_positive: m.isPositive,
				reasoning: m.reasoning,
			};
			if (m.barRange) {
				obj["bar_range"] = m.barRange;
			}
			return obj;
		});

		// Build drilling records for teacher input
		const drillingRecords = acc.drillingRecords.map((dr) => {
			const entry: Record<string, unknown> = {
				repetitions: dr.repetitionCount,
				first_scores: dr.firstScores,
				final_scores: dr.finalScores,
			};
			if (dr.barRange) {
				entry["bar_range"] = dr.barRange;
			}
			return entry;
		});

		// Bulk-read enriched chunks from DO storage
		const enrichedChunkCount = state.scoredChunks.length;
		const enrichedKeys = Array.from({ length: enrichedChunkCount }, (_, i) => `chunk_enriched:${i}`);
		let enrichedChunks: EnrichedChunk[] = [];
		if (enrichedKeys.length > 0) {
			try {
				const enrichedMap = await this.ctx.storage.get<EnrichedChunk>(enrichedKeys);
				enrichedChunks = enrichedKeys
					.map((k) => enrichedMap.get(k))
					.filter((v): v is EnrichedChunk => v !== undefined);
			} catch (err) {
				const error = err as Error;
				console.log(
					JSON.stringify({ level: "warn", message: "enriched chunk bulk read failed", error: error.message }),
				);
			}
		}

		const db = createDb(this.env.HYPERDRIVE);
		const ctx: ServiceContext = { db, env: this.env };

		// Query session history (last 5 sessions for this student)
		let sessionHistory: SessionHistoryRecord[] = [];
		try {
			const historyRows = await db
				.select({
					sessionId: sessions.id,
					startedAt: sessions.startedAt,
					synthesis: messages.content,
				})
				.from(sessions)
				.leftJoin(messages, sql`${messages.sessionId} = ${sessions.id} AND ${messages.messageType} = 'synthesis'`)
				.where(sql`${sessions.studentId} = ${state.studentId} AND ${sessions.id} != ${state.sessionId}`)
				.orderBy(sql`${sessions.startedAt} DESC`)
				.limit(5);
			sessionHistory = historyRows.map((r) => ({
				sessionId: r.sessionId,
				startedAt: r.startedAt.toISOString(),
				synthesis: r.synthesis ?? null,
			}));
		} catch (err) {
			const error = err as Error;
			console.log(JSON.stringify({ level: "warn", message: "session history query failed", error: error.message }));
		}

		// Query past diagnoses (last 20 for this student)
		let pastDiagnoses: PastDiagnosisRecord[] = [];
		try {
			const diagRows = await db
				.select()
				.from(diagnosisArtifacts)
				.where(sql`${diagnosisArtifacts.studentId} = ${state.studentId}`)
				.orderBy(sql`${diagnosisArtifacts.createdAt} DESC`)
				.limit(20);
			pastDiagnoses = diagRows.map((r) => ({
				sessionId: r.sessionId,
				primaryDimension: r.primaryDimension,
				barRangeStart: r.barRangeStart ?? null,
				barRangeEnd: r.barRangeEnd ?? null,
				artifactJson: r.artifactJson,
				createdAt: r.createdAt.toISOString(),
			}));
		} catch (err) {
			const error = err as Error;
			console.log(JSON.stringify({ level: "warn", message: "past diagnoses query failed", error: error.message }));
		}

		const synthInput: SynthesisInput = {
			studentId: state.studentId,
			conversationId: state.conversationId,
			sessionDurationMs,
			practicePattern,
			topMoments,
			drillingRecords,
			pieceMetadata: pieceCtx,
			enrichedChunks,
			baselines: state.baselines as Record<string, number> | null,
			sessionHistory,
			pastDiagnoses,
			pieceId: state.pieceIdentification?.pieceId ?? null,
		};

		// teacher.synthesize() throws on failure — try/catch handles it
		// synthesisCompleted stays true but DB needsSynthesis remains true, enabling deferred retry
		try {
			if (this.env.HARNESS_V6_ENABLED === "true") {
				let artifact: SynthesisArtifact | null = null;
				let validationError: string | null = null;
				const phase1Results: Array<{ tool: string; output: unknown }> = [];

				for await (const ev of synthesizeV6(
					ctx,
					synthInput,
					state.sessionId,
					(p) => this.ctx.waitUntil(p),
				)) {
					if (ev.type === "artifact") {
						artifact = ev.value;
					} else if (ev.type === "validation_error") {
						validationError = ev.zodError;
					} else if (ev.type === "phase_error") {
						console.error(
							JSON.stringify({
								level: "error",
								message: "v6 phase_error",
								phase: ev.phase,
								error: ev.error,
								sessionId: state.sessionId,
							}),
						);
					} else if (ev.type === "phase1_tool_result" && ev.ok) {
						phase1Results.push({ tool: ev.tool, output: ev.output });
					}
				}

				// Persist diagnosis artifacts (non-fatal)
				if (phase1Results.length > 0) {
					await persistDiagnosisArtifacts(
						db,
						phase1Results,
						state.sessionId,
						state.studentId,
						state.pieceIdentification?.pieceId ?? null,
					);
				}

				if (validationError !== null) {
					console.error(
						JSON.stringify({
							level: "error",
							message: "v6 validation_error",
							error: validationError,
							sessionId: state.sessionId,
						}),
					);
					return; // leave needsSynthesis=true for retry
				}

				if (artifact === null) {
					console.error(
						JSON.stringify({
							level: "error",
							message: "v6 produced no artifact",
							sessionId: state.sessionId,
						}),
					);
					return;
				}

				const loopComponents = (artifact.assigned_loops ?? []).map((ref) =>
					toLoopComponent({
						kind: "segment_loop",
						id: ref.id,
						studentId: state.studentId,
						pieceId: ref.pieceId,
						barsStart: ref.barsStart,
						barsEnd: ref.barsEnd,
						requiredCorrect: 5,
						attemptsCompleted: 0,
						status: "active",
						dimension: null,
					}),
				);
				const wsPayload = buildV6WsPayload(artifact, loopComponents);
				const sockets = this.ctx.getWebSockets();
				for (const sock of sockets) {
					this.sendWs(sock, wsPayload);
				}

				if (state.conversationId !== null) {
					try {
						await persistSynthesisMessage(
							db,
							state.conversationId,
							wsPayload.text,
							state.sessionId,
							wsPayload.components.length > 0
								? wsPayload.components
								: undefined,
						);
						await persistAccumulatedMoments(
							db,
							state.studentId,
							state.sessionId,
							state.conversationId,
							acc.teachingMoments,
						);
						await clearNeedsSynthesis(db, state.sessionId);
					} catch (err) {
						const error = err as Error;
						console.error(
							JSON.stringify({
								level: "error",
								message: "v6 synthesis DB persist failed",
								sessionId: state.sessionId,
								error: error.message,
							}),
						);
					}
				}
				return;
			}

			// Legacy path (HARNESS_V6_ENABLED !== "true"):
			const result = await teacherSynthesize(ctx, synthInput);

			// Send synthesis over WebSocket if connection still open
			const components = result.toolResults.flatMap((r) => r.componentsJson);
			const sockets = this.ctx.getWebSockets();
			for (const sock of sockets) {
				this.sendWs(sock, {
					type: "synthesis",
					text: result.text,
					components,
					isFallback: false,
				});
			}

			// Persist to DB
			if (state.conversationId !== null) {
				try {
					await persistSynthesisMessage(
						db,
						state.conversationId,
						result.text,
						state.sessionId,
						components.length > 0 ? components : undefined,
					);
					await persistAccumulatedMoments(
						db,
						state.studentId,
						state.sessionId,
						state.conversationId,
						acc.teachingMoments,
					);
					await clearNeedsSynthesis(db, state.sessionId);
				} catch (err) {
					const error = err as Error;
					console.error(
						JSON.stringify({
							level: "error",
							message: "synthesis DB persist failed",
							sessionId: state.sessionId,
							error: error.message,
						}),
					);
					// Not fatal: accumulator_json persisted in finalizeSession as safety net
				}
			}
		} catch (err) {
			const error = err as Error;
			console.error(
				JSON.stringify({
					level: "error",
					message: "teacher.synthesize failed",
					sessionId: state.sessionId,
					error: error.message,
				}),
			);
			// synthesisCompleted stays true (no duplicate), but needsSynthesis in DB stays true
			// so the deferred /synthesize endpoint can retry via the DO
		}

		// synthesisCompleted already set at method entry (claim-before-await pattern)
	}

	private buildPracticePattern(
		acc: SessionAccumulator,
		sessionDurationMs: number,
	): unknown[] {
		if (acc.modeTransitions.length === 0) {
			return [];
		}

		const entries: unknown[] = [];

		for (let i = 0; i < acc.modeTransitions.length; i++) {
			const tr = acc.modeTransitions[i];
			const endTs =
				i + 1 < acc.modeTransitions.length
					? acc.modeTransitions[i + 1].timestampMs
					: sessionDurationMs;

			const durationMin =
				Math.round((Math.max(0, endTs - tr.timestampMs) / 60_000) * 10) / 10;

			const entry: Record<string, unknown> = {
				mode: tr.to.toLowerCase(),
				duration_min: durationMin,
			};

			if (tr.to.toLowerCase() === "drilling") {
				const dr = acc.drillingRecords.find(
					(r) => r.startedAtChunk === tr.chunkIndex,
				);
				if (dr) {
					if (dr.barRange) {
						entry["bar_range"] = dr.barRange;
					}
					entry["repetitions"] = dr.repetitionCount;
				}
			}

			entries.push(entry);
		}

		return entries;
	}

	private async finalizeSession(): Promise<void> {
		const state = await this.readState();
		if (state.finalized) return; // idempotent

		// Safety net: if synthesis didn't complete, persist accumulator_json to sessions table
		if (!state.synthesisCompleted) {
			try {
				const db = createDb(this.env.HYPERDRIVE);
				const acc = SessionAccumulator.fromJSON(state.accumulator);
				await db
					.update(sessions)
					.set({
						accumulatorJson: acc.toJSON(),
						endedAt: new Date(),
					})
					.where(eq(sessions.id, state.sessionId));
			} catch (err) {
				const error = err as Error;
				console.error(
					JSON.stringify({
						level: "error",
						message: "finalizeSession safety-net persist failed",
						sessionId: state.sessionId,
						error: error.message,
					}),
				);
			}
		} else {
			// Normal path: just update endedAt
			try {
				const db = createDb(this.env.HYPERDRIVE);
				await db
					.update(sessions)
					.set({ endedAt: new Date() })
					.where(eq(sessions.id, state.sessionId));
			} catch (err) {
				const error = err as Error;
				console.error(
					JSON.stringify({
						level: "error",
						message: "finalizeSession endedAt update failed",
						sessionId: state.sessionId,
						error: error.message,
					}),
				);
			}
		}

		// Mark finalized
		const latestState = await this.readState();
		latestState.finalized = true;
		latestState.version++;
		await this.writeState(latestState);

		// Delete enriched chunk storage keys
		const enrichedKeyCount = latestState.scoredChunks.length;
		if (enrichedKeyCount > 0) {
			const keysToDelete = Array.from({ length: enrichedKeyCount }, (_, i) => `chunk_enriched:${i}`);
			try {
				await this.ctx.storage.delete(keysToDelete);
			} catch (err) {
				const error = err as Error;
				console.log(
					JSON.stringify({ level: "warn", message: "enriched chunk cleanup failed", error: error.message }),
				);
			}
		}

		// Close all WebSockets
		const sockets = this.ctx.getWebSockets();
		for (const sock of sockets) {
			try {
				sock.close(1000, "session_ended");
			} catch {
				// already closed
			}
		}

		console.log(
			JSON.stringify({
				level: "info",
				message: "session finalized",
				sessionId: state.sessionId,
				totalChunks: state.scoredChunks.length,
				synthesisCompleted: state.synthesisCompleted,
			}),
		);
	}

	// ─── State Management ─────────────────────────────────────────────────────

	private async readState(): Promise<SessionState> {
		const raw = await this.ctx.storage.get("state");
		if (!raw) throw new Error("No state in DO storage");
		return sessionStateSchema.parse(raw);
	}

	private async writeState(state: SessionState): Promise<void> {
		await this.ctx.storage.put("state", state);
	}

	private async decrementChunksInFlight(): Promise<void> {
		try {
			const state = await this.readState();
			state.chunksInFlight = Math.max(0, state.chunksInFlight - 1);
			state.version++;
			await this.writeState(state);
		} catch {
			// Best-effort
		}
	}

	private sendWs(ws: WebSocket, data: unknown): void {
		try {
			ws.send(JSON.stringify(data));
		} catch {
			// WebSocket may have closed between check and send
		}
	}

	// ─── Helpers ──────────────────────────────────────────────────────────────

	/**
	 * Attempt multi-signal piece identification via WASM (N-gram -> rerank -> DTW).
	 * Returns null if WASM not available or identification not confident enough.
	 */
	private async tryIdentifyPiece(
		perfNotes: PerfNote[],
		_totalNotes: number,
	): Promise<{
		pieceId: string;
		composer: string;
		title: string;
		confidence: number;
		method: string;
	} | null> {
		// Load N-gram index and rerank features from SCORES R2
		let ngramIndex: NgramIndex;
		let rerankFeatures: RerankFeatures;
		let catalog: Array<{ piece_id: string; composer: string; title: string }>;

		try {
			const [idxObj, featObj, catObj] = await Promise.all([
				this.env.SCORES.get("fingerprint/v1/ngram_index.json"),
				this.env.SCORES.get("fingerprint/v1/rerank_features.json"),
				this.env.SCORES.get("fingerprint/v1/catalog.json"),
			]);

			if (!idxObj || !featObj || !catObj) {
				console.log(
					JSON.stringify({
						level: "warn",
						message: "fingerprint data not found in SCORES R2",
					}),
				);
				return null;
			}

			ngramIndex = (await idxObj.json()) as NgramIndex;
			rerankFeatures = (await featObj.json()) as RerankFeatures;
			catalog = (await catObj.json()) as typeof catalog;
		} catch (err) {
			const error = err as Error;
			console.log(
				JSON.stringify({
					level: "warn",
					message: "fingerprint data load failed",
					error: error.message,
				}),
			);
			return null;
		}

		// Stage 1: N-gram recall
		const candidates = wasm.ngramRecall(perfNotes, ngramIndex);
		if (candidates.length === 0) return null;

		// Stage 2: Rerank
		const reranked = wasm.rerankCandidates(
			perfNotes,
			candidates,
			rerankFeatures,
		);
		if (reranked.length === 0) return null;

		const topCandidate = reranked[0];
		if (!topCandidate || topCandidate.similarity < 0.5) return null;

		// Stage 3: DTW confirmation
		const pieceId = topCandidate.piece_id;
		let scoreNotes: Array<{ onset: number; pitch: number }> = [];

		try {
			const scoreCtx = await this.loadScoreContext(pieceId);
			if (scoreCtx !== null) {
				scoreNotes = scoreCtx.score.bars.flatMap((bar) =>
					bar.notes.map((n) => ({ onset: n.onset_seconds, pitch: n.pitch })),
				);
			}
		} catch {
			// DTW without score data — skip DTW, use rerank confidence
		}

		if (scoreNotes.length > 0) {
			const dtwResult = wasm.dtwConfirm(perfNotes, scoreNotes, 0.3);
			if (!dtwResult.confirmed) return null;
		}

		// Look up metadata in catalog
		const entry = catalog.find((c) => c.piece_id === pieceId);
		if (!entry) return null;

		return {
			pieceId,
			composer: entry.composer,
			title: entry.title,
			confidence: topCandidate.similarity,
			method: "fingerprint",
		};
	}

	/**
	 * Load score context from SCORES R2.
	 * Returns null if not found.
	 */
	private async loadScoreContext(
		pieceId: string,
	): Promise<ScoreContext | null> {
		try {
			const obj = await this.env.SCORES.get(`scores/v1/${pieceId}.json`);
			if (!obj) return null;
			const scoreData = (await obj.json()) as ScoreContext["score"];

			return {
				piece_id: pieceId,
				composer: scoreData.composer,
				title: scoreData.title,
				score: scoreData,
				reference: null, // reference profiles not yet wired
				match_confidence: 1.0,
			};
		} catch {
			return null;
		}
	}

	/**
	 * Load piece metadata (composer/title) for synthesis context.
	 */
	private async loadPieceMetadata(
		pieceId: string,
	): Promise<{ composer: string; title: string; pieceId: string } | null> {
		try {
			const obj = await this.env.SCORES.get(`scores/v1/${pieceId}.json`);
			if (!obj) return null;
			const data = (await obj.json()) as { composer?: string; title?: string };
			return {
				pieceId,
				composer: data.composer ?? "Unknown",
				title: data.title ?? "Unknown",
			};
		} catch {
			return null;
		}
	}
}
