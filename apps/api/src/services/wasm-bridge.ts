// apps/api/src/services/wasm-bridge.ts
//
// Typed TypeScript wrappers over WASM modules.
// This is the ONLY file that imports from wasm/*/pkg/.
// All other code imports from this bridge.

// ═══════════════════════════════════════════════════
// Types (mirror Rust serde output)
// ═══════════════════════════════════════════════════

// --- STOP / teaching moment types ---

export interface ScoredChunk {
  chunk_index: number;
  /** Fixed 6-element array: [dynamics, timing, pedaling, articulation, phrasing, interpretation] */
  scores: [number, number, number, number, number, number];
}

export interface StudentBaselines {
  dynamics: number;
  timing: number;
  pedaling: number;
  articulation: number;
  phrasing: number;
  interpretation: number;
}

export interface RecentObservation {
  dimension: string;
}

export interface TeachingMoment {
  chunk_index: number;
  dimension: string;
  score: number;
  baseline: number;
  deviation: number;
  stop_probability: number;
  reasoning: string;
  is_positive: boolean;
}

// --- Score follower types ---

export interface PerfNote {
  pitch: number;
  onset: number;
  offset: number;
  velocity: number;
}

export interface PerfPedalEvent {
  time: number;
  value: number;
}

export interface ScoreNote {
  pitch: number;
  pitch_name: string;
  velocity: number;
  onset_tick: number;
  onset_seconds: number;
  duration_ticks: number;
  duration_seconds: number;
  track: number;
}

export interface ScorePedalEvent {
  type: string;
  tick: number;
  seconds: number;
}

export interface ScoreBar {
  bar_number: number;
  start_tick: number;
  start_seconds: number;
  time_signature: string;
  notes: ScoreNote[];
  pedal_events: ScorePedalEvent[];
  note_count: number;
  pitch_range: number[];
  mean_velocity: number;
}

export interface ScoreData {
  piece_id: string;
  composer: string;
  title: string;
  key_signature: string | null;
  time_signatures: unknown[];
  tempo_markings: unknown[];
  total_bars: number;
  bars: ScoreBar[];
}

export interface ReferenceBar {
  bar_number: number;
  velocity_mean: number;
  velocity_std: number;
  onset_deviation_mean_ms: number;
  onset_deviation_std_ms: number;
  pedal_duration_mean_beats: number | null;
  pedal_changes: number | null;
  note_duration_ratio_mean: number;
  performer_count: number;
}

export interface ReferenceProfile {
  piece_id: string;
  performer_count: number;
  bars: ReferenceBar[];
}

export interface ScoreContext {
  piece_id: string;
  composer: string;
  title: string;
  score: ScoreData;
  reference: ReferenceProfile | null;
  match_confidence: number;
}

export interface NoteAlignment {
  perf_onset: number;
  perf_pitch: number;
  perf_velocity: number;
  score_bar: number;
  score_beat: number;
  score_pitch: number;
  onset_deviation_ms: number;
}

export interface BarMap {
  chunk_index: number;
  bar_start: number;
  bar_end: number;
  alignments: NoteAlignment[];
  confidence: number;
  is_reanchored: boolean;
}

export interface FollowerState {
  last_known_bar: number | null;
}

export interface AlignChunkResult {
  bar_map: BarMap | null;
  state: FollowerState;
}

// --- Bar analysis types ---

export interface DimensionAnalysis {
  dimension: string;
  analysis: string;
  score_marking?: string;
  reference_comparison?: string;
}

export interface ChunkAnalysis {
  tier: number;
  /** Stringified bar range e.g. "4-7", or null for tier-3 */
  bar_range: string | null;
  dimensions: DimensionAnalysis[];
}

// --- Piece identify types ---

export interface NgramCandidate {
  piece_id: string;
  hit_count: number;
}

export interface RerankResult {
  piece_id: string;
  similarity: number;
}

export interface DtwConfirmResult {
  confirmed: boolean;
  cost: number;
  confidence: number;
}

export interface TextMatchResult {
  piece_id: string;
  confidence: number;
}

export interface CatalogEntry {
  piece_id: string;
  composer: string;
  title: string;
}

/** Inverted index: trigram key ("p1,p2,p3") -> Array<[piece_id, bar_number]> */
export type NgramIndex = Record<string, Array<[string, number]>>;

/** Per-piece 128-dim feature vectors */
export type RerankFeatures = Record<string, number[]>;

// ═══════════════════════════════════════════════════
// Lazy-loaded WASM module references
// ═══════════════════════════════════════════════════

// eslint-disable-next-line @typescript-eslint/consistent-type-imports
type ScoreAnalysisMod = typeof import("../wasm/score-analysis/pkg/score_analysis");
// eslint-disable-next-line @typescript-eslint/consistent-type-imports
type PieceIdentifyMod = typeof import("../wasm/piece-identify/pkg/piece_identify");

let scoreAnalysisModule: ScoreAnalysisMod | null = null;
let pieceIdentifyModule: PieceIdentifyMod | null = null;

function requireScoreAnalysis(): ScoreAnalysisMod {
  if (!scoreAnalysisModule) {
    throw new Error("score-analysis WASM not initialized");
  }
  return scoreAnalysisModule;
}

function requirePieceIdentify(): PieceIdentifyMod {
  if (!pieceIdentifyModule) {
    throw new Error("piece-identify WASM not initialized");
  }
  return pieceIdentifyModule;
}

// ═══════════════════════════════════════════════════
// Score Analysis wrappers
// ═══════════════════════════════════════════════════

/**
 * Select the top-1 teaching moment from a session's scored chunks.
 * Returns null if fewer than 2 chunks are provided.
 */
export function selectTeachingMoment(
  chunks: ScoredChunk[],
  baselines: StudentBaselines,
  recentObservations: RecentObservation[],
): TeachingMoment | null {
  return requireScoreAnalysis().select_teaching_moment(
    chunks,
    baselines,
    recentObservations,
  ) as TeachingMoment | null;
}

/**
 * Align a chunk of performance notes to a score using subsequence DTW.
 *
 * @param chunkIndex integer index of this chunk in the session
 * @param perfNotes array of performance notes from AMT
 * @param scoreBars array of ScoreBar from the loaded score JSON
 * @param followerState persistent state across chunks
 */
export function alignChunk(
  chunkIndex: number,
  perfNotes: PerfNote[],
  scoreBars: ScoreBar[],
  followerState: FollowerState,
): AlignChunkResult {
  return requireScoreAnalysis().align_chunk(
    chunkIndex,
    perfNotes,
    scoreBars,
    followerState,
  ) as AlignChunkResult;
}

/**
 * Tier 1 bar-aligned analysis: full analysis with score and reference comparison.
 *
 * @param barMap BarMap from a prior alignChunk call
 * @param perfNotes array of PerfNote
 * @param perfPedal array of PerfPedalEvent
 * @param scores 6-element array [dynamics, timing, pedaling, articulation, phrasing, interpretation]
 * @param scoreContext ScoreContext including score data and optional reference
 */
export function analyzeTier1(
  barMap: BarMap,
  perfNotes: PerfNote[],
  perfPedal: PerfPedalEvent[],
  scores: number[],
  scoreContext: ScoreContext,
): ChunkAnalysis {
  return requireScoreAnalysis().analyze_tier1(
    barMap,
    perfNotes,
    perfPedal,
    new Float64Array(scores),
    scoreContext,
  ) as ChunkAnalysis;
}

/**
 * Tier 2 absolute MIDI analysis: no score context required.
 *
 * @param perfNotes array of PerfNote
 * @param perfPedal array of PerfPedalEvent
 * @param scores 6-element array [dynamics, timing, pedaling, articulation, phrasing, interpretation]
 */
export function analyzeTier2(
  perfNotes: PerfNote[],
  perfPedal: PerfPedalEvent[],
  scores: number[],
): ChunkAnalysis {
  return requireScoreAnalysis().analyze_tier2(
    perfNotes,
    perfPedal,
    new Float64Array(scores),
  ) as ChunkAnalysis;
}

// ═══════════════════════════════════════════════════
// Piece Identify wrappers
// ═══════════════════════════════════════════════════

/**
 * Stage 1: N-gram recall.
 * Extracts pitch trigrams from notes, looks them up in the inverted index,
 * and returns top-10 candidates sorted by hit count.
 */
export function ngramRecall(notes: PerfNote[], index: NgramIndex): NgramCandidate[] {
  return requirePieceIdentify().ngram_recall(notes, index) as NgramCandidate[];
}

/**
 * Stage 2b: Rerank candidates by cosine similarity.
 * Returns top-2 results in descending similarity order.
 */
export function rerankCandidates(
  notes: PerfNote[],
  candidates: NgramCandidate[],
  features: RerankFeatures,
): RerankResult[] {
  return requirePieceIdentify().rerank_candidates(notes, candidates, features) as RerankResult[];
}

/**
 * Stage 3: DTW confirmation.
 * Runs subsequence DTW alignment to confirm or reject the top rerank candidate.
 *
 * @param perfNotes the identification window
 * @param scoreNotes flattened score notes { onset, pitch } from the candidate's score JSON
 * @param threshold normalized DTW cost below which the candidate is confirmed (use 0.3)
 */
export function dtwConfirm(
  perfNotes: PerfNote[],
  scoreNotes: Array<{ onset: number; pitch: number }>,
  threshold = 0.3,
): DtwConfirmResult {
  return requirePieceIdentify().dtw_confirm(perfNotes, scoreNotes, threshold) as DtwConfirmResult;
}

