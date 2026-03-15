# Smart Audio Gating and Ripple Visualization

Audio activity detection to gate chunk uploads during practice sessions, plus a new organic ripple visualization replacing the current wave animation.

## Problem

The current listening mode has two issues:

1. **Wasted inference cost:** `MediaRecorder.start(15000)` fires 15s chunks unconditionally. If the student takes 2 minutes to settle in, or pauses between pieces, those silence chunks still upload to R2 and trigger HF inference (~$0.01-0.02 per chunk).
2. **Dishonest visualization:** `FlowingWaves` uses a 15% minimum amplitude floor, so waves animate even during total silence. The student cannot tell whether the system is hearing anything.

## Design

Four components, each independently testable:

### 1. Audio Activity Detection (`useAudioActivity` hook)

Computes RMS energy from the existing `AnalyserNode` (already wired in `usePracticeSession`) and applies debounced thresholding.

**Interface:**
```ts
function useAudioActivity(analyserNode: AnalyserNode | null): {
  isPlaying: boolean;
  energy: number; // 0-1 smoothed RMS energy
}
```

**Algorithm:**
- Every animation frame, compute RMS energy from `analyserNode.getByteFrequencyData()`
- Compare against a fixed threshold (initial: `0.04` on a 0-1 scale)
- **Onset (silence -> playing):** Energy exceeds threshold for ~150ms (3-4 consecutive frames at 60fps)
- **Offset (playing -> silence):** Energy drops below threshold for ~2 seconds
- Asymmetric debounce: fast onset (don't miss phrase starts), slow offset (survive natural pauses between phrases)

**Threshold:** Exposed as a constant `ENERGY_THRESHOLD = 0.04`. No auto-calibration in V1.

### 2. Smart Chunking (modifications to `usePracticeSession`)

Switch from timeslice mode to manual chunk control, gated by `isPlaying`.

**MediaRecorder change:** `recorder.start()` (continuous, no timeslice) instead of `recorder.start(15000)`. Chunks are cut manually via `recorder.requestData()`.

**State machine:**
```
WAITING ──(isPlaying=true)──> BUFFERING ──(15s elapsed)──> CUT_CHUNK ──> BUFFERING
   ^                                                                        |
   +──────────────────────(isPlaying=false for 2s)──────────────────────────+
```

- **WAITING:** MediaRecorder runs (mic stays hot for instant response) but `requestData()` is not called. No uploads, no cost. Audio accumulates in browser's internal buffer and is discarded on next `requestData()` call when transitioning to BUFFERING.
- **BUFFERING:** Piano detected. Start a 15s interval timer. Each tick calls `requestData()`, producing a chunk blob for upload.
- **Transition to WAITING:** When `isPlaying` goes false (2s offset debounce), call `requestData()` once to flush the partial chunk (could be 3-12s of audio), upload it, then stop the timer and return to WAITING.

**Why keep MediaRecorder running during silence:** Stopping and restarting causes audible glitches and loses ~100ms of audio on restart (browser-dependent). Running continuously with no `requestData()` calls is essentially free.

**Partial chunks on offset:** The final flush chunk may be shorter than 15s. The HF inference handler already accepts variable-length audio. The DO tracks chunk index continuously across WAITING/BUFFERING transitions.

**Discarding silence buffer:** When transitioning from WAITING to BUFFERING, call `requestData()` once and discard the resulting blob (this is the accumulated silence). Then start the 15s timer for real chunks.

### 3. Organic Ripple Visualization (`ResonanceRipples` component)

Drop-in replacement for `FlowingWaves`. Same props: `{ analyserNode, active }`. Same canvas element and CSS sizing.

**Visual behavior:**

- **Idle (isPlaying=false):** A single faint ripple emits from center every ~3 seconds. Slow expansion, low opacity (0.08-0.12), organic wobbly edge. Barely perceptible.
- **Playing (isPlaying=true):** Ripples emit in response to audio energy. Louder dynamics = more frequent ripples, higher opacity, larger max radius. Soft playing = fewer, gentler ripples. Peak energy: 3-4 visible ripples simultaneously.

**Ripple lifecycle:** Each ripple is an object `{ birthTime, maxRadius, wobblePhase1, wobblePhase2 }`. Born at center (radius ~0), expands outward at constant speed, opacity decreases linearly with radius. Removed when radius exceeds max. Managed as an array of active ripples.

**Organic edge:** Radius perturbed by angle:
```
r(theta) = baseRadius + noise * sin(theta * 3 + phase1) + noise * sin(theta * 5 + phase2)
```
Noise amplitude: ~3-5% of radius. Two sine terms at different frequencies prevent ellipse appearance.

**Color:** Sage green `rgba(122, 154, 130, opacity)`. Stroke width 1.5-2px. No fill, stroked rings only.

**Rendering:** Canvas 2D, `requestAnimationFrame` loop. Idle: throttle to ~15fps. Active: full 60fps.

**Integration:** `ListeningMode` and `RecordingBar` both switch from `FlowingWaves` to `ResonanceRipples`. `FlowingWaves` component is deleted.

### 4. Dev Console Logging

Structured tagged logging, gated by `import.meta.env.DEV`. Zero production overhead.

**Utility:**
```ts
function createLogger(tag: string): { log, warn, error }
```
Each method checks `import.meta.env.DEV` before calling `console.log/warn/error`. Prefix: `[tag]`.

**Log output by component:**

```
[AudioActivity] Energy: 0.02 (below threshold 0.04) -- idle
[AudioActivity] Energy: 0.12 (above threshold 0.04) -- onset debounce 2/4
[AudioActivity] PLAYING detected (sustained 150ms)
[AudioActivity] SILENCE detected (sustained 2.0s)

[ChunkGate] State: WAITING -> BUFFERING
[ChunkGate] Chunk timer started (15s)
[ChunkGate] Chunk #0 cut: 15.0s -- uploading to R2
[ChunkGate] Upload complete: chunk #0 -> r2Key=abc123
[ChunkGate] WS sent: chunk_ready #0
[ChunkGate] State: BUFFERING -> WAITING (silence offset)
[ChunkGate] Partial chunk #1 cut: 8.4s (flush)
[ChunkGate] Session totals: 3 chunks sent, 2m 14s silence skipped

[Ripples] Active ripples: 3, energy: 0.18, fps: 60
```

`[AudioActivity]` energy line throttled to 1 log per second to avoid console flooding.

## Files Changed

| File | Change |
|---|---|
| `apps/web/src/hooks/useAudioActivity.ts` | NEW -- audio activity detection hook |
| `apps/web/src/hooks/usePracticeSession.ts` | MODIFY -- integrate smart chunking gated by `isPlaying` |
| `apps/web/src/components/ResonanceRipples.tsx` | NEW -- organic ripple visualization |
| `apps/web/src/components/FlowingWaves.tsx` | DELETE -- replaced by ResonanceRipples |
| `apps/web/src/components/ListeningMode.tsx` | MODIFY -- swap FlowingWaves for ResonanceRipples, pass `isPlaying` |
| `apps/web/src/components/RecordingBar.tsx` | MODIFY -- swap FlowingWaves for ResonanceRipples |
| `apps/web/src/lib/logger.ts` | NEW -- dev-only tagged console logger |

## No Backend Changes

The DO (`session.rs`) already handles variable-length chunks and arbitrary chunk timing. The only change is that fewer chunks arrive during silence. No API contract changes.

## Testing

- **Manual:** Open listening mode, verify console logs show WAITING state. Play piano (or any audio), verify transition to BUFFERING and chunk uploads. Stop playing, verify 2s offset then return to WAITING. Verify no chunks upload during silence.
- **Visual:** Verify idle ripples are barely visible. Verify playing ripples respond to dynamics. Verify ripple edges look organic, not circular.
- **Cost:** Compare chunk counts between old (unconditional) and new (gated) approaches for a typical session with pauses.
