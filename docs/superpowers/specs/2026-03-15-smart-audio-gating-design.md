# Smart Audio Gating and Ripple Visualization

Audio activity detection to gate chunk uploads during practice sessions, plus a new organic ripple visualization replacing the current wave animation.

## Problem

The current listening mode has two issues:

1. **Wasted inference cost:** `MediaRecorder.start(15000)` fires 15s chunks unconditionally. If the student takes 2 minutes to settle in, or pauses between pieces, those silence chunks still upload to R2 and trigger HF inference (~$0.01-0.02 per chunk).
2. **Dishonest visualization:** `FlowingWaves` uses a 15% minimum amplitude floor, so waves animate even during total silence. The student cannot tell whether the system is hearing anything.

## Design

Four components, each independently testable:

### 1. Audio Activity Detection (`useAudioActivity` hook)

Computes spectral energy from the existing `AnalyserNode` (already wired in `usePracticeSession`) and applies debounced thresholding.

**Not a standalone hook.** `useAudioActivity` is called internally by `usePracticeSession`, which already creates and owns the `AnalyserNode` (via a ref). `usePracticeSession` exposes `isPlaying` and `energy` in its return type so the visualization component can consume them.

**Interface (internal):**
```ts
// Called inside usePracticeSession, not by the parent component
function useAudioActivity(analyserNodeRef: React.RefObject<AnalyserNode | null>): {
  isPlaying: boolean;
  energy: number; // 0-1 smoothed spectral energy
}
```

**Return type addition to `usePracticeSession`:**
```ts
// Added to UsePracticeSessionReturn
isPlaying: boolean;
energy: number;
```

**Algorithm:**
- Uses a `requestAnimationFrame` loop (cancelled on unmount and when analyserNode becomes null)
- Each frame, computes spectral energy from `analyserNode.getByteFrequencyData()` -- this sums frequency-bin magnitudes (FFT output), which correlates with loudness. Not true RMS (which would use `getFloatTimeDomainData()`), but sufficient for piano onset detection and consistent with how the existing visualizer reads audio data.
- Normalize to 0-1 range: `energy = sum / (binCount * 255)`
- Compare against a fixed threshold (initial: `0.04` on a 0-1 scale)
- **Onset (silence -> playing):** Energy exceeds threshold for ~150ms (3-4 consecutive frames at 60fps)
- **Offset (playing -> silence):** Energy drops below threshold for ~2 seconds
- Asymmetric debounce: fast onset (don't miss phrase starts), slow offset (survive natural pauses between phrases)

**Threshold:** Exposed as a constant `ENERGY_THRESHOLD = 0.04`. No auto-calibration in V1.

**Cleanup:** The `requestAnimationFrame` loop is cancelled on unmount via the hook's cleanup return. When `analyserNodeRef.current` is null, the loop skips computation but keeps running (the node appears mid-lifecycle when `start()` creates it).

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

**Handling the silence buffer on transition:** When transitioning from WAITING to BUFFERING, the MediaRecorder's internal buffer contains accumulated silence. Rather than trying to discard it (which requires async coordination with `ondataavailable` and introduces race conditions), simply let the first chunk include some leading silence. The HF model already handles variable audio content, and a few seconds of silence at the start of a chunk does not meaningfully affect inference quality. This keeps the `ondataavailable` handler simple -- every blob it receives gets uploaded, no discard flag needed.

### 3. Organic Ripple Visualization (`ResonanceRipples` component)

Drop-in replacement for `FlowingWaves`. Same canvas element and CSS sizing.

**Props:**
```ts
interface ResonanceRipplesProps {
  energy: number;      // 0-1 smoothed spectral energy from usePracticeSession
  isPlaying: boolean;  // piano activity state from usePracticeSession
  active: boolean;     // session is in recording state (controls rAF loop lifecycle)
}
```

`energy` and `isPlaying` come from `usePracticeSession`'s return type (which internally uses `useAudioActivity`). The component does not own an `AnalyserNode` -- it receives pre-computed values.

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

**Rendering:** Canvas 2D, `requestAnimationFrame` loop. Idle: throttle to ~15fps by checking elapsed time in the rAF callback and skipping renders when delta < 66ms. Active: full 60fps.

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

`[AudioActivity]` idle energy readout throttled to 1 log per second to avoid console flooding. Debounce transition logs (onset/offset events) are unthrottled since they are rare discrete events.

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

### 5. Stop Behavior with Zero Chunks

When `stop()` is called and no chunks were ever sent (student never triggered `isPlaying`, or threshold was too high), the current guard shows "Play for at least 15 seconds so I can listen." This message is misleading with smart gating -- the student may have played but the mic didn't pick it up, or the threshold needs tuning.

**Behavior:** On `stop()`, if `chunkIndexRef.current === 0`:
- Flush the MediaRecorder buffer as a final chunk and upload it (last resort -- may contain useful audio)
- If the flushed blob is too small (< 1s of audio, roughly < 10KB for Opus), skip the upload and show: "I couldn't hear any playing. Make sure your microphone is picking up the piano."
- If the flushed blob is large enough, upload it, send `end_session`, and proceed normally. The DO will process whatever audio it gets.

This ensures we never silently discard a whole session where the threshold was miscalibrated.

## No Backend Changes

The DO (`session.rs`) already handles variable-length chunks and arbitrary chunk timing. The only change is that fewer chunks arrive during silence. No API contract changes.

## Testing

- **Manual:** Open listening mode, verify console logs show WAITING state. Play piano (or any audio), verify transition to BUFFERING and chunk uploads. Stop playing, verify 2s offset then return to WAITING. Verify no chunks upload during silence.
- **Visual:** Verify idle ripples are barely visible. Verify playing ripples respond to dynamics. Verify ripple edges look organic, not circular.
- **Cost:** Compare chunk counts between old (unconditional) and new (gated) approaches for a typical session with pauses.
