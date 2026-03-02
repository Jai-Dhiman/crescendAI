# Slice 3: On-Device Inference Pipeline

See `docs/architecture.md` for the full system architecture.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Process audio chunks through the on-device Core ML MuQ model as they arrive during a practice session, accumulating 6-dimension scores in a local session buffer (SwiftData).

**Architecture:** The iOS app chunks audio via AVAudioEngine (Slice 2), feeds each chunk to the Core ML MuQ model on the Neural Engine, and accumulates results in SwiftData. By the time the student asks "how was that?", most or all chunks have been analyzed locally.

**Tech Stack:** Swift, Core ML, SwiftData, AVFoundation

---

## Context

Slice 2 produces 15-second audio chunks every 15 seconds via AVAudioEngine. This slice processes them on-device using the Core ML MuQ model, which outputs 6 dimension scores directly (no 19-to-6 mapping needed). The key requirement is running inference in the background as chunks arrive without blocking audio capture, and accumulating results in SwiftData so they are available instantly when the student asks "how was that?"

## Design

### On-Device Session Buffer

All session and chunk data is stored locally in SwiftData. No cloud endpoints are needed for inference.

```swift
@Model class PracticeSession {
    var id: UUID
    var startedAt: Date
    var endedAt: Date?
    var chunks: [ChunkResult]
    var synced: Bool
}

@Model class ChunkResult {
    var index: Int
    var startOffset: TimeInterval
    var duration: TimeInterval
    var dynamics: Double
    var timing: Double
    var pedaling: Double
    var articulation: Double
    var phrasing: Double
    var interpretation: Double
    var stopProbability: Double  // populated by Slice 4
    var inferenceStatus: String  // pending, completed, failed
    var processingTimeMs: Int
}
```

### Inference Pipeline Per Chunk

1. 15s PCM chunk extracted from ring buffer (Slice 2)
2. Feed to Core ML MuQ model
3. Receive 6-dimension scores
4. Store result in SwiftData: ChunkResult
5. Run STOP classifier (Slice 4 adds this)

### Concurrency and Pipelining

- Core ML inference takes ~1-2s per chunk on the Neural Engine (A16+)
- Chunks arrive every 15s, so inference is well within budget
- Use a dedicated inference queue (DispatchQueue) to avoid blocking the audio capture thread
- If a chunk's inference isn't done when the next chunk arrives, queue it (should never happen at 15s intervals)
- If inference fails on a chunk: mark ChunkResult as failed, continue capturing. Log the error for diagnostics.

### What This Slice Does NOT Include

- Teaching moment scoring (Slice 4 adds `stopProbability` to chunks)
- The "how was that?" response logic (Slice 6)
- User authentication (Slice 5)
- D1 sync (Slice 5 handles cloud backup)
- Audio capture setup (Slice 2 handles AVAudioEngine and chunking)

### Tasks

**Task 1: Define SwiftData models for PracticeSession and ChunkResult**

- Create the `@Model` classes as specified in On-Device Session Buffer above
- Set up SwiftData container and model context
- Test: create a session, add chunks, verify persistence across app launches

**Task 2: Implement Core ML model loading**

- Load the .mlpackage from the app bundle (or downloaded location)
- Configure for Neural Engine execution (MLComputeUnits.all)
- Handle model loading errors gracefully
- Test: model loads successfully, first inference completes

**Task 3: Implement inference pipeline**

- Accept a 15s PCM chunk from the ring buffer (Slice 2)
- Run Core ML inference on a dedicated DispatchQueue
- Parse the 6-dimension output scores
- Create and persist a ChunkResult in SwiftData
- Test: feed a known audio file, verify scores are reasonable and stored correctly

**Task 4: Integrate with audio capture (Slice 2)**

- Wire the chunking timer callback to the inference pipeline
- Ensure inference runs on a background queue, not the audio capture thread
- Verify chunks are queued if inference is still running (edge case)
- Test: run a full practice session, verify all chunks are captured and scored

**Task 5: Handle inference failures and edge cases**

- If Core ML returns an error: mark ChunkResult as failed, log error, continue
- Handle app backgrounding: ensure inference completes for in-flight chunks
- Handle memory pressure: respond to memory warnings appropriately
- Test: simulate failure conditions, verify no data loss

### Open Questions

1. Should the Core ML model be loaded once at session start and kept in memory, or loaded per-chunk? Keeping it loaded avoids ~0.5s load time but uses ~300MB+ continuously.
2. What happens if the Neural Engine is unavailable (e.g., other apps using it)? Core ML should fall back to GPU/CPU automatically, but latency may increase.
3. Should failed chunks be retried? If so, when -- immediately, or at session end?
