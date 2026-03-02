# Slice 2: Continuous Audio Capture (iOS)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the iOS audio capture layer that records continuously during a practice session and produces 15-30 second audio chunks ready for MuQ inference.

**Architecture:** Native iOS app using AVAudioEngine for real-time audio capture with a rolling buffer. Audio is chunked on a timer and uploaded to the backend. Background audio mode keeps recording when screen is off.

**Tech Stack:** Swift, SwiftUI, AVAudioEngine, AVFoundation, URLSession

---

## Context

The practice companion needs continuous audio capture from the phone mic. Mobile Safari kills background audio; a native iOS app with the `audio` background mode capability can record reliably with the screen off. The existing `apps/ios/CrescendAI/` stub provides the starting point.

## Design

### Audio Capture

**AVAudioEngine** (not AVAudioRecorder) because:
- Real-time access to audio buffers (needed for silence detection later)
- Can tap the input node to get PCM samples as they arrive
- More control over format, sample rate, buffer size

**Recording format:**
- Sample rate: 24kHz (MuQ's native rate -- avoid resampling later)
- Channels: mono (piano recording, stereo unnecessary)
- Format: PCM Float32 in memory, encoded to AAC or Opus for upload

### Chunking Strategy

**Rolling buffer approach:**
- AVAudioEngine input tap writes PCM samples to a circular buffer in memory
- Buffer holds the last 5 minutes of audio (24000 samples/sec * 300 sec * 4 bytes = ~29 MB -- fine for modern iPhones)
- A background timer fires every 15 seconds
- On timer fire: extract the last 15s of audio from the buffer, encode to AAC, and enqueue for upload
- Chunks overlap by 0 seconds (adjacent, non-overlapping) for simplicity

**When "how was that?" is triggered:**
- The system already has chunks covering the full session, each with inference results
- No additional recording action needed -- the analysis buffer is already populated

### Background Recording

**iOS background mode:**
- Add `audio` to `UIBackgroundModes` in Info.plist
- Start an AVAudioSession with `.record` category before going to background
- AVAudioEngine continues running when app is backgrounded
- Display a persistent notification: "CrescendAI is listening to your practice"
- Handle interruptions (phone call, Siri) gracefully: pause recording, resume after

### Session Management

**Session lifecycle:**
1. User taps "Start Practice" -- AVAudioEngine starts, session timer begins
2. Recording runs continuously. Chunks are produced every 15s.
3. User can lock screen or switch apps -- recording continues
4. User taps "End Practice" or app detects prolonged silence (>5 min) -- session ends
5. Session metadata saved: start time, end time, number of chunks, total duration

### Upload Pipeline

**Per chunk:**
1. Encode PCM to AAC (using AVAudioConverter)
2. Upload to backend: `POST /api/session/{session_id}/chunk` with multipart audio
3. Backend stores in R2 and forwards to MuQ inference
4. Inference result returned and cached locally on device
5. If upload fails (no network): queue locally, retry when connection returns

**Bandwidth estimate:**
- AAC at 24kHz mono, ~64kbps = ~120KB per 15s chunk
- 1 hour session = 240 chunks = ~29 MB total upload
- Manageable on WiFi. On cellular, warn user about data usage.

### Data Model (on-device)

```swift
struct PracticeSession {
    let id: UUID
    let startedAt: Date
    var endedAt: Date?
    var chunks: [AudioChunk]
}

struct AudioChunk {
    let id: UUID
    let sessionId: UUID
    let index: Int
    let startOffset: TimeInterval  // seconds from session start
    let duration: TimeInterval     // 15s typically
    let localFileURL: URL          // AAC file on device
    var uploadStatus: UploadStatus // .pending, .uploading, .uploaded, .failed
    var inferenceResult: InferenceResult?
}

struct InferenceResult {
    let dimensions: [String: Float]  // 6 dimensions
    let teachingMomentScore: Float?  // from teaching moment model (slice 4)
    let processingTimeMs: Int
}
```

### What This Slice Does NOT Include

- Teaching moment scoring (Slice 4)
- Student model or persistence across sessions (Slice 5)
- The "how was that?" response logic (Slice 6)
- The UI beyond basic start/stop and status (Slice 9)
- Piece identification

### Tasks

**Task 1: Set up iOS project structure**
- Clean up existing `apps/ios/CrescendAI/` stub
- Add AVFoundation framework
- Configure audio background mode in Info.plist
- Set up AVAudioSession with `.record` category

**Task 2: Implement AVAudioEngine capture**
- Install input tap on AVAudioEngine
- Write PCM samples to a circular buffer (RingBuffer class)
- Handle audio session interruptions (phone calls, Siri)
- Test: verify audio is captured, buffer fills, no memory leaks

**Task 3: Implement chunking**
- Background timer fires every 15s
- Extract PCM from ring buffer, encode to AAC via AVAudioConverter
- Save chunk to temporary file on device
- Test: verify chunks are correct duration, no gaps, no corruption

**Task 4: Implement upload pipeline**
- POST chunk to backend endpoint (define API contract)
- Handle upload failures with local queue and retry
- Test: verify chunks arrive at backend, retry works on network loss

**Task 5: Session lifecycle**
- Start/stop session controls
- Background recording (screen off)
- Silence detection for auto-end (optional, can defer)
- Session metadata persistence (Core Data or simple JSON file)

**Task 6: Backend chunk ingestion endpoint**
- New endpoint: `POST /api/session/{session_id}/chunk`
- Receive audio, store in R2
- Forward to MuQ inference endpoint
- Return inference result
- Store session + chunk metadata in D1

### Open Questions

1. Should chunks be uploaded immediately (streaming) or batched? Immediate gives real-time analysis but uses more network. Batched is simpler.
2. Ring buffer size: 5 minutes enough? Could a student play for longer without asking "how was that?"
3. AAC encoding quality: 64kbps enough for MuQ, or does it need higher bitrate?
