import Foundation
import Observation

/// Coordinates the full practice session lifecycle: audio session setup,
/// continuous capture, chunk production, and inference.
@MainActor
@Observable
final class PracticeSessionManager {
    enum SessionState: Sendable {
        case idle
        case recording
        case paused
        case ended
    }

    private(set) var state: SessionState = .idle
    private(set) var currentSession: PracticeSession?

    var currentLevel: Float { captureEngine.currentLevel }

    private let audioSessionManager = AudioSessionManager()
    private let ringBuffer = RingBuffer(capacity: 24_000 * 300) // 5 minutes at 24kHz
    @ObservationIgnored private lazy var captureEngine = AudioCaptureEngine(ringBuffer: ringBuffer)
    private var chunkProducer: ChunkProducer?
    private var interruptionTask: Task<Void, Never>?
    private var chunkObservationTask: Task<Void, Never>?

    func startSession(inferenceProvider: (any InferenceProvider)? = nil) async throws {
        guard state == .idle || state == .ended else { return }

        try await audioSessionManager.configure()

        ringBuffer.reset()
        try captureEngine.start()

        let session = PracticeSession()
        currentSession = session

        let producer = ChunkProducer(ringBuffer: ringBuffer, inferenceProvider: inferenceProvider)
        producer.start(sessionId: session.id, startDate: session.startedAt)
        chunkProducer = producer

        state = .recording

        observeInterruptions()
        observeChunks()
    }

    func endSession() async {
        chunkProducer?.stop()
        captureEngine.stop()
        await audioSessionManager.deactivate()

        interruptionTask?.cancel()
        interruptionTask = nil
        chunkObservationTask?.cancel()
        chunkObservationTask = nil

        currentSession?.endedAt = Date()
        state = .ended
    }

    // MARK: - Interruption Handling

    private func observeInterruptions() {
        interruptionTask?.cancel()
        let sessionManager = audioSessionManager
        interruptionTask = Task {
            for await sessionState in await sessionManager.stateStream {
                guard !Task.isCancelled else { break }
                handleInterruptionState(sessionState)
            }
        }
    }

    private func handleInterruptionState(_ sessionState: AudioSessionManager.State) {
        switch sessionState {
        case .interrupted:
            captureEngine.stop()
            state = .paused
        case .active:
            if state == .paused {
                try? captureEngine.start()
                state = .recording
            }
        case .inactive:
            break
        }
    }

    // MARK: - Chunk Observation

    private func observeChunks() {
        guard let producer = chunkProducer else { return }
        chunkObservationTask?.cancel()
        chunkObservationTask = Task {
            for await chunk in producer.chunkStream {
                guard !Task.isCancelled else { break }
                currentSession?.chunks.append(chunk)
            }
        }
    }
}
