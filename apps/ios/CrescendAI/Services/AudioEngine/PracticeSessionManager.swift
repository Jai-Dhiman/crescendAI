import Foundation
import Observation
import Sentry
import SwiftData

/// Coordinates the full practice session lifecycle: audio session setup,
/// continuous capture, chunk production, inference, and SwiftData persistence.
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
    private(set) var currentSessionRecord: PracticeSessionRecord?

    var currentLevel: Float { captureEngine.currentLevel }

    private let modelContext: ModelContext
    private let audioSessionManager = AudioSessionManager()
    private let ringBuffer = RingBuffer(capacity: 24_000 * 300) // 5 minutes at 24kHz
    @ObservationIgnored private lazy var captureEngine = AudioCaptureEngine(ringBuffer: ringBuffer)
    private var chunkProducer: ChunkProducer?

    var chunkStream: AsyncStream<AudioChunk>? {
        chunkProducer?.chunkStream
    }
    private var interruptionTask: Task<Void, Never>?
    private var chunkObservationTask: Task<Void, Never>?

    init(modelContext: ModelContext) {
        self.modelContext = modelContext
    }

    func startSession(inferenceProvider: (any InferenceProvider)? = nil) async throws {
        guard state == .idle || state == .ended else { return }

        try await audioSessionManager.configure()

        ringBuffer.reset()
        try captureEngine.start()

        let session = PracticeSession()
        currentSession = session

        // Persist session record
        let record = PracticeSessionRecord(id: session.id, startedAt: session.startedAt)
        modelContext.insert(record)
        try modelContext.save()
        currentSessionRecord = record

        // Use factory default if no explicit provider
        let provider: any InferenceProvider
        if let inferenceProvider {
            provider = inferenceProvider
        } else {
            provider = await InferenceProviderFactory.create()
        }

        let producer = ChunkProducer(ringBuffer: ringBuffer, inferenceProvider: provider)
        producer.start(sessionId: session.id, startDate: session.startedAt)
        chunkProducer = producer

        state = .recording

        let startCrumb = Breadcrumb(level: .info, category: "practice")
        startCrumb.message = "Session started: \(session.id)"
        SentrySDK.addBreadcrumb(startCrumb)

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

        let endDate = Date()
        currentSession?.endedAt = endDate

        // Persist end timestamp
        currentSessionRecord?.endedAt = endDate
        do {
            try modelContext.save()
        } catch {
            print("[PracticeSessionManager] Failed to save session end: \(error)")
            SentrySDK.capture(error: error)
        }

        state = .ended

        let endCrumb = Breadcrumb(level: .info, category: "practice")
        endCrumb.message = "Session ended"
        SentrySDK.addBreadcrumb(endCrumb)
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
                persistChunk(chunk)
            }
        }
    }

    // MARK: - Persistence

    private func persistChunk(_ chunk: AudioChunk) {
        guard let sessionRecord = currentSessionRecord else { return }
        let record = chunk.toRecord(session: sessionRecord)
        modelContext.insert(record)
        do {
            try modelContext.save()
        } catch {
            print("[PracticeSessionManager] Failed to save chunk \(chunk.index): \(error)")
            SentrySDK.capture(error: error)
        }
    }
}
