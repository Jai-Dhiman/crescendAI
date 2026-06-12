import Foundation
import Sentry
import SwiftData

enum PracticeEvent {
    case sessionStarted(conversationId: String)
    case chunkUploaded(index: Int)
    case chunkUploadFailed(index: Int)
    case observation(text: String, dimension: String?, artifacts: [ArtifactConfig])
    case synthesis(text: String, artifacts: [ArtifactConfig])
    case reconnecting(attempt: Int)
    case error(String)
    case sessionEnded
}

enum PracticeSessionError: LocalizedError {
    case notConfigured

    var errorDescription: String? {
        "PracticeSessionService not configured — call configure(modelContext:) first"
    }
}

/// The 6-dimension scores carried by the server's `chunk_processed` WebSocket message.
/// These are the authoritative MuQ scores; the iOS client persists them rather than
/// computing anything on-device.
struct ChunkScores: Decodable, Equatable {
    let dynamics: Double
    let timing: Double
    let pedaling: Double
    let articulation: Double
    let phrasing: Double
    let interpretation: Double
}

@MainActor
protocol PracticeSessionServiceProtocol: AnyObject {
    var eventStream: AsyncStream<PracticeEvent> { get }
    var state: PracticeSessionService.State { get }
    var currentLevel: Float { get }
    var elapsedSeconds: TimeInterval { get }
    var conversationId: String? { get }
    func start() async throws
    func stop() async
    func askForFeedback() async
}

@MainActor
@Observable
final class PracticeSessionService: PracticeSessionServiceProtocol {

    enum State {
        case idle
        case connecting
        case recording
        case stopping
    }

    private(set) var state: State = .idle
    private(set) var currentLevel: Float = 0
    private(set) var elapsedSeconds: TimeInterval = 0
    private(set) var conversationId: String?

    var eventStream: AsyncStream<PracticeEvent> { _stream }

    private var _stream: AsyncStream<PracticeEvent>
    private var _continuation: AsyncStream<PracticeEvent>.Continuation

    private let session: URLSession
    private var sessionManager: PracticeSessionManager?
    private var webSocketTask: URLSessionWebSocketTask?
    private var sessionId: String?
    private var timerTask: Task<Void, Never>?
    private var uploadTask: Task<Void, Never>?
    private var wsTask: Task<Void, Never>?
    private var modelContext: ModelContext?
    private var synthesisReceived = false

    private struct PracticeStartResponse: Decodable {
        let sessionId: String
        let conversationId: String
    }

    init(session: URLSession = .shared) {
        self.session = session
        var cont: AsyncStream<PracticeEvent>.Continuation!
        _stream = AsyncStream(bufferingPolicy: .bufferingNewest(16)) { cont = $0 }
        _continuation = cont
    }

    func configure(modelContext: ModelContext) {
        self.modelContext = modelContext
    }

    func start() async throws {
        guard let mc = modelContext else {
            throw PracticeSessionError.notConfigured
        }
        state = .connecting

        // 1. Create server session
        var request = URLRequest(url: APIEndpoints.practiceStart())
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode([String: String]())

        let client = APIClient(session: session)
        let startResp: PracticeStartResponse = try await client.perform(request)
        self.sessionId = startResp.sessionId
        self.conversationId = startResp.conversationId

        // 2. Start audio capture (non-fatal if audio session unavailable)
        let manager = PracticeSessionManager(modelContext: mc)
        self.sessionManager = manager
        do {
            try await manager.startSession(inferenceProvider: nil)
        } catch {
            // Audio capture unavailable (e.g. simulator without mic access)
            // Continue with WebSocket/upload path; no chunks will be produced
            self.sessionManager = nil
        }

        // 3. Connect WebSocket
        connectWebSocket(sessionId: startResp.sessionId, conversationId: startResp.conversationId)

        // 4. Start chunk upload loop
        startChunkUploader(sessionId: startResp.sessionId)

        // 5. Start elapsed timer
        timerTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(1))
                elapsedSeconds += 1
            }
        }

        state = .recording
        _continuation.yield(.sessionStarted(conversationId: startResp.conversationId))
    }

    func stop() async {
        state = .stopping
        timerTask?.cancel()
        timerTask = nil
        uploadTask?.cancel()
        uploadTask = nil

        let hadAudioCapture = sessionManager != nil
        // Capture the session record before releasing the manager; baselines are
        // updated below once the WebSocket has delivered the remaining scores.
        let sessionRecord = sessionManager?.currentSessionRecord
        await sessionManager?.endSession()
        sessionManager = nil

        // Wait up to 30s for synthesis only when audio capture was active
        // (synthesis is triggered by the server after session audio ends)
        if hadAudioCapture && !synthesisReceived {
            let deadline = Date().addingTimeInterval(30)
            while !synthesisReceived && Date() < deadline {
                do {
                    try await Task.sleep(for: .milliseconds(200))
                } catch {
                    break
                }
            }
        }

        wsTask?.cancel()
        wsTask = nil
        webSocketTask?.cancel(with: .normalClosure, reason: nil)
        webSocketTask = nil

        // Fold this session's real per-chunk scores into the student's baselines.
        if let mc = modelContext, let sessionRecord {
            updateBaselinesOnSessionEnd(sessionRecord: sessionRecord, context: mc)
        }

        state = .idle
        _continuation.yield(.sessionEnded)
    }

    func askForFeedback() async {
        // Feedback is delivered via the WebSocket observation stream
    }

    // MARK: - WebSocket

    private func connectWebSocket(sessionId: String, conversationId: String, attempt: Int = 0) {
        wsTask?.cancel()
        wsTask = nil
        webSocketTask?.cancel(with: .normalClosure, reason: nil)

        let url = APIEndpoints.practiceWs(sessionId: sessionId, conversationId: conversationId)
        let task = session.webSocketTask(with: url)
        webSocketTask = task
        task.resume()

        wsTask = Task {
            do {
                try await receiveWebSocketMessages(task: task)
            } catch {
                guard !Task.isCancelled, state == .recording else { return }
                let maxAttempts = 5
                guard attempt < maxAttempts else {
                    _continuation.yield(.error("WebSocket disconnected after \(maxAttempts) attempts"))
                    return
                }
                let delay = min(pow(2.0, Double(attempt)), 30.0)
                _continuation.yield(.reconnecting(attempt: attempt + 1))
                try? await Task.sleep(for: .seconds(delay))
                guard !Task.isCancelled, state == .recording else { return }
                connectWebSocket(sessionId: sessionId, conversationId: conversationId, attempt: attempt + 1)
            }
        }
    }

    private func receiveWebSocketMessages(task: URLSessionWebSocketTask) async throws {
        while !Task.isCancelled {
            let message = try await task.receive()
            guard case .string(let text) = message,
                  let data = text.data(using: .utf8) else { continue }
            handleWebSocketData(data)
        }
    }

    private func handleWebSocketData(_ data: Data) {
        struct WSEvent: Decodable {
            let type: String
            let text: String?
            let dimension: String?
            let components: [ArtifactConfig]?
            let index: Int?
            let scores: ChunkScores?
        }

        guard let event = try? JSONDecoder().decode(WSEvent.self, from: data) else { return }

        switch event.type {
        case "chunk_processed":
            // Authoritative 6-dim scores computed server-side (MuQ). Persist them
            // onto this chunk's record so baselines/profile reflect real scores.
            if let index = event.index, let scores = event.scores,
               let mc = modelContext, let sessionRecord = sessionManager?.currentSessionRecord {
                Self.applyChunkScores(index: index, scores: scores, to: sessionRecord, in: mc)
            }
        case "observation":
            _continuation.yield(.observation(
                text: event.text ?? "",
                dimension: event.dimension,
                artifacts: event.components ?? []
            ))
        case "synthesis":
            synthesisReceived = true
            _continuation.yield(.synthesis(
                text: event.text ?? "",
                artifacts: event.components ?? []
            ))
            if let cid = conversationId, let mc = modelContext {
                let record = ConversationRecord(
                    conversationId: cid,
                    sessionId: UUID(uuidString: sessionId ?? "") ?? UUID(),
                    startedAt: Date(),
                    synthesisText: event.text
                )
                mc.insert(record)
                do {
                    try mc.save()
                } catch {
                    SentrySDK.capture(error: error)
                    _continuation.yield(.error("Failed to persist session record: \(error.localizedDescription)"))
                }
            }
        case "error":
            _continuation.yield(.error(event.text ?? "Unknown error"))
        default:
            break
        }
    }

    // MARK: - Chunk Upload

    private func startChunkUploader(sessionId: String) {
        guard let manager = sessionManager else { return }
        var localIndex = 0
        uploadTask = Task {
            guard let stream = manager.chunkStream else { return }
            for await chunk in stream {
                guard !Task.isCancelled else { break }
                guard let fileURL = chunk.localFileURL,
                      let audioData = try? Data(contentsOf: fileURL) else { continue }

                let uploaded = await uploadChunk(audioData, sessionId: sessionId, chunkIndex: localIndex)
                if uploaded {
                    _continuation.yield(.chunkUploaded(index: localIndex))
                } else {
                    _continuation.yield(.chunkUploadFailed(index: localIndex))
                }
                // Always advance the index: a failed chunk leaves a gap in the
                // server-side sequence, never a misalignment of later chunks.
                localIndex += 1
            }
        }
    }

    /// Upload a single audio chunk with bounded retry + exponential backoff.
    /// Returns true on a 2xx response. Retries network/transport errors, 5xx, and 429;
    /// fails fast on other 4xx (retrying a client error won't help). Every failed
    /// attempt is reported to Sentry. Returns false once attempts are exhausted.
    func uploadChunk(
        _ audioData: Data,
        sessionId: String,
        chunkIndex: Int,
        maxAttempts: Int = 3,
        baseDelayMs: Int = 400
    ) async -> Bool {
        var attempt = 0
        while attempt < maxAttempts {
            attempt += 1

            var req = URLRequest(url: APIEndpoints.practiceChunk(sessionId: sessionId, chunkIndex: chunkIndex))
            req.httpMethod = "POST"
            req.setValue("audio/aac", forHTTPHeaderField: "Content-Type")
            req.httpBody = audioData

            do {
                let (_, response) = try await session.data(for: req)
                guard let http = response as? HTTPURLResponse else {
                    // No HTTP response — treat as a retryable transport anomaly.
                    captureUploadFailure(chunkIndex: chunkIndex, statusCode: nil, error: nil, attempt: attempt)
                    if attempt < maxAttempts { await backoff(attempt: attempt, baseDelayMs: baseDelayMs) }
                    continue
                }
                if (200..<300).contains(http.statusCode) {
                    return true
                }
                captureUploadFailure(chunkIndex: chunkIndex, statusCode: http.statusCode, error: nil, attempt: attempt)
                let retryable = http.statusCode >= 500 || http.statusCode == 429
                if !retryable {
                    return false
                }
            } catch {
                // Network/transport failure — retryable.
                captureUploadFailure(chunkIndex: chunkIndex, statusCode: nil, error: error, attempt: attempt)
            }

            if attempt < maxAttempts {
                await backoff(attempt: attempt, baseDelayMs: baseDelayMs)
            }
        }
        return false
    }

    private func backoff(attempt: Int, baseDelayMs: Int) async {
        // Exponential: base * 2^(attempt-1).
        let delayMs = baseDelayMs << (attempt - 1)
        try? await Task.sleep(for: .milliseconds(delayMs))
    }

    private func captureUploadFailure(chunkIndex: Int, statusCode: Int?, error: Error?, attempt: Int) {
        let crumb = Breadcrumb(level: .error, category: "practice.upload")
        crumb.message = "Chunk \(chunkIndex) upload failed (attempt \(attempt), status: \(statusCode.map(String.init) ?? "n/a"))"
        SentrySDK.addBreadcrumb(crumb)

        if let error {
            SentrySDK.capture(error: error)
        } else {
            let err = NSError(
                domain: "PracticeChunkUpload",
                code: statusCode ?? -1,
                userInfo: [NSLocalizedDescriptionKey: "Chunk \(chunkIndex) upload failed with status \(statusCode ?? -1)"]
            )
            SentrySDK.capture(error: err)
        }
    }

    // MARK: - Real-score persistence

    /// Apply the server's authoritative scores onto the chunk record for `index`.
    /// Updates the existing (pending) record if present; otherwise inserts a completed
    /// one (covers the rare case where the score message arrives before the local
    /// chunk row was persisted). Static + parameterized so it is unit-testable without
    /// a live session.
    static func applyChunkScores(
        index: Int,
        scores: ChunkScores,
        to session: PracticeSessionRecord,
        in context: ModelContext
    ) {
        if let existing = session.chunks.first(where: { $0.index == index }) {
            existing.dynamics = scores.dynamics
            existing.timing = scores.timing
            existing.pedaling = scores.pedaling
            existing.articulation = scores.articulation
            existing.phrasing = scores.phrasing
            existing.interpretation = scores.interpretation
            existing.inferenceStatus = .completed
        } else {
            let record = ChunkResultRecord(
                index: index,
                startOffset: 0,
                duration: 0,
                dynamics: scores.dynamics,
                timing: scores.timing,
                pedaling: scores.pedaling,
                articulation: scores.articulation,
                phrasing: scores.phrasing,
                interpretation: scores.interpretation,
                inferenceStatus: .completed,
                session: session
            )
            context.insert(record)
        }
        do {
            try context.save()
        } catch {
            SentrySDK.capture(error: error)
        }
    }

    /// After a session ends, fold its real per-chunk scores into the student's EMA
    /// baselines. No-op if there is no local student or no completed chunks.
    private func updateBaselinesOnSessionEnd(sessionRecord: PracticeSessionRecord, context: ModelContext) {
        guard let student = (try? context.fetch(FetchDescriptor<Student>()))?.first else { return }
        if sessionRecord.student == nil {
            sessionRecord.student = student
        }
        StudentModelService.updateBaselines(student: student, session: sessionRecord)
        do {
            try context.save()
        } catch {
            SentrySDK.capture(error: error)
        }
    }
}
