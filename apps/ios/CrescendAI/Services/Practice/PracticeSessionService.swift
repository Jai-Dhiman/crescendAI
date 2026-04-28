import Foundation
import Sentry
import SwiftData

enum PracticeEvent {
    case sessionStarted(conversationId: String)
    case chunkUploaded(index: Int)
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
        }

        guard let event = try? JSONDecoder().decode(WSEvent.self, from: data) else { return }

        switch event.type {
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

                var req = URLRequest(url: APIEndpoints.practiceChunk(sessionId: sessionId, chunkIndex: localIndex))
                req.httpMethod = "POST"
                req.setValue("audio/aac", forHTTPHeaderField: "Content-Type")
                req.httpBody = audioData

                if let (_, response) = try? await session.data(for: req),
                   let httpResponse = response as? HTTPURLResponse,
                   (200..<300).contains(httpResponse.statusCode) {
                    _continuation.yield(.chunkUploaded(index: localIndex))
                }
                localIndex += 1
            }
        }
    }
}
