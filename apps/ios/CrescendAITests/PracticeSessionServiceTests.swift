import XCTest
import SwiftData
@testable import CrescendAI

@MainActor
final class PracticeSessionServiceTests: XCTestCase {

    override func tearDown() async throws {
        MockURLProtocol.requestHandler = nil
        try await super.tearDown()
    }

    // MARK: - RMS silence gate (static method, no network)

    func test_rms_silentSignalReturnsZero() {
        let samples = [Float](repeating: 0, count: 1000)
        XCTAssertEqual(ChunkProducer.rms(samples: samples), 0)
    }

    func test_rms_fullScaleSineReturnsNearPointSeven() {
        // RMS of a full-scale sine = 1/sqrt(2) ≈ 0.7071
        let N = 1024
        let samples = (0..<N).map { Float(sin(2 * .pi * Double($0) / Double(N))) }
        let result = ChunkProducer.rms(samples: samples)
        XCTAssertEqual(result, 0.7071, accuracy: 0.01)
    }

    // MARK: - PracticeSessionService.start() emits sessionStarted

    func test_start_emitsSessionStartedWithConversationId() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: config)

        MockURLProtocol.requestHandler = { request in
            if request.url?.path == "/api/practice/start" {
                let body = ["sessionId": "session-abc", "conversationId": "conv-xyz"]
                let data = try JSONEncoder().encode(body)
                let response = HTTPURLResponse(url: request.url!, statusCode: 201, httpVersion: nil, headerFields: nil)!
                return (response, data)
            }
            return (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!, Data())
        }

        let schema = Schema([Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self, CheckInRecord.self, ConversationRecord.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        let context = ModelContext(container)

        let service = PracticeSessionService(session: session)
        service.configure(modelContext: context)
        var receivedEvents: [PracticeEvent] = []

        let eventTask = Task {
            for await event in service.eventStream {
                receivedEvents.append(event)
                if case .sessionStarted = event { break }
            }
        }

        try await service.start()

        await eventTask.value

        XCTAssertTrue(receivedEvents.contains(where: {
            if case .sessionStarted(let cid) = $0 { return cid == "conv-xyz" }
            return false
        }))
        XCTAssertEqual(service.conversationId, "conv-xyz")

        Task { await service.stop() }
    }

    func test_start_throwsNetworkErrorWhenServerUnavailable() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: config)

        MockURLProtocol.requestHandler = { _ in
            throw URLError(.notConnectedToInternet)
        }

        let schema = Schema([Student.self, PracticeSessionRecord.self, ChunkResultRecord.self, ObservationRecord.self, CheckInRecord.self, ConversationRecord.self])
        let container = try ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
        let context = ModelContext(container)

        let service = PracticeSessionService(session: session)
        service.configure(modelContext: context)

        do {
            try await service.start()
            XCTFail("Expected throw")
        } catch APIError.networkError {
            // expected
        }
    }

    // MARK: - uploadChunk retry / backoff / failure surfacing

    /// Thread-safe request counter (MockURLProtocol's handler runs off the main actor).
    private final class RequestCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        func increment() -> Int { lock.lock(); defer { lock.unlock() }; value += 1; return value }
        var count: Int { lock.lock(); defer { lock.unlock() }; return value }
    }

    private func makeMockSession() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        return URLSession(configuration: config)
    }

    func test_uploadChunk_returnsTrueOnSuccess() async {
        let session = makeMockSession()
        let counter = RequestCounter()
        MockURLProtocol.requestHandler = { request in
            _ = counter.increment()
            return (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!, Data())
        }

        let service = PracticeSessionService(session: session)
        let result = await service.uploadChunk(Data([0x1]), sessionId: "s1", chunkIndex: 0, maxAttempts: 3, baseDelayMs: 1)

        XCTAssertTrue(result)
        XCTAssertEqual(counter.count, 1)
    }

    func test_uploadChunk_retriesOn5xxThenSucceeds() async {
        let session = makeMockSession()
        let counter = RequestCounter()
        MockURLProtocol.requestHandler = { request in
            let n = counter.increment()
            let status = n == 1 ? 503 : 200
            return (HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!, Data())
        }

        let service = PracticeSessionService(session: session)
        let result = await service.uploadChunk(Data([0x1]), sessionId: "s1", chunkIndex: 0, maxAttempts: 3, baseDelayMs: 1)

        XCTAssertTrue(result)
        XCTAssertEqual(counter.count, 2, "Should retry once after a 5xx, then succeed")
    }

    func test_uploadChunk_failsFastOnClientError() async {
        let session = makeMockSession()
        let counter = RequestCounter()
        MockURLProtocol.requestHandler = { request in
            _ = counter.increment()
            return (HTTPURLResponse(url: request.url!, statusCode: 400, httpVersion: nil, headerFields: nil)!, Data())
        }

        let service = PracticeSessionService(session: session)
        let result = await service.uploadChunk(Data([0x1]), sessionId: "s1", chunkIndex: 0, maxAttempts: 3, baseDelayMs: 1)

        XCTAssertFalse(result)
        XCTAssertEqual(counter.count, 1, "A 4xx is non-retryable; must not consume further attempts")
    }

    func test_uploadChunk_failsAfterExhaustingRetriesOnNetworkError() async {
        let session = makeMockSession()
        let counter = RequestCounter()
        MockURLProtocol.requestHandler = { _ in
            _ = counter.increment()
            throw URLError(.notConnectedToInternet)
        }

        let service = PracticeSessionService(session: session)
        let result = await service.uploadChunk(Data([0x1]), sessionId: "s1", chunkIndex: 0, maxAttempts: 3, baseDelayMs: 1)

        XCTAssertFalse(result)
        XCTAssertEqual(counter.count, 3, "Network errors are retryable; should attempt exactly maxAttempts times")
    }
}
