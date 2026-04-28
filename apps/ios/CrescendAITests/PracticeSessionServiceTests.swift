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
}
