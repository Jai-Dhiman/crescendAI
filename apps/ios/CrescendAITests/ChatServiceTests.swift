import XCTest
@testable import CrescendAI

@MainActor
final class ChatServiceTests: XCTestCase {

    override func tearDown() {
        MockURLProtocol.requestHandler = nil
        super.tearDown()
    }

    // MARK: - SSEParser unit tests

    func test_sseParser_parsesStartEvent() throws {
        let line = #"data: {"type":"start","conversationId":"c-001"}"#
        let event = SSEParser.parse(line: line)
        guard case .start(let cid) = event else {
            XCTFail("Expected .start, got \(String(describing: event))")
            return
        }
        XCTAssertEqual(cid, "c-001")
    }

    func test_sseParser_parsesDeltaEvent() {
        let line = #"data: {"type":"delta","text":"Hello world"}"#
        let event = SSEParser.parse(line: line)
        guard case .delta(let text) = event else {
            XCTFail("Expected .delta, got \(String(describing: event))")
            return
        }
        XCTAssertEqual(text, "Hello world")
    }

    func test_sseParser_parsesDoneEvent() {
        let event = SSEParser.parse(line: "data: [DONE]")
        guard case .done = event else {
            XCTFail("Expected .done, got \(String(describing: event))")
            return
        }
    }

    func test_sseParser_parsesToolResultWithComponents() throws {
        let line = #"data: {"type":"tool_result","componentsJson":[{"type":"keyboard_guide","config":{"title":"t","description":"d","hands":"both"}}]}"#
        let event = SSEParser.parse(line: line)
        guard case .toolResult(let artifacts) = event else {
            XCTFail("Expected .toolResult, got \(String(describing: event))")
            return
        }
        XCTAssertEqual(artifacts.count, 1)
        guard case .keyboardGuide(let config) = artifacts[0] else {
            XCTFail("Expected .keyboardGuide")
            return
        }
        XCTAssertEqual(config.title, "t")
    }

    func test_sseParser_returnsNilForNonDataLine() {
        let event = SSEParser.parse(line: ": keep-alive")
        XCTAssertNil(event)
    }

    // MARK: - ChatService integration

    func test_send_emitsDeltaEventFromSSEStream() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: config)

        let sseBody = """
        data: {"type":"start","conversationId":"c-001"}

        data: {"type":"delta","text":"Nice playing"}

        data: [DONE]

        """
        MockURLProtocol.requestHandler = { request in
            XCTAssertEqual(request.url?.path, "/api/chat")
            let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: ["Content-Type": "text/event-stream"])!
            return (response, Data(sseBody.utf8))
        }

        let service = ChatService(session: session)
        var events: [ChatEvent] = []

        for await event in service.send(message: "How am I doing?", conversationId: nil) {
            events.append(event)
        }

        XCTAssertTrue(events.contains(where: {
            if case .delta(let t) = $0 { return t == "Nice playing" }
            return false
        }))
        XCTAssertTrue(events.contains(where: { if case .done = $0 { return true }; return false }))
    }

    func test_send_emitsErrorEventOn4xx() async throws {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        let session = URLSession(configuration: config)

        MockURLProtocol.requestHandler = { request in
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: 401,
                httpVersion: nil,
                headerFields: nil
            )!
            return (response, Data())
        }

        let service = ChatService(session: session)
        var events: [ChatEvent] = []

        for await event in service.send(message: "test", conversationId: nil) {
            events.append(event)
        }

        XCTAssertTrue(events.contains(where: {
            if case .error = $0 { return true }
            return false
        }))
    }
}
