import XCTest
@testable import CrescendAI

final class ScoreHostBridgeTests: XCTestCase {

    func test_decodeReadyEvent() throws {
        let body: [String: Any] = ["type": "ready", "payload": [:] as [String: Any]]
        let event = try ScoreHostEvent(from: body)
        guard case .ready = event else {
            XCTFail("Expected .ready, got \(event)")
            return
        }
    }

    func test_decodeRenderedEvent() throws {
        let body: [String: Any] = ["type": "rendered", "payload": ["noteCount": 42] as [String: Any]]
        let event = try ScoreHostEvent(from: body)
        guard case .rendered(let noteCount) = event else {
            XCTFail("Expected .rendered, got \(event)")
            return
        }
        XCTAssertEqual(noteCount, 42)
    }

    func test_decodePlaybackEvent() throws {
        let body: [String: Any] = ["type": "playback", "payload": ["state": "playing"] as [String: Any]]
        let event = try ScoreHostEvent(from: body)
        guard case .playback(let state) = event else {
            XCTFail("Expected .playback, got \(event)")
            return
        }
        XCTAssertEqual(state, "playing")
    }

    func test_decodeActiveBarEvent() throws {
        let body: [String: Any] = ["type": "activeBar", "payload": ["bar": 7] as [String: Any]]
        let event = try ScoreHostEvent(from: body)
        guard case .activeBar(let bar) = event else {
            XCTFail("Expected .activeBar, got \(event)")
            return
        }
        XCTAssertEqual(bar, 7)
    }

    func test_decodeErrorEvent() throws {
        let body: [String: Any] = ["type": "error", "payload": ["reason": "score load failed"] as [String: Any]]
        let event = try ScoreHostEvent(from: body)
        guard case .error(let reason) = event else {
            XCTFail("Expected .error, got \(event)")
            return
        }
        XCTAssertEqual(reason, "score load failed")
    }

    func test_decodeUnknownTypeThrows() {
        let body: [String: Any] = ["type": "future_event", "payload": [:] as [String: Any]]
        XCTAssertThrowsError(try ScoreHostEvent(from: body))
    }

    func test_scoreHighlightSerializesCorrectly() throws {
        let config = ScoreHighlightConfig(
            pieceId: "czerny-op299-no1",
            highlights: [ScoreHighlight(bars: [1, 4], dimension: "dynamics", annotation: "forte")]
        )
        let artifact = ArtifactConfig.scoreHighlight(config)
        let json = try ScoreHostBridge.artifactJSON(for: artifact)
        let decoded = try JSONSerialization.jsonObject(with: json.data(using: .utf8)!) as! [String: Any]
        XCTAssertEqual(decoded["type"] as? String, "score_highlight")
        let inner = decoded["config"] as! [String: Any]
        XCTAssertEqual(inner["pieceId"] as? String, "czerny-op299-no1")
        let highlights = inner["highlights"] as! [[String: Any]]
        XCTAssertEqual(highlights.count, 1)
        XCTAssertEqual(highlights[0]["dimension"] as? String, "dynamics")
    }
}
