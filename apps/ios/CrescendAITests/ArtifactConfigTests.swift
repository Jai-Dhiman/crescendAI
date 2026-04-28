import XCTest
@testable import CrescendAI

final class ArtifactConfigTests: XCTestCase {

    func test_decodeExerciseSet() throws {
        let json = """
        {
            "type": "exercise_set",
            "config": {
                "sourcePassage": "Chopin Op.9 mm.1-4",
                "targetSkill": "legato phrasing",
                "exercises": [
                    {
                        "title": "Slow practice",
                        "instruction": "Play mm.1-4 at 60bpm",
                        "focusDimension": "phrasing",
                        "exerciseId": "ex-001"
                    }
                ]
            }
        }
        """.data(using: .utf8)!

        let artifact = try JSONDecoder().decode(ArtifactConfig.self, from: json)

        guard case .exerciseSet(let config) = artifact else {
            XCTFail("Expected .exerciseSet, got \(artifact)")
            return
        }
        XCTAssertEqual(config.sourcePassage, "Chopin Op.9 mm.1-4")
        XCTAssertEqual(config.exercises.count, 1)
        XCTAssertEqual(config.exercises[0].exerciseId, "ex-001")
    }

    func test_decodeKeyboardGuide() throws {
        let json = """
        {
            "type": "keyboard_guide",
            "config": {
                "title": "Hand positioning",
                "description": "Keep wrist level",
                "hands": "both"
            }
        }
        """.data(using: .utf8)!

        let artifact = try JSONDecoder().decode(ArtifactConfig.self, from: json)

        guard case .keyboardGuide(let config) = artifact else {
            XCTFail("Expected .keyboardGuide, got \(artifact)")
            return
        }
        XCTAssertEqual(config.title, "Hand positioning")
        XCTAssertEqual(config.hands, "both")
    }

    func test_decodeScoreHighlight() throws {
        let json = """
        {
            "type": "score_highlight",
            "config": {
                "pieceId": "chopin-op9-no2",
                "highlights": [
                    { "bars": [1, 4], "dimension": "dynamics", "annotation": "forte here" }
                ]
            }
        }
        """.data(using: .utf8)!

        let artifact = try JSONDecoder().decode(ArtifactConfig.self, from: json)

        guard case .scoreHighlight(let config) = artifact else {
            XCTFail("Expected .scoreHighlight, got \(artifact)")
            return
        }
        XCTAssertEqual(config.pieceId, "chopin-op9-no2")
        XCTAssertEqual(config.highlights[0].bars, [1, 4])
    }

    func test_decodeUnknownType() throws {
        let json = """
        {
            "type": "future_card",
            "config": { "foo": "bar" }
        }
        """.data(using: .utf8)!

        let artifact = try JSONDecoder().decode(ArtifactConfig.self, from: json)

        guard case .unknown(let typeName) = artifact else {
            XCTFail("Expected .unknown, got \(artifact)")
            return
        }
        XCTAssertEqual(typeName, "future_card")
    }

    func test_chatMessageHasArtifactsAndTeacherRole() {
        let artifact = ArtifactConfig.unknown(type: "test")
        let message = ChatMessage(
            role: .teacher,
            text: "Here is your feedback",
            artifacts: [artifact]
        )
        XCTAssertEqual(message.artifacts.count, 1)
        XCTAssertEqual(message.role, .teacher)
    }
}
