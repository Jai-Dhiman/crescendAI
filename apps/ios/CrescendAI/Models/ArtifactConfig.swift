import Foundation

struct ExerciseItem: Codable {
    let title: String
    let instruction: String
    let focusDimension: String
    let exerciseId: String
    let hands: String?
}

struct ExerciseSetConfig: Codable {
    let sourcePassage: String
    let targetSkill: String
    let exercises: [ExerciseItem]
}

struct ScoreHighlight: Codable {
    let bars: [Int]
    let dimension: String
    let annotation: String?
}

struct ScoreHighlightConfig: Codable {
    let pieceId: String
    let highlights: [ScoreHighlight]
}

struct KeyboardGuideConfig: Codable {
    let title: String
    let description: String
    let hands: String
    let fingering: String?
}

enum ArtifactConfig: Codable {
    case exerciseSet(ExerciseSetConfig)
    case scoreHighlight(ScoreHighlightConfig)
    case keyboardGuide(KeyboardGuideConfig)
    case unknown(type: String)

    private enum CodingKeys: String, CodingKey {
        case type, config
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type_ = try container.decode(String.self, forKey: .type)

        switch type_ {
        case "exercise_set":
            let config = try container.decode(ExerciseSetConfig.self, forKey: .config)
            self = .exerciseSet(config)
        case "score_highlight":
            let config = try container.decode(ScoreHighlightConfig.self, forKey: .config)
            self = .scoreHighlight(config)
        case "keyboard_guide":
            let config = try container.decode(KeyboardGuideConfig.self, forKey: .config)
            self = .keyboardGuide(config)
        default:
            self = .unknown(type: type_)
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .exerciseSet(let config):
            try container.encode("exercise_set", forKey: .type)
            try container.encode(config, forKey: .config)
        case .scoreHighlight(let config):
            try container.encode("score_highlight", forKey: .type)
            try container.encode(config, forKey: .config)
        case .keyboardGuide(let config):
            try container.encode("keyboard_guide", forKey: .type)
            try container.encode(config, forKey: .config)
        case .unknown(let type_):
            try container.encode(type_, forKey: .type)
        }
    }
}
