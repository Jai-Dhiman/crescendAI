import Foundation
import SwiftData

enum InferenceStatus: String, Codable, Sendable {
    case pending
    case completed
    case failed
}

@Model
final class ChunkResultRecord {
    var index: Int
    var startOffset: TimeInterval
    var duration: TimeInterval

    // 6 dimension scores
    var dynamics: Double
    var timing: Double
    var pedaling: Double
    var articulation: Double
    var phrasing: Double
    var interpretation: Double

    // STOP classifier output (populated by Slice 4)
    var stopProbability: Double

    // Inference metadata
    var inferenceStatusRaw: String
    var processingTimeMs: Int

    // Relationship
    var session: PracticeSessionRecord?

    var inferenceStatus: InferenceStatus {
        get { InferenceStatus(rawValue: inferenceStatusRaw) ?? .pending }
        set { inferenceStatusRaw = newValue.rawValue }
    }

    init(
        index: Int,
        startOffset: TimeInterval,
        duration: TimeInterval,
        dynamics: Double = 0,
        timing: Double = 0,
        pedaling: Double = 0,
        articulation: Double = 0,
        phrasing: Double = 0,
        interpretation: Double = 0,
        stopProbability: Double = 0,
        inferenceStatus: InferenceStatus = .pending,
        processingTimeMs: Int = 0,
        session: PracticeSessionRecord? = nil
    ) {
        self.index = index
        self.startOffset = startOffset
        self.duration = duration
        self.dynamics = dynamics
        self.timing = timing
        self.pedaling = pedaling
        self.articulation = articulation
        self.phrasing = phrasing
        self.interpretation = interpretation
        self.stopProbability = stopProbability
        self.inferenceStatusRaw = inferenceStatus.rawValue
        self.processingTimeMs = processingTimeMs
        self.session = session
    }
}
